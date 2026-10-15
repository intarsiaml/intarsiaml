//! `Backend::wrap_host`: zero-copy import of mmap'd weight files as `MTLBuffer`s.
//!
//! `newBufferWithBytesNoCopy` wants a page-aligned pointer and a length that is a multiple of the page size (16 KiB
//! on Apple Silicon, read from `getpagesize`). Policy:
//!
//!  * The base pointer is rounded DOWN and the end rounded UP to page boundaries. Neither can leave the mapping:
//!    virtual memory is mapped in whole pages, so the page that holds the first (last) byte of the slice is mapped in
//!    full. For an mmap'd file the tail of its last page reads as zeros. For a `HostMemory` that is not
//!    page-backed (a `Vec`), the extra bytes belong to other allocations; the buffer is read-only and the GPU never
//!    reads them, so this is harmless, but nothing here relies on it.
//!    Apple also wants no-copy memory to come from mmap/vm_allocate (not malloc). The only `HostMemory` implementation
//!    in the workspace is `MmapHost` (intarsia-models/src/safetensors.rs, a read-only `memmap2::Mmap`), so that
//!    holds; a future `HostMemory` over heap memory must not be wrapped (Metal may refuse it, and the rounded tail
//!    would belong to other allocations).
//!  * One `MTLBuffer` per `Arc<dyn HostMemory>` (keyed by the Arc's pointer, entry keeps the Arc alive so the key
//!    cannot be reused) covers the whole mapping; every tensor of a file shares it. A returned `Buffer` is a window:
//!    `MetalBuffer::base` = tensor start - buffer start, and `TensorRef::offset` stacks on top (`plan_gemv` binds at
//!    `base + offset`).
//!  * A mapping larger than `maxBufferLength` cannot be one buffer; each tensor then gets its own page-rounded buffer
//!    (no sharing), or `None` when even that exceeds the limit, and the loader falls back to alloc + upload.
//!  * `None` is also returned for empty ranges, ranges outside the slice, and when Metal refuses the memory; set
//!    `INTARSIA_METAL_DEBUG=1` to print the reason, or call [`MetalBackend::try_wrap_host`] for it.
//!  * The cache holds the `Arc` until the backend drops: a wrapped file stays mapped for the process lifetime.
//!  * Weight re-alignment (opt-in, `INTARSIA_METAL_REALIGN=1` or [`MetalBackend::set_realign`]): a range whose
//!    address is not 16-byte aligned (every tensor of a safetensors shard whose JSON header is 8 mod 16 bytes long)
//!    is copied into its own page-aligned `MTLBuffer` instead of windowing the shared mapping, so the 16-byte-load
//!    GEMV kernels (and the multi-row ones, which lose 20 percent at R = 8 on the 8-byte-load variant) are picked
//!    for it. The copy is read-only like any wrapped buffer. The source pages are then handed back with
//!    `madvise(MADV_DONTNEED)` (whole pages inside the range only), so the file-backed pages the copy faulted in do
//!    not stay resident next to the copy. The backend cannot tell a GEMV weight from a gathered embedding table
//!    (`wrap_host` carries no tensor name), so both are copied.

use crate::{buffer::MetalBuffer, MetalBackend, MtlBuffer};
use intarsia_core::backend::{Buffer, HostMemory};
use intarsia_core::{Error, Result};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::Arc;

struct Entry {
    _mem: Arc<dyn HostMemory>,
    /// `None`: the whole mapping does not fit one buffer, use per-tensor buffers.
    whole: Option<(MtlBuffer, usize)>,
}

#[derive(Default)]
pub(crate) struct HostCache {
    map: HashMap<usize, Entry>,
}

fn round_down(v: usize, page: usize) -> usize {
    v & !(page - 1)
}
fn round_up(v: usize, page: usize) -> usize {
    (v + page - 1) & !(page - 1)
}

/// `newBufferWithBytesNoCopy` over `[floor, floor + len)` (both page aligned), read-only use.
fn no_copy(be: &MetalBackend, floor: usize, len: usize) -> Result<MtlBuffer> {
    // SAFETY: the range lies in pages mapped for the life of the `Arc<dyn HostMemory>` the caller keeps in the
    // returned buffer; no deallocator, so Metal never frees it.
    let b = unsafe {
        be.device().newBufferWithBytesNoCopy_length_options_deallocator(
            NonNull::new(floor as *mut c_void).ok_or_else(|| Error::backend("null host pointer"))?,
            len,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    };
    b.ok_or_else(|| Error::backend(format!("newBufferWithBytesNoCopy refused {len} bytes at {floor:#x}")))
}

/// Copy `src` (a range of mapped host memory) into a fresh page-aligned, read-only `MTLBuffer` and release the
/// source pages. `MTLBuffer` contents are page aligned, so the copy is 16-byte aligned at `base` 0.
fn realign(be: &MetalBackend, src: &[u8]) -> Result<Buffer> {
    let mtl = be.new_mtl_buffer(src.len(), false)?;
    // SAFETY: `contents` points at `src.len()` writable bytes of the fresh buffer, which cannot overlap the mapping.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), mtl.contents().as_ptr() as *mut u8, src.len()) };
    // Drop the whole pages that lie inside the range from the process (file-backed, clean: the next touch, if any,
    // refaults them). Partial pages at either end belong to neighbouring tensors and stay.
    let page = be.page_size();
    let (a, b) = (round_up(src.as_ptr() as usize, page), round_down(src.as_ptr() as usize + src.len(), page));
    if b > a {
        // SAFETY: `[a, b)` lies inside the live mapping `src` is a slice of; MADV_DONTNEED only advises.
        unsafe { libc::madvise(a as *mut c_void, b - a, libc::MADV_DONTNEED) };
    }
    if std::env::var_os("INTARSIA_METAL_DEBUG").is_some() {
        eprintln!("intarsia-backend-metal: realign {} bytes (address mod 16 = {})", src.len(), src.as_ptr() as usize % 16);
    }
    be.realigned_bytes.fetch_add(src.len(), Ordering::Relaxed);
    be.realigned_tensors.fetch_add(1, Ordering::Relaxed);
    Ok(Buffer::new(MetalBuffer { mtl, base: 0, bytes: src.len(), read_only: true, _host: None }))
}

pub(crate) fn wrap(be: &MetalBackend, mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> Option<Buffer> {
    match try_wrap(be, mem, offset, len) {
        Ok(b) => Some(b),
        Err(e) => {
            if std::env::var_os("INTARSIA_METAL_DEBUG").is_some() {
                eprintln!("intarsia-backend-metal: wrap_host -> None: {e}");
            }
            None
        }
    }
}

pub(crate) fn try_wrap(be: &MetalBackend, mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> Result<Buffer> {
    let slice = mem.as_slice();
    if len == 0 || offset.checked_add(len).is_none_or(|e| e > slice.len()) {
        return Err(Error::invalid(format!("wrap_host range {offset}+{len} outside the {} byte mapping", slice.len())));
    }
    if be.realign.load(Ordering::Relaxed) && (slice.as_ptr() as usize + offset) % 16 != 0 {
        return realign(be, &slice[offset..offset + len]);
    }
    let page = be.page_size();
    let start = slice.as_ptr() as usize;
    let key = Arc::as_ptr(&mem) as *const () as usize;
    let max = be.device().maxBufferLength();
    let mut cache = be.hosts.lock().unwrap();
    if !cache.map.contains_key(&key) {
        let floor = round_down(start, page);
        let total = round_up(start + slice.len(), page) - floor;
        let whole = if total <= max { Some((no_copy(be, floor, total)?, floor)) } else { None };
        cache.map.insert(key, Entry { _mem: mem.clone(), whole });
    }
    let entry = &cache.map[&key];
    let (mtl, base) = match &entry.whole {
        Some((mtl, floor)) => (mtl.clone(), start + offset - floor),
        None => {
            let t0 = start + offset;
            let floor = round_down(t0, page);
            let total = round_up(t0 + len, page) - floor;
            if total > max {
                return Err(Error::backend(format!("tensor span {total} bytes exceeds maxBufferLength {max}")));
            }
            (no_copy(be, floor, total)?, t0 - floor)
        }
    };
    Ok(Buffer::new(MetalBuffer { mtl, base, bytes: len, read_only: true, _host: Some(mem) }))
}

impl MetalBackend {
    /// Switch weight re-alignment in `wrap_host` on or off (default: `INTARSIA_METAL_REALIGN` set and not `0`/empty).
    /// Affects later `wrap_host` calls only.
    pub fn set_realign(&self, on: bool) {
        self.realign.store(on, Ordering::Relaxed);
    }

    /// (tensors, bytes) copied by re-alignment so far.
    pub fn realigned(&self) -> (usize, usize) {
        (self.realigned_tensors.load(Ordering::Relaxed), self.realigned_bytes.load(Ordering::Relaxed))
    }

    /// [`Backend::wrap_host`] with the reason it declined.
    pub fn try_wrap_host(&self, mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> Result<Buffer> {
        try_wrap(self, mem, offset, len)
    }
}
