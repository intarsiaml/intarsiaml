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
//!  * With `MetalBackend::set_prewire(true)` (off by default; `intarsia serve` turns it on) a new whole-file buffer is pre-wired: a one-byte blit out of it in a committed command buffer of its own makes the
//!    driver map the whole file for the GPU in the background (about 10 ms per GB), so the first prefill chunk does not
//!    wait for it (`MetalBackend::prewire`). `Drop` waits for those buffers.

use crate::{buffer::MetalBuffer, MetalBackend, MtlBuffer};
use intarsia_core::backend::{Buffer, HostMemory};
use intarsia_core::{Error, Result};
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// Whole-file buffers may be pre-wired (see [`MetalBackend::prewire`]) up to this share of the device's recommended
/// working set; past it the first use of a buffer pays for its own wiring, as before.
const PREWIRE_SHARE: u64 = 2;

impl HostCache {
    pub(crate) fn prewired(&self) -> u64 {
        self.prewired
    }
}

struct Entry {
    _mem: Arc<dyn HostMemory>,
    /// `None`: the whole mapping does not fit one buffer, use per-tensor buffers.
    whole: Option<(MtlBuffer, usize)>,
}

#[derive(Default)]
pub(crate) struct HostCache {
    map: HashMap<usize, Entry>,
    /// Bytes handed to [`MetalBackend::prewire`] so far.
    prewired: u64,
    /// Their command buffers, until they have completed: [`MetalBackend::drain_prewire`] waits for them before the
    /// mappings can go away (the blit reads the no-copy buffer).
    prewire_cbs: Vec<crate::CommandBuffer>,
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
    let page = be.page_size();
    let start = slice.as_ptr() as usize;
    let key = Arc::as_ptr(&mem) as *const () as usize;
    let max = be.device().maxBufferLength();
    let mut cache = be.hosts.lock().unwrap();
    if !cache.map.contains_key(&key) {
        let floor = round_down(start, page);
        let total = round_up(start + slice.len(), page) - floor;
        let whole = if total <= max { Some((no_copy(be, floor, total)?, floor)) } else { None };
        if let Some((mtl, _)) = &whole {
            if cache.prewired + total as u64 <= be.device().recommendedMaxWorkingSetSize() / PREWIRE_SHARE {
                if let Some(cb) = be.prewire(mtl) {
                    cache.prewired += total as u64;
                    cache.prewire_cbs.retain(|c| c.status() != MTLCommandBufferStatus::Completed);
                    cache.prewire_cbs.push(cb);
                }
            }
        }
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
    /// Reference a freshly wrapped weight file from a committed command buffer of its own (a one-byte blit out of it), so
    /// the driver maps all of the file for the GPU now, in the background, while the loader carries on. The first command
    /// buffer that uses such a buffer otherwise waits for that (about 10 ms per GB: ~100 ms before the first prefill chunk
    /// of Qwen3-8B starts, and ~25 ms more before each of the next two command buffers, which a short chunk cannot hide
    /// behind the one still running). Nothing is computed, nothing is waited on.
    /// Returns the committed command buffer (the caller keeps it for [`Self::drain_prewire`]); `None` when pre-wiring is
    /// off or the command buffer could not be made.
    fn prewire(&self, buf: &MtlBuffer) -> Option<crate::CommandBuffer> {
        if !self.prewire.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let scratch = self.new_mtl_buffer(16, true).ok()?;
        let cb = self.queue().commandBuffer()?;
        let enc = cb.blitCommandEncoder()?;
        // SAFETY: both buffers are at least one byte long and distinct.
        unsafe { enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(buf, 0, &scratch, 0, 1) };
        enc.endEncoding();
        cb.commit();
        Some(cb)
    }

    /// Wait (`wait`) for the pre-wiring command buffers and drop the finished ones. They are not part of the stream (the
    /// first upload at load time must not wait for the wiring, that overlap is the point), so `synchronize` leaves them
    /// alone; this runs from `Drop`, before the backend lets go of the mappings they read, and from `synchronize` to
    /// forget the ones already finished. A failed one only means the first use of that file pays for the wiring, as before:
    /// it is reported once on stderr and otherwise ignored.
    pub(crate) fn drain_prewire(&self, wait: bool) {
        let mut cache = self.hosts.lock().unwrap();
        for cb in std::mem::take(&mut cache.prewire_cbs) {
            if wait {
                cb.waitUntilCompleted();
            }
            match cb.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("intarsia-backend-metal: pre-wiring a weight file failed (ignored): {:?}", cb.error()));
                }
                _ => cache.prewire_cbs.push(cb),
            }
        }
    }

    /// [`Backend::wrap_host`] with the reason it declined.
    pub fn try_wrap_host(&self, mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> Result<Buffer> {
        try_wrap(self, mem, offset, len)
    }
}
