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
//!  * `MetalBackend::set_keepwarm` (off by default; `intarsia serve` turns it on) keeps the driver's wiring of the whole-file
//!    buffers from decaying while the process idles, see there.

use crate::{buffer::MetalBuffer, MetalBackend, MtlBuffer};
use intarsia_core::backend::{Buffer, HostMemory};
use intarsia_core::{Error, Result};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Whole-file buffers are kept warm (see [`MetalBackend::set_keepwarm`]) up to this share of the device's recommended
/// working set; past it the first use of a buffer pays for its own wiring, as before.
const KEEPWARM_SHARE: u64 = 2;

struct Entry {
    _mem: Arc<dyn HostMemory>,
    /// `None`: the whole mapping does not fit one buffer, use per-tensor buffers.
    whole: Option<(MtlBuffer, usize)>,
}

#[derive(Default)]
pub(crate) struct HostCache {
    /// The keep-warm thread and its registry, while [`MetalBackend::set_keepwarm`] has it on.
    kw: Option<KeepWarm>,
    /// Bytes registered with it so far (bounded like [`KEEPWARM_SHARE`]).
    kw_bytes: u64,
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
            if let Some(kw) = &cache.kw {
                if cache.kw_bytes + total as u64 <= be.device().recommendedMaxWorkingSetSize() / KEEPWARM_SHARE {
                    kw.register(mtl.clone(), mem.clone());
                    cache.kw_bytes += total as u64;
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

/// Milliseconds since the first call in this process (a monotonic clock cheap to share through an atomic).
pub(crate) fn now_ms() -> u64 {
    static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// What the keep-warm thread shares with the backend: the whole-file weight buffers to touch (each with the mapping it
/// reads, so the pages cannot go away under a queued touch), the stop flag, and the engine's last-commit clock.
struct KwShared {
    bufs: Mutex<Vec<(MtlBuffer, Arc<dyn HostMemory>)>>,
    stop: AtomicBool,
    period_ms: u64,
    mode: KeepMode,
    /// `now_ms()` of the engine stream's last commit (`Stream::last_commit`).
    last_commit: Arc<AtomicU64>,
    /// Touch command buffers committed so far.
    touches: AtomicU64,
}

// SAFETY: Apple documents MTLBuffer as usable from any thread; the mappings are read-only `Arc`s.
unsafe impl Send for KwShared {}
unsafe impl Sync for KwShared {}

/// A value that crosses to the keep-warm thread although objc2 does not know it is thread-safe (MTLCommandQueue,
/// MTLBuffer: documented as usable from any thread).
struct Xfer<T>(T);
// SAFETY: see above; only used for those two types.
unsafe impl<T> Send for Xfer<T> {}

/// How the keep-warm thread keeps a file wired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeepMode {
    /// A one-byte GPU blit out of every file, from a queue of the thread's own.
    Blit,
    /// An `MTLResidencySet` holding every file, attached to the engine's queue; the thread re-requests its residency (a
    /// CPU-only call, no GPU work).
    Residency,
}

/// Keeps the GPU's wiring of the weight files from decaying (see [`MetalBackend::set_keepwarm`]).
pub(crate) struct KeepWarm {
    shared: Arc<KwShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl KeepWarm {
    fn register(&self, buf: MtlBuffer, mem: Arc<dyn HostMemory>) {
        self.shared.bufs.lock().unwrap().push((buf, mem));
        if let Some(t) = &self.thread {
            t.thread().unpark();
        }
    }
}

impl Drop for KeepWarm {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

/// Commit a command buffer on `queue` that blits one byte out of every registered buffer (no wait); `None` when empty.
fn keepwarm_touch(sh: &KwShared, queue: &ProtocolObject<dyn MTLCommandQueue>, scratch: &MtlBuffer) -> Option<crate::CommandBuffer> {
    let bufs = sh.bufs.lock().unwrap();
    if bufs.is_empty() {
        return None;
    }
    let cb = queue.commandBuffer()?;
    let enc = cb.blitCommandEncoder()?;
    for (i, (b, _)) in bufs.iter().enumerate() {
        // SAFETY: both buffers are at least 16 bytes long (the weight files are whole pages) and distinct.
        unsafe { enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(b, 0, scratch, (i % 16) as usize, 1) };
    }
    enc.endEncoding();
    cb.commit();
    sh.touches.fetch_add(1, Ordering::Relaxed);
    Some(cb)
}

/// The residency-set flavour of [`keepwarm_loop`]: all registered files go into one set on the engine's queue, and
/// `requestResidency` is called again every period while the engine is quiet.
fn residency_loop(sh: Arc<KwShared>, queue: Xfer<crate::Queue>, device: Xfer<crate::Device>) {
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLAllocation, MTLResidencySet, MTLResidencySetDescriptor};
    let (queue, device) = (queue.0, device.0);
    let period = Duration::from_millis(sh.period_ms);
    let mut set: Option<objc2::rc::Retained<ProtocolObject<dyn MTLResidencySet>>> = None;
    let mut seen = 0usize;
    while !sh.stop.load(Ordering::Acquire) {
        let n = sh.bufs.lock().unwrap().len();
        let quiet = now_ms().saturating_sub(sh.last_commit.load(Ordering::Relaxed)) >= sh.period_ms / 2;
        if n != seen || (quiet && n > 0) {
            if set.is_none() {
                let desc = MTLResidencySetDescriptor::new();
                match device.newResidencySetWithDescriptor_error(&desc) {
                    Ok(s) => {
                        queue.addResidencySet(&s);
                        set = Some(s);
                    }
                    Err(e) => {
                        eprintln!("intarsia-backend-metal: keep-warm: no residency set (off): {e:?}");
                        return;
                    }
                }
            }
            let set = set.as_ref().unwrap();
            if n != seen {
                let bufs = sh.bufs.lock().unwrap();
                for (b, _) in &bufs[seen..] {
                    set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&**b));
                }
                set.commit();
                seen = n;
            }
            set.requestResidency();
            sh.touches.fetch_add(1, Ordering::Relaxed);
        }
        std::thread::park_timeout(period);
    }
    if let Some(set) = set {
        queue.removeResidencySet(&set);
    }
}

fn keepwarm_loop(sh: Arc<KwShared>, queue: Xfer<crate::Queue>, scratch: Xfer<MtlBuffer>) {
    let (queue, scratch) = (queue.0, scratch.0);
    let period = Duration::from_millis(sh.period_ms);
    let mut prev: Option<crate::CommandBuffer> = None;
    let mut seen = 0usize;
    while !sh.stop.load(Ordering::Acquire) {
        let n = sh.bufs.lock().unwrap().len();
        // The engine's own command buffers touch every weight; only a quiet engine (nothing committed for half a period)
        // or a newly registered file needs a touch. Never queue a second one behind one that has not run yet.
        let quiet = now_ms().saturating_sub(sh.last_commit.load(Ordering::Relaxed)) >= sh.period_ms / 2;
        let pending = prev.as_ref().is_some_and(|c| !matches!(c.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error));
        if (n != seen || quiet) && !pending {
            prev = keepwarm_touch(&sh, &queue, &scratch);
            seen = n;
        }
        std::thread::park_timeout(period);
    }
    if let Some(cb) = prev {
        cb.waitUntilCompleted();
    }
}

impl MetalBackend {
    /// Keep the GPU's wiring of the wrapped weight files from decaying. The driver wires a file for the GPU on its first
    /// use (about 10 ms per GB) and drops that wiring again once the file has gone unused for about a second: the first
    /// request to a server that was idle for longer pays for all of it again (~170 ms for Qwen3-8B). This starts a thread
    /// that, every `period_ms` while the engine is quiet, commits a command buffer of its own (separate queue) with a
    /// one-byte blit out of each whole-file weight buffer: nothing is computed, nothing the engine does is ordered
    /// against it. A newly wrapped file is touched at once (what pre-wiring did). `0` stops it. The period must stay well
    /// below the decay horizon (1.0 s measured, 1.2 s decayed); the CLI uses 400 ms.
    pub fn set_keepwarm(&self, period_ms: u64) {
        self.set_keepwarm_mode(period_ms, KeepMode::Blit);
    }

    /// [`Self::set_keepwarm`] with the way the files are kept warm.
    pub fn set_keepwarm_mode(&self, period_ms: u64, mode: KeepMode) {
        let mut cache = self.hosts.lock().unwrap();
        if cache.kw.as_ref().is_some_and(|k| k.shared.period_ms == period_ms && k.shared.mode == mode) {
            return;
        }
        cache.kw = None; // joins a running thread
        cache.kw_bytes = 0;
        if period_ms == 0 {
            return;
        }
        let (Some(queue), Ok(scratch)) = (self.device().newCommandQueue(), self.new_mtl_buffer(256, true)) else {
            eprintln!("intarsia-backend-metal: keep-warm: could not make a queue or scratch buffer (off)");
            return;
        };
        if mode == KeepMode::Residency {
            use objc2::runtime::NSObjectProtocol;
            if !self.device().respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
                eprintln!("intarsia-backend-metal: keep-warm: this macOS has no MTLResidencySet (off)");
                return;
            }
        }
        let last_commit = self.stream.lock().unwrap().last_commit.clone();
        let shared = Arc::new(KwShared { bufs: Mutex::new(Vec::new()), stop: AtomicBool::new(false), period_ms, mode, last_commit, touches: AtomicU64::new(0) });
        let sh = shared.clone();
        let thread = match mode {
            KeepMode::Blit => {
                let (q, sc) = (Xfer(queue), Xfer(scratch));
                std::thread::Builder::new().name("metal-keepwarm".into()).spawn(move || keepwarm_loop(sh, q, sc)).ok()
            }
            KeepMode::Residency => {
                let (q, d) = (Xfer(self.queue.clone()), Xfer(self.device.clone()));
                std::thread::Builder::new().name("metal-keepwarm".into()).spawn(move || residency_loop(sh, q, d)).ok()
            }
        };
        let kw = KeepWarm { shared, thread };
        let limit = self.device().recommendedMaxWorkingSetSize() / KEEPWARM_SHARE;
        let mut bytes = 0u64;
        for e in cache.map.values() {
            if let Some((mtl, _)) = &e.whole {
                if bytes + mtl.length() as u64 <= limit {
                    kw.register(mtl.clone(), e._mem.clone());
                    bytes += mtl.length() as u64;
                }
            }
        }
        cache.kw_bytes = bytes;
        cache.kw = Some(kw);
    }

    /// Touch command buffers the keep-warm thread has committed (0 while it is off).
    pub fn keepwarm_touches(&self) -> u64 {
        self.hosts.lock().unwrap().kw.as_ref().map_or(0, |k| k.shared.touches.load(Ordering::Relaxed))
    }

    /// Stop the keep-warm thread and wait for its last command buffer; `Drop` runs this before the mappings can go.
    pub(crate) fn stop_keepwarm(&self) {
        let kw = self.hosts.lock().unwrap().kw.take();
        drop(kw);
    }

    /// [`Backend::wrap_host`] with the reason it declined.
    pub fn try_wrap_host(&self, mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> Result<Buffer> {
        try_wrap(self, mem, offset, len)
    }
}
