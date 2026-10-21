//! NCCL collectives for tensor parallelism across processes and hosts (docs/design/35b-tp2.md): one rank per process,
//! one GB10 per Spark, the two joined over ConnectX-7. libnccl is loaded at run time (`dlopen`, path in
//! `INTARSIA_NCCL_LIB`, default `libnccl.so.2`), so the build needs no NCCL headers or library and a binary without
//! it still runs everything else. The Sparks take the library from the vLLM container image (libnccl 2.30.7).
//!
//! Numerics: a sum of two operands commutes, so both ranks get the same bits, and they are the bits of the host
//! communicator (`f32(a) + f32(b)`, one rounding into the output dtype): f32 partials are summed in f32 by NCCL into a
//! scratch buffer and rounded once (round to nearest even) into the bf16 output; bf16 and f32 in-place sums are NCCL's.

use crate::device::CudaBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, StreamId, TensorRef};
use intarsia_core::comm::{Comm, CommCounters, CommStats};
use intarsia_core::{DType, Error, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// `ncclUniqueId`: 128 opaque bytes rank 0 creates and every rank passes to `ncclCommInitRank`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    pub internal: [u8; 128],
}

type NcclCommPtr = *mut c_void;

struct Lib {
    get_unique_id: unsafe extern "C" fn(*mut NcclUniqueId) -> c_int,
    comm_init_rank: unsafe extern "C" fn(*mut NcclCommPtr, c_int, NcclUniqueId, c_int) -> c_int,
    all_reduce: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, NcclCommPtr, *mut c_void) -> c_int,
    all_gather: unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, NcclCommPtr, *mut c_void) -> c_int,
    comm_abort: unsafe extern "C" fn(NcclCommPtr) -> c_int,
    get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    get_version: unsafe extern "C" fn(*mut c_int) -> c_int,
}

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;

const NCCL_SUM: c_int = 0;
fn nccl_dtype(d: DType) -> Result<c_int> {
    Ok(match d {
        DType::F16 => 6,
        DType::F32 => 7,
        DType::Bf16 => 9,
        DType::I32 => 2,
        DType::U8 => 1,
        other => return Err(Error::invalid(format!("nccl: unsupported dtype {}", other.name()))),
    })
}

fn load_lib() -> std::result::Result<Lib, String> {
    let path = std::env::var("INTARSIA_NCCL_LIB").unwrap_or_else(|_| "libnccl.so.2".into());
    let c = CString::new(path.clone()).map_err(|e| e.to_string())?;
    // SAFETY: dlopen / dlsym with NUL-terminated strings; the handle is never closed (the library stays loaded for the
    // process), so the function pointers stay valid.
    unsafe {
        let h = dlopen(c.as_ptr(), RTLD_NOW);
        if h.is_null() {
            let e = dlerror();
            let msg = if e.is_null() { "unknown error".into() } else { CStr::from_ptr(e).to_string_lossy().into_owned() };
            return Err(format!("cannot load NCCL from '{path}' (set INTARSIA_NCCL_LIB): {msg}"));
        }
        let sym = |name: &str| -> std::result::Result<*mut c_void, String> {
            let n = CString::new(name).unwrap();
            let p = dlsym(h, n.as_ptr());
            if p.is_null() {
                Err(format!("NCCL library '{path}' has no symbol {name}"))
            } else {
                Ok(p)
            }
        };
        Ok(Lib {
            get_unique_id: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut NcclUniqueId) -> c_int>(sym("ncclGetUniqueId")?),
            comm_init_rank: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut NcclCommPtr, c_int, NcclUniqueId, c_int) -> c_int>(
                sym("ncclCommInitRank")?,
            ),
            all_reduce: std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, NcclCommPtr, *mut c_void) -> c_int,
            >(sym("ncclAllReduce")?),
            all_gather: std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, NcclCommPtr, *mut c_void) -> c_int,
            >(sym("ncclAllGather")?),
            comm_abort: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(NcclCommPtr) -> c_int>(sym("ncclCommAbort")?),
            get_error_string: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(c_int) -> *const c_char>(sym("ncclGetErrorString")?),
            get_version: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_int) -> c_int>(sym("ncclGetVersion")?),
        })
    }
}

fn lib() -> Result<&'static Lib> {
    static LIB: OnceLock<std::result::Result<Lib, String>> = OnceLock::new();
    LIB.get_or_init(load_lib).as_ref().map_err(|e| Error::backend(e.clone()))
}

fn check(l: &Lib, what: &str, rc: c_int) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    // SAFETY: ncclGetErrorString returns a static NUL-terminated string.
    let msg = unsafe { CStr::from_ptr((l.get_error_string)(rc)) }.to_string_lossy().into_owned();
    Err(Error::backend(format!("{what}: {msg} (ncclResult {rc})")))
}

/// The NCCL library version ("2.30.7").
pub fn version() -> Result<String> {
    let l = lib()?;
    let mut v = 0;
    // SAFETY: valid out-pointer.
    check(l, "ncclGetVersion", unsafe { (l.get_version)(&mut v) })?;
    Ok(format!("{}.{}.{}", v / 10000, (v / 100) % 100, v % 100))
}

/// A fresh communicator id (rank 0 makes it and sends it to the other ranks).
pub fn unique_id() -> Result<[u8; 128]> {
    let l = lib()?;
    let mut id = NcclUniqueId { internal: [0; 128] };
    // SAFETY: valid out-pointer.
    check(l, "ncclGetUniqueId", unsafe { (l.get_unique_id)(&mut id) })?;
    Ok(id.internal)
}

/// Open CUDA device `ordinal` and join rank `rank` of the `world`-rank NCCL communicator `id`. Blocks until every rank
/// has joined. Call it before loading weights (a fresh process has the most room for NCCL's registered buffers).
pub fn create(ordinal: usize, rank: usize, world: usize, id: [u8; 128]) -> Result<(Arc<dyn Backend>, Arc<dyn Comm>)> {
    crate::disable::check_env()?;
    let l = lib()?;
    let be = Arc::new(CudaBackend::new(ordinal)?);
    be.bind()?;
    let mut comm: NcclCommPtr = std::ptr::null_mut();
    // SAFETY: valid out-pointer; the id is passed by value as NCCL expects.
    check(l, "ncclCommInitRank", unsafe { (l.comm_init_rank)(&mut comm, world as c_int, NcclUniqueId { internal: id }, rank as c_int) })?;
    let c = NcclComm {
        backend: be.clone(),
        comm,
        rank,
        world,
        aborted: AtomicBool::new(false),
        scratch: Mutex::new(None),
        reserved: AtomicBool::new(false),
        counters: CommCounters::default(),
    };
    Ok((be as Arc<dyn Backend>, Arc::new(c) as Arc<dyn Comm>))
}

struct NcclComm {
    backend: Arc<CudaBackend>,
    comm: NcclCommPtr,
    rank: usize,
    world: usize,
    aborted: AtomicBool,
    /// f32 landing buffer of an f32-partial -> bf16 reduce: grown on demand until [`Comm::reserve`] fixes its size
    /// (and address) for good.
    scratch: Mutex<Option<Buffer>>,
    reserved: AtomicBool,
    counters: CommCounters,
}

// SAFETY: the communicator handle is used from one thread at a time per collective (the model's forward thread);
// `abort` may come from another thread, which NCCL allows (ncclCommAbort is the documented way to unblock a rank).
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    fn live(&self) -> Result<&'static Lib> {
        if self.aborted.load(Ordering::Acquire) {
            return Err(Error::backend("communicator aborted (another rank failed)"));
        }
        lib()
    }

    fn reduce(&self, l: &Lib, src: *const c_void, dst: *mut c_void, n: usize, dt: DType, stream: StreamId) -> Result<()> {
        let s = self.backend.stream(stream)?;
        // SAFETY: src / dst are device ranges of `n` elements (bounds checked by the callers); the stream is live.
        check(l, "ncclAllReduce", unsafe { (l.all_reduce)(src, dst, n, nccl_dtype(dt)?, NCCL_SUM, self.comm, s) })
    }
}

impl Comm for NcclComm {
    fn name(&self) -> &str {
        "nccl"
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world(&self) -> usize {
        self.world
    }

    fn all_reduce_sum(&self, x: &TensorRef, stream: StreamId) -> Result<()> {
        self.all_reduce_sum_into(x, x, stream)
    }

    fn all_reduce_sum_into(&self, partial: &TensorRef, out: &TensorRef, stream: StreamId) -> Result<()> {
        let l = self.live()?;
        for t in [partial, out] {
            if !matches!(t.dtype, DType::Bf16 | DType::F16 | DType::F32) {
                return Err(Error::invalid(format!("nccl all_reduce supports bf16/f16/f32, got {}", t.dtype.name())));
            }
            t.check_bounds()?;
        }
        let n = partial.numel();
        if n != out.numel() {
            return Err(Error::invalid(format!("nccl all_reduce: {n} partial elements into {} output elements", out.numel())));
        }
        self.counters.call(partial.nbytes());
        self.backend.bind()?;
        let src = CudaBackend::ptr(&partial.buf, partial.offset, partial.nbytes())? as *const c_void;
        let dst = CudaBackend::ptr(&out.buf, out.offset, out.nbytes())? as *mut c_void;
        if partial.dtype == out.dtype {
            // in place, or out of place leaving the partial untouched
            return self.reduce(l, src, dst, n, out.dtype, stream);
        }
        if partial.dtype != DType::F32 {
            return Err(Error::invalid(format!(
                "nccl all_reduce: {} partials into {} (only f32 partials round into another dtype)",
                partial.dtype.name(),
                out.dtype.name()
            )));
        }
        let mut g = self.scratch.lock().unwrap();
        if g.as_ref().is_none_or(|b| b.bytes() < n * 4) {
            if self.reserved.load(Ordering::Acquire) {
                // a captured graph may hold the reserved buffer's address: never move it
                return Err(Error::invalid(format!(
                    "nccl all_reduce: {n} f32 partials exceed the reserved {} (Comm::reserve)",
                    g.as_ref().map_or(0, |b| b.bytes() / 4)
                )));
            }
            if g.is_some() {
                // the old buffer may still be read by queued work: drain before it is freed
                self.backend.synchronize(stream)?;
            }
            *g = Some(self.backend.alloc(n * 4, MemKind::Device)?);
        }
        let sc = CudaBackend::ptr(g.as_ref().unwrap(), 0, n * 4)?;
        self.reduce(l, src, sc as *mut c_void, n, DType::F32, stream)?;
        let code = if out.dtype == DType::Bf16 { 0 } else { 1 };
        let s = self.backend.stream(stream)?;
        // SAFETY: sc holds n f32, dst n elements of out.dtype; same stream as the reduce.
        let rc = unsafe { crate::ffi::ik_rt_f32_round(sc as *const c_void, dst, n, code, s) };
        if rc != 0 {
            return Err(Error::backend(format!("f32 partial rounding failed (cudaError {rc})")));
        }
        Ok(())
    }

    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()> {
        let l = self.live()?;
        send.check_bounds()?;
        recv.check_bounds()?;
        if send.dtype != recv.dtype || recv.numel() != send.numel() * self.world {
            return Err(Error::invalid(format!(
                "nccl all_gather: {} {} elements into {} {} (world {})",
                send.numel(),
                send.dtype.name(),
                recv.numel(),
                recv.dtype.name(),
                self.world
            )));
        }
        self.counters.call(send.nbytes());
        self.backend.bind()?;
        let s = self.backend.stream(stream)?;
        let src = CudaBackend::ptr(&send.buf, send.offset, send.nbytes())? as *const c_void;
        let dst = CudaBackend::ptr(&recv.buf, recv.offset, recv.nbytes())? as *mut c_void;
        // SAFETY: bounds checked above; live stream.
        check(l, "ncclAllGather", unsafe { (l.all_gather)(src, dst, send.numel(), nccl_dtype(send.dtype)?, self.comm, s) })
    }

    fn abort(&self) {
        if !self.aborted.swap(true, Ordering::AcqRel) {
            if let Ok(l) = lib() {
                // SAFETY: the handle came from ncclCommInitRank; abort is safe while other threads wait in a collective.
                unsafe {
                    (l.comm_abort)(self.comm);
                }
            }
        }
    }

    fn stats(&self) -> CommStats {
        self.counters.snapshot()
    }

    fn note_timed(&self, d: Duration) {
        self.counters.timed(d);
    }

    /// `ncclAllReduce` is a stream operation NCCL supports inside stream capture (thread-local mode, `ik_rt_graph_begin`),
    /// and the f32 rounding is one kernel; after `reserve` nothing allocates or synchronises.
    fn capture_safe(&self) -> bool {
        true
    }

    fn reserve(&self, max_elems: usize, stream: StreamId) -> Result<()> {
        let mut g = self.scratch.lock().unwrap();
        if g.as_ref().is_none_or(|b| b.bytes() < max_elems * 4) {
            if g.is_some() {
                self.backend.synchronize(stream)?;
            }
            *g = Some(self.backend.alloc(max_elems.max(1) * 4, MemKind::Device)?);
        }
        self.reserved.store(true, Ordering::Release);
        Ok(())
    }
}
