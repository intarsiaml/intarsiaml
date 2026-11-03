//! Direct RDMA all-reduce for the TP=2 f32 partial sums (lane dsv41/rdma, `INTARSIA_DSV41_RDMA`;
//! docs/results/2026-11-02-dsv41-rdma.md). A [`Comm`] that wraps the NCCL communicator and takes over exactly one call:
//! `all_reduce_sum_into` of an f32 partial into a bf16 output on stream 0 (80 of a decode pass's 83 collectives), through the
//! stage / collect kernel pair and host proxy of `kernels/cuda/ik_rdma.cu`. Every other call is NCCL's. The kernels are plain
//! stream work with their sequence number on the device, so a recorded CUDA graph replays them like any other node.
//!
//! The choice per call depends only on dtype, element count and stream, which both ranks see alike. The result is bitwise the
//! NCCL pair's (`ncclAllReduce` f32 + `ik_rt_f32_round`): at world 2 `a + b == b + a`, rounded once to nearest even.
//! The setting has three states (unset = auto, `1` = required, `0` = off) and the setup is a lock-step exchange between the
//! ranks that falls back to plain NCCL, with one log line, when anything about the ring fails ([`crate::rdma_policy`]).
//! With the ring off no wrapper exists: [`wrap`] returns the NCCL communicator itself.

use crate::device::CudaBackend;
use crate::ffi;
use crate::rdma_policy::{self, Outcome, RingOps, Wish};
use intarsia_core::backend::{StreamId, TensorRef};
use intarsia_core::comm::{ArNorm, Comm, CommCounters, CommStats};
use intarsia_core::{DType, Error, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

/// Largest partial the rings carry: 256 KiB of f32 (12 rows of the 5120-wide model); bigger calls (a prefill chunk) go to NCCL.
const MAX_BYTES: usize = 256 * 1024;
const SLOTS: c_int = 4;
/// The host round trip of the setup (a first packet may wait for an ARP / route lookup).
const PING_MS: c_int = 10_000;

fn lib_err(buf: &[c_char]) -> String {
    // SAFETY: the C side wrote a NUL-terminated string into the buffer (or left it zeroed).
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

/// One rank's end of the RDMA ring (the host proxy thread runs inside it).
struct Ring(*mut c_void);

// SAFETY: the ring's host state is atomics and pinned memory the proxy thread shares by design; the launch call only reads
// the (immutable after setup) pointers and enqueues kernels.
unsafe impl Send for Ring {}
unsafe impl Sync for Ring {}

pub struct RdmaComm {
    inner: Arc<dyn Comm>,
    backend: Arc<CudaBackend>,
    ring: Ring,
    counters: CommCounters,
}

/// The real ring behind the setup's [`RingOps`].
struct RealRing {
    rank: usize,
    h: *mut c_void,
}

impl RingOps for RealRing {
    fn open(&mut self, min_rails: i32) -> std::result::Result<(), String> {
        let (devs, gid) = rdma_policy::devices(std::env::var("NCCL_IB_HCA").ok().as_deref(), std::env::var("NCCL_IB_GID_INDEX").ok().as_deref())?;
        let deadline = rdma_policy::deadline_s(std::env::var("INTARSIA_DSV41_RDMA_DEADLINE_S").ok().as_deref());
        let fault = std::env::var("INTARSIA_DSV41_RDMA_FAULT").unwrap_or_default();
        let cd = CString::new(devs.clone()).map_err(|e| e.to_string())?;
        let cf = CString::new(fault).map_err(|e| e.to_string())?;
        let mut err = [0 as c_char; 256];
        // SAFETY: valid NUL-terminated strings and error buffer.
        let h = unsafe { ffi::ik_rdma_open(cd.as_ptr(), gid, self.rank as c_int, MAX_BYTES, SLOTS, min_rails, deadline, cf.as_ptr(), err.as_mut_ptr(), err.len() as c_int) };
        if h.is_null() {
            return Err(format!("ring on {devs} (gid {gid}): {}", lib_err(&err)));
        }
        self.h = h;
        Ok(())
    }
    fn info(&self) -> Vec<u8> {
        // SAFETY: plain FFI call.
        let n = unsafe { ffi::ik_rdma_info_size() } as usize;
        assert!(n <= rdma_policy::INFO_MAX, "the ring's info block outgrew the setup frame");
        let mut v = vec![0u8; n];
        // SAFETY: `v` holds the info block's size; the handle is open.
        unsafe { ffi::ik_rdma_info(self.h, v.as_mut_ptr() as *mut c_void) };
        v
    }
    fn connect(&mut self, peer_info: &[u8]) -> std::result::Result<(), String> {
        let mut err = [0 as c_char; 256];
        // SAFETY: the peer's block is at least the info block's size (a frame carries INFO_MAX bytes).
        let rc = unsafe { ffi::ik_rdma_connect(self.h, peer_info.as_ptr() as *const c_void, err.as_mut_ptr(), err.len() as c_int) };
        if rc != 0 {
            return Err(lib_err(&err));
        }
        Ok(())
    }
    fn ping(&mut self, timeout_ms: i32) -> std::result::Result<(), String> {
        let mut err = [0 as c_char; 256];
        // SAFETY: valid handle and buffer.
        let rc = unsafe { ffi::ik_rdma_ping(self.h, timeout_ms, err.as_mut_ptr(), err.len() as c_int) };
        if rc != 0 {
            return Err(lib_err(&err));
        }
        Ok(())
    }
    fn discard(&mut self) {
        if !self.h.is_null() {
            // SAFETY: a ring that was never started: everything is freed.
            unsafe { ffi::ik_rdma_close(self.h) };
            self.h = std::ptr::null_mut();
        }
    }
}

/// Set the ring up between the two ranks over the control stream `ctl` and wrap `inner`, or keep plain NCCL: `wish` is this
/// rank's setting. Auto falls back on any local or remote failure (one log line); required and a disagreement are errors.
pub fn wrap(backend: Arc<CudaBackend>, inner: Arc<dyn Comm>, rank: usize, ctl: &mut TcpStream, wish: Wish) -> Result<Arc<dyn Comm>> {
    let io = |e: std::io::Error| Error::backend(format!("RDMA setup: control stream: {e}"));
    ctl.set_read_timeout(Some(rdma_policy::SETUP_TIMEOUT)).map_err(io)?;
    ctl.set_write_timeout(Some(rdma_policy::SETUP_TIMEOUT)).map_err(io)?;
    let mut ring = RealRing { rank, h: std::ptr::null_mut() };
    let r = if wish == Wish::Off {
        rdma_policy::setup(ctl, rank, wish, &mut ring, PING_MS)
    } else {
        // the pinned memory and the verbs objects belong to this thread's device context
        backend.bind().map_err(|e| Error::backend(format!("RDMA setup: {e}")))?;
        rdma_policy::setup(ctl, rank, wish, &mut ring, PING_MS)
    };
    ctl.set_read_timeout(None).map_err(io)?;
    ctl.set_write_timeout(None).map_err(io)?;
    match r.map_err(Error::invalid)? {
        Outcome::Off => Ok(inner),
        Outcome::Fallback(why) => {
            eprintln!("tensor parallel: rank {rank}: RDMA exchange not used, f32 all-reduces stay on NCCL ({why}) [INTARSIA_DSV41_RDMA]");
            Ok(inner)
        }
        Outcome::Ready => {
            // SAFETY: plain FFI call; -1: the proxy thread is not pinned.
            unsafe { ffi::ik_rdma_start(ring.h, -1) };
            eprintln!("tensor parallel: rank {rank} f32 all-reduces over RDMA, up to {} KiB, the rest over NCCL (INTARSIA_DSV41_RDMA)", MAX_BYTES / 1024);
            Ok(Arc::new(RdmaComm { inner, backend, ring: Ring(ring.h), counters: CommCounters::default() }))
        }
    }
}

impl RdmaComm {
    /// Does this call take the RDMA path? Only dtype, element count and stream decide, so both ranks agree.
    fn eligible(partial: &TensorRef, out: &TensorRef, stream: StreamId) -> bool {
        let n = partial.numel();
        stream == 0 && partial.dtype == DType::F32 && out.dtype == DType::Bf16 && n == out.numel() && n > 0 && n % 4 == 0 && n * 4 <= MAX_BYTES
    }

    fn check_ring(&self) -> Result<()> {
        let mut why = [0 as c_char; 160];
        // SAFETY: valid ring handle and buffer.
        if unsafe { ffi::ik_rdma_failed(self.ring.0, why.as_mut_ptr(), why.len() as c_int) } != 0 {
            return Err(Error::backend(format!("RDMA ring failed: {}", lib_err(&why))));
        }
        Ok(())
    }
}

impl Comm for RdmaComm {
    fn name(&self) -> &str {
        "nccl+rdma"
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world(&self) -> usize {
        self.inner.world()
    }

    fn all_reduce_sum(&self, x: &TensorRef, stream: StreamId) -> Result<()> {
        self.inner.all_reduce_sum(x, stream)
    }

    fn all_reduce_sum_into(&self, partial: &TensorRef, out: &TensorRef, stream: StreamId) -> Result<()> {
        if !Self::eligible(partial, out, stream) {
            return self.inner.all_reduce_sum_into(partial, out, stream);
        }
        partial.check_bounds()?;
        out.check_bounds()?;
        self.check_ring()?;
        self.counters.call(partial.nbytes());
        self.backend.bind()?;
        let src = CudaBackend::ptr(&partial.buf, partial.offset, partial.nbytes())?;
        let dst = CudaBackend::ptr(&out.buf, out.offset, out.nbytes())?;
        if src as usize % 16 != 0 || dst as usize % 4 != 0 {
            return Err(Error::backend("RDMA all-reduce: the partial must be 16-byte and the output 4-byte aligned"));
        }
        let s = self.backend.stream(stream)?;
        // SAFETY: src holds n f32 and dst n bf16 (bounds checked above), n is a multiple of 4 and within the ring; same stream.
        let rc = unsafe { ffi::ik_rdma_exchange(self.ring.0, src as *const c_void, dst as *mut c_void, partial.numel(), s) };
        if rc != 0 {
            self.check_ring()?;
            return Err(Error::backend(format!("RDMA all-reduce launch failed (cudaError {rc})")));
        }
        Ok(())
    }

    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()> {
        self.inner.all_gather(send, recv, stream)
    }
    fn all_gather_quantum(&self) -> Option<usize> {
        self.inner.all_gather_quantum()
    }

    fn abort(&self) {
        // SAFETY: valid ring handle; sets the abort word the kernels' waits read.
        unsafe { ffi::ik_rdma_abort(self.ring.0) };
        self.inner.abort();
    }

    fn stats(&self) -> CommStats {
        let (a, b) = (self.inner.stats(), self.counters.snapshot());
        CommStats { calls: a.calls + b.calls, bytes: a.bytes + b.bytes, timed_calls: a.timed_calls, time_us: a.time_us }
    }
    fn note_timed(&self, d: Duration) {
        self.inner.note_timed(d)
    }
    fn debug_state(&self) -> Option<String> {
        self.inner.debug_state()
    }
    fn debug_counters(&self) -> Option<[u32; 2]> {
        self.inner.debug_counters()
    }
    fn capture_safe(&self) -> bool {
        self.inner.capture_safe()
    }
    fn reserve(&self, max_elems: usize, stream: StreamId) -> Result<()> {
        self.inner.reserve(max_elems, stream)
    }
    fn fuses_ar_norm(&self) -> bool {
        self.inner.fuses_ar_norm()
    }
    fn ar_norm_supported(&self, a: &ArNorm<'_>) -> bool {
        self.inner.ar_norm_supported(a)
    }
    fn all_reduce_add_rmsnorm(&self, a: &ArNorm<'_>, stream: StreamId) -> Result<()> {
        self.inner.all_reduce_add_rmsnorm(a, stream)
    }
    fn prepush(&self, partial: &TensorRef) -> Result<bool> {
        self.inner.prepush(partial)
    }
}

impl Drop for RdmaComm {
    fn drop(&mut self) {
        // SAFETY: valid ring handle; stops the proxy thread. The rings stay mapped (a graph may hold their addresses).
        unsafe { ffi::ik_rdma_close(self.ring.0) };
    }
}
