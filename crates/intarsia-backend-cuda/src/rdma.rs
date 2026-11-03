//! Direct RDMA all-reduce for the TP=2 f32 partial sums (lane dsv41/rdma, `INTARSIA_DSV41_RDMA=1`;
//! docs/results/2026-11-02-dsv41-rdma.md). A [`Comm`] that wraps the NCCL communicator and takes over exactly one call:
//! `all_reduce_sum_into` of an f32 partial into a bf16 output on stream 0 (80 of a decode pass's 83 collectives), through the
//! stage / collect kernel pair and host proxy of `kernels/cuda/ik_rdma.cu`. Every other call is NCCL's. The kernels are plain
//! stream work with their sequence number on the device, so a recorded CUDA graph replays them like any other node.
//!
//! The choice per call depends only on dtype, element count and stream, which both ranks see alike. The result is bitwise the
//! NCCL pair's (`ncclAllReduce` f32 + `ik_rt_f32_round`): at world 2 `a + b == b + a`, rounded once to nearest even.
//! With the switch off no wrapper exists: [`create_with_rdma`] returns the NCCL communicator itself.

use crate::device::CudaBackend;
use crate::ffi;
use intarsia_core::backend::{StreamId, TensorRef};
use intarsia_core::comm::{ArNorm, Comm, CommCounters, CommStats};
use intarsia_core::{DType, Error, Result};
use std::ffi::{c_char, c_int, c_void, CStr};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

/// Largest partial the rings carry: 256 KiB of f32 (12 rows of the 5120-wide model); bigger calls (a prefill chunk) go to NCCL.
const MAX_BYTES: usize = 256 * 1024;
const SLOTS: c_int = 4;

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

/// Both ranks say whether they run the RDMA exchange; a difference is an error (one side would wait for bytes that never
/// come). Rank 0 writes first, rank 1 reads first; both read what the other sent.
pub fn negotiate(ctl: &mut TcpStream, rank: usize, mine: bool) -> Result<bool> {
    let io = |e: std::io::Error| Error::backend(format!("INTARSIA_DSV41_RDMA handshake with the other rank: {e}"));
    let mut theirs = [0u8; 1];
    if rank == 0 {
        ctl.write_all(&[mine as u8]).map_err(io)?;
        ctl.flush().map_err(io)?;
        ctl.read_exact(&mut theirs).map_err(io)?;
    } else {
        ctl.read_exact(&mut theirs).map_err(io)?;
        ctl.write_all(&[mine as u8]).map_err(io)?;
        ctl.flush().map_err(io)?;
    }
    if (theirs[0] != 0) != mine {
        return Err(Error::invalid(format!(
            "INTARSIA_DSV41_RDMA is {} on this rank and {} on the other: set it the same on both",
            if mine { "1" } else { "unset" },
            if theirs[0] != 0 { "1" } else { "unset" }
        )));
    }
    Ok(mine)
}

/// The RDMA devices and GID index of the ring: the ones NCCL is told to use (`NCCL_IB_HCA`, one or two names, a `=` or `^`
/// prefix not allowed; `NCCL_IB_GID_INDEX`, default 3), so a launch needs no new variable.
fn devices_from_env() -> Result<(String, c_int)> {
    let hca = std::env::var("NCCL_IB_HCA").unwrap_or_default();
    if hca.is_empty() || hca.starts_with(['=', '^']) {
        return Err(Error::invalid("INTARSIA_DSV41_RDMA needs NCCL_IB_HCA to name the RDMA device(s), e.g. rocep1s0f1,roceP2p1s0f1".to_string()));
    }
    let gid = match std::env::var("NCCL_IB_GID_INDEX") {
        Ok(v) => v.trim().parse::<c_int>().map_err(|_| Error::invalid(format!("NCCL_IB_GID_INDEX '{v}' is not a number")))?,
        Err(_) => 3,
    };
    if hca.contains(':') {
        return Err(Error::invalid("INTARSIA_DSV41_RDMA: NCCL_IB_HCA names devices without ports (port 1 is used)".to_string()));
    }
    Ok((hca, gid))
}

/// Open the ring, swap queue-pair info with the other rank over the control stream, connect and start the proxy.
pub fn wrap(backend: Arc<CudaBackend>, inner: Arc<dyn Comm>, rank: usize, ctl: &mut TcpStream) -> Result<Arc<dyn Comm>> {
    // SAFETY: plain FFI call.
    if unsafe { ffi::ik_rdma_available() } == 0 {
        return Err(Error::Unsupported("INTARSIA_DSV41_RDMA=1: this binary was built without libibverbs".into()));
    }
    let (devs, gid) = devices_from_env()?;
    backend.bind()?;
    let cdevs = std::ffi::CString::new(devs.clone()).map_err(|e| Error::invalid(e.to_string()))?;
    let mut err = [0 as c_char; 256];
    // SAFETY: valid NUL-terminated device list and error buffer.
    let h = unsafe { ffi::ik_rdma_open(cdevs.as_ptr(), gid, rank as c_int, MAX_BYTES, SLOTS, err.as_mut_ptr(), err.len() as c_int) };
    if h.is_null() {
        return Err(Error::backend(format!("RDMA ring on {devs} (gid {gid}): {}", lib_err(&err))));
    }
    let ring = Ring(h);
    // SAFETY: plain FFI call.
    let n = unsafe { ffi::ik_rdma_info_size() } as usize;
    let mut mine = vec![0u8; n];
    // SAFETY: `mine` holds the info block's size.
    unsafe { ffi::ik_rdma_info(h, mine.as_mut_ptr() as *mut c_void) };
    let mut theirs = vec![0u8; n];
    let io = |e: std::io::Error| Error::backend(format!("RDMA queue-pair info exchange: {e}"));
    if rank == 0 {
        ctl.write_all(&mine).map_err(io)?;
        ctl.flush().map_err(io)?;
        ctl.read_exact(&mut theirs).map_err(io)?;
    } else {
        ctl.read_exact(&mut theirs).map_err(io)?;
        ctl.write_all(&mine).map_err(io)?;
        ctl.flush().map_err(io)?;
    }
    // SAFETY: `theirs` holds the peer's info block of the same size.
    let rc = unsafe { ffi::ik_rdma_connect(h, theirs.as_ptr() as *const c_void, err.as_mut_ptr(), err.len() as c_int) };
    if rc != 0 {
        return Err(Error::backend(format!("RDMA connect: {}", lib_err(&err))));
    }
    // both ranks connected their queue pairs before either starts sending: one more round on the control stream
    let (a, b) = ([1u8], &mut [0u8; 1]);
    if rank == 0 {
        ctl.write_all(&a).map_err(io)?;
        ctl.read_exact(b).map_err(io)?;
    } else {
        ctl.read_exact(b).map_err(io)?;
        ctl.write_all(&a).map_err(io)?;
    }
    // SAFETY: plain FFI call; -1: the proxy thread is not pinned.
    unsafe { ffi::ik_rdma_start(h, -1) };
    eprintln!("tensor parallel: rank {rank} f32 all-reduces over RDMA ({devs}, gid {gid}, up to {} KiB), the rest over NCCL (INTARSIA_DSV41_RDMA)", MAX_BYTES / 1024);
    Ok(Arc::new(RdmaComm { inner, backend, ring, counters: CommCounters::default() }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn negotiate_agrees_and_both_ranks_refuse_a_difference() {
        for (a, b, ok) in [(true, true, true), (false, false, true), (true, false, false), (false, true, false)] {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap();
            let t = std::thread::spawn(move || negotiate(&mut l.accept().unwrap().0, 1, b));
            let mut c = TcpStream::connect(addr).unwrap();
            let r0 = negotiate(&mut c, 0, a);
            let r1 = t.join().unwrap();
            assert_eq!((r0.is_ok(), r1.is_ok()), (ok, ok), "rank 0 {a}, rank 1 {b}");
            if ok {
                assert_eq!((r0.unwrap(), r1.unwrap()), (a, b));
            }
        }
    }

    #[test]
    fn devices_come_from_the_nccl_variables() {
        // the env is process-wide: set and read within one test
        std::env::set_var("NCCL_IB_HCA", "rocep1s0f1,roceP2p1s0f1");
        std::env::set_var("NCCL_IB_GID_INDEX", "3");
        assert_eq!(devices_from_env().unwrap(), ("rocep1s0f1,roceP2p1s0f1".to_string(), 3));
        std::env::set_var("NCCL_IB_HCA", "mlx5_0:1");
        assert!(devices_from_env().is_err());
        std::env::set_var("NCCL_IB_HCA", "^mlx5_1");
        assert!(devices_from_env().is_err());
        std::env::remove_var("NCCL_IB_HCA");
        assert!(devices_from_env().is_err());
    }
}
