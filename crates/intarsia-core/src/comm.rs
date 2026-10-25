//! Collective communication between the ranks of a tensor-parallel model (ARCHITECTURE 5).
//!
//! One process drives every rank, each from its own thread, on that rank's own backend (device) and
//! stream. A [`Comm`] is one rank's handle onto a group; the model calls it between the sharded
//! matmuls. The contract:
//!
//! * **Collective.** Every rank of the group must make the same sequence of calls with the same element
//!   counts and dtypes. A rank that fails must call [`Comm::abort`] so the others return an error
//!   instead of waiting forever.
//! * **Stream ordered.** On return the result is complete *as seen by later work on `stream`*; the caller
//!   does not synchronise. (A device implementation enqueues; the host-staged one blocks.)
//! * **Bit-identical results on every rank.** The reduction of two ranks is `a + b` in f32 with a single
//!   rounding to the tensor dtype (addition is commutative, so both ranks agree); with more ranks the sum
//!   runs in rank order. Ranks keep their own copy of the residual stream, so this is what keeps those
//!   copies equal.
//!
//! [`HostCommGroup`] is the backend-neutral implementation (download, exchange through host memory, sum,
//! upload). It works on any backend including the cpu reference, which is how tensor parallelism is tested
//! without a GPU. A device backend provides a faster group (HIP: peer-to-peer, `intarsia-backend-hip`).

use crate::backend::{Backend, StreamId, TensorRef};
use crate::dtype::{decode_to_f32, encode_from_f32, DType};
use crate::error::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long a rank waits for the others before the collective fails (a peer that died or hung).
pub const COMM_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct CommStats {
    /// Collective calls made by this rank.
    pub calls: u64,
    /// Payload bytes reduced / gathered by this rank (the local tensor size per call).
    pub bytes: u64,
    /// Calls whose duration was measured (profiled steps synchronise the stream around the call).
    pub timed_calls: u64,
    /// Measured time over `timed_calls`, microseconds.
    pub time_us: f64,
}

/// Atomic counters behind [`CommStats`], for implementors.
#[derive(Debug, Default)]
pub struct CommCounters {
    calls: AtomicU64,
    bytes: AtomicU64,
    timed: AtomicU64,
    nanos: AtomicU64,
}

impl CommCounters {
    pub fn call(&self, bytes: usize) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    pub fn timed(&self, d: Duration) {
        self.timed.fetch_add(1, Ordering::Relaxed);
        self.nanos.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }
    pub fn snapshot(&self) -> CommStats {
        CommStats {
            calls: self.calls.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            timed_calls: self.timed.load(Ordering::Relaxed),
            time_us: self.nanos.load(Ordering::Relaxed) as f64 / 1e3,
        }
    }
}

/// One rank's handle onto a communicator group.
pub trait Comm: Send + Sync {
    /// Implementation name for the plan ("host", "hip-p2p", ...).
    fn name(&self) -> &str;
    fn rank(&self) -> usize;
    fn world(&self) -> usize;

    /// In-place sum over all ranks of the contiguous tensor `x` (bf16, f16 or f32) living on this rank's
    /// backend. Every rank passes a tensor of the same dtype and element count.
    fn all_reduce_sum(&self, x: &TensorRef, stream: StreamId) -> Result<()>;

    /// `out = round_to(out.dtype)(sum over ranks of partial)`: the reduction takes each rank's `partial` as is (f32,
    /// bf16 or f16) and rounds once, into `out` (any of those dtypes, same element count). `partial` is left
    /// untouched and may be a different buffer than `out`. Summing f32 partials and rounding once matches what one
    /// unsharded matmul does (f32 accumulate, one rounding), where bf16 partials would round twice.
    fn all_reduce_sum_into(&self, partial: &TensorRef, out: &TensorRef, stream: StreamId) -> Result<()>;

    /// `recv` (`world * send.numel()` elements) receives every rank's `send` in rank order. Same dtype on
    /// both. `recv` may not alias `send`.
    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()>;

    /// Release peers blocked in a collective with an error. Idempotent; the group stays unusable after.
    fn abort(&self);

    /// Counters since creation.
    fn stats(&self) -> CommStats;

    /// Record the measured duration of one collective (profiled steps).
    fn note_timed(&self, d: Duration);

    /// Can the collectives be recorded into a captured graph (CUDA graph) on the rank's stream and replayed? True only
    /// when every call is a pure stream operation with no host synchronisation and no allocation once
    /// [`Comm::reserve`] ran. A group that says no keeps its model's decode eager at world > 1.
    fn capture_safe(&self) -> bool {
        false
    }

    /// Allocate now whatever a reduction of up to `max_elems` f32 partials needs, so that no later call allocates or
    /// moves a buffer (a captured graph keeps the addresses it saw). Calls above `max_elems` fail afterwards.
    fn reserve(&self, _max_elems: usize, _stream: StreamId) -> Result<()> {
        Ok(())
    }

    /// Does this communicator fuse the all-reduce with the residual add + rmsnorm that follows it
    /// ([`Comm::all_reduce_add_rmsnorm`])? Default false: the model then never defers a reduction, and its op sequence is
    /// exactly the unfused one.
    fn fuses_ar_norm(&self) -> bool {
        false
    }

    /// Would [`Comm::all_reduce_add_rmsnorm`] take these arguments? A pure function of the tensors' dtypes, shapes and
    /// offsets (never of their contents), so every rank, the plan recorder and a captured graph decide the same.
    fn ar_norm_supported(&self, _a: &ArNorm<'_>) -> bool {
        false
    }

    /// The all-reduce of [`Comm::all_reduce_sum_into`] followed by `Op::RmsNormAdd` on its result, as one collective:
    /// `out = round(sum over ranks of partial)`, `sum = round(out + residual)`, `norm = rmsnorm(sum) * (weight + w_offset)`,
    /// bit for bit the unfused pair (HIP: `kernels/hip/ar_norm.hip`). Same collective contract as the other calls; the
    /// caller asks [`Comm::ar_norm_supported`] first and runs the unfused pair when it says no.
    fn all_reduce_add_rmsnorm(&self, _a: &ArNorm<'_>, _stream: StreamId) -> Result<()> {
        Err(Error::Unsupported(format!("{} has no fused all-reduce + rmsnorm", self.name())))
    }
}

/// Operands of [`Comm::all_reduce_add_rmsnorm`]: the `[rows, H]` activations are in the model's activation dtype.
pub struct ArNorm<'a> {
    /// This rank's partial of the block output (f32, or the activation dtype), `rows * H` elements.
    pub partial: &'a TensorRef,
    /// Receives the reduced block output (the `x` of the `RmsNormAdd`; may be `partial` for equal dtypes).
    pub out: &'a TensorRef,
    /// The residual stream before this block's add.
    pub residual: &'a TensorRef,
    /// The norm weight `[H]`.
    pub weight: &'a TensorRef,
    /// The normed rows (first output of `RmsNormAdd`).
    pub norm: &'a TensorRef,
    /// The residual stream after the add (second output of `RmsNormAdd`).
    pub sum: &'a TensorRef,
    pub eps: f32,
    pub w_offset: f32,
}

fn check_reducible(x: &TensorRef) -> Result<usize> {
    if !matches!(x.dtype, DType::Bf16 | DType::F16 | DType::F32) {
        return Err(Error::invalid(format!("all_reduce_sum supports bf16/f16/f32, got {}", x.dtype.name())));
    }
    x.check_bounds()?;
    Ok(x.numel())
}

// ---------------------------------------------------------------------------
// host-staged implementation
// ---------------------------------------------------------------------------

struct RvState {
    generation: u64,
    arrived: usize,
    slots: Vec<Option<Vec<u8>>>,
    published: Option<Arc<Vec<Vec<u8>>>>,
    aborted: bool,
}

/// A reusable all-to-all rendezvous: every rank contributes bytes and gets everyone's back.
struct Rendezvous {
    world: usize,
    st: Mutex<RvState>,
    cv: Condvar,
}

impl Rendezvous {
    fn new(world: usize) -> Rendezvous {
        Rendezvous {
            world,
            st: Mutex::new(RvState { generation: 0, arrived: 0, slots: vec![None; world], published: None, aborted: false }),
            cv: Condvar::new(),
        }
    }

    fn abort(&self) {
        self.st.lock().unwrap().aborted = true;
        self.cv.notify_all();
    }

    fn exchange(&self, rank: usize, mine: Vec<u8>, timeout: Duration) -> Result<Arc<Vec<Vec<u8>>>> {
        let deadline = Instant::now() + timeout;
        let mut st = self.st.lock().unwrap();
        if st.aborted {
            return Err(Error::backend("communicator aborted (another rank failed)"));
        }
        st.slots[rank] = Some(mine);
        st.arrived += 1;
        if st.arrived == self.world {
            let all: Vec<Vec<u8>> = st.slots.iter_mut().map(|s| s.take().unwrap()).collect();
            st.published = Some(Arc::new(all));
            st.arrived = 0;
            st.generation += 1;
            self.cv.notify_all();
            return Ok(st.published.clone().unwrap());
        }
        let gen = st.generation;
        while st.generation == gen {
            if st.aborted {
                return Err(Error::backend("communicator aborted (another rank failed)"));
            }
            let now = Instant::now();
            if now >= deadline {
                st.aborted = true;
                self.cv.notify_all();
                return Err(Error::backend(format!(
                    "collective timed out after {timeout:?}: {} of {} ranks arrived",
                    st.arrived, self.world
                )));
            }
            st = self.cv.wait_timeout(st, deadline - now).unwrap().0;
        }
        if st.aborted {
            return Err(Error::backend("communicator aborted (another rank failed)"));
        }
        Ok(st.published.clone().unwrap())
    }
}

/// A group of ranks that exchange tensors through host memory. Backend neutral and slow (a download, a
/// rendezvous and an upload per call); the reference implementation and the test double.
pub struct HostCommGroup;

impl HostCommGroup {
    /// One [`Comm`] per backend, rank `i` bound to `backends[i]`.
    pub fn new(backends: Vec<Arc<dyn Backend>>) -> Vec<Arc<dyn Comm>> {
        Self::with_timeout(backends, COMM_TIMEOUT)
    }

    pub fn with_timeout(backends: Vec<Arc<dyn Backend>>, timeout: Duration) -> Vec<Arc<dyn Comm>> {
        let world = backends.len();
        let rv = Arc::new(Rendezvous::new(world));
        backends
            .into_iter()
            .enumerate()
            .map(|(rank, backend)| {
                Arc::new(HostComm { rank, world, backend, rv: rv.clone(), timeout, counters: CommCounters::default() })
                    as Arc<dyn Comm>
            })
            .collect()
    }
}

struct HostComm {
    rank: usize,
    world: usize,
    backend: Arc<dyn Backend>,
    rv: Arc<Rendezvous>,
    timeout: Duration,
    counters: CommCounters,
}

impl HostComm {
    fn read(&self, t: &TensorRef, stream: StreamId) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; t.nbytes()];
        self.backend.download(&t.buf, t.offset, &mut bytes, stream)?;
        Ok(bytes)
    }
}

impl Comm for HostComm {
    fn name(&self) -> &str {
        "host"
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
        let n = check_reducible(partial)?;
        if check_reducible(out)? != n {
            return Err(Error::invalid(format!("all_reduce_sum_into: {n} partial elements into {} output elements", out.numel())));
        }
        self.counters.call(partial.nbytes());
        if self.world == 1 {
            let same = partial.buf.ptr_eq(&out.buf) && partial.offset == out.offset && partial.dtype == out.dtype;
            if !same {
                // nothing to sum, but `out` is still owed the (converted) value
                let mut vals = vec![0f32; n];
                decode_to_f32(partial.dtype, &self.read(partial, stream)?, &mut vals);
                let mut bytes = vec![0u8; out.nbytes()];
                encode_from_f32(out.dtype, &vals, &mut bytes);
                return self.backend.upload(&out.buf, out.offset, &bytes, stream);
            }
            return Ok(());
        }
        let all = self.rv.exchange(self.rank, self.read(partial, stream)?, self.timeout)?;
        let mut acc = vec![0f32; n];
        decode_to_f32(partial.dtype, &all[0], &mut acc);
        let mut tmp = vec![0f32; n];
        for part in all.iter().skip(1) {
            decode_to_f32(partial.dtype, part, &mut tmp);
            for (a, b) in acc.iter_mut().zip(&tmp) {
                *a += *b;
            }
        }
        let mut bytes = vec![0u8; out.nbytes()];
        encode_from_f32(out.dtype, &acc, &mut bytes);
        self.backend.upload(&out.buf, out.offset, &bytes, stream)
    }

    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()> {
        send.check_bounds()?;
        recv.check_bounds()?;
        if send.dtype != recv.dtype || recv.numel() != send.numel() * self.world {
            return Err(Error::invalid(format!(
                "all_gather: recv {:?} {} is not {} x send {:?} {}",
                recv.shape,
                recv.dtype.name(),
                self.world,
                send.shape,
                send.dtype.name()
            )));
        }
        self.counters.call(send.nbytes());
        let all = self.rv.exchange(self.rank, self.read(send, stream)?, self.timeout)?;
        let cat: Vec<u8> = all.iter().flat_map(|p| p.iter().copied()).collect();
        self.backend.upload(&recv.buf, recv.offset, &cat, stream)
    }

    fn abort(&self) {
        self.rv.abort();
    }
    fn stats(&self) -> CommStats {
        self.counters.snapshot()
    }
    fn note_timed(&self, d: Duration) {
        self.counters.timed(d);
    }
}
