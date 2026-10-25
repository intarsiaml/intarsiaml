//! The HIP tensor-parallel collective: a two-rank one-shot all-reduce over PCIe peer-to-peer
//! (`kernels/hip/allreduce.hip`, design after radiance's libr4d, see `kernels/hip/NOTICE-radiance`) and, on the same
//! handshake, a two-rank all-gather (the vocab-parallel lm_head's logits).
//!
//! One process owns both GPUs. Each rank has, on its own device, a fine-grained (uncached) scratch buffer the peer
//! writes into, a block of flags the peer raises, and a counter block; a collective is a single kernel launch on the
//! rank's stream, made by the rank's own thread. There is no host rendezvous, no event and no copy engine in the
//! path, so the cost per call is one launch plus the on-device handshake.
//!
//! Both ranks must make the same sequence of calls (the per-block counters count launches and must stay equal).
//! Messages larger than one scratch slot are split into several launches.

use crate::device::HipBackend;
use crate::ffi;
use intarsia_core::backend::{StreamId, TensorRef};
use intarsia_core::comm::{ArNorm, Comm, CommCounters, CommStats};
use intarsia_core::{DType, Error, Result};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Blocks per launch. Fixed: every launch advances every block's counter, which keeps both ranks' counters equal.
const NBLOCKS: i32 = 16;
/// One scratch slot (there are two per rank). Larger messages are split.
const SLOT_BYTES: usize = 4 << 20;

fn rt(what: &str, rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        // SAFETY: hipGetErrorString returns a static string.
        let msg = unsafe { std::ffi::CStr::from_ptr(ffi::ik_rt_error_string(rc)) }.to_string_lossy().into_owned();
        Err(Error::backend(format!("{what}: {msg} (hipError {rc})")))
    }
}

/// What one rank owns on its device for the collective.
struct RankBuf {
    ordinal: i32,
    /// 2 slots of SLOT_BYTES, fine-grained; the peer pushes here
    scratch: *mut c_void,
    /// NBLOCKS u32, fine-grained; the peer raises these
    flags: *mut u32,
    /// NBLOCKS u32, ordinary device memory: this rank's launch counters
    ctrs: *mut u32,
}

// SAFETY: plain device addresses; all access is from kernels launched on the owning/peer device.
unsafe impl Send for RankBuf {}
unsafe impl Sync for RankBuf {}

impl RankBuf {
    fn new(backend: &HipBackend) -> Result<RankBuf> {
        backend.bind()?;
        let ordinal = backend.ordinal();
        let (mut s, mut f, mut c): (*mut c_void, *mut c_void, *mut c_void) =
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: valid out-pointers, device bound above.
        unsafe {
            rt("fine-grained scratch", ffi::ik_rt_malloc_finegrained(&mut s, 2 * SLOT_BYTES))?;
            rt("fine-grained flags", ffi::ik_rt_malloc_finegrained(&mut f, NBLOCKS as usize * 4))?;
            rt("counters", ffi::ik_rt_malloc(&mut c, NBLOCKS as usize * 4))?;
        }
        let rb = RankBuf { ordinal, scratch: s, flags: f as *mut u32, ctrs: c as *mut u32 };
        let st = backend.stream(0)?;
        // SAFETY: freshly allocated ranges of the stated sizes.
        unsafe {
            rt("zero flags", ffi::ik_rt_memset(f, 0, NBLOCKS as usize * 4, st))?;
            rt("zero counters", ffi::ik_rt_memset(c, 0, NBLOCKS as usize * 4, st))?;
            rt("zero scratch", ffi::ik_rt_memset(s, 0, 2 * SLOT_BYTES, st))?;
            rt("sync", ffi::ik_rt_stream_sync(st))?;
        }
        Ok(rb)
    }
}

impl Drop for RankBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by ik_rt_malloc*/finegrained on `ordinal`; freed once.
        unsafe {
            ffi::ik_rt_set_device(self.ordinal);
            ffi::ik_rt_free(self.scratch);
            ffi::ik_rt_free(self.flags as *mut c_void);
            ffi::ik_rt_free(self.ctrs as *mut c_void);
        }
    }
}

struct Group {
    bufs: [RankBuf; 2],
    aborted: AtomicBool,
}

struct P2pComm {
    rank: usize,
    backend: Arc<HipBackend>,
    group: Arc<Group>,
    drain: i32,
    counters: CommCounters,
}

/// The two-rank peer-to-peer group over `backends` (rank `i` = `backends[i]`). Fails when the two devices cannot
/// access each other's memory.
pub fn p2p_group(backends: &[Arc<HipBackend>]) -> Result<Vec<Arc<dyn Comm>>> {
    if backends.len() != 2 {
        return Err(Error::Unsupported(format!("the peer-to-peer all-reduce is a two-rank kernel, got {} ranks", backends.len())));
    }
    for (i, a) in backends.iter().enumerate() {
        let b = &backends[1 - i];
        let mut ok = 0;
        // SAFETY: valid out-pointer.
        rt("hipDeviceCanAccessPeer", unsafe { ffi::ik_rt_can_access_peer(a.ordinal(), b.ordinal(), &mut ok) })?;
        if ok == 0 {
            return Err(Error::Unsupported(format!(
                "HIP device {} cannot access device {} peer-to-peer (use --comm host)",
                a.ordinal(),
                b.ordinal()
            )));
        }
        // SAFETY: plain FFI.
        rt("hipDeviceEnablePeerAccess", unsafe { ffi::ik_rt_enable_peer_access(a.ordinal(), b.ordinal()) })?;
    }
    // INTARSIA_HIP_AR_DRAIN: how a rank makes its pushed data visible before it raises the flag: 3 = wait for the
    // stores to complete (default, radiance's cheaper mode; the scratch is uncached memory), 1 = a system fence in
    // every thread (safe baseline, about twice the latency)
    let drain = match std::env::var("INTARSIA_HIP_AR_DRAIN").ok().as_deref() {
        None | Some("") | Some("3") => 3,
        Some("1") => 1,
        Some(v) => return Err(Error::invalid(format!("INTARSIA_HIP_AR_DRAIN={v} (1 or 3)"))),
    };
    let group = Arc::new(Group {
        bufs: [RankBuf::new(&backends[0])?, RankBuf::new(&backends[1])?],
        aborted: AtomicBool::new(false),
    });
    Ok(backends
        .iter()
        .enumerate()
        .map(|(rank, b)| {
            Arc::new(P2pComm { rank, backend: b.clone(), group: group.clone(), drain, counters: CommCounters::default() })
                as Arc<dyn Comm>
        })
        .collect())
}

impl Comm for P2pComm {
    fn name(&self) -> &str {
        "hip-p2p"
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world(&self) -> usize {
        2
    }

    fn all_reduce_sum(&self, x: &TensorRef, stream: StreamId) -> Result<()> {
        self.all_reduce_sum_into(x, x, stream)
    }

    fn all_reduce_sum_into(&self, partial: &TensorRef, out: &TensorRef, stream: StreamId) -> Result<()> {
        if self.group.aborted.load(Ordering::Acquire) {
            return Err(Error::backend("communicator aborted (another rank failed)"));
        }
        // the kernel's (partial, out) pairs: bf16 -> bf16 and f32 -> f32 (in place or not), f32 -> bf16
        match (partial.dtype, out.dtype) {
            (DType::Bf16, DType::Bf16) | (DType::F32, DType::F32) | (DType::F32, DType::Bf16) => {}
            (a, b) => {
                return Err(Error::invalid(format!("hip all_reduce supports bf16->bf16, f32->f32, f32->bf16, got {}->{}", a.name(), b.name())))
            }
        }
        partial.check_bounds()?;
        out.check_bounds()?;
        if partial.numel() != out.numel() {
            return Err(Error::invalid(format!("hip all_reduce: {} partial elements into {} output elements", partial.numel(), out.numel())));
        }
        let (pe, oe) = (partial.dtype.size_bytes(), out.dtype.size_bytes());
        let total = partial.nbytes();
        if total % 16 != 0 {
            return Err(Error::invalid(format!("hip all_reduce needs a multiple of 16 bytes, got {total}")));
        }
        self.counters.call(total);
        self.backend.bind()?;
        let s = self.backend.stream(stream)?;
        let pbase = HipBackend::ptr(&partial.buf, partial.offset, total)?;
        let obase = HipBackend::ptr(&out.buf, out.offset, out.nbytes())?;
        let (me, peer) = (&self.group.bufs[self.rank], &self.group.bufs[1 - self.rank]);
        let (mut off, n_total) = (0usize, partial.numel());
        let per = SLOT_BYTES / pe; // elements per launch
        while off < n_total {
            let n = per.min(n_total - off);
            // SAFETY: both ranges are inside their tensors (bounds checked); scratch / flags / counters are live for
            // the group's lifetime, which the comm keeps alive through `group`.
            let rc = unsafe {
                ffi::ik_ar2_oneshot(
                    pbase.add(off * pe) as *mut c_void,
                    obase.add(off * oe) as *mut c_void,
                    partial.dtype as i32,
                    out.dtype as i32,
                    n as i64,
                    peer.scratch,
                    me.scratch,
                    peer.flags,
                    me.flags,
                    me.ctrs,
                    SLOT_BYTES as i64,
                    NBLOCKS,
                    self.drain,
                    s,
                )
            };
            if rc != 0 {
                return Err(Error::backend(format!("ik_ar2_oneshot returned {} ({rc})", ffi::status_name(rc))));
            }
            off += n;
        }
        Ok(())
    }

    fn all_gather_quantum(&self) -> Option<usize> {
        Some(16)
    }

    /// Rank-major gather (`ag2_kernel`): `recv[r * send.numel() ..]` is rank r's `send`. `send` may be this rank's own
    /// segment of `recv` (in place: the kernel skips the self copy), or disjoint from it. The same push handshake, scratch,
    /// flags and counters as the all-reduce, so both ranks must make the same sequence of collectives and either kind may
    /// follow the other. Messages above one scratch slot go in several launches.
    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()> {
        if self.group.aborted.load(Ordering::Acquire) {
            return Err(Error::backend("communicator aborted (another rank failed)"));
        }
        send.check_bounds()?;
        recv.check_bounds()?;
        if send.dtype != recv.dtype || recv.numel() != send.numel() * 2 {
            return Err(Error::invalid(format!(
                "hip all_gather: recv {:?} {} is not 2 x send {:?} {}",
                recv.shape,
                recv.dtype.name(),
                send.shape,
                send.dtype.name()
            )));
        }
        let total = send.nbytes();
        if total % 16 != 0 {
            return Err(Error::invalid(format!("hip all_gather needs a multiple of 16 bytes per rank, got {total}")));
        }
        self.counters.call(total);
        self.backend.bind()?;
        let s = self.backend.stream(stream)?;
        let sbase = HipBackend::ptr(&send.buf, send.offset, total)?;
        let rbase = HipBackend::ptr(&recv.buf, recv.offset, recv.nbytes())?;
        // in place (send is this rank's segment of recv) or disjoint; anything between would be read while it is written
        let (sa, ra) = (sbase as usize, rbase as usize);
        let mine = ra + self.rank * total;
        if sa != mine && sa < ra + 2 * total && ra < sa + total {
            return Err(Error::invalid("hip all_gather: send overlaps recv without being this rank's segment of it"));
        }
        let (me, peer) = (&self.group.bufs[self.rank], &self.group.bufs[1 - self.rank]);
        let mut off = 0usize;
        while off < total {
            let n = SLOT_BYTES.min(total - off);
            // SAFETY: the chunk is inside send, and inside both of recv's segments (bounds checked above); scratch / flags
            // / counters are live for the group's lifetime, which the comm keeps alive through `group`.
            let rc = unsafe {
                ffi::ik_ag2_gather(
                    sbase.add(off) as *mut c_void,
                    rbase.add(off) as *mut c_void,
                    n as i64,
                    total as i64,
                    peer.scratch,
                    me.scratch,
                    peer.flags,
                    me.flags,
                    me.ctrs,
                    SLOT_BYTES as i64,
                    NBLOCKS,
                    self.rank as i32,
                    self.drain,
                    s,
                )
            };
            if rc != 0 {
                return Err(Error::backend(format!("ik_ag2_gather returned {} ({rc})", ffi::status_name(rc))));
            }
            off += n;
        }
        Ok(())
    }

    fn abort(&self) {
        // A rank already spinning in the kernel gives up on its own after a bounded number of polls.
        self.group.aborted.store(true, Ordering::Release);
    }
    fn stats(&self) -> CommStats {
        self.counters.snapshot()
    }
    fn note_timed(&self, d: Duration) {
        self.counters.timed(d);
    }

    /// Yes: a collective is kernel launches on the caller's stream and nothing else (no host synchronisation, no event,
    /// no allocation), and everything the kernel keeps between calls lives on the device: the per-block launch counters
    /// (which pick the scratch parity and the flag value at run time, so a replayed launch takes the next sequence
    /// number like an eager one, and replays and eager calls can interleave), the flags and the fixed scratch slots. A
    /// recorded launch therefore stays valid for every replay. The f32 partial and the output are the model's
    /// workspace, so there is no comm-owned buffer that could grow and nothing to reserve. What a replay skips is the
    /// host-side `aborted` check; a replayed collective whose peer never arrives traps after the bounded wait, as an
    /// eager one already in the kernel does.
    fn capture_safe(&self) -> bool {
        true
    }

    fn fuses_ar_norm(&self) -> bool {
        true
    }
    fn ar_norm_supported(&self, a: &ArNorm<'_>) -> bool {
        self.ar_norm_shape(a).is_some()
    }
    fn all_reduce_add_rmsnorm(&self, a: &ArNorm<'_>, stream: StreamId) -> Result<()> {
        self.all_reduce_add_rmsnorm_impl(a, stream)
    }
}

impl P2pComm {
    /// `(rows, hidden)` when the fused all-reduce + residual add + rmsnorm (`kernels/hip/ar_norm.hip`) takes these operands,
    /// else None. A pure function of the tensors' dtypes, shapes and offsets, so both ranks, the plan recorder and a captured
    /// graph decide alike: bf16 activations `[rows, H]` with H a multiple of 8 (the norm's vector path), rows <= NBLOCKS, an
    /// f32 or bf16 partial, every operand 16-byte aligned, the partial within one scratch slot (decode and small verify
    /// steps; a prefill chunk keeps the unfused pair).
    fn ar_norm_shape(&self, a: &ArNorm<'_>) -> Option<(usize, usize)> {
        let act = [a.out, a.residual, a.weight, a.norm, a.sum];
        if act.iter().any(|t| t.dtype != DType::Bf16) || !matches!(a.partial.dtype, DType::F32 | DType::Bf16) {
            return None;
        }
        let h = a.weight.numel();
        if h < 8 || h % 8 != 0 || a.partial.numel() == 0 || a.partial.numel() % h != 0 {
            return None;
        }
        let rows = a.partial.numel() / h;
        let n = rows * h;
        // Decode-sized only: block b of the kernel owns whole rows and the grid is NBLOCKS, so more rows would be normed serially
        // within a block where the unfused norm runs one block per row (a prefill chunk would get slower). Above it, and above
        // one scratch slot, the model keeps the unfused pair.
        if rows > NBLOCKS as usize {
            return None;
        }
        if [a.out, a.residual, a.norm, a.sum].iter().any(|t| t.numel() != n || t.nbytes() != n * 2) || a.partial.nbytes() > SLOT_BYTES {
            return None;
        }
        // the unfused pair reads `out` and writes it in place for bf16 partials: it is `partial` or disjoint from it
        if a.partial.check_bounds().is_err() || act.iter().any(|t| t.check_bounds().is_err()) {
            return None;
        }
        if [a.partial, a.out, a.residual, a.weight, a.norm, a.sum].iter().any(|t| t.offset % 16 != 0) {
            return None;
        }
        Some((rows, h))
    }

    fn all_reduce_add_rmsnorm_impl(&self, a: &ArNorm<'_>, stream: StreamId) -> Result<()> {
        if self.group.aborted.load(Ordering::Acquire) {
            return Err(Error::backend("communicator aborted (another rank failed)"));
        }
        let (rows, h) = self
            .ar_norm_shape(a)
            .ok_or_else(|| Error::Unsupported("hip all_reduce_add_rmsnorm: operands not supported (ask ar_norm_supported)".into()))?;
        let total = a.partial.nbytes();
        self.counters.call(total);
        self.backend.bind()?;
        let s = self.backend.stream(stream)?;
        let ptr = |t: &TensorRef| HipBackend::ptr(&t.buf, t.offset, t.nbytes());
        let (me, peer) = (&self.group.bufs[self.rank], &self.group.bufs[1 - self.rank]);
        // SAFETY: every range is inside its tensor (`ar_norm_shape` bounds-checked them); scratch / flags / counters are
        // live for the group's lifetime, which the comm keeps alive through `group`.
        let rc = unsafe {
            ffi::ik_ar2_add_rmsnorm(
                ptr(a.partial)? as *mut c_void,
                ptr(a.out)? as *mut c_void,
                ptr(a.residual)? as *const c_void,
                ptr(a.weight)? as *const c_void,
                ptr(a.norm)? as *mut c_void,
                ptr(a.sum)? as *mut c_void,
                a.partial.dtype as i32,
                rows as i64,
                h as i64,
                a.eps,
                a.w_offset,
                peer.scratch,
                me.scratch,
                peer.flags,
                me.flags,
                me.ctrs,
                SLOT_BYTES as i64,
                NBLOCKS,
                self.drain,
                s,
            )
        };
        if rc != 0 {
            return Err(Error::backend(format!("ik_ar2_add_rmsnorm returned {} ({rc})", ffi::status_name(rc))));
        }
        Ok(())
    }
}
