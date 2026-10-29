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

use crate::device::{HipBackend, PushArm};
use crate::ffi;
use intarsia_core::backend::{StreamId, TensorRef};
use intarsia_core::comm::{ArNorm, Comm, CommCounters, CommStats};
use intarsia_core::{DType, Error, Result};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Blocks per launch. Fixed: every launch advances every block's counter, which keeps both ranks' counters equal.
const NBLOCKS: i32 = 16;
/// One scratch slot (there are two per rank). Larger messages are split.
const SLOT_BYTES: usize = 4 << 20;
/// Largest message (bytes) that takes the LL path when `INTARSIA_HIP_AR_DRAIN=7` (`INTARSIA_HIP_AR_LL_MAX` overrides): the LL lines
/// double the bytes on the wire, so bigger messages keep the fence path.
const LL_MAX_DEFAULT: usize = 64 << 10;
/// The wide fence path (lane pf27b): all-reduces above the LL limit (every prefill all-reduce) run on their own scratch, flags and
/// counters with more blocks and one slot that holds a whole 2048-row x 5120 bf16 partial, so a prefill all-reduce is one launch on
/// most of the card instead of five 4 MiB launches on 16 blocks. `INTARSIA_HIP_AR_WIDE=BLOCKS[,SLOT_MIB]` overrides, `0` turns it off.
const WIDE_BLOCKS_DEFAULT: i32 = 64;
const WIDE_SLOT_DEFAULT: usize = 20 << 20;

fn rt(what: &str, rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        // SAFETY: hipGetErrorString returns a static string.
        let msg = unsafe { std::ffi::CStr::from_ptr(ffi::ik_rt_error_string(rc)) }.to_string_lossy().into_owned();
        Err(Error::backend(format!("{what}: {msg} (hipError {rc})")))
    }
}

fn host_ctrs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_HIP_CTRS_HOST").is_ok_and(|v| v == "1"))
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
    /// 2 slots of `ll_slot_bytes`, fine-grained and zero-filled, the peer's tagged lines go here (null when LL is off)
    ll: *mut c_void,
}

// SAFETY: plain device addresses; all access is from kernels launched on the owning/peer device.
unsafe impl Send for RankBuf {}
unsafe impl Sync for RankBuf {}

impl RankBuf {
    fn new(backend: &HipBackend, ll_slot_bytes: usize) -> Result<RankBuf> {
        Self::sized(backend, SLOT_BYTES, NBLOCKS as usize, ll_slot_bytes)
    }

    /// `slot_bytes` per scratch slot (two slots), `nblocks` flags and counters
    fn sized(backend: &HipBackend, slot_bytes: usize, nblocks: usize, ll_slot_bytes: usize) -> Result<RankBuf> {
        backend.bind()?;
        let ordinal = backend.ordinal();
        let (mut s, mut f, mut c, mut l): (*mut c_void, *mut c_void, *mut c_void, *mut c_void) =
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: valid out-pointers, device bound above.
        unsafe {
            rt("fine-grained scratch", ffi::ik_rt_malloc_finegrained(&mut s, 2 * slot_bytes))?;
            rt("fine-grained flags", ffi::ik_rt_malloc_finegrained(&mut f, nblocks * 4))?;
            if ll_slot_bytes > 0 {
                rt("fine-grained LL scratch", ffi::ik_rt_malloc_finegrained(&mut l, 2 * ll_slot_bytes))?;
            }
            if host_ctrs() {
                // INTARSIA_HIP_CTRS_HOST=1 (debugging, w27): the launch counters live in pinned host memory (the kernels reach it through the unified
                // address space), so the host can read them at a hang without any HIP call
                rt("host counters", ffi::ik_rt_host_alloc(&mut c, nblocks * 4))?;
                std::ptr::write_bytes(c as *mut u8, 0, nblocks * 4);
            } else {
                rt("counters", ffi::ik_rt_malloc(&mut c, nblocks * 4))?;
            }
        }
        let rb = RankBuf { ordinal, scratch: s, flags: f as *mut u32, ctrs: c as *mut u32, ll: l };
        let st = backend.stream(0)?;
        // SAFETY: freshly allocated ranges of the stated sizes.
        unsafe {
            rt("zero flags", ffi::ik_rt_memset(f, 0, nblocks * 4, st))?;
            if !host_ctrs() {
                rt("zero counters", ffi::ik_rt_memset(c, 0, nblocks * 4, st))?;
            }
            rt("zero scratch", ffi::ik_rt_memset(s, 0, 2 * slot_bytes, st))?;
            if ll_slot_bytes > 0 {
                // zero tags: no line is valid before its launch writes it (the counters start at 1)
                rt("zero LL scratch", ffi::ik_rt_memset(l, 0, 2 * ll_slot_bytes, st))?;
            }
            rt("sync", ffi::ik_rt_stream_sync(st))?;
        }
        Ok(rb)
    }
}

impl Drop for RankBuf {
    fn drop(&mut self) {
        // INTARSIA_HIP_COMM_LEAK=1 (debugging, w25): never free the handshake buffers, so no later group can get the same address range
        if std::env::var("INTARSIA_HIP_COMM_LEAK").is_ok_and(|v| v == "1") {
            return;
        }
        // SAFETY: allocated by ik_rt_malloc*/finegrained on `ordinal`; freed once.
        unsafe {
            ffi::ik_rt_set_device(self.ordinal);
            ffi::ik_rt_free(self.scratch);
            ffi::ik_rt_free(self.flags as *mut c_void);
            if !self.ll.is_null() {
                ffi::ik_rt_free(self.ll);
            }
            if host_ctrs() {
                ffi::ik_rt_host_free(self.ctrs as *mut c_void);
            } else {
                ffi::ik_rt_free(self.ctrs as *mut c_void);
            }
        }
    }
}

/// The wide fence path's own buffers (see `WIDE_BLOCKS_DEFAULT`): separate from `Group::bufs`, so its counters, flags and scratch
/// parities advance only with its own launches (every one of them `nblocks` blocks) and the 16-block lockstep of the other
/// collectives is untouched.
struct Wide {
    bufs: [RankBuf; 2],
    nblocks: i32,
    slot_bytes: usize,
    /// launches each rank has enqueued on these buffers (lane arx): the device counters' value once they ran, so the next launch's
    /// `seq` is this + 1 and its slot `(this + 1) & 1`, which a GEMM armed to push (`P2pComm::prepush`) writes into ahead of it
    launches: [AtomicU64; 2],
    /// all-reduces (both ranks) that only reduced because their GEMM had pushed (`debug_state`, tests)
    pushed: AtomicU64,
}

struct Group {
    bufs: [RankBuf; 2],
    /// the wide fence path for all-reduces above `wide_min` bytes (None: off, they take `bufs` as before)
    wide: Option<Wide>,
    /// the largest all-reduce (bytes) that keeps the old path: the LL limit (drain 7 / 8) or `LL_MAX_DEFAULT`
    wide_min: usize,
    /// bytes of one LL slot (twice the largest LL message), 0 when the LL path is off
    ll_slot_bytes: usize,
    /// lane arx, `INTARSIA_HIP_AR_PUSH=1`: the wide all-reduce of a partial whose GEMM can push it (`Comm::prepush`) skips its own push
    push: bool,
    aborted: AtomicBool,
    /// `INTARSIA_TP_COLL_TRACE=1` (debugging, w26): every collective launch of each rank, packed as `op << 60 | dtype << 56 | out dtype << 52 |
    /// blocks << 40 | elements` (op 1 all-reduce, 2 all-gather, 3 all-reduce + rmsnorm). `debug_state` reports the first launch where the ranks differ.
    calls: Option<std::sync::Mutex<[Vec<u64>; 2]>>,
}

impl Group {
    fn note(&self, rank: usize, op: u64, dt: i32, odt: i32, n: usize) {
        if let Some(c) = &self.calls {
            let mut c = c.lock().unwrap();
            if c[rank].len() < 4_000_000 {
                c[rank].push(op << 60 | ((dt as u64) & 15) << 56 | ((odt as u64) & 15) << 52 | (NBLOCKS as u64) << 40 | (n as u64 & ((1 << 40) - 1)));
            }
        }
    }
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
    p2p_group_mode(backends, None, None)
}

/// [`p2p_group`] with the handshake mode given explicitly (tests): `drain` is `INTARSIA_HIP_AR_DRAIN`'s value and `ll_max` the
/// `INTARSIA_HIP_AR_LL_MAX` bytes; `None` reads the environment like `p2p_group`.
pub fn p2p_group_mode(backends: &[Arc<HipBackend>], drain: Option<i32>, ll_max: Option<usize>) -> Result<Vec<Arc<dyn Comm>>> {
    p2p_group_wide(backends, drain, ll_max, None)
}

/// The wide path's (blocks, slot bytes) from `INTARSIA_HIP_AR_WIDE` (`BLOCKS[,SLOT_MIB]`, `0` = off), else the defaults.
fn wide_env() -> Result<Option<(i32, usize)>> {
    let v = match std::env::var("INTARSIA_HIP_AR_WIDE") {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(Some((WIDE_BLOCKS_DEFAULT, WIDE_SLOT_DEFAULT))),
    };
    let bad = || Error::invalid(format!("INTARSIA_HIP_AR_WIDE={v} (BLOCKS[,SLOT_MIB], 1..=1024 blocks, or 0 for off)"));
    let mut it = v.split(',');
    let blocks: i32 = it.next().and_then(|b| b.trim().parse().ok()).ok_or_else(bad)?;
    if blocks == 0 {
        return Ok(None);
    }
    let slot = match it.next() {
        Some(m) => m.trim().parse::<usize>().ok().filter(|&m| (1..=256).contains(&m)).ok_or_else(bad)? << 20,
        None => WIDE_SLOT_DEFAULT,
    };
    if !(1..=1024).contains(&blocks) || it.next().is_some() {
        return Err(bad());
    }
    Ok(Some((blocks, slot)))
}

/// [`p2p_group_mode`] with the wide fence path given explicitly (tests and benchmarks): `wide` is `Some(Some((blocks, slot_bytes)))`,
/// `Some(None)` for off, `None` to read `INTARSIA_HIP_AR_WIDE`.
#[allow(clippy::type_complexity)]
pub fn p2p_group_wide(
    backends: &[Arc<HipBackend>],
    drain: Option<i32>,
    ll_max: Option<usize>,
    wide: Option<Option<(i32, usize)>>,
) -> Result<Vec<Arc<dyn Comm>>> {
    p2p_group_push(backends, drain, ll_max, wide, None)
}

/// [`p2p_group_wide`] with the GEMM push (lane arx, `INTARSIA_HIP_AR_PUSH`) given explicitly: `Some(on)`, or `None` to read the variable.
/// Only with the wide path on.
#[allow(clippy::type_complexity)]
pub fn p2p_group_push(
    backends: &[Arc<HipBackend>],
    drain: Option<i32>,
    ll_max: Option<usize>,
    wide: Option<Option<(i32, usize)>>,
    push: Option<bool>,
) -> Result<Vec<Arc<dyn Comm>>> {
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
    // INTARSIA_HIP_AR_DRAIN: how a rank makes its pushed data visible before it raises the flag: 1 = a system fence in
    // every thread (removes the rare stale peer read, 0/250 tapped fresh processes; costs ~2% decode on the 27B at
    // tp=2, see docs/results/2026-10-28-det-tp2-nondeterminism.md section 5; the fallback), 7 = (default) tagged-line (LL)
    // handshake for small messages: the reader validates the data itself, no flag, no fence (docs/results/2026-10-28-ars-ll-handshake.md),
    // 3 = wait for the stores to complete
    // (radiance's cheaper mode, the scratch is uncached memory: ~1.5% of fresh tp=2 processes add a stale peer row, a data
    // race, so the sums are wrong)
    let drain = match drain {
        Some(d) => d,
        None => match std::env::var("INTARSIA_HIP_AR_DRAIN").ok().as_deref() {
            None | Some("") | Some("7") => 7,
            Some("1") => 1,
            Some("3") => 3,
            // diagnostics (w26): 4 = mode 3 and the flag re-sent while waiting, 5 = mode 3 and ~25 us before the data is read
            Some("4") => 4,
            Some("5") => 5,
            // lane det: 6 = mode 3, then every thread reads back the last word it pushed before the flag goes out
            Some("6") => 6,
            // lane ars: 7 = tagged-line (LL) handshake for small messages, mode 1 above INTARSIA_HIP_AR_LL_MAX and for the fused
            // f32 hidden >= 4096 all-reduce+norm; 8 = 7 without the tag check (a negative control for the tests, wrong sums)
            Some("8") => 8,
            // lane pf27b: 9 = the fence path without its wait (tests only: a negative control, wrong sums)
            Some(v) => return Err(Error::invalid(format!("INTARSIA_HIP_AR_DRAIN={v} (1, 3, 4, 5, 6, 7 or 8)"))),
        },
    };
    let ll_slot_bytes = if drain >= 7 {
        let max = match ll_max {
            Some(m) => m,
            None => match std::env::var("INTARSIA_HIP_AR_LL_MAX").ok().as_deref() {
                None | Some("") => LL_MAX_DEFAULT,
                Some(v) => v.parse::<usize>().map_err(|_| Error::invalid(format!("INTARSIA_HIP_AR_LL_MAX={v} (a byte count)")))?,
            },
        };
        // a message of m bytes takes 2 m bytes of lines; a multiple of 16 so the parity slots stay aligned
        (2 * max.min(SLOT_BYTES / 2)).div_ceil(16) * 16
    } else {
        0
    };
    let wide = match wide {
        Some(w) => w,
        None => wide_env()?,
    };
    let wide = match wide {
        Some((nblocks, slot_bytes)) => Some(Wide {
            bufs: [
                RankBuf::sized(&backends[0], slot_bytes, nblocks as usize, 0)?,
                RankBuf::sized(&backends[1], slot_bytes, nblocks as usize, 0)?,
            ],
            nblocks,
            slot_bytes,
            launches: [AtomicU64::new(0), AtomicU64::new(0)],
            pushed: AtomicU64::new(0),
        }),
        None => None,
    };
    let push = match push {
        Some(p) => p,
        None => match std::env::var("INTARSIA_HIP_AR_PUSH").ok().as_deref() {
            Some("0") => false,
            None | Some("") | Some("1") => true,
            Some(v) => return Err(Error::invalid(format!("INTARSIA_HIP_AR_PUSH={v} (0 or 1)"))),
        },
    } && wide.is_some();
    let wide_min = if ll_slot_bytes > 0 { ll_slot_bytes / 2 } else { LL_MAX_DEFAULT };
    let group = Arc::new(Group {
        bufs: [RankBuf::new(&backends[0], ll_slot_bytes)?, RankBuf::new(&backends[1], ll_slot_bytes)?],
        wide,
        wide_min,
        ll_slot_bytes,
        push,
        aborted: AtomicBool::new(false),
        calls: std::env::var("INTARSIA_TP_COLL_TRACE").is_ok_and(|v| v == "1").then(|| std::sync::Mutex::new([Vec::new(), Vec::new()])),
    });
    if std::env::var("INTARSIA_HIP_COMM_LOG").is_ok_and(|v| v == "1") {
        for (r, b) in group.bufs.iter().enumerate() {
            eprintln!("COMM group: rank {r} device {} scratch {:p} flags {:p} ctrs {:p}", b.ordinal, b.scratch, b.flags, b.ctrs);
        }
    }
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
        // lane arx: an armed push applies to this call or to none (spent either way)
        let arm = if self.group.push { self.backend.swap_push_arm(None) } else { None };
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
        // above the LL limit the wide fence path, when on (a pure function of the message size, so both ranks agree)
        let wide = self.group.wide.as_ref().filter(|_| total > self.group.wide_min);
        let (bufs, slot, nblocks, ll_bytes) = match wide {
            Some(w) => (&w.bufs, w.slot_bytes, w.nblocks, 0usize),
            None => (&self.group.bufs, SLOT_BYTES, NBLOCKS, self.group.ll_slot_bytes),
        };
        let (me, peer) = (&bufs[self.rank], &bufs[1 - self.rank]);
        if let (Some(w), Some(a)) = (wide, arm) {
            if a.lay.pushed == 1 && a.out == pbase as usize && a.bytes == total && out.dtype == DType::Bf16 {
                // the GEMM pushed this partial into the slot of launch `a.launch`: nothing else may have run on the wide buffers since
                let n = w.launches[self.rank].fetch_add(1, Ordering::Relaxed);
                if n != a.launch {
                    return Err(Error::backend(format!("hip all_reduce: GEMM pushed for wide launch {} but this is launch {n}", a.launch)));
                }
                self.group.note(self.rank, 5, partial.dtype as i32, out.dtype as i32, partial.numel());
                // SAFETY: as below; `a.lay` is what the GEMM launch returned for exactly this partial.
                let rc = unsafe {
                    ffi::ik_ar2_pushed(pbase as *mut c_void, obase as *mut c_void, me.scratch, peer.flags, me.flags, me.ctrs, slot as i64, nblocks, self.drain, &a.lay, s)
                };
                if rc != 0 {
                    return Err(Error::backend(format!("ik_ar2_pushed returned {} ({rc})", ffi::status_name(rc))));
                }
                w.pushed.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        }
        let (mut off, n_total) = (0usize, partial.numel());
        let per = slot / pe; // elements per launch
        while off < n_total {
            let n = per.min(n_total - off);
            self.group.note(self.rank, if wide.is_some() { 4 } else { 1 }, partial.dtype as i32, out.dtype as i32, n);
            if let Some(w) = wide {
                w.launches[self.rank].fetch_add(1, Ordering::Relaxed);
            }
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
                    slot as i64,
                    nblocks,
                    self.drain,
                    peer.ll,
                    me.ll,
                    ll_bytes as i64,
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

    /// Lane arx (`INTARSIA_HIP_AR_PUSH=1`): arm the GEMM launch that writes `partial` (`matmul_mxfp4_a8_mma`'s direct kernel) to push
    /// it into the peer's slot of the next wide launch, for a bf16 partial on the wide path that fits one slot. The next collective spends
    /// the arm: its all-reduce of this partial then only reduces (`ik_ar2_pushed`) when the GEMM did push, else pushes as before.
    fn prepush(&self, partial: &TensorRef) -> Result<bool> {
        let Some(w) = self.group.wide.as_ref().filter(|_| self.group.push) else { return Ok(false) };
        let total = partial.nbytes();
        if partial.dtype != DType::Bf16 || total <= self.group.wide_min || total > w.slot_bytes || total % 16 != 0 {
            return Ok(false);
        }
        partial.check_bounds()?;
        let out = HipBackend::ptr(&partial.buf, partial.offset, total)? as usize;
        let launch = w.launches[self.rank].load(Ordering::Relaxed);
        let peer = w.bufs[1 - self.rank].scratch as usize + ((launch + 1) & 1) as usize * w.slot_bytes;
        let lay = ffi::IkArPush { cap: w.slot_bytes as i64, ..Default::default() };
        self.backend.swap_push_arm(Some(PushArm { out, bytes: total, peer, launch, lay }));
        Ok(true)
    }

    /// Rank-major gather (`ag2_kernel`): `recv[r * send.numel() ..]` is rank r's `send`. `send` may be this rank's own
    /// segment of `recv` (in place: the kernel skips the self copy), or disjoint from it. The same push handshake, scratch,
    /// flags and counters as the all-reduce, so both ranks must make the same sequence of collectives and either kind may
    /// follow the other. Messages above one scratch slot go in several launches.
    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()> {
        if self.group.push {
            self.backend.swap_push_arm(None); // an armed push is spent by the next collective (lane arx)
        }
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
            self.group.note(self.rank, 2, send.dtype as i32, send.dtype as i32, n);
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
                    peer.ll,
                    me.ll,
                    self.group.ll_slot_bytes as i64,
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
    fn debug_counters(&self) -> Option<[u32; 2]> {
        if !host_ctrs() {
            return None;
        }
        // SAFETY: pinned host memory of NBLOCKS u32, live as long as the group
        let rd = |b: &RankBuf| unsafe { std::ptr::read_volatile(b.ctrs) };
        Some([rd(&self.group.bufs[0]), rd(&self.group.bufs[1])])
    }

    fn debug_state(&self) -> Option<String> {
        // Read both ranks' launch counters and flags on a fresh non-blocking stream: it does not wait for a collective kernel that is
        // spinning on the rank's own stream. Equal counters across the ranks = the ranks launched the same number of collective kernels;
        // a flag is the last sequence number the peer raised in that block.
        let mut out = String::new();
        for (r, b) in self.group.bufs.iter().enumerate() {
            if host_ctrs() {
                // no HIP call: safe at a hang. (The flags are device memory; only the counters are read here.)
                let c: Vec<u32> = (0..NBLOCKS as usize).map(|i| unsafe { std::ptr::read_volatile(b.ctrs.add(i)) }).collect();
                let (lo, hi) = (c.iter().min().unwrap(), c.iter().max().unwrap());
                out.push_str(&format!("rank {r}: HOST launch counters {}; ", if lo == hi { format!("{lo}") } else { format!("{lo}..{hi} {c:?}") }));
                continue;
            }
            let mut s: *mut c_void = std::ptr::null_mut();
            let (mut c, mut f) = ([0u32; NBLOCKS as usize], [0u32; NBLOCKS as usize]);
            // SAFETY: valid out-pointers and host arrays of NBLOCKS u32; the device pointers are the group's own allocations.
            let rc = unsafe {
                ffi::ik_rt_set_device(b.ordinal);
                let mut rc = ffi::ik_rt_stream_create(&mut s);
                if rc == 0 {
                    rc = ffi::ik_rt_memcpy_d2h(c.as_mut_ptr() as *mut c_void, b.ctrs as *const c_void, NBLOCKS as usize * 4, s);
                }
                if rc == 0 {
                    rc = ffi::ik_rt_memcpy_d2h(f.as_mut_ptr() as *mut c_void, b.flags as *const c_void, NBLOCKS as usize * 4, s);
                }
                rc
            };
            if rc != 0 {
                out.push_str(&format!("rank {r}: read failed ({rc}); "));
                continue;
            }
            let summ = |v: &[u32]| {
                let (lo, hi) = (v.iter().min().unwrap(), v.iter().max().unwrap());
                if lo == hi { format!("{lo}") } else { format!("{lo}..{hi} {v:?}") }
            };
            out.push_str(&format!("rank {r}: launch counters {} flags {}; ", summ(&c), summ(&f)));
        }
        if let Some(w) = &self.group.wide {
            out.push_str(&format!(
                "wide launches {} / {}, pushed {}; ",
                w.launches[0].load(Ordering::Relaxed),
                w.launches[1].load(Ordering::Relaxed),
                w.pushed.load(Ordering::Relaxed)
            ));
        }
        if let Some(c) = &self.group.calls {
            let c = c.lock().unwrap();
            let dec = |w: u64| format!("op{} dt{} odt{} blocks{} n{}", w >> 60, (w >> 56) & 15, (w >> 52) & 15, (w >> 40) & 0xfff, w & ((1 << 40) - 1));
            out.push_str(&format!("host call log: rank 0 {} launches, rank 1 {}; ", c[0].len(), c[1].len()));
            match (0..c[0].len().max(c[1].len())).find(|&i| c[0].get(i) != c[1].get(i)) {
                Some(i) => out.push_str(&format!(
                    "FIRST DIFFERENCE at launch {i}: rank 0 {}, rank 1 {}",
                    c[0].get(i).map_or("none".into(), |&w| dec(w)),
                    c[1].get(i).map_or("none".into(), |&w| dec(w))
                )),
                None => out.push_str("the ranks' launch sequences are identical"),
            }
        }
        Some(out)
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
        if self.group.push {
            self.backend.swap_push_arm(None); // an armed push is spent by the next collective (lane arx)
        }
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
        self.group.note(self.rank, 3, a.partial.dtype as i32, a.partial.dtype as i32, rows * h);
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
                peer.ll,
                me.ll,
                self.group.ll_slot_bytes as i64,
                s,
            )
        };
        if rc != 0 {
            return Err(Error::backend(format!("ik_ar2_add_rmsnorm returned {} ({rc})", ffi::status_name(rc))));
        }
        Ok(())
    }
}

/// Lane pf27b's P2P micro-benchmark (`tests/comm.rs` `p2p_link_rate`, ignored): the link's copy-engine rate (`hipMemcpyAsync` into
/// the peer, both ranks at once and rank 0 alone) and the fence all-reduce's phases alone (`ik_ar2_bench`: the push into the peer's
/// fine-grained scratch with and without the system fence, and the local reduce) per block count. `us/call` is wall time per launch
/// pair over `iters` back-to-back launches; GB/s is bytes per rank per second. The returned text is one line per measurement.
#[doc(hidden)]
pub fn p2p_bench(backends: &[Arc<HipBackend>], sizes: &[usize], blocks: &[i32], iters: usize) -> Result<String> {
    if backends.len() != 2 {
        return Err(Error::invalid("p2p_bench needs two devices"));
    }
    for (i, a) in backends.iter().enumerate() {
        // SAFETY: plain FFI.
        rt("hipDeviceEnablePeerAccess", unsafe { ffi::ik_rt_enable_peer_access(a.ordinal(), backends[1 - i].ordinal()) })?;
    }
    let max = sizes.iter().copied().max().unwrap_or(0).div_ceil(16) * 16;
    // per rank: x, out (ordinary), scr (fine-grained, the peer pushes here), dst (ordinary, the peer's copy engine writes here)
    let mut bufs: Vec<[usize; 4]> = Vec::new();
    for b in backends {
        b.bind()?;
        let mut p = [std::ptr::null_mut::<c_void>(); 4];
        // SAFETY: valid out-pointers, device bound.
        unsafe {
            rt("x", ffi::ik_rt_malloc(&mut p[0], max))?;
            rt("out", ffi::ik_rt_malloc(&mut p[1], max))?;
            rt("scr", ffi::ik_rt_malloc_finegrained(&mut p[2], max))?;
            rt("dst", ffi::ik_rt_malloc(&mut p[3], max))?;
            let st = b.stream(0)?;
            for q in p {
                rt("memset", ffi::ik_rt_memset(q, 0, max, st))?;
            }
            rt("sync", ffi::ik_rt_stream_sync(st))?;
        }
        bufs.push(p.map(|q| q as usize));
    }
    // both ranks (or rank 0 only) run `f(rank, stream)` iters times after a barrier; us per launch
    let time = |both: bool, f: &(dyn Fn(usize, *mut c_void) -> i32 + Sync)| -> Result<f64> {
        let ranks = if both { 2 } else { 1 };
        let bar = std::sync::Barrier::new(ranks);
        let mut res = Ok(0.0);
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..ranks)
                .map(|r| {
                    let (b, bar) = (&backends[r], &bar);
                    sc.spawn(move || -> Result<f64> {
                        b.bind()?;
                        let st = b.stream(0)?;
                        for _ in 0..3 {
                            rt("warm", f(r, st))?;
                        }
                        // SAFETY: the rank's own stream
                        rt("sync", unsafe { ffi::ik_rt_stream_sync(st) })?;
                        bar.wait();
                        let t0 = std::time::Instant::now();
                        for _ in 0..iters {
                            rt("launch", f(r, st))?;
                        }
                        // SAFETY: as above
                        rt("sync", unsafe { ffi::ik_rt_stream_sync(st) })?;
                        Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
                    })
                })
                .collect();
            let mut worst: f64 = 0.0;
            for h in hs {
                match h.join().unwrap() {
                    Ok(us) => worst = worst.max(us),
                    Err(e) => res = Err(e),
                }
            }
            if res.is_ok() {
                res = Ok(worst);
            }
        });
        res
    };
    let mut out = String::new();
    let mut line = |what: &str, bytes: usize, us: f64| {
        out.push_str(&format!("P2P {what:<28} {bytes:>10} B {us:>9.1} us {:>7.2} GB/s\n", bytes as f64 / us / 1e3));
    };
    let p = |r: usize, i: usize| bufs[r][i] as *mut c_void;
    for &bytes in sizes {
        let n = bytes.div_ceil(16) * 16;
        let n16 = (n / 16) as i64;
        // SAFETY (all closures): every range is inside the buffers allocated above (n <= max).
        line("copy both", n, time(true, &|r, st| unsafe { ffi::ik_rt_memcpy_d2d(p(1 - r, 3), p(r, 0), n, st) })?);
        line("copy rank0", n, time(false, &|r, st| unsafe { ffi::ik_rt_memcpy_d2d(p(1 - r, 3), p(r, 0), n, st) })?);
        for &nb in blocks {
            line(&format!("push+fence both nb={nb}"), n, time(true, &|r, st| unsafe { ffi::ik_ar2_bench(0, p(r, 0), p(r, 1), p(1 - r, 2), p(r, 2), n16, nb, st) })?);
            line(&format!("push nofence both nb={nb}"), n, time(true, &|r, st| unsafe { ffi::ik_ar2_bench(2, p(r, 0), p(r, 1), p(1 - r, 2), p(r, 2), n16, nb, st) })?);
            line(&format!("push+fence rank0 nb={nb}"), n, time(false, &|r, st| unsafe { ffi::ik_ar2_bench(0, p(r, 0), p(r, 1), p(1 - r, 2), p(r, 2), n16, nb, st) })?);
            line(&format!("reduce local nb={nb}"), n, time(false, &|r, st| unsafe { ffi::ik_ar2_bench(1, p(r, 0), p(r, 1), p(1 - r, 2), p(r, 2), n16, nb, st) })?);
        }
    }
    for (r, b) in backends.iter().enumerate() {
        // SAFETY: allocated above on this device, freed once
        unsafe {
            ffi::ik_rt_set_device(b.ordinal());
            for &q in &bufs[r] {
                ffi::ik_rt_free(q as *mut c_void);
            }
        }
    }
    Ok(out)
}
