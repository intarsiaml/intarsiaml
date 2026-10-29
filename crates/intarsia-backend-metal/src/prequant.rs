//! Lane mnq: the producer of an fp8 GEMV's input also writes q(x) (`INTARSIA_METAL_NORM_FP8`, default on, `=0` is main).
//!
//! The decode fp8 GEMV (`fp8_gemv_*_m1`) stages q(x) = bf16(e4m3 QDQ(x / input_scale)) of its activation row in threadgroup
//! memory once per threadgroup (a divide, an encode and a decode per element, in every one of its few hundred threadgroups,
//! before the first weight byte is used). The kernel that produced x already holds x in registers, so it writes that row's q(x)
//! as well (`rmsnorm*_exq*`, `silu_glu_bf16_q`, `sigmoid_mul_bf16_q`: the bf16 output as before, plus `fp8_q_bf16` of the stored
//! value) and the GEMV (`fp8_gemv_*_m1_xd`) reads it from device memory: no staging pass, no threadgroup memory. The values
//! are the staging loop's, the GEMV's fma chain is the one-row kernel's, so the output bits are main's.
//!
//! The input scale is the consumer's (a static W8A8 scale per weight; the producer has no notion of the consumer). So a
//! producer that could feed a one-row fp8 GEMV is not encoded when it is launched but held ([`Pq::pending`], at most one):
//!  * the next launch is an M = 1 `matmul_fp8_gemv` whose x is the held producer's output and whose input scale is > 0: the
//!    producer is encoded as the q variant, bound to that GEMV's scales and the q slot, and the GEMV as `_m1_xd` ([`MetalBackend::pq_claim`]);
//!  * any other dispatch, commit, flush, synchronize, hold or discard encodes it first as the plain kernel
//!    (`Stream::drain_pending`, called by every path that encodes or commits), so the stream order is unchanged.
//!
//! The slot ([`Pq::slot`], one tracked buffer) then holds q(x) of one tensor ([`Pq::live`]: where it is, which scale it was
//! made with). The next GEMV on the same x (the GDN z after qkv, the attention k and v after q) uses it only if its input
//! scale is bit-equal to the one the slot was made with (read from the scales' shared memory); anything that writes x or the
//! slot's source range since drops it ([`MetalBackend::pq_note_writes`], and the host-side writers), and any other case runs
//! the staging GEMV on the bf16 x, which the producer wrote either way.
//!
//! Profiling (`INTARSIA_METAL_PROFILE`) labels dispatches by launch, so with a profile on the producers are encoded at once and
//! the handoff never happens.

use crate::dense::{Bind, Grid, OpP};
use crate::{set_bytes, MetalBackend, MtlBuffer, Pso};
use intarsia_core::backend::TensorRef;
use intarsia_core::{DType, Result};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// Bytes of the q(x) slot: rows of at most 8192 bf16 elements (the widest fp8 projection input of the models served here is 4096).
pub(crate) const SLOT_BYTES: usize = 16384;

/// `INTARSIA_METAL_NORM_FP8`: unset / `1` hands off from `silu_glu` and `sigmoid_mul` (the GDN out projection's and the
/// attention o_proj's input), `0` turns the handoff off (every producer and every fp8 GEMV is main's again), `2` adds the
/// row norms (`rmsnorm`, `rmsnorm_add`: slower, see docs/results/2026-10-29-mnq-metal-norm-fp8.md).
pub fn level() -> u8 {
    static L: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *L.get_or_init(|| match std::env::var("INTARSIA_METAL_NORM_FP8").as_deref() {
        Ok("0") => 0,
        Ok("2") => 2,
        _ => 1,
    })
}

pub fn enabled() -> bool {
    level() > 0
}

/// The row norms also write q(x) (`INTARSIA_METAL_NORM_FP8=2`).
pub(crate) fn norm_producers() -> bool {
    level() >= 2
}

/// A bf16 tensor's place: (MTLBuffer address, byte address inside it, elements).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct XKey {
    buf: usize,
    addr: usize,
    n: usize,
}

/// A producer launch held back until the next launch shows whether an fp8 GEMV reads its output.
pub(crate) struct Prod {
    key: XKey,
    pso: Pso,
    pso_q: Pso,
    binds: Vec<Bind>,
    p: OpP,
    grid: Grid,
    /// Set by the GEMV that claimed the output: the q slot and that GEMV's scales; the producer then runs as `pso_q`.
    q: Option<(MtlBuffer, Bind)>,
}

impl Prod {
    pub(crate) fn encode(&self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        let pso = if self.q.is_some() { &self.pso_q } else { &self.pso };
        let mut n = self.binds.len();
        // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`; the slot is SLOT_BYTES).
        unsafe {
            enc.setComputePipelineState(pso);
            for (i, b) in self.binds.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
            }
            if let Some((slot, sc)) = &self.q {
                enc.setBuffer_offset_atIndex(Some(slot), 0, n);
                enc.setBuffer_offset_atIndex(Some(&sc.mtl), sc.off, n + 1);
                n += 2;
            }
        }
        set_bytes(enc, &self.p, n);
        match self.grid {
            Grid::Threads(t) => {
                let w = pso.maxTotalThreadsPerThreadgroup().min(256);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: t, height: 1, depth: 1 },
                    MTLSize { width: w, height: 1, depth: 1 },
                );
            }
            Grid::Groups { x, y, threads } => enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: x, height: y, depth: 1 },
                MTLSize { width: threads, height: 1, depth: 1 },
            ),
        }
    }
}

/// The handoff state (inside the stream lock).
#[derive(Default)]
pub(crate) struct Pq {
    pub(crate) pending: Option<Prod>,
    /// What the slot holds: the tensor and the bits of the input scale its q(x) was made with.
    live: Option<(XKey, u32)>,
    slot: Option<MtlBuffer>,
    /// claims of a held producer, reuses of the slot by a later GEMV (tests, diagnostics)
    hits: [u64; 2],
}

impl Pq {
    pub(crate) fn drop_all(&mut self) {
        self.pending = None;
        self.live = None;
    }
}

impl MetalBackend {
    fn xkey(&self, t: &TensorRef) -> Option<XKey> {
        let mb = self.mbuf(&t.buf).ok()?;
        Some(XKey { buf: objc2::rc::Retained::as_ptr(&mb.mtl) as *const () as usize, addr: mb.base + t.offset, n: t.numel() })
    }

    /// Hold a producer launch (see the module doc). `key_t` is the output the GEMV would read; `binds` the plain kernel's
    /// bindings; the q kernel takes the slot and the scales after them. Encoded at once when the handoff cannot apply.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pq_launch(
        &self,
        key_t: &TensorRef,
        name: &str,
        name_q: &str,
        binds: &[&Bind],
        p: &OpP,
        grid: Grid,
        threads: usize,
    ) -> Result<()> {
        let plain = |this: &Self| this.dispatch(name, binds, p, grid);
        if !enabled() || key_t.dtype != DType::Bf16 || key_t.numel() * 2 > SLOT_BYTES {
            return plain(self);
        }
        let (Some(key), Some(pso), Some(pso_q)) = (self.xkey(key_t), self.pso(name), self.pso(name_q)) else {
            return plain(self);
        };
        if pso_q.maxTotalThreadsPerThreadgroup() < threads {
            return plain(self);
        }
        let prod = Prod { key, pso: pso.clone(), pso_q: pso_q.clone(), binds: binds.iter().map(|b| (*b).clone()).collect(), p: *p, grid, q: None };
        let mut st = self.stream.lock().unwrap();
        if st.prof.is_some() {
            drop(st);
            return plain(self);
        }
        self.open_locked(&mut st)?;
        // an older held producer goes first
        Self::drain_pending(&mut st);
        st.pq.pending = Some(prod);
        Ok(())
    }

    /// The M = 1 fp8 GEMV on `x` with scales `sc` (bound as `sb`) asks for a q(x) slot: `Some(slot)` when the producer of `x`
    /// is held (it becomes the q variant) or the slot already holds q(x) of `x` made with this very input scale.
    pub(crate) fn pq_claim(&self, x: &TensorRef, sc: &TensorRef, sb: &Bind) -> Option<MtlBuffer> {
        if !enabled() || sb.skip != 0 || sc.dtype != DType::F32 || sc.numel() != 2 {
            return None;
        }
        let key = self.xkey(x)?;
        // the input scale: weights' constants in shared memory, read here; the same bits the GEMV reads on the GPU
        let mb = self.mbuf(&sc.buf).ok()?;
        // SAFETY: the tensor lies inside its buffer (`OpCall::check`), 2 f32; shared storage.
        let is = unsafe { (mb.host_ptr().add(sc.offset) as *const f32).add(1).read_unaligned() };
        if !(is > 0.0 && is.is_finite()) {
            return None;
        }
        let bits = is.to_bits();
        let mut st = self.stream.lock().unwrap();
        if st.pq.live.is_some_and(|(k, b)| k == key && b == bits) {
            st.pq.hits[1] += 1;
            return st.pq.slot.clone();
        }
        if st.pq.pending.as_ref().is_some_and(|p| p.key == key && p.q.is_none()) {
            if st.pq.slot.is_none() {
                st.pq.slot = Some(self.new_mtl_buffer(SLOT_BYTES, false).ok()?);
            }
            let slot = st.pq.slot.clone()?;
            st.pq.pending.as_mut().unwrap().q = Some((slot.clone(), sb.clone()));
            st.pq.live = Some((key, bits));
            st.pq.hits[0] += 1;
            return Some(slot);
        }
        None
    }

    /// A launch is about to write `call`'s outputs and scratch: the slot's q(x) of a tensor they overlap is stale.
    pub(crate) fn pq_note_writes(&self, call: &intarsia_core::OpCall) {
        let mut st = self.stream.lock().unwrap();
        let Some((key, _)) = st.pq.live else { return };
        let hit = call.outputs.iter().chain(call.scratch.iter()).any(|t| {
            let Ok(mb) = self.mbuf(&t.buf) else { return true };
            let buf = objc2::rc::Retained::as_ptr(&mb.mtl) as *const () as usize;
            let (lo, hi) = (mb.base + t.offset, mb.base + t.offset + t.numel() * t.dtype.size_bytes());
            buf == key.buf && lo < key.addr + key.n * 2 && key.addr < hi
        });
        if hit {
            st.pq.live = None;
        }
    }

    /// (claims of a held producer, reuses of the slot by a later GEMV) on this backend so far.
    pub fn norm_fp8_hits(&self) -> (u64, u64) {
        let h = self.stream.lock().unwrap().pq.hits;
        (h[0], h[1])
    }

    /// A host write or a copy into some buffer: forget the slot.
    pub(crate) fn pq_kill(&self) {
        self.stream.lock().unwrap().pq.live = None;
    }
}

impl MetalBackend {
    /// `INTARSIA_METAL_NORM_FP8_STATS=1`: print the hand-off counts (called when the backend is dropped).
    pub(crate) fn pq_report(&self) {
        if std::env::var_os("INTARSIA_METAL_NORM_FP8_STATS").is_some_and(|v| v != "0") {
            let (claims, reuses) = self.norm_fp8_hits();
            eprintln!("norm_fp8: level {} claims {claims} reuses {reuses}", level());
        }
    }
}
