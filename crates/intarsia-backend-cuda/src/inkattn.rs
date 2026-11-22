//! Kernel table and device dispatch of the Inkling-2-Mini attention ops (`InkKvPrologue`, `RelSinkAttn`, `InkAttnDecode`;
//! contract `intarsia_core::inkling2::ops`; kernels/cuda/inkling2_attn.cu with the design notes in its header, ABI
//! kernels/cuda/ik_cuda_abi_ink.h; lane ink-cuda-attn). Admitted against the cpu reference (`intarsia-backend-cpu`'s
//! `ink_ref`) by the ordinary sweep (`intarsia kernels test --backend cuda --op ink_kv_prologue|rel_sink_attn|ink_attn_decode`).
//! The table is data and compiles on every host; the dispatch (`launch_inkattn`, called from `device.rs::launch`) needs CUDA.
//!
//! * `ink_kv_prologue_bf16`: one launch (plus the conv-ring write pass when a long chunk would overwrite ring rows other rows
//!   still read: T > Rr - K + 1).
//! * `rel_sink_attn_bf16`: split-KV stage 1 ((kv head, split, row) blocks), the combine when a row can have more than one split,
//!   and the cache-write pass when the write could race the reads (ring, T > Cr - window + 1). Any T; rows-exact (row t of a
//!   T-row call is bitwise the 1-row call at its position, host or device position): the split count and the key partition
//!   are functions of the row's position and the window only.
//! * `ink_attn_decode_bf16` (op 75, 1..=16 rows): the two in one stage-1 kernel, bitwise the pair.

use crate::kernels::ARCH;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("ink_kv_prologue_bf16", Op::InkKvPrologue),
    ("rel_sink_attn_bf16", Op::RelSinkAttn),
    ("ink_attn_decode_bf16", Op::InkAttnDecode),
];

/// bf16-rounded outputs: a one-ulp flip is allowed (the backend's shared budget).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);

/// Scratch stride: the most partials per row (`MAXT` of the tile kernel, >= `MAXS` of the split-KV kernel).
pub const MAX_SPLIT: usize = 40;

/// Shapes the prologue takes (d_rel <= 32 is checked at launch).
fn prologue_ok(s: &ShapeInfo) -> bool {
    s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0 && s.n_heads / s.n_kv_heads <= 4 && (1..=256).contains(&s.head_dim)
}

/// Shapes the attention takes.
fn attn_ok(s: &ShapeInfo) -> bool {
    s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && (1..=3).contains(&(s.n_heads / s.n_kv_heads))
        && matches!(s.head_dim, 8 | 16 | 32 | 64 | 128)
        && s.m >= 1
}

fn decode_ok(s: &ShapeInfo) -> bool {
    attn_ok(s) && s.head_dim >= 16 && s.m <= intarsia_core::inkling2::ops::ATTN_DECODE_MAX_ROWS
}

/// Decode / verify rows (T <= 16): the tile partials, f32 `acc [T,H,MAXT,D]` + `(m, l) [T,H,MAXT]`. Prefill chunks run the
/// flash kernel (no scratch), except head_dim 8 or `INTARSIA_INK_PREFILL=splitkv`: the split-KV partials (`MAXS` = 32 per row).
pub fn attn_scratch(s: &ShapeInfo) -> usize {
    if s.m <= intarsia_core::inkling2::ops::ATTN_DECODE_MAX_ROWS {
        s.m * s.n_heads * MAX_SPLIT * (s.head_dim + 2) * 4
    } else if matches!(s.head_dim, 16 | 32 | 64 | 128) && !prefill_splitkv() {
        0
    } else {
        s.m * s.n_heads * 32 * (s.head_dim + 2) * 4
    }
}

/// `INTARSIA_INK_PREFILL=splitkv`: prefill chunks on the split-KV kernel (diagnostic; the kernel library reads the same variable).
fn prefill_splitkv() -> bool {
    std::env::var("INTARSIA_INK_PREFILL").is_ok_and(|v| v == "splitkv")
}

/// `attn_scratch` plus the staged K / V cache rows `[2][T][Hk][D]` bf16 of the fused kernel.
pub fn decode_scratch(s: &ShapeInfo) -> usize {
    attn_scratch(s) + 2 * s.m * s.n_kv_heads * s.head_dim * 2
}

pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    TABLE
        .iter()
        .map(|(name, op)| {
            // f32 math, bf16 stores at the contract's rounding points; rows-exact by construction
            let k = KernelImpl::new(*op, *name, "cuda").arch([ARCH]).acts(bf).exact(Exactness::Deterministic).tol(BF16_TOL);
            match *name {
                "ink_kv_prologue_bf16" => k.when(prologue_ok),
                "rel_sink_attn_bf16" => k.weights(bf).when(attn_ok).scratch(attn_scratch),
                "ink_attn_decode_bf16" => k.weights(bf).when(decode_ok).scratch(decode_scratch),
                other => unreachable!("inkling2 attention kernel {other} has no registration"),
            }
        })
        .collect()
}

/// Mirror of `ik_ink_attrs` in kernels/cuda/ik_cuda_abi_ink.h.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IkInkAttrs {
    pub n_heads: i64,
    pub n_kv_heads: i64,
    pub head_dim: i64,
    pub d_rel: i64,
    pub pos0: i64,
    pub window: i64,
    pub tau_floor: i64,
    pub dyn_conv: i32,
    pub kv_ring: i32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub eps: f32,
    pub tau_alpha: f32,
    pub scale: f32,
    pub split_keys: i32,
}

/// Keys per split: `INTARSIA_INK_SPLIT_KEYS` (diagnostic for sweeps; it changes the rounding order of every call alike, so
/// rows stay exact) or the kernels' default (0).
fn split_keys() -> i32 {
    static K: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("INTARSIA_INK_SPLIT_KEYS").ok().and_then(|v| v.trim().parse().ok()).filter(|&k: &i32| k >= 1).unwrap_or(0))
}

#[cfg(ik_cuda)]
mod launch {
    use super::IkInkAttrs;
    use crate::device::CudaBackend;
    use crate::ffi::{status_name, IkScratch, IkTensor};
    use intarsia_core::backend::StreamId;
    use intarsia_core::{Error, Op, OpCall, Result};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type SC = *const IkScratch;
    type S = *mut c_void;
    type A = *const IkInkAttrs;

    extern "C" {
        fn ik_ink_kv_prologue_bf16(pq: T, pk: T, pv: T, kt: T, vt: T, qs: T, rs: T, rp: T, a: A, q: TM, rel: TM, k: TM, v: TM, kr: TM, vr: TM, pos: T, s: S) -> c_int;
        fn ik_rel_sink_attn_bf16(q: T, k: T, v: T, kc_in: T, vc_in: T, rel: T, pq: T, a: A, o: TM, kc: TM, vc: TM, sc: SC, pos: T, s: S) -> c_int;
        fn ik_ink_attn_decode_bf16(
            pq: T,
            pk: T,
            pv: T,
            kt: T,
            vt: T,
            qs: T,
            rs: T,
            rp: T,
            kc_in: T,
            vc_in: T,
            a: A,
            o: TM,
            kc: TM,
            vc: TM,
            kr: TM,
            vr: TM,
            sc: SC,
            pos: T,
            s: S,
        ) -> c_int;
    }

    impl CudaBackend {
        /// `Some(result)` when `kernel` is an Inkling-2 attention launcher of this lane, `None` otherwise.
        pub(crate) fn launch_inkattn(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
            let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
            Some(self.launch_inkattn_inner(kernel, *op, call, stream))
        }

        fn launch_inkattn_inner(&self, kernel: &str, op: Op, call: &OpCall, stream: StreamId) -> Result<()> {
            if op != call.op {
                return Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op)));
            }
            self.bind()?;
            let s = self.stream(stream)?;
            let inp: Vec<IkTensor> = call.inputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let mut out: Vec<IkTensor> = call.outputs.iter().map(Self::tensor).collect::<Result<_>>()?;
            let scratch = match &call.scratch {
                Some(t) => {
                    let ik = Self::tensor(t)?;
                    Some(IkScratch { data: ik.data, bytes: t.nbytes() as i64 })
                }
                None => None,
            };
            let sc = scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch);
            let pos_t: Option<IkTensor> = call.pos.as_ref().map(Self::tensor).transpose()?;
            let pos = pos_t.as_ref().map_or(std::ptr::null(), |t| t as *const IkTensor);
            let a = &call.attrs;
            let attrs = IkInkAttrs {
                n_heads: a.n_heads as i64,
                n_kv_heads: a.n_kv_heads as i64,
                head_dim: a.head_dim as i64,
                d_rel: a.ink.d_rel as i64,
                pos0: a.pos0 as i64,
                window: a.window as i64,
                tau_floor: a.ink.tau_floor as i64,
                dyn_conv: a.ink.dyn_conv as i32,
                kv_ring: a.ink.kv_ring as i32,
                qk_clip: a.ink.qk_clip,
                rel_clip: a.ink.rel_clip,
                eps: a.eps,
                tau_alpha: a.ink.tau_alpha,
                scale: a.scale,
                split_keys: super::split_keys(),
            };
            let ap = &attrs as *const IkInkAttrs;
            let i = |n: usize| &inp[n] as *const IkTensor;
            let o = out.as_mut_ptr();
            let on = |n: usize| o.wrapping_add(n);
            // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the launchers validate
            // dtype / rank / shape / scratch and return an error status instead of faulting, and never synchronise or allocate.
            let rc = unsafe {
                match kernel {
                    "ink_kv_prologue_bf16" => {
                        ik_ink_kv_prologue_bf16(i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7), ap, on(0), on(1), on(2), on(3), on(4), on(5), pos, s)
                    }
                    "rel_sink_attn_bf16" => ik_rel_sink_attn_bf16(i(0), i(1), i(2), i(3), i(4), i(5), i(6), ap, on(0), on(1), on(2), sc, pos, s),
                    "ink_attn_decode_bf16" => ik_ink_attn_decode_bf16(
                        i(0),
                        i(1),
                        i(2),
                        i(3),
                        i(4),
                        i(5),
                        i(6),
                        i(7),
                        i(8),
                        i(9),
                        ap,
                        on(0),
                        on(1),
                        on(2),
                        on(3),
                        on(4),
                        sc,
                        pos,
                        s,
                    ),
                    other => return Err(Error::NoKernel(format!("cuda kernel '{other}'"))),
                }
            };
            if rc != 0 {
                return Err(Error::backend(format!(
                    "ik_{kernel} returned {} ({rc}); inputs {:?} shapes {:?}",
                    status_name(rc),
                    call.inputs.iter().map(|t| t.dtype).collect::<Vec<_>>(),
                    call.inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
                )));
            }
            Ok(())
        }
    }
}
