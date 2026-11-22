//! Kernel table and device dispatch of the Inkling-2-Mini in-projection and sublayer-boundary ops (lane ink-cuda-mixmoe):
//! `InkInProj` (73), `ConvMix` (74), `AttnResMix` (67), `SconvRes` (69), `LoopBlend` (68). Kernels in kernels/cuda/ink_proj.cu
//! and ink_mix.cu, ABI kernels/cuda/ik_cuda_abi_ink.h; contract `intarsia_core::inkling2::ops`, admitted against the cpu
//! reference (`ink_ref`) by `intarsia kernels test --backend cuda --op <op>`. bf16 activations (the model's dtype).
//!
//! * `ink_in_proj_bf16_mma`: the tensor-core GEMV, one launch for the three segments, T <= 16 (decode, verify); its twin
//!   `_rows` is the same launcher (`rows_exact`: row t bitwise the one-row call, any T up to 16). `ink_in_proj_bf16_cublas`:
//!   prefill (T > 16), three cuBLAS GEMMs.
//! * The row kernels (ConvMix, AttnResMix, SconvRes): one block per row, the same code for every row count, so rows-exact and
//!   chunk-invariant (no twin needed). The blend (LoopBlend, ConvMix with msg) adds the h GEMV, the gate and the norm.

use crate::kernels::ARCH;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("ink_in_proj_bf16_mma", Op::InkInProj),
    ("ink_in_proj_bf16_mma_rows", Op::InkInProj),
    ("ink_in_proj_bf16_cublas", Op::InkInProj),
    ("conv_mix_bf16", Op::ConvMix),
    ("attn_res_mix_bf16", Op::AttnResMix),
    ("sconv_res_bf16", Op::SconvRes),
    ("loop_blend_bf16", Op::LoopBlend),
];

const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Most rows of the tensor-core GEMV form (ordinary calls; the launcher takes any T).
pub const PROJ_ROWS_MAX: usize = 16;
/// Widest row of the row kernels (6 elements per thread of 768), stored sources + last, streams and rank of the blend.
pub const MIX_D_MAX: usize = 4608;
pub const MIX_SOURCES_MAX: usize = 8;
pub const MIX_QUERIES_MAX: usize = 4;
pub const BLEND_STREAMS_MAX: usize = 3;
pub const BLEND_G_MAX: usize = 128;
/// Rows per blend chunk and channels per gate block (mirrors kBlendRows / kGateCh in ink_mix.cu).
const BLEND_ROWS: usize = 256;
const GATE_CH: usize = 128;

/// Scratch of the blend for `s` streams, `r` rows, width `d`, rank `g` (mirrors `blend_bytes` in ink_mix.cu).
pub fn blend_bytes(s: usize, r: usize, d: usize, g: usize) -> usize {
    let rnd = |v: usize| (v + 255) & !255;
    let rc = r.min(BLEND_ROWS);
    let nsl = d.div_ceil(GATE_CH);
    rnd(s * rc * 2 * d * 2) + rnd(s * rc * d * 4) + rnd(s * rc * g * 2) + rnd(s * rc * d * 4) + rnd(s * rc * nsl * 4)
}

/// ConvMix: the blend's scratch at its bounds (S and G are not in `ShapeInfo`): 3 streams, rank 128.
fn conv_mix_scratch(s: &ShapeInfo) -> usize {
    blend_bytes(BLEND_STREAMS_MAX, s.m, s.k, BLEND_G_MAX)
}

/// LoopBlend: S = `s.n` streams, rank at its bound.
fn loop_blend_scratch(s: &ShapeInfo) -> usize {
    blend_bytes(s.n.max(1), s.m, s.k, BLEND_G_MAX)
}

pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "cuda").arch([ARCH]).acts(bf).exact(Exactness::Deterministic);
            match *name {
                "ink_in_proj_bf16_mma" => {
                    k.weights(bf).when(|s| (1..=PROJ_ROWS_MAX).contains(&s.m) && s.k % 16 == 0).tol(BF16_TOL).priority(10)
                }
                // the verify / batched twin: the same launcher (`rows_exact`: only `exact_rows` calls select it)
                "ink_in_proj_bf16_mma_rows" => k
                    .weights(bf)
                    .when(|s| (1..=PROJ_ROWS_MAX).contains(&s.m) && s.k % 16 == 0)
                    .tol(BF16_TOL)
                    .priority(-1)
                    .rows_exact(),
                // prefill: cuBLAS (tensor cores; the algorithm may depend on the shape), as `matmul_bf16`
                "ink_in_proj_bf16_cublas" => k.weights(bf).exact(Exactness::Fast).when(|s| s.m > PROJ_ROWS_MAX).tol(BF16_TOL),
                "conv_mix_bf16" => {
                    k.weights(bf).when(|s| s.k > 0 && s.k <= MIX_D_MAX && s.n <= MIX_QUERIES_MAX).tol(BF16_TOL).tol_f32(F32_TOL).scratch(conv_mix_scratch)
                }
                "attn_res_mix_bf16" => k
                    .when(|s| s.k > 0 && s.k <= MIX_D_MAX && s.kv_len <= MIX_SOURCES_MAX && s.n <= MIX_QUERIES_MAX)
                    .tol(BF16_TOL)
                    .tol_f32(F32_TOL),
                "sconv_res_bf16" => k.weights(bf).when(|s| s.k > 0 && s.k <= MIX_D_MAX).tol(BF16_TOL),
                "loop_blend_bf16" => k
                    .weights(bf)
                    .when(|s| s.k > 0 && s.k <= MIX_D_MAX && (1..=BLEND_STREAMS_MAX).contains(&s.n))
                    .tol(BF16_TOL)
                    .scratch(loop_blend_scratch),
                other => unreachable!("inkmix kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_cuda)]
mod launch {
    use crate::device::CudaBackend;
    use crate::ffi::{status_name, IkScratch, IkTensor};
    use intarsia_core::backend::StreamId;
    use intarsia_core::{Error, Op, OpCall, Result};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type SC = *const IkScratch;
    type S = *mut c_void;

    extern "C" {
        fn ik_ink_in_proj_bf16_mma(x: T, w: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_ink_in_proj_bf16_mma_rows(x: T, w: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_ink_in_proj_bf16_cublas(x: T, w: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_sconv_res_bf16(x: T, taps: T, res: T, add: T, pos0: i64, pos: T, y: TM, ring: TM, sum: TM, s: S) -> c_int;
        fn ik_attn_res_mix_bf16(src: T, last: T, add: T, q: T, norm: T, eps: f32, out: TM, s: S) -> c_int;
        fn ik_conv_mix_bf16(
            x: T,
            taps: T,
            res: T,
            add: T,
            src: T,
            q: T,
            norm: T,
            msg: T,
            down: T,
            gain: T,
            forget: T,
            pos0: i64,
            pos: T,
            eps: f32,
            y: TM,
            ring: TM,
            sum: TM,
            mix: TM,
            sc: SC,
            s: S,
        ) -> c_int;
        fn ik_loop_blend_bf16(dest: T, msg: T, down: T, gain: T, forget: T, norm: T, eps: f32, out: TM, sc: SC, s: S) -> c_int;
    }

    impl CudaBackend {
        /// `Some(result)` when `kernel` is one of this module's launchers, `None` otherwise.
        pub(crate) fn launch_inkmix(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
            let (_, op) = super::TABLE.iter().find(|(n, _)| *n == kernel)?;
            Some(self.launch_inkmix_inner(kernel, *op, call, stream))
        }

        fn launch_inkmix_inner(&self, kernel: &str, op: Op, call: &OpCall, stream: StreamId) -> Result<()> {
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
            let i = |n: usize| &inp[n] as *const IkTensor;
            let o = out.as_mut_ptr();
            let on = |n: usize| o.wrapping_add(n);
            let pos0 = a.pos0 as i64;
            // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the launchers validate dtype /
            // rank / shape / scratch and return an error status instead of faulting, and never synchronise or allocate.
            let rc = unsafe {
                match kernel {
                    "ink_in_proj_bf16_mma" => ik_ink_in_proj_bf16_mma(i(0), i(1), on(0), on(1), on(2), s),
                    "ink_in_proj_bf16_mma_rows" => ik_ink_in_proj_bf16_mma_rows(i(0), i(1), on(0), on(1), on(2), s),
                    "ink_in_proj_bf16_cublas" => ik_ink_in_proj_bf16_cublas(i(0), i(1), on(0), on(1), on(2), s),
                    "sconv_res_bf16" => ik_sconv_res_bf16(i(0), i(1), i(2), i(3), pos0, pos, on(0), on(1), on(2), s),
                    "attn_res_mix_bf16" => ik_attn_res_mix_bf16(i(0), i(1), i(2), i(3), i(4), a.eps, on(0), s),
                    "conv_mix_bf16" => ik_conv_mix_bf16(
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
                        i(10),
                        pos0,
                        pos,
                        a.eps,
                        on(0),
                        on(1),
                        on(2),
                        on(3),
                        sc,
                        s,
                    ),
                    "loop_blend_bf16" => ik_loop_blend_bf16(i(0), i(1), i(2), i(3), i(4), i(5), a.eps, on(0), sc, s),
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
