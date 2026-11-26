//! Kernel table and device dispatch of the Inkling-2-Mini in-projection and sublayer-boundary ops: `InkInProj` (73), `ConvMix` (74), `AttnResMix` (67), `SconvRes` (69), `LoopBlend` (68) (lane ink-hip-mix; port map
//! docs/results/2026-11-25-inkling2-hip.md). Kernels in kernels/hip/ink_mix.hip and ink_proj.hip, ABI kernels/hip/ik_hip_abi_inkmix.h; contract
//! `intarsia_core::inkling2::ops`, admitted against the cpu reference (`ink_ref`) by `intarsia kernels test --backend hip --op <op>`.
//!
//! * `ink_in_proj_bf16_gemv`: one wave per weight row, T <= 16 (decode, verify); its twin `_rows` is the same launcher
//!   (`rows_exact`). `ink_in_proj_bf16_matmul`: prefill (T > 16), three `matmul_bf16` WMMA GEMMs.
//! * The row kernels (ConvMix, AttnResMix, SconvRes): one block per row, the same code for every row count, so rows-exact and
//!   chunk-invariant (no twin needed). The blend (LoopBlend, ConvMix with msg) adds the h wave-GEMV, the gate and the norm.
//!
//! An op of `OPS` counts as ported once `TABLE` lists it (until then it stays in `kernels::UNIMPLEMENTED`).

use crate::kernels::ARCH;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("ink_in_proj_bf16_gemv", Op::InkInProj),
    ("ink_in_proj_bf16_gemv_rows", Op::InkInProj),
    ("ink_in_proj_bf16_matmul", Op::InkInProj),
    ("conv_mix_bf16", Op::ConvMix),
    ("attn_res_mix_bf16", Op::AttnResMix),
    ("sconv_res_bf16", Op::SconvRes),
    ("loop_blend_bf16", Op::LoopBlend),
    // the router GEMV and the top-k in one launch (kernels/hip/ink_route_fused.hip)
    ("ink_route_bf16", Op::InkRoute),
];

/// The ops this module is the home of (a lane's kernels register only these).
pub const OPS: &[Op] = &[Op::InkInProj, Op::ConvMix, Op::AttnResMix, Op::SconvRes, Op::LoopBlend, Op::InkRoute];

const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Most rows of the GEMV form (ordinary calls; the launcher rejects more).
pub const PROJ_ROWS_MAX: usize = 16;
/// Widest row of the row kernels (6 elements per thread of 768), stored sources + last, streams and rank of the blend.
pub const MIX_D_MAX: usize = 4608;
pub const MIX_SOURCES_MAX: usize = 8;
pub const MIX_QUERIES_MAX: usize = 4;
pub const BLEND_STREAMS_MAX: usize = 3;
pub const BLEND_G_MAX: usize = 128;
/// Rows per blend chunk and channels per gate block (mirror kBlendRows / kGateCh in ink_mix.hip).
const BLEND_ROWS: usize = 256;
const GATE_CH: usize = 128;

/// Scratch of the blend for `s` streams, `r` rows, width `d`, rank `g` (mirrors `blend_bytes` in ink_mix.hip).
pub fn blend_bytes(s: usize, r: usize, d: usize, g: usize) -> usize {
    let rnd = |v: usize| (v + 255) & !255;
    let rc = r.min(BLEND_ROWS);
    let nsl = d.div_ceil(GATE_CH);
    rnd(s * rc * 2 * d * 2) + rnd(s * rc * d * 4) + rnd(s * rc * g * 2) + rnd(s * rc * d * 4) + rnd(s * rc * nsl * 4)
}

/// InkRoute: the logits `[T, E]` f32 when the call does not take them as an output.
fn route_scratch(s: &ShapeInfo) -> usize {
    s.m * s.experts * 4
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
            let k = KernelImpl::new(*op, *name, "hip").arch([ARCH]).acts(bf).exact(Exactness::Deterministic);
            match *name {
                "ink_in_proj_bf16_gemv" => {
                    k.weights(bf).when(|s| (1..=PROJ_ROWS_MAX).contains(&s.m) && s.k % 16 == 0).tol(BF16_TOL).priority(10)
                }
                // the verify / batched twin: the same launcher (`rows_exact`: only `exact_rows` calls select it)
                "ink_in_proj_bf16_gemv_rows" => k
                    .weights(bf)
                    .when(|s| (1..=PROJ_ROWS_MAX).contains(&s.m) && s.k % 16 == 0)
                    .tol(BF16_TOL)
                    .priority(-1)
                    .rows_exact(),
                // prefill: three WMMA GEMMs (matmul_bf16); `Fast` like CUDA's cuBLAS form: the GEMM's variant may depend on M, so a row
                // of a long call is not promised to equal the same row of a shorter one (not tested)
                "ink_in_proj_bf16_matmul" => k.weights(bf).exact(Exactness::Fast).when(|s| s.m > PROJ_ROWS_MAX).tol(BF16_TOL),
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
                // router GEMV + top-k: bitwise `Logits` + `MoeRoute` (one wave per logit in the lane order of `logits_bf16_gemv`, the last block
                // routes), rows-exact at any T <= 16 by construction; the math of the pair, so the f32 tolerance of `moe_route`
                "ink_route_bf16" => k
                    .weights(bf)
                    .when(|s| (1..=4).contains(&s.m) && s.k % 8 == 0 && s.experts <= 256 && (1..=32).contains(&s.top_k) && s.top_k <= s.experts)
                    .tol(F32_TOL)
                    .scratch(route_scratch),
                other => unreachable!("inkmix kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(ik_hip)]
pub(crate) mod launch {
    use crate::device::InkArgs;
    use crate::ffi::{IkScratch, IkTensor};
    use std::ffi::{c_int, c_void};

    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type SC = *const IkScratch;
    type S = *mut c_void;

    extern "C" {
        fn ik_ink_in_proj_bf16_gemv(x: T, w: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_ink_in_proj_bf16_gemv_rows(x: T, w: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_ink_in_proj_bf16_matmul(x: T, w: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_sconv_res_bf16(x: T, taps: T, res: T, add: T, pos0: i64, pos: T, y: TM, ring: TM, sum: TM, s: S) -> c_int;
        fn ik_attn_res_mix_bf16(src: T, last: T, add: T, q: T, norm: T, eps: f32, out: TM, s: S) -> c_int;
        #[allow(clippy::too_many_arguments)]
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
        #[allow(clippy::too_many_arguments)]
        fn ik_ink_route_bf16(
            x: T,
            w: T,
            bias: T,
            top_k: i64,
            scale: f32,
            norm: i32,
            temp: f32,
            sf: i32,
            weights: TM,
            ids: TM,
            logits: TM,
            sc: SC,
            s: S,
        ) -> c_int;
    }

    /// The launcher status of `kernel` when it is one of this module's, `None` otherwise. Called from `HipBackend::launch` with
    /// the call's tensors already converted (`a.i(n)` input n, `a.o(n)` output n, `a.sc` scratch, `a.pos` the device position or
    /// null, `a.tab*` the row table or null, `a.s` the stream, `a.call.attrs`); the caller checks the status, traces and reports.
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`.
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        let at = &a.call.attrs;
        let pos0 = at.pos0 as i64;
        Some(match kernel {
            "ink_in_proj_bf16_gemv" => ik_ink_in_proj_bf16_gemv(a.i(0), a.i(1), a.o(0), a.o(1), a.o(2), a.s),
            "ink_in_proj_bf16_gemv_rows" => ik_ink_in_proj_bf16_gemv_rows(a.i(0), a.i(1), a.o(0), a.o(1), a.o(2), a.s),
            "ink_in_proj_bf16_matmul" => ik_ink_in_proj_bf16_matmul(a.i(0), a.i(1), a.o(0), a.o(1), a.o(2), a.s),
            "sconv_res_bf16" => ik_sconv_res_bf16(a.i(0), a.i(1), a.i(2), a.i(3), pos0, a.pos, a.o(0), a.o(1), a.o(2), a.s),
            "attn_res_mix_bf16" => ik_attn_res_mix_bf16(a.i(0), a.i(1), a.i(2), a.i(3), a.i(4), at.eps, a.o(0), a.s),
            "conv_mix_bf16" => ik_conv_mix_bf16(
                a.i(0),
                a.i(1),
                a.i(2),
                a.i(3),
                a.i(4),
                a.i(5),
                a.i(6),
                a.i(7),
                a.i(8),
                a.i(9),
                a.i(10),
                pos0,
                a.pos,
                at.eps,
                a.o(0),
                a.o(1),
                a.o(2),
                a.o(3),
                a.sc,
                a.s,
            ),
            "loop_blend_bf16" => ik_loop_blend_bf16(a.i(0), a.i(1), a.i(2), a.i(3), a.i(4), a.i(5), at.eps, a.o(0), a.sc, a.s),
            "ink_route_bf16" => ik_ink_route_bf16(
                a.i(0),
                a.i(1),
                a.i(2),
                at.top_k as i64,
                at.scale,
                at.norm_topk as i32,
                at.temperature,
                at.dsv.score_func as i32,
                a.o(0),
                a.o(1),
                a.o(2),
                a.sc,
                a.s,
            ),
            _ => return None,
        })
    }
}

/// Tests: the blend gate kernel's fast fused-norm tail (lane ink-hip-dec2): 1 = on, 0 = the first tail, `-1` = the env
/// (`INTARSIA_HIP_INK_DEC_FAST` bit 2) or the default (on). The bits are the same either way (tests/ink_mix.rs).
pub fn set_mix_dec(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_mix_dec(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_mix_dec(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}

/// Tests / benchmarks: bit mask of the fast forms of the row kernel / blend launches / gate kernel (a process-wide switch of the kernel
/// library; 0 = the first forms of all, 3 = default, see ink_mix.hip `mix_mask`). The bits are the same either way (tests/ink_mix.rs). `INTARSIA_INK_MIX_FAST=0` does the same
/// at start-up.
pub fn set_mix_mode(v: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_ink_mix_mode(v: std::ffi::c_int);
        }
        // SAFETY: sets a plain int of the kernel library.
        unsafe { ik_ink_mix_mode(v) }
    }
    #[cfg(not(ik_hip))]
    let _ = v;
}
