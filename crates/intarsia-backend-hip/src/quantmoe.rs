//! Kernel table of the quantised-weight and MoE launchers (kernels/hip/{fp8,nvfp4,moe}.hip). Compiled on every
//! host (the table is data); the device dispatch is in `device.rs`. Registered through
//! `kernels::all()`.
//!
//! Selection: `*_gemv` (priority 10) serves decode shapes, `*_mma` (priority 0) every other shape, so each
//! (op, shape) has a kernel; the admission harness grades all of them against the cpu `ref_*` kernels.

use crate::kernels::{ARCH, ROWS_MAX};
use intarsia_core::quant::ops::{moe_block_scratch, moe_experts_scratch};
use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_fp8_gemv", Op::MatmulFp8),
    ("matmul_fp8_gemv_v2", Op::MatmulFp8),
    // row-exact twins (`rows_exact`, M = 1..=ROWS_MAX, bitwise the one-row v2 GEMV per row; tests/rows_exact.rs)
    ("matmul_fp8_gemv_rows", Op::MatmulFp8),
    // lane wg: the small-row WMMA family (INTARSIA_HIP_WMMA_ROWS=1; kernels/hip/ik_f8_rows.h). `_rows` is its `rows_exact` registration
    ("matmul_fp8_wmma", Op::MatmulFp8),
    ("matmul_fp8_wmma_rows", Op::MatmulFp8),
    ("matmul_fp8_mma", Op::MatmulFp8),
    ("matmul_fp8_mma_f8", Op::MatmulFp8),
    ("matmul_fp8_mma_v2", Op::MatmulFp8),
    ("matmul_nvfp4_gemv", Op::MatmulNvfp4),
    ("matmul_nvfp4_gemv_v2", Op::MatmulNvfp4),
    ("matmul_nvfp4_gemv_rows", Op::MatmulNvfp4),
    ("matmul_nvfp4_mma", Op::MatmulNvfp4),
    // M1 drill (kernels/hip/mxfp4.hip): OCP MXFP4 (Quark), W4A4 dynamic or W4A16 by the op's act input
    ("matmul_mxfp4_gemv", Op::MatmulMxfp4),
    ("matmul_mxfp4_mma", Op::MatmulMxfp4),
    ("matmul_mxfp4_pair_gemv", Op::MatmulMxfp4Pair),
    // row-exact twins of the two GEMVs (`rows_exact`, M <= ROWS_MAX = 16): M == 1 is the GEMV launcher itself, M = 2..16 runs
    // mxfp4_gemv_rows_kernel, which keeps the arithmetic of every output row of the one-row kernel (same lane-to-group map,
    // same fma chain, same warp sum; the activation QDQ is per group of 32) with a register shape that does not grow with M,
    // tests/rows_exact.rs
    ("matmul_mxfp4_gemv_rows", Op::MatmulMxfp4),
    ("matmul_mxfp4_pair_gemv_rows", Op::MatmulMxfp4Pair),
    ("moe_topk_bf16", Op::MoeTopk),
    ("moe_topk_bf16_v2", Op::MoeTopk),
    ("moe_experts_bf16", Op::MoeExperts),
    ("moe_experts_bf16_v2", Op::MoeExperts),
    ("moe_experts_bf16_v3", Op::MoeExperts),
    ("moe_experts_bf16_rows", Op::MoeExperts),
    ("moe_combine_bf16", Op::MoeCombine),
    // P2' (docs/results/2026-10-19-p2prime-moeblock.md): default ON, INTARSIA_HIP_MOE_BLOCK=0 disables
    ("moe_block_bf16", Op::MoeBlock),
    // row-exact form for batched decode (`rows_exact`, T = 1..=ROWS_MAX at any K <= 8; tests/moe_block.rs)
    ("moe_block_bf16_rows", Op::MoeBlock),
    // fused decode launchers: no other backend has them, the model falls back to the ops they replace
    ("matmul_pair_bf16_gemv", Op::MatmulPair),
    // lane wg: bf16 router / gate on the WMMA family (INTARSIA_HIP_WMMA_ROWS=1)
    ("matmul_pair_bf16_wmma", Op::MatmulPair),
    ("rmsnorm_add_matmul_pair_bf16_wmma", Op::RmsNormAddMatmulPair),
    ("matmul_fp8_pair_gemv", Op::MatmulFp8Pair),
    ("matmul_nvfp4_glu_gemv", Op::MatmulNvfp4Glu),
    ("gated_rmsnorm_bf16", Op::GatedRmsnorm),
    ("rmsnorm_add_matmul_pair_bf16_gemv", Op::RmsNormAddMatmulPair),
    ("gated_rmsnorm_matmul_fp8_gemv", Op::GatedRmsnormMatmulFp8),
    // core-op fusion phase 1 (kernels/hip/ik_hip_abi.h): row-local folds, each bitwise the two launches it replaces
    ("rmsnorm_add_matmul_fp8_pair_gemv", Op::RmsNormAddMatmulFp8Pair),
    ("sigmoid_mul_matmul_fp8_gemv", Op::SigmoidMulMatmulFp8),
    // lane wg: the decode folds of the fp8 GEMV on the WMMA family (INTARSIA_HIP_WMMA_ROWS=1)
    ("matmul_fp8_pair_wmma", Op::MatmulFp8Pair),
    ("gated_rmsnorm_matmul_fp8_wmma", Op::GatedRmsnormMatmulFp8),
    ("rmsnorm_add_matmul_fp8_pair_wmma", Op::RmsNormAddMatmulFp8Pair),
    ("sigmoid_mul_matmul_fp8_wmma", Op::SigmoidMulMatmulFp8),
    ("shared_down_combine_nvfp4_gemv", Op::SharedDownCombine),
];

/// Admission budget of the bf16-rounded outputs of every kernel here (a one-ulp flip, up to 7.8e-3 of the
/// largest output, is allowed: max_abs 1.5e-2 is about two ulps). rel_l2 is 1e-3: the admission sweep measures at most
/// 1.5e-4 (fp8 mma), so a bug that moves outputs by ~1e-3 relative is refused where the former 6e-3 admitted it. f32
/// outputs of the matmul kernels (lm_head logits) get `F32_OUT_TOL` through `KernelImpl::tol_f32`.
const TOL: Tol = Tol::new(1.5e-2, 1e-3);

/// Budget for f32 outputs of the quantised matmuls, applied per output tensor by dtype (`KernelImpl::tol_for`).
/// Measured worst: max_abs 1.8e-6, rel_l2 5.6e-7 over K/N-tail shapes.
pub const F32_OUT_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Budget of the fused kernels that end in a silu (`expf`, a division and a bf16 rounding of a product of two roundings):
/// the same as the elementwise kernels' (kernels.rs `BF16_TOL`).
const GLU_TOL: Tol = Tol::new(1.5e-2, 6e-3);

/// Lane wg: the small-row WMMA kernel family (kernels/hip/ik_f8_rows.h) replaces the fp8 GEMVs at M = 1..=ROWS_MAX, solo decode included, so a
/// row's bits are the same at every M. OPT-IN (INTARSIA_HIP_WMMA_ROWS=1): it changes the solo decode numerics once (the matrix core adds
/// in another order than the GEMV's dot chain), so a default flip is a separate decision (docs/results/2026-10-25-wg-hip-wmma-gemv.md).
/// With the knob unset or any other value every registration below is exactly main's.
pub fn wmma_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wmma_rows_from(std::env::var("INTARSIA_HIP_WMMA_ROWS").ok().as_deref()))
}

/// The value of INTARSIA_HIP_WMMA_ROWS -> family on: only exactly "1".
pub fn wmma_rows_from(v: Option<&str>) -> bool {
    v == Some("1")
}

/// `matmul_fp8_mma_f8` is the default prefill GEMM for the fp8 projections (M >= 5); INTARSIA_HIP_FP8_F8=0 is the kill switch
/// that puts `matmul_fp8_mma_v2` back (any other value, or unset, keeps f8). Its products are exact like the bf16 kernel's
/// and it agrees with the cpu reference to 5e-7 (f32 out), but the order of the f32 additions differs (fp8 x fp8 WMMA
/// instead of bf16 tiles), so it is not bitwise v2: the 35B's prefill KL21 against the vLLM goldens moves from 7.071e-3 to
/// 7.417e-3 (short) and from 4.583e-2 to 4.356e-2 (long), every gate verdict unchanged, for +10-12 % prefill
/// (docs/results/2026-10-19-hip-round4.md section 14). Off by default until round 5 (decided on those numbers,
/// docs/results/2026-10-19-hip-round5.md).
pub fn fp8_f8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fp8_f8_from(std::env::var("INTARSIA_HIP_FP8_F8").ok().as_deref()))
}

/// Whether the fp8 GEMV staging knobs are at their defaults (INTARSIA_HIP_FP8_Q = 2, INTARSIA_HIP_FP8_PAIR = 1,8; see
/// kernels/hip/fp8.hip staging_q / pair_cfg). The phase-1 fp8 folds hardcode those defaults, so with either knob moved the
/// unfused chain they must equal runs a different staging; the folds are then not selected (the model runs the unfused ops).
pub fn fp8_staging_default() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        fp8_staging_default_from(std::env::var("INTARSIA_HIP_FP8_Q").ok().as_deref(), std::env::var("INTARSIA_HIP_FP8_PAIR").ok().as_deref())
    })
}

/// Whether INTARSIA_HIP_FP8_Q is at its default (2, or unset). The row-exact fp8 GEMV (`matmul_fp8_gemv_rows`) runs the default
/// staging only, so with Q moved (Q = 3 quantises with a reciprocal and an fma correction, which can differ from the division
/// in a rare rounding) it is not accepted: a batched plan then fails to load instead of running a different arithmetic from
/// the one-row kernel the solo step runs.
pub fn fp8_q_is_default() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fp8_staging_default_from(std::env::var("INTARSIA_HIP_FP8_Q").ok().as_deref(), None))
}

/// [`fp8_staging_default`] on given values: Q is 2 unless it parses to 1 or 3 (the C++ rule); PAIR is the default unless it
/// parses as two integers other than (1, 8).
pub fn fp8_staging_default_from(q: Option<&str>, pair: Option<&str>) -> bool {
    let q_default = !matches!(q.and_then(|v| v.trim().parse::<i32>().ok()), Some(1) | Some(3));
    let pair_default = match pair {
        None => true,
        Some(v) => {
            let mut it = v.split(',').map(|x| x.trim().parse::<i32>());
            match (it.next(), it.next()) {
                (Some(Ok(r)), Some(Ok(w))) => (r, w) == (1, 8),
                _ => true, // sscanf fails: pair_cfg keeps the default
            }
        }
    };
    q_default && pair_default
}

/// The value of INTARSIA_HIP_FP8_F8 -> f8 on: on unless it is exactly "0".
pub fn fp8_f8_from(v: Option<&str>) -> bool {
    v != Some("0")
}

/// `moe_experts_bf16_v3` (one-launch decode experts, bitwise v2) is OPT-IN: selected only with INTARSIA_HIP_MOE_V3=1. Its down
/// blocks spin on a counter the gate_up blocks release; the mutant that drops the acquire fence before the h reads
/// (`v3-nofence`) is not killed by any test yet, and the gain is +0.8 % decode, so the default stays v2 until a test kills it
/// (docs/results/2026-10-19-hip-round5.md).
pub fn moe_v3_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| moe_v3_from(std::env::var("INTARSIA_HIP_MOE_V3").ok().as_deref()))
}

/// The value of INTARSIA_HIP_MOE_V3 -> v3 selected: only exactly "1".
pub fn moe_v3_from(v: Option<&str>) -> bool {
    v == Some("1")
}

/// `moe_block_bf16` (P2', `Op::MoeBlock`: top-k, routed experts, shared expert and combine in one launch, bitwise the default
/// chain) is ON BY DEFAULT: admissible unless INTARSIA_HIP_MOE_BLOCK=0 (the kill switch; docs/results/2026-10-19-p2prime-moeblock.md). Off, its
/// `when` is false, so `fused_ok` finds no kernel and the model runs the unfused chain exactly as before. Registered either
/// way, so the tests can launch it by name.
pub fn moe_block_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| moe_block_from(std::env::var("INTARSIA_HIP_MOE_BLOCK").ok().as_deref()))
}

/// The value of INTARSIA_HIP_MOE_BLOCK -> MoeBlock admissible: only exactly "0" disables; unset or anything else enables.
pub fn moe_block_from(v: Option<&str>) -> bool {
    v != Some("0")
}

/// The shapes `moe_block_bf16` serves: those the default chain runs with its decode kernels (T * K <= 16, K <= 8, the
/// experts' p1/v2 path; E <= 512, the top-k kernel's), H / I / IS multiples of 16 (`n` = I, `head_dim` = IS for this op).
pub fn moe_block_shape_ok(s: &intarsia_core::ShapeInfo) -> bool {
    s.m >= 1
        && (1..=8).contains(&s.top_k)
        && s.m * s.top_k <= 16
        && s.top_k <= s.experts
        && s.experts <= 512
        && s.k % 16 == 0
        && s.n % 16 == 0
        && s.head_dim % 16 == 0
        && s.k > 0
        && s.n > 0
        && s.head_dim > 0
        && s.out == Some(DType::Bf16)
}

/// Most rows for which the model takes `moe_block_bf16_rows`. The kernel is row-exact for 1..=`ROWS_MAX` tokens (tests/moe_block.rs), but
/// it reads each token's experts separately while the unfused `moe_experts_bf16_rows` shares an expert's weights between the tokens routed
/// to it, so past a few rows the fused block is slower (35B model-loop step, docs/results/2026-10-24-hr3-hip-batching-r3.md section 4).
pub const MOE_BLOCK_ROWS_MAX: usize = 4;

/// The shapes `moe_block_bf16_rows` serves: the same weights and widths as `moe_block_shape_ok`, 1..=`MOE_BLOCK_ROWS_MAX` tokens at any K <= 8.
pub fn moe_block_rows_shape_ok(s: &intarsia_core::ShapeInfo) -> bool {
    let mut one = s.clone();
    one.m = 1;
    s.m >= 1 && s.m <= MOE_BLOCK_ROWS_MAX && moe_block_shape_ok(&one)
}

pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "hip").arch([ARCH]);
            match *name {
                "matmul_fp8_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_rows_enabled() && (1..=4).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= 49152)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(10),
                "matmul_fp8_gemv_v2" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_rows_enabled() && (1..=4).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= 49152)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(20),
                // Row-exact twin (`rows_exact`: selected only by an `exact_rows` call): M in 1..=ROWS_MAX, row m bitwise the
                // one-row v2 GEMV on row m alone (the launcher splits M past the 48 KiB staging limit into groups).
                "matmul_fp8_gemv_rows" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_rows_enabled() && (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0 && s.k * 2 <= 49152 && fp8_q_is_default())
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(-1)
                    .rows_exact(),
                // lane wg: the WMMA family. `matmul_fp8_wmma` takes the ordinary decode shapes (M 1..=4, the GEMV's) and `_rows` the
                // `exact_rows` calls (batched decode, verify: M 1..=ROWS_MAX); one launcher, row m bitwise the one-row call on row m alone.
                "matmul_fp8_wmma" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_rows_enabled() && (1..=4).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k + 16)
                    .priority(30),
                "matmul_fp8_wmma_rows" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_rows_enabled() && (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k + 16)
                    .priority(0)
                    .rows_exact(),
                "matmul_fp8_mma" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL),
                // W8A8 prefill GEMM on the fp8 x fp8 WMMA (kernels/hip/ik_f8_gemm.h); weight-only activations run the bf16
                // kernel inside. DEFAULT ON (INTARSIA_HIP_FP8_F8=0 turns it off, see `fp8_f8_enabled`): off, matmul_fp8_mma_v2 is
                // the kernel for these shapes. Serves every M the gemv does not (>= 5), so a tail chunk is rounded like the rows
                // of a full one.
                "matmul_fp8_mma_f8" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.m >= 5 && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k + 16)
                    .priority(if fp8_f8_enabled() { 5 } else { -10 }),
                // matmul_fp8_mma with the W8A8 activations quantised once per call into scratch (not once per block along N)
                // and the weights decoded by the fp8 conversion instruction: the same bf16 tiles and WMMA order, so bitwise
                // matmul_fp8_mma (tests/fp8_mma_v2.rs), the kernel of every shape the GEMVs do not take. The scratch (M * K
                // bytes, 16 MB at a 4096-token chunk of K = 4096) is far below the MoE experts' (about 170 MB there), so the plan's
                // workspace (the largest of all) does not grow (docs/results/2026-10-19-hip-round3.md).
                "matmul_fp8_mma_v2" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k + 16)
                    .priority(1),
                "matmul_nvfp4_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=8).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(10),
                "matmul_nvfp4_gemv_v2" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=8).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(20),
                "matmul_nvfp4_gemv_rows" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(-1)
                    .rows_exact(),
                "matmul_nvfp4_mma" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL),
                "matmul_mxfp4_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=8).contains(&s.m) && s.k % 32 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2 + 16)
                    .priority(10),
                "matmul_mxfp4_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=8).contains(&s.m) && s.k % 32 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2 + 16)
                    .priority(10),
                "matmul_mxfp4_gemv_rows" | "matmul_mxfp4_pair_gemv_rows" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=crate::kernels::ROWS_MAX).contains(&s.m) && s.k % 32 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2 + 16)
                    .priority(-1)
                    .rows_exact(),
                "matmul_mxfp4_mma" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.k % 32 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2 + 16),
                "moe_topk_bf16" => k
                    .acts(bf)
                    .when(|s| s.experts <= 512 && (1..=32).contains(&s.top_k))
                    .exact(Exactness::Deterministic)
                    .tol(TOL),
                "moe_topk_bf16_v2" => k
                    .acts(bf)
                    .when(|s| s.experts <= 512 && (1..=32).contains(&s.top_k))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                // v3 (round 5): v2 with the decode path in one launch, bitwise v2 (tests/moe_v3.rs); same shapes. OPT-IN
                // (INTARSIA_HIP_MOE_V3=1, see `moe_v3_enabled`): above v2 when on, below v1 otherwise (registered either way,
                // so admission sweeps it and the tests can call it by name)
                "moe_experts_bf16" | "moe_experts_bf16_v2" | "moe_experts_bf16_v3" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| {
                        s.experts <= 1024 && s.k % 16 == 0 && s.n % 16 == 0 && 32768 / (2 * s.k.max(s.n).max(1)) >= 1
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_experts_scratch)
                    .priority(match *name {
                        "moe_experts_bf16_v3" => {
                            if moe_v3_enabled() {
                                20
                            } else {
                                -10
                            }
                        }
                        "moe_experts_bf16_v2" => 10,
                        _ => 0,
                    }),
                // Row-exact multi-token twin of the v2 experts (`rows_exact`, T = 1..=ROWS_MAX): token t bitwise the T = 1 call on
                // token t alone, never the tensor-core path (tests/rows_exact.rs). The ordinary registrations keep every T.
                "moe_experts_bf16_rows" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| {
                        (1..=ROWS_MAX).contains(&s.m)
                            && s.experts <= 1024
                            && s.k % 16 == 0
                            && s.n % 16 == 0
                            && 32768 / (2 * s.k.max(s.n).max(1)) >= 1
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_experts_scratch)
                    .priority(-1)
                    .rows_exact(),
                "moe_combine_bf16" => k.acts(bf).exact(Exactness::Deterministic).tol(TOL),
                // P2' (kernels/hip/moe_block.hip, tests/moe_block.rs): ON by default (INTARSIA_HIP_MOE_BLOCK=0 disables), see `moe_block_enabled`. Bitwise the chain
                // moe_topk_bf16_v2 -> moe_experts_bf16_v2 -> matmul_nvfp4_glu_gemv (gate / up gemv + silu_glu at T > 1) ->
                // shared_down_combine_nvfp4_gemv at every shape it takes; scratch: h [T * K, I] and sa [T, IS] bf16.
                "moe_block_bf16" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| moe_block_enabled() && moe_block_shape_ok(s))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_block_scratch)
                    .priority(10),
                // Row-exact form (lane hr3): a batched decode step (or a verify pass) of 1..=ROWS_MAX tokens, each token bitwise the
                // T = 1 launch on it alone, so the batched step keeps the fused MoE block at every row count (the ordinary form stops at
                // T * K <= 16, i.e. T <= 2 at top-8). Selected only by `exact_rows` calls.
                "moe_block_bf16_rows" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| moe_block_enabled() && moe_block_rows_shape_ok(s))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_block_scratch)
                    .priority(-1)
                    .rows_exact(),
                // Fused decode launchers: one activation row, bf16 in and out. Each is bitwise the chain it replaces (tests/
                // fused_decode.rs), so the budget is that of the kernels it is built from.
                "matmul_pair_bf16_gemv" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| !wmma_rows_enabled() && s.m == 1 && s.k % 8 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "matmul_fp8_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "matmul_nvfp4_glu_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.m == 1 && s.k % 16 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(GLU_TOL)
                    .priority(10),
                // the GDN output norm folded into the fp8 out_proj: dv = 128 rows, K = rows * 128 within the LDS staging
                "gated_rmsnorm_matmul_fp8_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_rows_enabled() && s.m == 1 && s.head_dim == 128 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                // norm prologue + pair GEMV: the row is staged in LDS (K <= 8192, K % 8 == 0); bitwise rmsnorm_add + pair
                // phase-1 folds. The fp8 ones serve one row (decode; the LDS holds the row, K <= 24576), the nvfp4 one the
                // 1..=8 rows of matmul_nvfp4_gemv_v2. With batching on (`--max-batch`, lane hb) the limits meet the batch-consistency rule
                // (`OpCall::with_rows`): a fold with no kernel at some row count is not taken by the solo step either; a row count
                // without a kernel runs the unfused ops.
                "rmsnorm_add_matmul_fp8_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        !wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16) && fp8_staging_default()
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "sigmoid_mul_matmul_fp8_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        !wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16) && fp8_staging_default()
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                // lane wg: the folds on the WMMA family; the same shapes as their GEMV twins (K <= 4096 for the ones that stage the row in one window)
                "matmul_fp8_pair_wmma" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "gated_rmsnorm_matmul_fp8_wmma" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        wmma_rows_enabled() && s.m == 1 && s.head_dim == 128 && s.k % 16 == 0 && s.k <= 4096 && s.out == Some(DType::Bf16)
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "rmsnorm_add_matmul_fp8_pair_wmma" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    // the norm runs in one window of the (half-width, weight-only) staging: K <= 2048
                    .when(|s| wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.k <= 2048 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "sigmoid_mul_matmul_fp8_wmma" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.k <= 4096 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "shared_down_combine_nvfp4_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=8).contains(&s.m) && s.k % 16 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "rmsnorm_add_matmul_pair_bf16_gemv" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| !wmma_rows_enabled() && s.m == 1 && s.k % 8 == 0 && (8..=8192).contains(&s.k) && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "matmul_pair_bf16_wmma" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "rmsnorm_add_matmul_pair_bf16_wmma" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| wmma_rows_enabled() && s.m == 1 && s.k % 16 == 0 && s.k <= 2048 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "gated_rmsnorm_bf16" => k.acts(bf).when(|s| s.out == Some(DType::Bf16)).exact(Exactness::Deterministic).tol(GLU_TOL),
                other => unreachable!("quantmoe kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_core::ShapeInfo;

    fn fp8_shape(m: usize, n: usize, k: usize) -> ShapeInfo {
        ShapeInfo { m, n, k, act: Some(DType::Bf16), weight: Some(DType::Fp8E4m3), out: Some(DType::Bf16), ..Default::default() }
    }

    /// The MatmulFp8 kernel `Registry::select` picks for this shape: the highest priority among the registrations that
    /// accept it (ties: registration order).
    fn pick(ks: &[KernelImpl], s: &ShapeInfo) -> String {
        let mut best: Option<&KernelImpl> = None;
        for k in ks.iter().filter(|k| k.op == Op::MatmulFp8 && k.accepts(s) && !k.rows_exact) {
            if best.is_none_or(|b| k.priority > b.priority) {
                best = Some(k);
            }
        }
        best.expect("a MatmulFp8 kernel for every shape").name.clone()
    }

    /// The f8 default (round 5): on when INTARSIA_HIP_FP8_F8 is unset or anything but "0"; "0" is the kill switch.
    #[test]
    fn fp8_f8_is_the_default() {
        assert!(fp8_f8_from(None));
        assert!(fp8_f8_from(Some("1")));
        assert!(fp8_f8_from(Some("")));
        assert!(!fp8_f8_from(Some("0")));
        assert_eq!(fp8_f8_enabled(), std::env::var("INTARSIA_HIP_FP8_F8").as_deref() != Ok("0"));
    }

    /// Pins the fp8 selection: the GEMVs for decode rows, `matmul_fp8_mma_f8` for every other shape with M >= 5 (prefill
    /// chunks, tails, verify rows) by default, `matmul_fp8_mma_v2` there under the kill switch INTARSIA_HIP_FP8_F8=0 and for the
    /// long decode rows f8 does not take. `matmul_fp8_mma` stays registered (admission sweeps it, and it is the bitwise
    /// reference of tests/fp8_mma_v2.rs) but is never chosen.
    #[test]
    fn fp8_selection_is_pinned() {
        let ks = all();
        let gemm = if fp8_f8_enabled() { "matmul_fp8_mma_f8" } else { "matmul_fp8_mma_v2" };
        let f8 = ks.iter().find(|k| k.name == "matmul_fp8_mma_f8").unwrap();
        let v2 = ks.iter().find(|k| k.name == "matmul_fp8_mma_v2").unwrap();
        if fp8_f8_enabled() {
            assert!(f8.priority > v2.priority, "f8 is the default GEMM");
        } else {
            assert!(f8.priority < v2.priority, "INTARSIA_HIP_FP8_F8=0 puts v2 back");
        }
        // the 35B's dense fp8 projections (N x K): qkv / q, z, o_proj / out_proj, k / v
        for (n, k) in [(8192usize, 2048usize), (4096, 2048), (2048, 4096), (512, 2048)] {
            for m in [5usize, 16, 17, 312, 1024, 4096] {
                assert_eq!(pick(&ks, &fp8_shape(m, n, k)), gemm, "m={m} n={n} k={k}");
            }
            for m in 1..=4 {
                assert_eq!(pick(&ks, &fp8_shape(m, n, k)), "matmul_fp8_gemv_v2", "m={m} n={n} k={k}");
            }
        }
        // decode rows too long to stage for the GEMV (M * K * 2 > 48 KB) take the GEMM; the e4m3 one serves M >= 5 only
        assert_eq!(pick(&ks, &fp8_shape(4, 1024, 8192)), "matmul_fp8_mma_v2");
        let v1 = ks.iter().find(|k| k.name == "matmul_fp8_mma").unwrap();
        let v2 = ks.iter().find(|k| k.name == "matmul_fp8_mma_v2").unwrap();
        assert!(v2.priority > v1.priority);
        for s in [fp8_shape(5, 17, 48), fp8_shape(4096, 8192, 2048), fp8_shape(1, 512, 2048), fp8_shape(64, 512, 2056)] {
            assert_eq!(v1.accepts(&s), v2.accepts(&s), "v2 must take exactly the shapes v1 takes");
        }
        assert!(!v2.accepts(&fp8_shape(64, 512, 2056)), "K % 16 != 0");
    }

    /// The v2 workspace is the quantised activations (M * K bytes), and at every chunk of the 35B it stays below what the
    /// MoE experts reserve for the same chunk, so the plan's workspace (the largest declared scratch) does not grow.
    #[test]
    fn fp8_v2_scratch_stays_below_the_moe_workspace() {
        let ks = all();
        let v2 = ks.iter().find(|k| k.name == "matmul_fp8_mma_v2").unwrap();
        assert_eq!(v2.scratch_bytes(&fp8_shape(4096, 2048, 4096)), 4096 * 4096 + 16);
        let moe = ks.iter().find(|k| k.name == "moe_experts_bf16_v2").unwrap();
        for t in [5usize, 16, 256, 1024, 2048, 4096, 8192] {
            let m = ShapeInfo { m: t, n: 512, k: 2048, experts: 256, top_k: 8, act: Some(DType::Bf16), ..Default::default() };
            assert!(v2.scratch_bytes(&fp8_shape(t, 2048, 4096)) < moe.scratch_bytes(&m), "chunk {t}");
        }
    }

    /// v3 is opt-in: only INTARSIA_HIP_MOE_V3=1 selects it.
    #[test]
    fn fp8_folds_follow_the_staging_defaults() {
        assert!(fp8_staging_default_from(None, None));
        assert!(fp8_staging_default_from(Some("2"), Some("1,8")));
        assert!(fp8_staging_default_from(Some("7"), Some("junk")));
        assert!(!fp8_staging_default_from(Some("1"), None));
        assert!(!fp8_staging_default_from(Some("3"), None));
        assert!(!fp8_staging_default_from(None, Some("2,8")));
        assert!(!fp8_staging_default_from(None, Some("1,16")));
    }

    /// P2' is on by default: `moe_block_bf16` is admissible unless INTARSIA_HIP_MOE_BLOCK=0 (only "0" disables), and then exactly at the shapes the
    /// default chain serves with its decode kernels (T * K <= 16, K <= 8, E <= 512, H / I / IS multiples of 16).
    #[test]
    fn moe_block_is_default_on_and_its_shapes_are_pinned() {
        assert!(moe_block_from(None));
        assert!(!moe_block_from(Some("0")));
        assert!(moe_block_from(Some("")));
        assert!(moe_block_from(Some("true")));
        assert!(moe_block_from(Some("1")));
        assert_eq!(moe_block_enabled(), std::env::var("INTARSIA_HIP_MOE_BLOCK").as_deref() != Ok("0"));
        let ks = all();
        let k = ks.iter().find(|k| k.name == "moe_block_bf16").unwrap();
        let sh = |m: usize, e: usize, top_k: usize, h: usize, i: usize, is: usize| ShapeInfo {
            m,
            k: h,
            n: i,
            head_dim: is,
            experts: e,
            top_k,
            act: Some(DType::Bf16),
            weight: Some(DType::U8),
            out: Some(DType::Bf16),
            ..Default::default()
        };
        // the 35B's decode step (T = 1) and T = 2; not T = 3 (the chain's experts leave the decode kernels at T * K > 16)
        for (m, ok) in [(1usize, true), (2, true), (3, false), (4096, false)] {
            assert_eq!(moe_block_shape_ok(&sh(m, 256, 8, 2048, 512, 512)), ok, "m={m}");
            assert_eq!(k.accepts(&sh(m, 256, 8, 2048, 512, 512)), ok && moe_block_enabled(), "m={m}");
        }
        assert!(moe_block_shape_ok(&sh(4, 16, 4, 128, 32, 48)));
        assert!(!moe_block_shape_ok(&sh(1, 16, 9, 128, 32, 48)), "K > 8");
        assert!(!moe_block_shape_ok(&sh(1, 1024, 8, 2048, 512, 512)), "E > 512");
        assert!(!moe_block_shape_ok(&sh(1, 256, 8, 2048, 512, 520)), "IS % 16");
        // scratch: h [T * K, I] and sa [T, IS] bf16, each rounded up to 256 bytes
        assert_eq!(k.scratch_bytes(&sh(1, 256, 8, 2048, 512, 512)), 8 * 512 * 2 + 512 * 2);
        assert_eq!(k.scratch_bytes(&sh(1, 16, 2, 256, 16, 32)), 256 + 256);
    }

    #[test]
    fn moe_v3_is_opt_in() {
        assert!(!moe_v3_from(None));
        assert!(!moe_v3_from(Some("0")));
        assert!(!moe_v3_from(Some("")));
        assert!(!moe_v3_from(Some("true")));
        assert!(moe_v3_from(Some("1")));
        assert_eq!(moe_v3_enabled(), std::env::var("INTARSIA_HIP_MOE_V3").as_deref() == Ok("1"));
    }

    /// Pins the MoE experts selection (round 5): `moe_experts_bf16_v2` for every shape the 35B draws, decode (T = 1, 2) and
    /// prefill chunks alike, by default; `moe_experts_bf16_v3` instead only with INTARSIA_HIP_MOE_V3=1. v3 takes the same
    /// shapes and the same scratch as v2, so the plan's workspace does not change either way; v1 and v3 stay registered.
    #[test]
    fn moe_experts_selection_is_pinned() {
        let ks = all();
        let get = |n: &str| ks.iter().find(|k| k.name == n).unwrap();
        let (v1, v2, v3) = (get("moe_experts_bf16"), get("moe_experts_bf16_v2"), get("moe_experts_bf16_v3"));
        assert!(v2.priority > v1.priority);
        let want = if moe_v3_enabled() {
            assert!(v3.priority > v2.priority, "INTARSIA_HIP_MOE_V3=1 puts v3 above v2");
            "moe_experts_bf16_v3"
        } else {
            assert!(v3.priority < v1.priority, "v3 is opt-in: below v1 and v2 by default");
            "moe_experts_bf16_v2"
        };
        for t in [1usize, 2, 3, 5, 16, 312, 4096] {
            let m = ShapeInfo { m: t, n: 512, k: 2048, experts: 256, top_k: 8, act: Some(DType::Bf16), ..Default::default() };
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == Op::MoeExperts && k.accepts(&m) && !k.rows_exact) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            assert_eq!(best.unwrap().name, want, "T = {t}");
            assert_eq!(v3.scratch_bytes(&m), v2.scratch_bytes(&m), "T = {t}");
        }
        for s in [
            ShapeInfo { m: 1, n: 512, k: 2048, experts: 2048, top_k: 8, act: Some(DType::Bf16), ..Default::default() },
            ShapeInfo { m: 1, n: 520, k: 2048, experts: 256, top_k: 8, act: Some(DType::Bf16), ..Default::default() },
            ShapeInfo { m: 1, n: 64, k: 16, experts: 16, top_k: 2, act: Some(DType::Bf16), ..Default::default() },
        ] {
            assert_eq!(v3.accepts(&s), v2.accepts(&s), "v3 must take exactly the shapes v2 takes");
        }
    }
}
