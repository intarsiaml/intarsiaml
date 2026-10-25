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
    ("matmul_mxfp4_mma_v1", Op::MatmulMxfp4),
    ("matmul_mxfp4_mma_fold", Op::MatmulMxfp4),
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
    // lane l7: the grouped form for the rows `moe_block_bf16_rows` does not serve (`rows_exact`, 5..=16 rows; kernels/hip/moe_block_grouped.hip)
    ("moe_block_bf16_grouped", Op::MoeBlock),
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

/// Lane wg: INTARSIA_HIP_FP8_FRAG=1 (with INTARSIA_HIP_WMMA_ROWS=1) stores the dense fp8 weights the WMMA table serves at every row count
/// 1..=ROWS_MAX in fragment order (`Layout::Fp8Frag`, `Backend::relayout_fp8_frag`): one 16-byte load per lane is a WMMA A fragment, which
/// takes the M = 1 W8A8 kernel to the GEMV's time. The tag lives on the allocation, so the fp8 GEMVs (which read row-major bytes) are never
/// selected for such a weight; the WMMA family and the prefill GEMMs read it. Off by default; weights it cannot serve stay row-major.
pub fn fp8_frag_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wmma_rows_enabled() && std::env::var("INTARSIA_HIP_FP8_FRAG").as_deref() == Ok("1"))
}

/// Whether a dense fp8 weight of kind `kind` may be stored in fragment order: the knob is on and the table runs that kind on the WMMA family
/// at every row count the registry gives it (1..=ROWS_MAX), because a fragment-order weight has no GEMV to fall back to.
pub fn fp8_frag_serves(kind: WmmaKind) -> bool {
    fp8_frag_enabled() && (1..=ROWS_MAX).all(|m| wmma_takes(kind, m))
}

/// What a WMMA small-row GEMV replaces: fp8 W8A8 (static input scale), fp8 W8A16 (weight-only), bf16 weights (router, gates, pairs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmmaKind {
    Fp8A8,
    Fp8A16,
    Bf16,
}

/// The kind of an fp8 matmul call (`OpAttrs::fp8_static_act` = W8A8).
pub fn fp8_kind(s: &intarsia_core::ShapeInfo) -> WmmaKind {
    if s.fp8_static_act { WmmaKind::Fp8A8 } else { WmmaKind::Fp8A16 }
}

/// THE dispatch table: the rows `lo..=hi` each kind runs on the WMMA family when INTARSIA_HIP_WMMA_ROWS=1 (`None` = never). Every
/// registration above asks `wmma_takes(kind, m)`; nothing else decides. A kind that is not 1..=ROWS_MAX switches kernels inside a
/// batch-size range, so one row's bits differ between rows=1 and rows=M across the switch: that is a numerics change, not row
/// invariance (tests/rows_exact.rs shows it at each boundary). Default (user decision 2026-10-25): W8A16 on 1..=ROWS_MAX (WMMA wins at every M, row
/// invariant), W8A8 and bf16 stay on the GEMV (the WMMA is 2-13 % slower at M=1, so a switch inside the batch range would break solo == batched,
/// and all-M WMMA would cost single-stream). Option 1 (all kinds WMMA at all M) is the override "fp8a8=all,bf16=all".
/// A/B override (read once): INTARSIA_HIP_WMMA_TABLE="fp8a8=4-16,fp8a16=1-16,bf16=off" (`lo-hi`, `off`, or `all`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WmmaTable {
    pub fp8a8: Option<(usize, usize)>,
    pub fp8a16: Option<(usize, usize)>,
    pub bf16: Option<(usize, usize)>,
}

impl WmmaTable {
    pub const DEFAULT: WmmaTable = WmmaTable { fp8a8: None, fp8a16: Some((1, ROWS_MAX)), bf16: None };

    /// Parse the override on top of the default; Err names the bad entry.
    pub fn parse(spec: &str) -> Result<WmmaTable, String> {
        let mut t = WmmaTable::DEFAULT;
        for ent in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let (k, v) = ent.split_once('=').ok_or_else(|| format!("{ent:?}: want kind=lo-hi|off|all"))?;
            let range = match v.trim() {
                "off" => None,
                "all" => Some((1, ROWS_MAX)),
                r => {
                    let (lo, hi) = r.split_once('-').ok_or_else(|| format!("{ent:?}: want lo-hi"))?;
                    let (lo, hi) = (lo.parse::<usize>().map_err(|e| format!("{ent:?}: {e}"))?, hi.parse::<usize>().map_err(|e| format!("{ent:?}: {e}"))?);
                    if lo == 0 || hi < lo {
                        return Err(format!("{ent:?}: need 1 <= lo <= hi"));
                    }
                    Some((lo, hi))
                }
            };
            match k.trim() {
                "fp8a8" => t.fp8a8 = range,
                "fp8a16" => t.fp8a16 = range,
                "bf16" => t.bf16 = range,
                other => return Err(format!("unknown kind {other:?} (fp8a8, fp8a16, bf16)")),
            }
        }
        Ok(t)
    }

    pub fn takes(&self, kind: WmmaKind, m: usize) -> bool {
        let r = match kind {
            WmmaKind::Fp8A8 => self.fp8a8,
            WmmaKind::Fp8A16 => self.fp8a16,
            WmmaKind::Bf16 => self.bf16,
        };
        r.is_some_and(|(lo, hi)| (lo..=hi).contains(&m))
    }
}

/// The table in force: all-off with the knob unset (registry identical to main), else the default or the override.
pub fn wmma_table() -> Option<&'static WmmaTable> {
    static T: std::sync::OnceLock<Option<WmmaTable>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        if !wmma_rows_enabled() {
            return None;
        }
        match std::env::var("INTARSIA_HIP_WMMA_TABLE") {
            Ok(spec) => Some(WmmaTable::parse(&spec).unwrap_or_else(|e| panic!("INTARSIA_HIP_WMMA_TABLE: {e}"))),
            Err(_) => Some(WmmaTable::DEFAULT),
        }
    })
    .as_ref()
}

pub fn wmma_takes(kind: WmmaKind, m: usize) -> bool {
    wmma_table().is_some_and(|t| t.takes(kind, m))
}

#[cfg(test)]
mod wmma_table_tests {
    use super::*;

    #[test]
    fn default_is_w8a16_only_and_whole_range() {
        for m in 1..=ROWS_MAX {
            assert!(WmmaTable::DEFAULT.takes(WmmaKind::Fp8A16, m));
            assert!(!WmmaTable::DEFAULT.takes(WmmaKind::Fp8A8, m) && !WmmaTable::DEFAULT.takes(WmmaKind::Bf16, m));
        }
        assert!(!WmmaTable::DEFAULT.takes(WmmaKind::Fp8A16, ROWS_MAX + 1));
        let all = WmmaTable::parse("fp8a8=all,bf16=all").unwrap();
        for kind in [WmmaKind::Fp8A8, WmmaKind::Fp8A16, WmmaKind::Bf16] {
            assert!((1..=ROWS_MAX).all(|m| all.takes(kind, m)));
        }
    }

    #[test]
    fn override_switches_one_kind_at_its_boundary() {
        let t = WmmaTable::parse("fp8a8=4-16, bf16=off").unwrap();
        assert!(!t.takes(WmmaKind::Fp8A8, 3) && t.takes(WmmaKind::Fp8A8, 4) && t.takes(WmmaKind::Fp8A8, 16));
        assert!(t.takes(WmmaKind::Fp8A16, 1));
        assert!(!t.takes(WmmaKind::Bf16, 1) && !t.takes(WmmaKind::Bf16, 16));
        assert_eq!(WmmaTable::parse("").unwrap(), WmmaTable::DEFAULT);
    }

    #[test]
    fn bad_override_is_an_error() {
        for bad in ["fp8a8", "fp8a8=0-4", "fp8a8=5-4", "x=1-2", "bf16=a-b"] {
            assert!(WmmaTable::parse(bad).is_err(), "{bad}");
        }
    }
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

/// `matmul_mxfp4_mma_fold` (lane mx, docs/results/2026-10-25-mx-hip-27b-prefill.md section 5): the DEFAULT W4A4 MXFP4 prefill
/// GEMM (M >= 9) since the user's decision of 2026-10-25 13:14 (docs/results/2026-10-25-mx-fold-default.md). It runs on the fp8
/// WMMA and its bits differ from `matmul_mxfp4_mma` (v2): the fp8 matrix core rounds its sums differently, and tiny activation
/// groups are rounded. INTARSIA_HIP_MXFP4_FOLD=0 returns to v2 (the kill switch).
pub fn mxfp4_fold_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mxfp4_fold_from(std::env::var("INTARSIA_HIP_MXFP4_FOLD").ok().as_deref()))
}

/// The value of INTARSIA_HIP_MXFP4_FOLD -> on unless it is exactly "0".
pub fn mxfp4_fold_from(v: Option<&str>) -> bool {
    v != Some("0")
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

/// Rows served by `moe_block_bf16_grouped` (lane l7): the rows `moe_block_bf16_rows` leaves to the unfused chain, up to the 16 the batched step takes.
pub const MOE_GROUPED_ROWS: std::ops::RangeInclusive<usize> = (MOE_BLOCK_ROWS_MAX + 1)..=16;

/// `moe_block_bf16_grouped` (lane l7, kernels/hip/moe_block_grouped.hip: topk, grouped routed experts, one shared-expert pass for all rows, bitwise the
/// T = 1 block per token) is admissible unless INTARSIA_HIP_MOE_GROUPED=0 (and INTARSIA_HIP_MOE_BLOCK is not 0). Off, `fused_ok` finds no MoeBlock
/// kernel for 5..=16 rows and the model runs the unfused chain exactly as before.
pub fn moe_grouped_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| moe_grouped_from(std::env::var("INTARSIA_HIP_MOE_GROUPED").ok().as_deref()))
}

/// The value of INTARSIA_HIP_MOE_GROUPED -> the grouped block admissible: only exactly "0" disables; unset or anything else enables.
pub fn moe_grouped_from(v: Option<&str>) -> bool {
    v != Some("0")
}

/// The shapes `moe_block_bf16_grouped` serves: the weights and widths of `moe_block_shape_ok` at any K <= 8, 5..=16 tokens.
pub fn moe_block_grouped_shape_ok(s: &intarsia_core::ShapeInfo) -> bool {
    let mut one = s.clone();
    one.m = 1;
    MOE_GROUPED_ROWS.contains(&s.m) && moe_block_shape_ok(&one)
}

/// Scratch bytes of `moe_block_bf16_grouped` (`ik_moe_block_grouped_scratch_bytes`): top-k weights f32 `[T, K]`, ids i32 `[T, K]`, routed bf16
/// `[T, H]`, shared activation bf16 `[T, IS]`, then the routed experts' scratch; each region rounded up to 256 bytes.
pub fn moe_block_grouped_scratch(s: &intarsia_core::ShapeInfo) -> usize {
    let r = |x: usize| (x + 255) & !255;
    r(s.m * s.top_k * 4) * 2 + r(s.m * s.k * 2) + r(s.m * s.head_dim * 2) + moe_experts_scratch(s)
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
                    .when(|s| !wmma_takes(fp8_kind(s), s.m) && (1..=4).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= 49152)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(10),
                "matmul_fp8_gemv_v2" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_takes(fp8_kind(s), s.m) && (1..=4).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= 49152)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(20),
                // Row-exact twin (`rows_exact`: selected only by an `exact_rows` call): M in 1..=ROWS_MAX, row m bitwise the
                // one-row v2 GEMV on row m alone (the launcher splits M past the 48 KiB staging limit into groups).
                "matmul_fp8_gemv_rows" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_takes(fp8_kind(s), s.m) && (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0 && s.k * 2 <= 49152 && fp8_q_is_default())
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(-1)
                    .rows_exact(),
                // lane wg: the WMMA family. `matmul_fp8_wmma` takes the ordinary decode shapes (M 1..=4, the GEMV's) and `_rows` the
                // `exact_rows` calls (batched decode, verify: M 1..=ROWS_MAX); one launcher, row m bitwise the one-row call on row m alone.
                "matmul_fp8_wmma" => k
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_takes(fp8_kind(s), s.m) && (1..=4).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k + 16)
                    .priority(30),
                "matmul_fp8_wmma_rows" => k
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_takes(fp8_kind(s), s.m) && (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k + 16)
                    .priority(0)
                    .rows_exact(),
                "matmul_fp8_mma" => k
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
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
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
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
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
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
                // matmul_mxfp4_mma before lane mx's v2 (kernels/hip/mxfp4.hip Mxfp4PolicyV2 / mx_gemm_run): the same bits, never
                // selected; the bitwise reference of tests/mxfp4_v2.rs. INTARSIA_HIP_MXFP4_V1=1 makes matmul_mxfp4_mma run it.
                "matmul_mxfp4_mma_v1" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.k % 32 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2 + 16)
                    .priority(-10),
                // DEFAULT (lane mx, user decision 2026-10-25 13:14; its bits differ from v2's): W4A4 folded onto the fp8 WMMA
                // (kernels/hip/mxfp4.hip launch_fold); INTARSIA_HIP_MXFP4_FOLD=0 (`mxfp4_fold_enabled`) turns it off, and then it
                // is never selected. M >= 9 in practice (the GEMVs outrank it); W4A16 calls run v2 inside (bitwise).
                "matmul_mxfp4_mma_fold" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.k % 64 == 0 && s.k <= 32768)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2 + s.m + s.n + 16)
                    .priority(if mxfp4_fold_enabled() { 5 } else { -20 }),
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
                // Grouped row-exact form (lane l7): 5..=16 tokens, each token bitwise the T = 1 launch on it alone; the routed experts of the unfused
                // `moe_experts_bf16_rows` plus one shared-expert pass over all rows. Selected only by `exact_rows` calls.
                "moe_block_bf16_grouped" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| moe_block_enabled() && moe_grouped_enabled() && moe_block_grouped_shape_ok(s))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_block_grouped_scratch)
                    .priority(-1)
                    .rows_exact(),
                // Fused decode launchers: one activation row, bf16 in and out. Each is bitwise the chain it replaces (tests/
                // fused_decode.rs), so the budget is that of the kernels it is built from.
                "matmul_pair_bf16_gemv" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| !wmma_takes(WmmaKind::Bf16, s.m) && s.m == 1 && s.k % 8 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "matmul_fp8_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| !wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
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
                    .when(|s| !wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.head_dim == 128 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
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
                        !wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16) && fp8_staging_default()
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "sigmoid_mul_matmul_fp8_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        !wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16) && fp8_staging_default()
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                // lane wg: the folds on the WMMA family; the same shapes as their GEMV twins (K <= 4096 for the ones that stage the row in one window)
                "matmul_fp8_pair_wmma" => k
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "gated_rmsnorm_matmul_fp8_wmma" => k
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.head_dim == 128 && s.k % 16 == 0 && s.k <= 4096 && s.out == Some(DType::Bf16)
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "rmsnorm_add_matmul_fp8_pair_wmma" => k
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    // the norm runs in one window of the (half-width, weight-only) staging: K <= 2048
                    .when(|s| wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.k % 16 == 0 && s.k <= 2048 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "sigmoid_mul_matmul_fp8_wmma" => k
                    .frag_ok() // reads fragment-order weights too (kernels/hip/fp8.hip frag_unwrap)
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| wmma_takes(fp8_kind(s), s.m) && s.m == 1 && s.k % 16 == 0 && s.k <= 4096 && s.out == Some(DType::Bf16))
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
                    .when(|s| !wmma_takes(WmmaKind::Bf16, s.m) && s.m == 1 && s.k % 8 == 0 && (8..=8192).contains(&s.k) && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "matmul_pair_bf16_wmma" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| wmma_takes(WmmaKind::Bf16, s.m) && s.m == 1 && s.k % 16 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "rmsnorm_add_matmul_pair_bf16_wmma" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| wmma_takes(WmmaKind::Bf16, s.m) && s.m == 1 && s.k % 16 == 0 && s.k <= 2048 && s.out == Some(DType::Bf16))
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

    /// With INTARSIA_HIP_WMMA_ROWS=1 (run the suite with it set; a no-op otherwise): the decode shapes (M 1..=4, the plain
    /// registration; the batched rows twins are `rows_exact` and picked by `exact_rows`) take `matmul_fp8_wmma` exactly where
    /// the table in force says so, per kind (W8A8 / W8A16), and the GEMV where it does not.
    #[test]
    fn wmma_fp8_selection_is_pinned() {
        let Some(t) = wmma_table() else { return };
        let ks = all();
        for (n, k) in [(8192usize, 2048usize), (4096, 2048), (2048, 4096), (512, 2048)] {
            for a8 in [false, true] {
                for m in 1..=4 {
                    let mut s = fp8_shape(m, n, k);
                    s.fp8_static_act = a8;
                    let kind = if a8 { WmmaKind::Fp8A8 } else { WmmaKind::Fp8A16 };
                    let want = if t.takes(kind, m) { "matmul_fp8_wmma" } else { "matmul_fp8_gemv_v2" };
                    assert_eq!(pick(&ks, &s), want, "m={m} n={n} k={k} a8={a8}");
                }
            }
        }
    }

    /// With the knobs on and the table giving W8A8 (or W8A16) every row count 1..=ROWS_MAX: a call with a fragment-order weight
    /// (`ShapeInfo::frag`) has a kernel at EVERY M the engine can emit (decode 1..=16, MTP verify streams x (k + 1) up to 64, prefill
    /// chunks), each of them a fragment reader; and the GEMVs are not among the candidates. Otherwise a tagged weight would fail mid-serve.
    #[test]
    fn wmma_frag_weights_have_a_reader_at_every_row_count() {
        let Some(t) = wmma_table() else { return };
        let ks = all();
        for (a8, kind) in [(true, WmmaKind::Fp8A8), (false, WmmaKind::Fp8A16)] {
            if !fp8_frag_serves(kind) {
                continue;
            }
            for (n, k) in [(8192usize, 2048usize), (2048, 2048), (512, 2048), (2048, 4096), (12288, 2048), (4096, 2048)] {
                for m in (1..=64usize).chain([96, 128, 256, 512, 1024, 4096]) {
                    let mut s = fp8_shape(m, n, k);
                    s.fp8_static_act = a8;
                    s.frag = true;
                    let acc: Vec<&KernelImpl> = ks.iter().filter(|x| x.op == Op::MatmulFp8 && x.accepts(&s)).collect();
                    assert!(!acc.is_empty(), "no kernel for a tagged weight at m={m} n={n} k={k} a8={a8}");
                    for x in &acc {
                        assert!(x.frag && !x.name.contains("gemv"), "{} accepts a tagged weight (m={m})", x.name);
                    }
                    if m <= ROWS_MAX {
                        assert!(t.takes(kind, m));
                        assert!(acc.iter().any(|x| x.name.starts_with("matmul_fp8_wmma")), "m={m}: the family does not serve the tagged weight: {:?}", acc.iter().map(|x| &x.name).collect::<Vec<_>>());
                    }
                }
            }
        }
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
        if wmma_rows_enabled() {
            return; // main's pins; `wmma_fp8_selection_is_pinned` pins the table's selection
        }
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

    /// The MXFP4 fold path (lane mx) is the default: on unless INTARSIA_HIP_MXFP4_FOLD is exactly "0" (the kill switch).
    #[test]
    fn mxfp4_fold_is_default_with_a_kill_switch() {
        assert!(mxfp4_fold_from(None));
        assert!(mxfp4_fold_from(Some("1")));
        assert!(mxfp4_fold_from(Some("")));
        assert!(!mxfp4_fold_from(Some("0")));
        assert_eq!(mxfp4_fold_enabled(), std::env::var("INTARSIA_HIP_MXFP4_FOLD").as_deref() != Ok("0"));
    }

    /// Pins the MXFP4 selection at the 27B's shapes: the GEMVs for 1..=8 rows, `matmul_mxfp4_mma_fold` above (the default),
    /// `matmul_mxfp4_mma` (v2, bitwise v1) there with INTARSIA_HIP_MXFP4_FOLD=0; `matmul_mxfp4_mma_v1` never.
    #[test]
    fn mxfp4_selection_is_pinned() {
        let ks = all();
        let pick_mx = |s: &ShapeInfo| -> String {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == Op::MatmulMxfp4 && k.accepts(s) && !k.rows_exact) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.expect("a MatmulMxfp4 kernel for every shape").name.clone()
        };
        let big = if mxfp4_fold_enabled() { "matmul_mxfp4_mma_fold" } else { "matmul_mxfp4_mma" };
        for (n, k) in [(17408usize, 5120usize), (10240, 5120), (6144, 5120), (12288, 5120), (1024, 5120), (48, 5120), (5120, 6144), (5120, 17408)] {
            for (m, want) in [(1usize, "matmul_mxfp4_gemv"), (8, "matmul_mxfp4_gemv"), (9, big), (64, big), (256, big), (4096, big)] {
                let s = ShapeInfo { m, n, k, act: Some(DType::Bf16), weight: Some(DType::U8), out: Some(DType::Bf16), ..Default::default() };
                assert_eq!(pick_mx(&s), want, "m={m} n={n} k={k}");
            }
        }
        // K % 64 == 32: the fold never takes it
        let s = ShapeInfo { m: 300, n: 512, k: 1056, act: Some(DType::Bf16), weight: Some(DType::U8), out: Some(DType::Bf16), ..Default::default() };
        assert_eq!(pick_mx(&s), "matmul_mxfp4_mma");
    }
}
