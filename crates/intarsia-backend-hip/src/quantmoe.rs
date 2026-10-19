//! Kernel table of the quantised-weight and MoE launchers (kernels/hip/{fp8,nvfp4,moe}.hip). Compiled on every
//! host (the table is data); the device dispatch is in `device.rs`. Registered through
//! `kernels::all()`.
//!
//! Selection: `*_gemv` (priority 10) serves decode shapes, `*_mma` (priority 0) every other shape, so each
//! (op, shape) has a kernel; the admission harness grades all of them against the cpu `ref_*` kernels.

use crate::kernels::ARCH;
use intarsia_core::quant::ops::moe_experts_scratch;
use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_fp8_gemv", Op::MatmulFp8),
    ("matmul_fp8_gemv_v2", Op::MatmulFp8),
    ("matmul_fp8_mma", Op::MatmulFp8),
    ("matmul_fp8_mma_f8", Op::MatmulFp8),
    ("matmul_fp8_mma_v2", Op::MatmulFp8),
    ("matmul_nvfp4_gemv", Op::MatmulNvfp4),
    ("matmul_nvfp4_gemv_v2", Op::MatmulNvfp4),
    ("matmul_nvfp4_mma", Op::MatmulNvfp4),
    ("moe_topk_bf16", Op::MoeTopk),
    ("moe_topk_bf16_v2", Op::MoeTopk),
    ("moe_experts_bf16", Op::MoeExperts),
    ("moe_experts_bf16_v2", Op::MoeExperts),
    ("moe_experts_bf16_v3", Op::MoeExperts),
    ("moe_combine_bf16", Op::MoeCombine),
    // fused decode launchers: no other backend has them, the model falls back to the ops they replace
    ("matmul_pair_bf16_gemv", Op::MatmulPair),
    ("matmul_fp8_pair_gemv", Op::MatmulFp8Pair),
    ("matmul_nvfp4_glu_gemv", Op::MatmulNvfp4Glu),
    ("gated_rmsnorm_bf16", Op::GatedRmsnorm),
    ("rmsnorm_add_matmul_pair_bf16_gemv", Op::RmsNormAddMatmulPair),
    ("gated_rmsnorm_matmul_fp8_gemv", Op::GatedRmsnormMatmulFp8),
    // core-op fusion phase 1 (kernels/hip/ik_hip_abi.h): row-local folds, each bitwise the two launches it replaces
    ("rmsnorm_add_matmul_fp8_pair_gemv", Op::RmsNormAddMatmulFp8Pair),
    ("sigmoid_mul_matmul_fp8_gemv", Op::SigmoidMulMatmulFp8),
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
                    .when(|s| (1..=4).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= 49152)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(10),
                "matmul_fp8_gemv_v2" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| (1..=4).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= 49152)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(20),
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
                "matmul_nvfp4_mma" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL),
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
                "moe_combine_bf16" => k.acts(bf).exact(Exactness::Deterministic).tol(TOL),
                // Fused decode launchers: one activation row, bf16 in and out. Each is bitwise the chain it replaces (tests/
                // fused_decode.rs), so the budget is that of the kernels it is built from.
                "matmul_pair_bf16_gemv" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.m == 1 && s.k % 8 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "matmul_fp8_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
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
                    .when(|s| s.m == 1 && s.head_dim == 128 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                // norm prologue + pair GEMV: the row is staged in LDS (K <= 8192, K % 8 == 0); bitwise rmsnorm_add + pair
                // phase-1 folds. The fp8 ones serve one row (decode; the LDS holds the row, K <= 24576), the nvfp4 one the
                // 1..=8 rows of matmul_nvfp4_gemv_v2. HIP has no batched step (no row-exact kernels), so the one-row limit
                // never meets the batch-consistency rule; a row count without a kernel runs the unfused ops.
                "rmsnorm_add_matmul_fp8_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16) && fp8_staging_default()
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "sigmoid_mul_matmul_fp8_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        s.m == 1 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16) && fp8_staging_default()
                    })
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
                    .when(|s| s.m == 1 && s.k % 8 == 0 && (8..=8192).contains(&s.k) && s.out == Some(DType::Bf16))
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
        for k in ks.iter().filter(|k| k.op == Op::MatmulFp8 && k.accepts(s)) {
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
            for k in ks.iter().filter(|k| k.op == Op::MoeExperts && k.accepts(&m)) {
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
