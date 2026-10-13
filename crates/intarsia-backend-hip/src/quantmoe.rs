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
    ("matmul_fp8_mma", Op::MatmulFp8),
    ("matmul_nvfp4_gemv", Op::MatmulNvfp4),
    ("matmul_nvfp4_mma", Op::MatmulNvfp4),
    ("moe_topk_bf16", Op::MoeTopk),
    ("moe_experts_bf16", Op::MoeExperts),
    ("moe_combine_bf16", Op::MoeCombine),
];

/// Admission budget of the bf16-rounded outputs of every kernel here (a one-ulp flip, up to 7.8e-3 of the
/// largest output, is allowed: max_abs 1.5e-2 is about two ulps). rel_l2 is 1e-3: the admission sweep measures at most
/// 1.5e-4 (fp8 mma), so a bug that moves outputs by ~1e-3 relative is refused where the former 6e-3 admitted it. f32
/// outputs of the matmul kernels (lm_head logits) get `F32_OUT_TOL` through `KernelImpl::tol_f32`.
const TOL: Tol = Tol::new(1.5e-2, 1e-3);

/// Budget for f32 outputs of the quantised matmuls, applied per output tensor by dtype (`KernelImpl::tol_for`).
/// Measured worst: max_abs 1.8e-6, rel_l2 5.6e-7 over K/N-tail shapes.
pub const F32_OUT_TOL: Tol = Tol::new(1e-3, 2e-4);

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
                "matmul_fp8_mma" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL),
                "matmul_nvfp4_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=8).contains(&s.m) && s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(10),
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
                "moe_experts_bf16" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| {
                        s.experts <= 1024 && s.k % 16 == 0 && s.n % 16 == 0 && 32768 / (2 * s.k.max(s.n).max(1)) >= 1
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_experts_scratch),
                "moe_combine_bf16" => k.acts(bf).exact(Exactness::Deterministic).tol(TOL),
                other => unreachable!("quantmoe kernel {other} has no registration"),
            }
        })
        .collect()
}
