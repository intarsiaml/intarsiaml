//! Kernel table of the quantised-weight and MoE launchers (kernels/cuda/{fp8,nvfp4,moe}.cu). Compiled on every
//! host (the table is data); the device dispatch is in `device_quantmoe.rs`. Registered through
//! `kernels::all()`.
//!
//! Selection: `*_gemv` (priority 10) serves decode shapes, `*_mma` (priority 0) every other shape, so each
//! (op, shape) has a kernel; the admission harness grades all of them against the cpu `ref_*` kernels.

use crate::kernels::{opt_in_priority, ARCH};
use intarsia_core::quant::ops::{moe_experts_decode_scratch, moe_experts_mma_scratch, moe_experts_scratch};
use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_fp8_gemv", Op::MatmulFp8),
    ("matmul_fp8_mma", Op::MatmulFp8),
    ("matmul_fp8_gemm", Op::MatmulFp8),
    ("matmul_fp8_gemm_f8", Op::MatmulFp8),
    ("matmul_nvfp4_gemv", Op::MatmulNvfp4),
    ("matmul_nvfp4_mma", Op::MatmulNvfp4),
    ("matmul_nvfp4_gemm", Op::MatmulNvfp4),
    ("moe_topk_bf16", Op::MoeTopk),
    ("moe_experts_bf16", Op::MoeExperts),
    ("moe_experts_decode_bf16", Op::MoeExperts),
    ("moe_experts_mma_bf16", Op::MoeExperts),
    ("moe_combine_bf16", Op::MoeCombine),
    // fused decode launchers (docs/results/2026-10-14-cuda-decode-fusion.md): each is bitwise the chain of launches it
    // replaces; the model falls back to that chain when the registry has no kernel for the exact call
    ("matmul_pair_bf16_gemv", Op::MatmulPair),
    ("matmul_fp8_pair_gemv", Op::MatmulFp8Pair),
    ("matmul_nvfp4_glu_gemv", Op::MatmulNvfp4Glu),
    ("gated_rmsnorm_bf16", Op::GatedRmsnorm),
];

/// Admission budget of the bf16-rounded outputs: a one-ulp flip (up to 7.8e-3 of the largest output) is allowed, which
/// is what max_abs 1.5e-2 (about two ulps) is for. rel_l2 is 1e-3, as in the HIP backend: the admission sweep measures
/// well below it, so a bug that moves outputs by ~1e-3 relative is refused where the former 6e-3 admitted it.
const TOL: Tol = Tol::new(1.5e-2, 1e-3);

/// Budget for the f32 outputs of the quantised matmuls (lm_head logits), picked per output tensor by dtype
/// (`KernelImpl::tol_for`); f32 outputs differ from the reference by accumulation order only.
pub const F32_OUT_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Budget of the fused kernels that end in a silu (`expf`, a division and a bf16 rounding of a product of two roundings):
/// the same as the elementwise kernels' (kernels.rs `BF16_TOL`).
const GLU_TOL: Tol = Tol::new(1.5e-2, 6e-3);

/// Fewest tokens for which the grouped tensor-core experts kernel is preferred over the warp-per-row one (GB10,
/// 256 experts, top-8: 1.43 ms vs 1.53 ms at 32 tokens, 0.98 ms vs 0.88 ms at 16).
const MMA_MIN_TOKENS: usize = 32;

/// Fewest tokens for which the e4m3 tensor-core GEMM serves (the gemv serves 1..=4).
const F8_MIN_TOKENS: usize = 5;

/// Fewest tokens for which the dense dequantise-once GEMMs are preferred over `*_mma`.
const GEMM_MIN_TOKENS: usize = 33;

/// Most tokens for which the two-launch decode experts kernel is preferred (45 us vs 59 us at 1 token; slower than
/// the grouping kernel from 2 tokens on, 130 us vs 119 us).
const DECODE_MAX_TOKENS: usize = 1;

pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "cuda").arch([ARCH]);
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
                // prefill-shaped batches: dequantise-once tensor-core GEMM with the cp.async weight pipeline
                "matmul_fp8_gemm" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.m >= GEMM_MIN_TOKENS && s.k % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2)
                    .priority(10),
                // OPT-IN (INTARSIA_CUDA_FAST_PREFILL=1, see `kernels::fast_prefill`).
                // the same on e4m3 tensor cores (no weight dequantisation); W8A16 batches fall back inside the launcher.
                // Serves every M the gemv does not (>= 5), the shapes `*_mma` and `*_gemm` shared: one rounding for every
                // chunk length (a tail chunk of 5..32 tokens must give the bits a full chunk gives its rows).
                "matmul_fp8_gemm_f8" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.m >= F8_MIN_TOKENS && s.k % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2)
                    .priority(opt_in_priority(20)),
                "matmul_nvfp4_gemm" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.m >= GEMM_MIN_TOKENS && s.k % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(10),
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
                // decode: no grouping launch, per-token reduce folded into the down projection (2 launches, not 4)
                "moe_experts_decode_bf16" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| {
                        s.m <= DECODE_MAX_TOKENS
                            && s.top_k <= 16
                            && s.k % 16 == 0
                            && s.n % 16 == 0
                            && s.top_k * s.n * 2 <= 49152
                            && s.k * 2 <= 49152
                    })
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_experts_decode_scratch)
                    .priority(10),
                // grouped tensor-core experts for batches (prefill): H % 128, I % 128, at least 32 tokens so a typical
                // active expert sees several pairs; decode and tiny batches keep the warp-per-row kernel
                "moe_experts_mma_bf16" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.m >= MMA_MIN_TOKENS && s.experts <= 1024 && s.k % 128 == 0 && s.n % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_experts_mma_scratch)
                    .priority(10),
                "moe_combine_bf16" => k.acts(bf).exact(Exactness::Deterministic).tol(TOL),
                // Fused decode launchers: one activation row, bf16 in and out. Each is bitwise the chain it replaces
                // (tests/fused_decode.rs), so the budget is that of the kernels it is built from.
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
                "gated_rmsnorm_bf16" => k.acts(bf).when(|s| s.out == Some(DType::Bf16)).exact(Exactness::Deterministic).tol(GLU_TOL),
                other => unreachable!("quantmoe kernel {other} has no registration"),
            }
        })
        .collect()
}
