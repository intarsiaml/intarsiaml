//! Kernel table of the quantised-weight and MoE launchers (kernels/cuda/{fp8,nvfp4,moe}.cu). Compiled on every
//! host (the table is data); the device dispatch is in `device_quantmoe.rs`. Registered through
//! `kernels::all()`.
//!
//! Selection: `*_gemv` (priority 10) serves decode shapes, `*_mma` (priority 0) every other shape, so each
//! (op, shape) has a kernel; the admission harness grades all of them against the cpu `ref_*` kernels.

use crate::kernels::{
    fp8_gemm_f8_priority, fp8_gemm_lt_priority, fp8_cublas, fp8_gemm_reg_priority, fp8_pair_reg, moe_nvfp4_priority, moe_w4a16_priority,
    nvfp4_reg, ARCH,
};
use intarsia_core::quant::ops::{moe_experts_decode_scratch, moe_experts_mma_scratch, moe_experts_scratch};
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_fp8_gemv", Op::MatmulFp8),
    ("matmul_fp8_mma", Op::MatmulFp8),
    ("matmul_fp8_gemm", Op::MatmulFp8),
    ("matmul_fp8_gemm_f8", Op::MatmulFp8),
    ("matmul_fp8_gemm_lt", Op::MatmulFp8),
    ("matmul_fp8_gemm_reg", Op::MatmulFp8),
    ("matmul_nvfp4_gemv", Op::MatmulNvfp4),
    ("matmul_nvfp4_mma", Op::MatmulNvfp4),
    ("matmul_nvfp4_gemm", Op::MatmulNvfp4),
    // prefill (M >= 256): weights dequantised in registers, bitwise `matmul_nvfp4_gemm` (kernels/cuda/nvfp4_gemm_reg.cu)
    ("matmul_nvfp4_gemm_reg", Op::MatmulNvfp4),
    // M1 drill (kernels/cuda/mxfp4.cu): OCP MXFP4 (Quark), W4A4 dynamic or W4A16 by the op's act input; any M
    ("matmul_mxfp4_gemv", Op::MatmulMxfp4),
    ("moe_topk_bf16", Op::MoeTopk),
    ("moe_experts_bf16", Op::MoeExperts),
    ("moe_experts_decode_bf16", Op::MoeExperts),
    ("moe_experts_decode_v2_bf16", Op::MoeExperts),
    ("moe_experts_mma_bf16", Op::MoeExperts),
    ("moe_experts_nvfp4_mma_bf16", Op::MoeExperts),
    ("moe_experts_w4a16_bf16", Op::MoeExperts),
    ("moe_combine_bf16", Op::MoeCombine),
    // fused decode launchers (docs/results/2026-10-14-cuda-decode-fusion.md): each is bitwise the chain of launches it
    // replaces; the model falls back to that chain when the registry has no kernel for the exact call
    ("matmul_pair_bf16_gemv", Op::MatmulPair),
    ("matmul_fp8_pair_gemv", Op::MatmulFp8Pair),
    // prefill (M >= 256): the gemm_reg GEMM on cuBLAS's bf16 kernels over exactly expanded e4m3 operands, the same bytes
    // at the 35B shapes on sm_121 / CUDA 13.0 (INTARSIA_CUDA_FP8_CUBLAS; docs/results/2026-10-22-s1-spark-prefill.md)
    ("matmul_fp8_gemm_cublas", Op::MatmulFp8),
    // prefill (M >= 256): the pair as one launcher of the two `matmul_fp8_gemm_reg` GEMMs (one activation quantiser when the
    // input scales are equal, one grid), bitwise the two single calls; docs/results/2026-10-22-s1-spark-prefill.md
    ("matmul_fp8_pair_gemm_reg", Op::MatmulFp8Pair),
    ("matmul_nvfp4_glu_gemv", Op::MatmulNvfp4Glu),
    // prefill (M >= 256): gate + up + silu_glu in one launch, bitwise two `matmul_nvfp4_gemm` and `silu_glu_bf16`
    ("matmul_nvfp4_glu_gemm", Op::MatmulNvfp4Glu),
    ("gated_rmsnorm_bf16", Op::GatedRmsnorm),
    // the residual add + norm (RmsNormAddMatmulPair) or the GDN output norm (GatedRmsnormMatmulFp8) folded into the GEMV that
    // follows it (docs/results/2026-10-17-cuda-norm-gemv.md)
    ("rmsnorm_add_matmul_pair_bf16_gemv", Op::RmsNormAddMatmulPair),
    ("gated_rmsnorm_matmul_fp8_gemv", Op::GatedRmsnormMatmulFp8),
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

/// Fewest tokens for `matmul_fp8_gemm_reg` (bitwise `matmul_fp8_gemm`, 128-row tiles): measured faster per 35B chunk
/// from 256 tokens up on spark1 (fp8 projections of one 256-token chunk 12.5 vs 14.7 ms, 2048 tokens 83 vs 104 ms;
/// docs/results/2026-10-19-cuda-prefill-3.md); not measured below 256, where `matmul_fp8_gemm` stays.
pub const FP8_REG_MIN_TOKENS: usize = 256;

/// Fewest tokens for the register-dequantising NVFP4 prefill kernels (`matmul_nvfp4_gemm_reg`, `matmul_nvfp4_glu_gemm`,
/// bitwise the kernels they replace): the 35B shared expert at prefill-chunk sizes; below, `matmul_nvfp4_gemm` and the
/// unfused gate / up / silu_glu chain stay (docs/results/2026-10-22-s1-spark-prefill.md).
pub const NVFP4_REG_MIN_TOKENS: usize = 256;

/// Most tokens for which the two-launch decode experts kernel is preferred (45 us vs 59 us at 1 token; slower than
/// the grouping kernel from 2 tokens on, 130 us vs 119 us).
const DECODE_MAX_TOKENS: usize = 1;

/// Tolerance of `moe_experts_nvfp4_mma_bf16` against the bf16-activation reference (`ref_moe_experts`). The kernel
/// quantises the activations to e2m1 (a 3-bit value grid), so it is not within `TOL` of the reference; the bound is
/// the quantisation error measured on the admission shapes by tests/moe_nvfp4_cpu.rs (rel_l2 0.163..0.166, max_abs
/// 0.14..0.19 of max(1, max|ref|); two e2m1 roundings of ~8 % each) with a 20 % / 50 % margin. Exactness against
/// the quantised arithmetic is tested separately (tests/moe_nvfp4.rs).
pub const NVFP4_ACT_TOL: Tol = Tol::new(0.30, 0.20);

/// Fewest tokens for the FP4 experts kernel, from the measured crossover against `moe_experts_mma_bf16` on spark1
/// (3 rounds, T = 64 .. 4096, docs/results/2026-10-17-cuda-moe-nvfp4-mma/raw/final/microbench.out): the FP4 kernel is
/// 6 to 14 % slower at T <= 512 (the two activation-quantisation launches are a fixed cost), about even at 1024
/// (2910 vs 2980 us and 3001 vs 3037 us in two rounds) and 22 to 40 % faster at 2048 and 4096.
pub const NVFP4_MIN_TOKENS: usize = 1024;

/// Fewest tokens for `moe_experts_w4a16_bf16` (kernels/cuda/moe_w4a16.cu; bitwise the same as `moe_experts_mma_bf16`),
/// from the measured crossover against `moe_experts_mma_bf16` on spark1: round 6 (docs/results/2026-10-19-cuda-round6.md,
/// 3 rounds, T = 512 .. 1024) has it 3 to 6 % slower at T <= 640 and 2 to 6 % faster from 768 on (768: 2836-2844 vs
/// 2938-2956 us; 1000: 2892-2913 vs 3062-3091 us), so it starts at 768 (it was 1024 from
/// docs/results/2026-10-18-cuda-moe-w4a16.md, which had them even at 1024 and 15 to 24 % apart at 2048 and 4096).
pub const W4A16_MIN_TOKENS: usize = 768;

/// Scratch bytes of `moe_experts_nvfp4_mma_bf16`: the grouped kernel's, plus the quantised activations
/// (`xq` u8 `[T,H/2]`, `xs` e4m3 `[T,H/16]`, `xg` f32 `[T]`, and the same three for the sorted `h` `[P,I]`), each region
/// rounded up to 256 bytes (kernels/cuda/moe_nvfp4_mma_sm120a.cu, `scratch_layout`).
pub fn moe_experts_nvfp4_scratch(s: &ShapeInfo) -> usize {
    let r = |x: usize| (x + 255) & !255;
    let pairs = s.m * s.top_k;
    moe_experts_mma_scratch(s)
        + r(s.m * (s.k / 2))
        + r(s.m * (s.k / 16))
        + r(s.m * 4)
        + r(pairs * (s.n / 2))
        + r(pairs * (s.n / 16))
        + r(pairs * 4)
}

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
                // OPT-IN (INTARSIA_CUDA_FP8_GEMM_F8=1 or INTARSIA_CUDA_FAST_PREFILL=1, see `kernels::fp8_gemm_f8`).
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
                    .priority(fp8_gemm_f8_priority(20)),
                // OPT-IN (INTARSIA_CUDA_FP8_GEMM_LT=1, see `kernels::fp8_gemm_lt`): the same W8A8 op on cuBLASLt's e4m3 GEMM
                // (kernels/cuda/fp8_gemm_lt.cu), docs/results/2026-10-21-cuda-round7.md. Same e4m3 activations as the default
                // path, different accumulation. Serves every M the gemv does not (one pinned algorithm per (N, K): a tail
                // chunk's rows get the bits a full chunk gives them). Above `matmul_fp8_gemm_f8` when both are on.
                "matmul_fp8_gemm_lt" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.m >= F8_MIN_TOKENS && s.k % 128 == 0 && s.n % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2)
                    .priority(fp8_gemm_lt_priority(25)),
                // The same GEMM with the weights dequantised in registers and 128 x 128 tiles (kernels/cuda/fp8_gemm.cu): bitwise
                // the results of `matmul_fp8_gemm`; INTARSIA_CUDA_FP8_GEMM_REG=0 falls back to it.
                "matmul_fp8_gemm_reg" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.m >= FP8_REG_MIN_TOKENS && s.k % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| s.m * s.k * 2)
                    .priority(fp8_gemm_reg_priority(15)),
                "matmul_nvfp4_gemm" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.m >= GEMM_MIN_TOKENS && s.k % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(10),
                "matmul_mxfp4_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.m >= 1 && s.k % 32 == 0)
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
                // the same kernel pair with the down projection restructured for occupancy (bitwise equal; kernels/cuda/moe_decode.cu)
                "moe_experts_decode_v2_bf16" => k
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
                    .priority(20),
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
                // The same grouped experts on the block-scaled FP4 tensor core (kernels/cuda/moe_nvfp4_mma_sm120a.cu):
                // W4A4, activations quantised to e2m1 + ue4m3. Registered only where the file was built for sm_120a /
                // sm_121a (build.rs, cfg `ik_sm120a`); selected only with INTARSIA_CUDA_MOE_NVFP4=1 (see `moe_nvfp4`).
                "moe_experts_nvfp4_mma_bf16" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| {
                        cfg!(ik_sm120a)
                            && s.m >= NVFP4_MIN_TOKENS
                            && s.experts <= 1024
                            && s.k % 128 == 0
                            && s.n % 128 == 0
                    })
                    .exact(Exactness::Deterministic)
                    .tol(NVFP4_ACT_TOL)
                    .scratch(moe_experts_nvfp4_scratch)
                    .priority(moe_nvfp4_priority(20)),
                // The same grouped experts with the weights dequantised in registers (kernels/cuda/moe_w4a16.cu): bitwise the
                // results of `moe_experts_mma_bf16`, a different schedule (two blocks per SM, no bf16 weight tile in shared memory).
                "moe_experts_w4a16_bf16" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.m >= W4A16_MIN_TOKENS && s.experts <= 1024 && s.k % 128 == 0 && s.n % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(moe_experts_mma_scratch)
                    .priority(moe_w4a16_priority(15)),
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
                // bitwise two `matmul_fp8_gemm_reg` calls: served exactly where those serve both halves (M >= 256, the reg
                // GEMM on; never with the opt-in e4m3 GEMMs, which would serve the singles). `INTARSIA_CUDA_FP8_PAIR_REG=0`
                // (or INTARSIA_NO_FUSION=1 in the model) gives the two single calls.
                // above matmul_fp8_gemm_reg (15) where both serve; bf16 output only. Scratch: {alpha, beta}, xq, the bf16 weights
                "matmul_fp8_gemm_cublas" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| fp8_cublas() && s.m >= FP8_REG_MIN_TOKENS && s.k % 128 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .scratch(|s| 256 + 2 * (s.m * s.k + s.n * s.k))
                    .priority(16),
                "matmul_fp8_pair_gemm_reg" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| fp8_pair_reg() && s.m >= FP8_REG_MIN_TOKENS && s.k % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .scratch(|s| 2 * s.m * s.k * 2)
                    .priority(15),
                "matmul_nvfp4_gemm_reg" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| nvfp4_reg() && s.m >= NVFP4_REG_MIN_TOKENS && s.k % 128 == 0 && s.n % 128 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .tol_f32(F32_OUT_TOL)
                    .priority(15),
                "matmul_nvfp4_glu_gemm" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| {
                        nvfp4_reg() && s.m >= NVFP4_REG_MIN_TOKENS && s.k % 128 == 0 && s.n % 64 == 0 && s.out == Some(DType::Bf16)
                    })
                    .exact(Exactness::Deterministic)
                    .tol(GLU_TOL)
                    .priority(15),
                "matmul_nvfp4_glu_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.m == 1 && s.k % 16 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(GLU_TOL)
                    .priority(10),
                // norm prologue + pair GEMV: the row is staged in shared memory (K <= 8192, K % 8 == 0); bitwise rmsnorm_add + pair
                "rmsnorm_add_matmul_pair_bf16_gemv" => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.m == 1 && s.k % 8 == 0 && (8..=8192).contains(&s.k) && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                // the GDN output norm folded into the fp8 out_proj: dv = 128 rows, K = rows * 128 within the smem staging
                "gated_rmsnorm_matmul_fp8_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.m == 1 && s.head_dim == 128 && s.k % 16 == 0 && s.k * 2 <= 49152 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(TOL)
                    .priority(10),
                "gated_rmsnorm_bf16" => k.acts(bf).when(|s| s.out == Some(DType::Bf16)).exact(Exactness::Deterministic).tol(GLU_TOL),
                other => unreachable!("quantmoe kernel {other} has no registration"),
            }
        })
        .collect()
}
