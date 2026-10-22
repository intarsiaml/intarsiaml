//! The kernel table: which launcher implements which op, for which shapes, and how closely it must
//! match the reference. Compiled on every host (the table is data); only `device.rs` needs CUDA.
//!
//! `name` is the launcher name without the `ik_` prefix (kernels/abi/ik_abi.h).

use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

/// Architecture the .cu files were compiled for (`INTARSIA_CUDA_ARCH`, default sm_121).
pub const ARCH: &str = env!("IK_CUDA_ARCH");

/// (launcher name, op) for every implementation, in registration order.
pub const TABLE: &[(&str, Op)] = &[
    ("embedding_bf16", Op::Embedding),
    ("rmsnorm_bf16", Op::RmsNorm),
    ("rmsnorm_add_bf16", Op::RmsNormAdd),
    ("add_bf16", Op::Add),
    ("qk_norm_bf16", Op::QkNorm),
    ("rope_bf16", Op::Rope),
    ("matmul_bf16", Op::Matmul),
    ("matmul_bf16_gemv", Op::Matmul),
    ("logits_bf16", Op::Logits),
    ("logits_bf16_gemv", Op::Logits),
    ("kv_append_bf16", Op::KvAppend),
    ("attn_prefill_bf16", Op::AttnPrefill),
    ("attn_decode_bf16", Op::AttnDecode),
    ("attn_prefill_bf16_flash", Op::AttnPrefill),
    ("attn_prefill_bf16_splitkv", Op::AttnPrefill),
    ("attn_decode_bf16_splitkv", Op::AttnDecode),
    ("attn_prefill_f8kv_flash", Op::AttnPrefill),
    ("attn_prefill_f8kv_splitkv", Op::AttnPrefill),
    ("attn_decode_f8kv_splitkv", Op::AttnDecode),
    ("silu_glu_bf16", Op::SiluGlu),
    // g4 (Gemma 4): the extended attributes (OpAttrs::window / rope_prop / scale on Add and Embedding, Op::GeluGlu), each a
    // separate launcher so the plain ones keep their bits; registered `.ext()`, see `all()`
    ("gelu_glu_bf16", Op::GeluGlu),
    ("rope_prop_bf16", Op::Rope),
    ("add_scaled_bf16", Op::Add),
    ("embedding_scaled_bf16", Op::Embedding),
    ("attn_prefill_bf16_ext", Op::AttnPrefill),
    ("attn_decode_bf16_ext", Op::AttnDecode),
    ("attn_prefill_bf16_splitkv_ext", Op::AttnPrefill),
    ("attn_decode_bf16_splitkv_ext", Op::AttnDecode),
    ("gate_split_bf16", Op::GateSplit),
    ("sigmoid_mul_bf16", Op::SigmoidMul),
    ("gdn_conv_bf16", Op::GdnConv),
    ("gdn_delta_rule_bf16", Op::GdnDeltaRule),
    ("gdn_delta_rule_bf16_lanes", Op::GdnDeltaRule),
    ("gdn_delta_rule_bf16_split", Op::GdnDeltaRule),
    ("argmax_f32", Op::Argmax),
    ("argmax_bf16", Op::Argmax),
];

/// bf16-rounded outputs: a one-ulp flip is allowed (same budget as the cpu backend).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
/// f32 outputs of bf16 inputs: only accumulation-order noise.
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

fn base(name: &str, op: Op) -> KernelImpl {
    KernelImpl::new(op, name, "cuda").arch([ARCH])
}

/// Most query rows the split-KV kernels take per launch (above this the flash prefill kernel wins).
const SPLITKV_MAX_ROWS: usize = 16;

/// Split-KV partitioning policy, mirroring `splitkv_count` in kernels/cuda/attention_splitkv.cu (the launcher
/// derives its split count from the key count only, so results are run-to-run identical; this copy must be an
/// upper bound of it, which `splitkv_scratch` relies on).
/// Returns the number of KV splits for `rows` query rows over `kv_len` keys and `n_kv_heads` kv heads.
pub fn splitkv_parts(kv_len: usize, n_kv_heads: usize, rows: usize) -> usize {
    const MIN_KEYS: usize = 64;
    const MAX_SPLIT: usize = 64;
    const TARGET_BLOCKS: usize = 256;
    let by_len = kv_len.div_ceil(MIN_KEYS);
    let by_fill = TARGET_BLOCKS.div_ceil((n_kv_heads * rows).max(1));
    by_len.min(by_fill).clamp(1, MAX_SPLIT)
}

/// Scratch the split-KV kernels need: float acc[T][N][nsplit][D] + float2 ml[T][N][nsplit]. Declared as an
/// upper bound that does not depend on `kv_len` (the launcher needs at most this for any context length), so a
/// short chunk at a long position can never need more than a plan probe at a short one reserved.
fn splitkv_scratch(s: &ShapeInfo) -> usize {
    let nsplit = splitkv_parts(usize::MAX / 2, s.n_kv_heads, s.m);
    s.m * s.n_heads * nsplit * (s.head_dim + 2) * 4
}

fn splitkv_ok(s: &ShapeInfo) -> bool {
    matches!(s.head_dim, 32 | 64 | 128 | 256)
        && s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && s.n_heads / s.n_kv_heads <= 8
}

fn flash_ok(s: &ShapeInfo) -> bool {
    matches!(s.head_dim, 64 | 128 | 256) && s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0
}

/// `INTARSIA_CUDA_FAST_PREFILL=1` selects the two prefill kernels of docs/results/2026-10-13-prefill-cuda.md
/// (`gdn_delta_rule_bf16_lanes`, `matmul_fp8_gemm_f8`) over the kernels they were measured against. Both pass admission
/// and are faster, but each reorders fp32 sums, and the 35B accuracy gates (docs/accuracy.md) of the two together do not
/// pass: the per-prompt limits are calibrated on an attention / prefill-chunk noise envelope that does not include GEMM or
/// recurrence rounding. They stay registered and tested; they are not selected until that gate is re-calibrated. The GDN
/// kernel alone has its own switch, `INTARSIA_CUDA_GDN_LANES` (see `gdn_lanes`).
pub fn fast_prefill() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_FAST_PREFILL").is_ok_and(|v| v == "1"))
}

/// `INTARSIA_CUDA_GDN_LANES=1` selects the several-lanes-per-row GatedDeltaNet recurrence (`gdn_delta_rule_bf16_lanes`,
/// prefill and decode) without the e4m3 GEMM that `INTARSIA_CUDA_FAST_PREFILL=1` also turns on; `INTARSIA_CUDA_FAST_PREFILL=1`
/// still selects it as before (docs/results/2026-10-19-cuda-prefill-3.md).
pub fn gdn_lanes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_GDN_LANES").is_ok_and(|v| v == "1") || fast_prefill())
}

/// Priority of the lanes GDN kernel: `on` when `gdn_lanes()`, below everything otherwise.
pub fn gdn_lanes_priority(on: i32) -> i32 {
    if gdn_lanes() {
        on
    } else {
        -10
    }
}

/// `INTARSIA_CUDA_MOE_NVFP4=1` selects the W4A4 block-scaled FP4 grouped experts kernel for prefill-shaped MoE batches
/// (`moe_experts_nvfp4_mma_bf16`). Off by default: the activation quantisation is a numerics change and is not default-on
/// until the accuracy gates pass (docs/results/2026-10-17-cuda-moe-nvfp4-mma.md).
pub fn moe_nvfp4() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_MOE_NVFP4").is_ok_and(|v| v == "1"))
}

/// Priority of the FP4 experts kernel: `on` when opted in, below everything otherwise.
pub fn moe_nvfp4_priority(on: i32) -> i32 {
    if moe_nvfp4() {
        on
    } else {
        -10
    }
}

/// The register-dequantising grouped experts kernel (`moe_experts_w4a16_bf16`, bitwise the results of
/// `moe_experts_mma_bf16`, docs/results/2026-10-18-cuda-moe-w4a16.md) is selected for prefill-shaped MoE batches by default;
/// `INTARSIA_CUDA_MOE_W4A16=0` falls back to `moe_experts_mma_bf16`.
pub fn moe_w4a16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_MOE_W4A16").map_or(true, |v| v != "0"))
}

/// Priority of the W4A16 experts kernel: `on` unless switched off, below everything then.
pub fn moe_w4a16_priority(on: i32) -> i32 {
    if moe_w4a16() {
        on
    } else {
        -10
    }
}

/// The register-dequantising fp8 GEMM (`matmul_fp8_gemm_reg`, bitwise the results of `matmul_fp8_gemm`,
/// docs/results/2026-10-19-cuda-prefill-3.md) is selected for prefill-shaped fp8 projections by default;
/// `INTARSIA_CUDA_FP8_GEMM_REG=0` falls back to `matmul_fp8_gemm`.
pub fn fp8_gemm_reg() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_FP8_GEMM_REG").map_or(true, |v| v != "0"))
}

/// Priority of the register-dequantising fp8 GEMM: `on` unless switched off, below everything then.
pub fn fp8_gemm_reg_priority(on: i32) -> i32 {
    if fp8_gemm_reg() {
        on
    } else {
        -10
    }
}

/// `matmul_fp8_pair_gemm_reg` (Op::MatmulFp8Pair at prefill, bitwise the two `matmul_fp8_gemm_reg` calls it replaces,
/// docs/results/2026-10-22-s1-spark-prefill.md) serves exactly where `matmul_fp8_gemm_reg` would serve both halves: the reg
/// GEMM on and neither opt-in e4m3 GEMM (`INTARSIA_CUDA_FP8_GEMM_F8` / `_LT` / `FAST_PREFILL`, which then serve the single
/// calls). `INTARSIA_CUDA_FP8_PAIR_REG=0` switches it off.
pub fn fp8_pair_reg() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INTARSIA_CUDA_FP8_PAIR_REG").map_or(true, |v| v != "0")
            && fp8_gemm_reg()
            && !fp8_gemm_f8()
            && !fp8_gemm_lt()
            && !fp8_cublas()
    })
}

/// `matmul_fp8_gemm_cublas`: the prefill fp8 projections on cuBLAS's bf16 GEMM over exactly expanded e4m3 operands
/// (the bytes of `matmul_fp8_gemm_reg` where cuBLAS runs the same m16n8k16 chain; docs/results/2026-10-22-s1-spark-prefill.md).
/// `INTARSIA_CUDA_FP8_CUBLAS=1` turns it on (and the gemm_reg pair off: the two single calls then run on cuBLAS); off with an
/// opt-in e4m3 GEMM, which serves those calls instead.
pub fn fp8_cublas_priority(on: i32) -> i32 {
    if fp8_cublas() {
        on
    } else {
        -10
    }
}

/// Whether `matmul_fp8_gemm_cublas` is selected (see `fp8_cublas_priority`).
pub fn fp8_cublas() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("INTARSIA_CUDA_FP8_CUBLAS").is_ok_and(|v| v == "1") && !fp8_gemm_f8() && !fp8_gemm_lt()
    })
}

/// The register-dequantising dense NVFP4 prefill kernels (`matmul_nvfp4_gemm_reg`, `matmul_nvfp4_glu_gemm`; bitwise the
/// kernels they replace, docs/results/2026-10-22-s1-spark-prefill.md) are on by default; `INTARSIA_CUDA_NVFP4_REG=0`
/// switches them off (the 35B shared expert then runs `matmul_nvfp4_gemm` x 3 and `silu_glu_bf16` again).
pub fn nvfp4_reg() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_NVFP4_REG").map_or(true, |v| v != "0"))
}

/// `INTARSIA_CUDA_FP8_GEMM_F8=1` selects the e4m3 tensor-core W8A8 GEMM (`matmul_fp8_gemm_f8`, every M >= 5) without the
/// lanes GDN kernel that `INTARSIA_CUDA_FAST_PREFILL=1` also turns on; `INTARSIA_CUDA_FAST_PREFILL=1` still selects it as
/// before. Opt-in: docs/results/2026-10-19-cuda-prefill-4.md.
pub fn fp8_gemm_f8() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_FP8_GEMM_F8").is_ok_and(|v| v == "1") || fast_prefill())
}

/// Priority of the e4m3 tensor-core fp8 GEMM: `on` when `fp8_gemm_f8()`, below everything otherwise.
pub fn fp8_gemm_f8_priority(on: i32) -> i32 {
    if fp8_gemm_f8() {
        on
    } else {
        -10
    }
}

/// `INTARSIA_CUDA_FP8_GEMM_LT=1` selects the cuBLASLt e4m3 W8A8 GEMM (`matmul_fp8_gemm_lt`, every M >= 5) for the fp8
/// projections. OPT-IN: same e4m3 activations as the default path, different accumulation, so not bitwise; its KL is
/// gated against vLLM's own run-to-run KL in docs/results/2026-10-21-cuda-round7.md.
pub fn fp8_gemm_lt() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_FP8_GEMM_LT").is_ok_and(|v| v == "1"))
}

/// Priority of the cuBLASLt e4m3 GEMM: `on` when `fp8_gemm_lt()`, below everything otherwise.
pub fn fp8_gemm_lt_priority(on: i32) -> i32 {
    if fp8_gemm_lt() {
        on
    } else {
        -10
    }
}

/// `gdn_delta_rule_bf16_split` (the one-thread-per-row GatedDeltaNet recurrence's arithmetic split over several lanes per
/// row, BITWISE the same results, docs/results/2026-10-19-cuda-prefill-4.md) is on by default; `INTARSIA_CUDA_GDN_SPLIT=0`
/// switches it off (the one-thread-per-row kernel `gdn_delta_rule_bf16` then serves dk 64 / 128 again).
pub fn gdn_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_CUDA_GDN_SPLIT").map_or(true, |v| v != "0"))
}

/// Priority of the split GDN kernel: `on` unless switched off, below everything then.
pub fn gdn_split_priority(on: i32) -> i32 {
    if gdn_split() {
        on
    } else {
        -10
    }
}

/// Every kernel this backend registers.
pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    let mut v = Vec::new();
    for (name, op) in TABLE.iter().copied() {
        let k = base(name, op);
        let k = match name {
            "embedding_bf16" => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT),
            "rmsnorm_bf16" | "rmsnorm_add_bf16" | "add_bf16" | "qk_norm_bf16" | "silu_glu_bf16" => {
                k.acts(bf).exact(Exactness::Deterministic).tol(BF16_TOL)
            }
            "rope_bf16" => k.acts(bf).when(|s| s.head_dim % 2 == 0).exact(Exactness::Deterministic).tol(BF16_TOL),
            // cuBLAS: tensor cores, algorithm (split-K) may change with the shape or the library version
            "matmul_bf16" => k.acts(bf).weights(bf).exact(Exactness::Fast).tol(BF16_TOL).priority(0),
            "logits_bf16" => k.acts(bf).weights(bf).exact(Exactness::Fast).tol(F32_TOL).priority(0),
            // native decode GEMV: fixed reduction order, run-to-run identical
            "matmul_bf16_gemv" => k
                .acts(bf)
                .weights(bf)
                .when(|s| (1..=4).contains(&s.m))
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(10),
            "logits_bf16_gemv" => k
                .acts(bf)
                .weights(bf)
                .when(|s| (1..=4).contains(&s.m))
                .exact(Exactness::Deterministic)
                .tol(F32_TOL)
                .priority(10),
            // the cache dtype (bf16 or fp8_e4m3) is the output's; the launcher dispatches on it
            "kv_append_bf16" => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT),
            "attn_prefill_bf16" | "attn_decode_bf16" => k
                .acts(bf)
                .weights(bf)
                .when(|s| s.head_dim > 0 && s.head_dim <= 256 && s.head_dim % 2 == 0)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL),
            // tensor-core flash prefill (mma.sync bf16): ranks above the SIMT fallback for chunks of > 16 rows
            "attn_prefill_bf16_flash" => k
                .acts(bf)
                .weights(bf)
                .when(|s| flash_ok(s) && s.m > SPLITKV_MAX_ROWS)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(10),
            // split-KV (flash-decoding): decode and short q_len (verify); needs scratch
            "attn_decode_bf16_splitkv" | "attn_prefill_bf16_splitkv" => k
                .acts(bf)
                .weights(bf)
                .when(|s| splitkv_ok(s) && s.m <= SPLITKV_MAX_ROWS)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(10)
                .scratch(splitkv_scratch),
            // fp8_e4m3 KV cache (OPT-IN: only an fp8 cache, `--kv-dtype fp8_e4m3`, selects them; the default cache is bf16):
            // the flash / split-KV kernels above over e4m3 K / V, converted exactly to bf16 / f32 on load. Bitwise the bf16
            // kernels on the upcast cache (tests/attn_f8kv.rs); the accuracy cost is the cache's rounding to e4m3
            // (docs/results/2026-10-19-cuda-round5.md).
            "attn_prefill_f8kv_flash" => k
                .acts(bf)
                .weights([DType::Fp8E4m3])
                .when(|s| flash_ok(s) && s.m > SPLITKV_MAX_ROWS)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(10),
            "attn_decode_f8kv_splitkv" | "attn_prefill_f8kv_splitkv" => k
                .acts(bf)
                .weights([DType::Fp8E4m3])
                .when(|s| splitkv_ok(s) && s.m <= SPLITKV_MAX_ROWS)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(10)
                .scratch(splitkv_scratch),
            // g4 (Gemma 4) extended kernels. All `.ext()`: only a call that uses the extended attributes (or, for the attention,
            // a head_dim the plain kernels do not take) can pick them, and at priority -1 they sit below every plain kernel, so
            // a plain call selects exactly what it did before.
            "gelu_glu_bf16" => k.acts(bf).ext().exact(Exactness::Deterministic).tol(BF16_TOL),
            // proportional rope: pairs (j, j + D/2) for j < rotary_dim / 2, exponent denominator D
            "rope_prop_bf16" => k
                .acts(bf)
                .ext()
                .when(|s| s.head_dim % 2 == 0)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(-1),
            "add_scaled_bf16" | "embedding_scaled_bf16" => {
                k.acts(bf).ext().exact(Exactness::Deterministic).tol(BF16_TOL).priority(-1)
            }
            // SIMT attention, head_dim <= 512 and the sliding window; correctness first (one block per query row and head)
            "attn_prefill_bf16_ext" | "attn_decode_bf16_ext" => k
                .acts(bf)
                .weights(bf)
                .ext()
                .when(|s| s.head_dim > 0 && s.head_dim <= 512 && s.head_dim % 2 == 0 && s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(-1),
            // the split-KV kernels with the sliding window (decode and short q_len); like the plain split-KV ones they need scratch.
            // Priority 9: below the plain split-KV kernels (10, which never accept an ext call), above the SIMT ones (-1)
            "attn_decode_bf16_splitkv_ext" | "attn_prefill_bf16_splitkv_ext" => k
                .acts(bf)
                .weights(bf)
                .ext()
                .when(|s| splitkv_ok(s) && s.m <= SPLITKV_MAX_ROWS)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(9)
                .scratch(splitkv_scratch),
            "gate_split_bf16" | "sigmoid_mul_bf16" => k.acts(bf).exact(Exactness::Deterministic).tol(BF16_TOL),
            "gdn_conv_bf16" => k.acts(bf).weights(bf).when(|s| (2..=8).contains(&s.k)).exact(Exactness::Deterministic).tol(BF16_TOL),
            // sequential recurrence, one warp per state row: fixed reduction order
            "gdn_delta_rule_bf16" => k
                .acts(bf)
                .weights(bf)
                .when(|s| matches!(s.head_dim, 32 | 64 | 128 | 256) && s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL),
            // OPT-IN (INTARSIA_CUDA_GDN_LANES=1 or INTARSIA_CUDA_FAST_PREFILL=1, see `gdn_lanes`).
            // several lanes per state row (kernels/cuda/gdn.cu): ~4x the threads of the one-thread-per-row kernel, 1.4x
            // faster at T = 2048 and 2.3x at T = 1. It serves EVERY T (decode included): the recurrence is token-sequential
            // and this kernel's per-token arithmetic does not depend on T, so a prompt gives the same bits however it is
            // chunked (prefix reuse, tail chunks) and in decode; the one-thread-per-row kernel stays for other head dims.
            "gdn_delta_rule_bf16_lanes" => k
                .acts(bf)
                .weights(bf)
                .when(|s| {
                    matches!(s.head_dim, 64 | 128)
                        && s.n_kv_heads > 0
                        && s.n_heads % s.n_kv_heads == 0
                })
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(gdn_lanes_priority(10)),
            // DEFAULT (INTARSIA_CUDA_GDN_SPLIT=0 switches it off, see `gdn_split`). The one-thread-per-row kernel's fp32
            // operations in the same order (kernels/cuda/gdn.cu), the element-wise part split over several lanes per row:
            // bitwise the same outputs and state for every T, so it replaces `gdn_delta_rule_bf16` for dk 64 / 128 in
            // prefill and decode. Below the opt-in lanes kernel (`INTARSIA_CUDA_GDN_LANES=1`), which is not bitwise.
            "gdn_delta_rule_bf16_split" => k
                .acts(bf)
                .weights(bf)
                .when(|s| {
                    matches!(s.head_dim, 64 | 128)
                        && s.n_kv_heads > 0
                        && s.n_heads % s.n_kv_heads == 0
                })
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(gdn_split_priority(5)),
            "argmax_f32" => k.acts([DType::F32]).exact(Exactness::Exact).tol(Tol::EXACT),
            "argmax_bf16" => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT),
            other => unreachable!("kernel {other} has no registration"),
        };
        v.push(k);
    }
    v.extend(crate::quantmoe::all());
    v.extend(crate::exl3::all());
    v
}

/// (launcher name, op) of every launcher: this file's table plus the quantised/MoE table.
pub fn full_table() -> Vec<(&'static str, Op)> {
    TABLE.iter().chain(crate::quantmoe::TABLE.iter()).chain(crate::exl3::TABLE.iter()).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn splitkv_scratch_bounds_every_context_length() {
        for (n, nkv, d) in [(32, 8, 128), (16, 2, 256), (6, 2, 32), (40, 8, 128)] {
            for m in 1..=SPLITKV_MAX_ROWS {
                let mut s = ShapeInfo { m, n_heads: n, n_kv_heads: nkv, head_dim: d, ..Default::default() };
                let bound = splitkv_scratch(&s);
                for kv in [m, 64, 285 + m, 1000, 4096, 17000, 262144] {
                    s.kv_len = kv;
                    let need = m * n * splitkv_parts(kv, nkv, m) * (d + 2) * 4;
                    assert!(need <= bound, "m={m} kv={kv}: need {need} > bound {bound}");
                }
            }
        }
    }

    /// The grouped experts kernel the registry picks for a MoE call of `m` tokens at the 35B layer shape (256 experts, top-8,
    /// H 2048, I 512) with the default environment: `moe_experts_w4a16_bf16` from 1024 tokens, `moe_experts_mma_bf16` for
    /// 32..1023, the warp-per-row decode kernel below. A kernel that is bitwise the old one cannot be told apart by an
    /// accuracy gate, so this is the check that it is the one in use.
    #[test]
    fn moe_experts_selection_by_batch_size() {
        let pick = |m: usize| {
            let s = ShapeInfo {
                m,
                n: 1024,
                k: 2048,
                experts: 256,
                top_k: 8,
                act: Some(DType::Bf16),
                weight: Some(DType::U8),
                out: Some(DType::Bf16),
                ..Default::default()
            };
            let mut best: Option<KernelImpl> = None;
            for k in all().into_iter().filter(|k| k.op == Op::MoeExperts && k.accepts(&s)) {
                if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.to_string())
        };
        if std::env::var("INTARSIA_CUDA_MOE_W4A16").is_ok() || std::env::var("INTARSIA_CUDA_MOE_NVFP4").is_ok() {
            return; // the default selection is what is pinned
        }
        assert_eq!(pick(2048), Some("moe_experts_w4a16_bf16".to_string()));
        assert_eq!(pick(4096), Some("moe_experts_w4a16_bf16".to_string()));
        assert_eq!(pick(1024), Some("moe_experts_w4a16_bf16".to_string()));
        assert_eq!(pick(1000), Some("moe_experts_w4a16_bf16".to_string()));
        assert_eq!(pick(768), Some("moe_experts_w4a16_bf16".to_string()));
        assert_eq!(pick(767), Some("moe_experts_mma_bf16".to_string()));
        assert_eq!(pick(256), Some("moe_experts_mma_bf16".to_string()));
    }

    /// The GatedDeltaNet recurrence kernel the registry picks at the 35B shape (dk = dv = 128, 16 key heads, 32 value heads)
    /// for decode (T = 1) and a prefill chunk (T = 2048), and at dk = 64: the split kernel (bitwise the one-thread-per-row
    /// kernel) by default, the one-thread-per-row kernel with `INTARSIA_CUDA_GDN_SPLIT=0`, the lanes kernel with
    /// `INTARSIA_CUDA_GDN_LANES=1` or `INTARSIA_CUDA_FAST_PREFILL=1` (the switches are read once per process, so the test
    /// pins whichever the environment of this run asks for). dk 32 and 256 always get `gdn_delta_rule_bf16`.
    #[test]
    fn gdn_delta_rule_selection() {
        let pick_dk = |m: usize, dk: usize| {
            let s = ShapeInfo {
                m,
                n: 128,
                k: 128,
                head_dim: dk,
                n_heads: 32,
                n_kv_heads: 16,
                act: Some(DType::Bf16),
                weight: Some(DType::Bf16),
                out: Some(DType::Bf16),
                ..Default::default()
            };
            let mut best: Option<KernelImpl> = None;
            for k in all().into_iter().filter(|k| k.op == Op::GdnDeltaRule && k.accepts(&s)) {
                if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.to_string())
        };
        let pick = |m: usize| pick_dk(m, 128);
        let lanes = std::env::var("INTARSIA_CUDA_GDN_LANES").is_ok_and(|v| v == "1")
            || std::env::var("INTARSIA_CUDA_FAST_PREFILL").is_ok_and(|v| v == "1");
        let split = std::env::var("INTARSIA_CUDA_GDN_SPLIT").map_or(true, |v| v != "0");
        let want = if lanes {
            "gdn_delta_rule_bf16_lanes"
        } else if split {
            "gdn_delta_rule_bf16_split"
        } else {
            "gdn_delta_rule_bf16"
        };
        for m in [1, 2, 2048] {
            assert_eq!(pick(m), Some(want.to_string()), "T={m}");
            assert_eq!(pick_dk(m, 64), Some(want.to_string()), "dk=64 T={m}");
            // head dims the split and lanes kernels do not take stay on the one-thread / one-warp-per-row kernel
            assert_eq!(pick_dk(m, 32), Some("gdn_delta_rule_bf16".to_string()), "dk=32 T={m}");
            assert_eq!(pick_dk(m, 256), Some("gdn_delta_rule_bf16".to_string()), "dk=256 T={m}");
        }
    }

    /// The opt-in cuBLASLt e4m3 GEMM's own range, whatever the environment: every M from 5 (the gemv keeps 1..4), K a
    /// multiple of 128, N a multiple of 16; off (below every other fp8 kernel) unless INTARSIA_CUDA_FP8_GEMM_LT=1.
    #[test]
    fn fp8_gemm_lt_range() {
        let lt = all().into_iter().find(|k| k.name == "matmul_fp8_gemm_lt").expect("registered");
        let s = |m: usize, n: usize, k: usize| ShapeInfo {
            m,
            n,
            k,
            act: Some(DType::Bf16),
            weight: Some(DType::Fp8E4m3),
            out: Some(DType::Bf16),
            ..Default::default()
        };
        assert!(!lt.accepts(&s(4, 8192, 2048)));
        for m in [5, 6, 255, 2048, 4096] {
            assert!(lt.accepts(&s(m, 8192, 2048)), "M={m}");
        }
        assert!(!lt.accepts(&s(2048, 8192, 2112)), "K % 128");
        assert!(!lt.accepts(&s(2048, 8200, 2048)), "N % 16");
        let on = std::env::var("INTARSIA_CUDA_FP8_GEMM_LT").is_ok_and(|v| v == "1");
        assert_eq!(lt.priority, if on { 25 } else { -10 });
    }

    /// The fp8 GEMM the registry picks for the 35B W8A8 projections (K 2048 / 4096): the gemv for 1..4 tokens, the mma
    /// kernel for 5..32, `matmul_fp8_gemm` for 33..255, `matmul_fp8_gemm_reg` from 256 (the 2048-token chunk included), unless
    /// switched off (`INTARSIA_CUDA_FP8_GEMM_REG=0`: `matmul_fp8_gemm`) or overridden by `INTARSIA_CUDA_FP8_GEMM_F8=1` / `INTARSIA_CUDA_FAST_PREFILL=1` (then `matmul_fp8_gemm_f8` from 5). The
    /// kernel is bitwise `matmul_fp8_gemm`, so no accuracy gate can tell which one ran: this pins it.
    #[test]
    fn fp8_gemm_selection_by_batch_size() {
        let pick = |m: usize, k: usize| {
            let s = ShapeInfo {
                m,
                n: 8192,
                k,
                act: Some(DType::Bf16),
                weight: Some(DType::Fp8E4m3),
                out: Some(DType::Bf16),
                ..Default::default()
            };
            let mut best: Option<KernelImpl> = None;
            for k in all().into_iter().filter(|k| k.op == Op::MatmulFp8 && k.accepts(&s)) {
                if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.to_string())
        };
        if std::env::var("INTARSIA_CUDA_FP8_GEMM_LT").is_ok_and(|v| v == "1") {
            // the cuBLASLt e4m3 GEMM serves every M the gemv does not, above the f8 kernel when both are on
            for k in [2048, 4096] {
                assert_eq!(pick(4, k).as_deref(), Some("matmul_fp8_gemv"));
                for m in [5, 32, 33, 255, 256, 2048, 4096] {
                    assert_eq!(pick(m, k).as_deref(), Some("matmul_fp8_gemm_lt"), "lt gate: M={m} K={k}");
                }
            }
            return;
        }
        if std::env::var("INTARSIA_CUDA_FP8_GEMM_F8").is_ok_and(|v| v == "1")
            || std::env::var("INTARSIA_CUDA_FAST_PREFILL").is_ok_and(|v| v == "1")
        {
            // the e4m3 tensor-core GEMM serves every M the gemv does not (one rounding for every chunk length)
            for k in [2048, 4096] {
                assert_eq!(pick(4, k).as_deref(), Some("matmul_fp8_gemv"));
                for m in [5, 32, 33, 255, 256, 2048, 4096] {
                    assert_eq!(pick(m, k).as_deref(), Some("matmul_fp8_gemm_f8"), "f8 gate: M={m} K={k}");
                }
            }
            return;
        }
        let big = if std::env::var("INTARSIA_CUDA_FP8_CUBLAS").is_ok_and(|v| v == "1") {
            "matmul_fp8_gemm_cublas" // lane S1: bf16 output (the probe shape)
        } else if std::env::var("INTARSIA_CUDA_FP8_GEMM_REG").is_ok_and(|v| v == "0") {
            "matmul_fp8_gemm"
        } else {
            "matmul_fp8_gemm_reg"
        };
        // the measured threshold (docs/results/2026-10-19-cuda-prefill-3.md), as a literal so a change shows up here
        assert_eq!(crate::quantmoe::FP8_REG_MIN_TOKENS, 256);
        for k in [2048, 4096] {
            assert_eq!(pick(1, k).as_deref(), Some("matmul_fp8_gemv"));
            assert_eq!(pick(4, k).as_deref(), Some("matmul_fp8_gemv"));
            assert_eq!(pick(5, k).as_deref(), Some("matmul_fp8_mma"));
            assert_eq!(pick(32, k).as_deref(), Some("matmul_fp8_mma"));
            assert_eq!(pick(33, k).as_deref(), Some("matmul_fp8_gemm"));
            assert_eq!(pick(255, k).as_deref(), Some("matmul_fp8_gemm"));
            for m in [256, 2048, 4096] {
                assert_eq!(pick(m, k).as_deref(), Some(big), "M={m} K={k}");
            }
        }
    }

    /// `Op::MatmulFp8Pair` (GDN qkv + z, attention k + v of the 35B): the gemv at one row, nothing from 2 to 255 rows (the
    /// model then runs the two single projections), the prefill pair launcher from 256 rows, exactly where
    /// `matmul_fp8_gemm_reg` serves both singles; never when an opt-in e4m3 GEMM or the kill switch is on.
    #[test]
    fn fp8_pair_selection_by_batch_size() {
        let pick = |m: usize, k: usize| {
            let s = ShapeInfo {
                m,
                n: 8192,
                k,
                act: Some(DType::Bf16),
                weight: Some(DType::Fp8E4m3),
                out: Some(DType::Bf16),
                ..Default::default()
            };
            let mut best: Option<KernelImpl> = None;
            for k in all().into_iter().filter(|k| k.op == Op::MatmulFp8Pair && k.accepts(&s)) {
                if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.to_string())
        };
        let env_on = |v: &str| std::env::var(v).is_ok_and(|x| x == "1");
        let off = env_on("INTARSIA_CUDA_FP8_GEMM_LT")
            || env_on("INTARSIA_CUDA_FP8_GEMM_F8")
            || env_on("INTARSIA_CUDA_FAST_PREFILL")
            || std::env::var("INTARSIA_CUDA_FP8_GEMM_REG").is_ok_and(|v| v == "0")
            || std::env::var("INTARSIA_CUDA_FP8_PAIR_REG").is_ok_and(|v| v == "0")
            || std::env::var("INTARSIA_CUDA_FP8_CUBLAS").is_ok_and(|v| v == "1");
        assert_eq!(pick(1, 2048).as_deref(), Some("matmul_fp8_pair_gemv"));
        for m in [2, 4, 5, 32, 33, 255] {
            assert_eq!(pick(m, 2048), None, "M={m}");
        }
        for m in [256, 2048, 4096] {
            let want = if off { None } else { Some("matmul_fp8_pair_gemm_reg") };
            assert_eq!(pick(m, 2048).as_deref(), want, "M={m}");
        }
    }

    /// The 35B shared expert's NVFP4 projections: `Op::MatmulNvfp4` (gate / up 512 x 2048, down 2048 x 512) and the fused
    /// `Op::MatmulNvfp4Glu` (gate + up + silu_glu) take the register-dequantising prefill kernels from 256 rows; below, the
    /// gemv / mma / gemm kernels as before and no GLU kernel past one row (the model runs gate, up and silu_glu).
    #[test]
    fn nvfp4_selection_by_batch_size() {
        let pick = |op: Op, m: usize, n: usize, k: usize| {
            let s = ShapeInfo {
                m,
                n,
                k,
                act: Some(DType::Bf16),
                weight: Some(DType::U8),
                out: Some(DType::Bf16),
                ..Default::default()
            };
            let mut best: Option<KernelImpl> = None;
            for k in all().into_iter().filter(|k| k.op == op && k.accepts(&s)) {
                if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.to_string())
        };
        assert_eq!(crate::quantmoe::NVFP4_REG_MIN_TOKENS, 256);
        let off = std::env::var("INTARSIA_CUDA_NVFP4_REG").is_ok_and(|v| v == "0");
        for (n, k) in [(512, 2048), (2048, 512)] {
            assert_eq!(pick(Op::MatmulNvfp4, 1, n, k).as_deref(), Some("matmul_nvfp4_gemv"));
            assert_eq!(pick(Op::MatmulNvfp4, 32, n, k).as_deref(), Some("matmul_nvfp4_mma"));
            assert_eq!(pick(Op::MatmulNvfp4, 255, n, k).as_deref(), Some("matmul_nvfp4_gemm"));
            for m in [256, 2048, 4096] {
                let want = if off { "matmul_nvfp4_gemm" } else { "matmul_nvfp4_gemm_reg" };
                assert_eq!(pick(Op::MatmulNvfp4, m, n, k).as_deref(), Some(want), "M={m} N={n} K={k}");
            }
        }
        assert_eq!(pick(Op::MatmulNvfp4, 2048, 520, 2048).as_deref(), Some("matmul_nvfp4_gemm"), "N % 128 != 0");
        assert_eq!(pick(Op::MatmulNvfp4Glu, 1, 512, 2048).as_deref(), Some("matmul_nvfp4_glu_gemv"));
        for m in [2, 5, 33, 255] {
            assert_eq!(pick(Op::MatmulNvfp4Glu, m, 512, 2048), None, "M={m}");
        }
        for m in [256, 2048] {
            let want = if off { None } else { Some("matmul_nvfp4_glu_gemm") };
            assert_eq!(pick(Op::MatmulNvfp4Glu, m, 512, 2048).as_deref(), want, "M={m}");
        }
    }

    /// The attention kernel the registry picks for the 35B (16 / 2 / 256) and 8B (32 / 8 / 128) heads by cache dtype: a bf16
    /// cache keeps the split-KV (decode, <= 16 rows) and flash (> 16 rows) kernels, an fp8_e4m3 cache gets their f8kv twins;
    /// no f8kv kernel can serve a bf16 cache (and the reverse).
    #[test]
    fn attention_selection_by_kv_dtype() {
        let pick = |op: Op, m: usize, n: usize, nkv: usize, d: usize, kv: DType| {
            let s = ShapeInfo {
                m,
                n_heads: n,
                n_kv_heads: nkv,
                head_dim: d,
                k: d,
                kv_len: 4096,
                act: Some(DType::Bf16),
                weight: Some(kv),
                ..Default::default()
            };
            let mut best: Option<KernelImpl> = None;
            for k in all().into_iter().filter(|k| k.op == op && k.accepts(&s)) {
                if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.to_string())
        };
        for (n, nkv, d) in [(16, 2, 256), (32, 8, 128)] {
            let bf = DType::Bf16;
            let f8 = DType::Fp8E4m3;
            assert_eq!(pick(Op::AttnDecode, 1, n, nkv, d, bf).as_deref(), Some("attn_decode_bf16_splitkv"));
            assert_eq!(pick(Op::AttnDecode, 1, n, nkv, d, f8).as_deref(), Some("attn_decode_f8kv_splitkv"));
            for m in [1, 5, 16] {
                assert_eq!(pick(Op::AttnPrefill, m, n, nkv, d, bf).as_deref(), Some("attn_prefill_bf16_splitkv"), "T={m}");
                assert_eq!(pick(Op::AttnPrefill, m, n, nkv, d, f8).as_deref(), Some("attn_prefill_f8kv_splitkv"), "T={m}");
            }
            for m in [17, 256, 2048] {
                assert_eq!(pick(Op::AttnPrefill, m, n, nkv, d, bf).as_deref(), Some("attn_prefill_bf16_flash"), "T={m}");
                assert_eq!(pick(Op::AttnPrefill, m, n, nkv, d, f8).as_deref(), Some("attn_prefill_f8kv_flash"), "T={m}");
            }
        }
        for name in ["attn_prefill_f8kv_flash", "attn_prefill_f8kv_splitkv", "attn_decode_f8kv_splitkv"] {
            let k = all().into_iter().find(|k| k.name == name).unwrap();
            let s = ShapeInfo { m: 1, n_heads: 16, n_kv_heads: 2, head_dim: 256, k: 256, act: Some(DType::Bf16), weight: Some(DType::Bf16), ..Default::default() };
            assert!(!k.accepts(&s), "{name} must not take a bf16 cache");
        }
    }

    #[test]
    fn names_unique_and_ops_covered() {
        let table = full_table();
        let names: HashSet<_> = table.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), table.len());
        // Sample is not implemented on the device (the engine samples on the host from f32 logits); the six fused decode
        // ops are registered (quantmoe.rs) and the model still falls back to the unfused ops for an exact call without one.
        // `QkNormRopeKv` is a fifth optional fused op that only the Metal backend registers. The phase-1 folds
        // (`RmsNormAddMatmulFp8Pair`, `SigmoidMulMatmulFp8`, `SharedDownCombine`) are HIP-only so far: CUDA keeps the unfused ops.
        // So are the phase-2 folds (`GateSplitQkNormRopeKv`, `GdnConvDelta`) and P2' (`MoeBlock`), and the MXFP4 k/v-style
        // pair (`MatmulMxfp4Pair`, HIP-only so far: the model keeps two `MatmulMxfp4` launches here).
        const NOT_ON_CUDA: [Op; 9] = [
            Op::Sample,
            Op::QkNormRopeKv,
            Op::RmsNormAddMatmulFp8Pair,
            Op::SigmoidMulMatmulFp8,
            Op::SharedDownCombine,
            Op::GateSplitQkNormRopeKv,
            Op::GdnConvDelta,
            Op::MoeBlock,
            Op::MatmulMxfp4Pair,
        ];
        for op in Op::ALL {
            if !NOT_ON_CUDA.contains(&op) {
                assert!(table.iter().any(|(_, o)| *o == op), "no cuda kernel for {op}");
            }
        }
        assert_eq!(all().len(), table.len());
    }

    #[test]
    fn every_launcher_is_declared_in_the_abi_header() {
        let hdr = [
            include_str!("../../../kernels/abi/ik_abi.h"),
            include_str!("../../../kernels/cuda/ik_cuda_abi.h"),
            include_str!("../../../kernels/cuda/ik_cuda_abi_g4.h"),
        ]
        .concat();
        for (name, _) in full_table() {
            assert!(hdr.contains(&format!("int ik_{name}(")), "ik_{name} missing from the ABI headers");
        }
    }

    #[test]
    fn op_prefix_matches_launcher_name() {
        for (name, op) in full_table() {
            assert!(name.starts_with(op.as_str()), "{name} is not named after op {op}");
        }
    }

    const PHASE1_PINS: &[&str] = &[
        "35b attn input norm: rmsnorm_add_bf16",
        "35b k/v pair: matmul_fp8_pair_gemv",
        "35b k proj: matmul_fp8_gemv",
        "35b output gate: sigmoid_mul_bf16",
        "35b o_proj: matmul_fp8_gemv",
        "35b shared down: matmul_nvfp4_gemv",
        "35b combine: moe_combine_bf16",
        "35b fold P5: -",
        "35b fold P4b: -",
        "35b fold P2a: -",
        "8b attn input norm: rmsnorm_add_bf16",
        "8b k/v pair: matmul_pair_bf16_gemv",
    ];

    /// Core-op fusion phase 1 (docs/results/2026-10-19-core-op-fusion-p1.md): this backend registers none of the three new
    /// ops, so `Model::try_fused` returns false for each and the model runs the unfused chain it ran before (the selection of
    /// those ops at the Qwen3.6-35B-A3B and Qwen3-8B decode shapes is pinned below: the names origin/main's table resolves).
    #[test]
    fn phase1_folds_are_absent_and_the_unfused_selection_is_pinned() {
        use intarsia_core::ShapeInfo;
        let ks = all();
        for op in [Op::RmsNormAddMatmulFp8Pair, Op::SigmoidMulMatmulFp8, Op::SharedDownCombine] {
            assert!(!ks.iter().any(|k| k.op == op), "{op} is registered: the model would take the fused path here");
        }
        let pick = |op: Op, s: &ShapeInfo| -> Option<String> {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.clone())
        };
        let (bf, f8, u8_) = (Some(DType::Bf16), Some(DType::Fp8E4m3), Some(DType::U8));
        let sh = |m: usize, n: usize, k: usize, w: Option<DType>| ShapeInfo { m, n, k, act: bf, out: bf, weight: w, ..Default::default() };
        // (what, op, shape): the ops each fold replaces, at the 35B decode shapes (hidden 2048, k / v 2 x 256, o_proj 2048 x
        // 4096, shared down 2048 x 512), and the 8B's input norm (hidden 4096, bf16 weights: no fold applies to the 8B)
        let cases: Vec<(&str, Op, ShapeInfo)> = vec![
            ("35b attn input norm", Op::RmsNormAdd, sh(1, 0, 2048, None)),
            ("35b k/v pair", Op::MatmulFp8Pair, sh(1, 512, 2048, f8)),
            ("35b k proj", Op::MatmulFp8, sh(1, 512, 2048, f8)),
            ("35b output gate", Op::SigmoidMul, sh(1, 0, 4096, None)),
            ("35b o_proj", Op::MatmulFp8, sh(1, 2048, 4096, f8)),
            ("35b shared down", Op::MatmulNvfp4, sh(1, 2048, 512, u8_)),
            ("35b combine", Op::MoeCombine, sh(1, 0, 2048, None)),
            ("35b fold P5", Op::RmsNormAddMatmulFp8Pair, sh(1, 512, 2048, f8)),
            ("35b fold P4b", Op::SigmoidMulMatmulFp8, sh(1, 2048, 4096, f8)),
            ("35b fold P2a", Op::SharedDownCombine, sh(1, 2048, 512, u8_)),
            ("8b attn input norm", Op::RmsNormAdd, sh(1, 0, 4096, None)),
            ("8b k/v pair", Op::MatmulPair, sh(1, 1024, 4096, bf)),
        ];
        let got: Vec<String> = cases.iter().map(|(w, op, s)| format!("{w}: {}", pick(*op, s).unwrap_or_else(|| "-".into()))).collect();
        eprintln!("PINS\n{}", got.join("\n"));
        assert_eq!(got, PHASE1_PINS, "the selection of the unfused ops moved");
    }

    #[rustfmt::skip]
    const MAIN_KERNEL_LIST: &[&str] = &[
        "add_bf16:add",
        "argmax_bf16:argmax",
        "argmax_f32:argmax",
        "attn_decode_bf16:attn_decode",
        "attn_decode_bf16_splitkv:attn_decode",
        "attn_decode_f8kv_splitkv:attn_decode",
        "attn_prefill_bf16:attn_prefill",
        "attn_prefill_bf16_flash:attn_prefill",
        "attn_prefill_bf16_splitkv:attn_prefill",
        "attn_prefill_f8kv_flash:attn_prefill",
        "attn_prefill_f8kv_splitkv:attn_prefill",
        "embedding_bf16:embedding",
        "gate_split_bf16:gate_split",
        "gated_rmsnorm_bf16:gated_rmsnorm",
        "gated_rmsnorm_matmul_fp8_gemv:gated_rmsnorm_matmul_fp8",
        "gdn_conv_bf16:gdn_conv",
        "gdn_delta_rule_bf16:gdn_delta_rule",
        "gdn_delta_rule_bf16_lanes:gdn_delta_rule",
        "gdn_delta_rule_bf16_split:gdn_delta_rule",
        "kv_append_bf16:kv_append",
        "logits_bf16:logits",
        "logits_bf16_gemv:logits",
        "matmul_bf16:matmul",
        "matmul_bf16_gemv:matmul",
        "matmul_exl3_gemm:matmul_exl3",
        "matmul_exl3_gemv:matmul_exl3",
        "matmul_fp8_gemm:matmul_fp8",
        "matmul_fp8_gemm_cublas:matmul_fp8",
        "matmul_fp8_gemm_f8:matmul_fp8",
        "matmul_fp8_gemm_lt:matmul_fp8",
        "matmul_fp8_gemm_reg:matmul_fp8",
        "matmul_fp8_gemv:matmul_fp8",
        "matmul_fp8_mma:matmul_fp8",
        "matmul_fp8_pair_gemm_reg:matmul_fp8_pair",
        "matmul_fp8_pair_gemv:matmul_fp8_pair",
        "matmul_mxfp4_gemv:matmul_mxfp4", // M1 drill (2026-10-21): OCP MXFP4, a new op, no change to existing candidates
        "matmul_nvfp4_gemm:matmul_nvfp4",
        "matmul_nvfp4_gemm_reg:matmul_nvfp4",
        "matmul_nvfp4_gemv:matmul_nvfp4",
        "matmul_nvfp4_glu_gemm:matmul_nvfp4_glu",
        "matmul_nvfp4_glu_gemv:matmul_nvfp4_glu",
        "matmul_nvfp4_mma:matmul_nvfp4",
        "matmul_pair_bf16_gemv:matmul_pair",
        "moe_combine_bf16:moe_combine",
        "moe_experts_bf16:moe_experts",
        "moe_experts_decode_bf16:moe_experts",
        "moe_experts_decode_v2_bf16:moe_experts",
        "moe_experts_mma_bf16:moe_experts",
        "moe_experts_nvfp4_mma_bf16:moe_experts",
        "moe_experts_w4a16_bf16:moe_experts",
        "moe_topk_bf16:moe_topk",
        "qk_norm_bf16:qk_norm",
        "rmsnorm_add_bf16:rmsnorm_add",
        "rmsnorm_add_matmul_pair_bf16_gemv:rmsnorm_add_matmul_pair",
        "rmsnorm_bf16:rmsnorm",
        "rope_bf16:rope",
        "sigmoid_mul_bf16:sigmoid_mul",
        "silu_glu_bf16:silu_glu",
    ];

    /// Core-op fusion phase 1 changes no kernel table here: the full registered list (name and op, sorted) is pinned to
    /// origin/main's (`4962d8fa`), so the plan has the same candidates as before. Round 5 adds the three fp8-KV attention
    /// kernels (`attn_*_f8kv_*`), selectable only for an fp8_e4m3 cache (docs/results/2026-10-19-cuda-round5.md).
    #[test]
    fn kernel_list_is_unchanged_by_core_op_fusion_p1() {
        let mut got: Vec<String> = all().iter().map(|k| format!("{}:{}", k.name, k.op)).collect();
        got.sort();
        if MAIN_KERNEL_LIST.is_empty() {
            panic!("LIST {}", got.join(" "));
        }
        // the g4 (Gemma 4) kernels are all `.ext()` (see `g4_kernels_are_ext_and_never_picked_by_plain_calls`); the rest is main's list
        let (g4, main): (Vec<String>, Vec<String>) = got.into_iter().partition(|n| G4_KERNELS.contains(&n.as_str()));
        assert_eq!(main, MAIN_KERNEL_LIST);
        assert_eq!(g4, G4_KERNELS);
    }

    /// The kernels g4 (Gemma 4) added, name and op, sorted.
    const G4_KERNELS: &[&str] = &[
        "add_scaled_bf16:add",
        "attn_decode_bf16_ext:attn_decode",
        "attn_decode_bf16_splitkv_ext:attn_decode",
        "attn_prefill_bf16_ext:attn_prefill",
        "attn_prefill_bf16_splitkv_ext:attn_prefill",
        "embedding_scaled_bf16:embedding",
        "gelu_glu_bf16:gelu_glu",
        "moe_experts_decode_gelu_bf16:moe_experts",
        "moe_experts_decode_v2_gelu_bf16:moe_experts",
        "moe_experts_gelu_bf16:moe_experts",
        "moe_experts_mma_gelu_bf16:moe_experts",
        "moe_experts_w4a16_gelu_bf16:moe_experts",
        "rope_prop_bf16:rope",
    ];

    /// Every g4 kernel is registered `.ext()` (the registry would otherwise offer a kernel that ignores `window`, `rope_prop`,
    /// `gelu` or `scale`), and a call that uses no extended attribute selects the same kernel as without them: the
    /// attention at the Qwen3.6-35B-A3B shapes and the rope / add / embedding / glu ops.
    #[test]
    fn g4_kernels_are_ext_and_never_picked_by_plain_calls() {
        let ks = all();
        for name in G4_KERNELS {
            let name = name.split(':').next().unwrap();
            assert!(ks.iter().find(|k| k.name == name).expect("registered").ext, "{name} must be .ext()");
        }
        let bf = Some(DType::Bf16);
        let pick = |op: Op, s: &ShapeInfo| -> Option<String> {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.clone())
        };
        let att = |m: usize, n: usize, nkv: usize, d: usize, ext: bool| ShapeInfo {
            m, n_heads: n, n_kv_heads: nkv, head_dim: d, k: d, act: bf, out: bf, weight: bf, ext, ..Default::default()
        };
        // plain calls: the pre-g4 choice; Gemma's shapes (window / D = 512) get the ext kernels
        assert_eq!(pick(Op::AttnDecode, &att(1, 16, 2, 256, false)).as_deref(), Some("attn_decode_bf16_splitkv"));
        assert_eq!(pick(Op::AttnPrefill, &att(64, 16, 2, 256, false)).as_deref(), Some("attn_prefill_bf16_flash"));
        assert_eq!(pick(Op::AttnDecode, &att(1, 16, 2, 256, true)).as_deref(), Some("attn_decode_bf16_splitkv_ext"));
        assert_eq!(pick(Op::AttnDecode, &att(1, 16, 8, 256, true)).as_deref(), Some("attn_decode_bf16_splitkv_ext"));
        assert_eq!(pick(Op::AttnPrefill, &att(8, 16, 8, 256, true)).as_deref(), Some("attn_prefill_bf16_splitkv_ext"));
        assert_eq!(pick(Op::AttnPrefill, &att(64, 16, 8, 256, true)).as_deref(), Some("attn_prefill_bf16_ext"));
        assert_eq!(pick(Op::AttnDecode, &att(1, 16, 2, 512, true)).as_deref(), Some("attn_decode_bf16_ext"));
        assert_eq!(pick(Op::AttnDecode, &att(1, 16, 2, 512, false)).as_deref(), Some("attn_decode_bf16_ext"));
        assert_eq!(pick(Op::AttnPrefill, &att(20, 16, 2, 512, true)).as_deref(), Some("attn_prefill_bf16_ext"));
        let el = |ext: bool| ShapeInfo { m: 4, k: 64, head_dim: 64, act: bf, out: bf, ext, ..Default::default() };
        for (op, plain, extk) in [
            (Op::Rope, "rope_bf16", "rope_prop_bf16"),
            (Op::Add, "add_bf16", "add_scaled_bf16"),
            (Op::Embedding, "embedding_bf16", "embedding_scaled_bf16"),
        ] {
            assert_eq!(pick(op, &el(false)).as_deref(), Some(plain));
            assert_eq!(pick(op, &el(true)).as_deref(), Some(extk));
        }
        assert_eq!(pick(Op::GeluGlu, &el(false)).as_deref(), Some("gelu_glu_bf16"));
    }

    const PHASE2_PINS: &[&str] = &[
        "35b m=1 gate split: gate_split_bf16",
        "35b m=1 q norm: qk_norm_bf16",
        "35b m=1 k norm: qk_norm_bf16",
        "35b m=1 rope: rope_bf16",
        "35b m=1 kv append: kv_append_bf16",
        "35b m=1 prologue: -",
        "35b m=1 fold P4a: -",
        "35b m=1 gdn conv: gdn_conv_bf16",
        "35b m=1 gdn delta: gdn_delta_rule_bf16_split",
        "35b m=1 fold P3: -",
        "35b m=32 gate split: gate_split_bf16",
        "35b m=32 q norm: qk_norm_bf16",
        "35b m=32 k norm: qk_norm_bf16",
        "35b m=32 rope: rope_bf16",
        "35b m=32 kv append: kv_append_bf16",
        "35b m=32 prologue: -",
        "35b m=32 fold P4a: -",
        "35b m=32 gdn conv: gdn_conv_bf16",
        "35b m=32 gdn delta: gdn_delta_rule_bf16_split",
        "35b m=32 fold P3: -",
    ];

    /// Core-op fusion phase 2 (docs/results/2026-10-19-core-op-fusion-p2.md): this backend registers neither new op
    /// (`GateSplitQkNormRopeKv`, `GdnConvDelta`), so `Model::fused_ok` is false for both and the model runs the chains it
    /// ran before. The selection of the ops those chains consist of, at the Qwen3.6-35B-A3B decode shapes (gated attention:
    /// 16 query / 2 kv heads of 256, bf16 cache; GatedDeltaNet: 16 key / 32 value heads of 128, conv width 8192, 4 taps),
    /// and at a 32-token chunk (the longest the HIP fold takes), is pinned below: the names origin/main's table resolves.
    #[test]
    fn phase2_folds_are_absent_and_the_unfused_selection_is_pinned() {
        use intarsia_core::ShapeInfo;
        let ks = all();
        for op in [Op::GateSplitQkNormRopeKv, Op::GdnConvDelta] {
            assert!(!ks.iter().any(|k| k.op == op), "{op} is registered: the model would take the fused path here");
        }
        let pick = |op: Op, s: &ShapeInfo| -> Option<String> {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.clone())
        };
        let bf = Some(DType::Bf16);
        let heads = |m: usize, n: usize, nkv: usize, d: usize| ShapeInfo {
            m,
            n_heads: n,
            n_kv_heads: nkv,
            head_dim: d,
            k: d,
            act: bf,
            out: bf,
            weight: bf,
            ..Default::default()
        };
        let gdn = |m: usize, k: usize| ShapeInfo {
            m,
            n: 128,
            k,
            n_heads: 32,
            n_kv_heads: 16,
            head_dim: 128,
            act: bf,
            out: bf,
            weight: bf,
            ..Default::default()
        };
        let mut cases: Vec<(String, Op, ShapeInfo)> = Vec::new();
        for m in [1usize, 32] {
            cases.extend([
                (format!("35b m={m} gate split"), Op::GateSplit, heads(m, 16, 0, 256)),
                (format!("35b m={m} q norm"), Op::QkNorm, heads(m, 16, 0, 256)),
                (format!("35b m={m} k norm"), Op::QkNorm, heads(m, 2, 0, 256)),
                (format!("35b m={m} rope"), Op::Rope, heads(m, 16, 0, 256)),
                (format!("35b m={m} kv append"), Op::KvAppend, heads(m, 2, 0, 256)),
                (format!("35b m={m} prologue"), Op::QkNormRopeKv, heads(m, 16, 2, 256)),
                (format!("35b m={m} fold P4a"), Op::GateSplitQkNormRopeKv, heads(m, 16, 2, 256)),
                (format!("35b m={m} gdn conv"), Op::GdnConv, ShapeInfo { m, n: 8192, k: 4, act: bf, out: bf, weight: bf, ..Default::default() }),
                (format!("35b m={m} gdn delta"), Op::GdnDeltaRule, gdn(m, 128)),
                (format!("35b m={m} fold P3"), Op::GdnConvDelta, gdn(m, 4)),
            ]);
        }
        let got: Vec<String> = cases.iter().map(|(w, op, s)| format!("{w}: {}", pick(*op, s).unwrap_or_else(|| "-".into()))).collect();
        eprintln!("PINS\n{}", got.join("\n"));
        assert_eq!(got, PHASE2_PINS, "the selection of the unfused ops moved");
    }

    const MOE_BLOCK_PINS: &[&str] = &[
        "35b m=1 topk: moe_topk_bf16",
        "35b m=1 experts: moe_experts_decode_v2_bf16",
        "35b m=1 shared glu: matmul_nvfp4_glu_gemv",
        "35b m=1 shared gate: matmul_nvfp4_gemv",
        "35b m=1 silu glu: silu_glu_bf16",
        "35b m=1 shared down: matmul_nvfp4_gemv",
        "35b m=1 shared down combine: -",
        "35b m=1 combine: moe_combine_bf16",
        "35b m=1 fold P2': -",
        "35b m=2 topk: moe_topk_bf16",
        "35b m=2 experts: moe_experts_bf16",
        "35b m=2 shared glu: -",
        "35b m=2 shared gate: matmul_nvfp4_gemv",
        "35b m=2 silu glu: silu_glu_bf16",
        "35b m=2 shared down: matmul_nvfp4_gemv",
        "35b m=2 shared down combine: -",
        "35b m=2 combine: moe_combine_bf16",
        "35b m=2 fold P2': -",
    ];

    /// Core-op fusion P2' (docs/results/2026-10-19-p2prime-moeblock.md): this backend does not register `MoeBlock`, so
    /// `Model::fused_ok` is false for it and `MoeWeights::forward_folded` runs the chain it ran before. The selection of the
    /// ops that chain consists of, at the Qwen3.6-35B-A3B decode shapes (256 experts, top-8, hidden 2048, routed and shared
    /// intermediate 512; one row and the two-row batched step), is pinned below: the names origin/main's table resolves.
    #[test]
    fn moe_block_is_absent_and_the_unfused_selection_is_pinned() {
        use intarsia_core::ShapeInfo;
        let ks = all();
        assert!(!ks.iter().any(|k| k.op == Op::MoeBlock), "MoeBlock is registered: the model would take the fused path here");
        let pick = |op: Op, s: &ShapeInfo| -> Option<String> {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.clone())
        };
        let (bf, u8_) = (Some(DType::Bf16), Some(DType::U8));
        let sh = |m: usize, n: usize, k: usize, w: Option<DType>| ShapeInfo { m, n, k, act: bf, out: bf, weight: w, ..Default::default() };
        let moe = |m: usize, n: usize, k: usize, w: Option<DType>| ShapeInfo { experts: 256, top_k: 8, ..sh(m, n, k, w) };
        let mut cases: Vec<(String, Op, ShapeInfo)> = Vec::new();
        for m in [1usize, 2] {
            cases.extend([
                (format!("35b m={m} topk"), Op::MoeTopk, ShapeInfo { k: 256, ..moe(m, 256, 256, None) }),
                (format!("35b m={m} experts"), Op::MoeExperts, moe(m, 512, 2048, u8_)),
                (format!("35b m={m} shared glu"), Op::MatmulNvfp4Glu, sh(m, 512, 2048, u8_)),
                (format!("35b m={m} shared gate"), Op::MatmulNvfp4, sh(m, 512, 2048, u8_)),
                (format!("35b m={m} silu glu"), Op::SiluGlu, sh(m, 0, 512, None)),
                (format!("35b m={m} shared down"), Op::MatmulNvfp4, sh(m, 2048, 512, u8_)),
                (format!("35b m={m} shared down combine"), Op::SharedDownCombine, sh(m, 2048, 512, u8_)),
                (format!("35b m={m} combine"), Op::MoeCombine, sh(m, 0, 2048, None)),
                (format!("35b m={m} fold P2'"), Op::MoeBlock, moe(m, 512, 2048, u8_)),
            ]);
        }
        let got: Vec<String> = cases.iter().map(|(w, op, s)| format!("{w}: {}", pick(*op, s).unwrap_or_else(|| "-".into()))).collect();
        eprintln!("PINS\n{}", got.join("\n"));
        assert_eq!(got, MOE_BLOCK_PINS, "the selection of the unfused ops moved");
    }
}
