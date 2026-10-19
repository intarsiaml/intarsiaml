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
    ("silu_glu_bf16", Op::SiluGlu),
    ("gate_split_bf16", Op::GateSplit),
    ("sigmoid_mul_bf16", Op::SigmoidMul),
    ("gdn_conv_bf16", Op::GdnConv),
    ("gdn_delta_rule_bf16", Op::GdnDeltaRule),
    ("gdn_delta_rule_bf16_lanes", Op::GdnDeltaRule),
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

/// Priority of an opt-in kernel: above everything it replaces when `fast_prefill()`, below everything otherwise.
pub fn opt_in_priority(on: i32) -> i32 {
    if fast_prefill() {
        on
    } else {
        -10
    }
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
            "argmax_f32" => k.acts([DType::F32]).exact(Exactness::Exact).tol(Tol::EXACT),
            "argmax_bf16" => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT),
            other => unreachable!("kernel {other} has no registration"),
        };
        v.push(k);
    }
    v.extend(crate::quantmoe::all());
    v
}

/// (launcher name, op) of every launcher: this file's table plus the quantised/MoE table.
pub fn full_table() -> Vec<(&'static str, Op)> {
    TABLE.iter().chain(crate::quantmoe::TABLE.iter()).copied().collect()
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
        assert_eq!(pick(1023), Some("moe_experts_mma_bf16".to_string()));
        assert_eq!(pick(256), Some("moe_experts_mma_bf16".to_string()));
    }

    /// The GatedDeltaNet recurrence kernel the registry picks at the 35B shape (dk = dv = 128, 16 key heads, 32 value heads)
    /// for decode (T = 1) and a prefill chunk (T = 2048): the one-warp-per-row kernel by default, the lanes kernel with
    /// `INTARSIA_CUDA_GDN_LANES=1` or `INTARSIA_CUDA_FAST_PREFILL=1` (the switch is read once per process, so the test pins
    /// whichever the environment of this run asks for).
    #[test]
    fn gdn_delta_rule_selection() {
        let pick = |m: usize| {
            let s = ShapeInfo {
                m,
                n: 128,
                k: 128,
                head_dim: 128,
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
        let lanes = std::env::var("INTARSIA_CUDA_GDN_LANES").is_ok_and(|v| v == "1")
            || std::env::var("INTARSIA_CUDA_FAST_PREFILL").is_ok_and(|v| v == "1");
        let want = if lanes { "gdn_delta_rule_bf16_lanes" } else { "gdn_delta_rule_bf16" };
        for m in [1, 2048] {
            assert_eq!(pick(m), Some(want.to_string()), "T={m}");
        }
    }

    /// The fp8 GEMM the registry picks for the 35B W8A8 projections (K 2048 / 4096): the gemv for 1..4 tokens, the mma
    /// kernel for 5..32, `matmul_fp8_gemm_reg` from `FP8_REG_MIN_TOKENS` (the 2048-token prefill chunk included), unless
    /// switched off (`INTARSIA_CUDA_FP8_GEMM_REG=0`: `matmul_fp8_gemm`) or overridden by `INTARSIA_CUDA_FAST_PREFILL=1`. The
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
        if std::env::var("INTARSIA_CUDA_FAST_PREFILL").is_ok() {
            return; // the default selection is what is pinned
        }
        let big = if std::env::var("INTARSIA_CUDA_FP8_GEMM_REG").is_ok_and(|v| v == "0") {
            "matmul_fp8_gemm"
        } else {
            "matmul_fp8_gemm_reg"
        };
        let min = crate::quantmoe::FP8_REG_MIN_TOKENS;
        for k in [2048, 4096] {
            assert_eq!(pick(1, k).as_deref(), Some("matmul_fp8_gemv"));
            assert_eq!(pick(4, k).as_deref(), Some("matmul_fp8_gemv"));
            assert_eq!(pick(5, k).as_deref(), Some("matmul_fp8_mma"));
            assert_eq!(pick(32, k).as_deref(), Some("matmul_fp8_mma"));
            if min > 33 {
                assert_eq!(pick(min - 1, k).as_deref(), Some("matmul_fp8_gemm"));
            }
            for m in [min, 2048, 4096] {
                assert_eq!(pick(m, k).as_deref(), Some(big), "M={m} K={k}");
            }
        }
    }

    #[test]
    fn names_unique_and_ops_covered() {
        let table = full_table();
        let names: HashSet<_> = table.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), table.len());
        // Sample is not implemented on the device (the engine samples on the host from f32 logits); the six fused decode
        // ops are registered (quantmoe.rs) and the model still falls back to the unfused ops for an exact call without one.
        // `QkNormRopeKv` is a fifth optional fused op that only the Metal backend registers.
        for op in Op::ALL {
            if op != Op::Sample && op != Op::QkNormRopeKv {
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
}
