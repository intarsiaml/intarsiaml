//! The kernel table: which launcher implements which op, for which shapes, and how closely it must
//! match the reference. Compiled on every host (the table is data); only `device.rs` needs ROCm.
//!
//! `name` is the launcher name without the `ik_` prefix (kernels/abi/ik_abi.h). This table is the dense BF16 set
//! (Qwen3 family plus the gated-attention helpers) and the GatedDeltaNet launchers; the quantised (fp8 / nvfp4)
//! and sparse-MoE launchers are in `quantmoe.rs`, and `all()` registers both.

use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

/// Architecture the .hip files were compiled for (`INTARSIA_HIP_ARCH`, default gfx1201).
pub const ARCH: &str = env!("IK_HIP_ARCH");

/// `hipcc --version` (first lines) of the toolchain that built the kernel library, `none` for the stub build. Part
/// of the backend fingerprint: a cargo target dir that outlives a ROCm image bump is rebuilt (build.rs tracks the
/// compiler), and the admission cache never mixes results of two compilers.
pub const COMPILER: &str = env!("IK_HIP_COMPILER");

/// Tuning / A-B environment knobs that change which kernel variant runs or its numerics, as `NAME=value` pairs of
/// the ones that are set (sorted). They are read inside the kernel library, so they belong in the fingerprint: a
/// cached ADMITTED result for the default variant says nothing about `INTARSIA_HIP_GEMV=4,8`.
pub const ENV_KNOBS: &[&str] = &[
    "INTARSIA_HIP_DISABLE",
    "INTARSIA_HIP_FP8_F8",
    "INTARSIA_HIP_FP8_PAIR",
    "INTARSIA_HIP_FP8_Q",
    "INTARSIA_HIP_FP8_ROWS",
    "INTARSIA_HIP_GDN_PD",
    "INTARSIA_HIP_GEMM_TILE",
    "INTARSIA_HIP_GEMV",
    "INTARSIA_HIP_MOE_PAIRS",
    "INTARSIA_HIP_MOE_PRE",
    "INTARSIA_HIP_MOE_WMMA",
    "INTARSIA_HIP_MOE_Z",
    "INTARSIA_HIP_XSTAGE",
];

pub fn env_knobs() -> String {
    ENV_KNOBS
        .iter()
        .filter_map(|n| std::env::var(n).ok().map(|v| format!("{n}={v}")))
        .collect::<Vec<_>>()
        .join(",")
}

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
    ("qk_norm_rope_kv_bf16", Op::QkNormRopeKv),
    ("attn_prefill_bf16", Op::AttnPrefill),
    ("attn_decode_bf16", Op::AttnDecode),
    ("attn_prefill_bf16_flash", Op::AttnPrefill),
    ("attn_prefill_bf16_splitkv", Op::AttnPrefill),
    ("attn_decode_bf16_splitkv", Op::AttnDecode),
    ("silu_glu_bf16", Op::SiluGlu),
    ("gdn_conv_bf16", Op::GdnConv),
    ("gdn_delta_rule_bf16", Op::GdnDeltaRule),
    ("gdn_delta_rule_bf16_v2", Op::GdnDeltaRule),
    ("gate_split_bf16", Op::GateSplit),
    ("sigmoid_mul_bf16", Op::SigmoidMul),
    ("argmax_f32", Op::Argmax),
    ("argmax_bf16", Op::Argmax),
];

/// Ops this backend does not implement (sampling stays on the cpu).
pub const UNIMPLEMENTED: &[Op] = &[Op::Sample];

/// bf16-rounded outputs: a one-ulp flip is allowed (same budget as the cpu backend).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
/// f32 outputs of bf16 inputs: only accumulation-order noise.
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

fn base(name: &str, op: Op) -> KernelImpl {
    KernelImpl::new(op, name, "hip").arch([ARCH])
}

/// Fewest tokens `gdn_delta_rule_bf16_v2` takes: every call, decode steps included (6.2 vs 9.2 us per decode launch).
pub const GDN_V2_MIN_TOKENS: usize = 1;

/// Most query rows the split-KV kernels take per launch (above this the flash prefill kernel wins).
const SPLITKV_MAX_ROWS: usize = 16;

/// Split-KV partitioning policy, mirroring `splitkv_count` in kernels/hip/attention_splitkv.hip (the launcher
/// derives its split count from the key count, the query rows and the kv heads, never from the data, so results
/// are run-to-run identical for a given call shape; this copy must be an
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
            // native WMMA GEMM: f32 accumulate in a fixed K order, run-to-run identical; any M. (A hipBLASLt route
            // would be Exactness::Fast; the dev image carries no hipblaslt, see docs/backends/hip.md.)
            "matmul_bf16" => k.acts(bf).weights(bf).exact(Exactness::Deterministic).tol(BF16_TOL).priority(0),
            "logits_bf16" => k.acts(bf).weights(bf).exact(Exactness::Deterministic).tol(F32_TOL).priority(0),
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
            // qk-norm + rope + kv-append in one launch, bitwise the three unfused launchers (tests/qk_rope_kv.rs), so the
            // budget is theirs; any row count (decode, verify, prefill chunks), bf16 or fp8 cache
            "qk_norm_rope_kv_bf16" => k
                .acts(bf)
                .when(|s| {
                    s.head_dim % 8 == 0
                        && (8..=256).contains(&s.head_dim)
                        && s.n_kv_heads > 0
                        && s.n_heads % s.n_kv_heads == 0
                        && s.n_heads / s.n_kv_heads <= 8
                        && matches!(s.weight, Some(DType::Bf16) | Some(DType::Fp8E4m3))
                })
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(10),
            "attn_prefill_bf16" | "attn_decode_bf16" => k
                .acts(bf)
                .weights(bf)
                .when(|s| s.head_dim > 0 && s.head_dim <= 256 && s.head_dim % 2 == 0)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL),
            // WMMA flash prefill: ranks above the SIMT fallback for chunks of > 16 rows
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
            // sequential recurrence, state rows in registers: fixed reduction order
            "gdn_delta_rule_bf16" => k
                .acts(bf)
                .weights(bf)
                .when(|s| matches!(s.head_dim, 32 | 64 | 128 | 256) && s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL),
            // the same recurrence without the rows kernel's stalls: bitwise gdn_delta_rule_bf16 (tests/gdn_v2.rs);
            // dk = 128 (the 35B), decode and prefill
            "gdn_delta_rule_bf16_v2" => k
                .acts(bf)
                .weights(bf)
                .when(|s| s.head_dim == 128 && s.m >= GDN_V2_MIN_TOKENS && s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0)
                .exact(Exactness::Deterministic)
                .tol(BF16_TOL)
                .priority(10),
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

    #[test]
    fn names_unique_and_ops_covered() {
        let table = full_table();
        let names: HashSet<_> = table.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), table.len());
        // every op is either implemented here or listed as not ported yet
        for op in Op::ALL {
            let have = table.iter().any(|(_, o)| *o == op);
            assert!(have != UNIMPLEMENTED.contains(&op), "{op}: implemented = {have}, listed unimplemented = {}", UNIMPLEMENTED.contains(&op));
        }
        assert_eq!(all().len(), table.len());
    }

    #[test]
    fn every_launcher_is_declared_in_the_abi_header() {
        // the shared header, plus the one of the launchers only this backend defines
        let hdr = [include_str!("../../../kernels/abi/ik_abi.h"), include_str!("../../../kernels/hip/ik_hip_abi.h")].concat();
        for (name, _) in full_table() {
            assert!(hdr.contains(&format!("int ik_{name}(")), "ik_{name} missing from kernels/abi/ik_abi.h");
        }
    }

    #[test]
    fn op_prefix_matches_launcher_name() {
        for (name, op) in full_table() {
            assert!(name.starts_with(op.as_str()), "{name} is not named after op {op}");
        }
    }

    /// The kernel `Registry::select` picks for `op` at shape `s` (highest priority among the registrations that accept it,
    /// ties in registration order), or None.
    fn pick(op: Op, s: &ShapeInfo) -> Option<String> {
        let ks = all();
        let mut best: Option<&KernelImpl> = None;
        for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
            if best.is_none_or(|b| k.priority > b.priority) {
                best = Some(k);
            }
        }
        best.map(|k| k.name.clone())
    }

    /// Round 4 selection (docs/results/2026-10-19-hip-round4.md): the fused attention prologue for the 35B's and the 8B's
    /// heads at every row count, and the stall-free delta rule for dk = 128 (other key widths keep the rows kernel).
    #[test]
    fn round4_selection_is_pinned() {
        let bf = Some(DType::Bf16);
        for (n, nkv, d) in [(16usize, 2usize, 256usize), (32, 8, 128)] {
            for m in [1usize, 2, 16, 312, 4096] {
                for cache in [DType::Bf16, DType::Fp8E4m3] {
                    let s = ShapeInfo { m, n_heads: n, n_kv_heads: nkv, head_dim: d, k: d, act: bf, out: bf, weight: Some(cache), ..Default::default() };
                    assert_eq!(pick(Op::QkNormRopeKv, &s).as_deref(), Some("qk_norm_rope_kv_bf16"), "m={m} n={n} d={d} {cache}");
                }
            }
        }
        // groups wider than 8 query heads per kv head, odd head widths: no fused prologue (the model runs the unfused ops)
        let wide = ShapeInfo { m: 1, n_heads: 32, n_kv_heads: 2, head_dim: 128, act: bf, out: bf, weight: bf, ..Default::default() };
        assert_eq!(pick(Op::QkNormRopeKv, &wide), None);
        let odd = ShapeInfo { m: 1, n_heads: 16, n_kv_heads: 2, head_dim: 260, act: bf, out: bf, weight: bf, ..Default::default() };
        assert_eq!(pick(Op::QkNormRopeKv, &odd), None);
        let gdn = |m: usize, dk: usize| ShapeInfo {
            m,
            n_heads: 32,
            n_kv_heads: 16,
            head_dim: dk,
            k: dk,
            n: 128,
            act: bf,
            out: bf,
            weight: bf,
            ..Default::default()
        };
        for m in [1usize, 2, 7, 312, 4096] {
            assert_eq!(pick(Op::GdnDeltaRule, &gdn(m, 128)).as_deref(), Some("gdn_delta_rule_bf16_v2"), "m={m}");
            assert_eq!(pick(Op::GdnDeltaRule, &gdn(m, 64)).as_deref(), Some("gdn_delta_rule_bf16"), "m={m} dk=64");
        }
    }

    #[test]
    fn arch_is_a_gfx12_target() {
        assert!(ARCH.starts_with("gfx12"), "the WMMA kernels need a gfx12 arch, got {ARCH}");
    }
}
