//! Registered bitwise equivalences between fused decode ops and the chains of ops they replace.
//!
//! The solo decode step keeps its fusions (`MatmulFp8Pair`, `GatedRmsnormMatmulFp8`, ...: fewer launches, no extra weight
//! or activation traffic). A batched step of `B >= 2` rows has no fused kernel for them (they are one-row kernels), so it
//! runs the unfused chain on all rows at once. That is bitwise the solo step only if the fused kernel is bitwise its
//! chain. This table lists the fusions for which that is *proven*: each entry names the test(s) that compare the fused
//! kernel with the chain launch by launch, bit for bit (`tests/fused_decode.rs` of the backend). The plan check
//! (`plan::check_batch_ops`) treats a fused op of the solo step and its chain in the batched step as the same work only for
//! an entry here, and `fusions_name_existing_tests` pins that every named test exists.
//!
//! Policy (docs/decisions/LOG.md, 2026-10-23 06:41): the solo step is never changed by batching, and the fused path is never
//! picked by how many sequences happen to be active; a fusion without an entry here makes the batched plan fail to load
//! rather than run unproven arithmetic.

use crate::ops::Op;

/// A fused op and the chain it replaces, bitwise, on one backend.
#[derive(Clone, Copy, Debug)]
pub struct Fusion {
    /// Backend whose fused kernel the tests exercise.
    pub backend: &'static str,
    pub fused: Op,
    /// The ops the fused kernel replaces, in order (one level: a member may itself be a fusion with its own entry).
    pub chain: &'static [Op],
    /// `path/relative/to/the/workspace/root.rs::test_fn` of the tests that prove it.
    pub tests: &'static [&'static str],
}

macro_rules! cuda_test {
    ($f:literal) => {
        concat!("crates/intarsia-backend-cuda/tests/fused_decode.rs::", $f)
    };
}

macro_rules! cuda_rows_test {
    ($f:literal) => {
        concat!("crates/intarsia-backend-cuda/tests/rows_exact.rs::", $f)
    };
}

macro_rules! hip_test {
    ($f:literal) => {
        concat!("crates/intarsia-backend-hip/tests/fused_decode.rs::", $f)
    };
}

/// Every proven fusion. CUDA's phase-1 fusions of the 35B and Qwen3-8B decode steps
/// (docs/results/2026-10-14-cuda-decode-fusion.md), and the six HIP decode fusions of the same names (the HIP backend's
/// `tests/fused_decode.rs` ports the CUDA tests, bitwise against the same unfused chains); other backends have none registered,
/// so their batched plans keep the stricter rule (the solo step's op multiset). Also here: HIP's phase-1 folds `RmsNormAddMatmulFp8Pair`, `SigmoidMulMatmulFp8`, `SharedDownCombine` and `MoeBlock`. They
/// are kernels of few rows (1, 1, 1..=8 and T * top_k <= 16); because each has an entry here, `Model::fused_ok` does not take
/// them away from the solo step when batching is on (without an entry it must: the batched step then could not run the same
/// work).
pub const FUSIONS: &[Fusion] = &[
    Fusion {
        backend: "cuda",
        fused: Op::MatmulPair,
        chain: &[Op::Matmul, Op::Matmul],
        tests: &[cuda_test!("matmul_pair_is_bitwise_two_gemvs")],
    },
    Fusion {
        backend: "cuda",
        fused: Op::MatmulFp8Pair,
        chain: &[Op::MatmulFp8, Op::MatmulFp8],
        tests: &[cuda_test!("matmul_fp8_pair_is_bitwise_two_gemvs")],
    },
    // CUDA's row-exact attention q + k + v (lane qkv16, rows_exact only, 12..=16 rows): the batched step's one launch for what the
    // solo step runs as the q projection and the k + v pair; the test compares every M with the one-row GEMV on each matrix
    Fusion {
        backend: "cuda",
        fused: Op::MatmulFp8Triple,
        chain: &[Op::MatmulFp8, Op::MatmulFp8, Op::MatmulFp8],
        tests: &[cuda_rows_test!("fp8_triple_rows_are_bitwise_the_one_row_gemvs")],
    },
    Fusion {
        backend: "cuda",
        fused: Op::MatmulNvfp4Glu,
        chain: &[Op::MatmulNvfp4, Op::MatmulNvfp4, Op::SiluGlu],
        tests: &[cuda_test!("matmul_nvfp4_glu_is_bitwise_gate_up_silu")],
    },
    Fusion {
        backend: "cuda",
        fused: Op::GatedRmsnorm,
        chain: &[Op::RmsNorm, Op::SiluGlu],
        tests: &[cuda_test!("gated_rmsnorm_is_bitwise_rmsnorm_then_silu_glu")],
    },
    Fusion {
        backend: "cuda",
        fused: Op::GatedRmsnormMatmulFp8,
        chain: &[Op::GatedRmsnorm, Op::MatmulFp8],
        tests: &[
            cuda_test!("gated_rmsnorm_matmul_fp8_is_bitwise_norm_then_gemv"),
            cuda_test!("gated_rmsnorm_matmul_fp8_many_draws_bitwise"),
        ],
    },
    // the test compares with `rmsnorm_add_bf16` then the pair kernel, which `matmul_pair_is_bitwise_two_gemvs` proves two GEMVs
    Fusion {
        backend: "cuda",
        fused: Op::RmsNormAddMatmulPair,
        chain: &[Op::RmsNormAdd, Op::MatmulPair],
        tests: &[
            cuda_test!("rmsnorm_add_matmul_pair_is_bitwise_norm_then_pair"),
            cuda_test!("rmsnorm_add_matmul_pair_many_rows_bitwise"),
        ],
    },
    // ---- HIP (docs/results/2026-10-23-hb-hip-hybrid-batching.md) ----
    Fusion {
        backend: "hip",
        fused: Op::MatmulPair,
        chain: &[Op::Matmul, Op::Matmul],
        tests: &[hip_test!("matmul_pair_is_bitwise_two_gemvs")],
    },
    Fusion {
        backend: "hip",
        fused: Op::MatmulFp8Pair,
        chain: &[Op::MatmulFp8, Op::MatmulFp8],
        tests: &[hip_test!("matmul_fp8_pair_is_bitwise_two_gemvs")],
    },
    Fusion {
        backend: "hip",
        fused: Op::MatmulNvfp4Glu,
        chain: &[Op::MatmulNvfp4, Op::MatmulNvfp4, Op::SiluGlu],
        tests: &[hip_test!("matmul_nvfp4_glu_is_bitwise_gate_up_silu")],
    },
    Fusion {
        backend: "hip",
        fused: Op::GatedRmsnorm,
        chain: &[Op::RmsNorm, Op::SiluGlu],
        tests: &[hip_test!("gated_rmsnorm_is_bitwise_rmsnorm_then_silu_glu")],
    },
    Fusion {
        backend: "hip",
        fused: Op::GatedRmsnormMatmulFp8,
        chain: &[Op::GatedRmsnorm, Op::MatmulFp8],
        tests: &[
            hip_test!("gated_rmsnorm_matmul_fp8_is_bitwise_norm_then_gemv"),
            hip_test!("gated_rmsnorm_matmul_fp8_many_draws_bitwise"),
        ],
    },
    Fusion {
        backend: "hip",
        fused: Op::RmsNormAddMatmulPair,
        chain: &[Op::RmsNormAdd, Op::MatmulPair],
        tests: &[
            hip_test!("rmsnorm_add_matmul_pair_is_bitwise_norm_then_pair"),
            hip_test!("rmsnorm_add_matmul_pair_many_rows_bitwise"),
        ],
    },
    Fusion {
        backend: "hip",
        fused: Op::RmsNormAddMatmulFp8Pair,
        chain: &[Op::RmsNormAdd, Op::MatmulFp8Pair],
        tests: &["crates/intarsia-backend-hip/tests/fusion_p1.rs::rmsnorm_add_matmul_fp8_pair_is_bitwise_norm_then_pair"],
    },
    Fusion {
        backend: "hip",
        fused: Op::SigmoidMulMatmulFp8,
        chain: &[Op::SigmoidMul, Op::MatmulFp8],
        tests: &["crates/intarsia-backend-hip/tests/fusion_p1.rs::sigmoid_mul_matmul_fp8_is_bitwise_gate_then_gemv"],
    },
    Fusion {
        backend: "hip",
        fused: Op::SharedDownCombine,
        chain: &[Op::MatmulNvfp4, Op::MoeCombine],
        tests: &["crates/intarsia-backend-hip/tests/fusion_p1.rs::shared_down_combine_is_bitwise_down_then_combine"],
    },
    Fusion {
        backend: "hip",
        fused: Op::MoeBlock,
        chain: &[Op::MoeTopk, Op::MoeExperts, Op::MatmulNvfp4Glu, Op::SharedDownCombine],
        tests: &["crates/intarsia-backend-hip/tests/moe_block.rs::moe_block_is_bitwise_the_chain"],
    },
    // HIP's attention prologue and GatedDeltaNet folds (core-op fusion phases 2 and 3; docs/results/2026-10-19-core-op-fusion-p2.md).
    // The solo step keeps them; a batched step with row tables (`OpCall::rows`, lane hr3,
    // docs/results/2026-10-24-hr3-hip-batching-r3.md) runs their chains as one launch per op for all rows.
    Fusion {
        backend: "hip",
        fused: Op::GateSplitQkNormRopeKv,
        chain: &[Op::GateSplit, Op::QkNormRopeKv],
        tests: &["crates/intarsia-backend-hip/tests/fusion_p2.rs::gate_split_qk_norm_rope_kv_is_bitwise_split_then_prologue"],
    },
    Fusion {
        backend: "hip",
        fused: Op::QkNormRopeKv,
        chain: &[Op::QkNorm, Op::Rope, Op::QkNorm, Op::Rope, Op::KvAppend, Op::KvAppend],
        tests: &[
            "crates/intarsia-backend-hip/tests/qk_rope_kv.rs::qk_norm_rope_kv_is_bitwise_the_unfused_chain",
            "crates/intarsia-backend-hip/tests/qk_rope_kv.rs::qk_norm_rope_kv_many_draws_bitwise",
        ],
    },
    Fusion {
        backend: "hip",
        fused: Op::GdnConvDelta,
        chain: &[Op::GdnConv, Op::GdnDeltaRule],
        tests: &[
            "crates/intarsia-backend-hip/tests/fusion_p2.rs::gdn_conv_delta_is_bitwise_conv_then_delta_rule",
            "crates/intarsia-backend-hip/tests/fusion_p2.rs::gdn_conv_delta_many_draws_bitwise",
        ],
    },
];

/// The proven fusion of `fused` on `backend`, if any.
pub fn fusion(backend: &str, fused: Op) -> Option<&'static Fusion> {
    FUSIONS.iter().find(|f| f.backend == backend && f.fused == fused)
}

/// `ops` with every fused op that has a proven fusion on `backend` replaced by its chain, recursively: the unfused work
/// the list stands for. Two op lists that expand to the same multiset do the same arithmetic.
pub fn expand(backend: &str, ops: &[Op]) -> Vec<Op> {
    fn push(backend: &str, op: Op, out: &mut Vec<Op>, depth: usize) {
        match fusion(backend, op) {
            Some(f) if depth < 8 => f.chain.iter().for_each(|&m| push(backend, m, out, depth + 1)),
            _ => out.push(op),
        }
    }
    let mut out = Vec::with_capacity(ops.len());
    for &op in ops {
        push(backend, op, &mut out, 0);
    }
    out
}

/// Every op `ops` stands for at any level of expansion: the ops themselves and the members of their proven chains,
/// members of members included (`GatedRmsnormMatmulFp8` reaches `GatedRmsnorm`, `RmsNorm`, `SiluGlu` and `MatmulFp8`).
pub fn reachable(backend: &str, ops: &[Op]) -> std::collections::BTreeSet<Op> {
    let mut seen = std::collections::BTreeSet::new();
    let mut todo: Vec<Op> = ops.to_vec();
    while let Some(op) = todo.pop() {
        if seen.insert(op) {
            if let Some(f) = fusion(backend, op) {
                todo.extend(f.chain.iter().copied());
            }
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every test an entry names exists as a `fn` in the named file (a renamed or deleted equivalence test must fail here,
    /// not silently leave a fusion "proven").
    #[test]
    fn fusions_name_existing_tests() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for f in FUSIONS {
            assert!(!f.tests.is_empty(), "{:?} on {} names no test", f.fused, f.backend);
            for t in f.tests {
                let (file, name) = t.split_once("::").expect("path::test");
                let src = std::fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
                assert!(
                    src.contains(&format!("fn {name}(")),
                    "{:?}: test {name} is not in {file} (the equivalence is no longer proven)",
                    f.fused
                );
            }
            assert!(file_is_the_backends(f), "{:?} cites a test file of another backend", f.fused);
        }
    }

    fn file_is_the_backends(f: &Fusion) -> bool {
        f.tests.iter().all(|t| t.contains(&format!("intarsia-backend-{}/", f.backend)))
    }

    #[test]
    fn chains_expand_to_unfused_ops() {
        let e = expand("cuda", &[Op::GatedRmsnormMatmulFp8, Op::MatmulFp8Pair, Op::RmsNormAddMatmulPair]);
        assert_eq!(
            e,
            vec![
                Op::RmsNorm,
                Op::SiluGlu,
                Op::MatmulFp8,
                Op::MatmulFp8,
                Op::MatmulFp8,
                Op::RmsNormAdd,
                Op::Matmul,
                Op::Matmul
            ]
        );
        let r = reachable("cuda", &[Op::GatedRmsnormMatmulFp8]);
        assert!([Op::GatedRmsnormMatmulFp8, Op::GatedRmsnorm, Op::RmsNorm, Op::SiluGlu, Op::MatmulFp8]
            .iter()
            .all(|o| r.contains(o)));
        // a backend without entries expands nothing
        assert_eq!(expand("metal", &[Op::MatmulPair]), vec![Op::MatmulPair]);
        // no chain contains a fusion that loops back
        for f in FUSIONS {
            assert!(expand(f.backend, &[f.fused]).iter().all(|o| fusion(f.backend, *o).is_none()));
        }
    }
}
