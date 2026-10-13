//! The kernel table: which launcher implements which op, for which shapes, and how closely it must
//! match the reference. Compiled on every host (the table is data); only `device.rs` needs CUDA.
//!
//! `name` is the launcher name without the `ik_` prefix (kernels/abi/ik_abi.h).

use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

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
    ("silu_glu_bf16", Op::SiluGlu),
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
            "argmax_f32" => k.acts([DType::F32]).exact(Exactness::Exact).tol(Tol::EXACT),
            "argmax_bf16" => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT),
            other => unreachable!("kernel {other} has no registration"),
        };
        v.push(k);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn names_unique_and_ops_covered() {
        let names: HashSet<_> = TABLE.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), TABLE.len());
        // Sample is not implemented on the device (the engine samples on the host from f32 logits)
        for op in Op::ALL {
            if op != Op::Sample {
                assert!(TABLE.iter().any(|(_, o)| *o == op), "no cuda kernel for {op}");
            }
        }
        assert_eq!(all().len(), TABLE.len());
    }

    #[test]
    fn every_launcher_is_declared_in_the_abi_header() {
        let hdr = include_str!("../../../kernels/abi/ik_abi.h");
        for (name, _) in TABLE {
            assert!(hdr.contains(&format!("int ik_{name}(")), "ik_{name} missing from kernels/abi/ik_abi.h");
        }
    }

    #[test]
    fn op_prefix_matches_launcher_name() {
        for (name, op) in TABLE {
            assert!(name.starts_with(op.as_str()), "{name} is not named after op {op}");
        }
    }
}
