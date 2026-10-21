#!/usr/bin/env python3
"""Mutations of the round-7 change (opt-in cuBLASLt e4m3 GEMM) for the mutation check (scripts/cuda-prefill4/mutcheck.sh):

  * the launcher kernels/cuda/fp8_gemm_lt.cu, killed by tests/quantmoe.rs (accuracy vs the bf16 GEMM and the f64
    reference in both activation modes, row invariance);
  * the gate (crates/intarsia-backend-cuda/src/kernels.rs) and the range (crates/intarsia-backend-cuda/src/quantmoe.rs),
    killed by the backend's lib tests (registry pins `fp8_gemm_lt_range`, `fp8_gemm_selection_by_batch_size`).

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list [file]           print the mutation names (only those of <file> if given)

A description starting with EQUIVALENT cannot change an output bit; PROBE marks a mutation whose effect depends on what
cuBLAS chooses (it is run to learn whether the pin matters, not as a kill target).
"""
import sys

LT = "kernels/cuda/fp8_gemm_lt.cu"
KR = "crates/intarsia-backend-cuda/src/kernels.rs"
QM = "crates/intarsia-backend-cuda/src/quantmoe.rs"

MUTATIONS = {
    "lt-ascale-in": (LT, "the weight (A) scale pointer reads the input scale (input_scale applied twice, weight_scale never)",
        [("    const void* sa = sc;      // weight_scale (A = weights)", "    const void* sa = sc + 1;  // weight_scale (A = weights)", 1)]),
    "lt-no-bscale": (LT, "the input (B) scale is never set (cuBLAS default 1: the input_scale factor is lost)",
        [("        cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof sb) != CUBLAS_STATUS_SUCCESS)",
          "        false)", 1)]),
    "lt-quant-mul": (LT, "the activations are quantised as x * input_scale instead of x / input_scale",
        [("ik_f_to_e4m3(__fdiv_rn(f[e], s_in))", "ik_f_to_e4m3(f[e] * s_in)", 1),
         ("ik_f_to_e4m3(__fdiv_rn(f[4 + e], s_in))", "ik_f_to_e4m3(f[4 + e] * s_in)", 1)]),
    "lt-no-weightonly": (LT, "no bf16 fallback for weight-only activations (input_scale <= 0: the e4m3 result stays)",
        [("    return ik_fp8_gemm_weight_only_internal(x->data, w->data, sc->data, out->data, f32_out ? 1 : 0, M, N, K, stream);",
          "    (void)f32_out;\n    return IK_OK;", 1)]),
    "lt-ld-swap": (LT, "the output layout is M x N (ld M) instead of N x M (ld N): rows and columns transposed",
        [("cublasLtMatrixLayoutInit(ld, dt, (uint64_t)N, (uint64_t)M, N)", "cublasLtMatrixLayoutInit(ld, dt, (uint64_t)M, (uint64_t)N, M)", 1)]),
    "PROBE-lt-plan-per-m": (LT, "PROBE: the algorithm is chosen at each call's own M (no pinning across M)",
        [("    IK_TRY(make_desc(ds, kRefM, N, K, f32_out, sc));", "    IK_TRY(make_desc(ds, g_cur_m, N, K, f32_out, sc));", 1),
         ("constexpr int kRefM = 2048;", "constexpr int kRefM = 2048;\nthread_local int64_t g_cur_m = 2048;", 1),
         ("    LtState& st = g_lt[dev];\n", "    LtState& st = g_lt[dev];\n    g_cur_m = M;\n", 1),
         ("    auto key = std::make_tuple(N, K, f32_out ? 1 : 0);", "    auto key = std::make_tuple(N * 1000003 + g_cur_m, K, f32_out ? 1 : 0);", 1)]),
    "lt-gate-on": (KR, "the cuBLASLt GEMM is on by default",
        [('std::env::var("INTARSIA_CUDA_FP8_GEMM_LT").is_ok_and(|v| v == "1"))\n}', 'std::env::var("INTARSIA_CUDA_FP8_GEMM_LT").map_or(true, |v| v != "0"))\n}', 1)]),
    "lt-min-gt": (QM, "the cuBLASLt GEMM from 6 tokens (> instead of >=): a 5-token tail chunk rounds differently",
        [(".when(|s| s.m >= F8_MIN_TOKENS && s.k % 128 == 0 && s.n % 16 == 0)", ".when(|s| s.m > F8_MIN_TOKENS && s.k % 128 == 0 && s.n % 16 == 0)", 1)]),
}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        f = sys.argv[2] if len(sys.argv) > 2 else None
        print(" ".join(m for m, v in MUTATIONS.items() if f is None or v[0].endswith(f)))
        return 0
    if len(sys.argv) != 3 or sys.argv[2] not in MUTATIONS:
        print(__doc__)
        return 2
    root, name = sys.argv[1], sys.argv[2]
    fname, desc, edits = MUTATIONS[name]
    path = f"{root}/{fname}"
    src = open(path).read()
    for find, repl, count in edits:
        if src.count(find) != count:
            print(f"mutation {name}: `{find[:60]}` occurs {src.count(find)} times, expected {count}")
            return 1
        src = src.replace(find, repl)
    open(path, "w").write(src)
    print(f"applied {name}: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
