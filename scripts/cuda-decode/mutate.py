#!/usr/bin/env python3
"""Apply one named mutation to a copy of the tree (usage: mutate.py <tree> <name>). Used by `mutate.sh`: each mutation must
make `cargo test -p intarsia-backend-cuda --test fused_decode` FAIL, which shows the fused-vs-unfused tests catch a
shifted second segment, a wrong scale, a missing bf16 rounding, a swapped operand or a dropped tail."""
import sys

MUTATIONS = {
    # bf16 pair: the second matrix's blocks start one block late (its first rows are never written)
    "pair-block": ("kernels/cuda/matmul.cu", [(
        "else gemv_body<1, ik_bf16, true>(x, w1, o1, N1, K, (int64_t)blockIdx.x - nb0);",
        "else gemv_body<1, ik_bf16, true>(x, w1, o1, N1, K, (int64_t)blockIdx.x - nb0 + 1);")]),
    # bf16 pair: the first matrix's block count drops its partial block (rows N0 % 8 are never written)
    "pair-nb0": ("kernels/cuda/matmul.cu", [(
        "const int64_t nb0 = (N0 + kWarpsPerBlock - 1) / kWarpsPerBlock;\n    if ((int64_t)blockIdx.x < nb0) gemv_body",
        "const int64_t nb0 = N0 / kWarpsPerBlock;\n    if ((int64_t)blockIdx.x < nb0) gemv_body")]),
    # fp8 pair: the second segment stages and scales with the first segment's scales
    "fp8-pair-scale": ("kernels/cuda/fp8.cu", [(
        "else fp8_gemv_body<1, OutT>(x, w1, sc1, o1, N1, K, (int64_t)blockIdx.x - nb0);",
        "else fp8_gemv_body<1, OutT>(x, w1, sc0, o1, N1, K, (int64_t)blockIdx.x - nb0);")]),
    # fp8 pair: the second segment writes the first segment's output
    "fp8-pair-out": ("kernels/cuda/fp8.cu", [(
        "else fp8_gemv_body<1, OutT>(x, w1, sc1, o1, N1, K, (int64_t)blockIdx.x - nb0);",
        "else fp8_gemv_body<1, OutT>(x, w1, sc1, o0, N1, K, (int64_t)blockIdx.x - nb0);")]),
    # nvfp4 gate+up+silu: the up projection is scaled with the gate's global scale
    "glu-scale": ("kernels/cuda/nvfp4.cu", [(
        "ik_warp_sum(au) * gsu[0]", "ik_warp_sum(au) * gsg[0]")]),
    # nvfp4 gate+up+silu: the gate value is not rounded to bf16 before the silu (the unfused GEMV stores it as bf16)
    "glu-round": ("kernels/cuda/nvfp4.cu", [(
        "const float g = ik_b2f(ik_f2b(ik_warp_sum(ag) * gsg[0]));", "const float g = ik_warp_sum(ag) * gsg[0];")]),
    # nvfp4 gate+up+silu: the up row accumulates the gate block scale
    "glu-blockscale": ("kernels/cuda/nvfp4.cu", [(
        "au = fmaf(su, scu, au);", "au = fmaf(su, scg, au);")]),
    # gated rmsnorm: silu is applied to the normalised value instead of z
    "gated-order": ("kernels/cuda/norm.cu", [(
        "f[k] = ik_silu(zf[k]) * nrm[k];", "f[k] = ik_silu(nrm[k]) * zf[k];")]),
    # gated rmsnorm: the normalised value skips its bf16 rounding (vector path)
    "gated-round": ("kernels/cuda/norm.cu", [(
        "ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)",
        "for (int k = 0; k < 8; ++k) nrm[k] = f[k];")]),
    # gated rmsnorm: the normalised value skips its bf16 rounding (scalar path)
    "gated-round-scalar": ("kernels/cuda/norm.cu", [(
        "const float nrm = ik_b2f(ik_f2b(ik_b2f(xr[i]) * r * (ik_b2f(w[i]) + woff)));",
        "const float nrm = ik_b2f(xr[i]) * r * (ik_b2f(w[i]) + woff);")]),
}

tree, name = sys.argv[1], sys.argv[2]
path, pairs = MUTATIONS[name]
p = f"{tree}/{path}"
s = open(p).read()
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit(f"mutation {name}: pattern found {s.count(old)} times in {path}, expected exactly once")
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"applied {name} to {p}")
