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
        "const int64_t nb0 = (N0 + kWarpsPerBlock - 1) / kWarpsPerBlock;\n    if ((int64_t)blockIdx.x < nb0) gemv_body<1, ik_bf16, true>(",
        "const int64_t nb0 = N0 / kWarpsPerBlock;\n    if ((int64_t)blockIdx.x < nb0) gemv_body<1, ik_bf16, true>(")]),
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
    # --- rmsnorm_add folded into the bf16 pair GEMV (cuda-norm-gemv; tests/fused_decode.rs) ---
    # the designated block does not store the new residual stream
    "norm-skip-sum": ("kernels/cuda/matmul.cu", [(
        "        if (writer) reinterpret_cast<uint4*>(sum)[i] = p;\n", "")]),
    # the designated block does not store the normalised row (the MoE experts and the shared expert read it)
    "norm-skip-normed": ("kernels/cuda/matmul.cu", [(
        "        if (writer) reinterpret_cast<uint4*>(normed)[i] = p;\n", "")]),
    # the sum is squared unrounded (rmsnorm_add squares the bf16-rounded sum that the next layer carries)
    "norm-sum-round": ("kernels/cuda/matmul.cu", [(
        "        ik_unpack8(p, f);  // continue from the bf16-rounded sum\n", "")]),
    # the zero-centred norm's +1 on the weight is dropped
    "norm-woff": ("kernels/cuda/matmul.cu", [(
        "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);", "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * g[k];")]),
    # the eight squares of a thread's chunk are added in the opposite order
    "norm-order": ("kernels/cuda/matmul.cu", [(
        "        for (int k = 0; k < 8; ++k) ss += f[k] * f[k];\n    }\n    ss = ik_block_sum(ss, red);",
        "        for (int k = 7; k >= 0; --k) ss += f[k] * f[k];\n    }\n    ss = ik_block_sum(ss, red);")]),
    # the second matrix's blocks start one block late
    "norm-pair-block": ("kernels/cuda/matmul.cu", [(
        "else gemv_body<1, ik_bf16, true, true>(xrow, w1, o1, N1, K, (int64_t)blockIdx.x - nb0);",
        "else gemv_body<1, ik_bf16, true, true>(xrow, w1, o1, N1, K, (int64_t)blockIdx.x - nb0 + 1);")]),
    # the first matrix's block count drops its partial block
    "norm-nb0": ("kernels/cuda/matmul.cu", [(
        "const int64_t nb0 = (N0 + kWarpsPerBlock - 1) / kWarpsPerBlock;\n    if ((int64_t)blockIdx.x < nb0) gemv_body<1, ik_bf16, true, true>(",
        "const int64_t nb0 = N0 / kWarpsPerBlock;\n    if ((int64_t)blockIdx.x < nb0) gemv_body<1, ik_bf16, true, true>(")]),
    # no barrier between the smem row being written and being read by the other warps
    "norm-sync": ("kernels/cuda/matmul.cu", [(
        "    __syncthreads();  // the whole normalised row is in smem before any warp reads it\n", "")]),
    # the GEMV reads the row from global memory (x) instead of the staged normalised row
    "norm-read-x": ("kernels/cuda/matmul.cu", [(
        "const ik_bf16* xrow = reinterpret_cast<const ik_bf16*>(xs);", "const ik_bf16* xrow = x;")]),
    # --- gated rmsnorm folded into the fp8 out_proj staging (cuda-norm-gemv; tests/fused_decode.rs) ---
    # the normalised value skips its bf16 rounding before the silu product
    "gn-round": ("kernels/cuda/fp8.cu", [(
        "    ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)\n",
        "    for (int k = 0; k < 8; ++k) nrm[k] = f[k];\n")]),
    # the zero-centred weight offset is dropped
    "gn-woff": ("kernels/cuda/fp8.cu", [(
        "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);", "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * g[k];")]),
    # the sum of squares of a head is added in the opposite order of lanes
    "gn-order": ("kernels/cuda/fp8.cu", [(
        "for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);",
        "for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);")]),
    # the eight squares of a lane's chunk are added in the opposite order
    "gn-chunk-order": ("kernels/cuda/fp8.cu", [(
        "    for (int k = 0; k < 8; ++k) ss += f[k] * f[k];\n#pragma unroll\n    for (int o = 8;",
        "    for (int k = 7; k >= 0; --k) ss += f[k] * f[k];\n#pragma unroll\n    for (int o = 8;")]),
    # the reduction spans 32 lanes: two heads share one sum of squares
    "gn-group": ("kernels/cuda/fp8.cu", [(
        "for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);",
        "for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);")]),
    # the norm weight is indexed by the wrong chunk
    "gn-weight": ("kernels/cuda/fp8.cu", [(
        "reinterpret_cast<const uint4*>(nw) + (i8 & 15)", "reinterpret_cast<const uint4*>(nw) + (i8 & 7)")]),
    # silu is applied to the normalised value instead of z
    "gn-silu": ("kernels/cuda/fp8.cu", [(
        "f[k] = ik_silu(zf[k]) * nrm[k];\n    return ik_pack8(f);", "f[k] = ik_silu(nrm[k]) * zf[k];\n    return ik_pack8(f);")]),
    # W8A8: the staged value is not quantised
    "gn-actq": ("kernels/cuda/fp8.cu", [(
        "\n                for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));",
        "\n                for (int e = 0; e < 8; ++e) f[e] = f[e];")]),
    # the last chunk group of an odd row count is dropped
    "gn-tail": ("kernels/cuda/fp8.cu", [(
        "if (i8 < nch) reinterpret_cast<uint4*>(xs)[i8] = v;", "if (i8 < nch - 1) reinterpret_cast<uint4*>(xs)[i8] = v;")]),
    # --- moe_experts_decode_v2 (tests/quantmoe.rs moe_experts_decode_matches_old_kernel) ---
    # v2 down: the last expert slot is not added to the total. (Summing the slots in reverse order was tried first and
    # SURVIVED on the test data: the K bf16-valued terms sit within a few binades, so their fp32 sums are exact and any
    # order gives the same bits. v2 keeps the original order by construction; the tests do NOT guard summation order.)
    "down-order": ("kernels/cuda/moe_decode.cu", [(
        "for (int k = 0; k < K; ++k) total += term[k][threadIdx.x];",
        "for (int k = 0; k < K - 1; ++k) total += term[k][threadIdx.x];")]),
    # v2 down: the slot term skips the rounding of the down projection before the routing weight
    "down-round": ("kernels/cuda/moe_decode.cu", [(
        "term[k][i] = ok ? round_bf16(round_bf16(s * gscales[v * 3 + 2]) * wts[(int64_t)t * K + k]) : 0.f;",
        "term[k][i] = ok ? round_bf16((s * gscales[v * 3 + 2]) * wts[(int64_t)t * K + k]) : 0.f;")]),
    # v2 down: the upper half of each lane's intermediate slice is replaced by its lower half
    "down-h": ("kernels/cuda/moe_decode.cu", [(
        "ik_unpack8(hp[2 * u + 1], hf + 8);", "ik_unpack8(hp[2 * u], hf + 8);")]),
    # v2 down: an invalid id still contributes (the term of an out-of-range expert is not zeroed)
    "down-invalid": ("kernels/cuda/moe_decode.cu", [(
        "wts[(int64_t)t * K + k]) : 0.f;", "wts[(int64_t)t * K + k]) : 1.f;")]),
    # v2 down: the second row of a block reads the first row's weight scales
    "down-scale": ("kernels/cuda/moe_decode.cu", [(
        "srow[i] = dns + row * (I / 16);", "srow[i] = dns + ((int64_t)v * H + r0) * (I / 16);")]),
}

# test selection per mutation (cargo test arguments after -p intarsia-backend-cuda)
V2 = ("down-order", "down-round", "down-h", "down-invalid", "down-scale")
def test_args(name):
    return "--test quantmoe moe_experts_decode_matches_old_kernel" if name in V2 else "--test fused_decode"

if sys.argv[1] == "--args":  # mutate.py --args <name>: print the cargo test arguments
    print(test_args(sys.argv[2]))
    sys.exit(0)
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
