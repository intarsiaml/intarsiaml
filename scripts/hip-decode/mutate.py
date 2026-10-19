#!/usr/bin/env python3
"""Apply one named mutation to a copy of the tree (usage: mutate.py <tree> <name>). Used by `remote.sh mutate`:
each mutation must make `cargo test -p intarsia-backend-hip --test decode_v2` FAIL, which shows the kernel-level tests
catch a wrong accumulation, a skipped iteration, a flipped tie-break, or a dropped state / tail write."""
import sys

MUTATIONS = {
    # fused single-tile conv: the thread's new state is written doubled
    "conv-state": ("kernels/hip/gdn.hip", [(
        "        for (int j = 0; j < K - 1; ++j) state[(int64_t)c * (K - 1) + j] = h[j];\n}\n\n// state[c] =",
        "        for (int j = 0; j < K - 1; ++j) state[(int64_t)c * (K - 1) + j] = h[j] * 2.f;\n}\n\n// state[c] =")]),
    # nvfp4 gemv v2: the first lane-iteration after the prefetch window is skipped (only visible for K > 2048)
    "nvfp4-iteration": ("kernels/hip/nvfp4.hip", [(
        "for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP) {  // K > 2048",
        "for (int u = lane + IK_WARP * (kPre + 1); u < nu; u += IK_WARP) {  // K > 2048")]),
    # moe top-k v2: ties go to the highest expert index instead of the lowest
    "topk-tiebreak": ("kernels/hip/moe.hip", [
        ("(topk_key16(b.bits) << 16) | (0xffffu - (uint32_t)e)", "(topk_key16(b.bits) << 16) | (uint32_t)e"),
        ("const int bi = (int)(0xffffu - (best & 0xffffu));", "const int bi = (int)(best & 0xffffu);")]),
    # rope: the in-kernel copy of the un-rotated tail skips every other value
    "rope-tail": ("kernels/hip/elementwise.hip", [(
        "for (int m = RD + k; m < D; m += half) out[base + m] = x[base + m];",
        "for (int m = RD + k; m < D; m += 2 * half) out[base + m] = x[base + m];")]),
    # moe experts v2 down kernel: the first half of the intermediate row is replaced by its second half
    "moe-down": ("kernels/hip/moe.hip", [(
        "                step(k, wv[k], sb[k], hv[XS ? 0 : k][0], hv[XS ? 0 : k][1]);",
        "                step(k, wv[k], sb[k], hv[XS ? 0 : k][1], hv[XS ? 0 : k][1]);")]),
    # fp8 gemv v2 (default staging Q=2): the output scale of the 2^-8-scaled hardware conversion is halved
    "fp8-scale": ("kernels/hip/fp8.hip", [(
        "((Q == 2 || Q == 3) ? 65536.f : 256.f)", "((Q == 2 || Q == 3) ? 32768.f : 256.f)")]),
    # fp8 gemv v2: the upper two bytes of each converted e4m3 pair read the lower pair's selector
    "fp8-convert": ("kernels/hip/fp8.hip", [(
        "e4m3x2_to_f16x2((uint32_t)c0, 0x030C020Cu)", "e4m3x2_to_f16x2((uint32_t)c0, 0x010C000Cu)")]),
    # --- fused decode kernels (tests/fused_decode.rs, hip-decode-2) ---
    # bf16 pair: the second matrix's blocks start one block late (its first rows are never written)
    "pair-block": ("kernels/hip/matmul.hip", [(
        "else gemv_body<1, R, U, OutT>(x, w1, o1, N1, K, (int64_t)blockIdx.x - nb0);",
        "else gemv_body<1, R, U, OutT>(x, w1, o1, N1, K, (int64_t)blockIdx.x - nb0 + 1);")]),
    # fp8 pair: the second segment stages and scales with the first segment's scales
    "fp8-pair-scale": ("kernels/hip/fp8.hip", [(
        "else fp8_gemv_body<1, R, OutT, Q, W>(x, w1, sc1, o1, N1, K, (int64_t)blockIdx.x - nb0);",
        "else fp8_gemv_body<1, R, OutT, Q, W>(x, w1, sc0, o1, N1, K, (int64_t)blockIdx.x - nb0);")]),
    # nvfp4 gate+up+silu: the up projection is scaled with the gate's global scale
    "glu-scale": ("kernels/hip/nvfp4.hip", [(
        "ik_warp_sum(acc[1]) * gsu[0]", "ik_warp_sum(acc[1]) * gsg[0]")]),
    # nvfp4 gate+up+silu: the gate value is not rounded to bf16 before the silu (the unfused GEMV stores it as bf16)
    "glu-round": ("kernels/hip/nvfp4.hip", [(
        "const float g = ik_b2f(ik_f2b(ik_warp_sum(acc[0]) * gsg[0]));", "const float g = ik_warp_sum(acc[0]) * gsg[0];")]),
    # gated rmsnorm: silu is applied to the normalised value instead of z
    "gated-order": ("kernels/hip/norm.hip", [(
        "f[k] = ik_silu(zf[k]) * nrm[k];", "f[k] = ik_silu(nrm[k]) * zf[k];")]),
    # gated rmsnorm: the normalised value skips its bf16 rounding
    "gated-round": ("kernels/hip/norm.hip", [(
        "ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)",
        "for (int k = 0; k < 8; ++k) nrm[k] = f[k];")]),
    # --- fp8 pair block shape (hip-fp8-gemv; the default is one row per warp, eight warps) ---
    # the kernel splits its blocks between the segments with a row count per block of two rows per warp, the launcher with R
    "pair-nb0": ("kernels/hip/fp8.hip", [(
        "    constexpr int per = W * R;\n    const int nb0 = (N0 + per - 1) / per;",
        "    constexpr int per = W * 2;\n    const int nb0 = (N0 + per - 1) / per;")]),
    # the launcher sizes the grid for the two segments together instead of one block count per segment
    "pair-grid": ("kernels/hip/fp8.hip", [(
        "const unsigned grid = (unsigned)((N0 + per - 1) / per + (N1 + per - 1) / per);\n    fp8_gemv_pair_kernel<R, ik_bf16, 2, W>",
        "const unsigned grid = (unsigned)((N0 + N1 + per - 1) / per);\n    fp8_gemv_pair_kernel<R, ik_bf16, 2, W>")]),
    # --- rmsnorm_add folded into the bf16 pair GEMV (hip-norm-gemv; tests/fused_decode.rs) ---
    # the designated block does not store the new residual stream
    "norm-skip-sum": ("kernels/hip/matmul.hip", [(
        "        if (writer) reinterpret_cast<uint4*>(sum)[i] = p;\n", "")]),
    # the designated block does not store the normalised row (the MoE experts and the shared expert read it)
    "norm-skip-normed": ("kernels/hip/matmul.hip", [(
        "        if (writer) reinterpret_cast<uint4*>(normed)[i] = p;\n", "")]),
    # the sum is squared unrounded (rmsnorm_add squares the bf16-rounded sum that the next layer carries)
    "norm-sum-round": ("kernels/hip/matmul.hip", [(
        "        ik_unpack8(p, f);  // continue from the bf16-rounded sum\n", "")]),
    # the zero-centred norm's +1 on the weight is dropped
    "norm-woff": ("kernels/hip/matmul.hip", [(
        "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);", "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * g[k];")]),
    # the eight squares of a thread's chunk are added in the opposite order (swapping neighbouring lanes before the butterfly
    # sum would be an equivalent mutant: the xor tree is invariant under it)
    "norm-order": ("kernels/hip/matmul.hip", [(
        "        for (int k = 0; k < 8; ++k) ss += f[k] * f[k];\n    }\n    ss = ik_block_sum(ss, red);",
        "        for (int k = 7; k >= 0; --k) ss += f[k] * f[k];\n    }\n    ss = ik_block_sum(ss, red);")]),
    # the second matrix's blocks start one block late
    "norm-pair-block": ("kernels/hip/matmul.hip", [(
        "w1, o1, N1, K, (int64_t)blockIdx.x - nb0, writer, xs, red);", "w1, o1, N1, K, (int64_t)blockIdx.x - nb0 + 1, writer, xs, red);")]),
    # no barrier between the LDS row being written and being read by the other warps
    "norm-sync": ("kernels/hip/matmul.hip", [(
        "    __syncthreads();  // the whole normalised row is in LDS before any warp reads it\n", "")]),
    # --- gated rmsnorm folded into the fp8 out_proj staging (hip-norm-gemv; tests/fused_decode.rs) ---
    # the normalised value skips its bf16 rounding before the silu product
    "gn-round": ("kernels/hip/fp8.hip", [(
        "    ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)\n",
        "    for (int k = 0; k < 8; ++k) nrm[k] = f[k];\n")]),
    # the zero-centred weight offset is dropped
    "gn-woff": ("kernels/hip/fp8.hip", [(
        "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);", "for (int k = 0; k < 8; ++k) f[k] = f[k] * r * g[k];")]),
    # the sum of squares of a head is added in the opposite order of lanes
    "gn-order": ("kernels/hip/fp8.hip", [(
        "for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor(ss, o, IK_WARP);", "for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor(ss, o, IK_WARP);")]),
    # the reduction spans 32 lanes: two heads share one sum of squares
    "gn-group": ("kernels/hip/fp8.hip", [(
        "for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor(ss, o, IK_WARP);", "for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, IK_WARP);")]),
    # the norm weight is indexed by the wrong chunk
    "gn-weight": ("kernels/hip/fp8.hip", [(
        "reinterpret_cast<const uint4*>(nw) + (i8 & 15)", "reinterpret_cast<const uint4*>(nw) + (i8 & 7)")]),
    # silu is applied to the normalised value instead of z
    "gn-silu": ("kernels/hip/fp8.hip", [(
        "f[k] = ik_silu(zf[k]) * nrm[k];\n    return ik_pack8(f);", "f[k] = ik_silu(nrm[k]) * zf[k];\n    return ik_pack8(f);")]),
    # --- fp8 x fp8 WMMA prefill GEMM (perf/hip-prefill; tests/fp8_f8.rs) and the MoE launch grid (tests/quantmoe.rs) ---
    # the output scale drops the input scale
    "f8-scale": ("kernels/hip/ik_f8_gemm.h", [("const float so = sc[0] * sc[1];", "const float so = sc[0];")]),
    # the token (B) fragments read the other lane half's 16 bytes: A and B disagree on which k a slot holds
    "f8-kperm": ("kernels/hip/ik_f8_gemm.h", [(
        "(wm * TM * 16 + j * 16 + lr) * kF8LD + g * 32 + lh]", "(wm * TM * 16 + j * 16 + lr) * kF8LD + g * 32 + (lh ^ 16)]")]),
    # the activations are quantised with a 6 % larger scale
    "f8-actq": ("kernels/hip/ik_f8_gemm.h", [(
        "lo |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[e], s_in))", "lo |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[e], s_in * 1.0625f))")]),
    # the last, partial 64-byte stage of K is dropped
    "f8-ktail": ("kernels/hip/ik_f8_gemm.h", [("const int nk = (K + kF8BK - 1) / kF8BK;", "const int nk = K / kF8BK;")]),
    # the weight-only half of the launcher no longer steps aside for W8A8: the bf16 kernel overwrites the e4m3 kernel's output
    "f8-w16-always": ("kernels/hip/fp8.hip", [("IK_DEVI bool skip(const DqArgs& a) { return a.sc[1] > 0.f; }", "IK_DEVI bool skip(const DqArgs& a) { return false; }")]),
    # ... and steps aside for weight-only activations too: nothing computes them
    "f8-w16-never": ("kernels/hip/fp8.hip", [("IK_DEVI bool skip(const DqArgs& a) { return a.sc[1] > 0.f; }", "IK_DEVI bool skip(const DqArgs& a) { return true; }")]),
    # the e4m3 kernel does not step aside for weight-only activations (the bf16 kernel runs after it, so this one is expected to survive)
    "f8-noexit": ("kernels/hip/ik_f8_gemm.h", [("    if (!(sc[1] > 0.f)) return;  // weight-only activations: the bf16 kernel handles this launch\n", "")]),
    # a block steps through the further pair tiles of a busy expert by one more than the grid's z
    "moe-z-step": ("kernels/hip/moe_wmma.h", [("pt += gridDim.z)", "pt += gridDim.z + 1)")]),
    # --- round 3 (perf/hip-round3): matmul_fp8_mma_v2 (tests/fp8_mma_v2.rs) and the 64-pair MoE tile (tests/moe_pairs.rs) ---
    # the e4m3 -> bf16 conversion decodes the low byte pair twice (bytes 2, 3 of the low word lost)
    "v2-wsel": ("kernels/hip/ik_quant.h", [(
        "make_uint4(ik_e4m3x2_to_bf16x2(lo, false), ik_e4m3x2_to_bf16x2(lo, true),",
        "make_uint4(ik_e4m3x2_to_bf16x2(lo, false), ik_e4m3x2_to_bf16x2(lo, false),")]),
    # the bf16 is taken from the low half of the converted f32 (truncation from the wrong end)
    "v2-perm": ("kernels/hip/ik_quant.h", [(
        "return __builtin_amdgcn_perm(__float_as_uint(f[1]), __float_as_uint(f[0]), 0x07060302u);",
        "return __builtin_amdgcn_perm(__float_as_uint(f[1]), __float_as_uint(f[0]), 0x05040100u);")]),
    # the second 8 weights of each 16-byte chunk repeat the first 8
    "v2-whi": ("kernels/hip/fp8.hip", [("hi = ik_e4m3x8_to_bf16(r.z, r.w);", "hi = ik_e4m3x8_to_bf16(r.x, r.y);")]),
    # the quantised activation rows are addressed with the bf16 row stride
    "v2-xstride": ("kernels/hip/fp8.hip", [("a.xq + (int64_t)m * a.K + k", "a.xq + (int64_t)m * a.K * 2 + k")]),
    # the quantiser covers only the first half of the activations (the rest of the scratch is stale)
    "v2-quant-half": ("kernels/hip/fp8.hip", [("sp, n8);\n    DqArgs a;\n    a.x = (const ik_bf16*)x->data;\n    a.w = (const uint8_t*)w->data;",
                                              "sp, n8 / 2);\n    DqArgs a;\n    a.x = (const ik_bf16*)x->data;\n    a.w = (const uint8_t*)w->data;")]),
    # W8A8 activations are not quantised at all (read and used as bf16)
    "v2-noquant": ("kernels/hip/fp8.hip", [
        ("        if (s.actq) {\n            const uint2 q", "        if (false) {\n            const uint2 q"),
        ("IK_DEVI uint4 xprep(const State& s, uint4 v) { return s.actq ? ik_e4m3x8_to_bf16(v.x, v.y) : v; }",
         "IK_DEVI uint4 xprep(const State& s, uint4 v) { return v; }")]),
    # the quantised activations are decoded as if they were bf16 (the decode step dropped)
    "v2-nodecode": ("kernels/hip/fp8.hip", [(
        "IK_DEVI uint4 xprep(const State& s, uint4 v) { return s.actq ? ik_e4m3x8_to_bf16(v.x, v.y) : v; }",
        "IK_DEVI uint4 xprep(const State& s, uint4 v) { return v; }")]),
    # 64-pair tile: the up accumulators are handed to the gate waves with the 32-pair stride (fragments overwrite each other)
    "pairs-sred": ("kernels/hip/moe_wmma.h", [(
        "sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] = acc[j][i];", "sred[(fw * 32 + lane) * 16 + j * 8 + i] = acc[j][i];")]),
    # 64-pair tile: the second activation row of each thread is stored 16 rows down instead of 32
    "pairs-xrow": ("kernels/hip/moe_wmma.h", [(
        "*reinterpret_cast<uint4*>(&sx[buf][(xr + 32 * r) * kMwLD + xch * 8]) = rx[r];",
        "*reinterpret_cast<uint4*>(&sx[buf][(xr + 16 * r) * kMwLD + xch * 8]) = rx[r];")]),
    # 64-pair tile: only the first two 16-pair fragments are written out
    "pairs-out": ("kernels/hip/moe_wmma.h", [(
        "        for (int j = 0; j < J; ++j) {\n            const int pi = pt * PAIRS + j * 16 + lr;",
        "        for (int j = 0; j < 2; ++j) {\n            const int pi = pt * PAIRS + j * 16 + lr;")]),
}

# the test target that must fail for each mutation (default decode_v2)
TESTS = {n: "fused_decode" for n in ("pair-block", "fp8-pair-scale", "glu-scale", "glu-round", "gated-order", "gated-round", "pair-nb0", "pair-grid", "norm-skip-sum", "norm-skip-normed", "norm-sum-round", "norm-woff", "norm-order", "norm-pair-block", "norm-sync", "gn-round", "gn-woff", "gn-order", "gn-group", "gn-weight", "gn-silu")}
TESTS.update({n: "fp8_f8" for n in ("f8-scale", "f8-kperm", "f8-actq", "f8-ktail", "f8-w16-always", "f8-w16-never", "f8-noexit")})
TESTS["moe-z-step"] = "quantmoe"
TESTS.update({n: "fp8_mma_v2" for n in ("v2-wsel", "v2-perm", "v2-whi", "v2-xstride", "v2-quant-half", "v2-noquant", "v2-nodecode")})
TESTS.update({n: "moe_pairs" for n in ("pairs-sred", "pairs-xrow", "pairs-out")})

tree, name = sys.argv[1], sys.argv[2]
if name == "--test-of":  # mutate.py --test-of <mutation>: print the cargo test target
    print(TESTS.get(sys.argv[3], "decode_v2"))
    sys.exit(0)
path, pairs = MUTATIONS[name]
p = f"{tree}/{path}"
s = open(p).read()
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit(f"mutation {name}: pattern not found exactly once in {path}: {old[:60]!r} ({s.count(old)})")
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"applied {name} to {p}")
