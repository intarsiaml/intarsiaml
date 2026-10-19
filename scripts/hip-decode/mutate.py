#!/usr/bin/env python3
"""Apply one named mutation to a copy of the tree (usage: mutate.py <tree> <name>). Used by `remote.sh mutate`:
each mutation must make `cargo test -p intarsia-backend-hip --test decode_v2` FAIL, which shows the kernel-level tests
catch a wrong accumulation, a skipped iteration, a flipped tie-break, or a dropped state / tail write."""
import sys

# moe_wmma.h holds two kernels with the same body text: moe_wmma_kernel (round 5: the 32-pair tile, and INTARSIA_HIP_MOE_MW=0)
# and moe_wmma_kernel_v (round 6: the 64-pair tile at the default MW=5). A pattern found in both is aimed at one of them by a
# third element (index, count): the file must hold exactly `count` matches, and the `index`-th (0-based) is replaced. The
# round-6 `moe-z-step` / `pairs-*` mutants take the second, moe_wmma_kernel_v: the kernel their tests select at default env
# (quantmoe's hot-experts case averages >= 32 pairs per expert, so the auto tile is 64; moe_pairs runs MW 1 / 4 / 5 at 64
# pairs against MW 0 at 32).
V_KERNEL = (1, 2)

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
    "moe-z-step": ("kernels/hip/moe_wmma.h", [("pt += gridDim.z)", "pt += gridDim.z + 1)", V_KERNEL)]),
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
        "sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] = acc[j][i];", "sred[(fw * 32 + lane) * 16 + j * 8 + i] = acc[j][i];", V_KERNEL)]),
    # 64-pair tile: the second activation row of each thread is stored 16 rows down instead of 32
    "pairs-xrow": ("kernels/hip/moe_wmma.h", [(
        "*reinterpret_cast<uint4*>(&sx[buf][(xr + 32 * r) * kMwLD + xch * 8]) = rx[r];",
        "*reinterpret_cast<uint4*>(&sx[buf][(xr + 16 * r) * kMwLD + xch * 8]) = rx[r];", V_KERNEL)]),
    # 64-pair tile: only the first two 16-pair fragments are written out
    "pairs-out": ("kernels/hip/moe_wmma.h", [(
        "        for (int j = 0; j < J; ++j) {\n            const int pi = pt * PAIRS + j * 16 + lr;",
        "        for (int j = 0; j < 2; ++j) {\n            const int pi = pt * PAIRS + j * 16 + lr;", V_KERNEL)]),
    # --- round 4 (docs/results/2026-10-19-hip-round4.md) ---
    # qk_norm_rope_kv: the sum of squares misses the last value of each 8-value chunk
    "qkr-sum": ("kernels/hip/elementwise.hip", [(
        "for (int e = 0; e < 8; ++e) ss += f[e] * f[e];\n    }\n    ss = ik_warp_sum(ss);",
        "for (int e = 0; e < 7; ++e) ss += f[e] * f[e];\n    }\n    ss = ik_warp_sum(ss);")]),
    # qk_norm_rope_kv: the zero-centred weight offset is dropped
    "qkr-woff": ("kernels/hip/elementwise.hip", [(
        "for (int e = 0; e < 8; ++e) f[e] = f[e] * r * (g[e] + woff);\n        reinterpret_cast<uint4*>(nr)[i]",
        "for (int e = 0; e < 8; ++e) f[e] = f[e] * r * g[e];\n        reinterpret_cast<uint4*>(nr)[i]")]),
    # qk_norm_rope_kv: every row of a chunk is rotated by the first row's angle
    "qkr-angle": ("kernels/hip/elementwise.hip", [(
        "sincos((double)(pos0 + t) * inv, &sd, &cd);\n        fsn[f]", "sincos((double)(pos0) * inv, &sd, &cd);\n        fsn[f]")]),
    # qk_norm_rope_kv: the rotation's first half adds the sine term instead of subtracting it
    "qkr-sign": ("kernels/hip/elementwise.hip", [(
        "lof = __builtin_fmaf(a, fc, -(b * fs));", "lof = __builtin_fmaf(a, fc, b * fs);")]),
    # qk_norm_rope_kv: hi fuses the other product (the contraction rope_kernel does not use; flips a bf16 rounding rarely)
    "qkr-hifma": ("kernels/hip/elementwise.hip", [(
        "hif = __builtin_fmaf(b, fc, a * fs);", "hif = __builtin_fmaf(a, fs, b * fc);")]),
    # qk_norm_rope_kv: the un-rotated tail is passed through for every other lane only
    "qkr-tail": ("kernels/hip/elementwise.hip", [(
        "for (int m = RD + lane; m < D; m += IK_WARP) {", "for (int m = RD + lane; m < D; m += 2 * IK_WARP) {")]),
    # qk_norm_rope_kv: only the first half of each v row is appended
    "qkr-v": ("kernels/hip/elementwise.hip", [(
        "for (int m = lane; m < D; m += IK_WARP) qkr_store(dv + m, vr[m]);", "for (int m = lane; m < D / 2; m += IK_WARP) qkr_store(dv + m, vr[m]);")]),
    # qk_norm_rope_kv: every k row of a chunk lands in the cache row of the first
    "qkr-kpos": ("kernels/hip/elementwise.hip", [(
        "dst_k = kcache + ((int64_t)kvh * cap + pos0 + t) * D;", "dst_k = kcache + ((int64_t)kvh * cap + pos0) * D;")]),
    # qk_norm_rope_kv: the fp8 cache gets half the value
    "qkr-fp8": ("kernels/hip/elementwise.hip", [(
        "else *reinterpret_cast<uint8_t*>(c) = f32_to_e4m3(ik_b2f(v));", "else *reinterpret_cast<uint8_t*>(c) = f32_to_e4m3(ik_b2f(v) * 0.5f);")]),
    # qk_norm_rope_kv: each query wave normalises the next head of its group
    "qkr-head": ("kernels/hip/elementwise.hip", [(
        "const ik_bf16* xr = is_k ? k + (t * NKV + kvh) * D : q + (t * N + (int64_t)kvh * G + w) * D;",
        "const ik_bf16* xr = is_k ? k + (t * NKV + kvh) * D : q + (t * N + (int64_t)kvh * G + (w + 1) % G) * D;")]),
    # delta rule v2: the key norm sums q * k instead of k * k
    "gdn2-knorm": ("kernels/hip/gdn.hip", [("kk += kr[i] * kr[i];", "kk += qr[i] * kr[i];")]),
    # delta rule v2: a ring slot reloads its own token instead of the one PD ahead
    "gdn2-ring": ("kernels/hip/gdn.hip", [("load(r, t + PD);  // this slot's next token", "load(r, t + PD - 1);  // this slot's next token")]),
    # delta rule v2: the gate of the token one later is read (lane + 1)
    "gdn2-gate": ("kernels/hip/gdn.hip", [("const int gl = (int)(t & 31);", "const int gl = (int)((t + 1) & 31);")]),
    # delta rule v2: the next gate batch is loaded one token late
    "gdn2-batch": ("kernels/hip/gdn.hip", [("load_gates(t0 + 32);", "load_gates(t0 + 33);")]),
    # delta rule v2: the xor-16 step of the warp sum is dropped (half the q/k norm)
    "gdn2-x16": ("kernels/hip/gdn.hip", [("    v += gdn_xor16(v);\n", "")]),
    # delta rule v2: the xor-1 DPP partner is lane ^ 2 (the row split adds the wrong half)
    "gdn2-x1": ("kernels/hip/gdn.hip", [("        kv += gdn_xorrow<1>(kv);", "        kv += gdn_xorrow<2>(kv);")]),
    # delta rule v2: the update loop reuses k of the first float4 only
    "gdn2-kreg": ("kernels/hip/gdn.hip", [("const float4 kk4 = kreg[i], qq4 = q4[i];", "const float4 kk4 = kreg[0], qq4 = q4[i];")]),
    # delta rule v2: the dot products fuse the other product of the first pair (one bf16 output in ~10^4 moves)
    "gdn2-dot": ("kernels/hip/gdn.hip", [("float t = __builtin_fmaf(s.x, k.x, s.y * k.y);", "float t = __builtin_fmaf(s.y, k.y, s.x * k.x);")]),
    # delta rule v2: the value is halved
    "gdn2-v": ("kernels/hip/gdn.hip", [("const float v = valid ? ik_b2f(r.v) : 0.f;", "const float v = valid ? ik_b2f(r.v) * 0.5f : 0.f;")]),
    # --- round 5: the f8 default (unit tests of the selection, no GPU: `cargo test -p intarsia-backend-hip --lib`) ---
    # back to the round-4 opt-in (unset = v2)
    "f8-optin": ("crates/intarsia-backend-hip/src/quantmoe.rs", [('    v != Some("0")\n', '    v == Some("1")\n')]),
    # the kill switch is ignored ("0" keeps f8)
    "f8-nokill": ("crates/intarsia-backend-hip/src/quantmoe.rs", [('    v != Some("0")\n', '    v != Some("off")\n')]),
    # f8 registered below v2 (v2 would stay the GEMM although f8 is "on")
    # v3 selected by default again (the opt-in ignored)
    "v3-default": ("crates/intarsia-backend-hip/src/quantmoe.rs", [('    v == Some("1")\n', '    v != Some("0")\n')]),
    # v3 registered above v2 whatever the knob says
    "v3-prio": ("crates/intarsia-backend-hip/src/quantmoe.rs", [("                                -10\n", "                                30\n")]),
    "f8-prio": ("crates/intarsia-backend-hip/src/quantmoe.rs", [(".priority(if fp8_f8_enabled() { 5 } else { -10 })", ".priority(if fp8_f8_enabled() { 0 } else { -10 })")]),
    # --- round 5: moe experts v3 (one-launch decode; tests/moe_v3.rs) ---
    # the arrive counter is not reset: the next launch's down blocks pass the wait at once and read the previous h
    "v3-noreset": ("kernels/hip/moe.hip", [("            __hip_atomic_store(sync, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);\n", "")]),
    # the wait ends one gate_up block early
    "v3-early": ("kernels/hip/moe.hip", [("< (unsigned)NG) __builtin_amdgcn_s_sleep(1);", "< (unsigned)NG - 1) __builtin_amdgcn_s_sleep(1);")]),
    # no wait at all
    "v3-nowait": ("kernels/hip/moe.hip", [("        while (__hip_atomic_load(sync, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)NG) __builtin_amdgcn_s_sleep(1);\n", "")]),
    # the ticket counter is not reset: the next launch has only down roles (no gate_up) and hangs (killed by the timeout)
    "v3-ticket": ("kernels/hip/moe.hip", [("            __hip_atomic_store(sync + 2, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);\n", "")]),
    # the down projection's first lane-iteration reads expert 0's intermediate row for every expert
    "v3-hrow": ("kernels/hip/moe.hip", [("                hload(k < K ? k : 0, lane, h0, h1);", "                hload(0, lane, h0, h1);")]),
    # the gate_up role stores its row in the neighbouring row's slot (h permuted)
    "v3-gurow": ("kernels/hip/moe.hip", [("            if (lane == 0) h[(int64_t)p * I + r] = ik_f2b(a * uu);", "            if (lane == 0) h[(int64_t)p * I + (r ^ 1)] = ik_f2b(a * uu);")]),
    # the acquire fence before the h reads is dropped (may survive: the L0 / L1 lines of h are rarely stale in practice)
    # --- round 5: flash v2 (tests/flash_v2.rs, run at the default HPB = 2) ---
    # every wave group takes the block's first head
    "fl2-head": ("kernels/hip/attention_flash.hip", [("    const int h = blockIdx.y * HPB + (warp >> 2), kvh = (blockIdx.y * HPB) / G;", "    const int h = blockIdx.y * HPB, kvh = (blockIdx.y * HPB) / G;")]),
    # the kv head from the block index instead of the block's first query head (equivalent only at HPB = 1)
    "fl2-kvh": ("kernels/hip/attention_flash.hip", [("    const int h = blockIdx.y * HPB + (warp >> 2), kvh = (blockIdx.y * HPB) / G;", "    const int h = blockIdx.y * HPB + (warp >> 2), kvh = blockIdx.y / G;")]),
    # rows from the wave index (group 1 computes rows 64.. of the tile, past the block)
    "fl2-row": ("kernels/hip/attention_flash.hip", [("    const int qrow = q0 + (warp & 3) * 16 + lr;", "    const int qrow = q0 + warp * 16 + lr;")]),
    # the tile load skips every other stride of chunks
    "fl2-stride": ("kernels/hip/attention_flash.hip", [("        for (int idx = tid; idx < BK * CH; idx += NT) {", "        for (int idx = tid; idx < BK * CH; idx += 2 * NT) {")]),
    "v3-nofence": ("kernels/hip/moe.hip", [('    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");  // every wave: h is read only after this\n', "")]),
    # round 6: MoE projection variants (moe_wmma.h moe_wmma_kernel_v; tests/moe_pairs.rs runs every variant against 0)
    # a partly filled tail tile computes one fragment too few (rem >> 4 instead of rounding up)
    "mw-jn": ("kernels/hip/moe_wmma.h", [("const int jn = rem >= PAIRS ? J : (rem + 15) >> 4;", "const int jn = rem >= PAIRS ? J : rem >> 4;")]),
    # a full tile skips its last fragment
    "mw-full": ("kernels/hip/moe_wmma.h", [("if (jn >= J) kloop(std::integral_constant<int, J>{});", "if (jn >= J) kloop(std::integral_constant<int, J - 1>{});")]),
    # a tile with two filled fragments computes one
    "mw-two": ("kernels/hip/moe_wmma.h", [("else if (J == 4 && jn == 2) kloop(std::integral_constant<int, (J == 4 ? 2 : 1)>{});", "else if (J == 4 && jn == 2) kloop(std::integral_constant<int, (J == 4 ? 1 : 1)>{});")]),
    # aliased up accumulators: the last k block's barrier dropped, so the up waves may overwrite activations still being read
    "mw-lastsync": ("kernels/hip/moe_wmma.h", [("                    __syncthreads();\n", "                    if (!ALIAS || kb + 1 < nkb) __syncthreads();\n")]),
    # round 6 selection pins (tests/round6_selection.rs)
    "mw-def64": ("kernels/hip/moe.hip", [("constexpr int kMwDefault64 = kMwSkip | kMwOcc, kMwDefault32 = 0;", "constexpr int kMwDefault64 = kMwSkip, kMwDefault32 = 0;")]),
    "mw-def32": ("kernels/hip/moe.hip", [("constexpr int kMwDefault64 = kMwSkip | kMwOcc, kMwDefault32 = 0;", "constexpr int kMwDefault64 = kMwSkip | kMwOcc, kMwDefault32 = kMwSkip;")]),
    "mw-tile": ("kernels/hip/moe.hip", [("const int def = pairs == 64 ? kMwDefault64 : kMwDefault32;", "const int def = pairs == 32 ? kMwDefault64 : kMwDefault32;")]),
    "mw-range": ("kernels/hip/moe.hip", [("return (v == 0 || v == kMwSkip || v == kMwOcc || v == (kMwSkip | kMwOcc)) ? v : def;", "return (v >= 0 && v <= 7) ? v : def;")]),
    "hpb-def": ("kernels/hip/attention_flash.hip", [("constexpr int kFlashHpbDefault = 4;", "constexpr int kFlashHpbDefault = 2;")]),
    "hpb-two": ("kernels/hip/attention_flash.hip", [("return (x == 1 || x == 2 || x == 4) ? x : kFlashHpbDefault;", "return (x == 1 || x == 4) ? x : kFlashHpbDefault;")]),
    # --- core-op fusion phase 1 (tests/fusion_p1.rs; docs/results/2026-10-19-core-op-fusion-p1.md) ---
    # P5: the second segment (v) reads the first segment's scales
    "p1-n5-sc0": ("kernels/hip/fp8.hip", [("(x, w1, sc1, o1, N1, K, (int64_t)blockIdx.x - nb0, res, nw", "(x, w1, sc0, o1, N1, K, (int64_t)blockIdx.x - nb0, res, nw")]),
    # P5: no block stores normed and sum
    "p1-n5-nowrite": ("kernels/hip/fp8.hip", [("    const bool writer = blockIdx.x == 0;\n", "    const bool writer = false;\n")]),
    # P5: the squares are summed from the unrounded sum (rmsnorm_kernel continues from the bf16-rounded one)
    "p1-n5-round": ("kernels/hip/fp8.hip", [("        ik_unpack8(p, f);  // continue from the bf16-rounded sum\n", "")]),
    # P5: the norm weight offset is dropped (zero-centred norms)
    "p1-n5-woff": ("kernels/hip/fp8.hip", [("f[k] = f[k] * r * (g[k] + w_off);", "f[k] = f[k] * r * g[k];")]),
    # P5: the GEMV stages the raw x instead of the normed row
    "p1-n5-stage": ("kernels/hip/fp8.hip", [("else if constexpr (PRO == 3) v = reinterpret_cast<const uint4*>(xs)[i8];", "else if constexpr (PRO == 3) v = reinterpret_cast<const uint4*>(x)[i8];")]),
    # P4b: sigmoid of the negated gate
    "p1-g4-sig": ("kernels/hip/fp8.hip", [("f[k] = f[k] * (1.f / (1.f + expf(-gv)));", "f[k] = f[k] * (1.f / (1.f + expf(gv)));")]),
    # P4b: the gate chunk read from a
    "p1-g4-gate": ("kernels/hip/fp8.hip", [("ik_unpack8(reinterpret_cast<const uint4*>(gate)[i8], g);", "ik_unpack8(reinterpret_cast<const uint4*>(a)[i8], g);")]),
    # P2a: every row uses row 0's shared-gate logit
    "p1-dc-logit0": ("kernels/hip/moe_shared.hip", [("const float gl = ik_b2f(logit[m]);", "const float gl = ik_b2f(logit[0]);")]),
    # P2a: the sigmoid of the gate is not rounded to bf16 (moe_combine_kernel rounds it)
    "p1-dc-sground": ("kernels/hip/moe_shared.hip", [("const float sg = ik_round_bf16(1.f / (1.f + expf(-gl)));", "const float sg = 1.f / (1.f + expf(-gl));")]),
    # P2a: the down projection is not rounded to bf16 before the combine (the unfused path stores it)
    "p1-dc-soround": ("kernels/hip/moe_shared.hip", [("const float so = ik_round_bf16(s * g);", "const float so = s * g;")]),
    # P2a: the first lane-iteration after the prefetch window is skipped (K > 2048 only)
    "p1-dc-tail": ("kernels/hip/moe_shared.hip", [("for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP) {  // K > 2048", "for (int u = lane + IK_WARP * (kPre + 1); u < nu; u += IK_WARP) {  // K > 2048")]),
    # P2a: every row combines with row 0's routed values and writes row 0
    "p1-dc-row": ("kernels/hip/moe_shared.hip", [("const int64_t i = (int64_t)m * N + n;", "const int64_t i = n;")]),
    # P5 selection: the one-row fold also taken for two rows (a prefill / verify shape it was never checked on)
    "p1-sel-m2": ("crates/intarsia-backend-hip/src/quantmoe.rs", [("""                "rmsnorm_add_matmul_fp8_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        s.m == 1""", """                "rmsnorm_add_matmul_fp8_pair_gemv" => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| {
                        s.m <= 2""")]),
    # P2a selection: nine rows (the GEMV launcher refuses them: the plan would fail instead of falling back)
    "p1-sel-dc9": ("crates/intarsia-backend-hip/src/quantmoe.rs", [("""                "shared_down_combine_nvfp4_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=8).contains(&s.m)""", """                "shared_down_combine_nvfp4_gemv" => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| (1..=9).contains(&s.m)""")]),
    # --- round 7 (docs/results/2026-10-19-hip-round7.md): moe_wmma_r7.h (kMwCvt fp8-conversion fp4 decode, kMwTwo two A
    # fragments per wave in down, kMwDeep six weight blocks in flight) and the per-projection selection in moe.hip. The round-6
    # entries mw-def64 / mw-def32 / mw-tile / mw-range mutate lines round 7 replaced; r7-def* / r7-tile / r7-range cover them.
    # even elements' magnitude one bit off / odd elements' magnitude one bit off
    "r7-even": ("kernels/hip/moe_wmma_r7.h", [("ev = ((c << 2) & 0x1C1C1C1Cu)", "ev = ((c << 1) & 0x1C1C1C1Cu)")]),
    "r7-odd": ("kernels/hip/moe_wmma_r7.h", [("od = ((c >> 2) & 0x1C1C1C1Cu)", "od = ((c >> 3) & 0x1C1C1C1Cu)")]),
    # even elements lose their sign / odd elements lose their sign
    "r7-esign": ("kernels/hip/moe_wmma_r7.h", [(" | ((c << 4) & 0x80808080u);", ";")]),
    "r7-osign": ("kernels/hip/moe_wmma_r7.h", [(" | (c & 0x80808080u);", ";")]),
    # the 2^6 fold of the block scale off by one binade
    "r7-x64": ("kernels/hip/moe_wmma_r7.h", [("ik_fp4x8_scaled_bf16_cvt(cw[s], sc * 64.f)", "ik_fp4x8_scaled_bf16_cvt(cw[s], sc * 32.f)")]),
    # elements 4/6 converted from the low half of the even word (elements 0/2 again)
    "r7-hiword": ("kernels/hip/moe_wmma_r7.h", [("e46 = __builtin_amdgcn_cvt_pk_f32_fp8((int)ev, true)", "e46 = __builtin_amdgcn_cvt_pk_f32_fp8((int)ev, false)")]),
    # the even and odd element of the first pair swapped
    "r7-swap": ("kernels/hip/moe_wmma_r7.h", [("r.x = __builtin_amdgcn_perm(__float_as_uint(o13[0] * sc64), __float_as_uint(e02[0] * sc64), 0x07060302u);", "r.x = __builtin_amdgcn_perm(__float_as_uint(e02[0] * sc64), __float_as_uint(o13[0] * sc64), 0x07060302u);")]),
    # kMwDeep: the ring refills a slot from one block too far (blocks RING+1.. instead of RING..)
    "r7-deep": ("kernels/hip/moe_wmma_r7.h", [("for (int f = 0; f < RF; ++f) W[i][f] = loadW(f, kb + RING);", "for (int f = 0; f < RF; ++f) W[i][f] = loadW(f, kb + RING + (RING > 3));")]),
    # two-fragment down: the second fragment computes the first one's rows (rows 16..31 of each wave never written)
    "r7-dnrow": ("kernels/hip/moe_wmma_r7.h", [("else return n0 + (wave * RF + f) * 16;", "else return n0 + wave * RF * 16;")]),
    # round-7 projections launched over half the feature tiles
    "r7-grid": ("kernels/hip/moe.hip", [("constexpr int feat = (V & (kMwCvt | kMwTwo)) ? mw_r7_feat(GATEUP, V)", "constexpr int feat = (V & (kMwCvt | kMwTwo)) ? 2 * mw_r7_feat(GATEUP, V)")]),
    # kMwSkip in moe_wmma_kernel_r7: a partly filled tail tile computes one fragment too few
    "r7-jn": ("kernels/hip/moe_wmma_r7.h", [("const int jn = rem >= PAIRS ? J : (rem + 15) >> 4;", "const int jn = rem >= PAIRS ? J : rem >> 4;")]),
    # kMwOcc aliasing in moe_wmma_kernel_r7: the k loop's last barrier dropped
    "r7-lastsync": ("kernels/hip/moe_wmma_r7.h", [("                    __syncthreads();\n                }\n            }\n        };", "                    if (!ALIAS || kb + 1 < nkb) __syncthreads();\n                }\n            }\n        };")]),
    # selection (tests/round6_selection.rs): each default, the tile mapping, the accepted values, the g:d split
    "r7-def64": ("kernels/hip/moe.hip", [("constexpr int kMwDefault64 = kMwCvt | kMwSkip | kMwOcc | kMwDeep,", "constexpr int kMwDefault64 = kMwCvt | kMwSkip | kMwOcc,")]),
    "r7-def32": ("kernels/hip/moe.hip", [("constexpr int kMwDefault32 = 0, kMwDefault32Dn", "constexpr int kMwDefault32 = kMwSkip, kMwDefault32Dn")]),
    "r7-def64dn": ("kernels/hip/moe.hip", [("kMwDefault64Dn = kMwCvt | kMwSkip | kMwTwo;", "kMwDefault64Dn = kMwCvt | kMwSkip;")]),
    "r7-def32dn": ("kernels/hip/moe.hip", [("kMwDefault32Dn = kMwCvt | kMwSkip | kMwDeep;", "kMwDefault32Dn = 0;")]),
    "r7-tile": ("kernels/hip/moe.hip", [("const int def = pairs == 64 ? (down ? kMwDefault64Dn : kMwDefault64) : (down ? kMwDefault32Dn : kMwDefault32);", "const int def = pairs == 32 ? (down ? kMwDefault64Dn : kMwDefault64) : (down ? kMwDefault32Dn : kMwDefault32);")]),
    "r7-range": ("kernels/hip/moe.hip", [("    return mw_accepted(v) ? v : def;", "    return (v >= 0 && v <= 7) ? v : def;")]),
    "r7-colon": ("kernels/hip/moe.hip", [("const char* f = down && colon ? colon + 1 : e;", "const char* f = e;")]),
    # core-op fusion phase 2 (tests/fusion_p2.rs, docs/results/2026-10-19-core-op-fusion-p2.md)
    # P4a gate_split_qk_norm_rope_kv: the gate copied from the query half
    "p2-ga-gate": ("kernels/hip/gate_qkr.hip", [("gate[qrow * D + m] = qg[qrow * 2 * D + D + m];", "gate[qrow * D + m] = qg[qrow * 2 * D + m];")]),
    # the query read with the split-off row's stride (D) instead of the gated row's (2D)
    "p2-ga-qstride": ("kernels/hip/gate_qkr.hip", [("k + (t * NKV + kvh) * D : qg + qrow * 2 * D;", "k + (t * NKV + kvh) * D : qg + qrow * D;")]),
    # the zero-centred norm's +1 dropped
    "p2-ga-woff": ("kernels/hip/gate_qkr.hip", [("f[e] = f[e] * r * (g[e] + woff);", "f[e] = f[e] * r * g[e];")]),
    # the fp8 cache value halved before the conversion
    "p2-ga-fp8": ("kernels/hip/gate_qkr.hip", [("gqr_f32_to_e4m3(ik_b2f(v));", "gqr_f32_to_e4m3(ik_b2f(v) * 0.5f);")]),
    # every token's v row written to the first new cache row
    "p2-ga-vpos": ("kernels/hip/gate_qkr.hip", [("CacheT* dv = vcache + ((int64_t)kvh * cap + pos0 + t) * D;", "CacheT* dv = vcache + ((int64_t)kvh * cap + pos0) * D;")]),
    # P3 gdn_conv_delta: the FIRST block of a key head stores its new q / k conv state (siblings may still read the old one)
    "p2-cd-writer0": ("kernels/hip/gdn_fused.hip", [("qk_writer = tk_ == nblk - 1;", "qk_writer = tk_ == 0;")]),
    # the key-head counter is not put back to 0: the next launch's tickets never reach nblk - 1
    "p2-cd-noreset": ("kernels/hip/gdn_fused.hip", [("        if (qk_writer && lane == 0) __hip_atomic_store(sync + kh, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);\n", "")]),
    # the v channels' new conv state not stored
    "p2-cd-novstate": ("kernels/hip/gdn_fused.hip", [("for (int t = 0; t < KH; ++t) cstate[(int64_t)cv * KH + t] = hv[t];", "for (int t = 0; t < KH; ++t) (void)hv[t];")]),
    # the conv output not rounded to bf16 before the delta rule reads it (the unfused path stores y in bf16)
    "p2-cd-round": ("kernels/hip/gdn_fused.hip", [("const float y = ik_b2f(ik_f2b(silu(acc)));", "const float y = silu(acc);")]),
    # the current input multiplied by the first tap instead of the last
    "p2-cd-tap": ("kernels/hip/gdn_fused.hip", [("float acc = cd_tap(tp, K - 1) * cur;", "float acc = cd_tap(tp, 0) * cur;")]),
    # the conv history not shifted between tokens
    "p2-cd-hist": ("kernels/hip/gdn_fused.hip", [("    for (int j = 0; j < K - 2; ++j) hh[j * S] = hh[(j + 1) * S];\n", "")]),
    # the two-launch form (no counter slot) skips its conv-state launch
    "p2-cd-slotstate": ("kernels/hip/gdn_fused.hip", [("    if (slot >= 0) return ik_launch_status();", "    return ik_launch_status();")]),
    # selection: the fold admitted one token past the conv's single tile
    "p2-sel-m33": ("crates/intarsia-backend-hip/src/kernels.rs", [("pub const GDN_CONV_DELTA_MAX_TOKENS: usize = 32;", "pub const GDN_CONV_DELTA_MAX_TOKENS: usize = 33;")]),
}

# the test target that must fail for each mutation (default decode_v2)
TESTS = {n: "fused_decode" for n in ("pair-block", "fp8-pair-scale", "glu-scale", "glu-round", "gated-order", "gated-round", "pair-nb0", "pair-grid", "norm-skip-sum", "norm-skip-normed", "norm-sum-round", "norm-woff", "norm-order", "norm-pair-block", "norm-sync", "gn-round", "gn-woff", "gn-order", "gn-group", "gn-weight", "gn-silu")}
TESTS.update({n: "fp8_f8" for n in ("f8-scale", "f8-kperm", "f8-actq", "f8-ktail", "f8-w16-always", "f8-w16-never", "f8-noexit")})
TESTS["moe-z-step"] = "quantmoe"
TESTS.update({n: "fp8_mma_v2" for n in ("v2-wsel", "v2-perm", "v2-whi", "v2-xstride", "v2-quant-half", "v2-noquant", "v2-nodecode")})
TESTS.update({n: "moe_pairs" for n in ("pairs-sred", "pairs-xrow", "pairs-out")})
TESTS.update({n: "qk_rope_kv" for n in ("qkr-sum", "qkr-woff", "qkr-angle", "qkr-sign", "qkr-hifma", "qkr-tail", "qkr-v", "qkr-kpos", "qkr-fp8", "qkr-head")})
TESTS.update({n: "lib" for n in ("f8-optin", "f8-nokill", "f8-prio", "v3-default", "v3-prio", "p1-sel-m2", "p1-sel-dc9")})
TESTS.update({n: "flash_v2" for n in ("fl2-head", "fl2-kvh", "fl2-row", "fl2-stride")})
TESTS.update({n: "moe_pairs" for n in ("mw-jn", "mw-full", "mw-two", "mw-lastsync")})
TESTS.update({n: "round6_selection" for n in ("mw-def64", "mw-def32", "mw-tile", "mw-range", "hpb-def", "hpb-two")})
TESTS.update({n: "moe_v3" for n in ("v3-noreset", "v3-early", "v3-nowait", "v3-ticket", "v3-hrow", "v3-gurow", "v3-nofence")})
TESTS.update({n: "fusion_p1" for n in ("p1-n5-sc0", "p1-n5-nowrite", "p1-n5-round", "p1-n5-woff", "p1-n5-stage", "p1-g4-sig", "p1-g4-gate", "p1-dc-logit0", "p1-dc-sground", "p1-dc-soround", "p1-dc-tail", "p1-dc-row")})
TESTS.update({n: "fusion_p2" for n in ("p2-ga-gate", "p2-ga-qstride", "p2-ga-woff", "p2-ga-fp8", "p2-ga-vpos", "p2-cd-writer0", "p2-cd-noreset", "p2-cd-novstate", "p2-cd-round", "p2-cd-tap", "p2-cd-hist", "p2-cd-slotstate")})
TESTS["p2-sel-m33"] = "lib"
TESTS.update({n: "gdn_v2" for n in ("gdn2-knorm", "gdn2-ring", "gdn2-gate", "gdn2-batch", "gdn2-x16", "gdn2-x1", "gdn2-kreg", "gdn2-dot", "gdn2-v")})
TESTS.update({n: "moe_pairs" for n in ("r7-even", "r7-odd", "r7-esign", "r7-osign", "r7-x64", "r7-hiword", "r7-swap", "r7-deep", "r7-dnrow", "r7-grid", "r7-jn", "r7-lastsync")})
TESTS.update({n: "round6_selection" for n in ("r7-def64", "r7-def32", "r7-def64dn", "r7-def32dn", "r7-tile", "r7-range", "r7-colon")})

tree, name = sys.argv[1], sys.argv[2]
if name == "--test-of":  # mutate.py --test-of <mutation>: print the cargo test target
    print(TESTS.get(sys.argv[3], "decode_v2"))
    sys.exit(0)
path, pairs = MUTATIONS[name]
p = f"{tree}/{path}"
s = open(p).read()
for pair in pairs:
    old, new = pair[0], pair[1]
    idx, want = pair[2] if len(pair) > 2 else (0, 1)  # (index, count): see V_KERNEL
    if s.count(old) != want:
        sys.exit(f"mutation {name}: pattern not found exactly {want} time(s) in {path}: {old[:60]!r} ({s.count(old)})")
    at = -1
    for _ in range(idx + 1):
        at = s.index(old, at + 1)
    s = s[:at] + new + s[at + len(old):]
open(p, "w").write(s)
print(f"applied {name} to {p}")
