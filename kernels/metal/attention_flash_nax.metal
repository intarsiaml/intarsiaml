// Flash-style causal prefill attention on the M5 neural accelerators (Metal 4, mpp::tensor_ops::matmul2d on
// simdgroup-scope cooperative tensors), GQA-shared work split. Compiled -std=metal4.0 into intarsia_metal4.metallib
// (build.rs METAL4_SOURCES). Launcher: crates/intarsia-backend-metal/src/flash_nax.rs. Same bindings and OpP fields as
// `attn_prefill_bf16_flash` (attention_flash.metal): 0 q [T,N,D], 1 k cache [Nkv,cap,D], 2 v cache, 3 out [T,N,D],
// 4 OpP (a = T, b = N, c = Nkv, d = D, e = cap, f = pos0, f0 = scale); 16-byte aligned operands with no leading skip.
// Query rows are positions pos0 .. pos0 + T, keys 0 .. pos0 + T are in the cache, row i sees keys 0 ..= pos0 + i.
//
// Structure after MLX's NAX attention (mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h and
// steel/attn/nax.h: 16x16 fragments, 8 elements per lane, matmul2d 16x32x16 on register-resident cooperative tensors,
// S = Q.K^T with transposed B, O += P.V): a simdgroup owns 16 query rows of one head and walks the key tiles; Q lives in
// registers, K and V fragments are read straight from device memory (no threadgroup staging; the simdgroups of a
// threadgroup that share a kv head read the same K / V lines, so the L1 serves all but the first).
//
// Lane layout (NAX fragment, MLX BaseNAXFrag::get_coord): lane = 4 * qid + r; row fm = (qid & 4) | ((lane >> 1) & 3) and
// fm + 8; columns 4 * cq .. 4 * cq + 3 with cq = (qid & 2) | (lane & 1). The contraction index of the A and B operands is
// the fragment column (A, B transposed) or row (B plain); an element permutation of the head dimension that is applied to
// both Q and K (S = Q.K^T contracts over d) or to V and the output columns (O's columns) does not change the result
// except by float summation order. Here the lane's 16 bytes (8 consecutive d) cover the same 4 columns of two
// consecutive 16-wide fragments: fragment pair p (32 d), lane column group cq holds d = 32 p + 8 cq + {0..3} in fragment
// 2p and d = 32 p + 8 cq + {4..7} in fragment 2p + 1, so every K / V / Q load and every output store is one 16-byte access.
//
// Online softmax in float, log2 domain, row max / sum in the fragment layout (a row lives on the four lanes that differ in
// bits 0 and 3 of the lane id). P is rounded to bf16 for the P.V product (the l sum uses the unrounded float P), as in
// attn_prefill_bf16_flash. Masking assigns -inf, so a fully masked tile leaves a row's max, sum and accumulator bitwise
// unchanged: block membership and extra tiles do not change a row's result (the matmul rows are independent). Key tiles
// start at absolute key 0, never at the query offset, so how a prompt is cut into chunks does not change a row's
// arithmetic. Key rows at or beyond pos0 + T are zeros (the cache there may hold anything and 0 * NaN poisons P.V).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace metal;

#define FN_UNROLL _Pragma("clang loop unroll(full)")
#ifndef RELAXED
#define RELAXED false
#endif
#ifndef PT
#define PT bfloat
#endif

template <typename T>
using frag_t = vec<T, 8>;

// C (two 16x16 fragments, a 16x32 tile) += A (16x16) . B (16x32 as two 16x16 fragments); TB: B fragments hold [n, k].
template <bool TB, typename AT, typename BT>
METAL_FUNC void nax_mma(thread frag_t<float>& C0, thread frag_t<float>& C1, thread const frag_t<AT>& A,
                        thread const frag_t<BT>& B0, thread const frag_t<BT>& B1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, RELAXED, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    FN_UNROLL
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = A[i];
        ct_b[i] = B0[i];
        ct_b[8 + i] = B1[i];
        ct_c[i] = C0[i];
        ct_c[8 + i] = C1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    FN_UNROLL
    for (short i = 0; i < 8; ++i) {
        C0[i] = ct_c[i];
        C1[i] = ct_c[8 + i];
    }
}

// 8 bf16 (16 bytes) at p, zeros when !ok
METAL_FUNC uint4 ld16(device const ushort* p, bool ok) {
    return ok ? *(device const uint4*)p : uint4(0);
}

// two frag_t<bfloat> from the 16-byte loads of rows fm (a) and fm + 8 (b): lo = d 0..3 of the lane's 8, hi = d 4..7
METAL_FUNC void split16(uint4 a, uint4 b, thread frag_t<bfloat>& lo, thread frag_t<bfloat>& hi) {
    const bfloat4 al = as_type<bfloat4>(a.xy), ah = as_type<bfloat4>(a.zw);
    const bfloat4 bl = as_type<bfloat4>(b.xy), bh = as_type<bfloat4>(b.zw);
    FN_UNROLL
    for (short j = 0; j < 4; ++j) {
        lo[j] = al[j];
        lo[4 + j] = bl[j];
        hi[j] = ah[j];
        hi[4 + j] = bh[j];
    }
}

template <int D, int BK, int NSG>
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void attn_nax(
    device const ushort* q [[buffer(0)]],
    device const ushort* kc [[buffer(1)]],
    device const ushort* vc [[buffer(2)]],
    device ushort* out [[buffer(3)]],
    constant OpP& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int TD = D / 16;   // d fragments
    constexpr int TK = BK / 16;  // key fragments per tile
    constexpr int DP = D / 32;   // d fragment pairs
    static_assert(BK % 32 == 0 && D % 32 == 0, "tile shape");

    const uint T = p.a, N = p.b, Nkv = p.c;
    const ulong cap = p.e, pos0 = p.f;
    const uint G = N / Nkv;
    const uint PQ = NSG / G;
    const uint BQ = 16 * PQ;
    const uint qb = grid.x - 1 - tg.x;  // longest blocks first
    const uint kvh = tg.y;
    const uint hg = sg / PQ, pq = sg % PQ;
    const uint head = kvh * G + hg;
    const uint qbase = qb * BQ + pq * 16;  // first position (within the call) of this simdgroup
    const ulong kv_len = min(pos0 + T, cap);
    const ulong kmax = min(kv_len, pos0 + min((ulong)T, (ulong)(qb + 1) * BQ));  // keys the block's last row sees
    const uint nkt = (uint)((kmax + BK - 1) / BK);

    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);  // rows fm and fm + 8
    const uint cq = (qid & 2) + (lane & 1);        // column group: d offset 8 * cq within a 32-wide pair

    device const ushort* kb = kc + (ulong)kvh * cap * D + 8 * cq;
    device const ushort* vb = vc + (ulong)kvh * cap * D + 8 * cq;

    // Q fragments in registers
    const uint pos0r = qbase + fm, pos1r = qbase + fm + 8;
    frag_t<bfloat> Qf[TD];
    FN_UNROLL
    for (int dp = 0; dp < DP; ++dp) {
        const uint4 a = ld16(q + ((ulong)pos0r * N + head) * D + 32 * dp + 8 * cq, pos0r < T);
        const uint4 b = ld16(q + ((ulong)pos1r * N + head) * D + 32 * dp + 8 * cq, pos1r < T);
        split16(a, b, Qf[2 * dp], Qf[2 * dp + 1]);
    }

    frag_t<float> O[TD];
    FN_UNROLL
    for (int i = 0; i < TD; ++i) O[i] = frag_t<float>(0.0f);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float c = p.f0 * 1.44269504088896340736f;  // scale * log2(e)
    const ulong row0 = pos0 + pos0r, row1 = pos0 + pos1r;
    const ulong first_row = pos0 + qbase;

    for (uint kt = 0; kt < nkt; ++kt) {
        const ulong key0 = (ulong)kt * BK;

        // S = Q.K^T
        frag_t<float> S[TK];
        FN_UNROLL
        for (int j = 0; j < TK; ++j) S[j] = frag_t<float>(0.0f);
        FN_UNROLL
        for (int kp = 0; kp < TK / 2; ++kp) {
            const ulong ka = key0 + 32 * kp + fm, kb2 = ka + 16;
            FN_UNROLL
            for (int dp = 0; dp < DP; ++dp) {
                const uint4 a0 = ld16(kb + ka * D + 32 * dp, ka < kv_len);
                const uint4 a1 = ld16(kb + (ka + 8) * D + 32 * dp, ka + 8 < kv_len);
                const uint4 b0 = ld16(kb + kb2 * D + 32 * dp, kb2 < kv_len);
                const uint4 b1 = ld16(kb + (kb2 + 8) * D + 32 * dp, kb2 + 8 < kv_len);
                frag_t<bfloat> k0l, k0h, k1l, k1h;
                split16(a0, a1, k0l, k0h);
                split16(b0, b1, k1l, k1h);
                nax_mma<true>(S[2 * kp], S[2 * kp + 1], Qf[2 * dp], k0l, k1l);
                nax_mma<true>(S[2 * kp], S[2 * kp + 1], Qf[2 * dp + 1], k0h, k1h);
            }
        }

        // scale, causal mask, online softmax. Element i * 4 + j of fragment f: row (i ? fm + 8 : fm), key 16 f + 4 cq + j.
        const bool masked = key0 + BK - 1 > first_row;
        float mx0 = -INFINITY, mx1 = -INFINITY;
        FN_UNROLL
        for (int f = 0; f < TK; ++f) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) {
                float s = S[f][e] * c;
                const ulong key = key0 + 16 * f + 4 * cq + (e & 3);
                if (masked && key > ((e & 4) ? row1 : row0)) s = -INFINITY;
                S[f][e] = s;
                if (e & 4) mx1 = max(mx1, s); else mx0 = max(mx0, s);
            }
        }
        mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(1)));
        mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(8)));
        mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(1)));
        mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(8)));
        const float mn0 = max(m0, mx0), mn1 = max(m1, mx1);
        const float mu0 = mn0 == -INFINITY ? 0.0f : mn0, mu1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float f0 = fast::exp2(m0 - mu0), f1 = fast::exp2(m1 - mu1);
        m0 = mn0;
        m1 = mn1;
        float ls0 = 0.0f, ls1 = 0.0f;
        frag_t<PT> Pf[TK];
        FN_UNROLL
        for (int f = 0; f < TK; ++f) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) {
                const float pv = fast::exp2(S[f][e] - ((e & 4) ? mu1 : mu0));
                if (e & 4) ls1 += pv; else ls0 += pv;
                Pf[f][e] = PT(pv);
            }
        }
        l0 = l0 * f0 + ls0;  // per-lane partials; the four lanes of a row are summed once at the end
        l1 = l1 * f1 + ls1;
        FN_UNROLL
        for (int i = 0; i < TD; ++i) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) O[i][e] *= (e & 4) ? f1 : f0;
        }

        // O += P.V
        FN_UNROLL
        for (int kk = 0; kk < TK; ++kk) {
            const ulong va = key0 + 16 * kk + fm;
            FN_UNROLL
            for (int dp = 0; dp < DP; ++dp) {
                const uint4 a = ld16(vb + va * D + 32 * dp, va < kv_len);
                const uint4 b = ld16(vb + (va + 8) * D + 32 * dp, va + 8 < kv_len);
                frag_t<bfloat> vl, vh;
                split16(a, b, vl, vh);
                nax_mma<false>(O[2 * dp], O[2 * dp + 1], Pf[kk], vl, vh);
            }
        }
    }

    // normalise and store
    l0 += simd_shuffle_xor(l0, ushort(1));
    l0 += simd_shuffle_xor(l0, ushort(8));
    l1 += simd_shuffle_xor(l1, ushort(1));
    l1 += simd_shuffle_xor(l1, ushort(8));
    const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    FN_UNROLL
    for (int i = 0; i < 2; ++i) {
        const uint pos = i ? pos1r : pos0r;
        if (pos >= T) continue;
        const float inv = i ? inv1 : inv0;
        device uint4* dst = (device uint4*)(out + ((ulong)pos * N + head) * D + 8 * cq);
        FN_UNROLL
        for (int dp = 0; dp < DP; ++dp) {
            uint w[4];
            FN_UNROLL
            for (int j = 0; j < 2; ++j) {
                const ushort a = f2bf(O[2 * dp][4 * i + 2 * j] * inv), b = f2bf(O[2 * dp][4 * i + 2 * j + 1] * inv);
                const ushort c2 = f2bf(O[2 * dp + 1][4 * i + 2 * j] * inv), d2 = f2bf(O[2 * dp + 1][4 * i + 2 * j + 1] * inv);
                w[j] = (uint)a | ((uint)b << 16);
                w[2 + j] = (uint)c2 | ((uint)d2 << 16);
            }
            dst[dp * 4] = uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

#define NAX_FLASH_INST(NAME, D, BK, NSG)                                                                              \
    template [[host_name(NAME)]] kernel void attn_nax<D, BK, NSG>(                                                    \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, constant OpP&, uint2, uint2, \
        uint, uint);

NAX_FLASH_INST("attn_nax_d128_s4_k32", 128, 32, 4)
NAX_FLASH_INST("attn_nax_d128_s4_k64", 128, 64, 4)
NAX_FLASH_INST("attn_nax_d128_s8_k32", 128, 32, 8)
NAX_FLASH_INST("attn_nax_d128_s8_k64", 128, 64, 8)
