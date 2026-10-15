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
template <bool TB, int ABL, typename AT, typename BT>
METAL_FUNC void nax_mma(thread frag_t<float>& C0, thread frag_t<float>& C1, thread const frag_t<AT>& A,
                        thread const frag_t<BT>& B0, thread const frag_t<BT>& B1) {
    if constexpr (ABL & 8) {
        FN_UNROLL
        for (short i = 0; i < 8; ++i) {
            C0[i] += float(A[i]) * float(B0[i]);
            C1[i] += float(A[i]) * float(B1[i]);
        }
        return;
    }
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
template <int ABL>
METAL_FUNC uint4 ld16(device const ushort* p, bool ok) {
    if constexpr (ABL & 4) return uint4(as_type<uint>((float)((ulong)p & 0xFFFF)));
    if constexpr (ABL & 16) {
        // hot window: fold the address into 16 KB around the base so everything hits the L1
        p = (device const ushort*)((ulong)p & ~(ulong)0x3E00);
    }
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

template <int D, int BK, int NSG, int ABL>
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
        const uint4 a = ld16<ABL>(q + ((ulong)pos0r * N + head) * D + 32 * dp + 8 * cq, pos0r < T);
        const uint4 b = ld16<ABL>(q + ((ulong)pos1r * N + head) * D + 32 * dp + 8 * cq, pos1r < T);
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
                const uint4 a0 = ld16<ABL>(kb + ka * D + 32 * dp, ka < kv_len);
                const uint4 a1 = ld16<ABL>(kb + (ka + 8) * D + 32 * dp, ka + 8 < kv_len);
                const uint4 b0 = ld16<ABL>(kb + kb2 * D + 32 * dp, kb2 < kv_len);
                const uint4 b1 = ld16<ABL>(kb + (kb2 + 8) * D + 32 * dp, kb2 + 8 < kv_len);
                frag_t<bfloat> k0l, k0h, k1l, k1h;
                split16(a0, a1, k0l, k0h);
                split16(b0, b1, k1l, k1h);
                nax_mma<true, ABL>(S[2 * kp], S[2 * kp + 1], Qf[2 * dp], k0l, k1l);
                nax_mma<true, ABL>(S[2 * kp], S[2 * kp + 1], Qf[2 * dp + 1], k0h, k1h);
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
                if (!(ABL & 2) && masked && key > ((e & 4) ? row1 : row0)) s = -INFINITY;
                S[f][e] = s;
                if (e & 4) mx1 = max(mx1, s); else mx0 = max(mx0, s);
            }
        }
        if (!(ABL & 1)) {
            mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(1)));
            mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(8)));
            mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(1)));
            mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(8)));
        }
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
                const float pv = (ABL & 1) ? S[f][e] : fast::exp2(S[f][e] - ((e & 4) ? mu1 : mu0));
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
                const uint4 a = ld16<ABL>(vb + va * D + 32 * dp, va < kv_len);
                const uint4 b = ld16<ABL>(vb + (va + 8) * D + 32 * dp, va + 8 < kv_len);
                frag_t<bfloat> vl, vh;
                split16(a, b, vl, vh);
                nax_mma<false, ABL>(O[2 * dp], O[2 * dp + 1], Pf[kk], vl, vh);
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

// ---------------------------------------------------------------------------------------------------------------------
// Threadgroup-staged variant: the threadgroup copies each K / V tile with coalesced 16-byte loads (register prefetch one
// tile ahead, as attention_flash.metal) into padded threadgroup memory, and the lanes read their 16-byte fragment
// pieces from there.

template <int ABL>
METAL_FUNC uint4 ldt(threadgroup const ushort* p) {
    if constexpr (ABL & 4) return uint4(as_type<uint>((float)((uint)(ulong)p & 0xFFFF)));
    return *(threadgroup const uint4*)p;
}

template <uint D, uint BK, uint NT>
METAL_FUNC void tg_load(thread uint4* raw, device const ushort* base, ulong key0, ulong limit, uint tid) {
    constexpr uint CH = D / 8;
    FN_UNROLL
    for (uint n = 0; n < (BK * CH) / NT; ++n) {
        const uint idx = tid + n * NT;
        const ulong key = key0 + idx / CH;
        raw[n] = key < limit ? ((device const uint4*)(base + key * D))[idx % CH] : uint4(0);
    }
}

template <uint D, uint BK, uint NT, uint LD>
METAL_FUNC void tg_put(threadgroup ushort* dst, thread const uint4* raw, uint tid) {
    constexpr uint CH = D / 8;
    FN_UNROLL
    for (uint n = 0; n < (BK * CH) / NT; ++n) {
        const uint idx = tid + n * NT;
        *(threadgroup uint4*)(dst + (idx / CH) * LD + (idx % CH) * 8) = raw[n];
    }
}

template <int D, int BK, int NSG, int PAD, int ABL>
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void attn_nax_tg(
    device const ushort* q [[buffer(0)]],
    device const ushort* kc [[buffer(1)]],
    device const ushort* vc [[buffer(2)]],
    device ushort* out [[buffer(3)]],
    constant OpP& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr int TD = D / 16;
    constexpr int TK = BK / 16;
    constexpr int DP = D / 32;
    constexpr uint LD = D + PAD;
    constexpr uint NT = NSG * 32;
    constexpr uint PFN = (BK * (D / 8)) / NT;
    static_assert(BK % 32 == 0 && D % 32 == 0 && (BK * (D / 8)) % NT == 0 && PFN > 0, "tile shape");

    threadgroup ushort Ks[BK * LD];
    threadgroup ushort Vs[BK * LD];

    const uint T = p.a, N = p.b, Nkv = p.c;
    const ulong cap = p.e, pos0 = p.f;
    const uint G = N / Nkv;
    const uint PQ = NSG / G;
    const uint BQ = 16 * PQ;
    const uint qb = grid.x - 1 - tg.x;
    const uint kvh = tg.y;
    const uint hg = sg / PQ, pq = sg % PQ;
    const uint head = kvh * G + hg;
    const uint qbase = qb * BQ + pq * 16;
    const ulong kv_len = min(pos0 + T, cap);
    const ulong kmax = min(kv_len, pos0 + min((ulong)T, (ulong)(qb + 1) * BQ));
    const uint nkt = (uint)((kmax + BK - 1) / BK);

    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint cq = (qid & 2) + (lane & 1);

    device const ushort* kb = kc + (ulong)kvh * cap * D;
    device const ushort* vb = vc + (ulong)kvh * cap * D;

    const uint pos0r = qbase + fm, pos1r = qbase + fm + 8;
    frag_t<bfloat> Qf[TD];
    FN_UNROLL
    for (int dp = 0; dp < DP; ++dp) {
        const uint4 a = ld16<0>(q + ((ulong)pos0r * N + head) * D + 32 * dp + 8 * cq, pos0r < T);
        const uint4 b = ld16<0>(q + ((ulong)pos1r * N + head) * D + 32 * dp + 8 * cq, pos1r < T);
        split16(a, b, Qf[2 * dp], Qf[2 * dp + 1]);
    }

    frag_t<float> O[TD];
    FN_UNROLL
    for (int i = 0; i < TD; ++i) O[i] = frag_t<float>(0.0f);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float c = p.f0 * 1.44269504088896340736f;
    const ulong row0 = pos0 + pos0r, row1 = pos0 + pos1r;
    const ulong first_row = pos0 + qbase;

    uint4 rk[PFN], rv[PFN];
    tg_load<D, BK, NT>(rk, kb, 0, kv_len, tid);
    tg_load<D, BK, NT>(rv, vb, 0, kv_len, tid);
    for (uint kt = 0; kt < nkt; ++kt) {
        const ulong key0 = (ulong)kt * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (!(ABL & 32)) {
            tg_put<D, BK, NT, LD>(Ks, rk, tid);
            tg_put<D, BK, NT, LD>(Vs, rv, tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!(ABL & 32) && kt + 1 < nkt) {
            tg_load<D, BK, NT>(rk, kb, key0 + BK, kv_len, tid);
            tg_load<D, BK, NT>(rv, vb, key0 + BK, kv_len, tid);
        }

        frag_t<float> S[TK];
        FN_UNROLL
        for (int j = 0; j < TK; ++j) S[j] = frag_t<float>(0.0f);
        FN_UNROLL
        for (int kp = 0; kp < TK / 2; ++kp) {
            const uint ra = 32 * kp + fm, rb = ra + 16;
            FN_UNROLL
            for (int dp = 0; dp < DP; ++dp) {
                const uint co = 32 * dp + 8 * cq;
                const uint4 a0 = ldt<ABL>(Ks + ra * LD + co);
                const uint4 a1 = ldt<ABL>(Ks + (ra + 8) * LD + co);
                const uint4 b0 = ldt<ABL>(Ks + rb * LD + co);
                const uint4 b1 = ldt<ABL>(Ks + (rb + 8) * LD + co);
                frag_t<bfloat> k0l, k0h, k1l, k1h;
                split16(a0, a1, k0l, k0h);
                split16(b0, b1, k1l, k1h);
                nax_mma<true, ABL>(S[2 * kp], S[2 * kp + 1], Qf[2 * dp], k0l, k1l);
                nax_mma<true, ABL>(S[2 * kp], S[2 * kp + 1], Qf[2 * dp + 1], k0h, k1h);
            }
        }

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
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        FN_UNROLL
        for (int i = 0; i < TD; ++i) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) O[i][e] *= (e & 4) ? f1 : f0;
        }

        FN_UNROLL
        for (int kk = 0; kk < TK; ++kk) {
            const uint rv0 = 16 * kk + fm;
            FN_UNROLL
            for (int dp = 0; dp < DP; ++dp) {
                const uint co = 32 * dp + 8 * cq;
                const uint4 a = ldt<ABL>(Vs + rv0 * LD + co);
                const uint4 b = ldt<ABL>(Vs + (rv0 + 8) * LD + co);
                frag_t<bfloat> vl, vh;
                split16(a, b, vl, vh);
                nax_mma<false, ABL>(O[2 * dp], O[2 * dp + 1], Pf[kk], vl, vh);
            }
        }
    }

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

#define NAX_TG_INST(NAME, D, BK, NSG, PAD, ABL)                                                                       \
    template [[host_name(NAME)]] kernel void attn_nax_tg<D, BK, NSG, PAD, ABL>(                                       \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, constant OpP&, uint2, uint2, \
        uint, uint, uint);

NAX_TG_INST("attn_nax_tg_s4_p8", 128, 32, 4, 8, 0)
NAX_TG_INST("attn_nax_tg_s4_p16", 128, 32, 4, 16, 0)
NAX_TG_INST("attn_nax_tg_s8_p8", 128, 32, 8, 8, 0)
NAX_TG_INST("attn_nax_tg_s8_p16", 128, 32, 8, 16, 0)
NAX_TG_INST("attn_nax_tgabl4", 128, 32, 8, 8, 4)
NAX_TG_INST("attn_nax_tgabl32", 128, 32, 8, 8, 32)
NAX_TG_INST("attn_nax_tgabl36", 128, 32, 8, 8, 36)
NAX_TG_INST("attn_nax_tgabl44", 128, 32, 8, 8, 44)

// ---------------------------------------------------------------------------------------------------------------------
// Cooperative-tensor-load variant: K and V fragments are loaded by the primitive's own cooperative_tensor::load from
// bounds-checked device tensor slices (natural head-dim order), Q fragments by the lane layout of MLX's BaseNAXFrag.

template <bool TB, typename AT, typename TS>
METAL_FUNC void nax_mma_ct(thread frag_t<float>& C0, thread frag_t<float>& C1, thread const frag_t<AT>& A, TS src) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, RELAXED, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, bfloat, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    ct_b.load(src);
    FN_UNROLL
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = A[i];
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

// 4 bf16 (8 bytes) of each of two rows into one fragment (elements 0..3 row r0, 4..7 row r1); zeros when !ok
METAL_FUNC frag_t<bfloat> ld_frag(device const ushort* r0, bool ok0, device const ushort* r1, bool ok1) {
    const uint2 a = ok0 ? *(device const uint2*)r0 : uint2(0);
    const uint2 b = ok1 ? *(device const uint2*)r1 : uint2(0);
    const bfloat4 al = as_type<bfloat4>(a), bl = as_type<bfloat4>(b);
    frag_t<bfloat> f;
    FN_UNROLL
    for (short j = 0; j < 4; ++j) {
        f[j] = al[j];
        f[4 + j] = bl[j];
    }
    return f;
}

template <int D, int BK, int NSG>
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void attn_nax_ct(
    device const ushort* q [[buffer(0)]],
    device const ushort* kc [[buffer(1)]],
    device const ushort* vc [[buffer(2)]],
    device ushort* out [[buffer(3)]],
    constant OpP& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int TD = D / 16;
    constexpr int TK = BK / 16;
    static_assert(BK % 32 == 0 && D % 32 == 0, "tile shape");

    const uint T = p.a, N = p.b, Nkv = p.c;
    const ulong cap = p.e, pos0 = p.f;
    const uint G = N / Nkv;
    const uint PQ = NSG / G;
    const uint BQ = 16 * PQ;
    const uint qb = grid.x - 1 - tg.x;
    const uint kvh = tg.y;
    const uint hg = sg / PQ, pq = sg % PQ;
    const uint head = kvh * G + hg;
    const uint qbase = qb * BQ + pq * 16;
    const ulong kv_len = min(pos0 + T, cap);
    const ulong kmax = min(kv_len, pos0 + min((ulong)T, (ulong)(qb + 1) * BQ));
    const uint nkt = (uint)((kmax + BK - 1) / BK);

    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint cq = (qid & 2) + (lane & 1);

    device bfloat* kb = (device bfloat*)(kc + (ulong)kvh * cap * D);
    device bfloat* vb = (device bfloat*)(vc + (ulong)kvh * cap * D);
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tK(kb, dextents<int, 2>(D, (int)kv_len));
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tV(vb, dextents<int, 2>(D, (int)kv_len));

    const uint pos0r = qbase + fm, pos1r = qbase + fm + 8;
    frag_t<bfloat> Qf[TD];
    FN_UNROLL
    for (int id = 0; id < TD; ++id) {
        const uint2 a = pos0r < T ? *(device const uint2*)(q + ((ulong)pos0r * N + head) * D + 16 * id + 4 * cq) : uint2(0);
        const uint2 b = pos1r < T ? *(device const uint2*)(q + ((ulong)pos1r * N + head) * D + 16 * id + 4 * cq) : uint2(0);
        const bfloat4 al = as_type<bfloat4>(a), bl = as_type<bfloat4>(b);
        FN_UNROLL
        for (short j = 0; j < 4; ++j) {
            Qf[id][j] = al[j];
            Qf[id][4 + j] = bl[j];
        }
    }

    frag_t<float> O[TD];
    FN_UNROLL
    for (int i = 0; i < TD; ++i) O[i] = frag_t<float>(0.0f);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float c = p.f0 * 1.44269504088896340736f;
    const ulong row0 = pos0 + pos0r, row1 = pos0 + pos1r;
    const ulong first_row = pos0 + qbase;

    for (uint kt = 0; kt < nkt; ++kt) {
        const int key0 = (int)((ulong)kt * BK);

        frag_t<float> S[TK];
        FN_UNROLL
        for (int j = 0; j < TK; ++j) S[j] = frag_t<float>(0.0f);
        // full tiles go through cooperative_tensor::load (no bounds check on the tensor slice: it must lie inside the
        // cache); the last, partial tile through bounds-checked lane loads (rows past kv_len are zeros)
        const bool full = (ulong)key0 + BK <= kv_len;
        if (full) {
            FN_UNROLL
            for (int kp = 0; kp < TK / 2; ++kp) {
                FN_UNROLL
                for (int id = 0; id < TD; ++id) {
                    auto sk = tK.template slice<16, 32>(16 * id, key0 + 32 * kp);
                    nax_mma_ct<true>(S[2 * kp], S[2 * kp + 1], Qf[id], sk);
                }
            }
        } else {
            FN_UNROLL
            for (int kp = 0; kp < TK / 2; ++kp) {
                const ulong ka = (ulong)key0 + 32 * kp + fm, kb2 = ka + 16;
                FN_UNROLL
                for (int id = 0; id < TD; ++id) {
                    const uint co = 16 * id + 4 * cq;
                    const frag_t<bfloat> b0 = ld_frag((device const ushort*)kb + ka * D + co, ka < kv_len,
                                                      (device const ushort*)kb + (ka + 8) * D + co, ka + 8 < kv_len);
                    const frag_t<bfloat> b1 = ld_frag((device const ushort*)kb + kb2 * D + co, kb2 < kv_len,
                                                      (device const ushort*)kb + (kb2 + 8) * D + co, kb2 + 8 < kv_len);
                    nax_mma<true, 0>(S[2 * kp], S[2 * kp + 1], Qf[id], b0, b1);
                }
            }
        }

        const bool masked = (ulong)key0 + BK - 1 > first_row;
        float mx0 = -INFINITY, mx1 = -INFINITY;
        FN_UNROLL
        for (int f = 0; f < TK; ++f) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) {
                float s = S[f][e] * c;
                const ulong key = (ulong)key0 + 16 * f + 4 * cq + (e & 3);
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
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        FN_UNROLL
        for (int i = 0; i < TD; ++i) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) O[i][e] *= (e & 4) ? f1 : f0;
        }

        if (full) {
            FN_UNROLL
            for (int kk = 0; kk < TK; ++kk) {
                FN_UNROLL
                for (int idp = 0; idp < TD / 2; ++idp) {
                    auto sv = tV.template slice<32, 16>(32 * idp, key0 + 16 * kk);
                    nax_mma_ct<false>(O[2 * idp], O[2 * idp + 1], Pf[kk], sv);
                }
            }
        } else {
            FN_UNROLL
            for (int kk = 0; kk < TK; ++kk) {
                const ulong va = (ulong)key0 + 16 * kk + fm;
                FN_UNROLL
                for (int idp = 0; idp < TD / 2; ++idp) {
                    const uint co = 32 * idp + 4 * cq;
                    const frag_t<bfloat> v0 = ld_frag((device const ushort*)vb + va * D + co, va < kv_len,
                                                      (device const ushort*)vb + (va + 8) * D + co, va + 8 < kv_len);
                    const frag_t<bfloat> v1 = ld_frag((device const ushort*)vb + va * D + co + 16, va < kv_len,
                                                      (device const ushort*)vb + (va + 8) * D + co + 16, va + 8 < kv_len);
                    nax_mma<false, 0>(O[2 * idp], O[2 * idp + 1], Pf[kk], v0, v1);
                }
            }
        }
    }

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
        device uint2* dst = (device uint2*)(out + ((ulong)pos * N + head) * D + 4 * cq);
        FN_UNROLL
        for (int id = 0; id < TD; ++id) {
            const ushort a = f2bf(O[id][4 * i] * inv), b = f2bf(O[id][4 * i + 1] * inv);
            const ushort c2 = f2bf(O[id][4 * i + 2] * inv), d2 = f2bf(O[id][4 * i + 3] * inv);
            dst[id * 4] = uint2((uint)a | ((uint)b << 16), (uint)c2 | ((uint)d2 << 16));
        }
    }
}

#define NAX_CT_INST(NAME, D, BK, NSG)                                                                                 \
    template [[host_name(NAME)]] kernel void attn_nax_ct<D, BK, NSG>(                                                 \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, constant OpP&, uint2, uint2, \
        uint, uint);

NAX_CT_INST("attn_nax_ct_s4_k32", 128, 32, 4)
NAX_CT_INST("attn_nax_ct_s8_k32", 128, 32, 8)
NAX_CT_INST("attn_nax_ct_s4_k64", 128, 64, 4)
NAX_CT_INST("attn_nax_ct_s8_k64", 128, 64, 8)
NAX_CT_INST("attn_nax_ct_s16_k32", 128, 32, 16)

#define NAX_FLASH_INST(NAME, D, BK, NSG, ABL)                                                                        \
    template [[host_name(NAME)]] kernel void attn_nax<D, BK, NSG, ABL>(                                               \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, constant OpP&, uint2, uint2, \
        uint, uint);

NAX_FLASH_INST("attn_nax_d128_s4_k32", 128, 32, 4, 0)
NAX_FLASH_INST("attn_nax_d128_s4_k64", 128, 64, 4, 0)
NAX_FLASH_INST("attn_nax_d128_s8_k32", 128, 32, 8, 0)
NAX_FLASH_INST("attn_nax_d128_s8_k64", 128, 64, 8, 0)
NAX_FLASH_INST("attn_nax_abl1", 128, 32, 4, 1)
NAX_FLASH_INST("attn_nax_abl2", 128, 32, 4, 2)
NAX_FLASH_INST("attn_nax_abl3", 128, 32, 4, 3)
NAX_FLASH_INST("attn_nax_abl4", 128, 32, 4, 4)
NAX_FLASH_INST("attn_nax_abl8", 128, 32, 4, 8)
NAX_FLASH_INST("attn_nax_abl12", 128, 32, 4, 12)
NAX_FLASH_INST("attn_nax_abl7", 128, 32, 4, 7)
NAX_FLASH_INST("attn_nax_abl15", 128, 32, 4, 15)
NAX_FLASH_INST("attn_nax_abl16", 128, 32, 4, 16)
NAX_FLASH_INST("attn_nax_abl24", 128, 32, 4, 24)
