// Flash-style causal prefill attention on simdgroup_matrix (Metal 3.1), GQA-shared K/V tiles.
// Launcher: crates/intarsia-backend-metal/src/flash.rs. Same bindings and OpP fields as `attn_prefill_bf16`
// (attention.metal): 0 q [T,N,D], 1 k cache [Nkv,cap,D], 2 v cache, 3 out [T,N,D], 4 OpP (a = T, b = N, c = Nkv, d = D,
// e = cap, f = pos0, f0 = scale). Needs 16-byte aligned operands with no leading skip (the launcher falls back to
// `attn_prefill_bf16` otherwise). The query rows are positions pos0 .. pos0 + T of a sequence whose keys 0 .. pos0 + T
// are in the cache (chunked prefill: the keys of earlier chunks plus the current one); row i sees keys 0 ..= pos0 + i.
//
// Work split. A threadgroup is NSG simdgroups and owns one kv head and a block of BQ = 8 * PQ query positions, where
// PQ = NSG / G and G = N / Nkv is the group size (1, 2, 4 or 8). Simdgroup `sg` is one query head of the group
// (hg = sg / PQ) and 8 consecutive positions (pq = sg % PQ), so the K/V tile staged in threadgroup memory is used by all G
// heads of the group and by every position slab: the K/V traffic is 1 / (G * PQ) of a per-(head, row) kernel's. Blocks
// are issued longest first (the last blocks of a chunk see the most keys).
//
// Per key tile of BK keys (tiles start at absolute key 0, never at the query offset, so a row's arithmetic does not depend
// on how the prompt is cut into chunks): S = Q.K^T on 8x8 simdgroup_matrix fragments with float accumulate (Q stays in
// registers, K comes from threadgroup memory through a transposed simdgroup_load), scale and causal mask in float, online
// softmax in the log2 domain with the row max / sum kept in the fragment layout (a row lives on the four lanes that differ
// in bits 0 and 3 of the lane id, hence the xor-1 / xor-8 shuffles), P rounded to bf16 straight from the S fragment
// registers (the C and A fragment layouts coincide), O += P.V. The next tile's global loads are issued before the
// current tile's compute and written to threadgroup memory after the next barrier (register prefetch; one tile of
// threadgroup memory, no double buffering). Reference for the fragment layout and the loop structure: mlx steel attention
// (mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h and attn/mma.h).
//
// Masking assigns -inf (never adds), so a fully masked tile leaves a row's max, sum and accumulator bitwise unchanged:
// block membership and the number of extra tiles a block walks do not change a row's result. Key rows at or beyond
// pos0 + T are staged as zeros (cache rows there may hold anything, and 0 * NaN would poison P.V).
//
// Precision: Q, K, V are the bf16 inputs themselves (no conversion, so no range limit), products and sums are float, P is
// rounded to bf16 (8 bits; the l sum uses the unrounded float P). A half-operand form (11-bit P, but |x| <= 65504)
// measured 6 percent slower and no more accurate on the goldens, see docs/results/2026-10-14-metal-prefill-p2-flash.md.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "common.h"
using namespace metal;

#define FL_UNROLL _Pragma("clang loop unroll(full)")

// The global loads of one tile (BK rows of D bf16, as 16-byte chunks spread over NT threads): rows at or beyond `limit`
// are zeros.
template <uint D, uint BK, uint NT>
METAL_FUNC void fl_load(thread uint4* raw, device const ushort* base, ulong key0, ulong limit, uint tid) {
    constexpr uint CH = D / 8;
    FL_UNROLL
    for (uint n = 0; n < (BK * CH) / NT; ++n) {
        const uint idx = tid + n * NT;
        const ulong key = key0 + idx / CH;
        raw[n] = key < limit ? ((device const uint4*)(base + key * D))[idx % CH] : uint4(0);
    }
}

// ... and their stores into threadgroup memory (row stride D + 8 bf16 = 16 bytes of padding)
template <uint D, uint BK, uint NT>
METAL_FUNC void fl_put(threadgroup bfloat* dst, thread const uint4* raw, uint tid) {
    constexpr uint CH = D / 8;
    constexpr uint LD = D + 8;
    FL_UNROLL
    for (uint n = 0; n < (BK * CH) / NT; ++n) {
        const uint idx = tid + n * NT;
        *(threadgroup uint4*)(dst + (idx / CH) * LD + (idx % CH) * 8) = raw[n];
    }
}

template <uint D, uint BK, uint NSG>
kernel void attn_flash(device const ushort* q [[buffer(0)]],
                       device const ushort* kc [[buffer(1)]],
                       device const ushort* vc [[buffer(2)]],
                       device ushort* out [[buffer(3)]],
                       constant OpP& p [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 grid [[threadgroups_per_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint LD = D + 8;
    constexpr uint TD = D / 8;
    constexpr uint TK = BK / 8;
    constexpr uint NT = NSG * 32;
    constexpr uint PFN = (BK * (D / 8)) / NT;  // 16-byte chunks per thread and tile, K and V each
    static_assert((BK * (D / 8)) % NT == 0 && PFN > 0, "a tile must be a whole number of chunks per thread");

    threadgroup bfloat Ks[BK * LD];
    threadgroup bfloat Vs[BK * LD];

    const uint T = p.a, N = p.b, Nkv = p.c;
    const ulong cap = p.e, pos0 = p.f;
    const uint G = N / Nkv;
    const uint PQ = NSG / G;
    const uint BQ = 8 * PQ;
    const uint qb = grid.x - 1 - tg.x;  // longest blocks first
    const uint kvh = tg.y;
    const uint hg = sg / PQ, pq = sg % PQ;
    const uint head = kvh * G + hg;
    const bool live = hg < G;  // group sizes that do not divide NSG (3, 5, 6, 7): the surplus simdgroups load and barrier but own no head
    const uint qbase = qb * BQ + pq * 8;  // first position (within the call) of this simdgroup
    const ulong kv_len = min(pos0 + T, cap);
    const ulong kmax = min(kv_len, pos0 + min((ulong)T, (ulong)(qb + 1) * BQ));  // keys the block's last row sees
    const uint nkt = (uint)((kmax + BK - 1) / BK);
    const ulong first_row = pos0 + (ulong)qb * BQ;  // smallest absolute row position of the block

    // fragment coordinates of this lane (mlx BaseMMAFrag::get_coord): row fm, columns fn and fn + 1
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;

    device const ushort* kb = kc + (ulong)kvh * cap * D;
    device const ushort* vb = vc + (ulong)kvh * cap * D;

    const uint pos = qbase + fm;  // this lane's query row (within the call)
    simdgroup_matrix<bfloat, 8, 8> Qf[TD];
    FL_UNROLL
    for (uint dd = 0; dd < TD; ++dd) {
        uint w = 0;
        if (pos < T && live) w = *(device const uint*)(q + ((ulong)pos * N + head) * D + dd * 8 + fn);
        Qf[dd].thread_elements()[0] = as_type<bfloat>((ushort)(w & 0xFFFF));
        Qf[dd].thread_elements()[1] = as_type<bfloat>((ushort)(w >> 16));
    }

    simdgroup_matrix<float, 8, 8> O[TD];
    FL_UNROLL
    for (uint id = 0; id < TD; ++id) O[id] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;
    const float c = p.f0 * 1.44269504088896340736f;  // scale * log2(e)

    uint4 rk[PFN], rv[PFN];
    fl_load<D, BK, NT>(rk, kb, 0, kv_len, tid);
    fl_load<D, BK, NT>(rv, vb, 0, kv_len, tid);
    for (uint kt = 0; kt < nkt; ++kt) {
        const ulong key0 = (ulong)kt * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);  // everyone is done with the previous tile
        fl_put<D, BK, NT>(Ks, rk, tid);
        fl_put<D, BK, NT>(Vs, rv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kt + 1 < nkt) {  // prefetch: the latency hides behind this tile's compute
            fl_load<D, BK, NT>(rk, kb, key0 + BK, kv_len, tid);
            fl_load<D, BK, NT>(rv, vb, key0 + BK, kv_len, tid);
        }

        // S = Q.K^T
        simdgroup_matrix<float, 8, 8> S[TK];
        FL_UNROLL
        for (uint j = 0; j < TK; ++j) S[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        FL_UNROLL
        for (uint dd = 0; dd < TD; ++dd) {
            simdgroup_matrix<bfloat, 8, 8> kf[TK];
            FL_UNROLL
            for (uint j = 0; j < TK; ++j) simdgroup_load(kf[j], Ks + (j * 8) * LD + dd * 8, LD, ulong2(0, 0), true);
            FL_UNROLL
            for (uint j = 0; j < TK; ++j) simdgroup_multiply_accumulate(S[j], Qf[dd], kf[j], S[j]);
        }

        // scale, causal mask (only tiles that reach past the block's first row), online softmax
        const bool masked = key0 + BK - 1 > first_row;
        const ulong row = pos0 + pos;
        float mx = -INFINITY;
        FL_UNROLL
        for (uint j = 0; j < TK; ++j) {
            FL_UNROLL
            for (uint e = 0; e < 2; ++e) {
                float s = S[j].thread_elements()[e] * c;
                if (masked && key0 + j * 8 + fn + e > row) s = -INFINITY;
                S[j].thread_elements()[e] = s;
                mx = max(mx, s);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
        mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
        const float m_new = max(m, mx);
        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
        const float factor = fast::exp2(m - m_use);
        m = m_new;
        float ls = 0.0f;
        simdgroup_matrix<bfloat, 8, 8> Pf[TK];
        FL_UNROLL
        for (uint j = 0; j < TK; ++j) {
            const float p0 = fast::exp2(S[j].thread_elements()[0] - m_use);
            const float p1 = fast::exp2(S[j].thread_elements()[1] - m_use);
            ls += p0 + p1;
            Pf[j].thread_elements()[0] = bfloat(p0);
            Pf[j].thread_elements()[1] = bfloat(p1);
        }
        l = l * factor + ls;  // per-lane partial; the four lanes of a row are summed once at the end
        FL_UNROLL
        for (uint id = 0; id < TD; ++id) {
            O[id].thread_elements()[0] *= factor;
            O[id].thread_elements()[1] *= factor;
        }

        // O += P.V
        FL_UNROLL
        for (uint kk = 0; kk < TK; ++kk) {
            FL_UNROLL
            for (uint id = 0; id < TD; ++id) {
                simdgroup_matrix<bfloat, 8, 8> vf;
                simdgroup_load(vf, Vs + (kk * 8) * LD + id * 8, LD);
                simdgroup_multiply_accumulate(O[id], Pf[kk], vf, O[id]);
            }
        }
    }

    // normalise and store
    l += simd_shuffle_xor(l, ushort(1));
    l += simd_shuffle_xor(l, ushort(8));
    const float inv = 1.0f / l;
    if (pos < T && live) {
        device uint* dst = (device uint*)(out + ((ulong)pos * N + head) * D + fn);
        FL_UNROLL
        for (uint id = 0; id < TD; ++id) {
            const ushort lo = f2bf(O[id].thread_elements()[0] * inv), hi = f2bf(O[id].thread_elements()[1] * inv);
            dst[id * 4] = (uint)lo | ((uint)hi << 16);
        }
    }
}

#define FLASH_INST(NAME, D, BK, NSG)                                                                                   \
    template [[host_name(NAME)]] kernel void attn_flash<D, BK, NSG>(                                                    \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, constant OpP&, uint2, uint2,  \
        uint, uint, uint);

// per head_dim: NSG = 4 serves groups 1, 2, 4; NSG = 8 serves group 8 (and is selectable for the others)
FLASH_INST("attn_flash_d128_s4", 128, 24, 4)
FLASH_INST("attn_flash_d128_s8", 128, 16, 8)
FLASH_INST("attn_flash_d64_s4", 64, 16, 4)
FLASH_INST("attn_flash_d64_s8", 64, 32, 8)
FLASH_INST("attn_flash_d256_s4", 256, 16, 4)
FLASH_INST("attn_flash_d256_s8", 256, 16, 8)
