// Device top-k of each logits row (`Backend::topk_rows`, lane ink-mtp): the k largest values of a row in the sampler's order
// (value descending, ties by ascending index; `sampler::topk_row`), NaN never selected, a row holding a NaN flagged.
//
// Two stages. Stage 1: grid (C chunks, R rows), each threadgroup caches its chunk of the row in threadgroup memory and selects
// the chunk's top k by k rounds of "the best element after the previous pick" in the total order (no marking: the order is
// strict). Candidates go to scratch as (value, index) pairs; missing ones (a chunk with fewer than k non-NaN elements) are
// (-inf, 0xffffffff), last in the order. Stage 2: grid (1, R), the same selection over the row's C * k candidates.
//
// OpP: a = V, b = k, c = chunk length, d = chunks C. s0 = skip of the logits (elements).

#include "common.h"

#define TOPK_TG 256
#define TOPK_CHUNK_MAX 7168  // 28 KiB of f32 in threadgroup memory
#define TOPK_CAND_MAX 2048   // C * k <= 32 * 64

inline float topk_ld(device const float* l, ulong i) { return l[i]; }
inline float topk_ld(device const ushort* l, ulong i) { return bf2f(l[i]); }

inline bool topk_before(float av, uint ai, float bv, uint bi) { return av > bv || (av == bv && ai < bi); }

// The best (value, index) over the threadgroup of what each thread offers; every thread gets it back.
inline void topk_tg_best(thread float& bv, thread uint& bi, threadgroup float* pv, threadgroup uint* pi, uint lane, uint sg, uint nsg) {
    float m = simd_max(bv);
    uint mi = simd_min(bv == m ? bi : 0xffffffffu);
    if (lane == 0) { pv[sg] = m; pi[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = pv[0];
    uint gi = pi[0];
    for (uint j = 1; j < nsg; ++j) {
        if (topk_before(pv[j], pi[j], gm, gi)) { gm = pv[j]; gi = pi[j]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = gm;
    bi = gi;
}

template <typename T>
kernel void topk_stage1_t(device const T* logits [[buffer(0)]],
                          device float2* scratch [[buffer(1)]],
                          constant OpP& p [[buffer(2)]],
                          uint2 g [[threadgroup_position_in_grid]],
                          uint2 tid2 [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    const uint tid = tid2.x;
    threadgroup float buf[TOPK_CHUNK_MAX];
    threadgroup float pv[32];
    threadgroup uint pi[32];
    threadgroup uint nanf;
    const uint v = p.a, k = p.b, c = g.x, row = g.y;
    const uint lo = min(c * p.c, v), hi = min(lo + p.c, v), n = hi - lo;
    device const T* l = logits + p.s0 + (ulong)row * v + lo;
    if (tid == 0) nanf = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool has_nan = false;
    for (uint i = tid; i < n; i += TOPK_TG) {
        const float x = topk_ld(l, i);
        buf[i] = x;
        has_nan = has_nan || isnan(x);
    }
    if (has_nan) nanf = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float2* out = scratch + ((ulong)row * p.d + c) * k;
    float lv = INFINITY;
    uint li = 0;
    for (uint r = 0; r < k; ++r) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        for (uint i = tid; i < n; i += TOPK_TG) {
            const float x = buf[i];
            const uint ix = lo + i;
            const bool after = r == 0 ? !isnan(x) : topk_before(lv, li, x, ix);
            if (after && topk_before(x, ix, bv, bi)) { bv = x; bi = ix; }
        }
        topk_tg_best(bv, bi, pv, pi, lane, sg, nsg);
        if (tid == 0) out[r] = float2(bv, as_type<float>(bi));
        lv = bv;
        li = bi;
        if (bi == 0xffffffffu) {
            // nothing left in this chunk: fill the rest with the sentinel
            for (uint q = r + 1 + tid; q < k; q += TOPK_TG) out[q] = float2(-INFINITY, as_type<float>(0xffffffffu));
            break;
        }
    }
    if (tid == 0 && nanf != 0) out[0] = float2(NAN, as_type<float>(0xffffffffu));  // the row's NaN flag for stage 2
}

kernel void topk_stage2(device const float2* scratch [[buffer(0)]],
                        device int* ids [[buffer(1)]],
                        device float* vals [[buffer(2)]],
                        device int* nan_out [[buffer(3)]],
                        constant OpP& p [[buffer(4)]],
                        uint2 g [[threadgroup_position_in_grid]],
                        uint2 tid2 [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]]) {
    const uint tid = tid2.x;
    threadgroup float cv[TOPK_CAND_MAX];
    threadgroup uint ci[TOPK_CAND_MAX];
    threadgroup float pv[32];
    threadgroup uint pi[32];
    threadgroup uint nanf;
    const uint k = p.b, row = g.y, n = p.d * k;
    device const float2* s = scratch + (ulong)row * n;
    if (tid == 0) nanf = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool has_nan = false;
    for (uint i = tid; i < n; i += TOPK_TG) {
        const float2 e = s[i];
        has_nan = has_nan || isnan(e.x);
        cv[i] = isnan(e.x) ? -INFINITY : e.x;
        ci[i] = isnan(e.x) ? 0xffffffffu : as_type<uint>(e.y);
    }
    if (has_nan) nanf = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lv = INFINITY;
    uint li = 0;
    for (uint r = 0; r < k; ++r) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        for (uint i = tid; i < n; i += TOPK_TG) {
            const float x = cv[i];
            const uint ix = ci[i];
            const bool after = ix != 0xffffffffu && (r == 0 || topk_before(lv, li, x, ix));
            if (after && topk_before(x, ix, bv, bi)) { bv = x; bi = ix; }
        }
        topk_tg_best(bv, bi, pv, pi, lane, sg, nsg);
        if (tid == 0) {
            ids[(ulong)row * k + r] = bi == 0xffffffffu ? 0 : (int)bi;
            vals[(ulong)row * k + r] = bv;
        }
        lv = bv;
        li = bi;
    }
    if (tid == 0) nan_out[row] = nanf != 0 ? 1 : 0;
}

template [[host_name("topk_f32_stage1")]] kernel void topk_stage1_t<float>(
    device const float*, device float2*, constant OpP&, uint2, uint2, uint, uint, uint);
template [[host_name("topk_bf16_stage1")]] kernel void topk_stage1_t<ushort>(
    device const ushort*, device float2*, constant OpP&, uint2, uint2, uint, uint, uint);
