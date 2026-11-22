// Screened greedy head (lane ink-fp8): HeadScreen and ScreenArgmax. Contract and proof: crates/intarsia-core/src/screen.rs;
// cpu reference crates/intarsia-backend-cpu/src/screen_ref.rs (the core's screen_row / argmax_row); launcher
// crates/intarsia-backend-metal/src/screen.rs. LogitsGather is in gemv_f32out.metal (it must share gemv_f32o_v's arithmetic).
//
// HeadScreen uses exactly the contract's f32 operations (explicit fma, the 32-lane strided sum of squares and its xor tree),
// so its candidates and counts are bitwise the cpu reference's. One 1024-thread threadgroup per activation row.

#include "common.h"

struct ScP {
    uint v;
    uint d;
    uint c;
    uint pad;
};

constant float SC_WIDEN = 9.5367431640625e-7f;  // 2^-20
constant float SC_HN_MARGIN = 1.00048828125f;    // 1 + 2^-11
constant int SC_NONFINITE = 0x7fffffff;

inline float2 sc_lohi(float a, float hn, float b) {
    const float m = fma(hn, b, fabs(a));
    const float lo = fma(-SC_WIDEN, m, fma(-hn, b, a));
    const float hi = fma(SC_WIDEN, m, fma(hn, b, a));
    return float2(lo, hi);
}

kernel void head_screen_f32(device const float* A [[buffer(0)]],
                            device const ushort* H [[buffer(1)]],
                            device const float* beta [[buffer(2)]],
                            device int* cand [[buffer(3)]],
                            device int* cnt [[buffer(4)]],
                            constant ScP& p [[buffer(5)]],
                            uint t [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgs [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float s_hn;
    threadgroup float s_part[32];
    threadgroup uint s_bad[32];
    threadgroup uint s_cnt[32];
    device const float* a = A + (ulong)t * p.v;
    device const ushort* h = H + (ulong)t * p.d;
    if (sg == 0) {
        float q = 0.0f;
        for (uint k = lane; k < p.d; k += 32) {
            const float x = bf2f(h[k]);
            q = fma(x, x, q);
        }
        // the xor tree p_l += p_{l^16}, ^8, ^4, ^2, ^1 (addition commutes, so every lane holds the cpu's lane-0 value)
        q = q + simd_shuffle_xor(q, 16);
        q = q + simd_shuffle_xor(q, 8);
        q = q + simd_shuffle_xor(q, 4);
        q = q + simd_shuffle_xor(q, 2);
        q = q + simd_shuffle_xor(q, 1);
        if (lane == 0) s_hn = precise::sqrt(q) * SC_HN_MARGIN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float hn = s_hn;
    // pass 1: L = max lo, and any non-finite screen value
    float lmax = -INFINITY;
    uint bad = isfinite(hn) ? 0u : 1u;
    for (uint n = tid; n < p.v; n += tgs) {
        const float an = a[n];
        bad |= isfinite(an) ? 0u : 1u;
        lmax = max(lmax, sc_lohi(an, hn, beta[n]).x);
    }
    lmax = simd_max(lmax);
    bad = simd_max(bad);
    if (lane == 0) {
        s_part[sg] = lmax;
        s_bad[sg] = bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float L = -INFINITY;
    uint anybad = 0;
    for (uint j = 0; j < nsg; ++j) {
        L = max(L, s_part[j]);
        anybad |= s_bad[j];
    }
    device int* cr = cand + (ulong)t * p.c;
    if (anybad != 0 || !isfinite(L)) {
        for (uint j = tid; j < p.c; j += tgs) cr[j] = -1;
        if (tid == 0) cnt[t] = SC_NONFINITE;
        return;
    }
    // pass 2: count the candidates of this thread's contiguous chunk, prefix over the threads, write in ascending n
    const uint chunk = (p.v + tgs - 1) / tgs;
    const uint n0 = min(tid * chunk, p.v), n1 = min(n0 + chunk, p.v);
    uint mine = 0;
    for (uint n = n0; n < n1; ++n) mine += sc_lohi(a[n], hn, beta[n]).y >= L ? 1u : 0u;
    const uint pre = simd_prefix_exclusive_sum(mine);
    const uint tot = simd_sum(mine);
    if (lane == 0) s_cnt[sg] = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0, all = 0;
    for (uint j = 0; j < nsg; ++j) {
        if (j < sg) base += s_cnt[j];
        all += s_cnt[j];
    }
    uint pos = base + pre;
    for (uint j = all + tid; j < p.c; j += tgs) cr[j] = -1;  // slots past the candidates (contract: -1)
    if (mine != 0 && pos < p.c) {
        for (uint n = n0; n < n1 && pos < p.c; ++n) {
            if (sc_lohi(a[n], hn, beta[n]).y >= L) cr[pos++] = (int)n;
        }
    }
    if (tid == 0) cnt[t] = (int)min(all, (uint)(SC_NONFINITE - 1));
}

// ScreenArgmax: the lowest-index maximum over the candidates (cand ascending, so the lowest j) or over the full row, with
// argmax_t's comparisons (strict >, +-0 equal, NaN never wins). One 1024-thread threadgroup per row.
kernel void screen_argmax_f32(device const int* cand [[buffer(0)]],
                              device const int* cnt [[buffer(1)]],
                              device const float* Y [[buffer(2)]],
                              device int* tok [[buffer(3)]],
                              constant ScP& p [[buffer(4)]],
                              uint t [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tgs [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float pv[32];
    threadgroup uint pi[32];
    device const float* y = Y + (ulong)t * p.v;
    device const int* cr = cand + (ulong)t * p.c;
    const int ct = cnt[t];
    const bool path = ct >= 1 && (uint)ct <= p.c;
    const uint neff = path ? (uint)ct : p.v;
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint j = tid; j < neff; j += tgs) {
        const float x = y[j];
        if (x > bv) { bv = x; bi = path ? (uint)cr[j] : j; }
    }
    const float m = simd_max(bv);
    const uint mi = simd_min(bv == m ? bi : 0xffffffffu);
    if (lane == 0) { pv[sg] = m; pi[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float gm = -INFINITY;
        uint gi = 0xffffffffu;
        for (uint j = 0; j < nsg; ++j) {
            if (pv[j] > gm || (pv[j] == gm && pi[j] < gi)) { gm = pv[j]; gi = pi[j]; }
        }
        // all -inf / NaN: argmax_t's `0xffffffff`, the cpu's 0; the lowest index is 0 in both semantics' intent
        tok[t] = gi == 0xffffffffu ? (path ? cr[0] : 0) : (int)gi;
    }
}
