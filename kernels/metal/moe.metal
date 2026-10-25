// Sparse-MoE ops of Qwen3.6-35B-A3B (port of kernels/hip/moe.hip; launcher crates/intarsia-backend-metal/src/moe.rs).
// Semantics: crates/intarsia-core/src/quant/reference.rs (`moe_topk`, `expert_forward`, `moe_experts_token`, `moe_combine`).
//   moe_topk_{bf16,f32}     one simdgroup per token: K rounds of a (value desc, index asc) argmax, softmax over the
//                           selected (norm) or all logits; weights f32 rounded to the logits dtype, ids i32.
//   moe_combine_bf16        out = rb(routed + rb(rb(sigmoid(gate[t])) * shared)), one thread per element.
//   MoeExperts, NVFP4 experts (low nibble = even column, value = e2m1 * e4m3(block of 16) * global), bf16 x:
//     decode (T*K small)    moe_dec_gu: one simdgroup per (pair, row r): gate + up rows -> h[pair, r] (bf16 scratch);
//                           moe_dec_dn_*: one simdgroup per (token, out row): the K down rows in k order, weighted sum.
//     prefill (grouped)     moe_group: one threadgroup counts / prefix-sums / scatters the valid pairs by expert into
//                           tiles of BM (8 / 16 / 32 / 64, per call) rows; moe_mm_{gu,dn}_m<BM>: one expert tile x 64
//                           weight rows on bf16 simdgroup matrices (f32 accumulate), each weight block decoded once per
//                           tile into threadgroup memory; h by slot, y by pair;
//                           moe_gather_*: per token the weighted sum of its y rows in k order.
// Every reduction has a fixed order (the atomic slot order of the scatter changes no value): deterministic.
// Weights are read as packed_uint2 (4-byte alignment suffices); scales byte-wise.
// Kernel arguments: OpP (common.h); per kernel below.

#include "common.h"

constant float moe_fp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// e4m3 byte -> f32 (subnormals m * 2^-9; 0x7F / 0xFF NaN), as intarsia-core dtype::fp8_e4m3_to_f32
inline float moe_e4m3(uint b) {
    const uint e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 0xFu && m == 7u) v = as_type<float>(0x7FC00000u);
    else if (e == 0u) v = (float)m * 0.001953125f;  // m / 8 * 2^-6
    else v = as_type<float>(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}

inline float moe_rb(float v) { return bf2f(f2bf(v)); }

// sum_j x[j] * fp4(code j) over one 16-element block (8 bytes, element 2i in the low nibble of byte i)
inline float moe_blk_dot(uint2 wv, thread const float* xv) {
    float t = 0.f;
    for (int q = 0; q < 2; ++q) {
        const uint v = wv[q];
        for (int b = 0; b < 8; ++b) t = fma(moe_fp4[(v >> (4 * b)) & 0xFu], xv[q * 8 + b], t);
    }
    return t;
}

// the decode half of moe_blk_dot, once per weight block (the row-exact kernels share it across the pairs of an expert)
inline void moe_blk_dec(uint2 wv, thread float* wf) {
    for (int q = 0; q < 2; ++q) {
        const uint v = wv[q];
        for (int b = 0; b < 8; ++b) wf[q * 8 + b] = moe_fp4[(v >> (4 * b)) & 0xFu];
    }
}
// moe_blk_dot over decoded weights: the same fma chain in the same order, so bitwise the same value
inline float moe_blk_dot_f(thread const float* wf, thread const float* xv) {
    float t = 0.f;
    for (int i = 0; i < 16; ++i) t = fma(wf[i], xv[i], t);
    return t;
}

// ---- MoeTopk ------------------------------------------------------------------------------------------------------
// a = T, b = E (<= 512), c = K (<= 32), d = norm; s0 = logits skip. Buffers: logits, weights f32, ids i32, p.

constant uint MOE_TOPK_SG = 4;  // tokens (simdgroups) per threadgroup

inline bool moe_better(float v1, uint i1, float v2, uint i2) {
    if (i2 == 0xFFFFFFFFu) return i1 != 0xFFFFFFFFu;
    if (i1 == 0xFFFFFFFFu) return false;
    return v1 > v2 || (v1 == v2 && i1 < i2);
}

template <typename T>
inline float moe_ld(device const T* p, ulong i);
template <>
inline float moe_ld<ushort>(device const ushort* p, ulong i) { return bf2f(p[i]); }
template <>
inline float moe_ld<float>(device const float* p, ulong i) { return p[i]; }

template <typename T>
inline void moe_topk_impl(device const T* logits, device float* wout, device int* iout, constant OpP& p, uint tg, uint sg,
                          uint lane) {
    const uint t = tg * MOE_TOPK_SG + sg;
    if (t >= p.a) return;
    const uint E = p.b, K = p.c;
    device const T* row = logits + p.s0 + (ulong)t * E;
    float v[16];
    uint taken = 0u;
    for (uint j = 0; j < 16; ++j) {
        const uint e = lane + 32u * j;
        v[j] = e < E ? moe_ld<T>(row, e) : 0.f;
    }
    float sv[32];
    uint si[32];
    for (uint k = 0; k < K; ++k) {
        float bv = 0.f;
        uint bi = 0xFFFFFFFFu;
        for (uint j = 0; j < 16; ++j) {
            const uint e = lane + 32u * j;
            if (e < E && !(taken & (1u << j)) && moe_better(v[j], e, bv, bi)) {
                bv = v[j];
                bi = e;
            }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const uint oi = simd_shuffle_xor(bi, o);
            if (moe_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (bi % 32u == lane) taken |= 1u << (bi / 32u);
        sv[k] = bv;
        si[k] = bi;
    }
    const float mx = sv[0];
    float den = 0.f;
    if (p.d != 0u) {
        for (uint k = 0; k < K; ++k) den += exp(sv[k] - mx);
    } else {
        for (uint j = 0; j < 16; ++j) {
            if (lane + 32u * j < E) den += exp(v[j] - mx);
        }
        den = simd_sum(den);
    }
    for (uint k = lane; k < K; k += 32u) {
        float w = exp(sv[k] - mx) / den;
        if (sizeof(T) == 2) w = moe_rb(w);
        wout[(ulong)t * K + k] = w;
        iout[(ulong)t * K + k] = (int)si[k];
    }
}

#define MOE_TOPK(NAME, T)                                                                                           \
    kernel void NAME(device const T* logits [[buffer(0)]], device float* w [[buffer(1)]], device int* ids [[buffer(2)]], \
                     constant OpP& p [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],                       \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        moe_topk_impl<T>(logits, w, ids, p, tg, sg, lane);                                                          \
    }

MOE_TOPK(moe_topk_bf16, ushort)
MOE_TOPK(moe_topk_f32, float)

// ---- MoeCombine ---------------------------------------------------------------------------------------------------
// a = T, b = H; s0..s3 = routed, shared, gate, out skips.

kernel void moe_combine_bf16(device const ushort* routed [[buffer(0)]], device const ushort* shared [[buffer(1)]],
                             device const ushort* gate [[buffer(2)]], device ushort* out [[buffer(3)]],
                             constant OpP& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const ulong n = (ulong)p.a * p.b;
    if (i >= n) return;
    const uint t = i / p.b;
    const float g = bf2f(gate[p.s2 + t]);
    const float sg = moe_rb(1.f / (1.f + exp(-g)));
    const float s = moe_rb(sg * bf2f(shared[p.s1 + i]));
    out[p.s3 + i] = f2bf(bf2f(routed[p.s0 + i]) + s);
}

// ---- MoeExperts ---------------------------------------------------------------------------------------------------
// Common arguments: a = T, b = E, c = K, d = H, e = I; s0 = x skip, s1 = gate_up scales skip, s2 = down scales skip,
// s3 = out skip. gate_up w [E, 2I, H/2], gate_up s [E, 2I, H/16], down w [E, H, I/2], down s [E, H, I/16], gs [E, 3].

constant uint MOE_DEC_SG = 4;  // rows (simdgroups) per threadgroup of the decode kernels

// decode gate + up: grid (ceil(I / 4), T*K); h [T*K, I] bf16
kernel void moe_dec_gu(device const ushort* x [[buffer(0)]], device const int* ids [[buffer(1)]],
                       device const uchar* guw [[buffer(2)]], device const uchar* gus [[buffer(3)]],
                       device const float* gsc [[buffer(4)]], device ushort* h [[buffer(5)]], constant OpP& p [[buffer(6)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * MOE_DEC_SG + sg, pr = tg.y;
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    if (r >= I) return;
    const int id = ids[pr];
    if (id < 0 || (uint)id >= E) return;
    device const ushort* xr = x + p.s0 + (ulong)(pr / K) * H;
    const ulong rg = (ulong)id * 2 * I + r, ru = rg + I;
    device const packed_uint2* wg = (device const packed_uint2*)(guw + rg * (H / 2));
    device const packed_uint2* wu = (device const packed_uint2*)(guw + ru * (H / 2));
    device const uchar* sgp = gus + p.s1 + rg * (H / 16);
    device const uchar* sup = gus + p.s1 + ru * (H / 16);
    float ag = 0.f, au = 0.f;
    for (uint u = lane; u < H / 16; u += 32u) {
        float xv[16];
        for (int j = 0; j < 16; ++j) xv[j] = bf2f(xr[u * 16 + j]);
        ag = fma(moe_blk_dot(uint2(wg[u]), xv), moe_e4m3(sgp[u]), ag);
        au = fma(moe_blk_dot(uint2(wu[u]), xv), moe_e4m3(sup[u]), au);
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0) {
        const float g = moe_rb(ag * gsc[id * 3]), uu = moe_rb(au * gsc[id * 3 + 1]);
        const float a = moe_rb(g / (1.f + exp(-g)));
        h[(ulong)pr * I + r] = f2bf(a * uu);
    }
}

// decode down + weighted sum: grid (ceil(H / 4), T)
template <typename OutT>
inline void moe_dec_dn_impl(device const ushort* h, device const int* ids, device const float* wts, device const uchar* dw,
                            device const uchar* ds, device const float* gsc, device OutT* out, constant OpP& p, uint2 tg,
                            uint sg, uint lane) {
    const uint n = tg.x * MOE_DEC_SG + sg, t = tg.y;
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    if (n >= H) return;
    float acc = 0.f;
    for (uint k = 0; k < K; ++k) {
        const uint pr = t * K + k;
        const int id = ids[pr];
        if (id < 0 || (uint)id >= E) continue;  // uniform over the simdgroup
        const ulong rn = (ulong)id * H + n;
        device const packed_uint2* wr = (device const packed_uint2*)(dw + rn * (I / 2));
        device const uchar* sr = ds + p.s2 + rn * (I / 16);
        device const ushort* hr = h + (ulong)pr * I;
        float s = 0.f;
        for (uint u = lane; u < I / 16; u += 32u) {
            float hv[16];
            for (int j = 0; j < 16; ++j) hv[j] = bf2f(hr[u * 16 + j]);
            s = fma(moe_blk_dot(uint2(wr[u]), hv), moe_e4m3(sr[u]), s);
        }
        s = simd_sum(s);
        const float y = moe_rb(s * gsc[id * 3 + 2]);
        acc += moe_rb(y * wts[pr]);
    }
    if (lane == 0) store_out(out + p.s3, (ulong)t * H + n, acc);
}

#define MOE_DEC_DN(NAME, OUT)                                                                                       \
    kernel void NAME(device const ushort* h [[buffer(0)]], device const int* ids [[buffer(1)]],                    \
                     device const float* wts [[buffer(2)]], device const uchar* dw [[buffer(3)]],                   \
                     device const uchar* ds [[buffer(4)]], device const float* gsc [[buffer(5)]],                   \
                     device OUT* out [[buffer(6)]], constant OpP& p [[buffer(7)]],                                   \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],         \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        moe_dec_dn_impl<OUT>(h, ids, wts, dw, ds, gsc, out, p, tg, sg, lane);                                       \
    }

MOE_DEC_DN(moe_dec_dn_bf16, ushort)
MOE_DEC_DN(moe_dec_dn_f32, float)

// ---- grouped (prefill) path ---------------------------------------------------------------------------------------
// The rows per tile (8 / 16 / 32 / 64) are chosen per call (p.f) from the mean rows per expert.

// one threadgroup (any size, a multiple of 32): pairs[slot] = pair id sorted by expert (invalid ids dropped), off[E+1]
// the slot offsets, tiles[E+1] the prefix of ceil(count / f). a = T, b = E (<= 1024), c = K, f = rows per tile.
kernel void moe_group(device const int* ids [[buffer(0)]], device int* pairs [[buffer(1)]], device int* off [[buffer(2)]],
                      device int* tiles [[buffer(3)]], constant OpP& p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
    threadgroup atomic_uint cnt[1024];
    const uint E = p.b, P = p.a * p.c, TM = p.f;
    for (uint e = tid; e < E; e += nt) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < P; i += nt) {
        const int id = ids[i];
        if (id >= 0 && (uint)id < E) atomic_fetch_add_explicit(&cnt[id], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint o = 0u, tl = 0u;
        for (uint e = 0; e < E; ++e) {
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            off[e] = (int)o;
            tiles[e] = (int)tl;
            atomic_store_explicit(&cnt[e], o, memory_order_relaxed);  // scatter cursor
            o += c;
            tl += (c + TM - 1) / TM;
        }
        off[E] = (int)o;
        tiles[E] = (int)tl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < P; i += nt) {
        const int id = ids[i];
        if (id >= 0 && (uint)id < E) pairs[atomic_fetch_add_explicit(&cnt[id], 1u, memory_order_relaxed)] = (int)i;
    }
}

// tile `ty` of `tm` rows -> (expert, first slot, rows); false past the last tile
inline bool moe_tile(device const int* off, device const int* tiles, uint E, uint tm, uint ty, thread uint& e,
                     thread uint& s0, thread uint& rows) {
    if ((int)ty >= tiles[E]) return false;
    uint lo = 0u, hi = E;  // largest e with tiles[e] <= ty (tiles[e + 1] > ty then)
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if ((uint)tiles[mid] <= ty) lo = mid;
        else hi = mid;
    }
    e = lo;
    s0 = (uint)off[e] + (ty - (uint)tiles[e]) * tm;
    rows = min(tm, (uint)off[e + 1] - s0);
    return true;
}

// ---- row-exact expert-grouped decode (continuous batching) ----------------------------------------------------------
// T <= 8 tokens (T * K <= 64 pairs) in one launch, each expert's weights read ONCE for all the tokens routed to it, and every
// pair's value bitwise what moe_dec_gu / moe_dec_dn compute for it alone (the T = 1 launch): per pair the same lane-strided
// block loop, the same fma chain, the same simd_sum, the same roundings. No sort pass: threadgroup row y is a pair `pr`; it
// works only if it is the first pair of its expert (or the first of the next 8 of that expert, for ids repeated beyond one
// per token) and then serves up to MOE_XR pairs of that expert, found by scanning the ids (at most 64, in L1). h is indexed by
// pair (token * K + k) as in the decode kernels, y by pair as in the grouped path; the final per-token weighted sum is
// moe_gather (k order, as moe_dec_dn).

constant constexpr uint MOE_XR = 8;  // pairs served per threadgroup row

// The pairs this threadgroup row serves: pl[0..rows) ascending, all with expert `e`. false: not a leader / invalid id. One
// simd ballot per 32 ids (every lane of a simdgroup calls it with the same arguments).
inline bool moe_lead(device const int* ids, uint P, uint E, uint pr, uint lane, thread uint& e, thread uint& rows,
                     thread uint* pl) {
    const int id = ids[pr];
    if (id < 0 || (uint)id >= E) return false;
    uint rank = 0;  // pairs before `pr` with the same expert
    for (uint base = 0; base < pr; base += 32u) {
        const int v = (base + lane < P) ? ids[base + lane] : -1;
        ulong m = (ulong)simd_ballot(v == id);
        const uint lim = min(pr - base, 32u);
        m &= (lim >= 32u) ? 0xFFFFFFFFul : ((1ul << lim) - 1ul);
        rank += (uint)popcount(m);
    }
    if (rank % MOE_XR != 0) return false;
    e = (uint)id;
    rows = 0;
    uint seen = 0;  // pairs of this expert met so far; keep those numbered rank .. rank + MOE_XR - 1
    for (uint base = 0; base < P && rows < MOE_XR; base += 32u) {
        const int v = (base + lane < P) ? ids[base + lane] : -1;
        ulong m = (ulong)simd_ballot(v == id);
        while (m != 0ul && rows < MOE_XR) {
            const uint b = (uint)ctz(m);
            m &= m - 1ul;
            if (seen >= rank) pl[rows++] = base + b;
            seen++;
        }
    }
    return true;
}

// gate + up: grid (ceil(I / 4), pairs); one simdgroup per row r, loops the pairs of its expert inside the weight-block loop
kernel void moe_dec_gu_x(device const ushort* x [[buffer(0)]], device const int* ids [[buffer(1)]],
                         device const uchar* guw [[buffer(2)]], device const uchar* gus [[buffer(3)]],
                         device const float* gsc [[buffer(4)]], device ushort* h [[buffer(5)]],
                         constant OpP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x * MOE_DEC_SG + sg;
    const uint T = p.a, E = p.b, K = p.c, H = p.d, I = p.e;
    uint e, rows;
    uint pl[MOE_XR];
    if (!moe_lead(ids, T * K, E, tg.y, lane, e, rows, pl)) return;  // uniform over the threadgroup
    if (r >= I) return;
    device const ushort* xr[MOE_XR];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0; q < MOE_XR; ++q) xr[q] = x + p.s0 + (ulong)(pl[min(q, rows - 1u)] / K) * H;
    const ulong rg = (ulong)e * 2 * I + r, ru = rg + I;
    device const packed_uint2* wg = (device const packed_uint2*)(guw + rg * (H / 2));
    device const packed_uint2* wu = (device const packed_uint2*)(guw + ru * (H / 2));
    device const uchar* sgp = gus + p.s1 + rg * (H / 16);
    device const uchar* sup = gus + p.s1 + ru * (H / 16);
    float ag[MOE_XR], au[MOE_XR];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0; q < MOE_XR; ++q) {
        ag[q] = 0.f;
        au[q] = 0.f;
    }
    for (uint u = lane; u < H / 16; u += 32u) {
        float wgf[16], wuf[16];
        moe_blk_dec(uint2(wg[u]), wgf);
        moe_blk_dec(uint2(wu[u]), wuf);
        const float sgv = moe_e4m3(sgp[u]), suv = moe_e4m3(sup[u]);
        _Pragma("clang loop unroll(full)")
        for (uint q = 0; q < MOE_XR; ++q) {
            if (q >= rows) break;
            float xv[16];
            for (int j = 0; j < 16; ++j) xv[j] = bf2f(xr[q][u * 16 + j]);
            ag[q] = fma(moe_blk_dot_f(wgf, xv), sgv, ag[q]);
            au[q] = fma(moe_blk_dot_f(wuf, xv), suv, au[q]);
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint q = 0; q < MOE_XR; ++q) {
        if (q >= rows) break;
        const float a = simd_sum(ag[q]), b = simd_sum(au[q]);
        if (lane == 0) {
            const float g = moe_rb(a * gsc[e * 3]), uu = moe_rb(b * gsc[e * 3 + 1]);
            const float act = moe_rb(g / (1.f + exp(-g)));
            h[(ulong)pl[q] * I + r] = f2bf(act * uu);
        }
    }
}

// down: grid (ceil(H / 4), pairs); y[pair, n] = bf16(simd_sum * gs), one weight row read per expert for all its pairs
kernel void moe_dec_dn_x(device const ushort* h [[buffer(0)]], device const int* ids [[buffer(1)]],
                         device const uchar* dw [[buffer(2)]], device const uchar* ds [[buffer(3)]],
                         device const float* gsc [[buffer(4)]], device ushort* y [[buffer(5)]],
                         constant OpP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint n = tg.x * MOE_DEC_SG + sg;
    const uint T = p.a, E = p.b, K = p.c, H = p.d, I = p.e;
    uint e, rows;
    uint pl[MOE_XR];
    if (!moe_lead(ids, T * K, E, tg.y, lane, e, rows, pl)) return;  // uniform over the threadgroup
    if (n >= H) return;
    device const ushort* hr[MOE_XR];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0; q < MOE_XR; ++q) hr[q] = h + (ulong)pl[min(q, rows - 1u)] * I;
    const ulong rn = (ulong)e * H + n;
    device const packed_uint2* wr = (device const packed_uint2*)(dw + rn * (I / 2));
    device const uchar* sr = ds + p.s2 + rn * (I / 16);
    float s[MOE_XR];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0; q < MOE_XR; ++q) s[q] = 0.f;
    for (uint u = lane; u < I / 16; u += 32u) {
        float wf[16];
        moe_blk_dec(uint2(wr[u]), wf);
        const float sc = moe_e4m3(sr[u]);
        _Pragma("clang loop unroll(full)")
        for (uint q = 0; q < MOE_XR; ++q) {
            if (q >= rows) break;
            float hv[16];
            for (int j = 0; j < 16; ++j) hv[j] = bf2f(hr[q][u * 16 + j]);
            s[q] = fma(moe_blk_dot_f(wf, hv), sc, s[q]);
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint q = 0; q < MOE_XR; ++q) {
        if (q >= rows) break;
        const float t = simd_sum(s[q]);
        if (lane == 0) y[(ulong)pl[q] * H + n] = f2bf(t * gsc[e * 3 + 2]);
    }
}

// NVFP4 -> bf16 as kernels/metal/nvfp4.metal: two codes (nibbles at bits 0..3, 16..19) -> value / 2^14 via half bits,
// an e4m3 scale byte -> value / 256 (NaN code -> NaN); the 2^22 is folded back into the scale, the product is exact
inline float2 moe_x2(uint t) {
    const uint h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    return float2(as_type<half2>(h));
}
inline float moe_sc(uint b) {
    if ((b & 0x7Fu) == 0x7Fu) return as_type<float>(0x7FC00000u);
    return float(as_type<half>((ushort)(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)))) * 4194304.0f;
}
inline uint moe_pack(float lo, float hi) { return (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xFFFF0000u); }

// one 16-element block (2 words, scale s from moe_sc) -> 16 bf16 at dst (16-byte aligned)
inline void moe_blk_bf16(uint2 v, float s, threadgroup bfloat* dst) {
    uint o[8];
    for (int i = 0; i < 2; ++i) {
        float2 c[4];  // c[k]: elements k and k + 4 of the word
        for (int k = 0; k < 4; ++k) c[k] = moe_x2((v[i] >> (4 * k)) & 0x000F000Fu) * s;
        o[4 * i] = moe_pack(c[0].x, c[1].x);
        o[4 * i + 1] = moe_pack(c[2].x, c[3].x);
        o[4 * i + 2] = moe_pack(c[0].y, c[1].y);
        o[4 * i + 3] = moe_pack(c[2].y, c[3].y);
    }
    threadgroup uint4* d = (threadgroup uint4*)dst;
    d[0] = uint4(o[0], o[1], o[2], o[3]);
    d[1] = uint4(o[4], o[5], o[6], o[7]);
}

constant constexpr uint MOE_WR = 64;            // weight rows per tile: 32 gate + 32 up columns (gu), 64 columns (dn)
constant constexpr uint MOE_LC = MOE_WR + 4;    // floats per epilogue row

// raw weight words and scale bytes of blocks bq .. bq + NB - 1 of the K step at k0 (zero past the matrix)
template <uint NB>
inline void moe_fetch(device const uchar* wp, device const uchar* sp, bool bok, uint bq, uint k0, uint KD,
                      thread uint2* wv, thread uint* sv) {
    for (uint q = 0; q < NB; ++q) {
        const uint gk = k0 + (bq + q) * 16u;
        const bool ok = bok && gk < KD;
        wv[q] = ok ? uint2(*(device const packed_uint2*)(wp + gk / 2)) : uint2(0u);
        sv[q] = ok ? (uint)sp[gk / 16] : 0u;
    }
}

// One expert tile: BM rows (slots) x MOE_WR weight rows, bf16 x bf16 -> f32 simdgroup matrices, 4 simdgroups of BM / SR
// rows x 16 * SR weight rows; every weight block is decoded once per tile into threadgroup memory. K steps of BK (BK / 2 contiguous
// bytes per weight row; the next step's raw weights are loaded into registers during the current step's MMAs).
//   MODE 0 (gate + up): A = x[pair / K] [., H], grid (ceil(I / 32), tiles); h[slot, n] = rb(rb(silu(g)) * u) bf16.
//   MODE 1 (down):      A = h[slot] [., I],     grid (ceil(H / 64), tiles); y[pair, n] = rb(acc * gs) bf16.
template <uint MODE, uint BM, uint BK>
inline void moe_mm_impl(device const ushort* A, device const int* pairs, device const int* off, device const int* tiles,
                        device const uchar* W, device const uchar* S, device const float* gsc, device ushort* out,
                        constant OpP& p, threadgroup float* smem, uint2 tg, uint tid, uint sg) {
    constexpr uint LD = BK + 8;                 // bf16 per staged row (a 16-byte multiple)
    constexpr uint AC = BK / 8;                 // 8-element A chunks per row
    constexpr uint NA = (BM * AC + 127) / 128;  // A chunks per thread
    constexpr uint NB = BK / 32;                // weight blocks per thread and step
    constexpr uint SR = BM == 8 ? 1 : 2;        // simdgroups SR x (4 / SR): MR rows x MC weight rows each
    constexpr uint MR = BM / SR / 8, MC = MOE_WR / (4 / SR) / 8;
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    const uint KD = MODE == 0 ? H : I;
    uint e, s0, rows;
    if (!moe_tile(off, tiles, E, BM, tg.y, e, s0, rows)) return;
    threadgroup bfloat* As = (threadgroup bfloat*)smem;
    threadgroup bfloat* Bs = As + BM * LD;
    // A rows of this thread's chunks (null: a padding row)
    device const ushort* ap[NA];
    for (uint j = 0; j < NA; ++j) {
        const uint c = tid + 128u * j, r = c / AC;
        ap[j] = nullptr;
        if (c < BM * AC && r < rows)
            ap[j] = MODE == 0 ? A + p.s0 + (ulong)((uint)pairs[s0 + r] / K) * H + (c % AC) * 8u
                              : A + (ulong)(s0 + r) * I + (c % AC) * 8u;
    }
    // B loader: weight row tid / 2, blocks (tid % 2) * NB + q of the step (NB * 8 contiguous bytes)
    const uint br = tid / 2u, bq = (tid % 2u) * NB;
    const uint n0 = tg.x * (MODE == 0 ? MOE_WR / 2 : MOE_WR);
    const uint bn = MODE == 0 ? n0 + br % 32u : n0 + br;
    const bool bok = bn < (MODE == 0 ? I : H);
    const ulong wrow = MODE == 0 ? (ulong)e * 2 * I + (br / 32u) * I + bn : (ulong)e * H + bn;
    device const uchar* wp = W + wrow * (KD / 2);
    device const uchar* sp = S + (MODE == 0 ? p.s1 : p.s2) + wrow * (KD / 16);
    uint2 wv[NB];
    uint sv[NB];
    moe_fetch<NB>(wp, sp, bok, bq, 0u, KD, wv, sv);
    const uint sr = (sg / (4 / SR)) * MR * 8, sc = (sg % (4 / SR)) * MC * 8;
    simdgroup_float8x8 acc[MR][MC];
    for (uint i = 0; i < MR; ++i)
        for (uint j = 0; j < MC; ++j) acc[i][j] = simdgroup_float8x8(0.f);
    for (uint k0 = 0; k0 < KD; k0 += BK) {
        for (uint j = 0; j < NA; ++j) {  // A: BM rows x AC chunks of 8 bf16 (x may sit at an odd element)
            const uint c = tid + 128u * j;
            if (c >= BM * AC) break;
            uint4 v = uint4(0u);
            if (ap[j] != nullptr && k0 + (c % AC) * 8u < KD) {
                device const packed_ushort4* q4 = (device const packed_ushort4*)(ap[j] + k0);
                const ushort4 a = ushort4(q4[0]), b = ushort4(q4[1]);
                v = uint4(a.x | ((uint)a.y << 16), a.z | ((uint)a.w << 16), b.x | ((uint)b.y << 16), b.z | ((uint)b.w << 16));
            }
            *(threadgroup uint4*)(As + (c / AC) * LD + (c % AC) * 8u) = v;
        }
        for (uint q = 0; q < NB; ++q) moe_blk_bf16(wv[q], moe_sc(sv[q]), Bs + br * LD + (bq + q) * 16u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < KD) moe_fetch<NB>(wp, sp, bok, bq, k0 + BK, KD, wv, sv);
        for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_matrix<bfloat, 8, 8> a[MR], b[MC];
            for (uint i = 0; i < MR; ++i) simdgroup_load(a[i], As + (sr + i * 8) * LD + kk, LD);
            for (uint j = 0; j < MC; ++j) simdgroup_load(b[j], Bs + (sc + j * 8) * LD + kk, LD, ulong2(0, 0), true);
            for (uint i = 0; i < MR; ++i)
                for (uint j = 0; j < MC; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue through threadgroup memory: C [BM][MOE_LC] f32 (gu: columns 0..31 gate, 32..63 up)
    for (uint i = 0; i < MR; ++i)
        for (uint j = 0; j < MC; ++j) simdgroup_store(acc[i][j], smem + (sr + i * 8) * MOE_LC + sc + j * 8, MOE_LC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (MODE == 0) {
        const float g0 = gsc[e * 3], g1 = gsc[e * 3 + 1];
        for (uint q = tid; q < BM * 32u; q += 128u) {
            const uint r = q / 32u, c = q % 32u;
            if (r >= rows || n0 + c >= I) continue;
            const float g = moe_rb(smem[r * MOE_LC + c] * g0), u = moe_rb(smem[r * MOE_LC + 32 + c] * g1);
            const float a = moe_rb(g / (1.f + exp(-g)));
            out[(ulong)(s0 + r) * I + n0 + c] = f2bf(a * u);
        }
    } else {
        const float g2 = gsc[e * 3 + 2];
        for (uint q = tid; q < BM * MOE_WR; q += 128u) {
            const uint r = q / MOE_WR, c = q % MOE_WR;
            if (r >= rows || n0 + c >= H) continue;
            out[(ulong)pairs[s0 + r] * H + n0 + c] = f2bf(smem[r * MOE_LC + c] * g2);
        }
    }
}

// threadgroup memory: the staged A + B tiles, reused by the f32 epilogue
#define MOE_SMEM(BM, BK) \
    (((BM + MOE_WR) * (BK + 8) * 2 > BM * MOE_LC * 4 ? (BM + MOE_WR) * (BK + 8) * 2 : BM * MOE_LC * 4) / 16)

// K step 128 (64-byte weight runs) up to 32 rows; 64 at 64 rows (threadgroup memory)
#define MOE_MM(NAME, MODE, BM, BK)                                                                                      \
    kernel void NAME(device const ushort* A [[buffer(0)]], device const int* pairs [[buffer(1)]],                  \
                     device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],                    \
                     device const uchar* W [[buffer(4)]], device const uchar* S [[buffer(5)]],                      \
                     device const float* gsc [[buffer(6)]], device ushort* out [[buffer(7)]],                       \
                     constant OpP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {        \
        threadgroup float4 smem[MOE_SMEM(BM, BK)];                                                                      \
        moe_mm_impl<MODE, BM, BK>(A, pairs, off, tiles, W, S, gsc, out, p, (threadgroup float*)smem, tg, tid, sg);      \
    }

MOE_MM(moe_mm_gu_m8, 0, 8, 128)
MOE_MM(moe_mm_gu_m16, 0, 16, 128)
MOE_MM(moe_mm_gu_m32, 0, 32, 128)
MOE_MM(moe_mm_gu_m64, 0, 64, 64)
MOE_MM(moe_mm_dn_m8, 1, 8, 128)
MOE_MM(moe_mm_dn_m16, 1, 16, 128)
MOE_MM(moe_mm_dn_m32, 1, 32, 128)
MOE_MM(moe_mm_dn_m64, 1, 64, 64)

// per token the weighted sum of its y rows in k order: one thread per (t, n), grid T*H
template <typename OutT>
inline void moe_gather_impl(device const ushort* y, device const int* ids, device const float* wts, device OutT* out,
                            constant OpP& p, uint i) {
    const uint E = p.b, K = p.c, H = p.d;
    if (i >= p.a * H) return;
    const uint t = i / H, n = i % H;
    float acc = 0.f;
    for (uint k = 0; k < K; ++k) {
        const uint pr = t * K + k;
        const int id = ids[pr];
        if (id < 0 || (uint)id >= E) continue;
        acc += moe_rb(bf2f(y[(ulong)pr * H + n]) * wts[pr]);
    }
    store_out(out + p.s3, i, acc);
}

#define MOE_GATHER(NAME, OUT)                                                                                       \
    kernel void NAME(device const ushort* y [[buffer(0)]], device const int* ids [[buffer(1)]],                    \
                     device const float* wts [[buffer(2)]], device OUT* out [[buffer(3)]],                          \
                     constant OpP& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {                           \
        moe_gather_impl<OUT>(y, ids, wts, out, p, i);                                                               \
    }

MOE_GATHER(moe_gather_bf16, ushort)
MOE_GATHER(moe_gather_f32, float)

// ---- row-exact decode with fat threadgroups (lane mm, docs/results/2026-10-25-mm-metal-moe-grouped.md) ---------------
// T <= 8 tokens in one launch, every pair's value bitwise what moe_dec_gu / moe_dec_dn compute for it alone: per (pair, row)
// the same lane-strided block loop, the same moe_blk_dot fma chain, the same scale fma, the same simd_sum and roundings.
// Threadgroup row y is pair y; a simdgroup covers MOE_PR gate+up rows / down rows of that pair's expert (the one-token kernels
// and moe_dec_{gu,dn}_x cover one), so a call runs MOE_PR times fewer threadgroups and each lane keeps MOE_PR weight blocks
// in flight; the activation block is loaded once for all of them. No grouping by expert: the pairs that share an expert read
// its blocks at about the same time and the second read comes from the cache (measured faster than serving the pairs of an
// expert from one threadgroup, the moe_lead scheme of the _x kernels). h by pair, y by pair; moe_gather sums per token in k
// order. NEG (negative control only, INTARSIA_METAL_NEGCTL): each block's scale fma is split into two fmas of half the
// block dot (two roundings instead of one), which the row gate must catch. (A plain multiply and add is not a control: the
// compiler contracts it back into the same fma.)

constant constexpr uint MOE_PR = 2;  // rows per simdgroup

// gate + up: grid (ceil(I / (4 * MOE_PR)), pairs)
template <bool NEG>
inline void moe_dec_gu_p_impl(device const ushort* x, device const int* ids, device const uchar* guw, device const uchar* gus,
                              device const float* gsc, device ushort* h, constant OpP& p, uint2 tg, uint sg, uint lane) {
    const uint r0 = (tg.x * MOE_DEC_SG + sg) * MOE_PR, pr = tg.y;
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    if (r0 >= I) return;
    const int id = ids[pr];
    if (id < 0 || (uint)id >= E) return;
    device const ushort* xr = x + p.s0 + (ulong)(pr / K) * H;
    device const packed_uint2* wg[MOE_PR];
    device const packed_uint2* wu[MOE_PR];
    device const uchar* sgp[MOE_PR];
    device const uchar* sup[MOE_PR];
    for (uint j = 0; j < MOE_PR; ++j) {
        const ulong rg = (ulong)id * 2 * I + min(r0 + j, I - 1u), ru = rg + I;
        wg[j] = (device const packed_uint2*)(guw + rg * (H / 2));
        wu[j] = (device const packed_uint2*)(guw + ru * (H / 2));
        sgp[j] = gus + p.s1 + rg * (H / 16);
        sup[j] = gus + p.s1 + ru * (H / 16);
    }
    float ag[MOE_PR], au[MOE_PR];
    for (uint j = 0; j < MOE_PR; ++j) {
        ag[j] = 0.f;
        au[j] = 0.f;
    }
    for (uint u = lane; u < H / 16; u += 32u) {
        float xv[16];
        for (int i = 0; i < 16; ++i) xv[i] = bf2f(xr[u * 16 + i]);
        for (uint j = 0; j < MOE_PR; ++j) {
            const float dg = moe_blk_dot(uint2(wg[j][u]), xv);
            ag[j] = NEG ? fma(0.5f * dg, moe_e4m3(sgp[j][u]), fma(0.5f * dg, moe_e4m3(sgp[j][u]), ag[j])) : fma(dg, moe_e4m3(sgp[j][u]), ag[j]);
            au[j] = fma(moe_blk_dot(uint2(wu[j][u]), xv), moe_e4m3(sup[j][u]), au[j]);
        }
    }
    for (uint j = 0; j < MOE_PR; ++j) {
        const float a = simd_sum(ag[j]), b = simd_sum(au[j]);
        if (lane == 0 && r0 + j < I) {
            const float g = moe_rb(a * gsc[id * 3]), uu = moe_rb(b * gsc[id * 3 + 1]);
            const float act = moe_rb(g / (1.f + exp(-g)));
            h[(ulong)pr * I + r0 + j] = f2bf(act * uu);
        }
    }
}

// down: grid (ceil(H / (4 * MOE_PR)), pairs); y[pair, n] = bf16(simd_sum * gs)
template <bool NEG>
inline void moe_dec_dn_p_impl(device const ushort* h, device const int* ids, device const uchar* dw, device const uchar* ds,
                              device const float* gsc, device ushort* y, constant OpP& p, uint2 tg, uint sg, uint lane) {
    const uint n0 = (tg.x * MOE_DEC_SG + sg) * MOE_PR, pr = tg.y;
    const uint E = p.b, H = p.d, I = p.e;
    if (n0 >= H) return;
    const int id = ids[pr];
    if (id < 0 || (uint)id >= E) return;
    device const packed_uint2* wr[MOE_PR];
    device const uchar* sr[MOE_PR];
    for (uint j = 0; j < MOE_PR; ++j) {
        const ulong rn = (ulong)id * H + min(n0 + j, H - 1u);
        wr[j] = (device const packed_uint2*)(dw + rn * (I / 2));
        sr[j] = ds + p.s2 + rn * (I / 16);
    }
    device const ushort* hr = h + (ulong)pr * I;
    float s[MOE_PR];
    for (uint j = 0; j < MOE_PR; ++j) s[j] = 0.f;
    for (uint u = lane; u < I / 16; u += 32u) {
        float hv[16];
        for (int i = 0; i < 16; ++i) hv[i] = bf2f(hr[u * 16 + i]);
        for (uint j = 0; j < MOE_PR; ++j) {
            const float d = moe_blk_dot(uint2(wr[j][u]), hv);
            s[j] = NEG ? fma(0.5f * d, moe_e4m3(sr[j][u]), fma(0.5f * d, moe_e4m3(sr[j][u]), s[j])) : fma(d, moe_e4m3(sr[j][u]), s[j]);
        }
    }
    for (uint j = 0; j < MOE_PR; ++j) {
        const float t = simd_sum(s[j]);
        if (lane == 0 && n0 + j < H) y[(ulong)pr * H + n0 + j] = f2bf(t * gsc[id * 3 + 2]);
    }
}

#define MOE_GU_P(NAME, NEG)                                                                                         \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const int* ids [[buffer(1)]],                    \
                     device const uchar* guw [[buffer(2)]], device const uchar* gus [[buffer(3)]],                  \
                     device const float* gsc [[buffer(4)]], device ushort* h [[buffer(5)]],                         \
                     constant OpP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        moe_dec_gu_p_impl<NEG>(x, ids, guw, gus, gsc, h, p, tg, sg, lane);                                          \
    }
#define MOE_DN_P(NAME, NEG)                                                                                         \
    kernel void NAME(device const ushort* h [[buffer(0)]], device const int* ids [[buffer(1)]],                    \
                     device const uchar* dw [[buffer(2)]], device const uchar* ds [[buffer(3)]],                    \
                     device const float* gsc [[buffer(4)]], device ushort* y [[buffer(5)]],                         \
                     constant OpP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        moe_dec_dn_p_impl<NEG>(h, ids, dw, ds, gsc, y, p, tg, sg, lane);                                            \
    }

MOE_GU_P(moe_dec_gu_p, false)
MOE_DN_P(moe_dec_dn_p, false)
MOE_GU_P(moe_dec_gu_p_neg, true)
MOE_DN_P(moe_dec_dn_p_neg, true)
