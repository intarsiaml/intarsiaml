// Sparse-MoE ops of Qwen3.6-35B-A3B (port of kernels/hip/moe.hip; launcher crates/intarsia-backend-metal/src/moe.rs).
// Semantics: crates/intarsia-core/src/quant/reference.rs (`moe_topk`, `expert_forward`, `moe_experts_token`, `moe_combine`).
//   moe_topk_{bf16,f32}     one simdgroup per token: K rounds of a (value desc, index asc) argmax, softmax over the
//                           selected (norm) or all logits; weights f32 rounded to the logits dtype, ids i32.
//   moe_combine_bf16        out = rb(routed + rb(rb(sigmoid(gate[t])) * shared)), one thread per element.
//   MoeExperts, NVFP4 experts (low nibble = even column, value = e2m1 * e4m3(block of 16) * global), bf16 x:
//     decode (T*K small)    moe_dec_gu: one simdgroup per (pair, row r): gate + up rows -> h[pair, r] (bf16 scratch);
//                           moe_dec_dn_*: one simdgroup per (token, out row): the K down rows in k order, weighted sum.
//     prefill (grouped)     moe_group: one threadgroup counts / prefix-sums / scatters the valid pairs by expert;
//                           moe_gemm_gu / moe_gemm_dn: 32x32 tiles of one expert's rows on simdgroup float8x8 matrices,
//                           each tile's weights decoded once into threadgroup memory; h by slot, y by pair;
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

constant uint MOE_TM = 32;  // rows (slots) per tile
constant uint MOE_TN = 32;  // output columns per tile
constant uint MOE_BK = 32;  // reduction step (two NVFP4 blocks)

// one threadgroup (any size, a multiple of 32): pairs[slot] = pair id sorted by expert (invalid ids dropped), off[E+1]
// the slot offsets, tiles[E+1] the prefix of ceil(count / MOE_TM). a = T, b = E (<= 1024), c = K.
kernel void moe_group(device const int* ids [[buffer(0)]], device int* pairs [[buffer(1)]], device int* off [[buffer(2)]],
                      device int* tiles [[buffer(3)]], constant OpP& p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
    threadgroup atomic_uint cnt[1024];
    const uint E = p.b, P = p.a * p.c;
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
            tl += (c + MOE_TM - 1) / MOE_TM;
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

// tile `ty` -> (expert, first slot, rows); false past the last tile
inline bool moe_tile(device const int* off, device const int* tiles, uint E, uint ty, thread uint& e, thread uint& s0,
                     thread uint& rows) {
    if ((int)ty >= tiles[E]) return false;
    uint lo = 0u, hi = E;  // largest e with tiles[e] <= ty (tiles[e + 1] > ty then)
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if ((uint)tiles[mid] <= ty) lo = mid;
        else hi = mid;
    }
    e = lo;
    s0 = (uint)off[e] + (ty - (uint)tiles[e]) * MOE_TM;
    rows = min(MOE_TM, (uint)off[e + 1] - s0);
    return true;
}

// decode one 16-element NVFP4 block (without the global scale) into dst[0], dst[stride], ...
inline void moe_blk_to_tg(device const uchar* w, device const uchar* s, threadgroup float* dst, uint stride) {
    const uint2 wv = uint2(*(device const packed_uint2*)w);
    const float sc = moe_e4m3(*s);
    for (int q = 0; q < 2; ++q) {
        for (int b = 0; b < 8; ++b) dst[(q * 8 + b) * stride] = moe_fp4[(wv[q] >> (4 * b)) & 0xFu] * sc;
    }
}

// tg memory strides (floats); A [TM][BK], B [BK][TN]
constant uint MOE_LA = MOE_BK + 4;
constant uint MOE_LB = MOE_TN + 4;

// gate + up of one tile: grid (ceil(I / TN), max tiles), 128 threads. h[slot, n] bf16.
kernel void moe_gemm_gu(device const ushort* x [[buffer(0)]], device const int* pairs [[buffer(1)]],
                        device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],
                        device const uchar* guw [[buffer(4)]], device const uchar* gus [[buffer(5)]],
                        device const float* gsc [[buffer(6)]], device ushort* h [[buffer(7)]], constant OpP& p [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    uint e, s0, rows;
    if (!moe_tile(off, tiles, E, tg.y, e, s0, rows)) return;
    const uint n0 = tg.x * MOE_TN;
    threadgroup float As[MOE_TM * MOE_LA];
    threadgroup float Bs[2][MOE_BK * MOE_LB];
    // A loader: row tid / 4, 8 columns from (tid % 4) * 8
    const uint ar = tid / 4u, ac = (tid % 4u) * 8u;
    device const ushort* xr = ar < rows ? x + p.s0 + (ulong)((uint)pairs[s0 + ar] / K) * H : x;
    // B loader: matrix tid / 64 (gate, up), row (tid % 64) / 2, block tid % 2
    const uint bm = tid / 64u, br = (tid % 64u) / 2u, bb = tid % 2u;
    const ulong wrow = (ulong)e * 2 * I + bm * I + n0 + br;
    const bool bok = n0 + br < I;
    // simdgroup: rows (sg / 2) * 16, cols (sg % 2) * 16; 2 x 2 tiles of 8 x 8 per matrix
    const uint mr = (sg / 2u) * 16u, mc = (sg % 2u) * 16u;
    simdgroup_float8x8 cg[2][2], cu[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            cg[i][j] = simdgroup_float8x8(0.f);
            cu[i][j] = simdgroup_float8x8(0.f);
        }
    for (uint k0 = 0; k0 < H; k0 += MOE_BK) {
        for (uint c = 0; c < 8u; ++c) {
            const uint kk = k0 + ac + c;
            As[ar * MOE_LA + ac + c] = (ar < rows && kk < H) ? bf2f(xr[kk]) : 0.f;
        }
        const uint kb = k0 + bb * 16u;
        threadgroup float* bd = &Bs[bm][(bb * 16u) * MOE_LB + br];
        if (bok && kb < H) {
            moe_blk_to_tg(guw + wrow * (H / 2) + kb / 2, gus + p.s1 + wrow * (H / 16) + kb / 16, bd, MOE_LB);
        } else {
            for (uint j = 0; j < 16u; ++j) bd[j * MOE_LB] = 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < MOE_BK; kk += 8u) {
            simdgroup_float8x8 a[2], bg[2], bu[2];
            for (int i = 0; i < 2; ++i) simdgroup_load(a[i], &As[(mr + 8 * i) * MOE_LA + kk], MOE_LA);
            for (int j = 0; j < 2; ++j) {
                simdgroup_load(bg[j], &Bs[0][kk * MOE_LB + mc + 8 * j], MOE_LB);
                simdgroup_load(bu[j], &Bs[1][kk * MOE_LB + mc + 8 * j], MOE_LB);
            }
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(cg[i][j], a[i], bg[j], cg[i][j]);
                    simdgroup_multiply_accumulate(cu[i][j], a[i], bu[j], cu[i][j]);
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue through threadgroup memory: Bs[0] = gate, Bs[1] = up as [TM][LB]
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            simdgroup_store(cg[i][j], &Bs[0][(mr + 8 * i) * MOE_LB + mc + 8 * j], MOE_LB);
            simdgroup_store(cu[i][j], &Bs[1][(mr + 8 * i) * MOE_LB + mc + 8 * j], MOE_LB);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float g0 = gsc[e * 3], g1 = gsc[e * 3 + 1];
    for (uint q = tid; q < MOE_TM * MOE_TN; q += 128u) {
        const uint r = q / MOE_TN, c = q % MOE_TN;
        if (r >= rows || n0 + c >= I) continue;
        const float g = moe_rb(Bs[0][r * MOE_LB + c] * g0), u = moe_rb(Bs[1][r * MOE_LB + c] * g1);
        const float a = moe_rb(g / (1.f + exp(-g)));
        h[(ulong)(s0 + r) * I + n0 + c] = f2bf(a * u);
    }
}

// down of one tile: grid (ceil(H / TN), max tiles), 128 threads. y[pair, n] bf16.
kernel void moe_gemm_dn(device const ushort* h [[buffer(0)]], device const int* pairs [[buffer(1)]],
                        device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],
                        device const uchar* dw [[buffer(4)]], device const uchar* ds [[buffer(5)]],
                        device const float* gsc [[buffer(6)]], device ushort* y [[buffer(7)]], constant OpP& p [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint E = p.b, H = p.d, I = p.e;
    uint e, s0, rows;
    if (!moe_tile(off, tiles, E, tg.y, e, s0, rows)) return;
    const uint n0 = tg.x * MOE_TN;
    threadgroup float As[MOE_TM * MOE_LA];
    threadgroup float Bs[MOE_BK * MOE_LB];
    const uint ar = tid / 4u, ac = (tid % 4u) * 8u;
    device const ushort* hr = h + (ulong)(s0 + min(ar, rows - 1u)) * I;
    // B loader: threads 0..63, row tid / 2, block tid % 2
    const uint br = (tid % 64u) / 2u, bb = tid % 2u;
    const bool bld = tid < 64u;
    const ulong wrow = (ulong)e * H + n0 + br;
    const bool bok = n0 + br < H;
    const uint mr = (sg / 2u) * 16u, mc = (sg % 2u) * 16u;
    simdgroup_float8x8 cc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) cc[i][j] = simdgroup_float8x8(0.f);
    for (uint k0 = 0; k0 < I; k0 += MOE_BK) {
        for (uint c = 0; c < 8u; ++c) {
            const uint kk = k0 + ac + c;
            As[ar * MOE_LA + ac + c] = (ar < rows && kk < I) ? bf2f(hr[kk]) : 0.f;
        }
        if (bld) {
            const uint kb = k0 + bb * 16u;
            threadgroup float* bd = &Bs[(bb * 16u) * MOE_LB + br];
            if (bok && kb < I) {
                moe_blk_to_tg(dw + wrow * (I / 2) + kb / 2, ds + p.s2 + wrow * (I / 16) + kb / 16, bd, MOE_LB);
            } else {
                for (uint j = 0; j < 16u; ++j) bd[j * MOE_LB] = 0.f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < MOE_BK; kk += 8u) {
            simdgroup_float8x8 a[2], b[2];
            for (int i = 0; i < 2; ++i) simdgroup_load(a[i], &As[(mr + 8 * i) * MOE_LA + kk], MOE_LA);
            for (int j = 0; j < 2; ++j) simdgroup_load(b[j], &Bs[kk * MOE_LB + mc + 8 * j], MOE_LB);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(cc[i][j], a[i], b[j], cc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_store(cc[i][j], &Bs[(mr + 8 * i) * MOE_LB + mc + 8 * j], MOE_LB);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float g2 = gsc[e * 3 + 2];
    for (uint q = tid; q < MOE_TM * MOE_TN; q += 128u) {
        const uint r = q / MOE_TN, c = q % MOE_TN;
        if (r >= rows || n0 + c >= H) continue;
        y[(ulong)pairs[s0 + r] * H + n0 + c] = f2bf(Bs[r * MOE_LB + c] * g2);
    }
}

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
