// Inkling-2-Mini MoE on Metal (lane ink-m-moe; launcher crates/intarsia-backend-metal/src/inkmoe.rs).
// Contracts: crates/intarsia-core/src/dsv41/ops.rs (`MoeRoute`), crates/intarsia-core/src/inkling2/ops.rs (`MoeExpertsMxfp8`
// and the `situ` activation of `MoeExperts`). f64 references: intarsia-backend-cpu dsv41_ref.rs (MoeRoute),
// intarsia-core quant/reference.rs `moe_experts_token_situ` (NVFP4 situ), intarsia-backend-cpu ink_ref.rs `experts_mxfp8`.
//
//   ink_moe_route          one simdgroup per row: score (sigmoid / sqrt-softplus / softmax of logits / temperature), top-K
//                          of score + bias as K rounds of a (value desc, index asc) argmax, weights = the unbiased scores
//                          of the picks (renormed when norm_topk and K > 1, + 1e-20 as the reference), times scale.
//   ink_moe_gu_*           gate + up of the decode / verify rows (any T): h[pair, r] bf16 with the activation epilogue
//                          (silu, gelu, situ: the routing weight folded in before w2). 8 lanes per weight row, each lane a
//                          32-element chunk per step (NVFP4: 16 bytes + 2 e4m3 scales, MXFP8: 32 bytes + 1 E8M0 scale),
//                          chunks lane-strided; a lane holds the gate row and the up row of one r (one x read for both).
//   ink_moe_dn_u_*         down per pair: y[pair, n] = bf16(dot * gs) (gs = 1 for MXFP8), 4 lanes per row.
//   ink_moe_dn_f_*         down + the per-token k-order sum in one launch: a threadgroup holds 8 output rows of one token
//                          and all K slots (simdgroup k % 4), then sums the K bf16 y values in k order (f32).
//   ink_moe_gather_*       the per-token k-order sum of the y rows (after ink_moe_dn_u).
//
// Union grouping (p.f bit 2): threadgroup row y of gu / dn_u is a pair; it works only if it leads its expert (the first pair
// of each run of INK_XR pairs of that expert, pair order) and then serves those pairs with one read of the expert's weights
// (the pairs' activations come from L1). Without it every pair reads its own weights (the grid is pair-fastest then, so the
// pairs of an expert read the same rows at about the same time).
// Rows-exact: a pair's value never depends on T, the routing of other tokens, union or not, or the kernel (dn_f or dn_u +
// gather): per (pair, row) the same chunk order (lane-strided over 8 / 4 lanes), the same fma chain per chunk, the same
// shuffle-xor reduction (1, 2, 4), the same roundings; the token sum is always in k order.
//
// Kernel arguments: OpP (common.h). Experts: a = T, b = E, c = K, d = H, e = I, f = flags (bits 0..1 epilogue: 0 silu,
// 1 gelu, 2 situ; bit 2 union), s0 = x skip, s1 = w13 scales skip, s2 = w2 scales skip, s3 = out skip, f0 = gate_cap,
// f1 = up_cap. Weight tensors are bound 16-byte aligned (`A16`) or at least 4-byte aligned (`!A16`: two 8-byte loads).

#include "common.h"

constant constexpr uint INK_XR = 4;  // pairs a union leader serves

// ---- MoeRoute -------------------------------------------------------------------------------------------------------
// a = T, b = E (<= 1024), c = K (<= 32), d = norm_topk, e = score func (0 softmax, 1 sigmoid, 2 sqrt-softplus);
// f0 = scale, f1 = temperature. Buffers: logits f32 [T, E], bias f32 [E], weights f32 [T, K], ids i32 [T, K].

constant constexpr uint INK_ROUTE_SG = 4;  // rows (simdgroups) per threadgroup

inline bool ink_better(float v1, uint i1, float v2, uint i2) {
    if (i2 == 0xFFFFFFFFu) return i1 != 0xFFFFFFFFu;
    if (i1 == 0xFFFFFFFFu) return false;
    return v1 > v2 || (v1 == v2 && i1 < i2);
}

// log(1 + z) for z >= 0 without the cancellation of log(1 + z) at small z
inline float ink_log1p(float z) {
    const float u = 1.f + z;
    return u == 1.f ? z : precise::log(u) * (z / (u - 1.f));
}

kernel void ink_moe_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                          device float* wout [[buffer(2)]], device int* iout [[buffer(3)]], constant OpP& p [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg * INK_ROUTE_SG + sg;
    if (t >= p.a) return;  // uniform over the simdgroup
    const uint E = p.b, K = p.c, NJ = (E + 31u) / 32u;
    device const float* row = logits + (ulong)t * E;
    float sc[32], v[32];
    for (uint j = 0; j < NJ; ++j) {
        const uint e = lane + 32u * j;
        const float l = e < E ? row[e] / p.f1 : 0.f;
        sc[j] = l;
    }
    if (p.e == 0u) {  // softmax over the row
        float mx = -INFINITY;
        for (uint j = 0; j < NJ; ++j)
            if (lane + 32u * j < E) mx = max(mx, sc[j]);
        mx = simd_max(mx);
        float s = 0.f;
        for (uint j = 0; j < NJ; ++j) {
            const bool ok = lane + 32u * j < E;
            sc[j] = ok ? precise::exp(sc[j] - mx) : 0.f;
            s += sc[j];
        }
        s = simd_sum(s);
        for (uint j = 0; j < NJ; ++j) sc[j] = precise::divide(sc[j], s);
    } else if (p.e == 1u) {
        for (uint j = 0; j < NJ; ++j) sc[j] = precise::divide(1.f, 1.f + precise::exp(-sc[j]));
    } else {
        for (uint j = 0; j < NJ; ++j) sc[j] = precise::sqrt(max(sc[j], 0.f) + ink_log1p(precise::exp(-fabs(sc[j]))));
    }
    for (uint j = 0; j < NJ; ++j) {
        const uint e = lane + 32u * j;
        v[j] = e < E ? sc[j] + bias[e] : 0.f;
    }
    uint taken = 0u;
    float pick = 0.f, den = 0.f;  // lane k < K: the unbiased score of pick k
    for (uint k = 0; k < K; ++k) {
        float bv = 0.f, bs = 0.f;
        uint bi = 0xFFFFFFFFu;
        for (uint j = 0; j < NJ; ++j) {
            const uint e = lane + 32u * j;
            if (e < E && !(taken & (1u << j)) && ink_better(v[j], e, bv, bi)) {
                bv = v[j];
                bs = sc[j];
                bi = e;
            }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o), os = simd_shuffle_xor(bs, o);
            const uint oi = simd_shuffle_xor(bi, o);
            if (ink_better(ov, oi, bv, bi)) {
                bv = ov;
                bs = os;
                bi = oi;
            }
        }
        if (bi % 32u == lane) taken |= 1u << (bi / 32u);
        den += bs;  // k order, the same on every lane
        if (lane == k) pick = bs;
        if (lane == k) iout[(ulong)t * K + k] = (int)bi;
    }
    if (lane < K) {
        const float w = (p.d != 0u && K > 1u) ? precise::divide(pick, den + 1e-20f) : pick;
        wout[(ulong)t * K + lane] = w * p.f0;
    }
}

// ---- expert weights ---------------------------------------------------------------------------------------------------

inline float ink_rb(float v) { return bf2f(f2bf(v)); }

// element j (0..31) of a 32-element bf16 chunk held as 16 words (element 2i low half of word i)
inline float ink_xe(thread const uint* xw, uint j) {
    const uint v = xw[j >> 1];
    return as_type<float>((j & 1u) ? (v & 0xFFFF0000u) : (v << 16));
}

// e4m3 scale byte -> value * 2^14 (the e2m1 codes decode as value / 2^14; NaN code -> NaN), as moe_mm.h's moe_sc
inline float ink_nv_sc(uint b) {
    if ((b & 0x7Fu) == 0x7Fu) return as_type<float>(0x7FC00000u);
    return float(as_type<half>((ushort)(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)))) * 4194304.0f;
}

// E8M0 scale byte -> 2^(s - 127) (s = 0: 2^-127, a subnormal; 255: NaN)
inline float ink_e8m0(uint s) {
    if (s == 255u) return as_type<float>(0x7FC00000u);
    return s == 0u ? as_type<float>(0x00400000u) : as_type<float>(s << 23);
}

// One weight chunk (32 consecutive inputs of a row) in registers. NVFP4: w[0] = 16 bytes (element 8q + b = nibble b of word
// q), sc[0..1] = the two block scales (ink_nv_sc). MXFP8: w[0..1] = 32 bytes (element 4q + b = byte b of word q), sc[0] =
// the E8M0 scale. The chain: elements in a fixed order per block, one scale fma per block into the row accumulator.
template <uint FMT>
struct InkChunk {
    uint4 w[2];  // NVFP4 uses w[0], MXFP8 both
    float sc[2]; // NVFP4 both, MXFP8 sc[0]
};

template <uint FMT, bool A16>
inline InkChunk<FMT> ink_load(device const uchar* wrow, device const uchar* srow, uint c) {
    InkChunk<FMT> k;
    if (FMT == 0) {
        device const uchar* wp = wrow + c * 16u;
        if (A16) {
            k.w[0] = *(device const uint4*)wp;
        } else {
            const uint2 a = uint2(*(device const packed_uint2*)wp), b = uint2(*(device const packed_uint2*)(wp + 8));
            k.w[0] = uint4(a.x, a.y, b.x, b.y);
        }
        k.sc[0] = ink_nv_sc(srow[2u * c]);
        k.sc[1] = ink_nv_sc(srow[2u * c + 1u]);
    } else {
        device const uchar* wp = wrow + c * 32u;
        for (uint q = 0; q < 2u; ++q) {
            if (A16) {
                k.w[q] = *(device const uint4*)(wp + 16u * q);
            } else {
                const uint2 a = uint2(*(device const packed_uint2*)(wp + 16u * q)),
                            b = uint2(*(device const packed_uint2*)(wp + 16u * q + 8u));
                k.w[q] = uint4(a.x, a.y, b.x, b.y);
            }
        }
        k.sc[0] = ink_e8m0(srow[c]);
    }
    return k;
}

template <uint FMT>
inline float ink_chunk_dot(thread const InkChunk<FMT>& k, thread const uint* xw, float acc) {
    if (FMT == 0) {
        _Pragma("clang loop unroll(full)")
        for (uint b = 0; b < 2u; ++b) {
            float t = 0.f;
            _Pragma("clang loop unroll(full)")
            for (uint q = 2u * b; q < 2u * b + 2u; ++q) {
                const uint v = k.w[0][q];
                _Pragma("clang loop unroll(full)")
                for (uint n = 0; n < 4u; ++n) {
                    const uint tt = (v >> (4u * n)) & 0x000F000Fu;  // nibbles n and n + 4
                    const float2 c = float2(as_type<half2>(((tt & 0x00070007u) << 9) | ((tt & 0x00080008u) << 12)));
                    t = fma(c.x, ink_xe(xw, 8u * q + n), t);
                    t = fma(c.y, ink_xe(xw, 8u * q + n + 4u), t);
                }
            }
            acc = fma(t, k.sc[b], acc);
        }
        return acc;
    } else {
        float t = 0.f;
        _Pragma("clang loop unroll(full)")
        for (uint q = 0; q < 8u; ++q) {
            const uint v = k.w[q >> 2][q & 3u];
            _Pragma("clang loop unroll(full)")
            for (uint n = 0; n < 2u; ++n) {
                const uint tt = (v >> (8u * n)) & 0x00FF00FFu;  // bytes n and n + 2
                const float2 c = float2(as_type<half2>(((tt & 0x007F007Fu) << 7) | ((tt & 0x00800080u) << 8)));
                t = fma(c.x, ink_xe(xw, 4u * q + n), t);
                t = fma(c.y, ink_xe(xw, 4u * q + n + 2u), t);
            }
        }
        // e4m3 decoded as value / 2^8: the 2^8 goes back exactly before the block scale
        return fma(t * 256.f, k.sc[0], acc);
    }
}

// 32 bf16 activations of chunk c of a row (16-byte aligned rows: 4 vector loads; else bf16 by bf16)
template <bool AX>
inline void ink_xload(device const ushort* xr, uint c, thread uint* xw) {
    if (AX) {
        device const uint4* q = (device const uint4*)(xr + c * 32u);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 4u; ++i) {
            const uint4 v = q[i];
            xw[4 * i] = v.x;
            xw[4 * i + 1] = v.y;
            xw[4 * i + 2] = v.z;
            xw[4 * i + 3] = v.w;
        }
    } else {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) xw[i] = (uint)xr[c * 32u + 2u * i] | ((uint)xr[c * 32u + 2u * i + 1u] << 16);
    }
}

inline float ink_gelu(float x) { return 0.5f * x * (1.f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

// the gate+up epilogue on the rounded g, u (contract: inkling2::ops `MoeExpertsMxfp8`, `situ`)
inline float ink_act(uint epi, float g, float u, float w, float gc, float uc) {
    if (epi == 2u) {
        const float sg = 1.f / (1.f + precise::exp(-g));
        return ink_rb(gc * precise::tanh(g / gc) * sg * uc * precise::tanh(u / uc) * w);
    }
    const float a = epi == 1u ? ink_gelu(g) : g / (1.f + precise::exp(-g));
    return ink_rb(ink_rb(a) * u);
}

// The pairs threadgroup row `pr` serves: pl[0..rows) ascending, all of expert `e`. Union: `pr` must be the first of a run
// of INK_XR pairs of its expert (moe.metal's moe_lead); else just `pr`. false: not a leader / invalid id.
inline bool ink_lead(device const int* ids, uint P, uint E, uint pr, bool uni, uint lane, thread uint& e, thread uint& rows,
                     thread uint* pl) {
    const int id = ids[pr];
    if (id < 0 || (uint)id >= E) return false;
    e = (uint)id;
    if (!uni) {
        rows = 1u;
        pl[0] = pr;
        return true;
    }
    uint rank = 0;
    for (uint base = 0; base < pr; base += 32u) {
        const int v = (base + lane < P) ? ids[base + lane] : -1;
        ulong m = (ulong)simd_ballot(v == id);
        const uint lim = min(pr - base, 32u);
        m &= (lim >= 32u) ? 0xFFFFFFFFul : ((1ul << lim) - 1ul);
        rank += (uint)popcount(m);
    }
    if (rank % INK_XR != 0) return false;
    rows = 0;
    uint seen = 0;
    for (uint base = 0; base < P && rows < INK_XR; base += 32u) {
        const int v = (base + lane < P) ? ids[base + lane] : -1;
        ulong m = (ulong)simd_ballot(v == id);
        while (m != 0ul && rows < INK_XR) {
            const uint b = (uint)ctz(m);
            m &= m - 1ul;
            if (seen >= rank) pl[rows++] = base + b;
            seen++;
        }
    }
    return true;
}

constant constexpr uint INK_SG = 4;   // simdgroups per threadgroup of the expert kernels
constant constexpr uint INK_LG = 8;   // lanes per gate+up row
constant constexpr uint INK_LD = 4;   // lanes per down row

// gate + up. Union: grid (I / 16, pairs), else (pairs, I / 16). h [T*K, I] bf16 by pair.
template <uint FMT, bool A16, bool AX>
inline void ink_gu_impl(device const ushort* x, device const int* ids, device const float* wts, device const uchar* w13,
                        device const uchar* s13, device const float* gsc, device ushort* h, constant OpP& p, uint2 tg,
                        uint sg, uint lane) {
    const uint E = p.b, K = p.c, H = p.d, I = p.e, P = p.a * p.c;
    const bool uni = (p.f & 4u) != 0u;
    const uint rb = uni ? tg.x : tg.y, pr = uni ? tg.y : tg.x;
    uint e, rows;
    uint pl[INK_XR];
    if (!ink_lead(ids, P, E, pr, uni, lane, e, rows, pl)) return;  // uniform over the threadgroup
    const uint r = rb * (INK_SG * 32u / INK_LG) + sg * (32u / INK_LG) + lane / INK_LG, li = lane % INK_LG;
    const uint rr = min(r, I - 1u);  // idle rows compute a copy (every lane takes part in the shuffles)
    constexpr uint WB = FMT == 0 ? 16u : 32u;  // weight bytes per 32-element chunk
    constexpr uint SB = FMT == 0 ? 2u : 1u;    // scale bytes per chunk
    const ulong rg = (ulong)e * 2u * I + rr, ru = rg + I;
    const uint nc = H / 32u;
    device const uchar* wg = w13 + rg * (nc * WB);
    device const uchar* wu = w13 + ru * (nc * WB);
    device const uchar* sgp = s13 + p.s1 + rg * (nc * SB);
    device const uchar* sup = s13 + p.s1 + ru * (nc * SB);
    device const ushort* xr[INK_XR];
    for (uint q = 0; q < INK_XR; ++q) xr[q] = x + p.s0 + (ulong)(pl[min(q, rows - 1u)] / K) * H;
    float ag[INK_XR], au[INK_XR];
    for (uint q = 0; q < INK_XR; ++q) {
        ag[q] = 0.f;
        au[q] = 0.f;
    }
    for (uint c = li; c < nc; c += INK_LG) {
        const InkChunk<FMT> kg = ink_load<FMT, A16>(wg, sgp, c);
        const InkChunk<FMT> ku = ink_load<FMT, A16>(wu, sup, c);
        for (uint q = 0; q < INK_XR; ++q) {
            if (q >= rows) break;
            uint xw[16];
            ink_xload<AX>(xr[q], c, xw);
            ag[q] = ink_chunk_dot<FMT>(kg, xw, ag[q]);
            au[q] = ink_chunk_dot<FMT>(ku, xw, au[q]);
        }
    }
    const uint epi = p.f & 3u;
    const float g0 = FMT == 0 ? gsc[e * 3u] : 1.f, g1 = FMT == 0 ? gsc[e * 3u + 1u] : 1.f;
    for (uint q = 0; q < INK_XR; ++q) {
        if (q >= rows) break;
        float a = ag[q], b = au[q];
        a += simd_shuffle_xor(a, 1);
        b += simd_shuffle_xor(b, 1);
        a += simd_shuffle_xor(a, 2);
        b += simd_shuffle_xor(b, 2);
        a += simd_shuffle_xor(a, 4);
        b += simd_shuffle_xor(b, 4);
        if (li == 0u && r < I) {
            const float g = ink_rb(a * g0), u = ink_rb(b * g1);
            h[(ulong)pl[q] * I + r] = f2bf(ink_act(epi, g, u, wts[pl[q]], p.f0, p.f1));
        }
    }
}

// one down row of one pair: the chain over the I / 32 chunks (lane-strided over INK_LD lanes), reduced
template <uint FMT, bool A16>
inline float ink_dn_row(device const uchar* wr, device const uchar* sr, device const ushort* hr, uint nc, uint li) {
    float s = 0.f;
    for (uint c = li; c < nc; c += INK_LD) {
        const InkChunk<FMT> kw = ink_load<FMT, A16>(wr, sr, c);
        uint xw[16];
        ink_xload<true>(hr, c, xw);
        s = ink_chunk_dot<FMT>(kw, xw, s);
    }
    s += simd_shuffle_xor(s, 1);
    s += simd_shuffle_xor(s, 2);
    return s;
}

// down per pair. Union: grid (H / 32, pairs), else (pairs, H / 32). y [T*K, H] bf16 by pair.
template <uint FMT, bool A16>
inline void ink_dn_u_impl(device const ushort* h, device const int* ids, device const uchar* w2, device const uchar* s2,
                          device const float* gsc, device ushort* y, constant OpP& p, uint2 tg, uint sg, uint lane) {
    const uint E = p.b, H = p.d, I = p.e, P = p.a * p.c;
    const bool uni = (p.f & 4u) != 0u;
    const uint nb = uni ? tg.x : tg.y, pr = uni ? tg.y : tg.x;
    uint e, rows;
    uint pl[INK_XR];
    if (!ink_lead(ids, P, E, pr, uni, lane, e, rows, pl)) return;
    const uint n = nb * (INK_SG * 32u / INK_LD) + sg * (32u / INK_LD) + lane / INK_LD, li = lane % INK_LD;
    const uint nn = min(n, H - 1u);
    constexpr uint WB = FMT == 0 ? 16u : 32u, SB = FMT == 0 ? 2u : 1u;
    const uint nc = I / 32u;
    const ulong rn = (ulong)e * H + nn;
    device const uchar* wr = w2 + rn * (nc * WB);
    device const uchar* sr = s2 + p.s2 + rn * (nc * SB);
    const float g2 = FMT == 0 ? gsc[e * 3u + 2u] : 1.f;
    for (uint q = 0; q < INK_XR; ++q) {
        if (q >= rows) break;
        const float t = ink_dn_row<FMT, A16>(wr, sr, h + (ulong)pl[q] * I, nc, li);
        if (li == 0u && n < H) y[(ulong)pl[q] * H + n] = f2bf(t * g2);
    }
}

// the per-token sum of K slot values in k order (situ: y as is, else rb(y * w)); invalid ids skipped
inline float ink_tok_sum(thread const float* ys, device const int* ids, device const float* wts, uint t, uint K, uint E,
                         uint epi) {
    float acc = 0.f;
    for (uint k = 0; k < K; ++k) {
        const uint pr = t * K + k;
        const int id = ids[pr];
        if (id < 0 || (uint)id >= E) continue;
        acc += epi == 2u ? ys[k] : ink_rb(ys[k] * wts[pr]);
    }
    return acc;
}

// down + the token sum: grid (T, H / 8). A threadgroup: 8 output rows of token tg.x; simdgroup sg serves slots sg, sg + 4, ..
// (8 rows x INK_LD lanes each), the y values meet in threadgroup memory, 8 threads sum them in k order.
template <uint FMT, bool A16, typename OutT>
inline void ink_dn_f_impl(device const ushort* h, device const int* ids, device const float* wts, device const uchar* w2,
                          device const uchar* s2, device const float* gsc, device OutT* out, constant OpP& p,
                          threadgroup float* ys, uint2 tg, uint sg, uint lane, uint tid) {
    const uint E = p.b, K = p.c, H = p.d, I = p.e, t = tg.x;
    const uint n = tg.y * 8u + lane / INK_LD, li = lane % INK_LD;
    const uint nn = min(n, H - 1u);
    constexpr uint WB = FMT == 0 ? 16u : 32u, SB = FMT == 0 ? 2u : 1u;
    const uint nc = I / 32u;
    for (uint k = sg; k < K; k += INK_SG) {
        const uint pr = t * K + k;
        const int id = ids[pr];
        if (id < 0 || (uint)id >= E) continue;  // uniform over the simdgroup
        const ulong rn = (ulong)id * H + nn;
        const float s = ink_dn_row<FMT, A16>(w2 + rn * (nc * WB), s2 + p.s2 + rn * (nc * SB), h + (ulong)pr * I, nc, li);
        const float g2 = FMT == 0 ? gsc[id * 3 + 2] : 1.f;
        if (li == 0u) ys[k * 8u + lane / INK_LD] = ink_rb(s * g2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u && tg.y * 8u + tid < H) {
        float yv[32];
        for (uint k = 0; k < K; ++k) yv[k] = ys[k * 8u + tid];
        store_out(out + p.s3, (ulong)t * H + tg.y * 8u + tid, ink_tok_sum(yv, ids, wts, t, K, E, p.f & 3u));
    }
}

// y rows -> out: one thread per (t, n), grid T * H
template <typename OutT>
inline void ink_gather_impl(device const ushort* y, device const int* ids, device const float* wts, device OutT* out,
                            constant OpP& p, uint i) {
    const uint E = p.b, K = p.c, H = p.d;
    if (i >= p.a * H) return;
    const uint t = i / H, n = i % H;
    float yv[32];
    for (uint k = 0; k < K; ++k) yv[k] = bf2f(y[(ulong)(t * K + k) * H + n]);
    store_out(out + p.s3, i, ink_tok_sum(yv, ids, wts, t, K, E, p.f & 3u));
}

#define INK_GU(NAME, FMT, A16, AX)                                                                                      \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const int* ids [[buffer(1)]],                        \
                     device const float* wts [[buffer(2)]], device const uchar* w13 [[buffer(3)]],                      \
                     device const uchar* s13 [[buffer(4)]], device const float* gsc [[buffer(5)]],                      \
                     device ushort* h [[buffer(6)]], constant OpP& p [[buffer(7)]],                                      \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],             \
                     uint lane [[thread_index_in_simdgroup]]) {                                                         \
        ink_gu_impl<FMT, A16, AX>(x, ids, wts, w13, s13, gsc, h, p, tg, sg, lane);                                      \
    }
#define INK_DN_U(NAME, FMT, A16)                                                                                        \
    kernel void NAME(device const ushort* h [[buffer(0)]], device const int* ids [[buffer(1)]],                        \
                     device const uchar* w2 [[buffer(2)]], device const uchar* s2 [[buffer(3)]],                        \
                     device const float* gsc [[buffer(4)]], device ushort* y [[buffer(5)]],                             \
                     constant OpP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                          \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {             \
        ink_dn_u_impl<FMT, A16>(h, ids, w2, s2, gsc, y, p, tg, sg, lane);                                               \
    }
#define INK_DN_F(NAME, FMT, A16, OUT)                                                                                   \
    kernel void NAME(device const ushort* h [[buffer(0)]], device const int* ids [[buffer(1)]],                        \
                     device const float* wts [[buffer(2)]], device const uchar* w2 [[buffer(3)]],                       \
                     device const uchar* s2 [[buffer(4)]], device const float* gsc [[buffer(5)]],                       \
                     device OUT* out [[buffer(6)]], constant OpP& p [[buffer(7)]],                                       \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],             \
                     uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {               \
        threadgroup float ys[32 * 8];                                                                                   \
        ink_dn_f_impl<FMT, A16, OUT>(h, ids, wts, w2, s2, gsc, out, p, ys, tg, sg, lane, tid);                          \
    }

// suffixes: nv = NVFP4, mx = MXFP8; a = 16-byte aligned weights, u = 4-byte aligned; x = 16-byte aligned activations
INK_GU(ink_moe_gu_nv_a_x, 0, true, true)
INK_GU(ink_moe_gu_nv_a, 0, true, false)
INK_GU(ink_moe_gu_nv_u_x, 0, false, true)
INK_GU(ink_moe_gu_nv_u, 0, false, false)
INK_GU(ink_moe_gu_mx_a_x, 1, true, true)
INK_GU(ink_moe_gu_mx_a, 1, true, false)
INK_GU(ink_moe_gu_mx_u_x, 1, false, true)
INK_GU(ink_moe_gu_mx_u, 1, false, false)
INK_DN_U(ink_moe_dn_u_nv_a, 0, true)
INK_DN_U(ink_moe_dn_u_nv_u, 0, false)
INK_DN_U(ink_moe_dn_u_mx_a, 1, true)
INK_DN_U(ink_moe_dn_u_mx_u, 1, false)
INK_DN_F(ink_moe_dn_f_nv_a_bf16, 0, true, ushort)
INK_DN_F(ink_moe_dn_f_nv_u_bf16, 0, false, ushort)
INK_DN_F(ink_moe_dn_f_mx_a_bf16, 1, true, ushort)
INK_DN_F(ink_moe_dn_f_mx_u_bf16, 1, false, ushort)
INK_DN_F(ink_moe_dn_f_nv_a_f32, 0, true, float)
INK_DN_F(ink_moe_dn_f_nv_u_f32, 0, false, float)
INK_DN_F(ink_moe_dn_f_mx_a_f32, 1, true, float)
INK_DN_F(ink_moe_dn_f_mx_u_f32, 1, false, float)

#define INK_GATHER(NAME, OUT)                                                                                           \
    kernel void NAME(device const ushort* y [[buffer(0)]], device const int* ids [[buffer(1)]],                        \
                     device const float* wts [[buffer(2)]], device OUT* out [[buffer(3)]],                              \
                     constant OpP& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {                               \
        ink_gather_impl<OUT>(y, ids, wts, out, p, i);                                                                   \
    }
INK_GATHER(ink_moe_gather_bf16, ushort)
INK_GATHER(ink_moe_gather_f32, float)
