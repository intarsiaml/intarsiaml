// Grouped (prefill) Inkling-2-Mini experts on the M5 neural accelerators, second generation (lane ink-prefill; launcher
// src/inkprefill.rs). Same bindings, grid, grouping (moe.metal's `moe_group`), staging layout, decode and epilogue as
// inkling2_moe_nax.metal's `inx_mm_impl`, so every variant computes the same products in the same accelerator K order per
// step; what changes is how the K loop is pipelined:
//
//  * V & 1: the A chunks (gathered x rows / h rows) of step k + 1 are loaded into registers while step k runs, like the
//    weight words already were (the first generation loaded A after the barrier and stored it at once: a device-latency
//    stall per step).
//  * V & 2: A and B tiles double-buffered in threadgroup memory: one barrier per K step instead of two, and the staging of
//    step k + 1 overlaps the accelerator work of step k (needs V & 1; BK 32 at 64 rows to fit 32 kB).
//  * V & 4 / 8 / 16 (diagnostics only, wrong results): skip the weight decode / the matmul / the A staging, to split the
//    time of a step into its parts.
//
// The K accumulation per output is the matmul2d order over the same BK steps, so for a given BK the variants give the same
// bits as each other; a different BK changes the step boundaries (results then differ in the last bits, admitted against
// the f64 reference like the first generation).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "moe_mm.h"
using namespace metal;
using namespace mpp::tensor_ops;

inline float in2_gelu(float x) { return 0.5f * x * (1.f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

inline float in2_act(uint epi, float g, float u, float w, float gc, float uc) {
    if (epi == 2u) {
        const float sg = 1.f / (1.f + precise::exp(-g));
        return moe_rb(gc * precise::tanh(g / gc) * sg * uc * precise::tanh(u / uc) * w);
    }
    const float a = epi == 1u ? in2_gelu(g) : g / (1.f + precise::exp(-g));
    return moe_rb(moe_rb(a) * u);
}

template <uint NB>
inline void in2_fetch_mx(device const uchar* wp, device const uchar* sp, bool bok, uint bq, uint k0, uint KD,
                         thread uint4* wv, thread uint* sv) {
    for (uint q = 0; q < NB; ++q) {
        const uint gk = k0 + (bq + q) * 16u;
        const bool ok = bok && gk < KD;
        if (ok) {
            const uint2 a = uint2(*(device const packed_uint2*)(wp + gk)), b = uint2(*(device const packed_uint2*)(wp + gk + 8));
            wv[q] = uint4(a.x, a.y, b.x, b.y);
        } else {
            wv[q] = uint4(0u);
        }
        sv[q] = ok ? (uint)sp[gk / 32] : 127u;
    }
}

inline void in2_blk_mx(uint4 v, uint s, threadgroup bfloat* dst) {
    const float sc = s == 0u ? as_type<float>(0x00400000u) : (s == 255u ? as_type<float>(0x7FC00000u) : as_type<float>(s << 23));
    uint o[8];
    for (int i = 0; i < 4; ++i) {
        const uint t0 = v[i] & 0x00FF00FFu, t1 = (v[i] >> 8) & 0x00FF00FFu;
        const float2 c0 = float2(as_type<half2>(((t0 & 0x007F007Fu) << 7) | ((t0 & 0x00800080u) << 8))) * 256.f * sc;
        const float2 c1 = float2(as_type<half2>(((t1 & 0x007F007Fu) << 7) | ((t1 & 0x00800080u) << 8))) * 256.f * sc;
        o[2 * i] = moe_pack(c0.x, c1.x);
        o[2 * i + 1] = moe_pack(c0.y, c1.y);
    }
    threadgroup uint4* d = (threadgroup uint4*)dst;
    d[0] = uint4(o[0], o[1], o[2], o[3]);
    d[1] = uint4(o[4], o[5], o[6], o[7]);
}

template <uint NA, uint AC, uint BM>
inline void in2_load_a(device const ushort* const thread* ap, uint tid, uint k0, uint KD, thread uint4* av) {
    for (uint j = 0; j < NA; ++j) {
        const uint c = tid + 128u * j;
        uint4 v = uint4(0u);
        if (c < BM * AC && ap[j] != nullptr && k0 + (c % AC) * 8u < KD) {
            device const packed_ushort4* q4 = (device const packed_ushort4*)(ap[j] + k0);
            const ushort4 a = ushort4(q4[0]), b = ushort4(q4[1]);
            v = uint4(a.x | ((uint)a.y << 16), a.z | ((uint)a.w << 16), b.x | ((uint)b.y << 16), b.z | ((uint)b.w << 16));
        }
        av[j] = v;
    }
}

template <uint FMT, uint MODE, uint BM, uint BK, uint V>
inline void in2_mm_impl(device const ushort* A, device const int* pairs, device const int* off, device const int* tiles,
                        device const uchar* W, device const uchar* S, device const float* gsc, device const float* wts,
                        device ushort* out, constant OpP& p, threadgroup float* smem, uint2 tg, uint tid, uint sg) {
    constexpr bool PA = (V & 1u) != 0, DB = (V & 2u) != 0;
    constexpr bool NODEC = (V & 4u) != 0, NOMM = (V & 8u) != 0, NOA = (V & 16u) != 0;
    constexpr uint TM = BM < 16 ? 16 : BM;
    constexpr uint GM = BM >= 32 ? 2 : 1;
    constexpr uint SM = TM / GM, SN = MOE_WR / (4 / GM);
    constexpr uint LD = BK + 8;
    constexpr uint AC = BK / 8;
    constexpr uint NA = (BM * AC + 127) / 128;
    constexpr uint NB = BK / 32;
    constexpr uint NBUF = DB ? 2 : 1;
    constexpr uint ASZ = TM * LD, BSZ = MOE_WR * LD;
    static_assert(!DB || PA, "double buffering needs the A prefetch");
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    const uint KD = MODE == 0 ? H : I;
    uint e, s0, rows;
    if (!moe_tile(off, tiles, E, BM, tg.y, e, s0, rows)) return;
    threadgroup bfloat* As0 = (threadgroup bfloat*)smem;
    threadgroup bfloat* Bs0 = As0 + NBUF * ASZ;
    if (BM < TM) {
        for (uint b = 0; b < NBUF; ++b)
            for (uint i = tid; i < (TM - BM) * LD / 8; i += 128u) *(threadgroup uint4*)(As0 + b * ASZ + BM * LD + i * 8u) = uint4(0u);
    }
    device const ushort* ap[NA];
    for (uint j = 0; j < NA; ++j) {
        const uint c = tid + 128u * j, r = c / AC;
        ap[j] = nullptr;
        if (c < BM * AC && r < rows)
            ap[j] = MODE == 0 ? A + p.s0 + (ulong)((uint)pairs[s0 + r] / K) * H + (c % AC) * 8u
                              : A + (ulong)(s0 + r) * I + (c % AC) * 8u;
    }
    const uint br = tid / 2u, bq = (tid % 2u) * NB;
    const uint n0 = tg.x * (MODE == 0 ? MOE_WR / 2 : MOE_WR);
    const uint bn = MODE == 0 ? n0 + br % 32u : n0 + br;
    const bool bok = bn < (MODE == 0 ? I : H);
    const ulong wrow = MODE == 0 ? (ulong)e * 2 * I + (br / 32u) * I + min(bn, I - 1u) : (ulong)e * H + min(bn, H - 1u);
    device const uchar* wp = W + wrow * (FMT == 0 ? KD / 2 : KD);
    device const uchar* sp = S + (MODE == 0 ? p.s1 : p.s2) + wrow * (FMT == 0 ? KD / 16 : KD / 32);
    uint2 wv[NB];
    uint4 wx[NB];
    uint sv[NB];
    uint4 av[PA ? NA : 1];
    if (FMT == 0) moe_fetch<NB>(wp, sp, bok, bq, 0u, KD, wv, sv);
    else in2_fetch_mx<NB>(wp, sp, bok, bq, 0u, KD, wx, sv);
    if (PA) in2_load_a<NA, AC, BM>(ap, tid, 0u, KD, av);
    const uint sr = (sg / (4 / GM)) * SM, sc = (sg % (4 / GM)) * SN;

    constexpr auto desc = matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    using ext_a = extents<int, BK, SM>;
    using ext_b = extents<int, BK, SN>;
    using ten_a = tensor<threadgroup bfloat, ext_a, tensor_inline>;
    using ten_b = tensor<threadgroup bfloat, ext_b, tensor_inline>;
    auto ct = op.template get_destination_cooperative_tensor<ten_a, ten_b, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) ct[i] = 0.f;

    // stage the registers (A chunks, raw weight words) of one step into buffer b
    auto stage = [&](uint b, uint k0) {
        threadgroup bfloat* As = As0 + b * ASZ;
        threadgroup bfloat* Bs = Bs0 + b * BSZ;
        if (!NOA) {
            for (uint j = 0; j < NA; ++j) {
                const uint c = tid + 128u * j;
                if (c >= BM * AC) break;
                uint4 v;
                if (PA) {
                    v = av[j];
                } else {
                    v = uint4(0u);
                    if (ap[j] != nullptr && k0 + (c % AC) * 8u < KD) {
                        device const packed_ushort4* q4 = (device const packed_ushort4*)(ap[j] + k0);
                        const ushort4 a = ushort4(q4[0]), bb = ushort4(q4[1]);
                        v = uint4(a.x | ((uint)a.y << 16), a.z | ((uint)a.w << 16), bb.x | ((uint)bb.y << 16), bb.z | ((uint)bb.w << 16));
                    }
                }
                *(threadgroup uint4*)(As + (c / AC) * LD + (c % AC) * 8u) = v;
            }
        }
        if (!NODEC) {
            for (uint q = 0; q < NB; ++q) {
                if (FMT == 0) moe_blk_bf16(wv[q], moe_sc(sv[q]), Bs + br * LD + (bq + q) * 16u);
                else in2_blk_mx(wx[q], sv[q], Bs + br * LD + (bq + q) * 16u);
            }
        } else {
            // keep the loads live: one word per thread into the tile
            uint acc = sv[0];
            for (uint q = 0; q < NB; ++q) acc ^= FMT == 0 ? wv[q].x : wx[q].x;
            if (acc == 0x9E3779B9u) *(threadgroup uint*)(Bs + br * LD + bq * 16u) = acc;
        }
    };
    auto fetch = [&](uint k1) {
        if (FMT == 0) moe_fetch<NB>(wp, sp, bok, bq, k1, KD, wv, sv);
        else in2_fetch_mx<NB>(wp, sp, bok, bq, k1, KD, wx, sv);
        if (PA) in2_load_a<NA, AC, BM>(ap, tid, k1, KD, av);
    };
    auto mma = [&](uint b) {
        if (NOMM) return;
        ext_a ea;
        ext_b eb;
        array<int, 2> st = {1, int(LD)};
        ten_a tA(As0 + b * ASZ + sr * LD, ea, st);
        ten_b tB(Bs0 + b * BSZ + sc * LD, eb, st);
        op.run(tA, tB, ct);
    };

    if (DB) {
        stage(0u, 0u);
        if (BK < KD) fetch(BK);
        uint b = 0u;
        for (uint k0 = 0; k0 < KD; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k0 + BK < KD) {
                stage(b ^ 1u, k0 + BK);
                if (k0 + 2 * BK < KD) fetch(k0 + 2 * BK);
            }
            mma(b);
            b ^= 1u;
        }
    } else {
        for (uint k0 = 0; k0 < KD; k0 += BK) {
            stage(0u, k0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k0 + BK < KD) fetch(k0 + BK);
            mma(0u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (DB) threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        extents<int, SN, SM> ec;
        array<int, 2> st = {1, int(MOE_LC)};
        tensor<threadgroup float, extents<int, SN, SM>, tensor_inline> tC(smem + sr * MOE_LC + sc, ec, st);
        ct.store(tC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (MODE == 0) {
        const float g0 = FMT == 0 ? gsc[e * 3] : 1.f, g1 = FMT == 0 ? gsc[e * 3 + 1] : 1.f;
        for (uint q = tid; q < BM * 32u; q += 128u) {
            const uint r = q / 32u, c = q % 32u;
            if (r >= rows || n0 + c >= I) continue;
            const float g = moe_rb(smem[r * MOE_LC + c] * g0), u = moe_rb(smem[r * MOE_LC + 32 + c] * g1);
            out[(ulong)(s0 + r) * I + n0 + c] = f2bf(in2_act(p.f, g, u, wts[pairs[s0 + r]], p.f0, p.f1));
        }
    } else {
        const float g2 = FMT == 0 ? gsc[e * 3 + 2] : 1.f;
        for (uint q = tid; q < BM * MOE_WR; q += 128u) {
            const uint r = q / MOE_WR, c = q % MOE_WR;
            if (r >= rows || n0 + c >= H) continue;
            out[(ulong)pairs[s0 + r] * H + n0 + c] = f2bf(smem[r * MOE_LC + c] * g2);
        }
    }
}

// threadgroup memory in float4: NBUF x (A + B) staged tiles, reused by the f32 epilogue
#define IN2_SMEM(TM, BK, NBUF) \
    ((((NBUF) * ((TM) + MOE_WR) * ((BK) + 8) * 2) > (TM) * MOE_LC * 4 ? ((NBUF) * ((TM) + MOE_WR) * ((BK) + 8) * 2) : (TM) * MOE_LC * 4) / 16)

#define IN2_MM(NAME, FMT, MODE, BM, BK, V)                                                                              \
    kernel void NAME(device const ushort* A [[buffer(0)]], device const int* pairs [[buffer(1)]],                       \
                     device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],                        \
                     device const uchar* W [[buffer(4)]], device const uchar* S [[buffer(5)]],                          \
                     device const float* gsc [[buffer(6)]], device const float* wts [[buffer(7)]],                      \
                     device ushort* out [[buffer(8)]], constant OpP& p [[buffer(9)]],                                   \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],               \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                                      \
        threadgroup float4 smem[IN2_SMEM(BM < 16 ? 16 : BM, BK, ((V) & 2u) ? 2 : 1)];                                   \
        in2_mm_impl<FMT, MODE, BM, BK, V>(A, pairs, off, tiles, W, S, gsc, wts, out, p, (threadgroup float*)smem, tg, tid, sg); \
    }

// Keep in step with NAX2 in src/inkprefill.rs. Name: ink_moe_nax2_<gu|dn>_<nv|mx>_m<BM>_k<BK>_v<V>.
#define IN2_SET(FN, FMT, BM, BK, V)                                  \
    IN2_MM(ink_moe_nax2_gu_##FN##_m##BM##_k##BK##_v##V, FMT, 0, BM, BK, V) \
    IN2_MM(ink_moe_nax2_dn_##FN##_m##BM##_k##BK##_v##V, FMT, 1, BM, BK, V)

IN2_SET(nv, 0, 64, 64, 0)
IN2_SET(nv, 0, 64, 64, 1)
IN2_SET(nv, 0, 64, 32, 1)
IN2_SET(nv, 0, 64, 32, 3)
IN2_SET(nv, 0, 32, 64, 3)
IN2_SET(nv, 0, 64, 64, 4)
IN2_SET(nv, 0, 64, 64, 8)
IN2_SET(nv, 0, 64, 64, 16)
IN2_SET(nv, 0, 64, 64, 28)
IN2_SET(mx, 1, 64, 64, 1)
IN2_SET(mx, 1, 64, 32, 3)
