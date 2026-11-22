// Grouped (prefill) Inkling-2-Mini experts on the M5 neural accelerators, third generation (lane ink-prefill; launcher
// src/inkprefill.rs). Same grouping (moe.metal's `moe_group`: pairs sorted by expert, tiles of BM rows), weight decode (each
// NVFP4 / MXFP8 block to bf16 once per tile, exact) and epilogues as inkling2_moe_nax.metal; the K loop is restructured:
//
//  * The A operand is read by the accelerator straight from device memory (a [P, KD] bf16 tensor in slot order), not staged
//    through threadgroup memory: `ink_moe_nax3_gather` first copies the x row of every pair into slot order (gate+up; the
//    down reads h, already in slot order). Threadgroup memory then holds only the decoded weights.
//  * The decoded weight tile is double-buffered: one barrier per K step, the decode of step k + 1 overlaps the matmul of step
//    k, the raw words of step k + 2 are in flight meanwhile.
//  * The weight tile is BN rows wide (64 or 128). For gate+up, each simdgroup's SN columns are SN / 2 gate columns followed
//    by the up columns of the same outputs, so the epilogue finds the pair in one simdgroup's sub-tile.
//
// The accumulation is the accelerator's over BK steps; results are admitted against the f64 reference (not bitwise the first
// generation or the decode kernels), as every NAX prefill path is.
//
// Bindings (mm): 0 A (xs for gate+up, h for down), 1 pairs, 2 off, 3 tiles, 4 W, 5 S, 6 gsc, 7 wts, 8 out, 9 OpP
// (a = T, b = E, c = K, d = H, e = I, f = epilogue, s1 / s2 = scale skips, f0 / f1 = caps). Grid (out columns / (BN / 2 or
// BN), max tiles), 128 threads.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "moe_mm.h"
using namespace metal;
using namespace mpp::tensor_ops;

inline float in3_gelu(float x) { return 0.5f * x * (1.f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

inline float in3_act(uint epi, float g, float u, float w, float gc, float uc) {
    if (epi == 2u) {
        const float sg = 1.f / (1.f + precise::exp(-g));
        return moe_rb(gc * precise::tanh(g / gc) * sg * uc * precise::tanh(u / uc) * w);
    }
    const float a = epi == 1u ? in3_gelu(g) : g / (1.f + precise::exp(-g));
    return moe_rb(moe_rb(a) * u);
}

// xs[s] = x[pairs[s] / K] for the valid slots s < off[E]; one thread per 8 elements. x may sit at an odd element (s0).
kernel void ink_moe_nax3_gather(device const ushort* x [[buffer(0)]], device const int* pairs [[buffer(1)]],
                                device const int* off [[buffer(2)]], device uint4* xs [[buffer(3)]],
                                constant OpP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    const uint H = p.d, K = p.c, E = p.b;
    const uint s = g.y, c = g.x;
    if (c >= H / 8u || s >= (uint)off[E]) return;
    device const packed_ushort4* q4 = (device const packed_ushort4*)(x + p.s0 + (ulong)((uint)pairs[s] / K) * H + c * 8u);
    const ushort4 a = ushort4(q4[0]), b = ushort4(q4[1]);
    xs[(ulong)s * (H / 8u) + c] = uint4(a.x | ((uint)a.y << 16), a.z | ((uint)a.w << 16), b.x | ((uint)b.y << 16), b.z | ((uint)b.w << 16));
}

template <uint NB>
inline void in3_fetch_mx(device const uchar* wp, device const uchar* sp, bool bok, uint bq, uint k0, uint KD,
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

inline void in3_blk_mx(uint4 v, uint s, threadgroup bfloat* dst) {
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

template <uint FMT, uint MODE, uint BM, uint BN, uint BK>
inline void in3_mm_impl(device bfloat* A, device const int* pairs, device const int* off, device const int* tiles,
                        device const uchar* W, device const uchar* S, device const float* gsc, device const float* wts,
                        device ushort* out, constant OpP& p, threadgroup float* smem, uint2 tg, uint tid, uint sg) {
    static_assert(BM == 32 || BM == 64, "tile rows");
    static_assert(BN == 64 || BN == 128, "tile columns");
    constexpr uint SM = BM / 2, SN = BN / 2;   // 2 x 2 simdgroups
    constexpr uint LD = BK + 8;
    constexpr uint TPR = 128 / BN;             // loader threads per weight row
    constexpr uint NB = BK / 16 / TPR;         // 16-element blocks per thread and step
    static_assert(NB >= 1, "BK too small for the loader");
    constexpr uint BSZ = BN * LD;
    constexpr uint LC = BN + 4;                // floats per epilogue row
    const uint E = p.b, H = p.d, I = p.e;
    const uint KD = MODE == 0 ? H : I;
    uint e, s0, rows;
    if (!moe_tile(off, tiles, E, BM, tg.y, e, s0, rows)) return;
    const uint P = (uint)off[E];
    threadgroup bfloat* Bs0 = (threadgroup bfloat*)smem;

    // B loader: smem row br, blocks bq .. bq + NB - 1 of each step
    const uint br = tid / TPR, bq = (tid % TPR) * NB;
    uint wcol;  // output column of smem row br
    ulong wrow;
    if (MODE == 0) {
        const uint sn = br / SN, j = br % SN;
        wcol = tg.x * (BN / 2) + sn * (SN / 2) + j % (SN / 2);
        wrow = (ulong)e * 2 * I + (j >= SN / 2 ? I : 0u) + min(wcol, I - 1u);
    } else {
        wcol = tg.x * BN + br;
        wrow = (ulong)e * H + min(wcol, H - 1u);
    }
    const bool bok = wcol < (MODE == 0 ? I : H);
    device const uchar* wp = W + wrow * (FMT == 0 ? KD / 2 : KD);
    device const uchar* sp = S + (MODE == 0 ? p.s1 : p.s2) + wrow * (FMT == 0 ? KD / 16 : KD / 32);
    uint2 wv[NB];
    uint4 wx[NB];
    uint sv[NB];
    auto fetch = [&](uint k1) {
        if (FMT == 0) moe_fetch<NB>(wp, sp, bok, bq, k1, KD, wv, sv);
        else in3_fetch_mx<NB>(wp, sp, bok, bq, k1, KD, wx, sv);
    };
    auto stage = [&](uint b) {
        threadgroup bfloat* Bs = Bs0 + b * BSZ + br * LD;
        for (uint q = 0; q < NB; ++q) {
            if (FMT == 0) moe_blk_bf16(wv[q], moe_sc(sv[q]), Bs + (bq + q) * 16u);
            else in3_blk_mx(wx[q], sv[q], Bs + (bq + q) * 16u);
        }
    };

    const uint sr = (sg / 2u) * SM, sc = (sg % 2u) * SN;
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A, dextents<int, 2>(int(KD), int(P)));
    constexpr auto desc = matmul2d_descriptor(SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    using ext_b = extents<int, BK, SN>;
    using ten_b = tensor<threadgroup bfloat, ext_b, tensor_inline>;
    using ten_a = decltype(tA.template slice<BK, SM>(0, 0));
    auto ct = op.template get_destination_cooperative_tensor<ten_a, ten_b, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) ct[i] = 0.f;
    const int m0 = int(s0 + sr);
    const bool inside = s0 + sr + SM <= P;  // the sub-tile's rows are all in the tensor (else the bounds-checked slice)

    fetch(0u);
    stage(0u);
    if (BK < KD) fetch(BK);
    uint b = 0u;
    for (uint k0 = 0; k0 < KD; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < KD) {
            stage(b ^ 1u);
            if (k0 + 2 * BK < KD) fetch(k0 + 2 * BK);
        }
        array<int, 2> st = {1, int(LD)};
        ext_b eb;
        ten_b tB(Bs0 + b * BSZ + sc * LD, eb, st);
        if (inside) {
            auto sA = tA.template slice<BK, SM>(int(k0), m0);
            op.run(sA, tB, ct);
        } else if (m0 < int(P)) {
            auto sA = tA.slice(int(k0), m0);
            op.run(sA, tB, ct);
        }
        b ^= 1u;
    }
    // epilogue through threadgroup memory, one row half (sr) at a time: C [SM][LC] f32
    for (uint hp = 0; hp < 2; ++hp) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg / 2u == hp) {
            extents<int, SN, SM> ec;
            array<int, 2> st = {1, int(LC)};
            tensor<threadgroup float, extents<int, SN, SM>, tensor_inline> tC(smem + sc, ec, st);
            ct.store(tC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint r0 = hp * SM;
        if (MODE == 0) {
            const float g0 = FMT == 0 ? gsc[e * 3] : 1.f, g1 = FMT == 0 ? gsc[e * 3 + 1] : 1.f;
            const uint n0 = tg.x * (BN / 2);
            for (uint q = tid; q < SM * (BN / 2); q += 128u) {
                const uint r = q / (BN / 2), c = q % (BN / 2);
                if (r0 + r >= rows || n0 + c >= I) continue;
                const uint sn = c / (SN / 2), jj = c % (SN / 2);
                const float g = moe_rb(smem[r * LC + sn * SN + jj] * g0), u = moe_rb(smem[r * LC + sn * SN + SN / 2 + jj] * g1);
                out[(ulong)(s0 + r0 + r) * I + n0 + c] = f2bf(in3_act(p.f, g, u, wts[pairs[s0 + r0 + r]], p.f0, p.f1));
            }
        } else {
            const float g2 = FMT == 0 ? gsc[e * 3 + 2] : 1.f;
            const uint n0 = tg.x * BN;
            for (uint q = tid; q < SM * BN; q += 128u) {
                const uint r = q / BN, c = q % BN;
                if (r0 + r >= rows || n0 + c >= H) continue;
                out[(ulong)pairs[s0 + r0 + r] * H + n0 + c] = f2bf(smem[r * LC + c] * g2);
            }
        }
    }
}

#define IN3_SMEM(BM, BN, BK) \
    (((2 * (BN) * ((BK) + 8) * 2) > ((BM) / 2) * ((BN) + 4) * 4 ? (2 * (BN) * ((BK) + 8) * 2) : ((BM) / 2) * ((BN) + 4) * 4) / 16)

#define IN3_MM(NAME, FMT, MODE, BM, BN, BK)                                                                             \
    kernel void NAME(device bfloat* A [[buffer(0)]], device const int* pairs [[buffer(1)]],                             \
                     device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],                        \
                     device const uchar* W [[buffer(4)]], device const uchar* S [[buffer(5)]],                          \
                     device const float* gsc [[buffer(6)]], device const float* wts [[buffer(7)]],                      \
                     device ushort* out [[buffer(8)]], constant OpP& p [[buffer(9)]],                                   \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],               \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                                      \
        threadgroup float4 smem[IN3_SMEM(BM, BN, BK)];                                                                  \
        in3_mm_impl<FMT, MODE, BM, BN, BK>(A, pairs, off, tiles, W, S, gsc, wts, out, p, (threadgroup float*)smem, tg, tid, sg); \
    }

// Keep in step with NAX3 in src/inkprefill.rs. Name: ink_moe_nax3_<gu|dn>_<nv|mx>_m<BM>_n<BN>_k<BK>.
#define IN3_SET(FN, FMT, BM, BN, BK)                                    \
    IN3_MM(ink_moe_nax3_gu_##FN##_m##BM##_n##BN##_k##BK, FMT, 0, BM, BN, BK) \
    IN3_MM(ink_moe_nax3_dn_##FN##_m##BM##_n##BN##_k##BK, FMT, 1, BM, BN, BK)

IN3_SET(nv, 0, 64, 64, 64)
IN3_SET(nv, 0, 64, 128, 32)
IN3_SET(nv, 0, 32, 128, 32)
IN3_SET(nv, 0, 32, 64, 64)
IN3_SET(mx, 1, 64, 64, 64)
IN3_SET(mx, 1, 64, 128, 32)
