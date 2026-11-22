// Grouped (prefill) Inkling-2-Mini experts on the M5 neural accelerators (lane ink-m-moe; launcher src/inkmoe.rs): moe_nax.metal's
// `moe_mm_nax_impl` with the weight format (NVFP4, MXFP8) and the gate+up epilogue (silu, gelu, situ with the routing weight
// folded in before w2) as parameters. Compiled -std=metal4.0 into intarsia_metal4.metallib (build.rs METAL4_SOURCES). The pairs
// come grouped by expert from moe.metal's `moe_group` (tiles of BM rows); x rows are gathered by pair, every weight block is
// decoded to bf16 once per tile (both formats are exact in bf16: e2m1 x e4m3 and e4m3 x 2^k), the accelerator accumulates in
// f32. h by slot, y[pair] = bf16(acc * gs) (gs = 1 for MXFP8); the token sum is inkling2_moe.metal's gather (k order).
// Not bitwise the decode kernels (the accelerator's K order): prefill only.
//
// Arguments: OpP; a = T, b = E, c = K, d = H, e = I, f = epilogue (0 silu, 1 gelu, 2 situ); s0 = x skip, s1 = w13 scales
// skip, s2 = w2 scales skip; f0 = gate_cap, f1 = up_cap.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "moe_mm.h"
using namespace metal;
using namespace mpp::tensor_ops;

inline float inx_gelu(float x) { return 0.5f * x * (1.f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

inline float inx_act(uint epi, float g, float u, float w, float gc, float uc) {
    if (epi == 2u) {
        const float sg = 1.f / (1.f + precise::exp(-g));
        return moe_rb(gc * precise::tanh(g / gc) * sg * uc * precise::tanh(u / uc) * w);
    }
    const float a = epi == 1u ? inx_gelu(g) : g / (1.f + precise::exp(-g));
    return moe_rb(moe_rb(a) * u);
}

// MXFP8: raw bytes and scale bytes of the 16-element groups bq .. bq + NB - 1 of the K step at k0 (zero past the matrix)
template <uint NB>
inline void inx_fetch_mx(device const uchar* wp, device const uchar* sp, bool bok, uint bq, uint k0, uint KD,
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

// 16 e4m3 bytes (element 4i + b = byte b of word i) times 2^(s - 127) -> 16 bf16 at dst (16-byte aligned); exact
inline void inx_blk_mx(uint4 v, uint s, threadgroup bfloat* dst) {
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

template <uint FMT, uint MODE, uint BM, uint BK>
inline void inx_mm_impl(device const ushort* A, device const int* pairs, device const int* off, device const int* tiles,
                        device const uchar* W, device const uchar* S, device const float* gsc, device const float* wts,
                        device ushort* out, constant OpP& p, threadgroup float* smem, uint2 tg, uint tid, uint sg) {
    constexpr uint TM = BM < 16 ? 16 : BM;
    constexpr uint GM = BM >= 32 ? 2 : 1;
    constexpr uint SM = TM / GM, SN = MOE_WR / (4 / GM);
    constexpr uint LD = BK + 8;
    constexpr uint AC = BK / 8;
    constexpr uint NA = (BM * AC + 127) / 128;
    constexpr uint NB = BK / 32;  // 16-element groups per thread and step
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    const uint KD = MODE == 0 ? H : I;
    uint e, s0, rows;
    if (!moe_tile(off, tiles, E, BM, tg.y, e, s0, rows)) return;
    threadgroup bfloat* As = (threadgroup bfloat*)smem;
    threadgroup bfloat* Bs = As + TM * LD;
    if (BM < TM) {
        for (uint i = tid; i < (TM - BM) * LD / 8; i += 128u) *(threadgroup uint4*)(As + BM * LD + i * 8u) = uint4(0u);
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
    uint2 wv[NB];  // NVFP4
    uint4 wx[NB];  // MXFP8
    uint sv[NB];
    if (FMT == 0) moe_fetch<NB>(wp, sp, bok, bq, 0u, KD, wv, sv);
    else inx_fetch_mx<NB>(wp, sp, bok, bq, 0u, KD, wx, sv);
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

    for (uint k0 = 0; k0 < KD; k0 += BK) {
        for (uint j = 0; j < NA; ++j) {
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
        for (uint q = 0; q < NB; ++q) {
            if (FMT == 0) moe_blk_bf16(wv[q], moe_sc(sv[q]), Bs + br * LD + (bq + q) * 16u);
            else inx_blk_mx(wx[q], sv[q], Bs + br * LD + (bq + q) * 16u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < KD) {
            if (FMT == 0) moe_fetch<NB>(wp, sp, bok, bq, k0 + BK, KD, wv, sv);
            else inx_fetch_mx<NB>(wp, sp, bok, bq, k0 + BK, KD, wx, sv);
        }
        ext_a ea;
        ext_b eb;
        array<int, 2> st = {1, int(LD)};
        ten_a tA(As + sr * LD, ea, st);
        ten_b tB(Bs + sc * LD, eb, st);
        op.run(tA, tB, ct);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
            out[(ulong)(s0 + r) * I + n0 + c] = f2bf(inx_act(p.f, g, u, wts[pairs[s0 + r]], p.f0, p.f1));
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

#define INX_SMEM(TM, BK) \
    ((((TM) + MOE_WR) * ((BK) + 8) * 2 > (TM) * MOE_LC * 4 ? ((TM) + MOE_WR) * ((BK) + 8) * 2 : (TM) * MOE_LC * 4) / 16)

#define INX_MM(NAME, FMT, MODE, BM, BK)                                                                                 \
    kernel void NAME(device const ushort* A [[buffer(0)]], device const int* pairs [[buffer(1)]],                       \
                     device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],                        \
                     device const uchar* W [[buffer(4)]], device const uchar* S [[buffer(5)]],                          \
                     device const float* gsc [[buffer(6)]], device const float* wts [[buffer(7)]],                      \
                     device ushort* out [[buffer(8)]], constant OpP& p [[buffer(9)]],                                   \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],               \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                                      \
        threadgroup float4 smem[INX_SMEM(BM < 16 ? 16 : BM, BK)];                                                       \
        inx_mm_impl<FMT, MODE, BM, BK>(A, pairs, off, tiles, W, S, gsc, wts, out, p, (threadgroup float*)smem, tg, tid, sg); \
    }

// Keep in step with NAX_GU / NAX_DN in src/inkmoe.rs (tiles of 8 / 16 / 32 / 64 rows; K step 128, 64 at 64 rows).
INX_MM(ink_moe_nax_gu_nv_m8, 0, 0, 8, 128)
INX_MM(ink_moe_nax_gu_nv_m16, 0, 0, 16, 128)
INX_MM(ink_moe_nax_gu_nv_m32, 0, 0, 32, 128)
INX_MM(ink_moe_nax_gu_nv_m64, 0, 0, 64, 64)
INX_MM(ink_moe_nax_dn_nv_m8, 0, 1, 8, 128)
INX_MM(ink_moe_nax_dn_nv_m16, 0, 1, 16, 128)
INX_MM(ink_moe_nax_dn_nv_m32, 0, 1, 32, 128)
INX_MM(ink_moe_nax_dn_nv_m64, 0, 1, 64, 64)
INX_MM(ink_moe_nax_gu_mx_m8, 1, 0, 8, 128)
INX_MM(ink_moe_nax_gu_mx_m16, 1, 0, 16, 128)
INX_MM(ink_moe_nax_gu_mx_m32, 1, 0, 32, 128)
INX_MM(ink_moe_nax_gu_mx_m64, 1, 0, 64, 64)
INX_MM(ink_moe_nax_dn_mx_m8, 1, 1, 8, 128)
INX_MM(ink_moe_nax_dn_mx_m16, 1, 1, 16, 128)
INX_MM(ink_moe_nax_dn_mx_m32, 1, 1, 32, 128)
INX_MM(ink_moe_nax_dn_mx_m64, 1, 1, 64, 64)
