// TIMING PROBES of the first-generation grouped NVFP4 experts (lane ink-prefill, coordinator brief 15). WRONG RESULTS.
// A copy of inkling2_moe_nax.metal's inx_mm_impl (NVFP4, 64-row tiles, K step 64) with one template knob:
//   PROBE 0  unchanged (a timing control for the copy itself)
//   PROBE 1  the NVFP4 -> bf16 decode arithmetic replaced by a bit copy (the same 32-byte threadgroup store per block)
//   PROBE 2  no weight / scale loads from device memory: thread-dependent words feed the unchanged decode
//   k64p0 / k128p0  PROBE 0 at K tile 64 / 128 without the threadgroup row pad (K-step probe; these give CORRECT results)
// Reachable only from inkmoe_bench built with the `ink-probe` cargo feature (src/inkprefill.rs `set_moe_probe`); never a
// registry kernel, so admission never lists it. Remove before relying on anything here.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "moe_mm.h"
using namespace metal;
using namespace mpp::tensor_ops;

inline float prb_gelu(float x) { return 0.5f * x * (1.f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

inline float prb_act(uint epi, float g, float u, float w, float gc, float uc) {
    if (epi == 2u) {
        const float sg = 1.f / (1.f + precise::exp(-g));
        return moe_rb(gc * precise::tanh(g / gc) * sg * uc * precise::tanh(u / uc) * w);
    }
    const float a = epi == 1u ? prb_gelu(g) : g / (1.f + precise::exp(-g));
    return moe_rb(moe_rb(a) * u);
}

// PROBE 2: words the compiler cannot fold (thread and step dependent), scale bytes in the normal e4m3 range
template <uint NB>
inline void prb_const(uint tid, uint k0, thread uint2* wv, thread uint* sv) {
    for (uint q = 0; q < NB; ++q) {
        const uint h = (tid * 0x9E3779B1u) ^ (k0 + q) * 0x85EBCA6Bu;
        wv[q] = uint2(h & 0x77777777u, (h >> 3) & 0x77777777u);
        sv[q] = 0x30u | (h >> 28);
    }
}

// PROBE 1: the raw words stored where the 16 decoded bf16 go (same store width, no arithmetic)
inline void prb_blk_copy(uint2 v, uint s, threadgroup bfloat* dst) {
    threadgroup uint4* d = (threadgroup uint4*)dst;
    d[0] = uint4(v.x, v.y, v.x ^ s, v.y);
    d[1] = uint4(v.y, v.x, v.y ^ s, v.x);
}

template <uint FMT, uint MODE, uint BM, uint BK, uint PROBE, uint PAD = 8>
inline void prb_mm_impl(device const ushort* A, device const int* pairs, device const int* off, device const int* tiles,
                        device const uchar* W, device const uchar* S, device const float* gsc, device const float* wts,
                        device ushort* out, constant OpP& p, threadgroup float* smem, uint2 tg, uint tid, uint sg) {
    constexpr uint TM = BM < 16 ? 16 : BM;
    constexpr uint GM = BM >= 32 ? 2 : 1;
    constexpr uint SM = TM / GM, SN = MOE_WR / (4 / GM);
    constexpr uint LD = BK + PAD;
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
    if (PROBE == 2) prb_const<NB>(tid, 0u, wv, sv);
    else if (FMT == 0) moe_fetch<NB>(wp, sp, bok, bq, 0u, KD, wv, sv);
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
            if (PROBE == 1) prb_blk_copy(wv[q], sv[q], Bs + br * LD + (bq + q) * 16u);
            else if (FMT == 0) moe_blk_bf16(wv[q], moe_sc(sv[q]), Bs + br * LD + (bq + q) * 16u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < KD) {
            if (PROBE == 2) prb_const<NB>(tid, k0 + BK, wv, sv);
            else if (FMT == 0) moe_fetch<NB>(wp, sp, bok, bq, k0 + BK, KD, wv, sv);
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
            out[(ulong)(s0 + r) * I + n0 + c] = f2bf(prb_act(p.f, g, u, wts[pairs[s0 + r]], p.f0, p.f1));
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

#define PRB_SMEM(TM, BK, PAD) \
    ((((TM) + MOE_WR) * ((BK) + (PAD)) * 2 > (TM) * MOE_LC * 4 ? ((TM) + MOE_WR) * ((BK) + (PAD)) * 2 : (TM) * MOE_LC * 4) / 16)

#define PRB_MMK(NAME, MODE, PROBE, BK, PAD)                                                                             \
    kernel void NAME(device const ushort* A [[buffer(0)]], device const int* pairs [[buffer(1)]],                       \
                     device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],                        \
                     device const uchar* W [[buffer(4)]], device const uchar* S [[buffer(5)]],                          \
                     device const float* gsc [[buffer(6)]], device const float* wts [[buffer(7)]],                      \
                     device ushort* out [[buffer(8)]], constant OpP& p [[buffer(9)]],                                   \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],               \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                                      \
        threadgroup float4 smem[PRB_SMEM(64, BK, PAD)];                                                                 \
        prb_mm_impl<0, MODE, 64, BK, PROBE, PAD>(A, pairs, off, tiles, W, S, gsc, wts, out, p, (threadgroup float*)smem, tg, tid, sg); \
    }
#define PRB_MM(NAME, MODE, PROBE) PRB_MMK(NAME, MODE, PROBE, 64, 8)

PRB_MM(ink_moe_probe_gu_nv_p0, 0, 0)
PRB_MM(ink_moe_probe_dn_nv_p0, 1, 0)
PRB_MM(ink_moe_probe_gu_nv_p1, 0, 1)
PRB_MM(ink_moe_probe_dn_nv_p1, 1, 1)
PRB_MM(ink_moe_probe_gu_nv_p2, 0, 2)
PRB_MM(ink_moe_probe_dn_nv_p2, 1, 2)
// K-step probes (brief 17; measure only): K tile 64 without the row pad (control for the pad), and 128 without the pad
// (2x; 64 + 64 rows x 128 x 2 B = 32768 B, exactly the 32 KB threadgroup limit). 4x (256) does not fit 32 KB at 64-row
// tiles and does not divide the down's K = 1152.
PRB_MMK(ink_moe_probe_gu_nv_k64p0, 0, 0, 64, 0)
PRB_MMK(ink_moe_probe_dn_nv_k64p0, 1, 0, 64, 0)
PRB_MMK(ink_moe_probe_gu_nv_k128p0, 0, 0, 128, 0)
PRB_MMK(ink_moe_probe_dn_nv_k128p0, 1, 0, 128, 0)
