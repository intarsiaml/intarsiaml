// Grouped (prefill) MoE experts on the M5 neural accelerators: moe_mm_{gu,dn}_m<BM> of moe.metal with the 8x8
// simdgroup_matrix MMAs replaced by mpp::tensor_ops::matmul2d on the staged threadgroup tiles. Compiled -std=metal4.0 into
// intarsia_metal4.metallib (build.rs METAL4_SOURCES). Same bindings, grid, staging (x rows gathered by pair, NVFP4 blocks
// decoded to bf16 once per tile) and epilogue as moe.metal's `moe_mm_impl` (the shared helpers are in moe_mm.h); only the K
// accumulation order differs (the accelerator's), so the result is KL-gated against the simdgroup form, not bitwise it
// (docs/results/2026-10-27-nxp-metal-nax-prefill.md). Every BM takes this path, so the tile choice stays free of the row
// bits (tests/moe.rs `moe_experts_rows_do_not_depend_on_the_tile`); BM = 8 runs as 16 rows, the pad rows zero.
//
// Each simdgroup owns an SM x SN sub-tile of the TM x 64 output tile (BM >= 32: 2 x 2 simdgroups, otherwise 1 x 4) and
// accumulates it over the K steps in a float cooperative tensor, which is stored to threadgroup memory for the epilogue.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "moe_mm.h"
using namespace metal;
using namespace mpp::tensor_ops;

template <uint MODE, uint BM, uint BK>
inline void moe_mm_nax_impl(device const ushort* A, device const int* pairs, device const int* off, device const int* tiles,
                            device const uchar* W, device const uchar* S, device const float* gsc, device ushort* out,
                            constant OpP& p, threadgroup float* smem, uint2 tg, uint tid, uint sg) {
    constexpr uint TM = BM < 16 ? 16 : BM;      // rows of the accelerator tile
    constexpr uint GM = BM >= 32 ? 2 : 1;       // simdgroups along M; 4 / GM along the weight rows
    constexpr uint SM = TM / GM, SN = MOE_WR / (4 / GM);
    constexpr uint LD = BK + 8;                 // bf16 per staged row (a 16-byte multiple)
    constexpr uint AC = BK / 8;                 // 8-element A chunks per row
    constexpr uint NA = (BM * AC + 127) / 128;  // A chunks per thread
    constexpr uint NB = BK / 32;                // weight blocks per thread and step
    const uint E = p.b, K = p.c, H = p.d, I = p.e;
    const uint KD = MODE == 0 ? H : I;
    uint e, s0, rows;
    if (!moe_tile(off, tiles, E, BM, tg.y, e, s0, rows)) return;
    threadgroup bfloat* As = (threadgroup bfloat*)smem;
    threadgroup bfloat* Bs = As + TM * LD;
    if (BM < TM) {  // pad rows of the accelerator tile: zero for good (the staging writes rows 0 .. BM - 1 only)
        for (uint i = tid; i < (TM - BM) * LD / 8; i += 128u) *(threadgroup uint4*)(As + BM * LD + i * 8u) = uint4(0u);
    }
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
        ext_a ea;
        ext_b eb;
        array<int, 2> st = {1, int(LD)};
        ten_a tA(As + sr * LD, ea, st);  // A sub-tile [SM rows][BK], W sub-tile [SN rows][BK] (transpose_right)
        ten_b tB(Bs + sc * LD, eb, st);
        op.run(tA, tB, ct);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue through threadgroup memory: C [TM][MOE_LC] f32 (gu: columns 0..31 gate, 32..63 up)
    {
        extents<int, SN, SM> ec;
        array<int, 2> st = {1, int(MOE_LC)};
        tensor<threadgroup float, extents<int, SN, SM>, tensor_inline> tC(smem + sr * MOE_LC + sc, ec, st);
        ct.store(tC);
    }
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

// threadgroup memory: the staged A + B tiles, reused by the f32 epilogue (TM parenthesised: it is an expression)
#define MOE_SMEM_N(TM, BK) \
    ((((TM) + MOE_WR) * ((BK) + 8) * 2 > (TM) * MOE_LC * 4 ? ((TM) + MOE_WR) * ((BK) + 8) * 2 : (TM) * MOE_LC * 4) / 16)

#define MOE_MM_NAX(NAME, MODE, BM, BK)                                                                                  \
    kernel void NAME(device const ushort* A [[buffer(0)]], device const int* pairs [[buffer(1)]],                       \
                     device const int* off [[buffer(2)]], device const int* tiles [[buffer(3)]],                        \
                     device const uchar* W [[buffer(4)]], device const uchar* S [[buffer(5)]],                          \
                     device const float* gsc [[buffer(6)]], device ushort* out [[buffer(7)]],                           \
                     constant OpP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],                          \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {            \
        threadgroup float4 smem[MOE_SMEM_N(BM < 16 ? 16 : BM, BK)];                                                     \
        moe_mm_nax_impl<MODE, BM, BK>(A, pairs, off, tiles, W, S, gsc, out, p, (threadgroup float*)smem, tg, tid, sg);  \
    }

// Keep in step with NAX_MM_GU / NAX_MM_DN in src/moe.rs. K step 128 up to 32 rows, 64 at 64 rows (threadgroup memory).
MOE_MM_NAX(moe_mm_nax_gu_m8, 0, 8, 128)
MOE_MM_NAX(moe_mm_nax_gu_m16, 0, 16, 128)
MOE_MM_NAX(moe_mm_nax_gu_m32, 0, 32, 128)
MOE_MM_NAX(moe_mm_nax_gu_m64, 0, 64, 64)
MOE_MM_NAX(moe_mm_nax_dn_m8, 1, 8, 128)
MOE_MM_NAX(moe_mm_nax_dn_m16, 1, 16, 128)
MOE_MM_NAX(moe_mm_nax_dn_m32, 1, 32, 128)
MOE_MM_NAX(moe_mm_nax_dn_m64, 1, 64, 64)
