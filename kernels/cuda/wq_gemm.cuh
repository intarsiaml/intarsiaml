/*
 * Dense weight-quantised GEMM for prefill: out[M,N] = scale * x[M,K] @ dequant(W)[N,K]^T on tensor cores, for
 *   FMT_NVFP4  W = e2m1 (u8 [N,K/2]) * e4m3 block scale (per 16 k) ; epilogue scale = global_scale
 *   FMT_FP8    W = e4m3 (u8 [N,K])                                  ; epilogue scale = weight_scale * input_scale
 * Same numerics as the nvfp4.cu / fp8.cu `*_mma` kernels (weights dequantised to bf16 exactly, f32 accumulate), but a
 * different memory schedule (see kernels/cuda/moe_grouped.cu for the measurements behind it): packed weights are
 * copied in 64-byte-per-row runs with cp.async (LPDDR reads of 32 B per row at a large row stride reach about half
 * the bandwidth), staged in shared memory and dequantised once per tile and 64-wide k step; the A tile has a 3-slot
 * cp.async ring; the epilogue writes full rows.
 *
 * Block tile BM x 128 weight rows (BM = 16 * MT * WM), 4 x WM warps. Requires K % 128 == 0 (a raw unit is 64 bytes of
 * packed weights = 128 fp4 or 64 fp8 elements; K % 64 == 0 would do for fp8 but keeps one rule).
 * Included by fp8_gemm.cu and nvfp4_gemm.cu (one kernel template, one instantiation set per file). Header-only
 * templates: no non-template __global__ here, so the two translation units never define the same symbol.
 */
#ifndef IK_WQ_GEMM_CUH
#define IK_WQ_GEMM_CUH

#include "ik_launch.h"
#include "ik_quant.cuh"

namespace wq {

constexpr int FMT_NVFP4 = 0;
constexpr int FMT_FP8 = 1;

constexpr int BK = 64;         // k per step
constexpr int LDS = BK + 8;    // padded smem row (bf16 elements)
constexpr int kWRows = 128;    // weight rows per block
constexpr int kRawStride = 80; // bytes per staged weight row: 64 payload + 16 pad
constexpr int kRawUnit = kWRows * kRawStride;

IK_DEVI float round_bf16(float v) { return ik_b2f(ik_f2b(v)); }

template <int FMT>
struct Fmt {
    static constexpr int kStepsPerUnit = FMT == FMT_NVFP4 ? 2 : 1;           // 64-wide k steps per 64-byte raw unit
    static constexpr int kRing = (3 + kStepsPerUnit - 1) / kStepsPerUnit;    // raw units in flight
    static constexpr int kRowBytesPerK = FMT == FMT_NVFP4 ? 1 : 2;           // row bytes = K * 1 / kRowBytesPerK
};

// ---- GEMM -------------------------------------------------------------------------------------------------------------
template <int FMT, int MT, int WM, typename OutT>
__global__ void __launch_bounds__(128 * WM)
wq_gemm_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ xq, const uint8_t* __restrict__ w,
               const uint8_t* __restrict__ bs, const float* __restrict__ gs, OutT* __restrict__ out, int M, int N, int K,
               int vec_out, int nfast, int only_wo) {
    using F = Fmt<FMT>;
    constexpr int BM = 16 * MT * WM;
    constexpr int NTHREADS = 128 * WM;
    constexpr int RQ = 512 / NTHREADS;  // 16-byte raw chunks per thread and unit (128 rows x 4)
    constexpr int SPU = F::kStepsPerUnit;
    constexpr int RING = F::kRing;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int XS = BM * LDS, WS = kWRows * LDS;
    ik_bf16* const xs0 = reinterpret_cast<ik_bf16*>(smem_raw);
    auto xs = [&](int slot) { return xs0 + slot * XS; };
    auto ws = [&](int buf) { return xs0 + 3 * XS + buf * WS; };
    unsigned char* const raw_base = smem_raw + (size_t)(3 * XS + 2 * WS) * 2;
    auto rawu = [&](int slot) { return raw_base + slot * kRawUnit; };

    // Block order: M tiles fastest (consecutive blocks share the weight rows in L2; the activations are re-read per
    // column tile) or, when the activations are the bigger operand, N tiles fastest (the activations are shared).
    const int nmt = (M + BM - 1) / BM, nnt = (N + kWRows - 1) / kWRows;
    const int bid = blockIdx.x;
    const int m0 = (nfast ? bid / nnt : bid % nmt) * BM;
    const int n0 = (nfast ? bid % nnt : bid / nmt) * kWRows;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp & 3, wm = warp >> 2;
    const int S = K / BK;
    const int U = (S + SPU - 1) / SPU;
    const int64_t row_bytes = FMT == FMT_NVFP4 ? K / 2 : K;
    const bool actq = FMT == FMT_FP8 && gs[1] > 0.f;
    if (only_wo && actq) return;  // block-uniform: this launch is the weight-only fallback of an e4m3-activation kernel
    const ik_bf16* const a_src = actq ? xq : x;
    const float s_out = FMT == FMT_FP8 ? gs[0] * (actq ? gs[1] : 1.f) : gs[0];

    // packed-weight copy: row r = (tid >> 2) + (NTHREADS / 4) q, 16-byte part tid & 3 of the unit's 64 bytes
    const int rpart = tid & 3;
    const uint8_t* rsrc[RQ];
    bool rvalid[RQ];
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
        const int r = (tid >> 2) + (NTHREADS / 4) * q;
        rvalid[q] = n0 + r < N;
        rsrc[q] = w + (int64_t)(rvalid[q] ? n0 + r : 0) * row_bytes + rpart * 16;
    }
    auto issue_raw = [&](int u) {
        const int slot = u % RING;
#pragma unroll
        for (int q = 0; q < RQ; ++q) {
            const int r = (tid >> 2) + (NTHREADS / 4) * q;
            ik_cp_async16(rawu(slot) + r * kRawStride + rpart * 16, rsrc[q] + (int64_t)u * 64, rvalid[q] ? 16 : 0);
        }
    };

    // this thread dequantises row `wr` of the tile (k offset `khalf * 32` of each step when WM == 2)
    const int wr = WM == 1 ? tid : tid >> 1;
    const int khalf = WM == 1 ? 0 : (tid & 1);
    const int wrow = min(n0 + wr, N - 1);
    const uint8_t* sp = FMT == FMT_NVFP4 ? bs + (int64_t)wrow * (K / 16) + khalf * 2 : bs;
    uint32_t rawsc = 0;
    auto load_sc = [&](int st) {
        if (FMT == FMT_NVFP4) {
            if (WM == 1) rawsc = *reinterpret_cast<const uint32_t*>(sp + st * (BK / 16));
            else rawsc = *reinterpret_cast<const uint16_t*>(sp + st * (BK / 16));
        }
    };
    auto dequant = [&](int st, int buf) {
        const int u = st / SPU;
        if (FMT == FMT_NVFP4) {
            const unsigned char* rp = rawu(u % RING) + wr * kRawStride + (st & 1) * 32 + khalf * 16;
            const uint4 q0 = *reinterpret_cast<const uint4*>(rp);
            const uint4 q1 = WM == 1 ? *reinterpret_cast<const uint4*>(rp + 16) : q0;
            const uint32_t wd[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
            for (int b = 0; b < 4 / WM; ++b) {
                uint4 lo, hi;
                ik_fp4x16_scaled_to_bf16(make_uint2(wd[2 * b], wd[2 * b + 1]),
                                         ik_e4m3_to_f((uint8_t)((rawsc >> (8 * b)) & 0xFFu)), lo, hi);
                uint4* dst = reinterpret_cast<uint4*>(ws(buf) + wr * LDS + khalf * 32 + b * 16);
                dst[0] = lo;
                dst[1] = hi;
            }
        } else {
            const unsigned char* rp = rawu(u % RING) + wr * kRawStride + khalf * 32;
#pragma unroll
            for (int b = 0; b < 4 / WM; ++b) {  // 16 fp8 per piece: 4 (WM == 1) or 2 (WM == 2)
                float f[16];
                ik_fp8x16_to_f32(*reinterpret_cast<const uint4*>(rp + b * 16), f);
                uint4* dst = reinterpret_cast<uint4*>(ws(buf) + wr * LDS + khalf * 32 + b * 16);
                dst[0] = ik_pack8(f);
                dst[1] = ik_pack8(f + 8);
            }
        }
    };

    // this thread's A rows: chunk c8 (16 B = 8 bf16) of rows r = (tid >> 3) + (NTHREADS / 8) q
    const int c8 = tid & 7;
    const ik_bf16* arow[MT];
    bool avalid[MT];
#pragma unroll
    for (int q = 0; q < MT; ++q) {
        const int r = (tid >> 3) + (NTHREADS / 8) * q;
        avalid[q] = m0 + r < M;
        arow[q] = avalid[q] ? a_src + (int64_t)(m0 + r) * K : a_src;
    }
    auto issue_a = [&](int st, int slot) {
#pragma unroll
        for (int q = 0; q < MT; ++q) {
            const int r = (tid >> 3) + (NTHREADS / 8) * q;
            ik_cp_async16(xs(slot) + r * LDS + c8 * 8, arow[q] + st * BK + c8 * 8, avalid[q] ? 16 : 0);
        }
    };

    float acc[MT][4][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

    // Raw unit u (steps u*SPU ..) is issued at iteration SPU*u - 3, i.e. into commit group SPU*u - 1; units that would
    // be issued before iteration 0 go into the prologue groups (0 and 1).
    // prologue group 0: A(0) + units with SPU*u - 1 <= 0 ; group 1: A(1) + the units with SPU*u - 1 == 1
    issue_a(0, 0);
    for (int u = 0; u < U && SPU * u - 1 <= 0; ++u) issue_raw(u);
    ik_cp_async_commit();
    if (S > 1) issue_a(1, 1);
    for (int u = 0; u < U && SPU * u - 1 <= 1; ++u)
        if (SPU * u - 1 == 1) issue_raw(u);
    ik_cp_async_commit();
    load_sc(0);
    ik_cp_async_wait_group<1>();
    __syncthreads();
    dequant(0, 0);
    if (S > 1) load_sc(1);
    for (int s = 0; s < S; ++s) {
        ik_cp_async_wait_group<1>();  // groups 0..s complete: A(s), and the raw unit of step s+1
        __syncthreads();
        if (s + 2 < S) issue_a(s + 2, (s + 2) % 3);
        if ((s + 3) % SPU == 0 && (s + 3) / SPU < U) issue_raw((s + 3) / SPU);
        ik_cp_async_commit();
        if (s + 1 < S) dequant(s + 1, (s + 1) & 1);
        if (s + 2 < S) load_sc(s + 2);
        const ik_bf16* xt = xs(s % 3);
        const ik_bf16* wt = ws(s & 1);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t b[4][2];
#pragma unroll
            for (int pj = 0; pj < 2; ++pj) {
                const int mi = lane >> 3;
                const int nr = wn * 32 + (2 * pj + (mi >> 1)) * 8 + (lane & 7);
                ik_ldmatrix_x4(b[2 * pj][0], b[2 * pj][1], b[2 * pj + 1][0], b[2 * pj + 1][1],
                               wt + nr * LDS + kk + (mi & 1) * 8);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                uint32_t a0, a1, a2, a3;
                const int r = wm * 16 * MT + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ik_ldmatrix_x4(a0, a1, a2, a3, xt + r * LDS + kk + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < 4; ++j) ik_mma_bf16_16816(acc[i][j], a0, a1, a2, a3, b[j][0], b[j][1]);
            }
        }
    }

    // ---- epilogue ----
    if (sizeof(OutT) == 2 && vec_out) {
        // stage the bf16 tile in shared memory and write full 256-byte row runs
        constexpr int OS = kWRows + 8;
        __syncthreads();
        ik_bf16* const ob = xs0;  // [BM][OS], aliases the A ring (3 * BM * 72 >= BM * 136)
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int rl = wm * 16 * MT + i * 16 + (lane >> 2) + hh * 8;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int cl = wn * 32 + j * 8 + (lane & 3) * 2;
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) =
                        __floats2bfloat162_rn(acc[i][j][2 * hh] * s_out, acc[i][j][2 * hh + 1] * s_out);
                }
            }
        __syncthreads();
        constexpr int CPR = kWRows / 8;
        for (int c = tid; c < BM * CPR; c += NTHREADS) {
            const int rl = c / CPR, part = c % CPR;
            if (m0 + rl >= M || n0 + part * 8 >= N) continue;  // N % 8 == 0 when vec_out
            *reinterpret_cast<uint4*>(reinterpret_cast<ik_bf16*>(out) + (int64_t)(m0 + rl) * N + n0 + part * 8) =
                *reinterpret_cast<const uint4*>(ob + rl * OS + part * 8);
        }
    } else {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int row = m0 + wm * 16 * MT + i * 16 + (lane >> 2) + hh * 8;
                    const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                    if (row >= M) continue;
                    if (col < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * hh] * s_out);
                    if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * hh + 1] * s_out);
                }
    }
}

template <int FMT, int MT, int WM, typename OutT>
void launch_gemm(const ik_bf16* x, const ik_bf16* xq, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out,
                 int M, int N, int K, int vec_out, cudaStream_t s, int only_wo) {
    // activation bytes vs weight bytes: the larger operand is the one consecutive blocks should share
    const int nfast = (double)M * 2.0 > (double)N * (FMT == FMT_NVFP4 ? 0.5625 : 1.0) ? 1 : 0;
    constexpr size_t smem = (size_t)(3 * 16 * MT * WM + 2 * kWRows) * LDS * 2 + (size_t)Fmt<FMT>::kRing * kRawUnit;
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(wq_gemm_kernel<FMT, MT, WM, OutT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        return true;
    }();
    (void)attr_set;
    constexpr int BM = 16 * MT * WM;
    const int64_t nblk = (int64_t)((M + BM - 1) / BM) * ((N + kWRows - 1) / kWRows);
    wq_gemm_kernel<FMT, MT, WM, OutT><<<dim3((unsigned)nblk), 128 * WM, smem, s>>>(x, xq, w, bs, gs, out, M, N, K,
                                                                                    vec_out, nfast, only_wo);
}

template <int FMT, typename OutT>
void gemm_by_m(const ik_bf16* x, const ik_bf16* xq, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int M,
               int N, int K, int vec_out, cudaStream_t s, int only_wo = 0) {
    if (M <= 16) launch_gemm<FMT, 1, 1>(x, xq, w, bs, gs, out, M, N, K, vec_out, s, only_wo);
    else if (M <= 32) launch_gemm<FMT, 1, 2>(x, xq, w, bs, gs, out, M, N, K, vec_out, s, only_wo);
    else launch_gemm<FMT, 2, 2>(x, xq, w, bs, gs, out, M, N, K, vec_out, s, only_wo);
}

}  // namespace wq

#endif /* IK_WQ_GEMM_CUH */
