/* Dequantise-to-bf16 WMMA GEMM shared by fp8.hip and nvfp4.hip (the `*_mma` launchers, any M).
 *
 * out[m, n] = scale * sum_k x[m, k] * W[n, k], with W dequantised tile by tile into LDS as bf16 (exact for e4m3 and
 * for e2m1 * e4m3) and bf16 WMMA with f32 accumulate in a fixed k order (deterministic). A weight format is a
 * `Policy` (below); the tile loop, LDS layout and fragment code are the same as the dense bf16 GEMM in matmul.hip.
 *
 * gfx12 wave32 16x16x16 fragment layout (verified on hardware by the matmul and attention admission sweeps):
 *   A (16 rows x 16 k): lane l holds row l % 16, k = (l / 16) * 8 .. +7   (8 bf16 = one uint4)
 *   B (16 k x 16 cols): lane l holds col l % 16, k = (l / 16) * 8 .. +7
 *   C/D (16 x 16 f32):  lane l holds col l % 16, rows (l / 16) * 8 + i, i = 0..7
 * The weights are the A operand (rows = output features), the activations the B operand, so each lane writes 8
 * contiguous outputs out[m, n0 .. n0 + 7].
 *
 * Policy contract (all static, device):
 *   struct State                      per-block constants (scales), made once from the launch args
 *   struct Raw                        one 16-element weight chunk as loaded from global memory
 *   bool  skip(const DqArgs&)        block-uniform: the launch is not for this policy's mode (exits before init)
 *   State init(const DqArgs&)        every thread reaches it at kernel start (may hold a __syncthreads())
 *   Raw   load(const DqArgs&, int64_t n, int k)   16 elements of row n starting at element k (in range)
 *   Raw   zero()
 *   void  dequant(const State&, const Raw&, uint4& lo, uint4& hi)   16 bf16 = elements 0..7, 8..15
 *   uint4 xload(const DqArgs&, const State&, int m, int k)   optional: the 8 activations of row m from k (in range); without
 *                                    it they are the bf16 x[m, k .. k + 7] (dq_xload below)
 *   uint4 xprep(const State&, uint4 x8)             activation transform of what xload gave (W8A8: quantise to e4m3)
 *   float out_scale(const State&)
 */
#ifndef IK_DQ_GEMM_H
#define IK_DQ_GEMM_H

#include "ik_quant.h"

struct DqArgs {
    const ik_bf16* x;
    const uint8_t* w;
    const uint8_t* bs;  // block scales (nvfp4)
    const float* sc;    // fp8: (weight_scale, input_scale); nvfp4: (global scale)
    const uint8_t* xq;  // matmul_fp8_mma_v2: the activations pre-quantised to e4m3 bytes [M, K] (Fp8PolicyQ only)
    int M;
    int64_t N;
    int K;
    int vec_out;        // N % 8 == 0 and out 16-byte aligned
};

typedef __bf16 ik_dq_v8bf __attribute__((ext_vector_type(8)));
typedef float ik_dq_v8f __attribute__((ext_vector_type(8)));

IK_DEVI ik_dq_v8f ik_dq_wmma(const uint4& a, const uint4& b, ik_dq_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ik_dq_v8bf, a),
                                                             __builtin_bit_cast(ik_dq_v8bf, b), c);
}

/* A policy's xload if it has one (overload resolution prefers the int argument), else the bf16 activation row. */
template <class P>
IK_DEVI auto dq_xload(const DqArgs& a, const typename P::State& st, int m, int k, int) -> decltype(P::xload(a, st, m, k)) {
    return P::xload(a, st, m, k);
}
template <class P>
IK_DEVI uint4 dq_xload(const DqArgs& a, const typename P::State&, int m, int k, long) {
    return *reinterpret_cast<const uint4*>(a.x + (int64_t)m * a.K + k);
}

constexpr int kDqBK = 32;
constexpr int kDqLD = kDqBK + 8;  // padded LDS row (elements): 80 B

template <class P, int WN, int WM, int TN, int TM, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32) dq_gemm_kernel(DqArgs a, OutT* __restrict__ out) {
    if (P::skip(a)) return;
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int CHW = BN * 2, CHX = BM * 4;  // 16-element weight chunks (2 per row), 8-element x chunks (4 per row)
    constexpr int CW = (CHW + NT - 1) / NT, CX = (CHX + NT - 1) / NT;
    __shared__ __align__(16) uint16_t sw[2][BN * kDqLD];
    __shared__ __align__(16) uint16_t sx[2][BM * kDqLD];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wn = wave % WN, wm = wave / WN;
    // token blocks vary fastest: blocks resident together share one weight panel (weights stream from DRAM once)
    const int m_blk = blockIdx.x * BM;
    const int64_t n_blk = (int64_t)blockIdx.y * BN;
    const int nk = (a.K + kDqBK - 1) / kDqBK;
    const typename P::State st = P::init(a);

    typename P::Raw rw[CW];
    uint4 rx[CX];
    auto gload = [&](int k0) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT, r = idx >> 1, ch = idx & 1;
            const int64_t n = n_blk + r;
            const int k = k0 + ch * 16;
            rw[c] = (idx < CHW && n < a.N && k < a.K) ? P::load(a, n, k) : P::zero();
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int m = m_blk + r;
            const int k = k0 + ch * 8;
            rx[c] = (idx < CHX && m < a.M && k < a.K) ? dq_xload<P>(a, st, m, k, 0) : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) {
                uint4 lo, hi;
                P::dequant(st, rw[c], lo, hi);
                uint16_t* d = &sw[buf][(idx >> 1) * kDqLD + (idx & 1) * 16];
                *reinterpret_cast<uint4*>(d) = lo;
                *reinterpret_cast<uint4*>(d + 8) = hi;
            }
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHX) *reinterpret_cast<uint4*>(&sx[buf][(idx >> 2) * kDqLD + (idx & 3) * 8]) = P::xprep(st, rx[c]);
        }
    };

    ik_dq_v8f acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    gload(0);
    sstore(0);
    __syncthreads();
    const int lr = lane & 15, lk = (lane >> 4) * 8;
    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) gload((kt + 1) * kDqBK);
#pragma unroll
        for (int kk = 0; kk < kDqBK; kk += 16) {
            uint4 af[TN], bf[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i)
                af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kDqLD + kk + lk]);
#pragma unroll
            for (int j = 0; j < TM; ++j)
                bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kDqLD + kk + lk]);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) acc[i][j] = ik_dq_wmma(af[i], bf[j], acc[i][j]);
        }
        if (kt + 1 < nk) sstore(buf ^ 1);
        __syncthreads();
    }

    const float so = P::out_scale(st);
#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m_blk + wm * TM * 16 + j * 16 + lr;
            const int64_t n = n_blk + wn * TN * 16 + i * 16 + lk;
            if (m >= a.M || n >= a.N) continue;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = acc[i][j][e] * so;
            OutT* o = out + (int64_t)m * a.N + n;
            if (a.vec_out) {
                if (sizeof(OutT) == 2) {
                    *reinterpret_cast<uint4*>(o) = ik_pack8(v);
                } else {
                    reinterpret_cast<float4*>(o)[0] = make_float4(v[0], v[1], v[2], v[3]);
                    reinterpret_cast<float4*>(o)[1] = make_float4(v[4], v[5], v[6], v[7]);
                }
            } else {
                for (int e = 0; e < 8 && n + e < a.N; ++e) ik_store(o + e, v[e]);
            }
        }
    }
}

/* Tile per M (weights x tokens): small M keeps the token tile small so the activations are not padded to 64+. */
template <class P, typename OutT>
void dq_gemm_run(const DqArgs& a, OutT* out, hipStream_t s) {
    const int64_t M = a.M, N = a.N;
#define IK_DQ_TILE(WN, WM, TN, TM)                                                                              \
    {                                                                                                          \
        dim3 grid((unsigned)((M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((N + WN * TN * 16 - 1) / (WN * TN * 16))); \
        dq_gemm_kernel<P, WN, WM, TN, TM, OutT><<<grid, WN * WM * 32, 0, s>>>(a, out);                         \
    }
    if (M <= 16) IK_DQ_TILE(4, 1, 2, 1)         // 128 x 16
    else if (M <= 32) IK_DQ_TILE(4, 1, 2, 2)    // 128 x 32
    else if (M <= 64) IK_DQ_TILE(2, 2, 4, 2)    // 128 x 64
    else IK_DQ_TILE(4, 2, 2, 4)                 // 128 x 128, 8 waves of 32 x 64
#undef IK_DQ_TILE
}

#endif /* IK_DQ_GEMM_H */
