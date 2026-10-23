/* W8A8 prefill GEMM on the gfx12 fp8 x fp8 WMMA (v_wmma_f32_16x16x16_fp8_fp8), used by matmul_fp8_mma_f8 (fp8.hip).
 *
 * Same op and numerics as ik_dq_gemm.h's Fp8Policy: with a static input scale the activations are quantised to e4m3
 * (RNE, saturating, x / input_scale), every product e4m3 x e4m3 is exact in f32, and y = weight_scale * input_scale *
 * sum. The dequantise-to-bf16 scheme converts every weight chunk once per token tile and every activation chunk once per
 * weight tile in the ALUs, which kept the fp8 projections of the 35B at ~20 TFLOPS (13 % of the bf16 WMMA peak); here the
 * e4m3 bytes go from global memory to LDS to the fragments unchanged (nothing is converted in the GEMM; the activations
 * are quantised once, by f8_actq_kernel, into scratch), and the matrix core does twice the MACs per instruction of the bf16
 * WMMA. Only the order of the f32 additions differs from the bf16 kernel (k permuted inside each group of 32, below).
 *
 * Fragments (gfx12 wave32, 16x16x16, 8-bit operands): lane l holds row/col l % 16 and 8 consecutive bytes; A is the weights
 * (rows = output features), B the activations. The 16 k of one instruction are two groups of 8 bytes, one per lane half.
 * The k slots only have to agree between A and B, so within each group of 32 k bytes lane half h takes bytes h * 16 .. +15
 * and the two instructions of the group use its low and high 8 bytes: one 16-byte LDS read feeds two WMMAs. Every k is
 * summed exactly once, in a fixed order that does not depend on M, N or the tile (deterministic, M-invariant).
 *
 * Block: WN x WM waves, each wave TN x TM fragments of 16 x 16 (weights x tokens); 64 k bytes per stage, double-buffered
 * through registers (global loads of the next stage are issued before the current stage is consumed). The output
 * layout is that of ik_dq_gemm.h (lane holds token l % 16 and features (l / 16) * 8 + i).
 */
#ifndef IK_F8_GEMM_H
#define IK_F8_GEMM_H

#include "ik_dq_gemm.h"

typedef int ik_f8_v2i __attribute__((ext_vector_type(2)));

IK_DEVI ik_dq_v8f ik_f8_wmma(uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi, ik_dq_v8f c) {
    const uint2 a = make_uint2(alo, ahi), b = make_uint2(blo, bhi);
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(ik_f8_v2i, a),
                                                                __builtin_bit_cast(ik_f8_v2i, b), c);
}

constexpr int kF8BK = 64;             // k bytes per stage
constexpr int kF8LD = kF8BK + 16;     // LDS row stride in bytes: 8 consecutive rows hit 8 distinct 16-byte bank groups

/* xq = e4m3(x / input_scale) as bytes, the rounding of Fp8Policy::xprep. Exits at once for weight-only activations. */
__global__ void __launch_bounds__(256)
f8_actq_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ xq, const float* __restrict__ sc, int64_t n8) {
    const float s_in = sc[1];
    if (!(s_in > 0.f)) return;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    float f[8];
    ik_unpack8(reinterpret_cast<const uint4*>(x)[i], f);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        lo |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[e], s_in)) << (8 * e);
        hi |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 + e], s_in)) << (8 * e);
    }
    reinterpret_cast<uint2*>(xq)[i] = make_uint2(lo, hi);
}

template <int WN, int WM, int TN, int TM, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32)
f8_gemm_kernel(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ w, const float* __restrict__ sc,
               OutT* __restrict__ out, int M, int64_t N, int K, int vec_out) {
    if (!(sc[1] > 0.f)) return;  // weight-only activations: the bf16 kernel handles this launch
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int CHW = BN * 4, CHX = BM * 4;  // 16-byte chunks of one stage (4 per row)
    constexpr int CW = (CHW + NT - 1) / NT, CX = (CHX + NT - 1) / NT;
    __shared__ __align__(16) uint8_t sw[2][BN * kF8LD];
    __shared__ __align__(16) uint8_t sx[2][BM * kF8LD];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wn = wave % WN, wm = wave / WN;
    // token blocks vary fastest: blocks resident together share one weight panel (weights stream from DRAM once)
    const int m_blk = blockIdx.x * BM;
    const int64_t n_blk = (int64_t)blockIdx.y * BN;
    const int nk = (K + kF8BK - 1) / kF8BK;

    uint4 rw[CW], rx[CX];
    auto gload = [&](int k0) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int64_t n = n_blk + r;
            const int k = k0 + ch * 16;
            rw[c] = (idx < CHW && n < N && k < K) ? *reinterpret_cast<const uint4*>(w + n * K + k) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int m = m_blk + r;
            const int k = k0 + ch * 16;
            rx[c] = (idx < CHX && m < M && k < K) ? *reinterpret_cast<const uint4*>(xq + (int64_t)m * K + k)
                                                   : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) *reinterpret_cast<uint4*>(&sw[buf][(idx >> 2) * kF8LD + (idx & 3) * 16]) = rw[c];
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHX) *reinterpret_cast<uint4*>(&sx[buf][(idx >> 2) * kF8LD + (idx & 3) * 16]) = rx[c];
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
    const int lr = lane & 15, lh = (lane >> 4) * 16;
    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) gload((kt + 1) * kF8BK);
#pragma unroll
        for (int g = 0; g < kF8BK / 32; ++g) {
            uint4 af[TN], bf[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i)
                af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kF8LD + g * 32 + lh]);
#pragma unroll
            for (int j = 0; j < TM; ++j)
                bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kF8LD + g * 32 + lh]);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = ik_f8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
                    acc[i][j] = ik_f8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
                }
        }
        if (kt + 1 < nk) sstore(buf ^ 1);
        __syncthreads();
    }

    const float so = sc[0] * sc[1];
#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m_blk + wm * TM * 16 + j * 16 + lr;
            const int64_t n = n_blk + wn * TN * 16 + i * 16 + (lane >> 4) * 8;
            if (m >= M || n >= N) continue;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = acc[i][j][e] * so;
            OutT* o = out + (int64_t)m * N + n;
            if (vec_out) {
                if (sizeof(OutT) == 2) {
                    *reinterpret_cast<uint4*>(o) = ik_pack8(v);
                } else {
                    reinterpret_cast<float4*>(o)[0] = make_float4(v[0], v[1], v[2], v[3]);
                    reinterpret_cast<float4*>(o)[1] = make_float4(v[4], v[5], v[6], v[7]);
                }
            } else {
                for (int e = 0; e < 8 && n + e < N; ++e) ik_store(o + e, v[e]);
            }
        }
    }
}

/* Tile per M, the same table as dq_gemm_run (small M keeps the token tile small): weights x tokens. */
template <typename OutT>
void f8_gemm_run(const uint8_t* xq, const uint8_t* w, const float* sc, OutT* out, int M, int64_t N, int K, int vec_out,
                 hipStream_t s) {
#define IK_F8_TILE(WN, WM, TN, TM)                                                                                   \
    {                                                                                                                \
        dim3 grid((unsigned)(((int64_t)M + WM * TM * 16 - 1) / (WM * TM * 16)),                                      \
                  (unsigned)((N + WN * TN * 16 - 1) / (WN * TN * 16)));                                              \
        f8_gemm_kernel<WN, WM, TN, TM, OutT><<<grid, WN * WM * 32, 0, s>>>(xq, w, sc, out, M, N, K, vec_out);       \
    }
    // Prefill chunks (M >= 2048, lane hp tile sweeps, docs/results/2026-10-22-hp-hip-prefill-roofline.md): 16 waves,
    // 256 x 128 (weights x tokens) for the widest projections, 128 x 256 for the others; the k order does not depend on the
    // tile, so every tile gives the same bits (the sweep checked all of them bitwise against 128 x 128, partial M too);
    // 128 x 256 only when it still gives every CU a block (else 128 x 128: k / v at M 2049 was +56 % on 36 blocks).
    if (M <= 16) IK_F8_TILE(4, 1, 2, 1)         // 128 x 16
    else if (M <= 32) IK_F8_TILE(4, 1, 2, 2)    // 128 x 32
    else if (M <= 64) IK_F8_TILE(2, 2, 4, 2)    // 128 x 64
    else if (M >= 2048 && N >= 8192) IK_F8_TILE(4, 4, 4, 2)  // 256 x 128, 16 waves of 64 x 32
    else if (M >= 2048 && ((N + 127) / 128) * (((int64_t)M + 255) / 256) >= 64)  // >= one block per CU (k/v: N = 512)
        IK_F8_TILE(4, 4, 2, 4)                  // 128 x 256, 16 waves of 32 x 64
    else IK_F8_TILE(4, 2, 2, 4)                 // 128 x 128, 8 waves of 32 x 64
#undef IK_F8_TILE
}

#endif /* IK_F8_GEMM_H */
