// matmul_nvfp4: out[M,N] = x[M,K] @ W^T with W[n,k] = e2m1(code) * e4m3(block_scale[n,k/16]) * global (W4A16).
//   w  u8   [N, K/2]   two codes per byte, element 2i in the low nibble
//   bs e4m3 [N, K/16]  plain row-major (checkpoint layout)
//   gs f32  [1]        weight_scale_2
//   *_gemv  M <= 8: warp-per-output-row, one 16-element block (8 weight bytes + 1 scale byte) per lane-iteration.
//   *_mma   any M: 64-column tiles, e2m1*e4m3 dequantised to bf16 in shared memory (exact), bf16 MMA, f32 accumulate.
// Deterministic (fixed reduction order). Activations bf16; output bf16 or f32 (the lm_head writes f32 logits).
// The fp4 -> e4m3 byte lookup (`prmt`) follows TensorFold's NVFP4 kernels (MIT, see ik_quant.cuh).
#include "ik_launch.h"
#include "ik_quant.cuh"
#include <initializer_list>

namespace {

constexpr int kWarps = 8;

// ---- GEMV --------------------------------------------------------------------------------------------------------

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
nvfp4_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                  const float* __restrict__ gs, OutT* __restrict__ out, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;  // whole warp exits together
    const uint2* wr = reinterpret_cast<const uint2*>(w + n * (K / 2));
    const uint8_t* sr = bs + n * (K / 16);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int u = lane; u < K / 16; u += IK_WARP) {
        const uint2 wv = __ldcs(wr + u);
        const float sc = ik_e4m3_to_f(sr[u]);
        float wf[16];
        ik_fp4x8_to_f32(wv.x, wf);
        ik_fp4x8_to_f32(wv.y, wf + 8);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * 16);
            float xf[16];
            ik_unpack8(__ldg(xp), xf);
            ik_unpack8(__ldg(xp + 1), xf + 8);
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
            acc[m] = fmaf(s, sc, acc[m]);
        }
    }
    const float g = gs[0];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) ik_store(out + (int64_t)m * N + n, s * g);
    }
}

// Shared-expert gate + up + silu_glu in one launch (M == 1, bf16 out): a warp owns row n of BOTH the gate and the up matrix,
// so it reads the activations once and ends with silu(gate_n) * up_n. Each row runs exactly the arithmetic of
// nvfp4_gemv_kernel<1, bf16> (same lane/iteration order, same fma chain, scale applied after the warp sum), the two results
// are rounded to bf16 as that kernel's store would, and the epilogue is silu_glu_kernel's: the output bits equal those of the
// three launches this replaces.
__global__ void __launch_bounds__(kWarps * IK_WARP)
nvfp4_glu_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ wg, const uint8_t* __restrict__ bsg,
                 const float* __restrict__ gsg, const uint8_t* __restrict__ wu, const uint8_t* __restrict__ bsu,
                 const float* __restrict__ gsu, ik_bf16* __restrict__ out, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;  // whole warp exits together
    const uint2* wrg = reinterpret_cast<const uint2*>(wg + n * (K / 2));
    const uint2* wru = reinterpret_cast<const uint2*>(wu + n * (K / 2));
    const uint8_t* srg = bsg + n * (K / 16);
    const uint8_t* sru = bsu + n * (K / 16);
    float ag = 0.f, au = 0.f;
#pragma unroll 4
    for (int u = lane; u < K / 16; u += IK_WARP) {
        const uint2 wgv = __ldcs(wrg + u), wuv = __ldcs(wru + u);
        const float scg = ik_e4m3_to_f(srg[u]), scu = ik_e4m3_to_f(sru[u]);
        float fg[16], fu[16];
        ik_fp4x8_to_f32(wgv.x, fg);
        ik_fp4x8_to_f32(wgv.y, fg + 8);
        ik_fp4x8_to_f32(wuv.x, fu);
        ik_fp4x8_to_f32(wuv.y, fu + 8);
        const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)u * 16);
        float xf[16];
        ik_unpack8(__ldg(xp), xf);
        ik_unpack8(__ldg(xp + 1), xf + 8);
        float sg = 0.f, su = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            sg = fmaf(fg[e], xf[e], sg);
            su = fmaf(fu[e], xf[e], su);
        }
        ag = fmaf(sg, scg, ag);
        au = fmaf(su, scu, au);
    }
    const float g = ik_b2f(ik_f2b(ik_warp_sum(ag) * gsg[0]));  // what the gate GEMV stores
    const float u = ik_b2f(ik_f2b(ik_warp_sum(au) * gsu[0]));  // what the up GEMV stores
    if (lane == 0) out[n] = ik_f2b(ik_silu(g) * u);
}

template <int M, typename OutT>
void gemv_run(const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t N, int64_t K,
              cudaStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    nvfp4_gemv_kernel<M, OutT><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, bs, gs, out, N, (int)K);
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t N,
              int64_t K, cudaStream_t s) {
    switch (M) {
        case 1: gemv_run<1>(x, w, bs, gs, out, N, K, s); break;
        case 2: gemv_run<2>(x, w, bs, gs, out, N, K, s); break;
        case 3: gemv_run<3>(x, w, bs, gs, out, N, K, s); break;
        case 4: gemv_run<4>(x, w, bs, gs, out, N, K, s); break;
        case 5: gemv_run<5>(x, w, bs, gs, out, N, K, s); break;
        case 6: gemv_run<6>(x, w, bs, gs, out, N, K, s); break;
        case 7: gemv_run<7>(x, w, bs, gs, out, N, K, s); break;
        case 8: gemv_run<8>(x, w, bs, gs, out, N, K, s); break;
        // 9..16: only the row-exact launcher (`launch` refuses M > 8 for the decode GEMV); the same kernel template, M more accumulators
        case 9: gemv_run<9>(x, w, bs, gs, out, N, K, s); break;
        case 10: gemv_run<10>(x, w, bs, gs, out, N, K, s); break;
        case 11: gemv_run<11>(x, w, bs, gs, out, N, K, s); break;
        case 12: gemv_run<12>(x, w, bs, gs, out, N, K, s); break;
        case 13: gemv_run<13>(x, w, bs, gs, out, N, K, s); break;
        case 14: gemv_run<14>(x, w, bs, gs, out, N, K, s); break;
        case 15: gemv_run<15>(x, w, bs, gs, out, N, K, s); break;
        case 16: gemv_run<16>(x, w, bs, gs, out, N, K, s); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

// ---- MMA (dequantise-to-bf16 tiles) -----------------------------------------------------------------------------

constexpr int BN = 64, BK = 64, LDS = BK + 8;

template <int BM, typename OutT>
__global__ void __launch_bounds__(128)
nvfp4_mma_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                 const float* __restrict__ gs, OutT* __restrict__ out, int M, int64_t N, int K) {
    __shared__ __align__(16) ik_bf16 xs[BM][LDS];
    __shared__ __align__(16) ik_bf16 ws[BN][LDS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t n0 = (int64_t)blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;
    const float g = gs[0];
    float acc[BM / 16][2][4];
#pragma unroll
    for (int i = 0; i < BM / 16; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * 8; idx += 128) {
            const int r = idx >> 3, c = (idx & 7) * 8;
            const int gm = m0 + r, gk = k0 + c;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gm < M && gk < K) v = *reinterpret_cast<const uint4*>(x + (int64_t)gm * K + gk);
            *reinterpret_cast<uint4*>(&xs[r][c]) = v;
        }
        // weights: 64 rows x 64 elements = 4 blocks of 16 per row; one block (8 bytes + 1 scale) per iteration
        for (int idx = tid; idx < BN * 4; idx += 128) {
            const int r = idx >> 2, c = (idx & 3) * 16;
            const int64_t gn = n0 + r;
            const int gk = k0 + c;
            uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
            if (gn < N && gk < K) {
                const uint2 wv = __ldcs(reinterpret_cast<const uint2*>(w + gn * (K / 2) + gk / 2));
                const float sc = ik_e4m3_to_f(bs[gn * (K / 16) + gk / 16]);
                float f[16];
                ik_fp4x8_to_f32(wv.x, f);
                ik_fp4x8_to_f32(wv.y, f + 8);
#pragma unroll
                for (int e = 0; e < 16; ++e) f[e] *= sc;  // exact: <= 6 significant bits
                lo = ik_pack8(f);
                hi = ik_pack8(f + 8);
            }
            *reinterpret_cast<uint4*>(&ws[r][c]) = lo;
            *reinterpret_cast<uint4*>(&ws[r][c + 8]) = hi;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[BM / 16][4];
#pragma unroll
            for (int i = 0; i < BM / 16; ++i) {
                const int r = i * 16 + (lane >> 2), c = kk + (lane & 3) * 2;
                a[i][0] = *reinterpret_cast<const uint32_t*>(&xs[r][c]);
                a[i][1] = *reinterpret_cast<const uint32_t*>(&xs[r + 8][c]);
                a[i][2] = *reinterpret_cast<const uint32_t*>(&xs[r][c + 8]);
                a[i][3] = *reinterpret_cast<const uint32_t*>(&xs[r + 8][c + 8]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int nr = warp * 16 + j * 8 + (lane >> 2), c = kk + (lane & 3) * 2;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&ws[nr][c]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&ws[nr][c + 8]);
#pragma unroll
                for (int i = 0; i < BM / 16; ++i) ik_mma_bf16_16816(acc[i][j], a[i][0], a[i][1], a[i][2], a[i][3], b0, b1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < BM / 16; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + (lane >> 2) + h * 8;
                const int64_t col = n0 + warp * 16 + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                if (col < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * h] * g);
                if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * h + 1] * g);
            }
}

template <int BM, typename OutT>
void mma_run(const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t M, int64_t N,
             int64_t K, cudaStream_t s) {
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    nvfp4_mma_kernel<BM, OutT><<<grid, 128, 0, s>>>(x, w, bs, gs, out, (int)M, N, (int)K);
}

template <typename OutT>
int mma_by_m(const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t M, int64_t N,
             int64_t K, cudaStream_t s) {
    if (M <= 16) mma_run<16>(x, w, bs, gs, out, M, N, K, s);
    else if (M <= 32) mma_run<32>(x, w, bs, gs, out, M, N, K, s);
    else mma_run<64>(x, w, bs, gs, out, M, N, K, s);
    return ik_launch_status();
}

// ---- launchers ---------------------------------------------------------------------------------------------------

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs, ik_tensor* out,
           void* stream, bool gemv, int max_gemv_rows = 8) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 8));
    IK_TRY(ik_check_a(bs, IK_DTYPE_FP8_E4M3, 2, 1));
    IK_TRY(ik_check_a(gs, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 16 != 0 || w->shape[1] != K / 2 || bs->shape[0] != N || bs->shape[1] != K / 16 || gs->shape[0] != 1 ||
        out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0) return cudaMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* bp = (const uint8_t*)bs->data;
    const float* gp = (const float*)gs->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > max_gemv_rows) return IK_ERR_SHAPE;
        return f32o ? gemv_by_m((int)M, xp, wp, bp, gp, (float*)out->data, N, K, s)
                    : gemv_by_m((int)M, xp, wp, bp, gp, (ik_bf16*)out->data, N, K, s);
    }
    return f32o ? mma_by_m(xp, wp, bp, gp, (float*)out->data, M, N, K, s)
                : mma_by_m(xp, wp, bp, gp, (ik_bf16*)out->data, M, N, K, s);
}

int glu(const ik_tensor* x, const ik_tensor* wg, const ik_tensor* bsg, const ik_tensor* gsg, const ik_tensor* wu,
        const ik_tensor* bsu, const ik_tensor* gsu, ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    for (const ik_tensor* w : {wg, wu}) IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 8));
    for (const ik_tensor* b : {bsg, bsu}) IK_TRY(ik_check_a(b, IK_DTYPE_FP8_E4M3, 2, 1));
    for (const ik_tensor* g : {gsg, gsu}) IK_TRY(ik_check_a(g, IK_DTYPE_F32, 1, 4));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = wg->shape[0];
    if (M != 1 || K % 16 != 0 || wu->shape[0] != N || wg->shape[1] != K / 2 || wu->shape[1] != K / 2 ||
        bsg->shape[0] != N || bsu->shape[0] != N || bsg->shape[1] != K / 16 || bsu->shape[1] != K / 16 ||
        gsg->shape[0] != 1 || gsu->shape[0] != 1 || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (N == 0) return IK_OK;
    if (K == 0)
        return cudaMemsetAsync(out->data, 0, ik_numel(out) * 2, IK_STREAM(stream)) == cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (K > 0x7fffffffLL || N > 0x7fffffffLL * kWarps) return IK_ERR_SHAPE;
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    nvfp4_glu_kernel<<<grid, kWarps * IK_WARP, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const uint8_t*)wg->data, (const uint8_t*)bsg->data, (const float*)gsg->data,
        (const uint8_t*)wu->data, (const uint8_t*)bsu->data, (const float*)gsu->data, (ik_bf16*)out->data, N, (int)K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_nvfp4_glu_gemv(const ik_tensor* x, const ik_tensor* w_gate, const ik_tensor* block_scales_gate,
                             const ik_tensor* global_scale_gate, const ik_tensor* w_up, const ik_tensor* block_scales_up,
                             const ik_tensor* global_scale_up, ik_tensor* out, void* stream) {
    return glu(x, w_gate, block_scales_gate, global_scale_gate, w_up, block_scales_up, global_scale_up, out, stream);
}

int ik_matmul_nvfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                         const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, true);
}
// Row-exact multi-row GEMV: row m of an M-row call (M <= 16) is bitwise the M = 1 `ik_matmul_nvfp4_gemv` call on row m alone
// (same kernel template: a warp per output row, the same lane-strided walk and fma chain per row, M independent accumulators).
int ik_matmul_nvfp4_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                              const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, true, 16);
}
int ik_matmul_nvfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                        const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, false);
}

}  // extern "C"
