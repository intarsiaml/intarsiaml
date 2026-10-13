// matmul_fp8: out[M,N] = x[M,K] @ W^T, W = float(w_e4m3[N,K]) * weight_scale (per tensor).
//   scales = f32[2] = (weight_scale, input_scale); input_scale > 0 selects W8A8 (static activation quantisation:
//   xq = e4m3(x / input_scale), RNE, saturating; y = weight_scale * input_scale * sum xq*wq), otherwise W8A16.
//   The mode is read from the device tensor inside the kernels (a uniform branch), so launching never needs a
//   host read and the same kernels serve both recipes.
//   *_gemv  decode: M <= 4, warp-per-output-row, x staged in shared memory (quantised once per block).
//   *_mma   any M: 64-column tiles, weights dequantised to bf16 in shared memory (exact), bf16 tensor-core MMA,
//           f32 accumulate. Used for prefill and as the correct path for every shape.
// Both are deterministic (fixed reduction order). Activations bf16; output bf16 or f32.
#include "ik_launch.h"
#include "ik_quant.cuh"

namespace {

constexpr int kWarps = 8;

// ---- GEMV --------------------------------------------------------------------------------------------------------

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                OutT* __restrict__ out, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* xs = reinterpret_cast<ik_bf16*>(smem);  // [M][K] bf16: x, or xq (fp8 values are exact in bf16)
    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        const ik_bf16 v = x[i];
        xs[i] = actq ? ik_f2b(ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(ik_b2f(v), s_in)))) : v;
    }
    __syncthreads();
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;
    const uint4* wr = reinterpret_cast<const uint4*>(w + n * K);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int u = lane; u < K / 16; u += IK_WARP) {
        float wf[16];
        ik_fp8x16_to_f32(__ldcs(wr + u), wf);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(xs + (int64_t)m * K + u * 16);
            float xf[16];
            ik_unpack8(xp[0], xf);
            ik_unpack8(xp[1], xf + 8);
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
            acc[m] += s;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) ik_store(out + (int64_t)m * N + n, s * s_out);
    }
}

template <int M, typename OutT>
void gemv_run(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    const size_t smem = (size_t)M * K * 2;
    fp8_gemv_kernel<M, OutT><<<grid, kWarps * IK_WARP, smem, s>>>(x, w, sc, out, (int)N, (int)K);
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    switch (M) {
        case 1: gemv_run<1>(x, w, sc, out, N, K, s); break;
        case 2: gemv_run<2>(x, w, sc, out, N, K, s); break;
        case 3: gemv_run<3>(x, w, sc, out, N, K, s); break;
        case 4: gemv_run<4>(x, w, sc, out, N, K, s); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

// ---- MMA (dequantise-to-bf16 tiles) -----------------------------------------------------------------------------

constexpr int BN = 64, BK = 64, LDS = BK + 8;  // LDS: padded row stride in bf16 elements (conflict-free fragment loads)

template <int BM, typename OutT>
__global__ void __launch_bounds__(128)
fp8_mma_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
               OutT* __restrict__ out, int M, int N, int K) {
    __shared__ __align__(16) ik_bf16 xs[BM][LDS];
    __shared__ __align__(16) ik_bf16 ws[BN][LDS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    float acc[BM / 16][2][4];
#pragma unroll
    for (int i = 0; i < BM / 16; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // activations: BM x 64, 8 elements per thread-iteration
        for (int idx = tid; idx < BM * 8; idx += 128) {
            const int r = idx >> 3, c = (idx & 7) * 8;
            const int gm = m0 + r, gk = k0 + c;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gm < M && gk < K) {
                v = *reinterpret_cast<const uint4*>(x + (int64_t)gm * K + gk);
                if (actq) {
                    float f[8];
                    ik_unpack8(v, f);
#pragma unroll
                    for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
                    v = ik_pack8(f);  // fp8 values are exact in bf16
                }
            }
            *reinterpret_cast<uint4*>(&xs[r][c]) = v;
        }
        // weights: 64 rows x 64 bytes -> bf16 (exact), 16 bytes per thread-iteration
        for (int idx = tid; idx < BN * 4; idx += 128) {
            const int r = idx >> 2, c = (idx & 3) * 16;
            const int gn = n0 + r, gk = k0 + c;
            uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
            if (gn < N && gk < K) {
                float f[16];
                ik_fp8x16_to_f32(__ldcs(reinterpret_cast<const uint4*>(w + (int64_t)gn * K + gk)), f);
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
                const int col = n0 + warp * 16 + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                if (col < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * h] * s_out);
                if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * h + 1] * s_out);
            }
}

template <int BM, typename OutT>
void mma_run(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t M, int64_t N, int64_t K, cudaStream_t s) {
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    fp8_mma_kernel<BM, OutT><<<grid, 128, 0, s>>>(x, w, sc, out, (int)M, (int)N, (int)K);
}

template <typename OutT>
int mma_by_m(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t M, int64_t N, int64_t K, cudaStream_t s) {
    if (M <= 16) mma_run<16>(x, w, sc, out, M, N, K, s);
    else if (M <= 32) mma_run<32>(x, w, sc, out, M, N, K, s);
    else mma_run<64>(x, w, sc, out, M, N, K, s);
    return ik_launch_status();
}

// ---- launchers ---------------------------------------------------------------------------------------------------

int check(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* out) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    if (K % 16 != 0) return IK_ERR_SHAPE;
    return IK_OK;
}

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream, bool gemv) {
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0) return cudaMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > 4 || M * K * 2 > 49152) return IK_ERR_SHAPE;
        return f32o ? gemv_by_m((int)M, xp, wp, sp, (float*)out->data, N, K, s)
                    : gemv_by_m((int)M, xp, wp, sp, (ik_bf16*)out->data, N, K, s);
    }
    return f32o ? mma_by_m(xp, wp, sp, (float*)out->data, M, N, K, s) : mma_by_m(xp, wp, sp, (ik_bf16*)out->data, M, N, K, s);
}

}  // namespace

extern "C" {

int ik_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, true);
}
int ik_matmul_fp8_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, false);
}

}  // extern "C"
