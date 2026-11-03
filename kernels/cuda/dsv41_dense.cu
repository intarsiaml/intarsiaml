// DeepSeek-V4.1 dense linears with f16 weights (lane dsv41/tp2): the Mia checkpoint keeps the MoE router
// (`layers.{l}.ffn.gate.weight`, [384, 5120]) and the indexer's `weights_proj` ([32, 5120]) in f16, and the backend's dense
// kernels take bf16 weights only (rounding the router to bf16 would move expert choices). y[m, n] = sum_k x[m, k] w[n, k],
// f32 accumulation, any float x / y dtype.
//
// Vector path (K % 8 == 0, 16-byte aligned rows): a block of 8 warps owns 8 / WPO outputs and a tile of R rows. Each output
// is split over WPO warps; lane l of part p reads 8 consecutive weights (one 16-byte load) at k8 = p * 32 + l + i * 32 * WPO
// and applies them to all R rows, so a weight is read once per R rows (the old one-row-per-block kernel read the router
// once per prompt token in prefill). Per row: 8 fmas in k order per load, lanes added by an xor butterfly, parts added in
// part order. WPO is a function of N only and R does not enter a row's sum, so a row's bits do not depend on M or on its
// place in the batch (decode == prefill bit for bit). The scalar kernel is the fallback for other K / alignments.
#include "dsv41_common.cuh"
#include <cstdlib>

using namespace dsv41;

namespace {
constexpr int kWarps = 8;

template <typename X, typename Y>
__global__ void dense_f16w_kernel(const X* __restrict__ x, const __half* __restrict__ w, Y* __restrict__ y, int64_t M, int64_t N,
                                  int64_t K) {
    const int lane = threadIdx.x & 31;
    const int64_t n = (int64_t)blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int64_t m = blockIdx.y;
    if (n >= N || m >= M) return;
    const X* xr = x + m * K;
    const __half* wr = w + n * K;
    float acc = 0.f;
    for (int64_t k = lane; k < K; k += 32) acc += ld<X>(xr, k) * __half2float(wr[k]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) st<Y>(y, m * N + n, acc);
}

// 8 consecutive x values of a row as floats (one or two 16-byte loads)
template <typename X> __device__ __forceinline__ void ld8(const X* p, float (&v)[8]);
template <> __device__ __forceinline__ void ld8<float>(const float* p, float (&v)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
template <> __device__ __forceinline__ void ld8<__nv_bfloat16>(const __nv_bfloat16* p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        v[2 * j] = f.x; v[2 * j + 1] = f.y;
    }
}
template <> __device__ __forceinline__ void ld8<__half>(const __half* p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h[j]);
        v[2 * j] = f.x; v[2 * j + 1] = f.y;
    }
}

template <typename X, typename Y, int R, int WPO>
__global__ void __launch_bounds__(kWarps * 32) dense_f16w_vec_kernel(const X* __restrict__ x, const __half* __restrict__ w,
                                                                      Y* __restrict__ y, int M, int N, int K) {
    __shared__ float part_sum[kWarps][R];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int part = warp % WPO;
    const int n = blockIdx.x * (kWarps / WPO) + warp / WPO;
    const int m0 = blockIdx.y * R;
    const int rn = min(R, M - m0);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    if (n < N) {
        const __half* wr = w + (size_t)n * K;
        const X* xr = x + (size_t)m0 * K;
        const int k8n = K >> 3;
#pragma unroll 2
        for (int k8 = part * 32 + lane; k8 < k8n; k8 += 32 * WPO) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(wr) + k8);
            const __half2* h = reinterpret_cast<const __half2*>(&u);
            float wf[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(h[j]);
                wf[2 * j] = f.x; wf[2 * j + 1] = f.y;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < rn) {
                    float xv[8];
                    ld8<X>(xr + (size_t)r * K + (size_t)k8 * 8, xv);
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] = fmaf(xv[j], wf[j], acc[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    }
    if constexpr (WPO == 1) {
        if (n < N && lane < R && lane < rn) {
            float v = 0.f;
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (r == lane) v = acc[r];
            st<Y>(y, (size_t)(m0 + lane) * N + n, v);
        }
    } else {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) part_sum[warp][r] = acc[r];
        }
        __syncthreads();
        if (part == 0 && n < N && lane < rn) {
            float v = part_sum[warp][lane];
#pragma unroll
            for (int p = 1; p < WPO; ++p) v += part_sum[warp + p][lane];
            st<Y>(y, (size_t)(m0 + lane) * N + n, v);
        }
    }
}

// Warps per output: the fewest (1, 2, 4, 8) that give at least 96 blocks (two per GB10 SM), a function of N only.
int wpo_for(int64_t N) {
    int wpo = 1;
    while (wpo < kWarps && cdiv(N * wpo, kWarps) < 96) wpo *= 2;
    return wpo;
}

template <typename X, typename Y, int R>
void launch_vec(const X* x, const __half* w, Y* y, int M, int N, int K, cudaStream_t s) {
    const int wpo = wpo_for(N);
    dim3 grid((unsigned)cdiv((int64_t)N * wpo, kWarps), (unsigned)cdiv(M, R));
    switch (wpo) {
        case 1: dense_f16w_vec_kernel<X, Y, R, 1><<<grid, kWarps * 32, 0, s>>>(x, w, y, M, N, K); break;
        case 2: dense_f16w_vec_kernel<X, Y, R, 2><<<grid, kWarps * 32, 0, s>>>(x, w, y, M, N, K); break;
        case 4: dense_f16w_vec_kernel<X, Y, R, 4><<<grid, kWarps * 32, 0, s>>>(x, w, y, M, N, K); break;
        default: dense_f16w_vec_kernel<X, Y, R, 8><<<grid, kWarps * 32, 0, s>>>(x, w, y, M, N, K); break;
    }
}

// INTARSIA_DSV41_F16W_SCALAR=1: the scalar kernel at every shape (the kernel before the vector path, same sum order as
// then: the numerics A/B of the vector path)
static bool f16w_scalar() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("INTARSIA_DSV41_F16W_SCALAR");
        on = (e && e[0] == '1' && e[1] == 0) ? 1 : 0;
    }
    return on == 1;
}

static int f16w_rows() {
    static int r = -1;
    if (r < 0) {
        const char* e = getenv("INTARSIA_DSV41_F16W_ROWS");
        // 8 rows per weight read by default; 16 was the earlier default (slower, docs/results/2026-11-02-dsv41-moegate.md),
        // 1 is the decode tile at every M. R never enters a row's sum, so every value gives the same bits.
        r = 8;
        if (e && e[0] == '1' && e[1] == 0) r = 1;
        else if (e && e[0] == '1' && e[1] == '6' && e[2] == 0) r = 16;
    }
    return r;
}

template <typename X, typename Y>
int dense_f16w_typed(const X* xp, const __half* wp, Y* yp, int64_t M, int64_t N, int64_t K, bool vec, cudaStream_t s) {
    if (vec) {
        // R = 1 for decode (no idle row registers), 8 rows per weight read in prefill. INTARSIA_DSV41_F16W_ROWS=1 keeps
        // R = 1 at every M, =16 the earlier prefill tile (diagnosis; R never enters a row's sum, so the bits are the same)
        const int R = f16w_rows();
        if (M == 1 || R == 1) launch_vec<X, Y, 1>(xp, wp, yp, (int)M, (int)N, (int)K, s);
        else if (R == 16) launch_vec<X, Y, 16>(xp, wp, yp, (int)M, (int)N, (int)K, s);
        else launch_vec<X, Y, 8>(xp, wp, yp, (int)M, (int)N, (int)K, s);
    } else {
        dim3 grid((unsigned)cdiv(N, kWarps), (unsigned)M);
        dense_f16w_kernel<X, Y><<<grid, kWarps * 32, 0, s>>>(xp, wp, yp, M, N, K);
    }
    return ik_launch_status();
}

template <typename X>
int dense_f16w_out(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, int64_t M, int64_t N, int64_t K, bool vec, cudaStream_t s) {
    const X* xp = (const X*)x->data;
    const __half* wp = (const __half*)w->data;
    switch (y->dtype) {
        case IK_DTYPE_F32: return dense_f16w_typed<X, float>(xp, wp, (float*)y->data, M, N, K, vec, s);
        case IK_DTYPE_BF16: return dense_f16w_typed<X, __nv_bfloat16>(xp, wp, (__nv_bfloat16*)y->data, M, N, K, vec, s);
        default: return dense_f16w_typed<X, __half>(xp, wp, (__half*)y->data, M, N, K, vec, s);
    }
}
}  // namespace

static int dense_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream) {
    IK_TRY(check_float(x, 2));
    IK_TRY(check_float(y, 2));
    if (w->dtype != IK_DTYPE_F16 || w->ndim != 2) return IK_ERR_DTYPE;
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (w->shape[1] != K || y->shape[0] != M || y->shape[1] != N || M > 65535) return IK_ERR_SHAPE;
    if (M * N == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    // vector path: whole 16-byte loads of every row (row strides of 8 elements, 16-byte aligned bases), int32 indexing
    const bool vec = !f16w_scalar() && K % 8 == 0 && ((uintptr_t)x->data & 15u) == 0 && ((uintptr_t)w->data & 15u) == 0 && M * K < (1LL << 31) &&
                     N * K < (1LL << 31) && M * N < (1LL << 31);
    switch (x->dtype) {
        case IK_DTYPE_F32: return dense_f16w_out<float>(x, w, y, M, N, K, vec, s);
        case IK_DTYPE_BF16: return dense_f16w_out<__nv_bfloat16>(x, w, y, M, N, K, vec, s);
        default: return dense_f16w_out<__half>(x, w, y, M, N, K, vec, s);
    }
}

// One launcher per registered kernel name (Op::Logits / Op::Matmul), both the same f16-weight GEMV.
extern "C" int ik_logits_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream) {
    return dense_f16w(x, w, y, stream);
}
extern "C" int ik_matmul_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream) {
    return dense_f16w(x, w, y, stream);
}
