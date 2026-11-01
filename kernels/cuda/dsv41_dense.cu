// DeepSeek-V4.1 dense linears with f16 weights (lane dsv41/tp2): the Mia checkpoint keeps the MoE router
// (`layers.{l}.ffn.gate.weight`, [384, 5120]) and the indexer's `weights_proj` ([32, 5120]) in f16, and the backend's dense
// kernels take bf16 weights only (rounding the router to bf16 would move expert choices). y[m, n] = sum_k x[m, k] w[n, k],
// f32 accumulation, any float x / y dtype. One warp per output, eight per block; small N (<= a few thousand) by design.
#include "dsv41_common.cuh"

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

template <typename X>
int dense_f16w_out(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, int64_t M, int64_t N, int64_t K, cudaStream_t s) {
    dim3 grid((unsigned)cdiv(N, kWarps), (unsigned)M);
    const X* xp = (const X*)x->data;
    const __half* wp = (const __half*)w->data;
    switch (y->dtype) {
        case IK_DTYPE_F32: dense_f16w_kernel<X, float><<<grid, kWarps * 32, 0, s>>>(xp, wp, (float*)y->data, M, N, K); break;
        case IK_DTYPE_BF16:
            dense_f16w_kernel<X, __nv_bfloat16><<<grid, kWarps * 32, 0, s>>>(xp, wp, (__nv_bfloat16*)y->data, M, N, K);
            break;
        default: dense_f16w_kernel<X, __half><<<grid, kWarps * 32, 0, s>>>(xp, wp, (__half*)y->data, M, N, K); break;
    }
    return ik_launch_status();
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
    switch (x->dtype) {
        case IK_DTYPE_F32: return dense_f16w_out<float>(x, w, y, M, N, K, s);
        case IK_DTYPE_BF16: return dense_f16w_out<__nv_bfloat16>(x, w, y, M, N, K, s);
        default: return dense_f16w_out<__half>(x, w, y, M, N, K, s);
    }
}

// One launcher per registered kernel name (Op::Logits / Op::Matmul), both the same f16-weight GEMV.
extern "C" int ik_logits_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream) {
    return dense_f16w(x, w, y, stream);
}
extern "C" int ik_matmul_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream) {
    return dense_f16w(x, w, y, stream);
}
