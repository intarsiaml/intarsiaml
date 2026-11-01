// DeepSeek-V4.1 hyper-connections: hc_expand, hc_mixes (projection + RMS statistic + split + 4x4 Sinkhorn, all f32 as
// Block.hc_mixes / hc_split_sinkhorn, model.py:950-957, kernel.py:406-475), hc_pre, hc_post (model.py:959-968).
// Contract: docs/design/dsv41-ops.md. Written from the official inference code (MIT, DeepSeek).
#include "dsv41_common.cuh"

using namespace dsv41;

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// hc_mixes: rows per block of the partial-dot kernel and the largest (2+S)S it takes (S <= 4)
constexpr int kRows = 4;
constexpr int kMaxMix = 24;
// at least this many x elements per split (two per thread), so a split's fixed reduction cost stays small
constexpr int64_t kMinSplitElems = 512;
// blocks the split targets for small T (GB10: 48 SMs)
constexpr int64_t kTargetBlocks = 192;

template <typename Ti, typename To>
__global__ void hc_expand_kernel(const Ti* __restrict__ x, To* __restrict__ y, int64_t T, int64_t S, int64_t D) {
    const int64_t n = T * S * D;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t d = i % D, t = i / (S * D);
        st(y, i, ld(x, t * D + d));
    }
}

// partial sums of one split of the flattened row: (fn @ x)[j] for j < M and sum(x^2), for kRows rows at once
template <typename T>
__global__ void __launch_bounds__(kThreads) hc_mix_partial(const T* __restrict__ x, const float* __restrict__ fn, int64_t rows,
                                                           int64_t sd, int m, int nsplit, float* __restrict__ part) {
    __shared__ float red[kWarps][kRows * (kMaxMix + 1)];
    const int sp = blockIdx.x;
    const int64_t r0 = (int64_t)blockIdx.y * kRows;
    const int64_t per = cdiv(sd, nsplit);
    const int64_t lo = sp * per, hi = min(lo + per, sd);
    float acc[kRows][kMaxMix + 1];
#pragma unroll
    for (int r = 0; r < kRows; ++r)
#pragma unroll
        for (int j = 0; j <= kMaxMix; ++j) acc[r][j] = 0.f;
    for (int64_t e = lo + threadIdx.x; e < hi; e += kThreads) {
        float xv[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) xv[r] = (r0 + r < rows) ? ld(x, (r0 + r) * sd + e) : 0.f;
#pragma unroll
        for (int r = 0; r < kRows; ++r) acc[r][kMaxMix] = fmaf(xv[r], xv[r], acc[r][kMaxMix]);
#pragma unroll
        for (int j = 0; j < kMaxMix; ++j) {
            if (j < m) {
                const float f = __ldg(fn + (int64_t)j * sd + e);
#pragma unroll
                for (int r = 0; r < kRows; ++r) acc[r][j] = fmaf(f, xv[r], acc[r][j]);
            }
        }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < kRows; ++r)
#pragma unroll
        for (int j = 0; j <= kMaxMix; ++j) {
            const float v = ik_warp_sum(acc[r][j]);
            if (lane == 0) red[w][r * (kMaxMix + 1) + j] = v;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < kRows * (m + 1); i += kThreads) {
        const int r = i / (m + 1), j = i % (m + 1);
        if (r0 + r >= rows) continue;
        const int jj = (j == m) ? kMaxMix : j;
        float s = 0.f;
        for (int ww = 0; ww < kWarps; ++ww) s += red[ww][r * (kMaxMix + 1) + jj];
        part[((r0 + r) * nsplit + sp) * (m + 1) + j] = s;
    }
}

IK_DEVI float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }

// one thread per row: sum the splits in order, then hc_split_sinkhorn
__global__ void hc_mix_finish(const float* __restrict__ part, int64_t rows, int64_t sd, int s, int nsplit, const float* __restrict__ scale,
                              const float* __restrict__ base, float eps, int iters, float hc_eps, float* __restrict__ pre,
                              float* __restrict__ post, float* __restrict__ comb) {
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int m = (2 + s) * s;
    float mix[kMaxMix + 1];
    for (int j = 0; j <= m; ++j) mix[j] = 0.f;
    for (int sp = 0; sp < nsplit; ++sp) {
        const float* p = part + (r * nsplit + sp) * (m + 1);
        for (int j = 0; j <= m; ++j) mix[j] += p[j];
    }
    const float rs = 1.f / sqrtf(mix[m] / (float)sd + eps);
    for (int j = 0; j < m; ++j) mix[j] *= rs;
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    for (int j = 0; j < s; ++j) {
        pre[r * s + j] = sigmoidf_(mix[j] * s0 + base[j]) + hc_eps;
        post[r * s + j] = 2.f * sigmoidf_(mix[s + j] * s1 + base[s + j]);
    }
    float c[16];
    for (int j = 0; j < s * s; ++j) c[j] = mix[2 * s + j] * s2 + base[2 * s + j];
    // comb = softmax(comb, -1) + eps; comb = comb / (comb.sum(-2) + eps)
    for (int j = 0; j < s; ++j) {
        float mx = -INFINITY;
        for (int k = 0; k < s; ++k) mx = fmaxf(mx, c[j * s + k]);
        float sum = 0.f;
        for (int k = 0; k < s; ++k) {
            c[j * s + k] = expf(c[j * s + k] - mx);
            sum += c[j * s + k];
        }
        for (int k = 0; k < s; ++k) c[j * s + k] = c[j * s + k] / sum + hc_eps;
    }
    for (int k = 0; k < s; ++k) {
        float sum = 0.f;
        for (int j = 0; j < s; ++j) sum += c[j * s + k];
        for (int j = 0; j < s; ++j) c[j * s + k] /= sum + hc_eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < s; ++j) {
            float sum = 0.f;
            for (int k = 0; k < s; ++k) sum += c[j * s + k];
            for (int k = 0; k < s; ++k) c[j * s + k] /= sum + hc_eps;
        }
        for (int k = 0; k < s; ++k) {
            float sum = 0.f;
            for (int j = 0; j < s; ++j) sum += c[j * s + k];
            for (int j = 0; j < s; ++j) c[j * s + k] /= sum + hc_eps;
        }
    }
    for (int j = 0; j < s * s; ++j) comb[r * s * s + j] = c[j];
}

template <typename Ti, typename To>
__global__ void hc_pre_kernel(const Ti* __restrict__ x, const float* __restrict__ mix, To* __restrict__ y, int64_t T, int s, int64_t D) {
    const int64_t n = T * D;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = i / D, d = i % D;
        float acc = 0.f;
        for (int c = 0; c < s; ++c) acc = fmaf(mix[t * s + c], ld(x, (t * s + c) * D + d), acc);
        st(y, i, acc);
    }
}

template <typename Tx, typename Tr, typename To>
__global__ void hc_post_kernel(const Tx* __restrict__ x, const Tr* __restrict__ res, const float* __restrict__ post,
                               const float* __restrict__ comb, To* __restrict__ y, int64_t T, int s, int64_t D) {
    const int64_t n = T * D;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = i / D, d = i % D;
        const float xv = ld(x, i);
        float rv[8];
        for (int j = 0; j < s; ++j) rv[j] = ld(res, (t * s + j) * D + d);
        for (int k = 0; k < s; ++k) {
            float v = post[t * s + k] * xv;
            for (int j = 0; j < s; ++j) v = fmaf(comb[(t * s + j) * s + k], rv[j], v);
            st(y, (t * s + k) * D + d, v);
        }
    }
}

}  // namespace

/* Split count of hc_mixes' partial-dot kernel (mirrored by `hc_nsplit` in crates/intarsia-backend-cuda/src/dsv41.rs). */
static int64_t hc_nsplit(int64_t rows, int64_t sd) {
    const int64_t rb = cdiv(rows, kRows);
    int64_t mx = sd / kMinSplitElems;
    if (mx < 1) mx = 1;
    int64_t n = cdiv(kTargetBlocks, rb);
    if (n > mx) n = mx;
    if (n < 1) n = 1;
    return n;
}

extern "C" int ik_hc_expand(const ik_tensor* x, ik_tensor* y, void* stream) {
    IK_TRY(check_float(x, 2));
    IK_TRY(check_float(y, 3));
    const int64_t T = x->shape[0], D = x->shape[1], S = y->shape[1];
    if (y->shape[0] != T || y->shape[2] != D) return IK_ERR_SHAPE;
    if (T * S * D == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    DSV_FLOAT_DISPATCH(x->dtype, Ti, DSV_FLOAT_DISPATCH(y->dtype, To,
        hc_expand_kernel<Ti, To><<<grid1d(T * S * D, kThreads), kThreads, 0, s>>>((const Ti*)x->data, (To*)y->data, T, S, D)));
    return ik_launch_status();
}

extern "C" int ik_hc_mixes(const ik_tensor* x, const ik_tensor* fn, const ik_tensor* scale, const ik_tensor* base, float eps,
                           int64_t iters, float hc_eps, ik_tensor* pre, ik_tensor* post, ik_tensor* comb, const ik_scratch* scratch,
                           void* stream) {
    IK_TRY(check_float(x, 3));
    const int64_t T = x->shape[0], S = x->shape[1], D = x->shape[2];
    const int64_t m = (2 + S) * S, sd = S * D;
    if (S < 1 || m > kMaxMix || iters < 1) return IK_ERR_SHAPE;
    IK_TRY(ik_check(fn, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(scale, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(base, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(pre, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(post, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(comb, IK_DTYPE_F32, 3));
    if (fn->shape[0] != m || fn->shape[1] != sd || scale->shape[0] != 3 || base->shape[0] != m) return IK_ERR_SHAPE;
    if (pre->shape[0] != T || pre->shape[1] != S || post->shape[0] != T || post->shape[1] != S || comb->shape[0] != T ||
        comb->shape[1] != S || comb->shape[2] != S)
        return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    const int64_t ns = hc_nsplit(T, sd);
    const int64_t need = T * ns * (m + 1) * 4;
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    float* part = (float*)scratch->data;
    cudaStream_t s = IK_STREAM(stream);
    dim3 g((unsigned)ns, (unsigned)cdiv(T, kRows));
    DSV_FLOAT_DISPATCH(x->dtype, Tx,
        hc_mix_partial<Tx><<<g, kThreads, 0, s>>>((const Tx*)x->data, (const float*)fn->data, T, sd, (int)m, (int)ns, part));
    IK_TRY(ik_launch_status());
    hc_mix_finish<<<(unsigned)cdiv(T, 64), 64, 0, s>>>(part, T, sd, (int)S, (int)ns, (const float*)scale->data,
                                                       (const float*)base->data, eps, (int)iters, hc_eps, (float*)pre->data,
                                                       (float*)post->data, (float*)comb->data);
    return ik_launch_status();
}

extern "C" int ik_hc_pre(const ik_tensor* x, const ik_tensor* mix, ik_tensor* y, void* stream) {
    IK_TRY(check_float(x, 3));
    IK_TRY(ik_check(mix, IK_DTYPE_F32, 2));
    IK_TRY(check_float(y, 2));
    const int64_t T = x->shape[0], S = x->shape[1], D = x->shape[2];
    if (mix->shape[0] != T || mix->shape[1] != S || y->shape[0] != T || y->shape[1] != D || S > 8) return IK_ERR_SHAPE;
    if (T * D == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    DSV_FLOAT_DISPATCH(x->dtype, Ti, DSV_FLOAT_DISPATCH(y->dtype, To,
        hc_pre_kernel<Ti, To><<<grid1d(T * D, kThreads), kThreads, 0, s>>>((const Ti*)x->data, (const float*)mix->data,
                                                                         (To*)y->data, T, (int)S, D)));
    return ik_launch_status();
}

extern "C" int ik_hc_post(const ik_tensor* x, const ik_tensor* res, const ik_tensor* post, const ik_tensor* comb, ik_tensor* y,
                          void* stream) {
    IK_TRY(check_float(x, 2));
    IK_TRY(check_float(res, 3));
    IK_TRY(ik_check(post, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(comb, IK_DTYPE_F32, 3));
    IK_TRY(check_float(y, 3));
    const int64_t T = res->shape[0], S = res->shape[1], D = res->shape[2];
    if (x->shape[0] != T || x->shape[1] != D || post->shape[0] != T || post->shape[1] != S || comb->shape[0] != T ||
        comb->shape[1] != S || comb->shape[2] != S || y->shape[0] != T || y->shape[1] != S || y->shape[2] != D || S > 8)
        return IK_ERR_SHAPE;
    if (T * D == 0) return IK_OK;
    if (x->dtype != res->dtype) return IK_ERR_DTYPE;
    cudaStream_t s = IK_STREAM(stream);
    DSV_FLOAT_DISPATCH(x->dtype, Tx, DSV_FLOAT_DISPATCH(y->dtype, To,
        hc_post_kernel<Tx, Tx, To><<<grid1d(T * D, kThreads), kThreads, 0, s>>>((const Tx*)x->data, (const Tx*)res->data,
                                                                              (const float*)post->data, (const float*)comb->data,
                                                                              (To*)y->data, T, (int)S, D)));
    return ik_launch_status();
}
