// DeepSeek-V4.1 fused hyper-connection sublayer entry, Op::HcSublayer (docs/design/dsv41-ops.md): optionally the previous
// sublayer's HcPost (h = post_mix x_prev + comb^T res, stored to h_out), then HcMixes + Sinkhorn on h, the collapse of h with
// the previous sublayer's pre-mix (HcPre) and the sublayer RmsNorm, in two launches:
//  1. partial dots: grid (nsplit, row tiles of 16). A block stages one 128-element chunk of h for its rows in shared memory
//     (computing and storing h with the fold), then each warp owns a few of the (2+S)S projection rows (and the sum of
//     squares), every lane 4 elements, fn read once per chunk as float4 and reused for all rows of the tile. With more than
//     one chunk per block the lanes accumulate across chunks before the warp reduction. Writes f32 partials
//     [T, nsplit, (2+S)S + 1] (scratch).
//  2. finish: one block per row. All threads collapse h with pre_in (f32, rounded to n's dtype as HcPre stores it), reduce the
//     sum of squares, write n = y * rsqrt(mean(y^2) + eps) * (w + w_offset); warp 0 sums the partials in split order and lane
//     0 runs hc_split_sinkhorn (shared with hc_mixes, dsv41_hc.cuh).
// Written from DeepSeek's official inference code (MIT): Block.hc_mixes / hc_pre / hc_post, model.py:950-968, kernel.py:406-475.
#include "dsv41_hc.cuh"

using namespace dsv41;

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunk = 128;   // h elements per chunk: 32 lanes x 4
constexpr int kTileRows = 16; // rows per block of the partial kernel
constexpr int kTargetBlocks = 1536;  // ~32 blocks per SM: the fold's staging is latency-bound
constexpr int kRowsPerWarp = 4;  // projection rows (incl. the sum of squares) per warp: 8 warps x 4 >= (2+S)S + 1 = 25

template <typename T> IK_DEVI T cvt(float v);
template <> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half_rn(v); }

template <typename T> IK_DEVI float back(T v);
template <> __device__ __forceinline__ float back<float>(float v) { return v; }
template <> __device__ __forceinline__ float back<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float back<__half>(__half v) { return __half2float(v); }

// Tx: x / res and post_x dtype; Th: h_out dtype (fold only)
template <typename Tx, typename Th, bool FOLD>
__global__ void __launch_bounds__(kThreads) hc_sub_partial(const Tx* __restrict__ x, const float* __restrict__ fn, const Tx* __restrict__ px,
                                                         const float* __restrict__ pmix, const float* __restrict__ pcomb, Th* __restrict__ hout,
                                                         int64_t T, int S, int64_t D, int m, int nsplit, float* __restrict__ part) {
    __shared__ __align__(16) float xs[kTileRows][kChunk];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t sd = (int64_t)S * D;
    const int64_t t0 = (int64_t)blockIdx.y * kTileRows;
    const int nt = (int)min((int64_t)kTileRows, T - t0);
    const int64_t nchunk = cdiv(sd, kChunk);
    float acc[kRowsPerWarp][kTileRows];
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i)
#pragma unroll
        for (int r = 0; r < kTileRows; ++r) acc[i][r] = 0.f;
    for (int64_t ch = blockIdx.x; ch < nchunk; ch += nsplit) {
        const int64_t e0 = ch * kChunk;
        __syncthreads();
        for (int v = threadIdx.x; v < kTileRows * kChunk; v += kThreads) {
            const int r = v / kChunk, c = v % kChunk;
            const int64_t e = e0 + c;
            float val = 0.f;
            if (r < nt && e < sd) {
                const int64_t t = t0 + r;
                if (FOLD) {
                    const int k = (int)(e / D);
                    const int64_t d = e % D;
                    float h = pmix[t * S + k] * ld(px, t * D + d);
                    for (int j = 0; j < S; ++j) h = fmaf(pcomb[(t * S + j) * S + k], ld(x, (t * S + j) * D + d), h);
                    const Th hq = cvt<Th>(h);
                    hout[t * sd + e] = hq;
                    val = back<Th>(hq);
                } else {
                    val = ld(x, t * sd + e);
                }
            }
            xs[r][c] = val;
        }
        __syncthreads();
        const int64_t e = e0 + lane * 4;
        const bool full = e + 4 <= sd;
#pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i) {
            const int j = warp + kWarps * i;
            if (j > m) continue;  // warp-uniform
            float4 f;
            if (j == m) {
                f = make_float4(0.f, 0.f, 0.f, 0.f);
            } else if (full) {
                f = __ldg(reinterpret_cast<const float4*>(fn + (int64_t)j * sd + e));
            } else {
                const float* fr = fn + (int64_t)j * sd;
                f.x = e < sd ? fr[e] : 0.f;
                f.y = e + 1 < sd ? fr[e + 1] : 0.f;
                f.z = e + 2 < sd ? fr[e + 2] : 0.f;
                f.w = e + 3 < sd ? fr[e + 3] : 0.f;
            }
#pragma unroll
            for (int r = 0; r < kTileRows; ++r) {
                if (r >= nt) break;
                const float4 xv = *reinterpret_cast<const float4*>(&xs[r][lane * 4]);
                if (j == m) {
                    acc[i][r] = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, fmaf(xv.z, xv.z, fmaf(xv.w, xv.w, acc[i][r]))));
                } else {
                    acc[i][r] = fmaf(f.x, xv.x, fmaf(f.y, xv.y, fmaf(f.z, xv.z, fmaf(f.w, xv.w, acc[i][r]))));
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        const int j = warp + kWarps * i;
        if (j > m) continue;
#pragma unroll
        for (int r = 0; r < kTileRows; ++r) {
            if (r >= nt) break;
            const float v = ik_warp_sum(acc[i][r]);
            if (lane == 0) part[((t0 + r) * nsplit + blockIdx.x) * (m + 1) + j] = v;
        }
    }
}

// Th: dtype of the h read here (h_out with the fold, x without); To: n's dtype; Tw: norm weight dtype
template <int S, typename Th, typename To, typename Tw, bool SINK = true>
__global__ void __launch_bounds__(kThreads) hc_sub_finish(const Th* __restrict__ h, const float* __restrict__ pre_in, const Tw* __restrict__ w,
                                                        const float* __restrict__ part, const float* __restrict__ scale,
                                                        const float* __restrict__ base, int64_t D, int nsplit, float eps, float w_offset,
                                                        int iters, float hc_eps, To* __restrict__ n, float* __restrict__ pre,
                                                        float* __restrict__ post, float* __restrict__ comb) {
    constexpr int M = (2 + S) * S;
    extern __shared__ float ys[];  // [D]
    __shared__ float red[kWarps];
    __shared__ float sums[32];
    const int64_t t = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float pm[S];
#pragma unroll
    for (int c = 0; c < S; ++c) pm[c] = pre_in[t * S + c];
    float ss = 0.f;
    const Th* hr = h + t * S * D;
    for (int64_t d = threadIdx.x; d < D; d += kThreads) {
        float y = 0.f;
#pragma unroll
        for (int c = 0; c < S; ++c) y = fmaf(pm[c], back<Th>(hr[c * D + d]), y);
        y = back<To>(cvt<To>(y));  // HcPre stores y in n's dtype
        ys[d] = y;
        ss = fmaf(y, y, ss);
    }
    ss = ik_block_sum(ss, red);
    const float rs = 1.f / sqrtf(ss / (float)D + eps);
    for (int64_t d = threadIdx.x; d < D; d += kThreads) n[t * D + d] = cvt<To>(ys[d] * rs * (back<Tw>(w[d]) + w_offset));
    if constexpr (!SINK) return;
    if (warp != 0) return;
    float v = 0.f;
    if (lane <= M)
        for (int sp = 0; sp < nsplit; ++sp) v += part[(t * nsplit + sp) * (M + 1) + lane];
    sums[lane] = v;
    __syncwarp();
    hc_split_sinkhorn_warp<S>(sums, (int64_t)S * D, scale, base, eps, iters, hc_eps, pre + t * S, post + t * S, comb + t * S * S);
}

// The two halves of the op for INTARSIA_DSV41_HC_SIDE, same bits as the fused launches: the main stream's half needs nothing
// the dots and the Sinkhorn produce (they are the next sublayer's inputs), so those can run beside the next kernels.
// h = post_mix x_prev + comb^T res alone: the partial kernel's fold without its dots.
template <typename Tx, typename Th>
__global__ void __launch_bounds__(kThreads) hc_sub_post_only(const Tx* __restrict__ x, const Tx* __restrict__ px, const float* __restrict__ pmix,
                                                            const float* __restrict__ pcomb, Th* __restrict__ hout, int S, int64_t D) {
    const int64_t sd = (int64_t)S * D, t = blockIdx.y;
    const int64_t e = (int64_t)blockIdx.x * kThreads + threadIdx.x;
    if (e >= sd) return;
    const int k = (int)(e / D);
    const int64_t d = e % D;
    float h = pmix[t * S + k] * ld(px, t * D + d);
    for (int j = 0; j < S; ++j) h = fmaf(pcomb[(t * S + j) * S + k], ld(x, (t * S + j) * D + d), h);
    hout[t * sd + e] = cvt<Th>(h);
}

// The finish's Sinkhorn alone: one warp per row.
template <int S>
__global__ void __launch_bounds__(32) hc_sub_sinkhorn(const float* __restrict__ part, const float* __restrict__ scale, const float* __restrict__ base,
                                                      int64_t D, int nsplit, float eps, int iters, float hc_eps, float* __restrict__ pre,
                                                      float* __restrict__ post, float* __restrict__ comb) {
    constexpr int M = (2 + S) * S;
    __shared__ float sums[32];
    const int64_t t = blockIdx.x;
    const int lane = threadIdx.x;
    float v = 0.f;
    if (lane <= M)
        for (int sp = 0; sp < nsplit; ++sp) v += part[(t * nsplit + sp) * (M + 1) + lane];
    sums[lane] = v;
    __syncwarp();
    hc_split_sinkhorn_warp<S>(sums, (int64_t)S * D, scale, base, eps, iters, hc_eps, pre + t * S, post + t * S, comb + t * S * S);
}

}  // namespace

/* Splits of hc_sublayer's partial kernel (mirrored by `hc_sub_nsplit` in crates/intarsia-backend-cuda/src/dsv41.rs). */
static int64_t hc_sub_nsplit(int64_t rows, int64_t sd) {
    const int64_t tiles = cdiv(rows, kTileRows);
    int64_t n = cdiv(kTargetBlocks, tiles < 1 ? 1 : tiles);
    const int64_t mx = cdiv(sd, kChunk);
    if (n > mx) n = mx;
    if (n < 1) n = 1;
    return n;
}

template <typename Tx, typename Th, typename To, typename Tw, int S>
static int run_sub(const ik_tensor* x, const ik_tensor* fn, const ik_tensor* scale, const ik_tensor* base, const ik_tensor* pre_in,
                   const ik_tensor* w, const ik_tensor* px, const ik_tensor* pmix, const ik_tensor* pcomb, float eps, float w_offset, int iters,
                   float hc_eps, ik_tensor* n, ik_tensor* pre, ik_tensor* post, ik_tensor* comb, ik_tensor* hout, bool fold, float* part,
                   int64_t ns, cudaStream_t s, int mode) {
    const int64_t T = x->shape[0], D = x->shape[2];
    const int m = (2 + S) * S;
    if (mode == 1) {
        // main half: the post (fold), then n; part, pre, post and comb are not touched
        if (fold) {
            dim3 g((unsigned)cdiv((int64_t)S * D, kThreads), (unsigned)T);
            hc_sub_post_only<Tx, Th><<<g, kThreads, 0, s>>>((const Tx*)x->data, (const Tx*)px->data, (const float*)pmix->data,
                                                            (const float*)pcomb->data, (Th*)hout->data, S, D);
            IK_TRY(ik_launch_status());
        }
        const size_t bytes = (size_t)D * sizeof(float);
        if (bytes > 48 * 1024) cudaFuncSetAttribute(hc_sub_finish<S, Th, To, Tw, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
        const void* hs = fold ? hout->data : x->data;
        hc_sub_finish<S, Th, To, Tw, false><<<(unsigned)T, kThreads, bytes, s>>>((const Th*)hs, (const float*)pre_in->data, (const Tw*)w->data, part,
                                                                                (const float*)scale->data, (const float*)base->data, D, (int)ns, eps,
                                                                                w_offset, iters, hc_eps, (To*)n->data, (float*)pre->data,
                                                                                (float*)post->data, (float*)comb->data);
        return ik_launch_status();
    }
    if (mode == 2) {
        // side half: the dots of the stored h, then the Sinkhorn
        const void* hs = fold ? hout->data : x->data;
        dim3 g((unsigned)ns, (unsigned)cdiv(T, kTileRows));
        hc_sub_partial<Th, Th, false><<<g, kThreads, 0, s>>>((const Th*)hs, (const float*)fn->data, nullptr, nullptr, nullptr, nullptr,
                                                            T, S, D, m, (int)ns, part);
        IK_TRY(ik_launch_status());
        hc_sub_sinkhorn<S><<<(unsigned)T, 32, 0, s>>>(part, (const float*)scale->data, (const float*)base->data, D, (int)ns, eps, iters, hc_eps,
                                                      (float*)pre->data, (float*)post->data, (float*)comb->data);
        return ik_launch_status();
    }
    dim3 g((unsigned)ns, (unsigned)cdiv(T, kTileRows));
    if (fold)
        hc_sub_partial<Tx, Th, true><<<g, kThreads, 0, s>>>((const Tx*)x->data, (const float*)fn->data, (const Tx*)px->data,
                                                           (const float*)pmix->data, (const float*)pcomb->data, (Th*)hout->data, T, S, D,
                                                           m, (int)ns, part);
    else
        hc_sub_partial<Tx, Th, false><<<g, kThreads, 0, s>>>((const Tx*)x->data, (const float*)fn->data, nullptr, nullptr, nullptr, nullptr,
                                                            T, S, D, m, (int)ns, part);
    IK_TRY(ik_launch_status());
    const size_t smem = (size_t)D * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(hc_sub_finish<S, Th, To, Tw>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    const void* hsrc = fold ? hout->data : x->data;
    hc_sub_finish<S, Th, To, Tw><<<(unsigned)T, kThreads, smem, s>>>((const Th*)hsrc, (const float*)pre_in->data, (const Tw*)w->data, part,
                                                                    (const float*)scale->data, (const float*)base->data, D, (int)ns, eps,
                                                                    w_offset, iters, hc_eps, (To*)n->data, (float*)pre->data,
                                                                    (float*)post->data, (float*)comb->data);
    return ik_launch_status();
}

extern "C" int ik_hc_sublayer_mode(int mode, const ik_tensor* x, const ik_tensor* fn, const ik_tensor* scale, const ik_tensor* base, const ik_tensor* pre_in,
                              const ik_tensor* norm_w, const ik_tensor* post_x, const ik_tensor* post_mix, const ik_tensor* post_comb,
                              float eps, float w_offset, int64_t iters, float hc_eps, ik_tensor* n, ik_tensor* pre, ik_tensor* post,
                              ik_tensor* comb, ik_tensor* h_out, const ik_scratch* scratch, void* stream) {
    IK_TRY(check_float(x, 3));
    const int64_t T = x->shape[0], S = x->shape[1], D = x->shape[2];
    const int64_t m = (2 + S) * S, sd = S * D;
    if (S < 1 || S > 4 || iters < 1 || sd % 4 != 0 || D > 12288) return IK_ERR_SHAPE;
    IK_TRY(ik_check(fn, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(scale, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(base, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(pre_in, IK_DTYPE_F32, 2));
    IK_TRY(check_float(norm_w, 1));
    IK_TRY(check_float(n, 2));
    IK_TRY(ik_check(pre, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(post, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(comb, IK_DTYPE_F32, 3));
    if (fn->shape[0] != m || fn->shape[1] != sd || scale->shape[0] != 3 || base->shape[0] != m || pre_in->shape[0] != T ||
        pre_in->shape[1] != S || norm_w->shape[0] != D || n->shape[0] != T || n->shape[1] != D || pre->shape[0] != T || pre->shape[1] != S ||
        post->shape[0] != T || post->shape[1] != S || comb->shape[0] != T || comb->shape[1] != S || comb->shape[2] != S)
        return IK_ERR_SHAPE;
    if (!ik_aligned16(fn->data)) return IK_ERR_INVALID_ARG;
    const bool fold = post_x && ik_numel(post_x) > 0;
    if (fold) {
        IK_TRY(ik_check(post_x, x->dtype, 2));
        IK_TRY(ik_check(post_mix, IK_DTYPE_F32, 2));
        IK_TRY(ik_check(post_comb, IK_DTYPE_F32, 3));
        IK_TRY(check_float(h_out, 3));
        if (post_x->shape[0] != T || post_x->shape[1] != D || post_mix->shape[0] != T || post_mix->shape[1] != S || post_comb->shape[0] != T ||
            post_comb->shape[1] != S || post_comb->shape[2] != S || h_out->shape[0] != T || h_out->shape[1] != S || h_out->shape[2] != D)
            return IK_ERR_SHAPE;
    }
    if (T == 0) return IK_OK;
    const int64_t ns = hc_sub_nsplit(T, sd);
    const int64_t need = T * ns * (m + 1) * 4;
    // mode 0: the whole op; 1: the main half, no scratch; 2: the side half (INTARSIA_DSV41_HC_SIDE)
    if (mode < 0 || mode > 2) return IK_ERR_INVALID_ARG;
    if (mode != 1 && (!scratch || !scratch->data || scratch->bytes < need)) return IK_ERR_INVALID_ARG;
    float* part = mode == 1 ? nullptr : (float*)scratch->data;
    cudaStream_t s = IK_STREAM(stream);
    // dtypes: x, post_x, h_out and n share one float dtype (the model: bf16; the sweep also all-f32); the norm weight is that
    // dtype or f32 (the official RMSNorm keeps its weight in f32)
    const int hdt = fold ? h_out->dtype : x->dtype;
    if (hdt != x->dtype || n->dtype != x->dtype || (norm_w->dtype != x->dtype && norm_w->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    const bool wf32 = norm_w->dtype == IK_DTYPE_F32;
#define HC_SUB_RUN(TT, TW, SS)                                                                                                \
    return run_sub<TT, TT, TT, TW, SS>(x, fn, scale, base, pre_in, norm_w, post_x, post_mix, post_comb, eps, w_offset, (int)iters, \
                                       hc_eps, n, pre, post, comb, h_out, fold, part, ns, s, mode)
#define HC_SUB_S(TT, TW)                \
    switch (S) {                        \
        case 1: HC_SUB_RUN(TT, TW, 1);  \
        case 2: HC_SUB_RUN(TT, TW, 2);  \
        case 3: HC_SUB_RUN(TT, TW, 3);  \
        default: HC_SUB_RUN(TT, TW, 4); \
    }
#define HC_SUB_W(TT)              \
    if (wf32) {                   \
        HC_SUB_S(TT, float)       \
    } else {                      \
        HC_SUB_S(TT, TT)          \
    }
    switch (x->dtype) {
        case IK_DTYPE_F32: HC_SUB_S(float, float)
        case IK_DTYPE_BF16: HC_SUB_W(__nv_bfloat16)
        case IK_DTYPE_F16: HC_SUB_W(__half)
        default: return IK_ERR_DTYPE;
    }
#undef HC_SUB_RUN
#undef HC_SUB_W
#undef HC_SUB_S
}

extern "C" int ik_hc_sublayer(const ik_tensor* x, const ik_tensor* fn, const ik_tensor* scale, const ik_tensor* base, const ik_tensor* pre_in,
                              const ik_tensor* norm_w, const ik_tensor* post_x, const ik_tensor* post_mix, const ik_tensor* post_comb,
                              float eps, float w_offset, int64_t iters, float hc_eps, ik_tensor* n, ik_tensor* pre, ik_tensor* post,
                              ik_tensor* comb, ik_tensor* h_out, const ik_scratch* scratch, void* stream) {
    return ik_hc_sublayer_mode(0, x, fn, scale, base, pre_in, norm_w, post_x, post_mix, post_comb, eps, w_offset, iters, hc_eps, n, pre, post, comb,
                               h_out, scratch, stream);
}
