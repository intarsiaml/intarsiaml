/* hc_split_sinkhorn (kernel.py:406-475) of one row in f32, shared by hc_mixes and hc_sublayer (dsv41_hc.cu, dsv41_hc_sub.cu).
 * Written from DeepSeek's official inference code (MIT). */
#ifndef IK_DSV41_HC_CUH
#define IK_DSV41_HC_CUH

#include "dsv41_common.cuh"

namespace dsv41 {

IK_DEVI float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }

/* hc_split_sinkhorn of one row, one warp: sums holds the row's M = (2+S)S raw projections (fn @ x) and, at [M], sum(x^2)
 * over its sd elements; writes pre / post [S], comb [S*S] of the row. Lane l < S*S holds comb[l / S][l % S], lanes j < S compute pre[j] /
 * post[j]. Every row / column sum gathers its S terms by shuffles in index order, so each f32 sum is formed in the reference's
 * loop order (and so are the maxima). All 32 lanes must call it; sums is shared memory. */
template <int S>
IK_DEVI void hc_split_sinkhorn_warp(const float* sums, int64_t sd, const float* __restrict__ scale, const float* __restrict__ base, float eps,
                                    int iters, float hc_eps, float* pre, float* post, float* comb) {
    constexpr int M = (2 + S) * S;
    const int lane = threadIdx.x & 31;
    const float rs = 1.f / sqrtf(sums[M] / (float)sd + eps);
    if (lane < S) {
        pre[lane] = sigmoidf_(sums[lane] * rs * scale[0] + base[lane]) + hc_eps;
        post[lane] = 2.f * sigmoidf_(sums[S + lane] * rs * scale[1] + base[S + lane]) ;
    }
    const int l = lane < S * S ? lane : 0;
    const int j = l / S, k = l % S;
    float c = sums[2 * S + l] * rs * scale[2] + base[2 * S + l];
    // softmax over the row, + eps
    float mx = -INFINITY;
#pragma unroll
    for (int kk = 0; kk < S; ++kk) mx = fmaxf(mx, __shfl_sync(0xffffffffu, c, j * S + kk));
    c = expf(c - mx);
    float sum = 0.f;
#pragma unroll
    for (int kk = 0; kk < S; ++kk) sum += __shfl_sync(0xffffffffu, c, j * S + kk);
    c = c / sum + hc_eps;
    // column normalisation
    sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < S; ++jj) sum += __shfl_sync(0xffffffffu, c, jj * S + k);
    c /= sum + hc_eps;
    for (int it = 1; it < iters; ++it) {
        sum = 0.f;
#pragma unroll
        for (int kk = 0; kk < S; ++kk) sum += __shfl_sync(0xffffffffu, c, j * S + kk);
        c /= sum + hc_eps;
        sum = 0.f;
#pragma unroll
        for (int jj = 0; jj < S; ++jj) sum += __shfl_sync(0xffffffffu, c, jj * S + k);
        c /= sum + hc_eps;
    }
    if (lane < S * S) comb[lane] = c;
}

}  // namespace dsv41

#endif /* IK_DSV41_HC_CUH */
