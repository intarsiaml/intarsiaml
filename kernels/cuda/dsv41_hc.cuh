/* hc_split_sinkhorn (kernel.py:406-475) of one row in f32, shared by hc_mixes and hc_sublayer (dsv41_hc.cu, dsv41_hc_sub.cu).
 * Written from DeepSeek's official inference code (MIT). */
#ifndef IK_DSV41_HC_CUH
#define IK_DSV41_HC_CUH

#include "dsv41_common.cuh"

namespace dsv41 {

IK_DEVI float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }

/* sums: the row's M = (2+S)S raw projections (fn @ x) and, at [M], sum(x^2) over its sd elements; writes pre / post [S],
 * comb [S*S] of the row. One thread. */
template <int S>
IK_DEVI void hc_split_sinkhorn(const float* sums, int64_t sd, const float* __restrict__ scale, const float* __restrict__ base, float eps,
                               int iters, float hc_eps, float* pre, float* post, float* comb) {
    constexpr int M = (2 + S) * S;
    float mix[M];
    const float rs = 1.f / sqrtf(sums[M] / (float)sd + eps);
#pragma unroll
    for (int j = 0; j < M; ++j) mix[j] = sums[j] * rs;
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
#pragma unroll
    for (int j = 0; j < S; ++j) {
        pre[j] = sigmoidf_(mix[j] * s0 + base[j]) + hc_eps;
        post[j] = 2.f * sigmoidf_(mix[S + j] * s1 + base[S + j]);
    }
    float c[S * S];
#pragma unroll
    for (int j = 0; j < S * S; ++j) c[j] = mix[2 * S + j] * s2 + base[2 * S + j];
    // comb = softmax(comb, -1) + eps; comb = comb / (comb.sum(-2) + eps)
#pragma unroll
    for (int j = 0; j < S; ++j) {
        float mx = -INFINITY;
#pragma unroll
        for (int k = 0; k < S; ++k) mx = fmaxf(mx, c[j * S + k]);
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < S; ++k) {
            c[j * S + k] = expf(c[j * S + k] - mx);
            sum += c[j * S + k];
        }
#pragma unroll
        for (int k = 0; k < S; ++k) c[j * S + k] = c[j * S + k] / sum + hc_eps;
    }
#pragma unroll
    for (int k = 0; k < S; ++k) {
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < S; ++j) sum += c[j * S + k];
#pragma unroll
        for (int j = 0; j < S; ++j) c[j * S + k] /= sum + hc_eps;
    }
    for (int it = 1; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < S; ++j) {
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < S; ++k) sum += c[j * S + k];
#pragma unroll
            for (int k = 0; k < S; ++k) c[j * S + k] /= sum + hc_eps;
        }
#pragma unroll
        for (int k = 0; k < S; ++k) {
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < S; ++j) sum += c[j * S + k];
#pragma unroll
            for (int j = 0; j < S; ++j) c[j * S + k] /= sum + hc_eps;
        }
    }
#pragma unroll
    for (int j = 0; j < S * S; ++j) comb[j] = c[j];
}

}  // namespace dsv41

#endif /* IK_DSV41_HC_CUH */
