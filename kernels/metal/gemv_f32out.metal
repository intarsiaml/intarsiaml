// Batch-1 BF16 GEMV with f32 output, for `Logits` with M = 1: y[N] = W[N,K] . x[K]. Same structure and layout contract
// as gemv_bf16.metal (whose kernels write bf16): `gemv_f32o_v` needs K % 8 == 0 and 16-byte aligned W and x, one simdgroup
// per row, four 16-byte loads in flight per lane, 4 rows per threadgroup; `gemv_f32o_scalar` takes any K and any 2-byte
// alignment (w_skip / x_skip), 8 rows per threadgroup. y is f32 and 4-byte aligned. GemvParams is the one of
// gemv_bf16.metal (y_skip is 0 here).

#include "common.h"

struct GemvParams {
    uint k;
    uint n;
    uint w_skip;
    uint x_skip;
    uint y_skip;
    uint pad0;
    uint pad1;
    uint pad2;
};

inline float lo_f(uint u) { return as_type<float>(u << 16); }
inline float hi_f(uint u) { return as_type<float>(u & 0xffff0000u); }

inline float dot8(uint4 w, uint4 x) {
    float a = lo_f(w.x) * lo_f(x.x);
    a = fma(hi_f(w.x), hi_f(x.x), a);
    a = fma(lo_f(w.y), lo_f(x.y), a);
    a = fma(hi_f(w.y), hi_f(x.y), a);
    a = fma(lo_f(w.z), lo_f(x.z), a);
    a = fma(hi_f(w.z), hi_f(x.z), a);
    a = fma(lo_f(w.w), lo_f(x.w), a);
    a = fma(hi_f(w.w), hi_f(x.w), a);
    return a;
}

kernel void gemv_f32o_v(device const uint4* W [[buffer(0)]],
                        device const uint4* X [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant GemvParams& p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint k8 = p.k / 8;
    const uint row = tg * 4 + sg;
    device const uint4* wr = W + (ulong)min(row, p.n - 1) * k8;  // rows past N read a valid row and are not stored
    float acc = 0.0f;
    const uint block = 32 * 4;
    const uint nblk = k8 / block;
    uint i = lane;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 xv[4], wv[4];
        for (int u = 0; u < 4; ++u) xv[u] = X[i + 32 * u];
        for (int u = 0; u < 4; ++u) wv[u] = wr[i + 32 * u];
        for (int u = 0; u < 4; ++u) acc += dot8(wv[u], xv[u]);
    }
    for (i = nblk * block + lane; i < k8; i += 32) acc += dot8(wr[i], X[i]);
    const float s = simd_sum(acc);
    if (lane == 0 && row < p.n) Y[p.y_skip + row] = s;
}

kernel void gemv_f32o_scalar(device const ushort* W [[buffer(0)]],
                             device const ushort* X [[buffer(1)]],
                             device float* Y [[buffer(2)]],
                             constant GemvParams& p [[buffer(3)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8 + sg;
    device const ushort* w = W + p.w_skip + (ulong)min(row, p.n - 1) * p.k;
    device const ushort* x = X + p.x_skip;
    float acc = 0.0f;
    for (uint i = lane; i < p.k; i += 32) acc = fma(bf2f(w[i]), bf2f(x[i]), acc);
    const float s = simd_sum(acc);
    if (lane == 0 && row < p.n) Y[p.y_skip + row] = s;
}
