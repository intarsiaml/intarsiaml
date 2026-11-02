// MPROF roofs of this box: attainable DRAM read bandwidth, fp32 FMA peak, simdgroup 8x8 MMA peak. Standalone micro-benchmark.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void bw_read(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    uint4 acc = uint4(0);
    for (uint i = gid; i < n4; i += gsz) acc ^= src[i];
    if ((acc.x | acc.y | acc.z | acc.w) == 0x9e3779b9u) out[0] = 1;
}

kernel void fma_f32(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a0 = gid * 1e-9f, a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f, a4 = a0 + 4.0f, a5 = a0 + 5.0f, a6 = a0 + 6.0f, a7 = a0 + 7.0f;
    const float b = 0.99999f, c = 1e-7f;
    for (uint i = 0; i < iters; ++i) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

kernel void mma_f32(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(1.0f + gid * 1e-9f), b = make_filled_simdgroup_matrix<float, 8, 8>(0.5f);
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    simdgroup_float8x8 s = c0;
    out[gid] = s.thread_elements()[0] + c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0];
}

kernel void mma_f16(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8, 8>(half(1.0f)), b = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    out[gid] = c0.thread_elements()[0] + c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0];
}
