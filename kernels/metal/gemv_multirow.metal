// Multi-row (R activation rows) bit-exact BF16 GEMVs for speculative-decoding verification (G0 prototype,
// docs/results/2026-10-15-metal-g0-multirow.md).
//
// Contract: output row r of every kernel here is BITWISE what the serving 1-row decode GEMV produces for activation row r:
// each thread, each simdgroup and each threadgroup accumulates one activation row in exactly the order of the 1-row
// kernel it mirrors, the weights are streamed once per threadgroup and used against all R rows, and R only adds independent
// accumulators. Nothing here may be retuned per R (SG, UNROLL, lane->chunk mapping are those of the mirrored kernel).
//
//   gemv_bf16_rt_u2s8_mr<R>        == gemv_bf16_rt_u2s8        (kernels.rs DEFAULT_GEMV: 16-byte loads, K % 8 == 0)
//   gemv_bf16_rt_a8u4s8_mr<R>      == gemv_bf16_rt_a8u4s8      (DEFAULT_GEMV_A8: 8-byte loads, K % 4 == 0)
//   gemv_pair_bf16_rt_u2s8_mr<R>   == gemv_pair_bf16_rt_u2s8   (MatmulPair, 16-byte)
//   gemv_pair_bf16_rt_a8u4s8_mr<R> == gemv_pair_bf16_rt_a8u4s8 (MatmulPair, 8-byte)
//   gemv_f32o_v_mr<R>              == gemv_f32o_v             (Logits, f32 out, 16-byte)
//
// Layouts: X [R, K] (row stride K), Y [R, N] (row stride N, element offset y_skip), W [N, K] as in the 1-row kernels.
// R rows of X are read at stride K*2 bytes, so a row has the alignment of row 0 whenever K % 8 == 0 (K % 4 == 0 for the
// 8-byte kernels), which is also what the 1-row kernels need.

#include <metal_stdlib>
using namespace metal;

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

struct PairParams {
    uint k;
    uint n0;
    uint n1;
    uint y0_skip;
    uint y1_skip;
    uint pad0;
    uint pad1;
    uint pad2;
};

inline float lo_f(uint u) { return as_type<float>(u << 16); }
inline float hi_f(uint u) { return as_type<float>(u & 0xffff0000u); }

// Same fma chain as gemv_bf16.metal / gemv_f32out.metal dot8 / dot4.
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

inline float dot4(uint2 w, uint2 x) {
    float a = lo_f(w.x) * lo_f(x.x);
    a = fma(hi_f(w.x), hi_f(x.x), a);
    a = fma(lo_f(w.y), lo_f(x.y), a);
    a = fma(hi_f(w.y), hi_f(x.y), a);
    return a;
}

inline ushort to_bf16(float v) { return as_type<ushort>(bfloat(v)); }

// One row's worth of the rt kernels' body, R activation rows at once. `part` is threadgroup float[R * SG].
// 16-byte: lane L of the SG*32-lane threadgroup reads chunk (step*UNROLL + u)*lanes + L of the weight row.
template <int R, int UNROLL, int SG>
inline void rt_row16(device const uint4* wr, device const uint4* X, uint k8, uint sg, uint lane,
                     threadgroup float* part, thread float* out) {
    constexpr uint lanes = 32 * SG;
    constexpr uint block = lanes * UNROLL;
    const uint nblk = k8 / block;
    uint i = sg * 32 + lane;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 wv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) wv[u] = wr[i + lanes * u];
        for (int r = 0; r < R; ++r) {
            uint4 xv[UNROLL];
            for (int u = 0; u < UNROLL; ++u) xv[u] = X[(ulong)r * k8 + i + lanes * u];
            for (int u = 0; u < UNROLL; ++u) acc[r] += dot8(wv[u], xv[u]);
        }
    }
    for (i = nblk * block + sg * 32 + lane; i < k8; i += lanes) {
        const uint4 wv = wr[i];
        for (int r = 0; r < R; ++r) acc[r] += dot8(wv, X[(ulong)r * k8 + i]);
    }
    for (int r = 0; r < R; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == 0) part[r * SG + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < R; ++r) {
        float t = part[r * SG];
        for (int j = 1; j < SG; ++j) t += part[r * SG + j];
        out[r] = t;
    }
}

template <int R, int UNROLL, int SG>
inline void rt_row8(device const uint2* wr, device const uint2* X, uint k4, uint sg, uint lane,
                    threadgroup float* part, thread float* out) {
    constexpr uint lanes = 32 * SG;
    constexpr uint block = lanes * UNROLL;
    const uint nblk = k4 / block;
    uint i = sg * 32 + lane;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint2 wv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) wv[u] = wr[i + lanes * u];
        for (int r = 0; r < R; ++r) {
            uint2 xv[UNROLL];
            for (int u = 0; u < UNROLL; ++u) xv[u] = X[(ulong)r * k4 + i + lanes * u];
            for (int u = 0; u < UNROLL; ++u) acc[r] += dot4(wv[u], xv[u]);
        }
    }
    for (i = nblk * block + sg * 32 + lane; i < k4; i += lanes) {
        const uint2 wv = wr[i];
        for (int r = 0; r < R; ++r) acc[r] += dot4(wv, X[(ulong)r * k4 + i]);
    }
    for (int r = 0; r < R; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == 0) part[r * SG + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < R; ++r) {
        float t = part[r * SG];
        for (int j = 1; j < SG; ++j) t += part[r * SG + j];
        out[r] = t;
    }
}

// ---- Matmul ---------------------------------------------------------------------------------------------------------
template <int R>
kernel void gemv_bf16_rt_u2s8_mr(device const uint4* W [[buffer(0)]],
                                 device const uint4* X [[buffer(1)]],
                                 device ushort* Y [[buffer(2)]],
                                 constant GemvParams& p [[buffer(3)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[R * 8];
    const uint k8 = p.k / 8;
    float t[R];
    rt_row16<R, 2, 8>(W + (ulong)tg * k8, X, k8, sg, lane, part, t);
    if (sg == 0 && lane == 0) {
        for (int r = 0; r < R; ++r) Y[p.y_skip + (ulong)r * p.n + tg] = to_bf16(t[r]);
    }
}

template <int R>
kernel void gemv_bf16_rt_a8u4s8_mr(device const uint2* W [[buffer(0)]],
                                   device const uint2* X [[buffer(1)]],
                                   device ushort* Y [[buffer(2)]],
                                   constant GemvParams& p [[buffer(3)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[R * 8];
    const uint k4 = p.k / 4;
    float t[R];
    rt_row8<R, 4, 8>(W + (ulong)tg * k4, X, k4, sg, lane, part, t);
    if (sg == 0 && lane == 0) {
        for (int r = 0; r < R; ++r) Y[p.y_skip + (ulong)r * p.n + tg] = to_bf16(t[r]);
    }
}

// ---- MatmulPair (threadgroups [0, n0) work on W0, the rest on W1) ----------------------------------------------------
template <int R>
kernel void gemv_pair_bf16_rt_u2s8_mr(device const uint4* W0 [[buffer(0)]],
                                      device const uint4* W1 [[buffer(1)]],
                                      device const uint4* X [[buffer(2)]],
                                      device ushort* Y0 [[buffer(3)]],
                                      device ushort* Y1 [[buffer(4)]],
                                      constant PairParams& p [[buffer(5)]],
                                      uint tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[R * 8];
    const bool second = tg >= p.n0;
    device const uint4* W = second ? W1 : W0;
    device ushort* Y = (second ? Y1 : Y0) + (second ? p.y1_skip : p.y0_skip);
    const uint n = second ? p.n1 : p.n0;
    const uint row = second ? tg - p.n0 : tg;
    const uint k8 = p.k / 8;
    float t[R];
    rt_row16<R, 2, 8>(W + (ulong)row * k8, X, k8, sg, lane, part, t);
    if (sg == 0 && lane == 0) {
        for (int r = 0; r < R; ++r) Y[(ulong)r * n + row] = to_bf16(t[r]);
    }
}

template <int R>
kernel void gemv_pair_bf16_rt_a8u4s8_mr(device const uint2* W0 [[buffer(0)]],
                                        device const uint2* W1 [[buffer(1)]],
                                        device const uint2* X [[buffer(2)]],
                                        device ushort* Y0 [[buffer(3)]],
                                        device ushort* Y1 [[buffer(4)]],
                                        constant PairParams& p [[buffer(5)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[R * 8];
    const bool second = tg >= p.n0;
    device const uint2* W = second ? W1 : W0;
    device ushort* Y = (second ? Y1 : Y0) + (second ? p.y1_skip : p.y0_skip);
    const uint n = second ? p.n1 : p.n0;
    const uint row = second ? tg - p.n0 : tg;
    const uint k4 = p.k / 4;
    float t[R];
    rt_row8<R, 4, 8>(W + (ulong)row * k4, X, k4, sg, lane, part, t);
    if (sg == 0 && lane == 0) {
        for (int r = 0; r < R; ++r) Y[(ulong)r * n + row] = to_bf16(t[r]);
    }
}

// ---- Logits, f32 out (one simdgroup per row, 4 rows per threadgroup, UNROLL 4, simd_sum only) ------------------------
template <int R>
kernel void gemv_f32o_v_mr(device const uint4* W [[buffer(0)]],
                           device const uint4* X [[buffer(1)]],
                           device float* Y [[buffer(2)]],
                           constant GemvParams& p [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint k8 = p.k / 8;
    const uint row = tg * 4 + sg;
    device const uint4* wr = W + (ulong)min(row, p.n - 1) * k8;  // rows past N read a valid row and are not stored
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    const uint block = 32 * 4;
    const uint nblk = k8 / block;
    uint i = lane;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 wv[4];
        for (int u = 0; u < 4; ++u) wv[u] = wr[i + 32 * u];
        for (int r = 0; r < R; ++r) {
            uint4 xv[4];
            for (int u = 0; u < 4; ++u) xv[u] = X[(ulong)r * k8 + i + 32 * u];
            for (int u = 0; u < 4; ++u) acc[r] += dot8(wv[u], xv[u]);
        }
    }
    for (i = nblk * block + lane; i < k8; i += 32) {
        const uint4 wv = wr[i];
        for (int r = 0; r < R; ++r) acc[r] += dot8(wv, X[(ulong)r * k8 + i]);
    }
    for (int r = 0; r < R; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == 0 && row < p.n) Y[p.y_skip + (ulong)r * p.n + row] = s;
    }
}

#define INST_MR(KERNEL, R, ...)                                                                                       \
    template [[host_name(#KERNEL "_mr" #R)]] kernel void KERNEL##_mr<R>(__VA_ARGS__);

#define INST_MR_ALL(KERNEL, ...)  \
    INST_MR(KERNEL, 1, __VA_ARGS__) INST_MR(KERNEL, 2, __VA_ARGS__) INST_MR(KERNEL, 3, __VA_ARGS__)   \
    INST_MR(KERNEL, 4, __VA_ARGS__) INST_MR(KERNEL, 5, __VA_ARGS__) INST_MR(KERNEL, 6, __VA_ARGS__)   \
    INST_MR(KERNEL, 7, __VA_ARGS__) INST_MR(KERNEL, 8, __VA_ARGS__) INST_MR(KERNEL, 16, __VA_ARGS__)

INST_MR_ALL(gemv_bf16_rt_u2s8, device const uint4*, device const uint4*, device ushort*, constant GemvParams&, uint, uint, uint)
INST_MR_ALL(gemv_bf16_rt_a8u4s8, device const uint2*, device const uint2*, device ushort*, constant GemvParams&, uint, uint, uint)
INST_MR_ALL(gemv_pair_bf16_rt_u2s8, device const uint4*, device const uint4*, device const uint4*, device ushort*, device ushort*, constant PairParams&, uint, uint, uint)
INST_MR_ALL(gemv_pair_bf16_rt_a8u4s8, device const uint2*, device const uint2*, device const uint2*, device ushort*, device ushort*, constant PairParams&, uint, uint, uint)
INST_MR_ALL(gemv_f32o_v, device const uint4*, device const uint4*, device float*, constant GemvParams&, uint, uint, uint)
