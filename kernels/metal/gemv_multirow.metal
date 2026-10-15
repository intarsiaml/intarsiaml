// Multi-row (R activation rows) bit-exact BF16 GEMVs for speculative-decoding verification (G0 prototype,
// docs/results/2026-10-15-metal-g0-multirow.md).
//
// Contract: output row r of every kernel here is BITWISE what the serving 1-row decode GEMV produces for activation row r:
// each thread, each simdgroup and each threadgroup accumulates one activation row in exactly the order of the 1-row
// kernel it mirrors, the weights are streamed once per threadgroup and used against all R rows, and R only adds independent
// accumulators. Nothing here may be retuned per R (SG, UNROLL, lane->chunk mapping are those of the mirrored kernel).
//
//   gemv_bf16_rt_u2s8_mr<R,P>      == gemv_bf16_rt_u2s8        (kernels.rs DEFAULT_GEMV: 16-byte loads, K % 8 == 0)
//   gemv_bf16_rt_a8u4s8_mr<R>      == gemv_bf16_rt_a8u4s8      (DEFAULT_GEMV_A8: 8-byte loads, K % 4 == 0)
//   gemv_pair_bf16_rt_u2s8_mr<R>   == gemv_pair_bf16_rt_u2s8   (MatmulPair, 16-byte)
//   gemv_pair_bf16_rt_a8u4s8_mr<R> == gemv_pair_bf16_rt_a8u4s8 (MatmulPair, 8-byte)
//   gemv_f32o_v_mr<R>              == gemv_f32o_v             (Logits, f32 out, 16-byte)
//
// Template parameters: R activation rows, P output rows per threadgroup (rt kernels) or per simdgroup (f32-out kernel).
// P > 1 shares each loaded x chunk between P weight rows (x is re-read from cache by every threadgroup, so its traffic
// is R*K*2 bytes per threadgroup, i.e. R times the 1-row kernel's per weight byte at P = 1); every (weight row, activation
// row) accumulator still follows the 1-row kernel's lane mapping and reduction order, so P does not change any bit.
// Host names: <kernel>_mr<R>x<P>.
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

// Shared body of the rt kernels: P output rows (first = row0, clamped to nvalid-1 for the loads) x R activation rows.
// 16-byte: lane L of the SG*32-lane threadgroup reads chunk (step*UNROLL + u)*lanes + L of the weight row.
// `part` is threadgroup float[P * R * SG]; `out[p][r]` is valid on the thread with sg == 0 && lane == 0.
template <int R, int P, int UNROLL, int SG>
inline void rt_rows16(device const uint4* W, uint row0, uint nvalid, device const uint4* X, uint k8, uint sg, uint lane,
                      threadgroup float* part, thread float* out) {
    constexpr uint lanes = 32 * SG;
    constexpr uint block = lanes * UNROLL;
    device const uint4* wr[P];
    for (int p = 0; p < P; ++p) wr[p] = W + (ulong)min(row0 + p, nvalid - 1) * k8;
    const uint nblk = k8 / block;
    uint i = sg * 32 + lane;
    float acc[P * R];
    for (int j = 0; j < P * R; ++j) acc[j] = 0.0f;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 wv[P][UNROLL];
        for (int p = 0; p < P; ++p)
            for (int u = 0; u < UNROLL; ++u) wv[p][u] = wr[p][i + lanes * u];
        for (int r = 0; r < R; ++r) {
            uint4 xv[UNROLL];
            for (int u = 0; u < UNROLL; ++u) xv[u] = X[(ulong)r * k8 + i + lanes * u];
            for (int p = 0; p < P; ++p)
                for (int u = 0; u < UNROLL; ++u) acc[p * R + r] += dot8(wv[p][u], xv[u]);
        }
    }
    for (i = nblk * block + sg * 32 + lane; i < k8; i += lanes) {
        uint4 wv[P];
        for (int p = 0; p < P; ++p) wv[p] = wr[p][i];
        for (int r = 0; r < R; ++r) {
            const uint4 xv = X[(ulong)r * k8 + i];
            for (int p = 0; p < P; ++p) acc[p * R + r] += dot8(wv[p], xv);
        }
    }
    for (int j = 0; j < P * R; ++j) {
        const float s = simd_sum(acc[j]);
        if (lane == 0) part[j * SG + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < P * R; ++j) {
        float t = part[j * SG];
        for (int q = 1; q < SG; ++q) t += part[j * SG + q];
        out[j] = t;
    }
}

template <int R, int P, int UNROLL, int SG>
inline void rt_rows8(device const uint2* W, uint row0, uint nvalid, device const uint2* X, uint k4, uint sg, uint lane,
                     threadgroup float* part, thread float* out) {
    constexpr uint lanes = 32 * SG;
    constexpr uint block = lanes * UNROLL;
    device const uint2* wr[P];
    for (int p = 0; p < P; ++p) wr[p] = W + (ulong)min(row0 + p, nvalid - 1) * k4;
    const uint nblk = k4 / block;
    uint i = sg * 32 + lane;
    float acc[P * R];
    for (int j = 0; j < P * R; ++j) acc[j] = 0.0f;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint2 wv[P][UNROLL];
        for (int p = 0; p < P; ++p)
            for (int u = 0; u < UNROLL; ++u) wv[p][u] = wr[p][i + lanes * u];
        for (int r = 0; r < R; ++r) {
            uint2 xv[UNROLL];
            for (int u = 0; u < UNROLL; ++u) xv[u] = X[(ulong)r * k4 + i + lanes * u];
            for (int p = 0; p < P; ++p)
                for (int u = 0; u < UNROLL; ++u) acc[p * R + r] += dot4(wv[p][u], xv[u]);
        }
    }
    for (i = nblk * block + sg * 32 + lane; i < k4; i += lanes) {
        uint2 wv[P];
        for (int p = 0; p < P; ++p) wv[p] = wr[p][i];
        for (int r = 0; r < R; ++r) {
            const uint2 xv = X[(ulong)r * k4 + i];
            for (int p = 0; p < P; ++p) acc[p * R + r] += dot4(wv[p], xv);
        }
    }
    for (int j = 0; j < P * R; ++j) {
        const float s = simd_sum(acc[j]);
        if (lane == 0) part[j * SG + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < P * R; ++j) {
        float t = part[j * SG];
        for (int q = 1; q < SG; ++q) t += part[j * SG + q];
        out[j] = t;
    }
}

// ---- Matmul ---------------------------------------------------------------------------------------------------------
template <int R, int P, int RB = R>
kernel void gemv_bf16_rt_u2s8_mr(device const uint4* W [[buffer(0)]],
                                 device const uint4* X [[buffer(1)]],
                                 device ushort* Y [[buffer(2)]],
                                 constant GemvParams& p [[buffer(3)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[P * R * 8];
    static_assert(R % RB == 0, "R must be a multiple of RB");
    const uint k8 = p.k / 8;
    float t[P * R];
    const uint row0 = tg * P;
    for (int rb = 0; rb < R; rb += RB) {
        float tb[P * RB];
        rt_rows16<RB, P, 2, 8>(W, row0, p.n, X + (ulong)rb * k8, k8, sg, lane, part + rb * P * 8, tb);
        for (int q = 0; q < P; ++q)
            for (int r = 0; r < RB; ++r) t[q * R + rb + r] = tb[q * RB + r];
    }
    if (sg == 0 && lane == 0) {
        for (int q = 0; q < P; ++q)
            if (row0 + q < p.n)
                for (int r = 0; r < R; ++r) Y[p.y_skip + (ulong)r * p.n + row0 + q] = to_bf16(t[q * R + r]);
    }
}

template <int R, int P, int RB = R>
kernel void gemv_bf16_rt_a8u4s8_mr(device const uint2* W [[buffer(0)]],
                                   device const uint2* X [[buffer(1)]],
                                   device ushort* Y [[buffer(2)]],
                                   constant GemvParams& p [[buffer(3)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[P * R * 8];
    static_assert(R % RB == 0, "R must be a multiple of RB");
    const uint k4 = p.k / 4;
    float t[P * R];
    const uint row0 = tg * P;
    for (int rb = 0; rb < R; rb += RB) {
        float tb[P * RB];
        rt_rows8<RB, P, 4, 8>(W, row0, p.n, X + (ulong)rb * k4, k4, sg, lane, part + rb * P * 8, tb);
        for (int q = 0; q < P; ++q)
            for (int r = 0; r < RB; ++r) t[q * R + rb + r] = tb[q * RB + r];
    }
    if (sg == 0 && lane == 0) {
        for (int q = 0; q < P; ++q)
            if (row0 + q < p.n)
                for (int r = 0; r < R; ++r) Y[p.y_skip + (ulong)r * p.n + row0 + q] = to_bf16(t[q * R + r]);
    }
}

// ---- MatmulPair (the first ceil(n0/P) threadgroups work on W0, the next ceil(n1/P) on W1) -----------------------------
template <int R, int P, int RB = R>
kernel void gemv_pair_bf16_rt_u2s8_mr(device const uint4* W0 [[buffer(0)]],
                                      device const uint4* W1 [[buffer(1)]],
                                      device const uint4* X [[buffer(2)]],
                                      device ushort* Y0 [[buffer(3)]],
                                      device ushort* Y1 [[buffer(4)]],
                                      constant PairParams& p [[buffer(5)]],
                                      uint tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[P * R * 8];
    static_assert(R % RB == 0, "R must be a multiple of RB");
    const uint groups0 = (p.n0 + P - 1) / P;
    const bool second = tg >= groups0;
    device const uint4* W = second ? W1 : W0;
    device ushort* Y = (second ? Y1 : Y0) + (second ? p.y1_skip : p.y0_skip);
    const uint n = second ? p.n1 : p.n0;
    const uint row0 = (second ? tg - groups0 : tg) * P;
    const uint k8 = p.k / 8;
    float t[P * R];
    for (int rb = 0; rb < R; rb += RB) {
        float tb[P * RB];
        rt_rows16<RB, P, 2, 8>(W, row0, n, X + (ulong)rb * k8, k8, sg, lane, part + rb * P * 8, tb);
        for (int q = 0; q < P; ++q)
            for (int r = 0; r < RB; ++r) t[q * R + rb + r] = tb[q * RB + r];
    }
    if (sg == 0 && lane == 0) {
        for (int q = 0; q < P; ++q)
            if (row0 + q < n)
                for (int r = 0; r < R; ++r) Y[(ulong)r * n + row0 + q] = to_bf16(t[q * R + r]);
    }
}

template <int R, int P, int RB = R>
kernel void gemv_pair_bf16_rt_a8u4s8_mr(device const uint2* W0 [[buffer(0)]],
                                        device const uint2* W1 [[buffer(1)]],
                                        device const uint2* X [[buffer(2)]],
                                        device ushort* Y0 [[buffer(3)]],
                                        device ushort* Y1 [[buffer(4)]],
                                        constant PairParams& p [[buffer(5)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[P * R * 8];
    static_assert(R % RB == 0, "R must be a multiple of RB");
    const uint groups0 = (p.n0 + P - 1) / P;
    const bool second = tg >= groups0;
    device const uint2* W = second ? W1 : W0;
    device ushort* Y = (second ? Y1 : Y0) + (second ? p.y1_skip : p.y0_skip);
    const uint n = second ? p.n1 : p.n0;
    const uint row0 = (second ? tg - groups0 : tg) * P;
    const uint k4 = p.k / 4;
    float t[P * R];
    for (int rb = 0; rb < R; rb += RB) {
        float tb[P * RB];
        rt_rows8<RB, P, 4, 8>(W, row0, n, X + (ulong)rb * k4, k4, sg, lane, part + rb * P * 8, tb);
        for (int q = 0; q < P; ++q)
            for (int r = 0; r < RB; ++r) t[q * R + rb + r] = tb[q * RB + r];
    }
    if (sg == 0 && lane == 0) {
        for (int q = 0; q < P; ++q)
            if (row0 + q < n)
                for (int r = 0; r < R; ++r) Y[(ulong)r * n + row0 + q] = to_bf16(t[q * R + r]);
    }
}

// ---- Logits, f32 out (one simdgroup per P rows, 4 simdgroups per threadgroup, UNROLL 4, simd_sum only) ----------------
template <int R, int P, int RB = R>
kernel void gemv_f32o_v_mr(device const uint4* W [[buffer(0)]],
                           device const uint4* X [[buffer(1)]],
                           device float* Y [[buffer(2)]],
                           constant GemvParams& p [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    static_assert(R % RB == 0, "R must be a multiple of RB");
    const uint k8 = p.k / 8;
    const uint row0 = (tg * 4 + sg) * P;
    device const uint4* wr[P];
    for (int q = 0; q < P; ++q) wr[q] = W + (ulong)min(row0 + q, p.n - 1) * k8;  // rows past N read a valid row, not stored
    for (int rb = 0; rb < R; rb += RB) {
        device const uint4* Xb = X + (ulong)rb * k8;
        float acc[P * RB];
        for (int j = 0; j < P * RB; ++j) acc[j] = 0.0f;
        const uint block = 32 * 4;
        const uint nblk = k8 / block;
        uint i = lane;
        for (uint b = 0; b < nblk; ++b, i += block) {
            uint4 wv[P][4];
            for (int q = 0; q < P; ++q)
                for (int u = 0; u < 4; ++u) wv[q][u] = wr[q][i + 32 * u];
            for (int r = 0; r < RB; ++r) {
                uint4 xv[4];
                for (int u = 0; u < 4; ++u) xv[u] = Xb[(ulong)r * k8 + i + 32 * u];
                for (int q = 0; q < P; ++q)
                    for (int u = 0; u < 4; ++u) acc[q * RB + r] += dot8(wv[q][u], xv[u]);
            }
        }
        for (i = nblk * block + lane; i < k8; i += 32) {
            uint4 wv[P];
            for (int q = 0; q < P; ++q) wv[q] = wr[q][i];
            for (int r = 0; r < RB; ++r) {
                const uint4 xv = Xb[(ulong)r * k8 + i];
                for (int q = 0; q < P; ++q) acc[q * RB + r] += dot8(wv[q], xv);
            }
        }
        for (int q = 0; q < P; ++q) {
            for (int r = 0; r < RB; ++r) {
                const float s = simd_sum(acc[q * RB + r]);
                if (lane == 0 && row0 + q < p.n) Y[p.y_skip + (ulong)(rb + r) * p.n + row0 + q] = s;
            }
        }
    }
}

#define INST_RP(KERNEL, R, P, ...)                                                                                    \
    template [[host_name(#KERNEL "_mr" #R "x" #P)]] kernel void KERNEL##_mr<R, P>(__VA_ARGS__);

#define INST_RS(KERNEL, R, P, RB, ...)                                                                                \
    template [[host_name(#KERNEL "_mr" #R "x" #P "s" #RB)]] kernel void KERNEL##_mr<R, P, RB>(__VA_ARGS__);

#define INST_SPLIT(KERNEL, ...)                                                                                       \
    INST_RS(KERNEL, 6, 1, 3, __VA_ARGS__) INST_RS(KERNEL, 6, 1, 2, __VA_ARGS__) INST_RS(KERNEL, 8, 1, 4, __VA_ARGS__) \
    INST_RS(KERNEL, 8, 1, 2, __VA_ARGS__) INST_RS(KERNEL, 8, 2, 4, __VA_ARGS__) INST_RS(KERNEL, 16, 1, 8, __VA_ARGS__) \
    INST_RS(KERNEL, 16, 1, 4, __VA_ARGS__)

#define INST_R(KERNEL, R, ...) \
    INST_RP(KERNEL, R, 1, __VA_ARGS__) INST_RP(KERNEL, R, 2, __VA_ARGS__) INST_RP(KERNEL, R, 4, __VA_ARGS__)

#define INST_MR_ALL(KERNEL, ...)                                                                                      \
    INST_R(KERNEL, 1, __VA_ARGS__) INST_R(KERNEL, 2, __VA_ARGS__) INST_R(KERNEL, 3, __VA_ARGS__)                      \
    INST_R(KERNEL, 4, __VA_ARGS__) INST_R(KERNEL, 5, __VA_ARGS__) INST_R(KERNEL, 6, __VA_ARGS__)                      \
    INST_R(KERNEL, 7, __VA_ARGS__) INST_R(KERNEL, 8, __VA_ARGS__) INST_R(KERNEL, 16, __VA_ARGS__)

INST_MR_ALL(gemv_bf16_rt_u2s8, device const uint4*, device const uint4*, device ushort*, constant GemvParams&, uint, uint, uint)
INST_SPLIT(gemv_bf16_rt_u2s8, device const uint4*, device const uint4*, device ushort*, constant GemvParams&, uint, uint, uint)
INST_MR_ALL(gemv_bf16_rt_a8u4s8, device const uint2*, device const uint2*, device ushort*, constant GemvParams&, uint, uint, uint)
INST_SPLIT(gemv_bf16_rt_a8u4s8, device const uint2*, device const uint2*, device ushort*, constant GemvParams&, uint, uint, uint)
INST_MR_ALL(gemv_pair_bf16_rt_u2s8, device const uint4*, device const uint4*, device const uint4*, device ushort*, device ushort*, constant PairParams&, uint, uint, uint)
INST_SPLIT(gemv_pair_bf16_rt_u2s8, device const uint4*, device const uint4*, device const uint4*, device ushort*, device ushort*, constant PairParams&, uint, uint, uint)
INST_MR_ALL(gemv_pair_bf16_rt_a8u4s8, device const uint2*, device const uint2*, device const uint2*, device ushort*, device ushort*, constant PairParams&, uint, uint, uint)
INST_SPLIT(gemv_pair_bf16_rt_a8u4s8, device const uint2*, device const uint2*, device const uint2*, device ushort*, device ushort*, constant PairParams&, uint, uint, uint)
INST_MR_ALL(gemv_f32o_v, device const uint4*, device const uint4*, device float*, constant GemvParams&, uint, uint, uint)
INST_SPLIT(gemv_f32o_v, device const uint4*, device const uint4*, device float*, constant GemvParams&, uint, uint, uint)
