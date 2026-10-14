// BF16 GEMV for batch-1 decode: y[N] = W[N,K] . x[K]   (Matmul with M = 1; W row-major [N,K], i.e. y = x W^T).
// bf16 in, f32 accumulate, bf16 out (round to nearest even). Spike kernels (docs/research/macos-metal-backend-scope.md).
//
// Layout contract: every pointer is bound at a byte offset the host aligned to 16 bytes
// (vector kernels, which also need K % 8 == 0; only y may carry a `y_skip`) or to 4 bytes (`gemv_bf16_scalar`, which takes the remaining 0 or 1
// ushort of misalignment of W, x, y as `*_skip`). The host picks the kernel (crates/intarsia-backend-metal/src/kernels.rs).
//
// Vector kernels: one simdgroup (32 lanes) handles ROWS output rows. Per step a lane loads UNROLL consecutive-in-lane
// 16-byte chunks (8 bf16) of x once and of each of its ROWS weight rows; lane l reads chunk (step*UNROLL + u)*32 + l,
// so a simdgroup's load of one u is a contiguous 512 B. SG simdgroups per threadgroup. Reduction is simd_sum.

#include <metal_stdlib>
using namespace metal;

struct GemvParams {
    uint k;
    uint n;
    uint w_skip;  // scalar kernel only: leading ushort elements to skip in W / x / y
    uint x_skip;
    uint y_skip;
    uint pad0;
    uint pad1;
    uint pad2;
};

// bf16 pair packed in a uint -> two f32 (bf16 is the top half of an f32)
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

inline float dot4(uint2 w, uint2 x) {
    float a = lo_f(w.x) * lo_f(x.x);
    a = fma(hi_f(w.x), hi_f(x.x), a);
    a = fma(lo_f(w.y), lo_f(x.y), a);
    a = fma(hi_f(w.y), hi_f(x.y), a);
    return a;
}

inline ushort to_bf16(float v) { return as_type<ushort>(bfloat(v)); }

template <int ROWS, int UNROLL, int SG>
kernel void gemv_bf16_t(device const uint4* W [[buffer(0)]],
                        device const uint4* X [[buffer(1)]],
                        device ushort* Y [[buffer(2)]],
                        constant GemvParams& p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint k8 = p.k / 8;
    const uint row0 = (tg * SG + sg) * ROWS;
    device const uint4* wr[ROWS];
    for (int r = 0; r < ROWS; ++r) {
        const uint row = min(row0 + r, p.n - 1);  // rows past N read a valid row and are not stored
        wr[r] = W + (ulong)row * k8;
    }
    float acc[ROWS];
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    const uint block = 32 * UNROLL;
    const uint nblk = k8 / block;
    uint i = lane;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 xv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) xv[u] = X[i + 32 * u];
        for (int r = 0; r < ROWS; ++r) {
            uint4 wv[UNROLL];
            for (int u = 0; u < UNROLL; ++u) wv[u] = wr[r][i + 32 * u];
            for (int u = 0; u < UNROLL; ++u) acc[r] += dot8(wv[u], xv[u]);
        }
    }
    for (i = nblk * block + lane; i < k8; i += 32) {
        const uint4 xv = X[i];
        for (int r = 0; r < ROWS; ++r) acc[r] += dot8(wr[r][i], xv);
    }
    for (int r = 0; r < ROWS; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < p.n) Y[p.y_skip + row0 + r] = to_bf16(s);
    }
}

#define INST(ROWS, UNROLL, SG)                                                                                        \
    template [[host_name("gemv_bf16_r" #ROWS "u" #UNROLL "s" #SG)]] kernel void gemv_bf16_t<ROWS, UNROLL, SG>(      \
        device const uint4*, device const uint4*, device ushort*, constant GemvParams&, uint, uint, uint);

INST(1, 1, 8)
INST(1, 2, 8)
INST(1, 4, 4)
INST(1, 4, 8)
INST(1, 4, 16)
INST(1, 8, 8)
INST(2, 1, 4)
INST(2, 2, 4)
INST(2, 4, 4)
INST(4, 1, 4)
INST(4, 2, 4)
INST(8, 1, 2)
INST(8, 2, 2)

// 8-byte variant for operands that are only 8-byte aligned (safetensors shards whose header length is 8 mod 16 put
// every tensor at 8 mod 16) and K % 4 == 0: one simdgroup per row, UNROLL 8-byte loads per lane per step.
template <int UNROLL, int SG>
kernel void gemv_bf16_a8_t(device const uint2* W [[buffer(0)]],
                           device const uint2* X [[buffer(1)]],
                           device ushort* Y [[buffer(2)]],
                           constant GemvParams& p [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint k4 = p.k / 4;
    const uint row = tg * SG + sg;
    device const uint2* wr = W + (ulong)min(row, p.n - 1) * k4;
    float acc = 0.0f;
    const uint block = 32 * UNROLL;
    const uint nblk = k4 / block;
    uint i = lane;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint2 xv[UNROLL];
        uint2 wv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) xv[u] = X[i + 32 * u];
        for (int u = 0; u < UNROLL; ++u) wv[u] = wr[i + 32 * u];
        for (int u = 0; u < UNROLL; ++u) acc += dot4(wv[u], xv[u]);
    }
    for (i = nblk * block + lane; i < k4; i += 32) acc += dot4(wr[i], X[i]);
    const float s = simd_sum(acc);
    if (lane == 0 && row < p.n) Y[p.y_skip + row] = to_bf16(s);
}

#define INST_A8(UNROLL, SG)                                                                                           \
    template [[host_name("gemv_bf16_a8u" #UNROLL "s" #SG)]] kernel void gemv_bf16_a8_t<UNROLL, SG>(                 \
        device const uint2*, device const uint2*, device ushort*, constant GemvParams&, uint, uint, uint);

INST_A8(4, 4)
INST_A8(8, 4)
INST_A8(8, 8)

// Any K, any 2-byte alignment: one simdgroup per row, 2-byte loads. Correctness fallback, not a fast path.
kernel void gemv_bf16_scalar(device const ushort* W [[buffer(0)]],
                             device const ushort* X [[buffer(1)]],
                             device ushort* Y [[buffer(2)]],
                             constant GemvParams& p [[buffer(3)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8 + sg;
    const uint rr = min(row, p.n - 1);
    device const ushort* w = W + p.w_skip + (ulong)rr * p.k;
    device const ushort* x = X + p.x_skip;
    float acc = 0.0f;
    for (uint i = lane; i < p.k; i += 32) acc = fma(as_type<float>((uint)w[i] << 16), as_type<float>((uint)x[i] << 16), acc);
    const float s = simd_sum(acc);
    if (lane == 0 && row < p.n) Y[p.y_skip + row] = to_bf16(s);
}

// ---- MatmulPair: two GEMVs of one x in one dispatch -------------------------------------------------------------------
// y0[N0] = W0[N0,K] . x and y1[N1] = W1[N1,K] . x. Threadgroups [0, ceil(N0/SG)) work on the first matrix, the rest on
// the second; each row is computed exactly as `gemv_bf16_t<1, 4, 4>` does (same lane/step order, same simd_sum), so
// both outputs are bitwise those of two separate GEMVs. Vector loads only: K % 8 == 0 and 16-byte aligned W0, W1, x.
// The host (crates/intarsia-backend-metal/src/fused.rs) falls back to two GEMV dispatches for anything else.
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

kernel void gemv_pair_bf16(device const uint4* W0 [[buffer(0)]],
                           device const uint4* W1 [[buffer(1)]],
                           device const uint4* X [[buffer(2)]],
                           device ushort* Y0 [[buffer(3)]],
                           device ushort* Y1 [[buffer(4)]],
                           constant PairParams& p [[buffer(5)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr int UNROLL = 4;
    constexpr uint SG = 4;
    const uint groups0 = (p.n0 + SG - 1) / SG;
    const bool second = tg >= groups0;
    device const uint4* W = second ? W1 : W0;
    device ushort* Y = (second ? Y1 : Y0) + (second ? p.y1_skip : p.y0_skip);
    const uint n = second ? p.n1 : p.n0;
    const uint k8 = p.k / 8;
    const uint row = ((second ? tg - groups0 : tg) * SG + sg);
    device const uint4* wr = W + (ulong)min(row, n - 1) * k8;
    float acc = 0.0f;
    const uint block = 32 * UNROLL;
    const uint nblk = k8 / block;
    uint i = lane;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 xv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) xv[u] = X[i + 32 * u];
        uint4 wv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) wv[u] = wr[i + 32 * u];
        for (int u = 0; u < UNROLL; ++u) acc += dot8(wv[u], xv[u]);
    }
    for (i = nblk * block + lane; i < k8; i += 32) {
        const uint4 xv = X[i];
        acc += dot8(wr[i], xv);
    }
    const float s = simd_sum(acc);
    if (lane == 0 && row < n) Y[row] = to_bf16(s);
}
