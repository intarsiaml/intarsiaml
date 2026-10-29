/* W4A8 helpers (lane W4A8): MXFP4 weights x dynamic per-token fp8 (e4m3) activations. Header-only, shared by
 * kernels/hip/mxfp4_a8.hip (decode: M = 1..16 rows) and any other .hip translation unit that wants the same activation
 * quantisation (the prefill GEMM includes this header). No namespace and no includes of .hip files; every name starts
 * with mx_a8 / kMxA8, so it cannot clash with mxfp4.hip's or ik_dq_gemm.h's helpers.
 *
 * Math (the CPU reference implements exactly this):
 *   amax[m]  = max_k |x[m, k]|                                  (f32)
 *   xs[m]    = max(amax / 448, 1 / (448 * 512))                  (f32 division, f32 constant)
 *   xq[m, k] = e4m3fn(clamp(x[m, k] / xs[m], -448, 448))         (IEEE f32 division, round to nearest even, saturating)
 *   out[m, n] = round(warp_sum(sum_g fma(t_g, 2^(sw[n, g] - 127), ..)) * xs[m]),  t_g = sum_{e < 32} e2m1(w) * xq
 * Every product e2m1 * e4m3 is exact in f32 (e2m1 has 2 significant bits, e4m3 4), so the only roundings are the f32 sums.
 *
 * Scratch (one buffer per call, 16-byte aligned base; the launcher owner sizes it with mx_a8_scratch_bytes):
 *   xq  u8 [M, K]   at byte 0                                  (row m = the K e4m3 bytes of row m)
 *   xs  f32 [M]     at byte mx_a8_xs_offset(M, K) = (M * K + 15) & ~15
 */
#ifndef IK_MX_A8_H
#define IK_MX_A8_H

#include "ik_quant.h"  // ik_hip.h (IK_DEVI, IK_WARP, ik_bf16, ik_unpack8, ik_block_max, ik_launch_status), ik_f_to_e4m3

constexpr float kMxA8Max = 448.f;                         // largest finite e4m3fn
constexpr float kMxA8MinScale = 1.0f / (448.0f * 512.0f);  // f32 constant: lower bound of the per-token scale

/* e8m0 byte -> 2^(b - 127); 0 -> 2^-127 (f32 subnormal), 255 -> NaN (same as mxfp4.hip's mx_e8m0) */
IK_DEVI float mx_a8_e8m0(uint32_t b) {
    return b == 0u ? __uint_as_float(0x00400000u) : (b == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(b << 23));
}

/* ---- activations: 8 bf16 -> 8 e4m3 bytes ------------------------------------------------------------------------------- */

/* x / s, clamped to +-448, RNE e4m3: the sequence of f8r_quant8 (ik_f8_rows.h), byte e of the result = element e */
IK_DEVI uint2 mx_a8_quant8(const uint4& v, float s) {
    float f[8];
    ik_unpack8(v, f);
#if __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32)
    float y[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) y[e] = fminf(fmaxf(__fdiv_rn(f[e], s), -kMxA8Max), kMxA8Max);
    int c0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[0], y[1], 0, false);
    c0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[2], y[3], c0, true);
    int c1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[4], y[5], 0, false);
    c1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[6], y[7], c1, true);
    return make_uint2((uint32_t)c0, (uint32_t)c1);
#else
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        lo |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[e], s)) << (8 * e);
        hi |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 + e], s)) << (8 * e);
    }
    return make_uint2(lo, hi);
#endif
}

/* The two per-row steps of the activation quantisation, shared by the pre-pass kernel below and by the fused decode kernel
 * (mxfp4_a8_rows_fused_kernel, which runs them per row in its own block into LDS): both therefore produce bit-identical xs / xq.
 * Both are block-cooperative: every thread of a 256-thread block calls them with the same arguments.
 * mx_a8_row_scale: xs of one row = max(amax / 448, 1 / (448 * 512)); the amax is a max over the exact |x|, so it does not
 * depend on the order of the reduction. `red` is float[256 / IK_WARP] of block shared memory; the result is block-uniform.
 * mx_a8_row_quant: the K e4m3 bytes of the row (C = K / 8 chunks), qp may point to global or shared memory. */
constexpr int kMxA8QuantThreads = 256;
IK_DEVI float mx_a8_row_scale(const uint4* xp, int C, float* red) {
    float amax = 0.f;
    for (int c = threadIdx.x; c < C; c += kMxA8QuantThreads) {
        float f[8];
        ik_unpack8(xp[c], f);
#pragma unroll
        for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
    }
    amax = ik_block_max(amax, red);
    return fmaxf(__fdiv_rn(amax, kMxA8Max), kMxA8MinScale);
}
IK_DEVI void mx_a8_row_quant(const uint4* xp, uint2* qp, int C, float s) {
    for (int c = threadIdx.x; c < C; c += kMxA8QuantThreads) qp[c] = mx_a8_quant8(xp[c], s);
}

/* One block (256 threads) per activation row m: xs[m] and the K e4m3 bytes xq[m * K ..]. K % 8 == 0, x and xq 16-byte aligned
 * (K % 32 == 0 in every caller). The block max is order independent, so the scale does not depend on the block shape. */
static __global__ void __launch_bounds__(256)
mx_a8_quant_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ xq, float* __restrict__ xs, int K) {
    __shared__ float red[256 / IK_WARP];
    const int64_t row = blockIdx.x;
    const uint4* xp = reinterpret_cast<const uint4*>(x + row * K);
    uint2* qp = reinterpret_cast<uint2*>(xq + row * K);
    const int C = K >> 3;  // 8-element chunks
    const float s = mx_a8_row_scale(xp, C, red);
    if (threadIdx.x == 0) xs[row] = s;
    mx_a8_row_quant(xp, qp, C, s);
}

/* Same quantisation (same helpers, so the same xs / e4m3 bytes), written in the fragment order of the W4A8 prefill GEMM's direct
 * kernel (mxfp4_a8_gemm.hip, lane PF27) instead of row-major: for every 16-token tile t and 32-k group g one 512-byte block, in
 * which lane l (0..31) of a wave finds its 16 bytes at l * 16 = token 16 t + (l & 15), k = 32 g + 16 (l >> 4) .. + 16:
 *   xt[(t * KG + g) * 512 + ((m & 15) + 16 * ((k & 31) >> 4)) * 16 + (k & 15)] = xq[m, k],   KG = 2 * ceil(K / 64)
 * which is the uint4 a wave's LDS read of a8_gemm_kernel gives that lane, with KG = the group count of K padded to a multiple of 64
 * (a K % 64 == 32 tail adds one zero group per tile). One block per row of the 16-padded M (gridDim.x = Mp); rows M .. Mp - 1 are
 * written as zeros (no xs). K % 32 == 0. */
static __global__ void __launch_bounds__(256)
mx_a8_quant_tiled_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ xt, float* __restrict__ xs, int M, int K) {
    __shared__ float red[256 / IK_WARP];
    const int64_t row = blockIdx.x;
    const int C = K >> 3, KGP = ((K + 63) >> 6) * 2;  // 8-element chunks; 32-k groups per tile, K padded to 64
    const int CP = KGP * 4;                            // chunks including the pad group
    uint8_t* tile = xt + (row >> 4) * (int64_t)KGP * 512 + (row & 15) * 16;
    // chunk c (k = 8 c): group c >> 2, half (c >> 1) & 1 (lanes 16..31), byte (c & 1) * 8 of the lane's 16
    for (int c = C + threadIdx.x; c < CP; c += kMxA8QuantThreads)  // the K % 64 == 32 pad group: zeros
        *reinterpret_cast<uint2*>(tile + (c >> 2) * 512 + ((c >> 1) & 1) * 256 + (c & 1) * 8) = make_uint2(0u, 0u);
    if (row >= M) {  // block-uniform
        for (int c = threadIdx.x; c < C; c += kMxA8QuantThreads)
            *reinterpret_cast<uint2*>(tile + (c >> 2) * 512 + ((c >> 1) & 1) * 256 + (c & 1) * 8) = make_uint2(0u, 0u);
        return;
    }
    const uint4* xp = reinterpret_cast<const uint4*>(x + row * K);
    const float s = mx_a8_row_scale(xp, C, red);
    if (threadIdx.x == 0) xs[row] = s;
    for (int c = threadIdx.x; c < C; c += kMxA8QuantThreads)
        *reinterpret_cast<uint2*>(tile + (c >> 2) * 512 + ((c >> 1) & 1) * 256 + (c & 1) * 8) = mx_a8_quant8(xp[c], s);
}

/* ---- scratch layout / launch (host) ------------------------------------------------------------------------------------ */

static inline size_t mx_a8_al16(size_t b) { return (b + 15) & ~(size_t)15; }
/* byte offset of xs[M] in the scratch (xq occupies [0, M * K)) */
static inline size_t mx_a8_xs_offset(int64_t M, int64_t K) { return mx_a8_al16((size_t)M * (size_t)K); }
/* bytes the scratch must hold: aligned xq + aligned M floats + 16 slack */
static inline size_t mx_a8_scratch_bytes(int64_t M, int64_t K) {
    return mx_a8_al16((size_t)M * (size_t)K) + mx_a8_al16((size_t)M * 4) + 16;
}

/* Quantise x [M, K] bf16 into the scratch (layout above). 16-byte aligned x and scratch, K % 8 == 0 (else IK_ERR_SHAPE /
 * IK_ERR_INVALID_ARG). M == 0 launches nothing. */
static inline int mx_a8_quant_launch(const ik_bf16* x, void* scratch, int64_t M, int64_t K, hipStream_t s) {
    if (M < 0 || K < 0 || (K & 7) != 0 || M > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (!x || !scratch || (reinterpret_cast<uintptr_t>(scratch) & 15u) || (reinterpret_cast<uintptr_t>(x) & 15u))
        return IK_ERR_INVALID_ARG;
    if (M == 0 || K == 0) return IK_OK;
    uint8_t* xq = static_cast<uint8_t*>(scratch);
    float* xs = reinterpret_cast<float*>(xq + mx_a8_xs_offset(M, K));
    mx_a8_quant_kernel<<<(unsigned)M, 256, 0, s>>>(x, xq, xs, (int)K);
    return ik_launch_status();
}

/* ---- weights: e2m1 -> e4m3 --------------------------------------------------------------------------------------------- */

/* 8 e2m1 codes (element j = nibble j of w) -> 8 e4m3 bytes in element order: .x = elements 0..3, .y = elements 4..7 (byte e
 * of a dword = element e of that half). Magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} = e4m3 {00, 30, 38, 3C, 40, 44, 48, 4C}; the
 * sign (bit 3 of the code) becomes bit 7 of the byte. */
IK_DEVI uint2 mx_a8_e2m1x8_to_e4m3(uint32_t w) {
    constexpr uint32_t P0 = 0x3C383000u, P1 = 0x4C484440u;  // magnitude index 0 1 2 3 | 4 5 6 7
    const uint32_t ie = w & 0x07070707u, io = (w >> 4) & 0x07070707u;
    const uint32_t ev = __builtin_amdgcn_perm(P1, P0, ie) | ((w << 4) & 0x80808080u);  // elements 0, 2, 4, 6
    const uint32_t od = __builtin_amdgcn_perm(P1, P0, io) | (w & 0x80808080u);         // elements 1, 3, 5, 7
    // interleave: bytes 0-3 of the selector = ev, 4-7 = od
    return make_uint2(__builtin_amdgcn_perm(od, ev, 0x05010400u), __builtin_amdgcn_perm(od, ev, 0x07030602u));
}

/* ---- 32-element dot: t = sum_e w[e] * xq[e] ---------------------------------------------------------------------------- */

/* wd: the 8 dwords mx_a8_e2m1x8_to_e4m3 made from a group's four weight words (.x then .y of each), xa / xc: the group's 32
 * xq bytes (xa = elements 0..15, xc = 16..31). Fixed accumulation order, independent of everything else. */
#if __has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8)
#if defined(__HIP_DEVICE_COMPILE__)  // device pass only: the host pass does not see the gfx12 builtin and would report the fallback
#warning "mx_a8: dot4 fp8 path"
#endif
IK_DEVI float mx_a8_dot32(const uint32_t* wd, const uint4& xa, const uint4& xc) {
    float t = 0.f;
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[0], xa.x, t);
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[1], xa.y, t);
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[2], xa.z, t);
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[3], xa.w, t);
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[4], xc.x, t);
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[5], xc.y, t);
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[6], xc.z, t);
    t = __builtin_amdgcn_dot4_f32_fp8_fp8(wd[7], xc.w, t);
    return t;
}
#else
#if defined(__HIP_DEVICE_COMPILE__)
#warning "mx_a8: fp32 fallback path"
#endif
/* 4 e4m3 bytes -> 4 floats (exact) */
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
typedef float mx_a8_v2f __attribute__((ext_vector_type(2)));
IK_DEVI void mx_a8_e4m3x4_to_f32(uint32_t w, float* f) {
    const mx_a8_v2f lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false);
    const mx_a8_v2f hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true);
    f[0] = lo[0];
    f[1] = lo[1];
    f[2] = hi[0];
    f[3] = hi[1];
}
#else
IK_DEVI void mx_a8_e4m3x4_to_f32(uint32_t w, float* f) {
#pragma unroll
    for (int j = 0; j < 4; ++j) f[j] = ik_e4m3_to_f((w >> (8 * j)) & 0xffu);
}
#endif
IK_DEVI float mx_a8_dot32(const uint32_t* wd, const uint4& xa, const uint4& xc) {
    const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w};
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float wf[4], xf[4];
        mx_a8_e4m3x4_to_f32(wd[i], wf);
        mx_a8_e4m3x4_to_f32(xw[i], xf);
#pragma unroll
        for (int j = 0; j < 4; ++j) t = fmaf(wf[j], xf[j], t);
    }
    return t;
}
#endif

#endif /* IK_MX_A8_H */
