// matmul_mxfp4: out[M,N] = q(x)[M,K] @ W^T with W[n,k] = e2m1(code) * 2^(scale[n,k/32] - 127) (OCP MXFP4, AMD Quark).
// Port of kernels/hip/mxfp4.hip (same layout, same QDQ math); launcher crates/intarsia-backend-metal/src/mxfp4.rs.
//   w     u8 [N, K/2]   two codes per byte, element 2i in the low nibble (bound 4-byte aligned, no skip)
//   scale u8 [N, K/32]  e8m0, plain row-major
//   act   f32 [1]       0: q(x) = x (W4A16); 1: q(x) = dynamic MXFP4 QDQ of x per 32-element group along K (W4A4, scale
//                       rule "even", crates/intarsia-core/src/quant/mxfp4.rs)
// Two dispatches per launch:
//   mxfp4_xprep_bf16       xs[M, K] (bf16 scratch) = act ? QDQ(x) : x, one thread per 32-element group. The QDQ result
//                          (e2m1 * 2^e) is exact in bf16.
//   mxfp4_gemv_bf16/_f32   one simdgroup per output row n, a lane per 32-element group (16 weight bytes + 1 scale byte +
//                          32 activations per lane-iteration), MB activation rows per threadgroup row of the grid.
//                          _m1 serves M == 1, _m8 any M (8 rows per grid row). Fixed reduction order: deterministic.
// Kernel arguments: OpP (common.h), a = M, b = N, c = K; xprep s0 = x skip; gemv s0 = scale skip, s1 = out skip.

#include "common.h"

constant float kFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// e8m0 byte -> 2^(b - 127); 0 -> 2^-127 (f32 subnormal), 255 -> NaN
inline float mx_e8m0(uint b) {
    return b == 0u ? as_type<float>(0x00400000u) : (b == 255u ? as_type<float>(0x7FC00000u) : as_type<float>(b << 23));
}

// Quark "even" shared scale of a group with largest magnitude amax (>= 0): biased e8m0 byte
inline uint mx_scale_even(float amax) {
    const uint ab = as_type<uint>(amax);
    if ((ab & 0x7FFFFFFFu) > 0x7F800000u) return 255u;  // NaN
    const uint r = (ab + (1u << 21)) & 0x7F800000u;
    const int biased = (int)(r >> 23);
    if (biased == 0) return 0u;
    if (biased == 255) return 254u;
    int e = biased - 129;
    e = e < -127 ? -127 : (e > 127 ? 127 : e);
    return (uint)(e + 127);
}

// round to the nearest e2m1 value, ties to the even code, saturating at +-6 (crates/intarsia-core quant::nvfp4::f32_to_fp4)
inline float mx_fp4_round(float v) {
    const float a = fabs(v);
    float r;
    if (a <= 0.25f) r = 0.f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.f;
    else if (a < 3.5f) r = 3.f;
    else if (a <= 5.f) r = 4.f;
    else r = 6.f;
    return copysign(r, v);
}

// QDQ one group of 32 activations in place
inline void mx_qdq32(thread float* f) {
    float amax = 0.f;
    for (int e = 0; e < 32; ++e) amax = fmax(amax, fabs(f[e]));
    const uint sb = mx_scale_even(amax);
    const float s = mx_e8m0(sb), inv = mx_e8m0(254u - sb);  // 1 / 2^(sb-127) = 2^(127-sb), exact
    for (int e = 0; e < 32; ++e) f[e] = mx_fp4_round(f[e] * inv) * s;
}

// ---- activation pre-pass ------------------------------------------------------------------------------------------

kernel void mxfp4_xprep_bf16(device const ushort* x [[buffer(0)]], device const float* act [[buffer(1)]],
                             device ushort* xs [[buffer(2)]], constant OpP& p [[buffer(3)]],
                             uint g [[thread_position_in_grid]]) {
    const uint groups = p.a * (p.c / 32);
    if (g >= groups) return;
    device const ushort* xp = x + p.s0 + (ulong)g * 32;
    device ushort* dp = xs + (ulong)g * 32;
    if (act[0] == 0.f) {
        for (int e = 0; e < 32; ++e) dp[e] = xp[e];
        return;
    }
    float f[32];
    for (int e = 0; e < 32; ++e) f[e] = bf2f(xp[e]);
    mx_qdq32(f);
    for (int e = 0; e < 32; ++e) dp[e] = f2bf(f[e]);  // exact: e2m1 * 2^e fits bf16
}

// ---- GEMV ---------------------------------------------------------------------------------------------------------

constant uint MX_SG = 4;  // simdgroups (= output rows) per threadgroup

template <int MB, typename OutT>
inline void mxfp4_gemv_impl(device const ushort* xs, device const uchar* w, device const uchar* sc, device OutT* out,
                            constant OpP& p, uint2 tg, uint sg, uint lane) {
    const ulong n = (ulong)tg.x * MX_SG + sg;
    if (n >= p.b) return;  // whole simdgroup exits together
    const uint M = p.a, K = p.c;
    const uint m0 = tg.y * MB;
    const uint ng = K / 32;
    device const packed_uint4* wr = (device const packed_uint4*)(w + n * (K / 2));
    device const uchar* sr = sc + p.s0 + n * ng;
    float acc[MB];
    for (int m = 0; m < MB; ++m) acc[m] = 0.f;
    for (uint u = lane; u < ng; u += 32) {
        const uint4 wv = uint4(wr[u]);
        const float s = mx_e8m0(sr[u]);
        float wf[32];
        for (int q = 0; q < 4; ++q) {
            const uint v = wv[q];
            for (int b = 0; b < 8; ++b) wf[q * 8 + b] = kFp4[(v >> (4 * b)) & 0xFu];
        }
        for (int m = 0; m < MB; ++m) {
            if (m0 + m >= M) break;
            device const ushort* xp = xs + (ulong)(m0 + m) * K + (ulong)u * 32;
            float t = 0.f;
            for (int e = 0; e < 32; ++e) t = fma(wf[e], bf2f(xp[e]), t);
            acc[m] = fma(t, s, acc[m]);
        }
    }
    for (int m = 0; m < MB; ++m) {
        if (m0 + m >= M) break;
        const float t = simd_sum(acc[m]);
        if (lane == 0) store_out(out + p.s1, (ulong)(m0 + m) * p.b + n, t);
    }
}

#define MX_GEMV(NAME, MB, OUT)                                                                                      \
    kernel void NAME(device const ushort* xs [[buffer(0)]], device const uchar* w [[buffer(1)]],                    \
                     device const uchar* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        mxfp4_gemv_impl<MB, OUT>(xs, w, sc, out, p, tg, sg, lane);                                                  \
    }

MX_GEMV(mxfp4_gemv_bf16_m1, 1, ushort)
MX_GEMV(mxfp4_gemv_bf16_m8, 8, ushort)
MX_GEMV(mxfp4_gemv_f32_m1, 1, float)
MX_GEMV(mxfp4_gemv_f32_m8, 8, float)
