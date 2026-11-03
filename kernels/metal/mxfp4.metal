// matmul_mxfp4: out[M,N] = q(x)[M,K] @ W^T with W[n,k] = e2m1(code) * 2^(scale[n,k/32] - 127) (OCP MXFP4, AMD Quark).
// Port of kernels/hip/mxfp4.hip (same layout, same QDQ math); launcher crates/intarsia-backend-metal/src/mxfp4.rs.
//   w     u8 [N, K/2]   two codes per byte, element 2i in the low nibble (bound 4-byte aligned, no skip)
//   scale u8 [N, K/32]  e8m0, plain row-major
//   act   f32 [1]       0: q(x) = x (W4A16); 1: q(x) = dynamic MXFP4 QDQ of x per 32-element group along K (W4A4, scale
//                       rule "even", crates/intarsia-core/src/quant/mxfp4.rs)
// Two dispatches per launch (one when the GEMV reads x itself, see `fused`):
//   mxfp4_xprep_bf16       xs[M, K] (bf16 scratch) = act ? QDQ(x) : x, one thread per 32-element group. The QDQ result
//                          (e2m1 * 2^e) is exact in bf16. Skipped when x is 16-byte aligned (OpP d = 1, "fused"): the GEMV then
//                          loads x (buffer 5) and QDQs a group in the lane that owns it when act (buffer 6) says so.
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

// e2m1 codes -> floats without a table, two per step (the half trick of nvfp4.metal `nv_x2`): the 3 magnitude bits of a code placed at half
// bits 9..11 and its sign at bit 15 are a half whose value is the e2m1 value * 2^-14 exactly (exponent 0 is the denormal 0.5 * 2^-14),
// so one mask-shift pair per nibble pair gives two floats. A dot product over such values is 2^-14 times the true one: `mx_dec8`
// returns the 8 codes of a word, in element order, as floats * 2^-14, and the caller multiplies a group's sum by `MX_FOLD` (2^14, exact)
// before the e8m0 scale. The products and the order of the f32 fma chain are those of the table decode (every value and product is exact
// in f32 either way; they differ only if a product falls below 2^-126, i.e. an activation below about 2^-111).
inline float2 mx_x2(uint t) {
    const uint h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    return float2(as_type<half2>(h));
}
constant constexpr float MX_FOLD = 16384.0f;

inline void mx_dec8(uint v, thread float* o) {
    for (int k = 0; k < 4; ++k) {
        const float2 c = mx_x2((v >> (4 * k)) & 0x000F000Fu);  // elements k and k + 4 of the word
        o[k] = c.x;
        o[k + 4] = c.y;
    }
}

// ---- GEMV ---------------------------------------------------------------------------------------------------------

constant uint MX_SG = 4;  // simdgroups (= output rows) per threadgroup

// M = 1: one simdgroup per output row. The 32 activations of a group are converted to f32 once (the weights of a group are then decoded and
// consumed word by word, so the live state stays small); same products in the same order as `mxfp4_gemv_impl`.
template <typename OutT>
inline void mxfp4_gemv1_impl(device const ushort* xs, device const uchar* w, device const uchar* sc, device OutT* out,
                             constant OpP& p, device const ushort* x, device const float* act, uint2 tg, uint sg, uint lane) {
    const ulong n = (ulong)tg.x * MX_SG + sg;
    if (n >= p.b) return;  // whole simdgroup exits together
    const uint ng = p.c / 32;
    device const packed_uint4* wr = (device const packed_uint4*)(w + n * (p.c / 2));
    device const uchar* sr = sc + p.s0 + n * ng;
    // fused (p.d == 1): the activations come straight from x, QDQ'd in the lane when act says so (the pre-pass's result, bit for bit:
    // a lane's group is the QDQ group); else from the pre-pass scratch xs
    const bool fused = p.d == 1u;
    const bool qdq = fused && act[0] != 0.f;
    device const ushort* xsrc = fused ? x : xs;
    float acc = 0.f;
    for (uint u = lane; u < ng; u += 32) {
        const uint4 wv = uint4(wr[u]);
        const float s = mx_e8m0(sr[u]);
        device const uint4* xp = (device const uint4*)(xsrc + (ulong)u * 32);
        float xf[32];
        for (int q = 0; q < 4; ++q) {
            const uint4 xv = xp[q];
            for (int j = 0; j < 4; ++j) {
                xf[q * 8 + 2 * j] = as_type<float>(xv[j] << 16);
                xf[q * 8 + 2 * j + 1] = as_type<float>(xv[j] & 0xFFFF0000u);
            }
        }
        if (qdq) mx_qdq32(xf);
        float t = 0.f;
        for (int q = 0; q < 4; ++q) {
            float wf[8];
            mx_dec8(wv[q], wf);
            for (int b = 0; b < 8; ++b) t = fma(wf[b], xf[q * 8 + b], t);
        }
        acc = fma(t * MX_FOLD, s, acc);
    }
    const float t = simd_sum(acc);
    if (lane == 0) store_out(out + p.s1, n, t);
}

// M > 1: MB activation rows per grid row; each weight group is decoded once and applied to every row.
template <int MB, typename OutT>
inline void mxfp4_gemv_impl(device const ushort* xs, device const uchar* w, device const uchar* sc, device OutT* out,
                            constant OpP& p, device const ushort* x, device const float* act, uint2 tg, uint sg, uint lane) {
    const ulong n = (ulong)tg.x * MX_SG + sg;
    if (n >= p.b) return;  // whole simdgroup exits together
    const uint M = p.a, K = p.c;
    const uint m0 = tg.y * MB;
    const uint ng = K / 32;
    device const packed_uint4* wr = (device const packed_uint4*)(w + n * (K / 2));
    device const uchar* sr = sc + p.s0 + n * ng;
    const bool fused = p.d == 1u;  // see mxfp4_gemv1_impl
    const bool qdq = fused && act[0] != 0.f;
    device const ushort* xsrc = fused ? x : xs;
    float acc[MB];
    for (int m = 0; m < MB; ++m) acc[m] = 0.f;
    for (uint u = lane; u < ng; u += 32) {
        const uint4 wv = uint4(wr[u]);
        const float s = mx_e8m0(sr[u]);
        float wf[32];
        for (int q = 0; q < 4; ++q) mx_dec8(wv[q], wf + q * 8);
        for (int m = 0; m < MB; ++m) {
            if (m0 + m >= M) break;
            // 32 bf16 of the activation row as four 16-byte loads (xsrc is 16-byte aligned, K % 32 == 0)
            device const uint4* xp = (device const uint4*)(xsrc + (ulong)(m0 + m) * K + (ulong)u * 32);
            float xf[32];
            for (int q = 0; q < 4; ++q) {
                const uint4 xv = xp[q];
                for (int j = 0; j < 4; ++j) {
                    xf[q * 8 + 2 * j] = as_type<float>(xv[j] << 16);
                    xf[q * 8 + 2 * j + 1] = as_type<float>(xv[j] & 0xFFFF0000u);
                }
            }
            if (qdq) mx_qdq32(xf);
            float t = 0.f;
            for (int e = 0; e < 32; ++e) t = fma(wf[e], xf[e], t);
            acc[m] = fma(t * MX_FOLD, s, acc[m]);
        }
    }
    for (int m = 0; m < MB; ++m) {
        if (m0 + m >= M) break;
        const float t = simd_sum(acc[m]);
        if (lane == 0) store_out(out + p.s1, (ulong)(m0 + m) * p.b + n, t);
    }
}

#define MX_GEMV(NAME, IMPL, OUT)                                                                                    \
    kernel void NAME(device const ushort* xs [[buffer(0)]], device const uchar* w [[buffer(1)]],                    \
                     device const uchar* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], device const ushort* x [[buffer(5)]],                            \
                     device const float* act [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],              \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        IMPL(xs, w, sc, out, p, x, act, tg, sg, lane);                                                              \
    }

MX_GEMV(mxfp4_gemv_bf16_m1, mxfp4_gemv1_impl<ushort>, ushort)
MX_GEMV(mxfp4_gemv_bf16_m8, (mxfp4_gemv_impl<8, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_m1, mxfp4_gemv1_impl<float>, float)
MX_GEMV(mxfp4_gemv_f32_m8, (mxfp4_gemv_impl<8, float>), float)

// ---- weight decode for the NAX GEMM (matmul_mxfp4_nax: gemm_nax_fp8.metal's GEMM, launcher mxfp4.rs) --------------------
// `mxfp4_w_to_bf16`: W[N,K/2] codes * e8m0 scales -> bf16 [N,K] (e2m1 * 2^e: at most two significant bits, so the bf16 is
// exact, as it is in the GEMV above; a NaN scale byte gives NaN), and the GEMM's scale pair [1, 0] (no weight scale, no input
// scale) after the weights. p.b = N, p.c = K (K % 32 == 0), p.s0 = scale skip; one thread decodes one 32-weight group.
// bind 0 W (4-byte aligned), 1 scales, 2 Wb (16-byte aligned), 3 scale pair, 4 OpP.
kernel void mxfp4_w_to_bf16(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                            device uint4* wb [[buffer(2)]], device float* pair [[buffer(3)]],
                            constant OpP& p [[buffer(4)]], uint g [[thread_position_in_grid]]) {
    if (g == 0) {
        pair[0] = 1.0f;
        pair[1] = 0.0f;
    }
    if ((ulong)g >= (ulong)p.b * (p.c / 32)) return;
    const uint4 wv = uint4(*(device const packed_uint4*)(w + (ulong)g * 16));
    const float s = mx_e8m0(sc[p.s0 + (ulong)g]);
    uint o[16];
    for (int q = 0; q < 4; ++q) {
        const uint v = wv[q];
        for (int b = 0; b < 4; ++b) {  // elements 8q + 2b (low nibble) and 8q + 2b + 1 of the group
            const float lo = kFp4[(v >> (8 * b)) & 0xFu] * s, hi = kFp4[(v >> (8 * b + 4)) & 0xFu] * s;
            o[q * 4 + b] = (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xFFFF0000u);
        }
    }
    for (int i = 0; i < 4; ++i) wb[4 * (ulong)g + i] = uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}
