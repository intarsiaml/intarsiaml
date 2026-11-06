// matmul_mxfp4: out[M,N] = q(x)[M,K] @ W^T with W[n,k] = e2m1(code) * 2^(scale[n,k/32] - 127) (OCP MXFP4, AMD Quark).
// Port of kernels/hip/mxfp4.hip (same layout, same QDQ math); launcher crates/intarsia-backend-metal/src/mxfp4.rs.
//   w     u8 [N, K/2]   two codes per byte, element 2i in the low nibble (bound 4-byte aligned, no skip)
//   scale u8 [N, K/32]  e8m0, plain row-major
//   act   f32 [1]       0: q(x) = x (W4A16); 1: q(x) = dynamic MXFP4 QDQ of x per 32-element group along K (W4A4, scale
//                       rule "even", crates/intarsia-core/src/quant/mxfp4.rs)
// Two dispatches per launch (one when the GEMV reads x itself, see `fused`):
//   mxfp4_xprep_bf16       xs[M, K] (bf16 scratch) = act ? QDQ(x) : x, one thread per 32-element group. The QDQ result
//                          (e2m1 * 2^e) is exact in bf16. Skipped for W4A16 calls (OpAttrs::mxfp4_w4a16, act is 0) with 16-byte
//                          aligned x (OpP d = 1, "fused"): the GEMV then loads x (buffer 5) itself. W4A4 keeps the pre-pass: a
//                          QDQ in every output row's simdgroup repeats each group's QDQ N times (measured 1.9x to 3.4x slower).
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
                             constant OpP& p, device const ushort* x, uint2 tg, uint sg, uint lane) {
    const ulong n = (ulong)tg.x * MX_SG + sg;
    if (n >= p.b) return;  // whole simdgroup exits together
    const uint ng = p.c / 32;
    device const packed_uint4* wr = (device const packed_uint4*)(w + n * (p.c / 2));
    device const uchar* sr = sc + p.s0 + n * ng;
    // fused (p.d == 1, W4A16 only): the activations come straight from x (the pre-pass would copy them); else from the pre-pass scratch xs
    device const ushort* xsrc = p.d == 1u ? x : xs;
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
                            constant OpP& p, device const ushort* x, uint2 tg, uint sg, uint lane) {
    const ulong n = (ulong)tg.x * MX_SG + sg;
    if (n >= p.b) return;  // whole simdgroup exits together
    const uint M = p.a, K = p.c;
    const uint m0 = tg.y * MB;
    const uint ng = K / 32;
    device const packed_uint4* wr = (device const packed_uint4*)(w + n * (K / 2));
    device const uchar* sr = sc + p.s0 + n * ng;
    device const ushort* xsrc = p.d == 1u ? x : xs;  // see mxfp4_gemv1_impl
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
            float t = 0.f;
            for (int q = 0; q < 4; ++q) {
                const uint4 xv = xp[q];
                for (int j = 0; j < 4; ++j) {
                    const uint pr = xv[j];
                    t = fma(wf[q * 8 + 2 * j], as_type<float>(pr << 16), t);
                    t = fma(wf[q * 8 + 2 * j + 1], as_type<float>(pr & 0xFFFF0000u), t);
                }
            }
            acc[m] = fma(t * MX_FOLD, s, acc[m]);
        }
    }
    for (int m = 0; m < MB; ++m) {
        if (m0 + m >= M) break;
        const float t = simd_sum(acc[m]);
        if (lane == 0) store_out(out + p.s1, (ulong)(m0 + m) * p.b + n, t);
    }
}

// M = 2..8 (lane df, `INTARSIA_METAL_MXFP4_MR`): P output rows per simdgroup, exactly M activation rows. The `_m8` form above
// widens every activation row's bf16 once per output row and is issue-bound (M = 8 costs 4.8x M = 1 at the 27B shapes); here the
// activations of one 8-element word are widened once per simdgroup and used for its P weight rows, and each weight word is decoded
// once for all M rows. Bits: every (weight row, activation row) pair keeps `mxfp4_gemv1_impl`'s chain exactly: the lane's groups
// u = lane, lane + 32, ..; per group t = fma chain over the 32 elements in element order (here carried across the four words in
// `t[p][m]`), then acc = fma(t * MX_FOLD, s, acc); then simd_sum. Only the loop nesting and the registers change.
template <int M, int P, typename OutT>
inline void mxfp4_gemv_mr_impl(device const ushort* xs, device const uchar* w, device const uchar* sc, device OutT* out,
                               constant OpP& p, device const ushort* x, uint2 tg, uint sg, uint lane) {
    const ulong n0 = ((ulong)tg.x * MX_SG + sg) * P;
    if (n0 >= p.b) return;  // whole simdgroup exits together
    const uint K = p.c;
    const uint ng = K / 32;
    device const ushort* xsrc = p.d == 1u ? x : xs;  // see mxfp4_gemv1_impl
    // rows past N (a tail simdgroup) read row n0's weights and store nothing
    device const packed_uint4* wr[P];
    device const uchar* sr[P];
    for (int q = 0; q < P; ++q) {
        const ulong n = n0 + q < p.b ? n0 + q : n0;
        wr[q] = (device const packed_uint4*)(w + n * (K / 2));
        sr[q] = sc + p.s0 + n * ng;
    }
    float acc[P][M];
    for (int q = 0; q < P; ++q)
        for (int m = 0; m < M; ++m) acc[q][m] = 0.f;
    for (uint u = lane; u < ng; u += 32) {
        uint4 wv[P];
        for (int q = 0; q < P; ++q) wv[q] = uint4(wr[q][u]);
        float t[P][M];
        for (int q = 0; q < P; ++q)
            for (int m = 0; m < M; ++m) t[q][m] = 0.f;
        for (int c = 0; c < 4; ++c) {
            float wf[P][8];
            for (int q = 0; q < P; ++q) mx_dec8(wv[q][c], wf[q]);
            for (int m = 0; m < M; ++m) {
                const uint4 xv = ((device const uint4*)(xsrc + (ulong)m * K + (ulong)u * 32))[c];
                float xf[8];
                for (int j = 0; j < 4; ++j) {
                    xf[2 * j] = as_type<float>(xv[j] << 16);
                    xf[2 * j + 1] = as_type<float>(xv[j] & 0xFFFF0000u);
                }
                for (int q = 0; q < P; ++q)
                    for (int b = 0; b < 8; ++b) t[q][m] = fma(wf[q][b], xf[b], t[q][m]);
            }
        }
        for (int q = 0; q < P; ++q) {
            const float s = mx_e8m0(sr[q][u]);
            for (int m = 0; m < M; ++m) acc[q][m] = fma(t[q][m] * MX_FOLD, s, acc[q][m]);
        }
    }
    for (int q = 0; q < P; ++q) {
        for (int m = 0; m < M; ++m) {
            const float r = simd_sum(acc[q][m]);
            if (lane == 0 && n0 + q < p.b) store_out(out + p.s1, (ulong)m * p.b + n0 + q, r);
        }
    }
}

// M = 2..8 on the simdgroup matrix instructions (lane dm, `INTARSIA_METAL_MXFP4_MR=8`, opt-in). Bitwise `mxfp4_gemv1_impl` per row ONLY
// because fp32 `simdgroup_multiply_accumulate` was measured to be a sequential k-ascending chain of rounded adds on the arch this is
// registered for (lane dm probe, docs/results/2026-11-06-metal-dflash2.md; every MXFP4 product is exact in f32, so a rounded add per
// product is the fma chain of the one-row kernel). Tile: 8 weight rows (A, n x k) x 8 activation rows (B, k x m; rows >= M are zero
// and never stored); one group u = 4 chained MMAs from C = 0 (t of the one-row kernel for all 64 (n, m) pairs at once), then per
// element acc = fma(t * MX_FOLD, s[n], acc). The one-row kernel's reduction is kept exactly: its lane l sums groups u = l, l + 32, ..
// in order and then `simd_sum`s over lanes; here the simdgroup walks class l = 0..31 (groups l, l + 32, ..), parks class l's 8 x 8
// accumulator in threadgroup slot l, and at the end lane l reads slot l for each pair and the same `simd_sum` runs.
// Fragment coordinates are not assumed: an index matrix loaded once tells each thread which (row, col) its two elements are.
constant constexpr uint MMA_SG = 2;  // simdgroups (8 output rows each) per threadgroup

template <typename OutT>
inline void mxfp4_gemv_mma_impl(device const ushort* xs, device const uchar* w, device const uchar* sc, device OutT* out,
                                constant OpP& p, device const ushort* x, uint2 tg, uint sg, uint lane,
                                threadgroup float* slots, threadgroup float* idx) {
    // slots: [MMA_SG][32][64] accumulators; idx: [64] = 0..63
    for (uint i = sg * 32 + lane; i < 64; i += MMA_SG * 32) idx[i] = (float)i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 im;
    simdgroup_load(im, idx, 8);
    const uint e0 = (uint)im.thread_elements()[0], e1 = (uint)im.thread_elements()[1];
    const uint r0 = e0 / 8, c0 = e0 % 8, r1 = e1 / 8, c1 = e1 % 8;
    const ulong n0 = ((ulong)tg.x * MMA_SG + sg) * 8;
    if (n0 >= p.b) return;  // whole simdgroup exits together (no barrier follows)
    const uint M = p.a, K = p.c, ng = K / 32;
    device const ushort* xsrc = p.d == 1u ? x : xs;  // see mxfp4_gemv1_impl
    // A elements: weight rows n0 + r (rows past N read row n0, never stored); B elements: activation rows c (>= M give 0)
    const ulong na0 = n0 + r0 < p.b ? n0 + r0 : n0, na1 = n0 + r1 < p.b ? n0 + r1 : n0;
    device const uint* wa0 = (device const uint*)(w + na0 * (K / 2));
    device const uint* wa1 = (device const uint*)(w + na1 * (K / 2));
    // C / acc elements: (n = r, m = c); their scales are rows n0 + r
    const ulong ns0 = na0, ns1 = na1;
    threadgroup float* myslots = slots + sg * 32 * 64;
    for (uint l = 0; l < 32; ++l) {
        float a0 = 0.f, a1 = 0.f;
        for (uint u = l; u < ng; u += 32) {
            simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
            for (int q = 0; q < 4; ++q) {
                simdgroup_float8x8 A, B;
                // A[r][c] = element u*32 + q*8 + c of weight row n0 + r (word q of the group, nibble c)
                const uint wd0 = wa0[u * 4 + q], wd1 = wa1[u * 4 + q];
                const float2 d0 = mx_x2(((wd0 >> (4 * c0)) & 0xFu) | (((wd1 >> (4 * c1)) & 0xFu) << 16));
                A.thread_elements()[0] = d0.x;
                A.thread_elements()[1] = d0.y;
                // B[r][c] = x[c][u*32 + q*8 + r]
                const ulong kk0 = (ulong)u * 32 + q * 8 + r0, kk1 = (ulong)u * 32 + q * 8 + r1;
                const float b0 = c0 < M ? as_type<float>((uint)xsrc[(ulong)c0 * K + kk0] << 16) : 0.f;
                const float b1 = c1 < M ? as_type<float>((uint)xsrc[(ulong)c1 * K + kk1] << 16) : 0.f;
                B.thread_elements()[0] = b0;
                B.thread_elements()[1] = b1;
                simdgroup_multiply_accumulate(c, A, B, c);
            }
            a0 = fma(c.thread_elements()[0] * MX_FOLD, mx_e8m0(sc[p.s0 + ns0 * ng + u]), a0);
            a1 = fma(c.thread_elements()[1] * MX_FOLD, mx_e8m0(sc[p.s0 + ns1 * ng + u]), a1);
        }
        myslots[l * 64 + e0] = a0;
        myslots[l * 64 + e1] = a1;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = 0; n < 8; ++n) {
        for (uint m = 0; m < M; ++m) {
            const float r = simd_sum(myslots[lane * 64 + n * 8 + m]);
            if (lane == 0 && n0 + n < p.b) store_out(out + p.s1, (ulong)m * p.b + n0 + n, r);
        }
    }
}

#define MX_GEMV_MMA(NAME, OUT)                                                                                     \
    kernel void NAME(device const ushort* xs [[buffer(0)]], device const uchar* w [[buffer(1)]],                    \
                     device const uchar* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], device const ushort* x [[buffer(5)]],                            \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        threadgroup float slots[MMA_SG * 32 * 64];                                                                  \
        threadgroup float idx[64];                                                                                  \
        mxfp4_gemv_mma_impl<OUT>(xs, w, sc, out, p, x, tg, sg, lane, slots, idx);                                   \
    }

#define MX_GEMV(NAME, IMPL, OUT)                                                                                  \
    kernel void NAME(device const ushort* xs [[buffer(0)]], device const uchar* w [[buffer(1)]],                    \
                     device const uchar* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], device const ushort* x [[buffer(5)]],                            \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        IMPL(xs, w, sc, out, p, x, tg, sg, lane);                                                                   \
    }

MX_GEMV(mxfp4_gemv_bf16_m1, mxfp4_gemv1_impl<ushort>, ushort)
MX_GEMV(mxfp4_gemv_bf16_m8, (mxfp4_gemv_impl<8, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_m1, mxfp4_gemv1_impl<float>, float)
MX_GEMV(mxfp4_gemv_f32_m8, (mxfp4_gemv_impl<8, float>), float)
// lane df: M = 2..8 rows, P = 2 or 4 output rows per simdgroup (`INTARSIA_METAL_MXFP4_MR`)
MX_GEMV(mxfp4_gemv_bf16_mr2p2, (mxfp4_gemv_mr_impl<2, 2, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr2p2, (mxfp4_gemv_mr_impl<2, 2, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr3p2, (mxfp4_gemv_mr_impl<3, 2, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr3p2, (mxfp4_gemv_mr_impl<3, 2, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr4p2, (mxfp4_gemv_mr_impl<4, 2, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr4p2, (mxfp4_gemv_mr_impl<4, 2, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr5p2, (mxfp4_gemv_mr_impl<5, 2, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr5p2, (mxfp4_gemv_mr_impl<5, 2, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr6p2, (mxfp4_gemv_mr_impl<6, 2, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr6p2, (mxfp4_gemv_mr_impl<6, 2, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr7p2, (mxfp4_gemv_mr_impl<7, 2, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr7p2, (mxfp4_gemv_mr_impl<7, 2, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr8p2, (mxfp4_gemv_mr_impl<8, 2, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr8p2, (mxfp4_gemv_mr_impl<8, 2, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr2p4, (mxfp4_gemv_mr_impl<2, 4, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr2p4, (mxfp4_gemv_mr_impl<2, 4, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr3p4, (mxfp4_gemv_mr_impl<3, 4, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr3p4, (mxfp4_gemv_mr_impl<3, 4, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr4p4, (mxfp4_gemv_mr_impl<4, 4, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr4p4, (mxfp4_gemv_mr_impl<4, 4, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr5p4, (mxfp4_gemv_mr_impl<5, 4, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr5p4, (mxfp4_gemv_mr_impl<5, 4, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr6p4, (mxfp4_gemv_mr_impl<6, 4, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr6p4, (mxfp4_gemv_mr_impl<6, 4, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr7p4, (mxfp4_gemv_mr_impl<7, 4, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr7p4, (mxfp4_gemv_mr_impl<7, 4, float>), float)
MX_GEMV(mxfp4_gemv_bf16_mr8p4, (mxfp4_gemv_mr_impl<8, 4, ushort>), ushort)
MX_GEMV(mxfp4_gemv_f32_mr8p4, (mxfp4_gemv_mr_impl<8, 4, float>), float)

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
// lane dm: M = 2..8 on simdgroup matrices (`INTARSIA_METAL_MXFP4_MR=8`)
MX_GEMV_MMA(mxfp4_gemv_bf16_mma, ushort)
MX_GEMV_MMA(mxfp4_gemv_f32_mma, float)
