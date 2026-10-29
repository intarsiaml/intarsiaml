// matmul_fp8: out[M,N] = q(x)[M,K] @ W^T * (weight_scale * input_scale), W[n,k] OCP e4m3 bytes (DType::Fp8E4m3).
// Port of kernels/hip/fp8.hip (same semantics as crates/intarsia-core/src/quant/reference.rs matmul_fp8);
// launcher crates/intarsia-backend-metal/src/fp8.rs.
//   x      bf16 [M,K]
//   w      u8   [N,K]   e4m3, bound 4-byte aligned; the launcher requires an 8-byte aligned address (rows are K % 16 == 0 bytes)
//   scales f32  [2]     (weight_scale, input_scale), read here on the GPU. input_scale > 0: W8A8 static, q(x) = e4m3(x / is)
//                       (round to nearest even, saturating at 448, NaN -> 0x7F, `fp8_enc`), result * ws * is; else W8A16, q(x) = x.
// e4m3 -> float: the 7 magnitude bits are placed in the exponent / mantissa of a half (<< 7), which reads as value / 256
// (subnormals included); the factor 256 is folded into the final scale. Codes 0x7F / 0xFF (NaN) are caught per word.
// Entry points:
//   fp8_xprep_bf16        xs[M,K] (bf16 scratch) = QDQ(x) (exact in bf16) or x; one thread per 8 elements (GEMM only)
//   fp8_gemv_{bf16,f32}_{m1,m4,m8}
//                         decode GEMV, M <= 1 / 4 / 8 per call. A threadgroup of FP8_SG simdgroups stages its q(x) rows once in
//                         threadgroup memory (bf16, launcher-sized M * K * 2 bytes), each simdgroup owns RS output
//                         rows (4 for _m1, 2 for _m8; a lane takes 16-element chunks of K, UN of them loaded before any is
//                         used, 2 x 8-byte loads per row and chunk), f32 accumulate, simd_sum.
//   fp8_mma_{bf16,f32}    GEMM for any M, 64 x 64 tile, 4 simdgroups of 32 x 32, K step 32, bf16 operand tiles (e4m3 is
//                         exact in bf16), simdgroup_matrix bf16 x bf16 -> f32; reads xs.
// Kernel arguments: OpP (common.h), a = M, b = N, c = K; xprep s0 = x skip; gemv / mma s0 = x skip (0 for mma: xs),
// s1 = out skip.

#include "common.h"

// ---- e4m3 ----------------------------------------------------------------------------------------------------------

// a word holds a NaN code (0x7F / 0xFF) in some byte
inline bool fp8_word_nan(uint w) {
    const uint t = (w & 0x7F7F7F7Fu) ^ 0x7F7F7F7Fu;  // a zero byte <=> NaN code
    return ((t - 0x01010101u) & ~t & 0x80808080u) != 0u;
}

// 16 e4m3 bytes (4 words) . 16 activations; the result is the dot / 256, NaN when the chunk holds a NaN code
inline float fp8_dot16(uint2 a, uint2 b, const thread float* x) {
    const uint v[4] = {a.x, a.y, b.x, b.y};
    float t = 0.0f;
    bool nan = false;
    for (int j = 0; j < 4; ++j) {
        const float2 p = fp8_x2(v[j]), q = fp8_x2(v[j] >> 8);  // elements (0, 2) and (1, 3) of the word
        t = fma(p.x, x[4 * j], t);
        t = fma(q.x, x[4 * j + 1], t);
        t = fma(p.y, x[4 * j + 2], t);
        t = fma(q.y, x[4 * j + 3], t);
        nan = nan || fp8_word_nan(v[j]);
    }
    return nan ? as_type<float>(0x7FC00000u) : t;
}

// the weight half of `fp8_dot16`: 16 e4m3 bytes -> 16 floats (value / 256, element order 0..15) and the NaN flag
inline void fp8_dec16(uint2 a, uint2 b, thread float* wv, thread bool& nan) {
    const uint v[4] = {a.x, a.y, b.x, b.y};
    nan = false;
    for (int j = 0; j < 4; ++j) {
        const float2 p = fp8_x2(v[j]), q = fp8_x2(v[j] >> 8);
        wv[4 * j] = p.x;
        wv[4 * j + 1] = q.x;
        wv[4 * j + 2] = p.y;
        wv[4 * j + 3] = q.y;
        nan = nan || fp8_word_nan(v[j]);
    }
}

// the activation half: the same fma chain as `fp8_dot16` over decoded weights
inline float fp8_dot16d(const thread float* wv, bool nan, const thread float* x) {
    float t = 0.0f;
    for (int e = 0; e < 16; ++e) t = fma(wv[e], x[e], t);
    return nan ? as_type<float>(0x7FC00000u) : t;
}

// bf16 pair in a word -> two floats
inline void fp8_unpack2(uint w, thread float* o) {
    o[0] = as_type<float>(w << 16);
    o[1] = as_type<float>(w & 0xFFFF0000u);
}

// ---- activation pre-pass (GEMM) ---------------------------------------------------------------------------------------

kernel void fp8_xprep_bf16(device const ushort* x [[buffer(0)]], device const float* sc [[buffer(1)]],
                           device ushort* xs [[buffer(2)]], constant OpP& p [[buffer(3)]],
                           uint g [[thread_position_in_grid]]) {
    const ulong total = (ulong)p.a * p.c;
    const ulong e0 = (ulong)g * 8;
    if (e0 >= total) return;
    const float is = sc[1];
    for (int e = 0; e < 8; ++e) {
        const ushort v = x[p.s0 + e0 + e];
        xs[e0 + e] = is > 0.0f ? f2bf(fp8_qx(bf2f(v), is)) : v;
    }
}

// ---- decode GEMV ------------------------------------------------------------------------------------------------------

constant constexpr uint FP8_SG = 8;  // simdgroups per threadgroup

// XD (the `_xd` multi-row forms): x is q(x) already, [M, K] bf16 written once per call by fp8_xprep_bf16 (the same
// expression the staging loop below evaluates), and is read straight from device memory: no threadgroup staging, so no
// per-threadgroup q(x) of all the rows and no 32 KB of threadgroup memory capping the threadgroups per core. Per row the
// values and the fma chain are those of the staged forms, so the bits are the one-row kernel's.
template <int MB, int RS, int UN, typename OutT, bool XD = false, bool NEG = false>
inline void fp8_gemv_impl(device const ushort* x, device const uchar* w, device const float* sc, device OutT* out,
                          constant OpP& p, threadgroup ushort* xs, uint2 tg, uint tid, uint sg, uint lane) {
    const uint M = p.a, N = p.b, K = p.c;
    const uint m0 = tg.y * MB;
    const uint rows = min((uint)MB, M - m0);
    const float ws = sc[0], is = sc[1];
    if constexpr (!XD) {
        // stage q(x) of this threadgroup's activation rows
        for (uint i = tid; i < rows * K; i += FP8_SG * 32) {
            const ushort v = x[p.s0 + (ulong)m0 * K + i];
            xs[i] = is > 0.0f ? f2bf(fp8_qx(bf2f(v), is)) : v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device const ushort* xd = x + (ulong)m0 * K;  // XD: q(x) rows of this threadgroup (the prep buffer has no skip)

    const uint nb = (tg.x * FP8_SG + sg) * RS;
    if (nb >= N) return;  // whole simdgroup exits together
    device const uchar* wr[RS];
    for (int r = 0; r < RS; ++r) wr[r] = w + (ulong)min(nb + r, N - 1) * K;
    float acc[RS][MB];
    for (int r = 0; r < RS; ++r)
        for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
    const uint nch = K / 16;
    for (uint u0 = lane; u0 < nch; u0 += 32 * UN) {
        uint2 wa[UN][RS], wb[UN][RS];  // all loads of the step first: UN * RS * 16 bytes in flight per lane
        for (int j = 0; j < UN; ++j) {
            const uint u = min(u0 + 32 * j, nch - 1);
            for (int r = 0; r < RS; ++r) {
                wa[j][r] = *(device const uint2*)(wr[r] + (ulong)u * 16);
                wb[j][r] = *(device const uint2*)(wr[r] + (ulong)u * 16 + 8);
            }
        }
        for (int j = 0; j < UN; ++j) {
            const uint u = u0 + 32 * j;
            if (u >= nch) break;
            if constexpr (MB > 1) {
                // multi-row forms: decode each weight chunk ONCE and apply it to every activation row (the e4m3 decode is the
                // dominant ALU cost; per row it is the same fma chain over the same values in the same order as `fp8_dot16`,
                // so every row's bits are those of the one-row kernel)
                float wv[RS][16];
                bool wnan[RS];
                for (int r = 0; r < RS; ++r) fp8_dec16(wa[j][r], wb[j][r], wv[r], wnan[r]);
                for (int m = 0; m < MB; ++m) {
                    if (m >= (int)rows) break;
                    uint4 x0, x1;
                    if constexpr (XD) {
                        device const uint4* xp = (device const uint4*)(xd + (ulong)m * K + (ulong)u * 16);
                        x0 = xp[0];
                        x1 = xp[1];
                    } else {
                        threadgroup const uint4* xp = (threadgroup const uint4*)(xs + (ulong)m * K + (ulong)u * 16);
                        x0 = xp[0];
                        x1 = xp[1];
                    }
                    float xf[16];
                    fp8_unpack2(x0.x, xf);
                    fp8_unpack2(x0.y, xf + 2);
                    fp8_unpack2(x0.z, xf + 4);
                    fp8_unpack2(x0.w, xf + 6);
                    fp8_unpack2(x1.x, xf + 8);
                    fp8_unpack2(x1.y, xf + 10);
                    fp8_unpack2(x1.z, xf + 12);
                    fp8_unpack2(x1.w, xf + 14);
                    for (int r = 0; r < RS; ++r) {
                        const float d = fp8_dot16d(wv[r], wnan[r], xf);
                        // NEG (negative control only): another association on odd blocks, which the row gate must catch
                        if (NEG && (u & 1u)) acc[r][m] = (acc[r][m] + 0.5f * d) + 0.5f * d;
                        else acc[r][m] += d;
                    }
                }
            } else {
                for (int m = 0; m < MB; ++m) {
                    if (m >= (int)rows) break;
                    uint4 x0, x1;
                    if constexpr (XD) {
                        // one row, q(x) already in device memory (written by the producer of x, lane mnq): the same values
                        // the staging loop would put in threadgroup memory, read from there instead
                        device const uint4* xp = (device const uint4*)(xd + (ulong)m * K + (ulong)u * 16);
                        x0 = xp[0];
                        x1 = xp[1];
                    } else {
                        threadgroup const uint4* xp = (threadgroup const uint4*)(xs + (ulong)m * K + (ulong)u * 16);
                        x0 = xp[0];
                        x1 = xp[1];
                    }
                    float xf[16];
                    fp8_unpack2(x0.x, xf);
                    fp8_unpack2(x0.y, xf + 2);
                    fp8_unpack2(x0.z, xf + 4);
                    fp8_unpack2(x0.w, xf + 6);
                    fp8_unpack2(x1.x, xf + 8);
                    fp8_unpack2(x1.y, xf + 10);
                    fp8_unpack2(x1.z, xf + 12);
                    fp8_unpack2(x1.w, xf + 14);
                    for (int r = 0; r < RS; ++r) acc[r][m] += fp8_dot16(wa[j][r], wb[j][r], xf);
                }
            }
        }
    }
    const float scale = ws * (is > 0.0f ? is : 1.0f) * 256.0f;
    for (int m = 0; m < MB; ++m) {
        if (m >= (int)rows) break;
        for (int r = 0; r < RS; ++r) {
            const float t = simd_sum(acc[r][m]);
            if (lane == 0 && nb + r < N) store_out(out + p.s1, (ulong)(m0 + m) * N + nb + r, t * scale);
        }
    }
}

#define FP8_GEMV(NAME, MB, RS, UN, OUT)                                                                                 \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], threadgroup ushort* xs [[threadgroup(0)]],                      \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],           \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        fp8_gemv_impl<MB, RS, UN, OUT>(x, w, sc, out, p, xs, tg, tid, sg, lane);                                       \
    }
#define FP8_GEMV_XD(NAME, MB, RS, UN, OUT) FP8_GEMV_XDN(NAME, MB, RS, UN, OUT, false)
#define FP8_GEMV_XDN(NAME, MB, RS, UN, OUT, NEG)                                                                              \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        fp8_gemv_impl<MB, RS, UN, OUT, true, NEG>(x, w, sc, out, p, nullptr, tg, tid, sg, lane);                            \
    }

FP8_GEMV(fp8_gemv_bf16_m1, 1, 4, 4, ushort)
FP8_GEMV(fp8_gemv_bf16_m8, 8, 2, 2, ushort)
FP8_GEMV(fp8_gemv_f32_m1, 1, 4, 4, float)
FP8_GEMV(fp8_gemv_f32_m8, 8, 2, 2, float)
// _m4: the multi-row form for K up to 4096 (4 rows of q(x) fill the 32 KB of threadgroup memory there, 8 rows only up to
// K = 2048); `matmul_fp8_gemv_rows` takes it past M = 4 at K = 4096 (the 35B's GDN / attention output projections)
FP8_GEMV(fp8_gemv_bf16_m4, 4, 2, 2, ushort)
FP8_GEMV(fp8_gemv_f32_m4, 4, 2, 2, float)
// _m1_xd (lane mnq): the one-row GEMV over a q(x) row the producer of x wrote ([1, K] bf16, no skip): no staging pass and no
// threadgroup memory. Same fma chain as `_m1`, so the same bits.
FP8_GEMV_XD(fp8_gemv_bf16_m1_xd, 1, 4, 4, ushort)
FP8_GEMV_XD(fp8_gemv_f32_m1_xd, 1, 4, 4, float)
// `matmul_fp8_gemv_rows` over the q(x) prep buffer (lane mm, docs/results/2026-10-25-mm-metal-moe-grouped.md): 4 rows per
// threadgroup (2..8 rows: one or two threadgroup rows; with no staging the tile is free of the threadgroup-memory limit, and
// 4 measured faster than 8 or 2 at every 35B shape); _neg is the negative control (INTARSIA_METAL_NEGCTL=fp8_xd)
FP8_GEMV_XD(fp8_gemv_bf16_m4_xd, 4, 2, 2, ushort)
FP8_GEMV_XD(fp8_gemv_f32_m4_xd, 4, 2, 2, float)
FP8_GEMV_XDN(fp8_gemv_bf16_m4_xd_neg, 4, 2, 2, ushort, true)
FP8_GEMV_XDN(fp8_gemv_f32_m4_xd_neg, 4, 2, 2, float, true)

// ---- multi-row GEMV on simdgroup matrices (lane mm step 2) ---------------------------------------------------------
// `matmul_fp8_gemv_rows` for M = 5..=8 rows: C'(n, m) = sum_k W(n, k) q(x)(m, k) on 8x8 simdgroup matrices, row-exact.
// Measured on endor (M5 Max, docs/results/2026-10-25-mm-metal-fp8-mmt.md): simdgroup_multiply_accumulate on float 8x8
// equals, bit for bit, the chain c = fma(a[i][k], b[k][j], c) for k = 0..7 from c, per element (subnormals, inf, NaN and
// cancellation included), and simd_sum equals the xor butterfly 1, 2, 4, 8, 16, i.e. the pairwise tree over lanes 0..31
// in index order. So the one-row kernel's value for (row m, output n) is rebuilt from matrices: its lane l owns blocks
// u = l, l + 32, ... and sums their 16-term chains (`fp8_dot16`, NaN when the block holds a NaN code) in that order;
// here block u's chain for 8 outputs x 8 rows is two MMAs from 0 (k = 0..7, then 8..15), a block whose weight row holds a
// NaN code is set to NaN, and it is added to the accumulator of class u mod 32. The 8 simdgroups of a threadgroup take
// classes 4s..4s+3, each folds its 4 ((0+1)+(2+3)), and the threadgroup folds the 8 results ((0+1)+(2+3))+((4+5)+(6+7)):
// the pairwise tree of simd_sum. Then the same scale and rounding.
// Operands: a lane's weight operand is two adjacent bytes of one weight row (one ushort load), its activation operand two
// adjacent rows of one k of the block-transposed q(x) xt[u][k][m] (bf16, MP = 8 R rows, zero padded; fp8_xprep_t).
// Layout of a lane's 2 elements of an 8x8 matrix (measured): row (lane & 16 ? 4 : 0) + ((lane >> 1) & 3), columns
// ((lane >> 3) & 1) * 4 + (lane & 1) * 2 and + 1. One threadgroup per 8 outputs, all rows; padded to 8 rows, so it costs
// the same at M = 2 as at 8 and the launcher keeps the `_xd` form below 5 rows. R = 2 (16 rows) is not instantiated.
// NEG (negative control only): the two MMAs of a block in the other order.
kernel void fp8_xprep_t(device const ushort* x [[buffer(0)]], device const float* sc [[buffer(1)]],
                        device uint* xt [[buffer(2)]], constant OpP& p [[buffer(3)]], uint g [[thread_position_in_grid]]) {
    const uint M = p.a, K = p.c, MP = p.d;
    const uint e = g * 2;  // two rows m, m + 1 of one (u, k)
    if (e >= K * MP) return;
    const uint m = e % MP, kk = e / MP;  // kk = u * 16 + k
    const float is = sc[1];
    uint v[2];
    for (int i = 0; i < 2; ++i) {
        const uint mi = m + i;
        uint b = 0;
        if (mi < M) {
            const ushort xv = x[p.s0 + (ulong)mi * K + kk];
            b = is > 0.0f ? f2bf(fp8_qx(bf2f(xv), is)) : xv;  // the expression the staged GEMV evaluates
        }
        v[i] = b;
    }
    xt[g] = v[0] | (v[1] << 16);
}

template <typename OutT, bool NEG, int R>
inline void fp8_rows_mmt_impl(device const ushort* xt, device const uchar* w, device const float* sc, device OutT* out,
                              constant OpP& p, threadgroup float* red, uint tg, uint sg, uint lane, uint tid) {
    constexpr uint MP = 8 * R;
    const uint M = p.a, N = p.b, K = p.c;
    const uint n0 = tg * 8;
    const uint fm = ((lane & 16u) ? 4u : 0u) + ((lane >> 1) & 3u);
    const uint fn = ((lane >> 3) & 1u) * 4u + (lane & 1u) * 2u;
    device const uchar* wr = w + (ulong)min(n0 + fm, N - 1) * K;
    simdgroup_float8x8 acc[R][4];
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < 4; ++c) acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint nch = K / 16;
    for (uint ub = 4 * sg; ub < nch; ub += 32) {
        uint flags = 0;  // bit c: the block of class c of weight row fm holds a NaN code
        float2 wa[4], wb[4];
        for (int c = 0; c < 4; ++c) {
            const ulong u = min(ub + c, nch - 1);
            const uint v1 = *(device const ushort*)(wr + u * 16 + fn), v2 = *(device const ushort*)(wr + u * 16 + 8 + fn);
            const uint word = v1 | (v2 << 16);
            const float2 pp = fp8_x2(word), qq = fp8_x2(word >> 8);
            wa[c] = float2(pp.x, qq.x);  // (n = fm, k = fn, fn + 1)
            wb[c] = float2(pp.y, qq.y);  // (n = fm, k = 8 + fn, 9 + fn)
            flags |= (fp8_word_nan(word) || fp8_word_nan(word >> 8)) ? (1u << c) : 0u;
        }
        // the 16 bytes of weight row fm lie in the 4 lanes that differ in lane bits 0 and 3
        flags |= simd_shuffle_xor(flags, 1);
        flags |= simd_shuffle_xor(flags, 8);
        for (int c = 0; c < 4; ++c) {
            const uint u = ub + c;
            if (u >= nch) break;  // uniform over the simdgroup
            simdgroup_float8x8 a1m, a2m;
            a1m.thread_elements()[0] = wa[c].x;
            a1m.thread_elements()[1] = wa[c].y;
            a2m.thread_elements()[0] = wb[c].x;
            a2m.thread_elements()[1] = wb[c].y;
            for (int r = 0; r < R; ++r) {
                simdgroup_float8x8 b1m, b2m;
                device const ushort* xb = xt + (ulong)u * 16 * MP + fm * MP + fn + 8 * r;
                const uint x1 = *(device const uint*)xb, x2 = *(device const uint*)(xb + 8 * MP);
                b1m.thread_elements()[0] = as_type<float>(x1 << 16);  // (k = fm, m = fn), (k = fm, m = fn + 1)
                b1m.thread_elements()[1] = as_type<float>(x1 & 0xFFFF0000u);
                b2m.thread_elements()[0] = as_type<float>(x2 << 16);  // k = 8 + fm
                b2m.thread_elements()[1] = as_type<float>(x2 & 0xFFFF0000u);
                simdgroup_float8x8 t = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                if (NEG) {
                    simdgroup_multiply_accumulate(t, a2m, b2m, t);
                    simdgroup_multiply_accumulate(t, a1m, b1m, t);
                } else {
                    simdgroup_multiply_accumulate(t, a1m, b1m, t);
                    simdgroup_multiply_accumulate(t, a2m, b2m, t);
                }
                float2 tv = float2(t.thread_elements()[0], t.thread_elements()[1]);
                if (flags & (1u << c)) tv = float2(as_type<float>(0x7FC00000u));
                acc[r][c].thread_elements()[0] += tv.x;
                acc[r][c].thread_elements()[1] += tv.y;
            }
        }
    }
    // the pairwise tree of simd_sum: classes 4s..4s+3 here, then the 8 simdgroups; red[r][sg][m][n]
    for (int r = 0; r < R; ++r) {
        const float qx = (acc[r][0].thread_elements()[0] + acc[r][1].thread_elements()[0]) +
                         (acc[r][2].thread_elements()[0] + acc[r][3].thread_elements()[0]);
        const float qy = (acc[r][0].thread_elements()[1] + acc[r][1].thread_elements()[1]) +
                         (acc[r][2].thread_elements()[1] + acc[r][3].thread_elements()[1]);
        red[(r * 8 + sg) * 64 + fn * 8 + fm] = qx;
        red[(r * 8 + sg) * 64 + (fn + 1) * 8 + fm] = qy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float ws = sc[0], is = sc[1];
    const float scale = ws * (is > 0.0f ? is : 1.0f) * 256.0f;
    for (uint e = tid; e < 64 * R; e += 256) {
        const uint r = e / 64, i = e % 64;
        const uint m = 8 * r + i / 8, n = n0 + i % 8;
        threadgroup const float* rt = red + r * 512 + i;
        const float v = ((rt[0] + rt[64]) + (rt[128] + rt[192])) + ((rt[256] + rt[320]) + (rt[384] + rt[448]));
        if (m < M && n < N) store_out(out + p.s1, (ulong)m * N + n, v * scale);
    }
}

#define FP8_ROWS_MMT(NAME, OUT, NEG, R)                                                                             \
    kernel void NAME(device const ushort* xt [[buffer(0)]], device const uchar* w [[buffer(1)]],                   \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],                       \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                     uint tid [[thread_index_in_threadgroup]]) {                                                    \
        threadgroup float red[R * 8 * 64];                                                                          \
        fp8_rows_mmt_impl<OUT, NEG, R>(xt, w, sc, out, p, red, tg, sg, lane, tid);                                  \
    }

FP8_ROWS_MMT(fp8_rows_mmt_bf16, ushort, false, 1)
FP8_ROWS_MMT(fp8_rows_mmt_f32, float, false, 1)
FP8_ROWS_MMT(fp8_rows_mmt_bf16_neg, ushort, true, 1)
FP8_ROWS_MMT(fp8_rows_mmt_f32_neg, float, true, 1)

// ---- GEMM -------------------------------------------------------------------------------------------------------------

constant constexpr uint FP8_BM = 64;
constant constexpr uint FP8_BN = 64;
constant constexpr uint FP8_BK = 32;
constant constexpr uint FP8_LD = 40;  // bf16 elements per staged row (80 bytes, 16-byte multiple, skews the banks)

// float -> bf16 bits of an exactly representable value
inline uint fp8_bf(float v) { return as_type<uint>(v) >> 16; }
inline uint fp8_pack(float lo, float hi) { return fp8_bf(lo) | (as_type<uint>(hi) & 0xFFFF0000u); }

// 16 e4m3 bytes (4 words) -> 16 bf16 (8 words)
inline void fp8_tile16(uint2 a, uint2 b, thread uint* o) {
    const uint v[4] = {a.x, a.y, b.x, b.y};
    for (int j = 0; j < 4; ++j) {
        if (fp8_word_nan(v[j])) {
            for (int e = 0; e < 4; e += 2) {
                o[2 * j + e / 2] = fp8_pack(fp8_dec((v[j] >> (8 * e)) & 0xFFu), fp8_dec((v[j] >> (8 * e + 8)) & 0xFFu));
            }
            continue;
        }
        const float2 p = fp8_x2(v[j]) * 256.0f, q = fp8_x2(v[j] >> 8) * 256.0f;
        o[2 * j] = fp8_pack(p.x, q.x);      // elements 0, 1
        o[2 * j + 1] = fp8_pack(p.y, q.y);  // elements 2, 3
    }
}

template <typename OutT>
inline void fp8_mma_impl(device const ushort* X, device const uchar* W, device const float* sc, device OutT* Y,
                         constant OpP& p, threadgroup float* smem, uint2 tgp, uint tid, uint sg, uint lane) {
    threadgroup bfloat* As = (threadgroup bfloat*)smem;
    threadgroup bfloat* Bs = As + FP8_BM * FP8_LD;
    const uint M = p.a, N = p.b, K = p.c;
    const uint n0 = tgp.x * FP8_BN, m0 = tgp.y * FP8_BM;
    const uint sm = sg / 2, sn = sg % 2;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += FP8_BK) {
        for (uint c = tid; c < FP8_BM * 4; c += 128) {  // A: 64 rows x 4 chunks of 8 bf16
            const uint r = c / 4, q = c % 4;
            const uint gm = m0 + r, gk = k0 + q * 8;
            uint4 v = uint4(0u);
            if (gm < M && gk < K) v = uint4(*(device const packed_uint4*)(X + (ulong)gm * K + gk));
            *(threadgroup uint4*)(As + r * FP8_LD + q * 8) = v;
        }
        {  // B: 64 rows x 2 chunks of 16 e4m3
            const uint r = tid / 2, q = tid % 2;
            const uint gn = n0 + r, gk = k0 + q * 16;
            uint o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (gn < N && gk < K) {
                device const uchar* wp = W + (ulong)gn * K + gk;
                fp8_tile16(*(device const uint2*)wp, *(device const uint2*)(wp + 8), o);
            }
            threadgroup uint4* bp = (threadgroup uint4*)(Bs + r * FP8_LD + q * 16);
            bp[0] = uint4(o[0], o[1], o[2], o[3]);
            bp[1] = uint4(o[4], o[5], o[6], o[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < FP8_BK / 8; ++kk) {
            simdgroup_matrix<bfloat, 8, 8> a[4], b[4];
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sm * 32 + i * 8) * FP8_LD + kk * 8, FP8_LD);
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(b[j], Bs + (sn * 32 + j * 8) * FP8_LD + kk * 8, FP8_LD, ulong2(0, 0), true);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* cs = smem + sg * 1024;
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], cs + (i * 8) * 32 + j * 8, 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float is = sc[1];
    const float scale = sc[0] * (is > 0.0f ? is : 1.0f);
    for (uint e = lane; e < 1024; e += 32) {
        const uint gm = m0 + sm * 32 + e / 32, gn = n0 + sn * 32 + e % 32;
        if (gm < M && gn < N) store_out(Y + p.s1, (ulong)gm * N + gn, cs[e] * scale);
    }
}

#define FP8_MMA(NAME, OUT)                                                                                          \
    kernel void NAME(device const ushort* X [[buffer(0)]], device const uchar* W [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* Y [[buffer(3)]],                             \
                     constant OpP& p [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],                     \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        threadgroup float4 smem[1024];                                                                              \
        fp8_mma_impl<OUT>(X, W, sc, Y, p, (threadgroup float*)smem, tgp, tid, sg, lane);                            \
    }

FP8_MMA(fp8_mma_bf16, ushort)
FP8_MMA(fp8_mma_f32, float)

// ---- weight decode for the NAX GEMM (gemm_nax_fp8.metal) ---------------------------------------------------------------
// `fp8_w_to_bf16`: W[N,K] e4m3 bytes -> bf16 [N,K] (the staged tile of `fp8_mma_impl`, `fp8_tile16`: exact, NaN codes kept
// as NaN) into the plan scratch. p.b = N, p.c = K (K % 16 == 0); one thread decodes 16 weights; bind 0 W (8-byte aligned),
// 1 Wb (16-byte aligned), 2 OpP.
kernel void fp8_w_to_bf16(device const uchar* w [[buffer(0)]], device uint4* wb [[buffer(1)]], constant OpP& p [[buffer(2)]],
                          uint g [[thread_position_in_grid]]) {
    if ((ulong)g >= (ulong)p.b * p.c / 16) return;
    device const uchar* wp = w + (ulong)g * 16;
    uint o[8];
    fp8_tile16(*(device const uint2*)wp, *(device const uint2*)(wp + 8), o);
    wb[2 * (ulong)g] = uint4(o[0], o[1], o[2], o[3]);
    wb[2 * (ulong)g + 1] = uint4(o[4], o[5], o[6], o[7]);
}
