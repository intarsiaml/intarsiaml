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

// two e4m3 bytes at bits 0..7 and 16..23 of t (the other bits are ignored) -> value / 256
inline float2 fp8_x2(uint t) {
    const uint h = ((t & 0x007F007Fu) << 7) | ((t & 0x00800080u) << 8);
    return float2(as_type<half2>(h));
}

// a word holds a NaN code (0x7F / 0xFF) in some byte
inline bool fp8_word_nan(uint w) {
    const uint t = (w & 0x7F7F7F7Fu) ^ 0x7F7F7F7Fu;  // a zero byte <=> NaN code
    return ((t - 0x01010101u) & ~t & 0x80808080u) != 0u;
}

// exact value of an e4m3 code
inline float fp8_dec(uint b) {
    if ((b & 0x7Fu) == 0x7Fu) return as_type<float>(0x7FC00000u);
    return fp8_x2(b).x * 256.0f;
}

// intarsia_core::dtype::f32_to_fp8_e4m3: round to nearest, ties to the even code, saturating at +-448, NaN -> 0x7F
inline uint fp8_enc(float v) {
    const uint s = (as_type<uint>(v) >> 24) & 0x80u;
    const float a = fabs(v);
    if (!(a == a)) return 0x7Fu;
    if (a >= 448.0f) return s | 0x7Eu;
    uint code;
    if (a < 0.015625f) {
        code = (uint)rint(a * 512.0f);  // subnormal grid 2^-9, rint = ties to even
    } else {
        uint b = as_type<uint>(a);  // 3 mantissa bits kept: round to nearest even on bit 20
        b += 0x7FFFFu + ((b >> 20) & 1u);
        code = (b >> 20) - 960u;  // re-bias the exponent: (127 - 7) << 3
    }
    return s | code;
}

// q(x) for one element: static W8A8 (is > 0) or the activation itself
inline float fp8_qx(float x, float is) {
    return is > 0.0f ? fp8_dec(fp8_enc(precise::divide(x, is))) : x;
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

// the q(x) prep as f32 for the `_xf` GEMVs: the float the bf16 forms unpack (bf16 -> f32 is exact), so the same values
kernel void fp8_xprep_f32(device const ushort* x [[buffer(0)]], device const float* sc [[buffer(1)]],
                          device float* xs [[buffer(2)]], constant OpP& p [[buffer(3)]],
                          uint g [[thread_position_in_grid]]) {
    const ulong total = (ulong)p.a * p.c;
    const ulong e0 = (ulong)g * 8;
    if (e0 >= total) return;
    const float is = sc[1];
    for (int e = 0; e < 8; ++e) {
        const ushort v = x[p.s0 + e0 + e];
        xs[e0 + e] = bf2f(is > 0.0f ? f2bf(fp8_qx(bf2f(v), is)) : v);
    }
}

// ---- decode GEMV ------------------------------------------------------------------------------------------------------

constant constexpr uint FP8_SG = 8;  // simdgroups per threadgroup

// XD (the `_xd` multi-row forms): x is q(x) already, [M, K] bf16 written once per call by fp8_xprep_bf16 (the same
// expression the staging loop below evaluates), and is read straight from device memory: no threadgroup staging, so no
// per-threadgroup q(x) of all the rows and no 32 KB of threadgroup memory capping the threadgroups per core. Per row the
// values and the fma chain are those of the staged forms, so the bits are the one-row kernel's.
template <int MB, int RS, int UN, typename OutT, bool XD = false, bool NEG = false, bool XF = false, bool SW = false>
inline void fp8_gemv_impl(device const ushort* x, device const uchar* w, device const float* sc, device OutT* out,
                          constant OpP& p, threadgroup ushort* xs, uint2 tg, uint tid, uint sg, uint lane) {
    const uint M = p.a, N = p.b, K = p.c;
    // SW: the row tile is tg.x (fastest), so the tiles that read the same weight rows run next to each other
    const uint m0 = (SW ? tg.x : tg.y) * MB;
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
    // XD: q(x) rows of this threadgroup (the prep buffer has no skip); XF: f32 rows, so the offset is in floats
    device const ushort* xd = XF ? (device const ushort*)((device const float*)x + (ulong)m0 * K) : x + (ulong)m0 * K;

    const uint nb = ((SW ? tg.y : tg.x) * FP8_SG + sg) * RS;
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
                    float xf[16];
                    if constexpr (XD && XF) {
                        // f32 q(x) rows (fp8_xprep_f32): the floats the unpack below produces, loaded as they are
                        device const float4* xp = (device const float4*)((device const float*)xd + (ulong)m * K + (ulong)u * 16);
                        for (int q = 0; q < 4; ++q) {
                            const float4 v = xp[q];
                            xf[4 * q] = v.x;
                            xf[4 * q + 1] = v.y;
                            xf[4 * q + 2] = v.z;
                            xf[4 * q + 3] = v.w;
                        }
                    } else {
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
                    fp8_unpack2(x0.x, xf);
                    fp8_unpack2(x0.y, xf + 2);
                    fp8_unpack2(x0.z, xf + 4);
                    fp8_unpack2(x0.w, xf + 6);
                    fp8_unpack2(x1.x, xf + 8);
                    fp8_unpack2(x1.y, xf + 10);
                    fp8_unpack2(x1.z, xf + 12);
                    fp8_unpack2(x1.w, xf + 14);
                    }
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
                    threadgroup const uint4* xp = (threadgroup const uint4*)(xs + (ulong)m * K + (ulong)u * 16);
                    float xf[16];
                    const uint4 x0 = xp[0], x1 = xp[1];
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
#define FP8_GEMV_XFN(NAME, MB, RS, UN, OUT, NEG)                                                                              \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        fp8_gemv_impl<MB, RS, UN, OUT, true, NEG, true>(x, w, sc, out, p, nullptr, tg, tid, sg, lane);                            \
    }
#define FP8_GEMV_XFS(NAME, MB, RS, UN, OUT, NEG, XF)                                                                              \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        fp8_gemv_impl<MB, RS, UN, OUT, true, NEG, XF, true>(x, w, sc, out, p, nullptr, tg, tid, sg, lane);                            \
    }

FP8_GEMV(fp8_gemv_bf16_m1, 1, 4, 4, ushort)
FP8_GEMV(fp8_gemv_bf16_m8, 8, 2, 2, ushort)
FP8_GEMV(fp8_gemv_f32_m1, 1, 4, 4, float)
FP8_GEMV(fp8_gemv_f32_m8, 8, 2, 2, float)
// _m4: the multi-row form for K up to 4096 (4 rows of q(x) fill the 32 KB of threadgroup memory there, 8 rows only up to
// K = 2048); `matmul_fp8_gemv_rows` takes it past M = 4 at K = 4096 (the 35B's GDN / attention output projections)
FP8_GEMV(fp8_gemv_bf16_m4, 4, 2, 2, ushort)
FP8_GEMV(fp8_gemv_f32_m4, 4, 2, 2, float)
// `matmul_fp8_gemv_rows` over the q(x) prep buffer (lane mm, docs/results/2026-10-25-mm-metal-moe-grouped.md): 4 rows per
// threadgroup (2..8 rows: one or two threadgroup rows; with no staging the tile is free of the threadgroup-memory limit, and
// 4 measured faster than 8 or 2 at every 35B shape); _neg is the negative control (INTARSIA_METAL_NEGCTL=fp8_xd)
FP8_GEMV_XD(fp8_gemv_bf16_m4_xd, 4, 2, 2, ushort)
FP8_GEMV_XD(fp8_gemv_f32_m4_xd, 4, 2, 2, float)
FP8_GEMV_XDN(fp8_gemv_bf16_m4_xd_neg, 4, 2, 2, ushort, true)
FP8_GEMV_XDN(fp8_gemv_f32_m4_xd_neg, 4, 2, 2, float, true)
// ---- multi-row GEMV over f32 q(x), word-interleaved (lane mm step 2) --------------------------------------------------
// The same values and, per (row, output), the same fma chain as the one-row kernel: per 16-byte block the 16-term chain
// from 0 in element order (fp8_dot16), the NaN-code test of the block (exact: a byte's low 7 bits are 0x7F iff adding 1
// sets its bit 7, no carry leaves the byte), `acc += chain` over the lane's blocks in ascending order, simd_sum, scale.
// What changes is only the schedule: each weight word (4 elements) is decoded once for all MB rows and applied to every
// row before the next word, so a lane holds 4 decoded weights per weight row instead of 16 and the tile can cover all 8
// rows (one decode per weight per call); x comes as f32 from fp8_xprep_f32 (float4 loads, no unpack).
inline bool fp8_block_nan(uint4 q) {
    const uint m = 0x7F7F7F7Fu, o = 0x01010101u;
    return ((((q.x & m) + o) | ((q.y & m) + o) | ((q.z & m) + o) | ((q.w & m) + o)) & 0x80808080u) != 0u;
}

template <int MB, int RS, typename OutT, bool NEG>
inline void fp8_rows2_impl(device const float* x, device const uchar* w, device const float* sc, device OutT* out,
                           constant OpP& p, uint2 tg, uint sg, uint lane) {
    const uint M = p.a, N = p.b, K = p.c;
    const uint m0 = tg.y * MB;
    const uint rows = min((uint)MB, M - m0);
    const float ws = sc[0], is = sc[1];
    const uint nb = (tg.x * FP8_SG + sg) * RS;
    if (nb >= N) return;  // whole simdgroup exits together
    device const uchar* wr[RS];
    for (int r = 0; r < RS; ++r) wr[r] = w + (ulong)min(nb + r, N - 1) * K;
    device const float* xr = x + (ulong)m0 * K;
    float acc[RS][MB];
    for (int r = 0; r < RS; ++r)
        for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
    const uint nch = K / 16;
    for (uint u = lane; u < nch; u += 32) {
        uint4 wq[RS];
        bool nan[RS];
        for (int r = 0; r < RS; ++r) {
            const uint2 a = *(device const uint2*)(wr[r] + (ulong)u * 16), b = *(device const uint2*)(wr[r] + (ulong)u * 16 + 8);
            wq[r] = uint4(a.x, a.y, b.x, b.y);
            nan[r] = fp8_block_nan(wq[r]);
        }
        float t[RS][MB];
        for (int r = 0; r < RS; ++r)
            for (int m = 0; m < MB; ++m) t[r][m] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            float wv[RS][4];
            for (int r = 0; r < RS; ++r) {
                const uint v = wq[r][j];
                const float2 pp = fp8_x2(v), qq = fp8_x2(v >> 8);  // elements (0, 2) and (1, 3) of the word
                wv[r][0] = pp.x;
                wv[r][1] = qq.x;
                wv[r][2] = pp.y;
                wv[r][3] = qq.y;
            }
            for (int m = 0; m < MB; ++m) {
                if (m >= (int)rows) break;
                const float4 xv = *(device const float4*)(xr + (ulong)m * K + (ulong)u * 16 + 4 * j);
                for (int r = 0; r < RS; ++r) {
                    t[r][m] = fma(wv[r][0], xv.x, t[r][m]);
                    t[r][m] = fma(wv[r][1], xv.y, t[r][m]);
                    t[r][m] = fma(wv[r][2], xv.z, t[r][m]);
                    t[r][m] = fma(wv[r][3], xv.w, t[r][m]);
                }
            }
        }
        for (int r = 0; r < RS; ++r)
            for (int m = 0; m < MB; ++m) {
                const float d = nan[r] ? as_type<float>(0x7FC00000u) : t[r][m];
                // NEG (negative control only): another association on odd blocks, which the row gate must catch
                if (NEG && (u & 1u)) acc[r][m] = (acc[r][m] + 0.5f * d) + 0.5f * d;
                else acc[r][m] += d;
            }
    }
    const float scale = ws * (is > 0.0f ? is : 1.0f) * 256.0f;
    for (int m = 0; m < MB; ++m) {
        if (m >= (int)rows) break;
        for (int r = 0; r < RS; ++r) {
            const float tt = simd_sum(acc[r][m]);
            if (lane == 0 && nb + r < N) store_out(out + p.s1, (ulong)(m0 + m) * N + nb + r, tt * scale);
        }
    }
}

#define FP8_ROWS2(NAME, MB, RS, OUT, NEG)                                                                           \
    kernel void NAME(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                      \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        fp8_rows2_impl<MB, RS, OUT, NEG>(x, w, sc, out, p, tg, sg, lane);                                           \
    }

FP8_ROWS2(fp8r2_bf16_m4_r2, 4, 2, ushort, false)
FP8_ROWS2(fp8r2_f32_m4_r2, 4, 2, float, false)
FP8_ROWS2(fp8r2_bf16_m8_r2, 8, 2, ushort, false)
FP8_ROWS2(fp8r2_f32_m8_r2, 8, 2, float, false)
FP8_ROWS2(fp8r2_bf16_m8_r1, 8, 1, ushort, false)
FP8_ROWS2(fp8r2_f32_m8_r1, 8, 1, float, false)
FP8_ROWS2(fp8r2_bf16_m4_r4, 4, 4, ushort, false)
FP8_ROWS2(fp8r2_f32_m4_r4, 4, 4, float, false)
FP8_ROWS2(fp8r2_bf16_m8_r4, 8, 4, ushort, false)
FP8_ROWS2(fp8r2_f32_m8_r4, 8, 4, float, false)
FP8_ROWS2(fp8r2_bf16_m4_r1, 4, 1, ushort, false)
FP8_ROWS2(fp8r2_f32_m4_r1, 4, 1, float, false)

// swapped grid (INTARSIA_METAL_FP8_SW=MB,UN,XF)
FP8_GEMV_XFS(fp8s_m2_u1_x0, 2, 2, 1, ushort, false, false)
FP8_GEMV_XFS(fp8s_m2_u1_x1, 2, 2, 1, ushort, false, true)
FP8_GEMV_XFS(fp8s_m2_u2_x0, 2, 2, 2, ushort, false, false)
FP8_GEMV_XFS(fp8s_m2_u2_x1, 2, 2, 2, ushort, false, true)
FP8_GEMV_XFS(fp8s_m4_u1_x0, 4, 2, 1, ushort, false, false)
FP8_GEMV_XFS(fp8s_m4_u1_x1, 4, 2, 1, ushort, false, true)
FP8_GEMV_XFS(fp8s_m4_u2_x0, 4, 2, 2, ushort, false, false)
FP8_GEMV_XFS(fp8s_m4_u2_x1, 4, 2, 2, ushort, false, true)
FP8_GEMV_XFS(fp8s_m8_u1_x0, 8, 2, 1, ushort, false, false)
FP8_GEMV_XFS(fp8s_m8_u1_x1, 8, 2, 1, ushort, false, true)
FP8_GEMV_XFS(fp8s_m8_u2_x0, 8, 2, 2, ushort, false, false)
FP8_GEMV_XFS(fp8s_m8_u2_x1, 8, 2, 2, ushort, false, true)
// ---- multi-row GEMV on simdgroup matrices, row-exact (lane mm step 2) --------------------------------------------------
// Measured on endor (M5 Max, .mm/mmaprobe): simdgroup_multiply_accumulate on float 8x8 equals, bit for bit, the chain
// c = fma(a[m][k], b[k][n], c) for k = 0..7 from c (subnormals, inf, NaN, cancellation included), and simd_sum equals
// the xor butterfly 1, 2, 4, 8, 16, i.e. the pairwise tree over lanes 0..31 in index order. So the one-row kernel's value
// for (row m, output n) can be rebuilt from matrices: lane l of the one-row kernel owns blocks u = l, l + 32, ... and
// sums their 16-term chains in that order; here block u's chain for 8 rows x 8 outputs is two MMAs from c = 0 (k = 0..7,
// then 8..15), a block whose weight row holds a NaN code is set to NaN (as fp8_dot16 does), and it is added to the
// accumulator of class u mod 32. The 8 simdgroups of a threadgroup take classes 4s..4s+3 (so a step reads 32 consecutive
// blocks per weight row, as the one-row kernel does), each folds its 4 classes ((0+1)+(2+3)), and the threadgroup folds
// the 8 results ((0+1)+(2+3))+((4+5)+(6+7)): the pairwise tree of simd_sum. Then the same scale and rounding.
// Layout of a lane's 2 elements of an 8x8 matrix (measured): row (lane & 16 ? 4 : 0) + ((lane >> 1) & 3), columns
// ((lane >> 3) & 1) * 4 + (lane & 1) * 2 and +1. x: f32 q(x) rows (fp8_xprep_f32). Threadgroup: 8 outputs, all M <= 8 rows.
// NEG (negative control only): the two MMAs of a block in the other order (k = 8..15 first).
template <typename OutT, bool NEG, int T, int ABL = 0>
inline void fp8_rows_mma_impl(device const float* x, device const uchar* w, device const float* sc, device OutT* out,
                              constant OpP& p, threadgroup float* red, uint tg, uint sg, uint lane, uint tid) {
    const uint M = p.a, N = p.b, K = p.c;
    const uint n0 = tg * 8 * T;
    const uint fm = ((lane & 16u) ? 4u : 0u) + ((lane >> 1) & 3u);
    const uint fn = ((lane >> 3) & 1u) * 4u + (lane & 1u) * 2u;
    device const uchar* w0[T];
    device const uchar* w1[T];
    for (int t = 0; t < T; ++t) {
        w0[t] = w + (ulong)min(n0 + 8 * t + fn, N - 1) * K;
        w1[t] = w + (ulong)min(n0 + 8 * t + fn + 1, N - 1) * K;
    }
    const bool xrow = fm < M;
    device const float* xr = x + (ulong)min(fm, M - 1) * K;
    simdgroup_float8x8 acc[T][4];
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 4; ++c) acc[t][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint nch = K / 16;
    for (uint ub = 4 * sg; ub < nch; ub += 32) {
        uint flags = 0;  // bit 8t + 2c (+1): the block of class c, tile t, output fn (fn + 1) holds a NaN code
        float2 bw0[T][4], bw1[T][4];
        for (int c = 0; c < 4; ++c) {
            const ulong u = min(ub + c, nch - 1);
            for (int t = 0; t < T; ++t) {
                const uint a0 = w0[t][u * 16 + fm], a1 = w0[t][u * 16 + 8 + fm];
                const uint b0 = w1[t][u * 16 + fm], b1 = w1[t][u * 16 + 8 + fm];
                bw0[t][c] = fp8_x2(a0 | (a1 << 16));  // (k = fm, n = fn) of the two halves
                bw1[t][c] = fp8_x2(b0 | (b1 << 16));  // (k = fm, n = fn + 1)
                flags |= ((((a0 & 0x7Fu) == 0x7Fu) || ((a1 & 0x7Fu) == 0x7Fu)) ? 1u : 0u) << (8 * t + 2 * c);
                flags |= ((((b0 & 0x7Fu) == 0x7Fu) || ((b1 & 0x7Fu) == 0x7Fu)) ? 2u : 0u) << (8 * t + 2 * c);
            }
        }
        // the 16 bytes of a block row lie in the 8 lanes that differ in lane bits 1, 2 and 4
        if (ABL != 2) {
        flags |= simd_shuffle_xor(flags, 2);
        flags |= simd_shuffle_xor(flags, 4);
        flags |= simd_shuffle_xor(flags, 16);
        }
        for (int c = 0; c < 4; ++c) {
            const uint u = ub + c;
            if (u >= nch) break;  // uniform over the simdgroup
            simdgroup_float8x8 a1m, a2m;
            const float2 xa = ABL == 3 ? float2(xr[0], xr[1]) : xrow ? *(device const float2*)(xr + (ulong)u * 16 + fn) : float2(0.0f);
            const float2 xb = ABL == 3 ? float2(xr[2], xr[3]) : xrow ? *(device const float2*)(xr + (ulong)u * 16 + 8 + fn) : float2(0.0f);
            a1m.thread_elements()[0] = xa.x;
            a1m.thread_elements()[1] = xa.y;
            a2m.thread_elements()[0] = xb.x;
            a2m.thread_elements()[1] = xb.y;
            for (int t = 0; t < T; ++t) {
                simdgroup_float8x8 b1m, b2m;
                b1m.thread_elements()[0] = bw0[t][c].x;
                b1m.thread_elements()[1] = bw1[t][c].x;
                b2m.thread_elements()[0] = bw0[t][c].y;
                b2m.thread_elements()[1] = bw1[t][c].y;
                simdgroup_float8x8 r = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                if (ABL == 1) {
                    r.thread_elements()[0] = bw0[t][c].x * xa.x + bw0[t][c].y * xb.y;
                    r.thread_elements()[1] = bw1[t][c].x * xa.y + bw1[t][c].y * xb.x;
                } else if (NEG) {
                    simdgroup_multiply_accumulate(r, a2m, b2m, r);
                    simdgroup_multiply_accumulate(r, a1m, b1m, r);
                } else {
                    simdgroup_multiply_accumulate(r, a1m, b1m, r);
                    simdgroup_multiply_accumulate(r, a2m, b2m, r);
                }
                float2 tv = float2(r.thread_elements()[0], r.thread_elements()[1]);
                if (flags & (1u << (8 * t + 2 * c))) tv.x = as_type<float>(0x7FC00000u);
                if (flags & (2u << (8 * t + 2 * c))) tv.y = as_type<float>(0x7FC00000u);
                acc[t][c].thread_elements()[0] += tv.x;
                acc[t][c].thread_elements()[1] += tv.y;
            }
        }
    }
    // the pairwise tree of simd_sum: classes 4s..4s+3 here, then the 8 simdgroups
    for (int t = 0; t < T; ++t) {
        const float qx = (acc[t][0].thread_elements()[0] + acc[t][1].thread_elements()[0]) +
                         (acc[t][2].thread_elements()[0] + acc[t][3].thread_elements()[0]);
        const float qy = (acc[t][0].thread_elements()[1] + acc[t][1].thread_elements()[1]) +
                         (acc[t][2].thread_elements()[1] + acc[t][3].thread_elements()[1]);
        red[(t * 8 + sg) * 64 + fm * 8 + fn] = qx;
        red[(t * 8 + sg) * 64 + fm * 8 + fn + 1] = qy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float ws = sc[0], is = sc[1];
    const float scale = ws * (is > 0.0f ? is : 1.0f) * 256.0f;
    for (uint e = tid; e < 64 * T; e += 256) {
        const uint t = e / 64, i = e % 64;
        const uint m = i / 8, n = n0 + 8 * t + i % 8;
        threadgroup const float* rt = red + t * 512 + i;
        const float r = ((rt[0] + rt[64]) + (rt[128] + rt[192])) + ((rt[256] + rt[320]) + (rt[384] + rt[448]));
        if (m < M && n < N) store_out(out + p.s1, (ulong)m * N + n, r * scale);
    }
}

#define FP8_ROWS_MMA(NAME, OUT, NEG, T) FP8_ROWS_MMA_A(NAME, OUT, NEG, T, 0)
#define FP8_ROWS_MMA_A(NAME, OUT, NEG, T, ABL)                                                                      \
    kernel void NAME(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],                       \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                     uint tid [[thread_index_in_threadgroup]]) {                                                    \
        threadgroup float red[T * 8 * 64];                                                                          \
        fp8_rows_mma_impl<OUT, NEG, T, ABL>(x, w, sc, out, p, red, tg, sg, lane, tid);                                   \
    }

FP8_ROWS_MMA(fp8_rows_mma_bf16, ushort, false, 1)
FP8_ROWS_MMA(fp8_rows_mma_f32, float, false, 1)
FP8_ROWS_MMA(fp8_rows_mma_bf16_neg, ushort, true, 1)
FP8_ROWS_MMA(fp8_rows_mma_f32_neg, float, true, 1)
FP8_ROWS_MMA(fp8_rows_mma_bf16_t2, ushort, false, 2)
FP8_ROWS_MMA(fp8_rows_mma_f32_t2, float, false, 2)
FP8_ROWS_MMA(fp8_rows_mma_bf16_neg_t2, ushort, true, 2)
FP8_ROWS_MMA(fp8_rows_mma_f32_neg_t2, float, true, 2)
FP8_ROWS_MMA(fp8_rows_mma_bf16_t4, ushort, false, 4)
FP8_ROWS_MMA(fp8_rows_mma_f32_t4, float, false, 4)
FP8_ROWS_MMA(fp8_rows_mma_bf16_neg_t4, ushort, true, 4)
FP8_ROWS_MMA(fp8_rows_mma_f32_neg_t4, float, true, 4)
FP8_ROWS_MMA_A(fp8_rows_mma_bf16_a1, ushort, false, 1, 1)
FP8_ROWS_MMA_A(fp8_rows_mma_bf16_a2, ushort, false, 1, 2)
FP8_ROWS_MMA_A(fp8_rows_mma_bf16_a3, ushort, false, 1, 3)

// ---- multi-row GEMV on simdgroup matrices, weights as the left operand (lane mm step 2) ------------------------------
// The transposed form of fp8_rows_mma: C'(n, m) = sum_k W(n, k) q(x)(m, k), so a lane's weight operand is two adjacent
// bytes of one weight row (one ushort load) and its activation operand two adjacent rows of one k of the prepared,
// block-transposed q(x): xt[u][k][m] (bf16, MP rows, padded with zeros), written by fp8_xprep_t. Each element is the
// fma chain over k = 0..15 from 0 (fma commutes in its multiplicands, the measured MMA chain is per element), so the
// per-block values, the 32 classes and the pairwise fold are those of fp8_rows_mma, i.e. of the one-row kernel.
// MP = 8 or 16 rows (one or two activation tiles per weight tile: the weights are read once for all rows).
// NEG (negative control only): the two MMAs of a block in the other order.
// the e4m3-byte q(x) layout (static W8A8 only) measured slower than bf16 on endor (decode ALU > the traffic it saves): off
constant constexpr bool MMT_XB = false;

kernel void fp8_xprep_t(device const ushort* x [[buffer(0)]], device const float* sc [[buffer(1)]],
                        device uchar* xt [[buffer(2)]], constant OpP& p [[buffer(3)]], uint g [[thread_position_in_grid]]) {
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
            // static W8A8: the e4m3 code of q(x) (its value is exactly fp8_qx's); W8A16: the bf16 activation
            b = MMT_XB && is > 0.0f ? fp8_enc(precise::divide(bf2f(xv), is)) : is > 0.0f ? f2bf(fp8_qx(bf2f(xv), is)) : xv;
        }
        v[i] = b;
    }
    if (MMT_XB && is > 0.0f) ((device ushort*)xt)[g] = ushort(v[0] | (v[1] << 8));
    else ((device uint*)xt)[g] = v[0] | (v[1] << 16);
}

// q(x) e4m3 code -> its value (fp8_dec)
inline float fp8_xv(uint b, float v256) { return (b & 0x7Fu) == 0x7Fu ? as_type<float>(0x7FC00000u) : v256 * 256.0f; }

template <typename OutT, bool NEG, int R, bool XB>
inline void fp8_rows_mmt_impl(device const uchar* xt, device const uchar* w, device const float* sc, device OutT* out,
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
                if (XB) {
                    device const uchar* xb = xt + (ulong)u * 16 * MP + fm * MP + fn + 8 * r;
                    const uint v1 = *(device const ushort*)xb, v2 = *(device const ushort*)(xb + 8 * MP);
                    const uint word = v1 | (v2 << 16);
                    const float2 pp = fp8_x2(word), qq = fp8_x2(word >> 8);
                    b1m.thread_elements()[0] = fp8_xv(v1, pp.x);  // (k = fm, m = fn), (k = fm, m = fn + 1)
                    b1m.thread_elements()[1] = fp8_xv(v1 >> 8, qq.x);
                    b2m.thread_elements()[0] = fp8_xv(v2, pp.y);  // k = 8 + fm
                    b2m.thread_elements()[1] = fp8_xv(v2 >> 8, qq.y);
                } else {
                    device const ushort* xb = (device const ushort*)xt + (ulong)u * 16 * MP + fm * MP + fn + 8 * r;
                    const uint x1 = *(device const uint*)xb, x2 = *(device const uint*)(xb + 8 * MP);
                    b1m.thread_elements()[0] = as_type<float>(x1 << 16);  // (k = fm, m = fn), (k = fm, m = fn + 1)
                    b1m.thread_elements()[1] = as_type<float>(x1 & 0xFFFF0000u);
                    b2m.thread_elements()[0] = as_type<float>(x2 << 16);  // k = 8 + fm
                    b2m.thread_elements()[1] = as_type<float>(x2 & 0xFFFF0000u);
                }
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
    kernel void NAME(device const uchar* xt [[buffer(0)]], device const uchar* w [[buffer(1)]],                    \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],                       \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                     uint tid [[thread_index_in_threadgroup]]) {                                                    \
        threadgroup float red[R * 8 * 64];                                                                          \
        fp8_rows_mmt_impl<OUT, NEG, R, MMT_XB>(xt, w, sc, out, p, red, tg, sg, lane, tid);                         \
    }

FP8_ROWS_MMT(fp8_rows_mmt_bf16, ushort, false, 1)
FP8_ROWS_MMT(fp8_rows_mmt_f32, float, false, 1)
FP8_ROWS_MMT(fp8_rows_mmt_bf16_neg, ushort, true, 1)
FP8_ROWS_MMT(fp8_rows_mmt_f32_neg, float, true, 1)
FP8_ROWS_MMT(fp8_rows_mmt_bf16_r2, ushort, false, 2)
FP8_ROWS_MMT(fp8_rows_mmt_f32_r2, float, false, 2)
FP8_ROWS_MMT(fp8_rows_mmt_bf16_neg_r2, ushort, true, 2)
FP8_ROWS_MMT(fp8_rows_mmt_f32_neg_r2, float, true, 2)

// lane mm step 2 experiments (INTARSIA_METAL_FP8_XD_EXP=MB,RS,XF[,UN]): bf16 out
FP8_GEMV_XDN(fp8x_m4_r1_u2_h, 4, 1, 2, ushort, false)
FP8_GEMV_XFN(fp8x_m4_r1_u2_f, 4, 1, 2, ushort, false)
FP8_GEMV_XDN(fp8x_m4_r2_u2_h, 4, 2, 2, ushort, false)
FP8_GEMV_XFN(fp8x_m4_r2_u2_f, 4, 2, 2, ushort, false)
FP8_GEMV_XDN(fp8x_m4_r4_u2_h, 4, 4, 2, ushort, false)
FP8_GEMV_XFN(fp8x_m4_r4_u2_f, 4, 4, 2, ushort, false)
FP8_GEMV_XDN(fp8x_m8_r1_u2_h, 8, 1, 2, ushort, false)
FP8_GEMV_XFN(fp8x_m8_r1_u2_f, 8, 1, 2, ushort, false)
FP8_GEMV_XDN(fp8x_m8_r2_u2_h, 8, 2, 2, ushort, false)
FP8_GEMV_XFN(fp8x_m8_r2_u2_f, 8, 2, 2, ushort, false)
FP8_GEMV_XDN(fp8x_m8_r4_u2_h, 8, 4, 2, ushort, false)
FP8_GEMV_XFN(fp8x_m8_r4_u2_f, 8, 4, 2, ushort, false)
FP8_GEMV_XDN(fp8x_m8_r2_u4_h, 8, 2, 4, ushort, false)
FP8_GEMV_XFN(fp8x_m8_r2_u4_f, 8, 2, 4, ushort, false)
FP8_GEMV_XDN(fp8x_m8_r1_u4_h, 8, 1, 4, ushort, false)
FP8_GEMV_XFN(fp8x_m8_r1_u4_f, 8, 1, 4, ushort, false)
FP8_GEMV_XDN(fp8x_m8_r1_u8_h, 8, 1, 8, ushort, false)
FP8_GEMV_XFN(fp8x_m8_r1_u8_f, 8, 1, 8, ushort, false)
FP8_GEMV_XDN(fp8x_m4_r2_u4_h, 4, 2, 4, ushort, false)
FP8_GEMV_XFN(fp8x_m4_r2_u4_f, 4, 2, 4, ushort, false)
FP8_GEMV_XDN(fp8x_m4_r2_u1_h, 4, 2, 1, ushort, false)
FP8_GEMV_XFN(fp8x_m4_r2_u1_f, 4, 2, 1, ushort, false)
FP8_GEMV_XDN(fp8x_m8_r2_u1_h, 8, 2, 1, ushort, false)
FP8_GEMV_XFN(fp8x_m8_r2_u1_f, 8, 2, 1, ushort, false)
FP8_GEMV_XDN(fp8x_m4_r1_u4_h, 4, 1, 4, ushort, false)
FP8_GEMV_XFN(fp8x_m4_r1_u4_f, 4, 1, 4, ushort, false)

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
