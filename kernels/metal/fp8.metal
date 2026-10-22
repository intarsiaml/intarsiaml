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
//   fp8_gemv_{bf16,f32}_{m1,m8}
//                         decode GEMV, M <= 1 / 8 per call. A threadgroup of FP8_SG simdgroups stages its q(x) rows once in
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

// The GEMV is split in two so the fused pair kernels below can feed it staged rows of their own: `fp8_gemv_stage` fills the
// threadgroup's q(x) rows from memory, `fp8_gemv_body` (after a threadgroup barrier) runs the dot products and writes
// `out`. Scalars instead of `OpP`: the pair kernels run it once per matrix with that matrix's N / output skip.
inline void fp8_gemv_stage(device const ushort* x, ulong xskip, uint M, uint K, uint MB, float is, threadgroup ushort* xs,
                           uint m0, uint tid) {
    const uint rows = min(MB, M - m0);
    for (uint i = tid; i < rows * K; i += FP8_SG * 32) {
        const ushort v = x[xskip + (ulong)m0 * K + i];
        xs[i] = is > 0.0f ? f2bf(fp8_qx(bf2f(v), is)) : v;
    }
}

template <int MB, int RS, int UN, typename OutT>
inline void fp8_gemv_body(device const uchar* w, device const float* sc, device OutT* out, uint M, uint N, uint K,
                          ulong oskip, threadgroup ushort* xs, uint2 tg, uint sg, uint lane) {
    const uint m0 = tg.y * MB;
    const uint rows = min((uint)MB, M - m0);
    const float ws = sc[0], is = sc[1];
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
    const float scale = ws * (is > 0.0f ? is : 1.0f) * 256.0f;
    for (int m = 0; m < MB; ++m) {
        if (m >= (int)rows) break;
        for (int r = 0; r < RS; ++r) {
            const float t = simd_sum(acc[r][m]);
            if (lane == 0 && nb + r < N) store_out(out + oskip, (ulong)(m0 + m) * N + nb + r, t * scale);
        }
    }
}

template <int MB, int RS, int UN, typename OutT>
inline void fp8_gemv_impl(device const ushort* x, device const uchar* w, device const float* sc, device OutT* out,
                          constant OpP& p, threadgroup ushort* xs, uint2 tg, uint tid, uint sg, uint lane) {
    fp8_gemv_stage(x, p.s0, p.a, p.c, MB, sc[1], xs, tg.y * MB, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fp8_gemv_body<MB, RS, UN, OutT>(w, sc, out, p.a, p.b, p.c, p.s1, xs, tg, sg, lane);
}

#define FP8_GEMV(NAME, MB, RS, UN, OUT)                                                                                 \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], threadgroup ushort* xs [[threadgroup(0)]],                      \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],           \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        fp8_gemv_impl<MB, RS, UN, OUT>(x, w, sc, out, p, xs, tg, tid, sg, lane);                                       \
    }

FP8_GEMV(fp8_gemv_bf16_m1, 1, 4, 4, ushort)
FP8_GEMV(fp8_gemv_bf16_m8, 8, 2, 2, ushort)
FP8_GEMV(fp8_gemv_f32_m1, 1, 4, 4, float)
FP8_GEMV(fp8_gemv_f32_m8, 8, 2, 2, float)

// ---- fused decode GEMVs (Op::MatmulFp8Pair, Op::RmsNormAddMatmulFp8Pair) -------------------------------------------------
// Two projections of one activation in one dispatch. Threadgroups [0, T0) serve matrix 0 (T0 = ceil(N0 / (FP8_SG * RS))), the
// rest matrix 1; each stages q(x) with ITS OWN input scale and then runs exactly the single-matrix GEMV body on its rows, so
// every output element is bitwise what `fp8_gemv_*` writes for it (same row grouping, same lane-strided K order, same
// simd_sum, same final scale). The norm variant first computes RmsNormAdd (the `rmsnorm_add_bf16` arithmetic: 256 threads,
// thread-strided fma partial sums, simd_sum, serial sum of the 8 simdgroup partials) into the staged rows; threadgroup 0
// also stores `normed` and `sum`.
// Kernel arguments: Fp8PairP (a = M, b = N0, c = N1, d = K; skips in binding order of the pair kernel). Entry points:
//   fp8_pair_gemv_{bf16,f32}_{m1,m8}     x, w0, sc0, w1, sc1, out0, out1, p
//   fp8_pair_norm_gemv_bf16_m1           x, res, nw, w0, sc0, w1, sc1, normed, sum, out0, out1, p      (M = 1)

struct Fp8PairP {
    uint m, n0, n1, k;
    uint s[8];  // pair: x, out0, out1; norm pair: x, res, nw, normed, sum, out0, out1
    float eps, w_offset;
    uint pad[2];
};

template <int MB, int RS, int UN, typename OutT>
inline void fp8_pair_impl(device const ushort* x, device const uchar* w0, device const float* sc0, device const uchar* w1,
                          device const float* sc1, device OutT* out0, device OutT* out1, constant Fp8PairP& p,
                          threadgroup ushort* xs, uint2 tg, uint tid, uint sg, uint lane) {
    const uint t0 = (p.n0 + FP8_SG * RS - 1) / (FP8_SG * RS);
    const bool second = tg.x >= t0;
    device const float* sc = second ? sc1 : sc0;
    fp8_gemv_stage(x, p.s[0], p.m, p.k, MB, sc[1], xs, tg.y * MB, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fp8_gemv_body<MB, RS, UN, OutT>(second ? w1 : w0, sc, second ? out1 : out0, p.m, second ? p.n1 : p.n0, p.k,
                                    second ? p.s[2] : p.s[1], xs, uint2(second ? tg.x - t0 : tg.x, tg.y), sg, lane);
}

#define FP8_PAIR(NAME, MB, RS, UN, OUT)                                                                                 \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w0 [[buffer(1)]],                    \
                     device const float* sc0 [[buffer(2)]], device const uchar* w1 [[buffer(3)]],                   \
                     device const float* sc1 [[buffer(4)]], device OUT* out0 [[buffer(5)]],                         \
                     device OUT* out1 [[buffer(6)]], constant Fp8PairP& p [[buffer(7)]],                            \
                     threadgroup ushort* xs [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],          \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        fp8_pair_impl<MB, RS, UN, OUT>(x, w0, sc0, w1, sc1, out0, out1, p, xs, tg, tid, sg, lane);                    \
    }

FP8_PAIR(fp8_pair_gemv_bf16_m1, 1, 4, 4, ushort)
FP8_PAIR(fp8_pair_gemv_bf16_m8, 8, 2, 2, ushort)
FP8_PAIR(fp8_pair_gemv_f32_m1, 1, 4, 4, float)
FP8_PAIR(fp8_pair_gemv_f32_m8, 8, 2, 2, float)

// threadgroup-wide sum in the order of `tg_sum` (dense_ops.metal): simd_sum, then a serial sum of the partials
inline float fp8_tg_sum(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint j = 0; j < nsg; ++j) t += part[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

template <int MB, int RS, int UN>
inline void fp8_pair_norm_impl(device const ushort* x, device const ushort* res, device const ushort* nw,
                               device const uchar* w0, device const float* sc0, device const uchar* w1,
                               device const float* sc1, device ushort* normed, device ushort* sum, device ushort* out0,
                               device ushort* out1, constant Fp8PairP& p, threadgroup ushort* xs, threadgroup float* part,
                               uint2 tg, uint tid, uint sg, uint lane) {
    const uint M = p.m, h = p.k;
    const uint m0 = tg.y * MB;
    const uint rows = min((uint)MB, M - m0);
    const uint t0 = (p.n0 + FP8_SG * RS - 1) / (FP8_SG * RS);
    const bool second = tg.x >= t0;
    device const float* sc = second ? sc1 : sc0;
    const float is = sc[1];
    const bool store = tg.x == 0;
    for (uint r = 0; r < rows; ++r) {
        const ulong base = (ulong)(m0 + r) * h;
        float ss = 0.0f;
        for (uint i = tid; i < h; i += FP8_SG * 32) {
            const float s = bf2f(f2bf(bf2f(x[p.s[0] + base + i]) + bf2f(res[p.s[1] + base + i])));
            ss = fma(s, s, ss);
        }
        const float tot = fp8_tg_sum(ss, part, lane, sg, FP8_SG);
        const float rr = 1.0f / precise::sqrt(tot / (float)h + p.eps);
        for (uint i = tid; i < h; i += FP8_SG * 32) {
            const ushort sb = f2bf(bf2f(x[p.s[0] + base + i]) + bf2f(res[p.s[1] + base + i]));
            const ushort nb = f2bf(bf2f(sb) * rr * (bf2f(nw[p.s[2] + i]) + p.w_offset));
            if (store) {
                sum[p.s[4] + base + i] = sb;
                normed[p.s[3] + base + i] = nb;
            }
            xs[(ulong)r * h + i] = is > 0.0f ? f2bf(fp8_qx(bf2f(nb), is)) : nb;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fp8_gemv_body<MB, RS, UN, ushort>(second ? w1 : w0, sc, second ? out1 : out0, M, second ? p.n1 : p.n0, h,
                                      second ? p.s[6] : p.s[5], xs, uint2(second ? tg.x - t0 : tg.x, tg.y), sg, lane);
}

#define FP8_PAIR_NORM(NAME, MB, RS, UN)                                                                                 \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const ushort* res [[buffer(1)]],                  \
                     device const ushort* nw [[buffer(2)]], device const uchar* w0 [[buffer(3)]],                   \
                     device const float* sc0 [[buffer(4)]], device const uchar* w1 [[buffer(5)]],                   \
                     device const float* sc1 [[buffer(6)]], device ushort* normed [[buffer(7)]],                    \
                     device ushort* sum [[buffer(8)]], device ushort* out0 [[buffer(9)]],                           \
                     device ushort* out1 [[buffer(10)]], constant Fp8PairP& p [[buffer(11)]],                       \
                     threadgroup ushort* xs [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],         \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        threadgroup float part[32];                                                                                 \
        fp8_pair_norm_impl<MB, RS, UN>(x, res, nw, w0, sc0, w1, sc1, normed, sum, out0, out1, p, xs, part, tg, tid, sg,  \
                                       lane);                                                                       \
    }

FP8_PAIR_NORM(fp8_pair_norm_gemv_bf16_m1, 1, 4, 4)
FP8_PAIR_NORM(fp8_pair_norm_gemv_bf16_m8, 8, 2, 2)

// ---- SigmoidMulMatmulFp8: out = fp8 GEMV(bf16(a * sigmoid(g))) -------------------------------------------------------------
// The product is staged as the bf16 `sigmoid_mul_bf16` stores it (gdn.metal: f2bf(a * 1 / (1 + exp(-g)))) and then quantised
// like any other activation. Binding: a, g, w, sc, out, p. Fp8PairP: m, n0 = N, k = K, s = skips of a, g, out (bf16 only).
template <int MB, int RS, int UN>
inline void fp8_sigmoid_impl(device const ushort* a, device const ushort* g, device const uchar* w,
                             device const float* sc, device ushort* out, constant Fp8PairP& p, threadgroup ushort* xs,
                             uint2 tg, uint tid, uint sg, uint lane) {
    const uint M = p.m, K = p.k;
    const uint m0 = tg.y * MB;
    const uint rows = min((uint)MB, M - m0);
    const float is = sc[1];
    for (uint i = tid; i < rows * K; i += FP8_SG * 32) {
        const ulong e = (ulong)m0 * K + i;
        const float gv = bf2f(g[p.s[1] + e]);
        const ushort v = f2bf(bf2f(a[p.s[0] + e]) * (1.0f / (1.0f + precise::exp(-gv))));
        xs[i] = is > 0.0f ? f2bf(fp8_qx(bf2f(v), is)) : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fp8_gemv_body<MB, RS, UN, ushort>(w, sc, out, M, p.n0, K, p.s[2], xs, tg, sg, lane);
}

#define FP8_SIGMOID(NAME, MB, RS, UN)                                                                                   \
    kernel void NAME(device const ushort* a [[buffer(0)]], device const ushort* g [[buffer(1)]],                    \
                     device const uchar* w [[buffer(2)]], device const float* sc [[buffer(3)]],                     \
                     device ushort* out [[buffer(4)]], constant Fp8PairP& p [[buffer(5)]],                          \
                     threadgroup ushort* xs [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],          \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        fp8_sigmoid_impl<MB, RS, UN>(a, g, w, sc, out, p, xs, tg, tid, sg, lane);                                       \
    }

FP8_SIGMOID(fp8_sigmoid_gemv_bf16_m1, 1, 4, 4)
FP8_SIGMOID(fp8_sigmoid_gemv_bf16_m8, 8, 2, 2)

// ---- GatedRmsnormMatmulFp8 (one output row): out = fp8 GEMV(silu_glu(z, rmsnorm(x, w))) ----------------------------------
// x, z are [R, D] row blocks of the K = R * D activation (D = 128 per GatedDeltaNet value head). Every threadgroup normalises
// all R rows exactly as `rmsnorm_bf16` does a row of D <= 256 elements with D threads (thread i owns element i; fma square,
// simd_sum per 32 elements, serial sum of the D / 32 partials, 1 / precise::sqrt(mean + eps)), rounds the norm to bf16, then
// applies `silu_glu_bf16` (f2bf(z / (1 + exp(-z)) * normed)) and stages the quantised result. A simdgroup takes rows
// sg, sg + 8, ...; lane l of chunk j holds element 32 * j + l. Binding: x, z, nw, w, sc, out, p. Fp8PairP: n0 = N, n1 = D, k = K.
inline void fp8_gated_stage(device const ushort* x, device const ushort* z, device const ushort* nw, float is, constant Fp8PairP& p,
                            threadgroup ushort* xs, uint sg, uint lane) {
    const uint D = p.n1, R = p.k / D, nch = D / 32;
    for (uint r = sg; r < R; r += FP8_SG) {
        const ulong base = (ulong)r * D;
        float part[8];  // D <= 256
        for (uint j = 0; j < nch; ++j) {
            const float v = bf2f(x[p.s[0] + base + j * 32 + lane]);
            part[j] = simd_sum(fma(v, v, 0.0f));
        }
        float tot = 0.0f;
        for (uint j = 0; j < nch; ++j) tot += part[j];
        const float rr = 1.0f / precise::sqrt(tot / (float)D + p.eps);
        for (uint j = 0; j < nch; ++j) {
            const uint i = j * 32 + lane;
            const ushort nb = f2bf(bf2f(x[p.s[0] + base + i]) * rr * (bf2f(nw[p.s[2] + i]) + p.w_offset));
            const float gv = bf2f(z[p.s[1] + base + i]);
            const ushort v = f2bf(gv / (1.0f + precise::exp(-gv)) * bf2f(nb));
            xs[base + i] = is > 0.0f ? f2bf(fp8_qx(bf2f(v), is)) : v;
        }
    }
}

kernel void fp8_gated_norm_gemv_bf16_m1(device const ushort* x [[buffer(0)]], device const ushort* z [[buffer(1)]],
                                        device const ushort* nw [[buffer(2)]], device const uchar* w [[buffer(3)]],
                                        device const float* sc [[buffer(4)]], device ushort* out [[buffer(5)]],
                                        constant Fp8PairP& p [[buffer(6)]], threadgroup ushort* xs [[threadgroup(0)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    fp8_gated_stage(x, z, nw, sc[1], p, xs, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fp8_gemv_body<1, 4, 4, ushort>(w, sc, out, 1, p.n0, p.k, p.s[3], xs, tg, sg, lane);
}

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
