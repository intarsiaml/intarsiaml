// matmul_fp8: out[M,N] = q(x)[M,K] @ W^T * (weight_scale * input_scale), W[n,k] OCP e4m3 bytes (DType::Fp8E4m3).
// Port of kernels/hip/fp8.hip (same semantics as crates/intarsia-core/src/quant/reference.rs matmul_fp8);
// launcher crates/intarsia-backend-metal/src/fp8.rs.
//   x      bf16 [M,K]
//   w      u8   [N,K]   e4m3, bound 4-byte aligned; the launcher requires an 8-byte aligned address (rows are K % 16 == 0 bytes)
//   scales f32  [2]     (weight_scale, input_scale), read here on the GPU. input_scale > 0: W8A8 static, q(x) = e4m3(x / is)
//                       (round to nearest even, saturating at 448, NaN stays NaN, `fp8_qx`), result * ws * is; else W8A16, q(x) = x.
// e4m3 -> float: the 7 magnitude bits are placed in the exponent / mantissa of a half (<< 7), which reads as value / 256
// (subnormals included); the factor 256 is folded into the final scale. Codes 0x7F / 0xFF (NaN) are caught per word.
// Entry points:
//   fp8_xprep_bf16        xs[M,K] (bf16 scratch) = QDQ(x) (exact in bf16) or x; one thread per 8 elements (GEMM only)
//   fp8_gv_{bf16,f32}_s<SG>r<R>
//                         decode GEMV, M = 1, split-K: a threadgroup of SG simdgroups owns R output rows and streams them
//                         together, one 8-byte weight load per row and chunk, the thread quantises only its 8 activations
//                         (no staging, no barrier before the loads), f32 accumulate, simd_sum, fixed-order combine.
//   fp8_gemv_{bf16,f32}_m8
//                         decode GEMV, M <= 8 per call. A threadgroup of FP8_SG simdgroups stages its q(x) rows once in
//                         threadgroup memory (bf16, launcher-sized M * K * 2 bytes), each simdgroup owns 2 output
//                         rows (a lane takes 16-element chunks of K, UN of them loaded before any is used, 2 x 8-byte
//                         loads per row and chunk), f32 accumulate, simd_sum.
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

// 1/is for the quantiser below; 0 (-> a true division per element) when is is outside the range where its correction is safe
inline float fp8_rcp(float is) { return (is >= 8.6736174e-19f && is <= 1.1529215e18f) ? precise::divide(1.0f, is) : 0.0f; }

// q(x) for one element: static W8A8 (is > 0, r = fp8_rcp(is)) or the activation itself. The value of
// intarsia_core::dtype::f32_to_fp8_e4m3(x / is) (round to nearest, ties to the even code, saturating at +-448, NaN stays
// NaN), computed on the value instead of the code: the quotient is x * r corrected by one fma (the correctly rounded
// f32 quotient; a result >= 1024 saturates whatever its last bit), then rounded to 4 significant bits (3 stored mantissa
// bits) by an add on the bit pattern, or to the 2^-9 grid below 2^-6 by the add-and-subtract trick.
inline float fp8_qx(float x, float is, float r) {
    if (!(is > 0.0f)) return x;
    float q;
    if (r != 0.0f) {
        const float q0 = x * r;
        q = fabs(q0) < 1024.0f ? fma(fma(-q0, is, x), r, q0) : q0;
    } else {
        q = precise::divide(x, is);
    }
    const float a = fabs(q);
    if (!(a == a)) return a;
    float v;
    if (a < 0.015625f) {
        v = (a + 16384.0f) - 16384.0f;  // ulp(2^14) = 2^-9, ties to even
    } else {
        uint b = as_type<uint>(a);
        b += 0x7FFFFu + ((b >> 20) & 1u);
        v = as_type<float>(b & 0xFFF00000u);
    }
    if (a >= 448.0f) v = 448.0f;
    return copysign(v, q);
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
    const float is = sc[1], r = fp8_rcp(is);
    for (int e = 0; e < 8; ++e) {
        const ushort v = x[p.s0 + e0 + e];
        xs[e0 + e] = is > 0.0f ? f2bf(fp8_qx(bf2f(v), is, r)) : v;
    }
}

// ---- decode GEMV ------------------------------------------------------------------------------------------------------

constant constexpr uint FP8_SG = 8;  // simdgroups per threadgroup

template <int MB, int RS, int UN, typename OutT>
inline void fp8_gemv_impl(device const ushort* x, device const uchar* w, device const float* sc, device OutT* out,
                          constant OpP& p, threadgroup ushort* xs, uint2 tg, uint tid, uint sg, uint lane) {
    const uint M = p.a, N = p.b, K = p.c;
    const uint m0 = tg.y * MB;
    const uint rows = min((uint)MB, M - m0);
    const float ws = sc[0], is = sc[1], r = fp8_rcp(is);
    // stage q(x) of this threadgroup's activation rows
    for (uint i = tid; i < rows * K; i += FP8_SG * 32) {
        const ushort v = x[p.s0 + (ulong)m0 * K + i];
        xs[i] = is > 0.0f ? f2bf(fp8_qx(bf2f(v), is, r)) : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

FP8_GEMV(fp8_gemv_bf16_m8, 8, 2, 2, ushort)
FP8_GEMV(fp8_gemv_f32_m8, 8, 2, 2, float)

// ---- decode GEMV, M = 1: split-K (`fp8_gv`) ---------------------------------------------------------------------------
// A threadgroup of SG simdgroups owns R output rows and streams them together: thread t of the threadgroup takes the
// 8-byte chunks t, t + 32 SG, ... of every row (one 8-byte load per row and chunk, so a row-aligned address is all it
// needs and the K % 16 == 0 rows need no head / tail), and quantises only the 8 activations of its chunk, so there is
// no staging and no barrier before the weight loads are in flight. Per-simdgroup partials are combined in a fixed order
// through threadgroup memory (deterministic, no atomics). R is chosen per call so that N / R covers the GPU.

// 8 e4m3 bytes (2 words) . 8 activations; the result is the dot / 256, NaN when a byte holds a NaN code
inline float fp8_dot8(uint2 v, const thread float* x) {
    const uint w[2] = {v.x, v.y};
    float t = 0.0f;
    uint nan = 0u;
    for (int j = 0; j < 2; ++j) {
        const float2 p = fp8_x2(w[j]), q = fp8_x2(w[j] >> 8);  // elements (0, 2) and (1, 3) of the word
        t = fma(p.x, x[4 * j], t);
        t = fma(q.x, x[4 * j + 1], t);
        t = fma(p.y, x[4 * j + 2], t);
        t = fma(q.y, x[4 * j + 3], t);
        nan |= ((w[j] & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;  // bit 7 of a byte <=> its magnitude bits are 0x7F
    }
    return nan != 0u ? as_type<float>(0x7FC00000u) : t;
}

constant constexpr uint FP8_GV_UN = 2;  // chunks per thread and row loaded before any is used (K <= 8 * 32 * SG * UN in one step)

template <int R, int SG, typename OutT>
inline void fp8_gv_impl(device const ushort* x, device const uchar* w, device const float* sc, device OutT* out,
                        constant OpP& p, threadgroup float* part, uint tg, uint tid, uint sg, uint lane) {
    const uint N = p.b, K = p.c;
    const float ws = sc[0], is = sc[1], rcp = fp8_rcp(is);
    const uint nch = K / 8;
    const uint r0 = tg * R;
    device const uchar* wr[R];
    for (int r = 0; r < R; ++r) wr[r] = w + (ulong)min(r0 + r, N - 1) * K;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    constexpr uint lanes = 32 * SG;
    for (uint c0 = tid; c0 < nch; c0 += lanes * FP8_GV_UN) {
        uint2 wv[FP8_GV_UN][R];
        for (uint j = 0; j < FP8_GV_UN; ++j) {
            const uint c = c0 + lanes * j;
            for (int r = 0; r < R; ++r) wv[j][r] = c < nch ? *(device const uint2*)(wr[r] + (ulong)c * 8) : uint2(0u);
        }
        for (uint j = 0; j < FP8_GV_UN; ++j) {
            const uint c = c0 + lanes * j;
            if (c >= nch) break;
            device const packed_ushort4* xp = (device const packed_ushort4*)(x + p.s0 + (ulong)c * 8);
            const ushort4 a = ushort4(xp[0]), b = ushort4(xp[1]);
            float xf[8] = {bf2f(a.x), bf2f(a.y), bf2f(a.z), bf2f(a.w), bf2f(b.x), bf2f(b.y), bf2f(b.z), bf2f(b.w)};
            if (is > 0.0f) {
                for (int e = 0; e < 8; ++e) xf[e] = fp8_qx(xf[e], is, rcp);
            }
            for (int r = 0; r < R; ++r) acc[r] += fp8_dot8(wv[j][r], xf);
        }
    }
    for (int r = 0; r < R; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == 0) part[r * SG + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R && r0 + tid < N) {
        float t = part[tid * SG];
        for (int s = 1; s < SG; ++s) t += part[tid * SG + s];
        store_out(out + p.s1, r0 + tid, t * (ws * (is > 0.0f ? is : 1.0f) * 256.0f));
    }
}

#define FP8_GV(NAME, R, SG, OUT)                                                                                        \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const float* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                           \
                     constant OpP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],                       \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],          \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        threadgroup float part[R * SG];                                                                              \
        fp8_gv_impl<R, SG, OUT>(x, w, sc, out, p, part, tg, tid, sg, lane);                                          \
    }

#define FP8_GV_RS(SG)                                                                                                   \
    FP8_GV(fp8_gv_bf16_s##SG##r1, 1, SG, ushort)                                                                        \
    FP8_GV(fp8_gv_bf16_s##SG##r2, 2, SG, ushort)                                                                        \
    FP8_GV(fp8_gv_bf16_s##SG##r4, 4, SG, ushort)                                                                        \
    FP8_GV(fp8_gv_bf16_s##SG##r8, 8, SG, ushort)                                                                        \
    FP8_GV(fp8_gv_f32_s##SG##r1, 1, SG, float)                                                                          \
    FP8_GV(fp8_gv_f32_s##SG##r2, 2, SG, float)                                                                          \
    FP8_GV(fp8_gv_f32_s##SG##r4, 4, SG, float)                                                                          \
    FP8_GV(fp8_gv_f32_s##SG##r8, 8, SG, float)
FP8_GV_RS(1)
FP8_GV_RS(2)
FP8_GV_RS(4)
FP8_GV_RS(8)

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
