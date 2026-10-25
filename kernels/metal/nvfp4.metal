// matmul_nvfp4: out[M,N] = x[M,K] @ W^T, W[n,k] = e2m1(code) * e4m3(block[n,k/16]) * global (NVIDIA NVFP4, group 16).
// Port of kernels/hip/nvfp4.hip (same semantics as crates/intarsia-core/src/quant/reference.rs matmul_nvfp4);
// launcher crates/intarsia-backend-metal/src/nvfp4.rs.
//   x      bf16 [M,K]
//   w      u8   [N,K/2]  two codes per byte, element 2i in the low nibble; bound 4-byte aligned, the launcher requires an
//                        8-byte aligned address (a row is K / 2 = 8 * (K / 16) bytes)
//   bs     u8   [N,K/16] e4m3 block scales, plain row-major, any byte alignment (bound skip added)
//   gs     f32  [1]      global scale, read here on the GPU
// Decoding reads a code as the bits of a half (<< 9: 2^14 times smaller than the e2m1 value) and a block scale as the bits of
// a half (<< 7: 256 times smaller than the e4m3 value); both factors, 2^22 together, are folded into the final scale. The
// e4m3 NaN code (0x7F / 0xFF) of a block scale gives NaN.
// Entry points:
//   nv_gemv_{bf16,f32}_{m1,m8}  decode GEMV for M <= 1 / 8 per call, one dispatch, no scratch. A threadgroup of NV_SG
//                               simdgroups, each owning NV_RS_* output rows; a lane takes a 16-element K block at a time (one
//                               8-byte weight load and one scale byte per row), f32 accumulate, simd_sum.
//   nv_mma_{bf16,f32}           GEMM for any M, 64 x 64 tile, 4 simdgroups of 32 x 32, K step 32. The B tile holds
//                               e2m1 * e4m3(block) as bf16 (exact: at most 5 significant bits), the global scale is applied
//                               in the epilogue; simdgroup_matrix bf16 x bf16 -> f32.
// Kernel arguments: OpP (common.h), a = M, b = N, c = K; s0 = x skip, s1 = block-scale skip, s2 = out skip.

#include "common.h"

// two codes (nibbles at bits 0..3 and 16..19 of t) -> value / 2^14
inline float2 nv_x2(uint t) {
    const uint h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    return float2(as_type<half2>(h));
}

// e4m3 block scale byte -> value / 256 (NaN for the NaN code)
inline float nv_scale(uint b) {
    if ((b & 0x7Fu) == 0x7Fu) return as_type<float>(0x7FC00000u);
    return float(as_type<half>((ushort)(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8))));
}

constant constexpr float NV_FOLD = 4194304.0f;  // 2^22 = 2^14 * 256

// ---- decode GEMV ------------------------------------------------------------------------------------------------------

constant constexpr uint NV_SG = 8;  // simdgroups per threadgroup

// one block: 16 codes (2 words) . 16 activations, in the units of nv_x2
inline float nv_dot16(uint2 v, const thread float* x) {
    float t = 0.0f;
    const uint w[2] = {v.x, v.y};
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            const float2 c = nv_x2((w[i] >> (4 * k)) & 0x000F000Fu);  // elements k and k + 4 of the word
            t = fma(c.x, x[8 * i + k], t);
            t = fma(c.y, x[8 * i + k + 4], t);
        }
    }
    return t;
}

// the weight half of `nv_dot16`: the 16 codes as floats in the order the fma chain consumes them (word i, nibble k: elements
// 8i + k then 8i + k + 4)
inline void nv_dec16(uint2 v, thread float* cw) {
    const uint w[2] = {v.x, v.y};
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            const float2 c = nv_x2((w[i] >> (4 * k)) & 0x000F000Fu);
            cw[8 * i + 2 * k] = c.x;
            cw[8 * i + 2 * k + 1] = c.y;
        }
    }
}

// the activation half: the chain of `nv_dot16` over decoded codes
inline float nv_dot16d(const thread float* cw, const thread float* x) {
    float t = 0.0f;
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            t = fma(cw[8 * i + 2 * k], x[8 * i + k], t);
            t = fma(cw[8 * i + 2 * k + 1], x[8 * i + k + 4], t);
        }
    }
    return t;
}

template <int MB, int RS, typename OutT>
inline void nv_gemv_impl(device const ushort* x, device const uchar* w, device const uchar* bs, device const float* gs,
                         device OutT* out, constant OpP& p, uint2 tg, uint sg, uint lane) {
    const uint M = p.a, N = p.b, K = p.c;
    const uint nb = (tg.x * NV_SG + sg) * RS;
    if (nb >= N) return;  // whole simdgroup exits together
    const uint m0 = tg.y * MB;
    const uint rows = min((uint)MB, M - m0);
    device const uchar* wr[RS];
    device const uchar* br[RS];
    for (int r = 0; r < RS; ++r) {
        const uint n = min(nb + r, N - 1);
        wr[r] = w + (ulong)n * (K / 2);
        br[r] = bs + p.s1 + (ulong)n * (K / 16);
    }
    float acc[RS][MB];
    for (int r = 0; r < RS; ++r)
        for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
    for (uint u = lane; u < K / 16; u += 32) {
        uint2 wv[RS];
        float s[RS];
        for (int r = 0; r < RS; ++r) {
            wv[r] = *(device const uint2*)(wr[r] + (ulong)u * 8);
            s[r] = nv_scale(br[r][u]);
        }
        if constexpr (MB > 1) {
            // multi-row forms: decode each weight block ONCE and apply it to every activation row; per row the same fma chain
            // over the same values in the same order as `nv_dot16`, so every row's bits are those of the one-row kernel
            float cw[RS][16];
            for (int r = 0; r < RS; ++r) nv_dec16(wv[r], cw[r]);
            for (int m = 0; m < MB; ++m) {
                if (m >= (int)rows) break;
                device const packed_ushort4* xp = (device const packed_ushort4*)(x + p.s0 + (ulong)(m0 + m) * K + (ulong)u * 16);
                float xf[16];
                for (int q = 0; q < 4; ++q) {
                    const ushort4 v = ushort4(xp[q]);
                    xf[4 * q] = bf2f(v.x);
                    xf[4 * q + 1] = bf2f(v.y);
                    xf[4 * q + 2] = bf2f(v.z);
                    xf[4 * q + 3] = bf2f(v.w);
                }
                for (int r = 0; r < RS; ++r) acc[r][m] = fma(nv_dot16d(cw[r], xf), s[r], acc[r][m]);
            }
        } else {
            for (int m = 0; m < MB; ++m) {
                if (m >= (int)rows) break;
                device const packed_ushort4* xp = (device const packed_ushort4*)(x + p.s0 + (ulong)(m0 + m) * K + (ulong)u * 16);
                float xf[16];
                for (int q = 0; q < 4; ++q) {
                    const ushort4 v = ushort4(xp[q]);
                    xf[4 * q] = bf2f(v.x);
                    xf[4 * q + 1] = bf2f(v.y);
                    xf[4 * q + 2] = bf2f(v.z);
                    xf[4 * q + 3] = bf2f(v.w);
                }
                for (int r = 0; r < RS; ++r) acc[r][m] = fma(nv_dot16(wv[r], xf), s[r], acc[r][m]);
            }
        }
    }
    const float scale = gs[0] * NV_FOLD;
    for (int m = 0; m < MB; ++m) {
        if (m >= (int)rows) break;
        for (int r = 0; r < RS; ++r) {
            const float t = simd_sum(acc[r][m]);
            if (lane == 0 && nb + r < N) store_out(out + p.s2, (ulong)(m0 + m) * N + nb + r, t * scale);
        }
    }
}

#define NV_GEMV(NAME, MB, RS, OUT)                                                                                  \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                     \
                     device const uchar* bs [[buffer(2)]], device const float* gs [[buffer(3)]],                    \
                     device OUT* out [[buffer(4)]], constant OpP& p [[buffer(5)]],                                  \
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],         \
                     uint lane [[thread_index_in_simdgroup]]) {                                                     \
        nv_gemv_impl<MB, RS, OUT>(x, w, bs, gs, out, p, tg, sg, lane);                                              \
    }

NV_GEMV(nv_gemv_bf16_m1, 1, 4, ushort)
NV_GEMV(nv_gemv_bf16_m8, 8, 2, ushort)
NV_GEMV(nv_gemv_f32_m1, 1, 4, float)
NV_GEMV(nv_gemv_f32_m8, 8, 2, float)
// M split across threadgroups for the narrow matrices of a batched step (lane mm): the same per-row arithmetic
NV_GEMV(nv_gemv_bf16_m2, 2, 2, ushort)
NV_GEMV(nv_gemv_bf16_m4, 4, 2, ushort)
NV_GEMV(nv_gemv_f32_m2, 2, 2, float)
NV_GEMV(nv_gemv_f32_m4, 4, 2, float)

// ---- GEMM -------------------------------------------------------------------------------------------------------------

constant constexpr uint NV_BM = 64;
constant constexpr uint NV_BN = 64;
constant constexpr uint NV_BK = 32;
constant constexpr uint NV_LD = 40;  // bf16 elements per staged row (80 bytes, 16-byte multiple, skews the banks)

inline uint nv_pack(float lo, float hi) { return (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xFFFF0000u); }

template <typename OutT>
inline void nv_mma_impl(device const ushort* X, device const uchar* W, device const uchar* BS, device const float* gs,
                        device OutT* Y, constant OpP& p, threadgroup float* smem, uint2 tgp, uint tid, uint sg,
                        uint lane) {
    threadgroup bfloat* As = (threadgroup bfloat*)smem;
    threadgroup bfloat* Bs = As + NV_BM * NV_LD;
    const uint M = p.a, N = p.b, K = p.c;
    const uint n0 = tgp.x * NV_BN, m0 = tgp.y * NV_BM;
    const uint sm = sg / 2, sn = sg % 2;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += NV_BK) {
        for (uint c = tid; c < NV_BM * 4; c += 128) {  // A: 64 rows x 4 chunks of 8 bf16 (x may sit at an odd element)
            const uint r = c / 4, q = c % 4;
            const uint gm = m0 + r, gk = k0 + q * 8;
            uint4 v = uint4(0u);
            if (gm < M && gk < K) {
                device const packed_ushort4* xp = (device const packed_ushort4*)(X + p.s0 + (ulong)gm * K + gk);
                const ushort4 a = ushort4(xp[0]), b = ushort4(xp[1]);
                v = uint4(a.x | ((uint)a.y << 16), a.z | ((uint)a.w << 16), b.x | ((uint)b.y << 16),
                          b.z | ((uint)b.w << 16));
            }
            *(threadgroup uint4*)(As + r * NV_LD + q * 8) = v;
        }
        {  // B: 64 rows x 2 blocks of 16 codes, dequantised to bf16 (without the global scale)
            const uint r = tid / 2, q = tid % 2;
            const uint gn = n0 + r, gk = k0 + q * 16;
            uint o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (gn < N && gk < K) {
                const uint2 v = *(device const uint2*)(W + (ulong)gn * (K / 2) + gk / 2);
                const float s = nv_scale(BS[p.s1 + (ulong)gn * (K / 16) + gk / 16]) * NV_FOLD;
                const uint w2[2] = {v.x, v.y};
                for (int i = 0; i < 2; ++i) {
                    float2 c[4];  // c[k]: elements k and k + 4 of the word
                    for (int k = 0; k < 4; ++k) c[k] = nv_x2((w2[i] >> (4 * k)) & 0x000F000Fu) * s;
                    o[4 * i] = nv_pack(c[0].x, c[1].x);
                    o[4 * i + 1] = nv_pack(c[2].x, c[3].x);
                    o[4 * i + 2] = nv_pack(c[0].y, c[1].y);
                    o[4 * i + 3] = nv_pack(c[2].y, c[3].y);
                }
            }
            threadgroup uint4* bp = (threadgroup uint4*)(Bs + r * NV_LD + q * 16);
            bp[0] = uint4(o[0], o[1], o[2], o[3]);
            bp[1] = uint4(o[4], o[5], o[6], o[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < NV_BK / 8; ++kk) {
            simdgroup_matrix<bfloat, 8, 8> a[4], b[4];
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sm * 32 + i * 8) * NV_LD + kk * 8, NV_LD);
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(b[j], Bs + (sn * 32 + j * 8) * NV_LD + kk * 8, NV_LD, ulong2(0, 0), true);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* cs = smem + sg * 1024;
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], cs + (i * 8) * 32 + j * 8, 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = gs[0];
    for (uint e = lane; e < 1024; e += 32) {
        const uint gm = m0 + sm * 32 + e / 32, gn = n0 + sn * 32 + e % 32;
        if (gm < M && gn < N) store_out(Y + p.s2, (ulong)gm * N + gn, cs[e] * scale);
    }
}

#define NV_MMA(NAME, OUT)                                                                                           \
    kernel void NAME(device const ushort* X [[buffer(0)]], device const uchar* W [[buffer(1)]],                     \
                     device const uchar* BS [[buffer(2)]], device const float* gs [[buffer(3)]],                    \
                     device OUT* Y [[buffer(4)]], constant OpP& p [[buffer(5)]],                                    \
                     uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],          \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {         \
        threadgroup float4 smem[1024];                                                                              \
        nv_mma_impl<OUT>(X, W, BS, gs, Y, p, (threadgroup float*)smem, tgp, tid, sg, lane);                         \
    }

NV_MMA(nv_mma_bf16, ushort)
NV_MMA(nv_mma_f32, float)
