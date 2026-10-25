// matmul_fp8: out[M,N] = x[M,K] @ W^T, W = float(w_e4m3[N,K]) * weight_scale (per tensor).
//   scales = f32[2] = (weight_scale, input_scale); input_scale > 0 selects W8A8 (static activation quantisation:
//   xq = e4m3(x / input_scale), RNE, saturating; y = weight_scale * input_scale * sum xq*wq), otherwise W8A16.
//   The mode is read from the device tensor inside the kernels (a uniform branch), so launching never needs a
//   host read and the same kernels serve both recipes.
//   *_gemv  decode: M <= 4, warp-per-output-row, x staged in shared memory (quantised once per block).
//   *_mma   any M: bf16 WMMA with the weights dequantised to bf16 in shared memory (exact), f32 accumulate
//           (ik_dq_gemm.h). Used for prefill and as the correct path for every shape.
// Both are deterministic (fixed reduction order). Activations bf16; output bf16 or f32.
// HIP port of kernels/cuda/fp8.cu; the "dequantise to bf16, bf16 MMA with exact products" scheme follows the ideas of
// TensorFold's NVFP4 kernels (MIT, Copyright (c) 2026 TensorFold contributors, see ik_quant.h and
// LICENSES/TensorFold-MIT.txt). gfx12 also has an fp8 x fp8 WMMA (v_wmma_f32_16x16x16_fp8_fp8) that would
// serve the W8A8 case natively; not used yet (docs/results/2026-10-13-phase2a-hip.md).
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "ik_dq_gemm.h"
#include "ik_f8_gemm.h"
#include "ik_hip_abi.h"

namespace {

constexpr int kWarps = 8;

// ---- GEMV --------------------------------------------------------------------------------------------------------
// One warp owns R output rows. The weight loads of the first batch (UN 16-byte chunks per row and lane) are issued
// before the activations are staged in shared memory, so their latency overlaps the staging (the e4m3 activation
// quantisation is the longest part of the prologue); later batches are loaded as the previous one is consumed.

constexpr int kUn = 4;

// e4m3 as fp16: (sign << 15) | (magnitude << 7) is the e4m3 value times 2^-8, exactly, subnormals included (an e4m3
// subnormal m * 2^-9 becomes the fp16 subnormal m * 2^-17). With W8A8 the activations are e4m3 too, so they are exact
// fp16 and v_dot2_f32_f16 multiplies and accumulates the pairs in f32: ~3 ALU ops per element instead of ~18 for the
// generic decode. The 2^8 is folded into the output scale. (NaN codes are not preserved here; checkpoints have none.)
typedef _Float16 ik_h2 __attribute__((ext_vector_type(2)));

IK_DEVI uint32_t e4m3x2_to_f16x2(uint32_t w, uint32_t sel) {
    const uint32_t t = __builtin_amdgcn_perm(0u, w, sel);  // [0, b_lo, 0, b_hi]
    return (((t >> 1) & 0x3F803F80u) | (t & 0x80008000u));
}

// The GDN output norm folded into the staging (PRO == 2): the value gated_rmsnorm_kernel<true> stores for the 8-element chunk
// `i8` of the rows `x` / `z` of 128 values (sixteen chunks per row), as bf16 chunks. A row's sum of squares is that kernel's
// block reduction over its 32 threads, of which only the first sixteen hold a chunk: the xor butterfly of ik_warp_sum starting
// at offset 8 inside a 16-lane group sums the same values in the same order (its first step, offset 16, adds an exact zero),
// so the sixteen lanes of a row may share a warp with the next row's. Every lane of a group must call it.
IK_DEVI uint4 gated_norm_chunk(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z, const ik_bf16* __restrict__ nw,
                               int i8, float eps, float woff) {
    float f[8], g[8], zf[8];
    ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(x) + i8), f);
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor(ss, o, IK_WARP);
    const float r = rsqrtf(ss / 128.f + eps);
    ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(nw) + (i8 & 15)), g);
    ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(z) + i8), zf);
#pragma unroll
    for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);
    float nrm[8];
    ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)
#pragma unroll
    for (int k = 0; k < 8; ++k) f[k] = ik_silu(zf[k]) * nrm[k];
    return ik_pack8(f);
}

// The attention input norm folded into the k / v staging (PRO == 3, core-op fusion P5): rmsnorm_kernel<true, true> for one row
// of K values, statement for statement (the same 8-element chunk per thread and iteration, the bf16 rounding of the sum before
// it is squared and normalised, ik_block_sum over the same 256 threads: the block size launch_rows picks for every K >= 2048;
// for a narrower row its extra threads contribute exact zeros). `xs` holds the rounded sum, then the normalised row; the block
// with `writer` also stores both, into buffers that are not the x / res every block reads. Every thread of the block must call
// it (block barriers). The staging loop afterwards reads each chunk back on the thread that wrote it.
__device__ __forceinline__ void norm_add_stage(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res,
                                               const ik_bf16* __restrict__ nw, ik_bf16* __restrict__ normed,
                                               ik_bf16* __restrict__ sum, uint4* xs, int K, float eps, float w_off,
                                               bool writer) {
    __shared__ float red[32];
    const int64_t nv = K / 8;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
        float f[8], g[8];
        ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(x) + i), f);
        ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(res) + i), g);
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] += g[k];
        const uint4 p = ik_pack8(f);
        if (writer) reinterpret_cast<uint4*>(sum)[i] = p;
        xs[i] = p;  // the rounded sum, until the second loop replaces it with the normalised row
        ik_unpack8(p, f);  // continue from the bf16-rounded sum
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
    }
    ss = ik_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)K + eps);
    for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
        float f[8], g[8];
        ik_unpack8(xs[i], f);
        ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(nw) + i), g);
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + w_off);  // (named apart from gated_norm_chunk's line: mutate.py gn-woff)
        const uint4 p = ik_pack8(f);
        xs[i] = p;
        if (writer) reinterpret_cast<uint4*>(normed)[i] = p;
    }
    __syncthreads();
}

// The attention output gate folded into the o_proj staging (PRO == 4, core-op fusion P4b): chunk `i8` of
// sigmoid_mul(a, gate) as sigmoid_mul_kernel (elementwise.hip) stores it, element by element: bf16(a * (1 / (1 + exp(-g)))).
IK_DEVI uint4 sigmoid_mul_chunk(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ gate, int i8) {
    float f[8], g[8];
    ik_unpack8(reinterpret_cast<const uint4*>(a)[i8], f);
    ik_unpack8(reinterpret_cast<const uint4*>(gate)[i8], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float gv = g[k];
        f[k] = f[k] * (1.f / (1.f + expf(-gv)));
    }
    return ik_pack8(f);
}

// Q selects how the block stages the activations (everything after the prologue is shared):
//   0  the original: one bf16 per thread-iteration, software e4m3 conversion
//   1  v2: eight bf16 per thread (one 16-byte load, one 16-byte shared store), same software conversion (bit-identical)
//   2  v2: eight per thread, the gfx12 fp8 conversion instruction (v_cvt_pk_fp8_f32) after an explicit clamp to +-448 and
//      the division by the input scale as before; the staged values are e4m3 * 2^-8 (the same trick as the weights), so the
//      output scale is 2^16 instead of 2^8 (all powers of two: bit-identical to Q = 0, 1)
//   3  as 2, with the division by the input scale as multiply-by-reciprocal plus one fma correction
// W is the number of warps per block (kWarps; the pair kernel's block shape is tunable).
// `blk` is the block's index along the rows of this matrix: blockIdx.x for fp8_gemv_kernel, the index inside its segment
// for the pair kernel below, which therefore runs this very code (its staging uses the segment's own scales).
// PRO == 2 stages the output of the GDN gated rmsnorm of `x` (the core, rows of 128) and `pz` (the gate) with weight `pnw`
// instead of reading `x` as it is (M == 1, Q == 2); PRO == 0 is the plain kernel.
// PRO == 3 (core-op fusion P5) stages rmsnorm_add(x, res = pz, pnw): the block first runs rmsnorm_kernel<true, true> for the
// row into its LDS (norm_add_stage above), and the block with `pwriter` also stores the sum (`psum`) and the normalised row
// (`pnormed`). PRO == 4 (P4b) stages sigmoid_mul(x, gate = pz), sigmoid_mul_kernel's expression per element. Both then
// quantise the staged chunk exactly as PRO == 0 quantises the chunk it reads from global memory.
template <int M, int R, typename OutT, int Q, int W = kWarps, int PRO = 0>
__device__ __forceinline__ void fp8_gemv_body(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w,
                                              const float* __restrict__ sc, OutT* __restrict__ out, int N, int K,
                                              int64_t blk, const ik_bf16* __restrict__ pz = nullptr,
                                              const ik_bf16* __restrict__ pnw = nullptr, float peps = 0.f,
                                              float pwoff = 0.f, ik_bf16* __restrict__ pnormed = nullptr,
                                              ik_bf16* __restrict__ psum = nullptr, bool pwriter = false) {
    static_assert(PRO == 0 || (M == 1 && Q == 2), "the folded prologues are the default single-row staging");
    static_assert(PRO != 3 || W == kWarps, "the folded norm reduces over rmsnorm_kernel's 256 threads");
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* xs = reinterpret_cast<ik_bf16*>(smem);  // [M][K] bf16: x, or (W8A8) xq as fp16 bits (exact)
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = (blk * W + warp) * R;
    const int nu = K / 16;
    const uint4* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = n0 + r < N ? reinterpret_cast<const uint4*>(w + (n0 + r) * K) : nullptr;
    uint4 wb[R][kUn];
    auto load = [&](int i0) {  // iterations i0 .. i0 + kUn - 1 of this lane
#pragma unroll
        for (int j = 0; j < kUn; ++j) {
            const int u = lane + IK_WARP * (i0 + j);
#pragma unroll
            for (int r = 0; r < R; ++r) wb[r][j] = (wr[r] && u < nu) ? ik_ldcs(wr[r] + u) : make_uint4(0, 0, 0, 0);
        }
    };
    load(0);

    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    if constexpr (PRO == 3) norm_add_stage(x, pz, pnw, pnormed, psum, reinterpret_cast<uint4*>(xs), K, peps, pwoff, pwriter);
    if constexpr (Q == 0) {
        for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
            const ik_bf16 v = x[i];
            if (actq) {
                const _Float16 q = (_Float16)ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(ik_b2f(v), s_in)));
                xs[i].bits = __builtin_bit_cast(uint16_t, q);
            } else {
                xs[i] = v;
            }
        }
    } else {
        const float r_in = Q == 3 ? 1.f / s_in : 0.f;
        for (int i8 = threadIdx.x; i8 < M * K / 8; i8 += blockDim.x) {
            uint4 v;
            if constexpr (PRO == 2) v = gated_norm_chunk(x, pz, pnw, i8, peps, pwoff);
            else if constexpr (PRO == 3) v = reinterpret_cast<const uint4*>(xs)[i8];  // this thread's own chunk of the normed row
            else if constexpr (PRO == 4) v = sigmoid_mul_chunk(x, pz, i8);
            else v = reinterpret_cast<const uint4*>(x)[i8];
            uint4 o;
            if (!actq) {
                o = v;
            } else {
                float f[8];
                ik_unpack8(v, f);
                if constexpr (Q == 1) {
                    uint32_t ow[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const _Float16 a = (_Float16)ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[2 * e], s_in)));
                        const _Float16 b = (_Float16)ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[2 * e + 1], s_in)));
                        ow[e] = (uint32_t)__builtin_bit_cast(uint16_t, a) | ((uint32_t)__builtin_bit_cast(uint16_t, b) << 16);
                    }
                    o = make_uint4(ow[0], ow[1], ow[2], ow[3]);
                } else {
#if __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32)
                    float y[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float q = Q == 3 ? f[e] * r_in : __fdiv_rn(f[e], s_in);
                        if (Q == 3) q = fmaf(fmaf(-q, s_in, f[e]), r_in, q);
                        y[e] = fminf(fmaxf(q, -448.f), 448.f);
                    }
                    int c0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[0], y[1], 0, false);
                    c0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[2], y[3], c0, true);
                    int c1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[4], y[5], 0, false);
                    c1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[6], y[7], c1, true);
                    o = make_uint4(e4m3x2_to_f16x2((uint32_t)c0, 0x010C000Cu), e4m3x2_to_f16x2((uint32_t)c0, 0x030C020Cu),
                                   e4m3x2_to_f16x2((uint32_t)c1, 0x010C000Cu), e4m3x2_to_f16x2((uint32_t)c1, 0x030C020Cu));
#else
                    o = make_uint4(0, 0, 0, 0);  // no fp8 conversion instruction in this compiler: Q = 2, 3 are unavailable
#endif
                }
            }
            reinterpret_cast<uint4*>(xs)[i8] = o;
        }
    }
    __syncthreads();
    if (!wr[0]) return;  // (all rows of the warp are past N; rows are consecutive)

    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    const int iters = (nu + IK_WARP - 1 - lane) / IK_WARP;  // iterations this lane runs
    if (actq) {
        for (int i0 = 0; i0 < iters; i0 += kUn) {
            if (i0 > 0) load(i0);
#pragma unroll
            for (int j = 0; j < kUn; ++j) {
                const int u = lane + IK_WARP * (i0 + j);
                if (u >= nu) break;
                uint4 xv[M][2];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const uint4* xp = reinterpret_cast<const uint4*>(xs + (int64_t)m * K + u * 16);
                    xv[m][0] = xp[0];
                    xv[m][1] = xp[1];
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t ww[4] = {wb[r][j].x, wb[r][j].y, wb[r][j].z, wb[r][j].w};
                    uint32_t wh[8];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        wh[2 * q] = e4m3x2_to_f16x2(ww[q], 0x010C000Cu);
                        wh[2 * q + 1] = e4m3x2_to_f16x2(ww[q], 0x030C020Cu);
                    }
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        const uint32_t xw[8] = {xv[m][0].x, xv[m][0].y, xv[m][0].z, xv[m][0].w,
                                                xv[m][1].x, xv[m][1].y, xv[m][1].z, xv[m][1].w};
                        float s = acc[r][m];
#pragma unroll
                        for (int q = 0; q < 8; ++q)
                            s = __builtin_amdgcn_fdot2(__builtin_bit_cast(ik_h2, wh[q]), __builtin_bit_cast(ik_h2, xw[q]), s, false);
                        acc[r][m] = s;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float s = ik_warp_sum(acc[r][m]);
                if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, s * ((Q == 2 || Q == 3) ? 65536.f : 256.f) * s_out);
            }
        return;
    }
    for (int i0 = 0; i0 < iters; i0 += kUn) {
        if (i0 > 0) load(i0);
#pragma unroll
        for (int j = 0; j < kUn; ++j) {
            const int u = lane + IK_WARP * (i0 + j);
            if (u >= nu) break;
            float xf[M][16];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4* xp = reinterpret_cast<const uint4*>(xs + (int64_t)m * K + u * 16);
                ik_unpack8(xp[0], xf[m]);
                ik_unpack8(xp[1], xf[m] + 8);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float wf[16];
                ik_fp8x16_to_f32(wb[r][j], wf);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    float s = 0.f;
#pragma unroll
                    for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[m][e], s);
                    acc[r][m] += s;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float s = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, s * s_out);
        }
}

template <int M, int R, typename OutT, int Q>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                OutT* __restrict__ out, int N, int K) {
    fp8_gemv_body<M, R, OutT, Q>(x, w, sc, out, N, K, (int64_t)blockIdx.x);
}

// Two projections of one activation row in one launch (GDN qkv + z, attention k + v): blocks [0, nb0) run the first
// matrix with its own scales, the rest the second. Each block stages the activations itself, so the two segments need
// not share an input scale (the checkpoint's do).
template <int R, typename OutT, int Q, int W = kWarps>
__global__ void __launch_bounds__(W * IK_WARP)
fp8_gemv_pair_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, const float* __restrict__ sc0,
                     OutT* __restrict__ o0, int N0, const uint8_t* __restrict__ w1, const float* __restrict__ sc1,
                     OutT* __restrict__ o1, int N1, int K) {
    constexpr int per = W * R;
    const int nb0 = (N0 + per - 1) / per;
    if ((int)blockIdx.x < nb0) fp8_gemv_body<1, R, OutT, Q, W>(x, w0, sc0, o0, N0, K, (int64_t)blockIdx.x);
    else fp8_gemv_body<1, R, OutT, Q, W>(x, w1, sc1, o1, N1, K, (int64_t)blockIdx.x - nb0);
}

// rows per warp: INTARSIA_HIP_FP8_ROWS=1|2|4 forces one, otherwise 1 for small N (keeps enough blocks in flight)
// (v2: two rows per warp at every N; measured DRAM-cold on the 35B's projection shapes it is the best or tied for N = 512 to
// 8192, and 1.4 to 6.7 us faster than one row per warp on N = 2048 and 4096, which v1's default of one row serves)
int gemv_rows(int64_t N, bool v2) {
    static const int forced = [] {
        const char* e = getenv("INTARSIA_HIP_FP8_ROWS");
        return e ? atoi(e) : 0;
    }();
    if (forced == 1 || forced == 2 || forced == 4) return forced;
    return (v2 || N >= 6144) ? 2 : 1;
}

template <int M, int R, typename OutT, int Q>
void gemv_run_r(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps * R - 1) / (kWarps * R));
    const size_t smem = (size_t)M * K * 2;
    fp8_gemv_kernel<M, R, OutT, Q><<<grid, kWarps * IK_WARP, smem, s>>>(x, w, sc, out, (int)N, (int)K);
}

template <int M, typename OutT, int Q>
void gemv_run_q(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    switch (gemv_rows(N, Q != 0)) {
        case 1: gemv_run_r<M, 1, OutT, Q>(x, w, sc, out, N, K, s); break;
        case 4: gemv_run_r<M, 4, OutT, Q>(x, w, sc, out, N, K, s); break;
        default: gemv_run_r<M, 2, OutT, Q>(x, w, sc, out, N, K, s); break;
    }
}

// activation staging variant of the v2 kernel: INTARSIA_HIP_FP8_Q = 1, 2, 3 (see fp8_gemv_kernel), default 2
int staging_q() {
    static const int q = [] {
        const char* e = getenv("INTARSIA_HIP_FP8_Q");
        const int v = e ? atoi(e) : 2;
        return (v == 1 || v == 3) ? v : 2;
    }();
    return q;
}

template <int M, typename OutT>
void gemv_run(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s, bool v2) {
    if (!v2) return gemv_run_q<M, OutT, 0>(x, w, sc, out, N, K, s);
    switch (staging_q()) {
        case 2: return gemv_run_q<M, OutT, 2>(x, w, sc, out, N, K, s);
        case 3: return gemv_run_q<M, OutT, 3>(x, w, sc, out, N, K, s);
        case 1: return gemv_run_q<M, OutT, 1>(x, w, sc, out, N, K, s);
        default: return gemv_run_q<M, OutT, 2>(x, w, sc, out, N, K, s);
    }
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s,
              bool v2) {
    switch (M) {
        case 1: gemv_run<1>(x, w, sc, out, N, K, s, v2); break;
        case 2: gemv_run<2>(x, w, sc, out, N, K, s, v2); break;
        case 3: gemv_run<3>(x, w, sc, out, N, K, s, v2); break;
        case 4: gemv_run<4>(x, w, sc, out, N, K, s, v2); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

// ---- row-exact multi-row GEMV (matmul_fp8_gemv_rows; lane hb, batched decode) ---------------------------------------
// The default v2 kernel (2 rows per warp, staging Q = 2) for M = 1..kRowsGroup rows in one launch: the kernel's arithmetic for
// output (m, n) is its one-row arithmetic (a lane's u = lane + 32 i walk, one fdot2 / fma chain per row, one warp sum), M only
// sizes the staged activations and the accumulator array. gemv_by_m stops at 4 rows (the ordinary decode registrations);
// this table goes up to the most rows whose staged x (M * K * 2 bytes) fits the 48 KiB the kernel is allowed. The launcher
// below splits a larger M into near-equal groups of at most that many rows (a group is a launch; the weights are read once
// per group, so M = 16 at K = 2048 is two weight passes, at K = 4096 three).
constexpr int kRowsGroup = 12;
constexpr int kRowsMax = 16;

template <typename OutT>
int gemv_rows_by_m(int M, const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K,
                   hipStream_t s) {
    switch (M) {
#define IK_FP8_ROWS_CASE(MV) \
    case MV: gemv_run_r<MV, 2, OutT, 2>(x, w, sc, out, N, K, s); break;
        IK_FP8_ROWS_CASE(1) IK_FP8_ROWS_CASE(2) IK_FP8_ROWS_CASE(3) IK_FP8_ROWS_CASE(4) IK_FP8_ROWS_CASE(5) IK_FP8_ROWS_CASE(6)
        IK_FP8_ROWS_CASE(7) IK_FP8_ROWS_CASE(8) IK_FP8_ROWS_CASE(9) IK_FP8_ROWS_CASE(10) IK_FP8_ROWS_CASE(11)
        IK_FP8_ROWS_CASE(12)
#undef IK_FP8_ROWS_CASE
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

#include "ik_f8_rows.h"

// ---- MMA policy (weights dequantised to bf16 tiles) ---------------------------------------------------------------

struct Fp8Policy {
    IK_DEVI bool skip(const DqArgs&) { return false; }
    struct State {
        bool actq;
        float s_in, s_out;
    };
    typedef uint4 Raw;  // 16 e4m3 bytes
    IK_DEVI State init(const DqArgs& a) {
        State s;
        s.actq = a.sc[1] > 0.f;
        s.s_in = s.actq ? a.sc[1] : 1.f;
        s.s_out = a.sc[0] * s.s_in;
        return s;
    }
    IK_DEVI Raw load(const DqArgs& a, int64_t n, int k) {
        return *reinterpret_cast<const uint4*>(a.w + n * a.K + k);
    }
    IK_DEVI Raw zero() { return make_uint4(0, 0, 0, 0); }
    IK_DEVI void dequant(const State&, const Raw& r, uint4& lo, uint4& hi) {
        float f[16];
        ik_fp8x16_to_f32(r, f);  // exact in bf16
        lo = ik_pack8(f);
        hi = ik_pack8(f + 8);
    }
    IK_DEVI uint4 xprep(const State& s, uint4 v) {
        if (!s.actq) return v;
        float f[8];
        ik_unpack8(v, f);
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s.s_in)));
        return ik_pack8(f);  // fp8 values are exact in bf16
    }
    IK_DEVI float out_scale(const State& s) { return s.s_out; }
};

// The weight-only half of matmul_fp8_mma_f8: the same GEMM, exiting at once when the scale is positive (the e4m3 kernel did the launch).
struct Fp8PolicyW16 : Fp8Policy {
    IK_DEVI bool skip(const DqArgs& a) { return a.sc[1] > 0.f; }
};

// matmul_fp8_mma_v2: the GEMM of matmul_fp8_mma with the two conversions of its tile loop made cheap, bitwise the same.
//  * W8A8 activations arrive quantised (f8_actq_kernel, once per call: the rounding of Fp8Policy::xprep, as e4m3 bytes in
//    scratch) instead of being divided, rounded and decoded again by every block along N (64 times at N = 8192), which was
//    most of the ALU work of the tile loop; they are decoded to the same bf16 when the tile is stored. Weight-only
//    activations (input_scale <= 0: the quantiser exits) are read as bf16, as before.
//  * the weight chunk is decoded by the gfx12 conversion instruction (ik_e4m3x8_to_bf16) instead of in software.
// The LDS tiles hold the same bf16 values and the WMMAs run in the same order, so the output bits are matmul_fp8_mma's.
struct Fp8PolicyQ : Fp8Policy {
    IK_DEVI uint4 xload(const DqArgs& a, const State& s, int m, int k) {
        if (s.actq) {
            const uint2 q = *reinterpret_cast<const uint2*>(a.xq + (int64_t)m * a.K + k);
            return make_uint4(q.x, q.y, 0u, 0u);
        }
        return *reinterpret_cast<const uint4*>(a.x + (int64_t)m * a.K + k);
    }
    IK_DEVI void dequant(const State&, const Raw& r, uint4& lo, uint4& hi) {
        lo = ik_e4m3x8_to_bf16(r.x, r.y);
        hi = ik_e4m3x8_to_bf16(r.z, r.w);
    }
    IK_DEVI uint4 xprep(const State& s, uint4 v) { return s.actq ? ik_e4m3x8_to_bf16(v.x, v.y) : v; }
};

// ---- launchers ---------------------------------------------------------------------------------------------------

int check(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* out) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    if (K % 16 != 0) return IK_ERR_SHAPE;
    return IK_OK;
}

// variant: 0 = tensor-core GEMM, 1 = GEMV, 2 = GEMV v2 (vectorised activation staging)
int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream, int variant) {
    const bool gemv = variant != 0;
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == hipSuccess
                   ? IK_OK
                   : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > 4 || M * K * 2 > 49152) return IK_ERR_SHAPE;
        return f32o ? gemv_by_m((int)M, xp, wp, sp, (float*)out->data, N, K, s, variant == 2)
                    : gemv_by_m((int)M, xp, wp, sp, (ik_bf16*)out->data, N, K, s, variant == 2);
    }
    DqArgs a;
    a.x = xp;
    a.w = wp;
    a.bs = nullptr;
    a.sc = sp;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    if (f32o) dq_gemm_run<Fp8Policy>(a, (float*)out->data, s);
    else dq_gemm_run<Fp8Policy>(a, (ik_bf16*)out->data, s);
    return ik_launch_status();
}

// matmul_fp8_gemv_rows (HIP only: declared in ik_hip_abi.h): M = 1..kRowsMax rows, row m bitwise the one-row
// ik_matmul_fp8_gemv_v2 on row m alone (tests/rows_exact.rs). K % 16 == 0 and K * 2 <= 48 KiB as the one-row kernel.
int launch_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0) return launch(x, w, scales, out, stream, 2);
    if (M > kRowsMax || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    const int64_t gmax = std::min<int64_t>(kRowsGroup, 49152 / (K * 2));
    if (gmax < 1) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    const int64_t groups = (M + gmax - 1) / gmax;
    int64_t r0 = 0;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t m = (M - r0 + (groups - g) - 1) / (groups - g);  // near-equal groups, each at most gmax rows
        IK_TRY(f32o ? gemv_rows_by_m((int)m, xp + r0 * K, wp, sp, (float*)out->data + r0 * N, N, K, s)
                    : gemv_rows_by_m((int)m, xp + r0 * K, wp, sp, (ik_bf16*)out->data + r0 * N, N, K, s));
        r0 += m;
    }
    return IK_OK;
}

// matmul_fp8_wmma (HIP only: ik_hip_abi.h; lane wg): M = 1..64 rows on the matrix cores, one family for every M (ik_f8_rows.h). Row m
// is bitwise the same call on row m alone, whatever the other rows hold. W8A8 and W8A16 in one launch (the mode is the device-side
// input scale).
// M >= INTARSIA_HIP_WMMA_DIRECT_MIN (default below) with scratch given: the activations are quantised once per call into scratch and the B fragments
// load from global (DIRECT); below it, or without scratch, each block quantises into LDS (STAGED). The bits are the same either way.
int wmma_direct_min() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_HIP_WMMA_DIRECT_MIN");
        return e ? atoi(e) : 4;
    }();
    return v;
}

int launch_wmma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0) return launch(x, w, scales, out, stream, 2);
    if (M > 64 || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    if (!ik_aligned16(xp) || !ik_aligned16(wp)) return IK_ERR_INVALID_ARG;
    uint8_t* scr = nullptr;
    if (scratch && scratch->data && scratch->bytes >= M * K && !(reinterpret_cast<uintptr_t>(scratch->data) & 15u)) scr = (uint8_t*)scratch->data;
    return out->dtype == IK_DTYPE_F32 ? f8r_dispatch<false>(xp, wp, sp, (float*)out->data, (int)M, N, K, scr, wmma_direct_min(), s)
                                      : f8r_dispatch<false>(xp, wp, sp, (ik_bf16*)out->data, (int)M, N, K, scr, wmma_direct_min(), s);
}

// ---- bf16 weights on the family (lane wg): the MoE router / shared-expert gate and the GDN b / a gates ----------------------------
int bf16_check(const ik_tensor* x, const ik_tensor* w, const ik_tensor* out) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_BF16, 2, 16));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (w->shape[1] != K || out->shape[0] != M || out->shape[1] != N || K % 16 != 0) return IK_ERR_SHAPE;
    return IK_OK;
}

int launch_bf16_wmma(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    IK_TRY(bf16_check(x, w, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0) return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == hipSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (M > 64 || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    if (!ik_aligned16(xp) || !ik_aligned16(wp)) return IK_ERR_INVALID_ARG;
    return out->dtype == IK_DTYPE_F32 ? f8r_dispatch<true>(xp, wp, nullptr, (float*)out->data, (int)M, N, K, nullptr, wmma_direct_min(), s)
                                      : f8r_dispatch<true>(xp, wp, nullptr, (ik_bf16*)out->data, (int)M, N, K, nullptr, wmma_direct_min(), s);
}

int bf16_pair_wmma(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* o0, ik_tensor* o1, void* stream) {
    IK_TRY(bf16_check(x, w0, o0));
    IK_TRY(bf16_check(x, w1, o1));
    if (o0->dtype != IK_DTYPE_BF16 || o1->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (M != 1 || K == 0 || N0 == 0 || N1 == 0 || N0 > 0x7fffffffLL || N1 > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (!ik_aligned16(x->data) || !ik_aligned16(w0->data) || !ik_aligned16(w1->data)) return IK_ERR_INVALID_ARG;
    F8rCfg c;
    if (!f8r_cfg(1, K, false, 0, false, 2, &c)) return IK_ERR_SHAPE;
    const unsigned grid = (unsigned)((N0 + 15) / 16 + (N1 + 15) / 16);
    hipStream_t s = IK_STREAM(stream);
    if (c.nj == 16)
        f8r_bf16_pair_kernel<16><<<grid, kF8rThreads, c.smem, s>>>((const ik_bf16*)x->data, (const uint8_t*)w0->data, (ik_bf16*)o0->data, (int)N0, (const uint8_t*)w1->data, (ik_bf16*)o1->data, (int)N1, (int)K, c.kw);
    else
        f8r_bf16_pair_kernel<8><<<grid, kF8rThreads, c.smem, s>>>((const ik_bf16*)x->data, (const uint8_t*)w0->data, (ik_bf16*)o0->data, (int)N0, (const uint8_t*)w1->data, (ik_bf16*)o1->data, (int)N1, (int)K, c.kw);
    return ik_launch_status();
}

int bf16_norm_pair_wmma(const ik_tensor* x, const ik_tensor* res, const ik_tensor* nw, float eps, float woff, const ik_tensor* w0,
                        const ik_tensor* w1, ik_tensor* normed, ik_tensor* sum, ik_tensor* o0, ik_tensor* o1, void* stream) {
    IK_TRY(bf16_check(x, w0, o0));
    IK_TRY(bf16_check(x, w1, o1));
    IK_TRY(ik_check(res, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(nw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(normed, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(sum, IK_DTYPE_BF16, 2));
    if (o0->dtype != IK_DTYPE_BF16 || o1->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (M != 1 || res->shape[0] != 1 || res->shape[1] != K || nw->shape[0] != K || normed->shape[0] != 1 || normed->shape[1] != K ||
        sum->shape[0] != 1 || sum->shape[1] != K)
        return IK_ERR_SHAPE;
    if (!ik_aligned16(res->data) || !ik_aligned16(nw->data) || !ik_aligned16(normed->data) || !ik_aligned16(sum->data) ||
        !ik_aligned16(x->data) || !ik_aligned16(w0->data) || !ik_aligned16(w1->data))
        return IK_ERR_SHAPE;
    if (K == 0 || N0 == 0 || N1 == 0 || N0 > 0x7fffffffLL || N1 > 0x7fffffffLL) return IK_ERR_SHAPE;
    F8rCfg c;
    // the norm runs once, in one window (it writes `normed` / `sum`)
    if (!f8r_cfg(1, K, true, (size_t)K * 2, false, 2, &c)) return IK_ERR_SHAPE;  // bf16 weights: the window is in elements, 2 B each
    const unsigned grid = (unsigned)((N0 + 15) / 16 + (N1 + 15) / 16);
    hipStream_t s = IK_STREAM(stream);
#define IK_BF_NP(NJV)                                                                                                                  \
    f8r_bf16_norm_pair_kernel<NJV><<<grid, kF8rThreads, c.smem, s>>>(                                                                  \
        (const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)nw->data, eps, woff, (ik_bf16*)normed->data,               \
        (ik_bf16*)sum->data, (const uint8_t*)w0->data, (ik_bf16*)o0->data, (int)N0, (const uint8_t*)w1->data, (ik_bf16*)o1->data,     \
        (int)N1, (int)K, c.kw)
    if (c.nj == 16) IK_BF_NP(16);
    else IK_BF_NP(8);
#undef IK_BF_NP
    return ik_launch_status();
}

// matmul_fp8_mma_f8 (HIP only: declared in ik_hip_abi.h): the W8A8 GEMM on the fp8 x fp8 WMMA (ik_f8_gemm.h). input_scale > 0 is known only on the device, so one
// launch of each kind is made and each exits at once when the scale is not for it: the activation quantiser and the e4m3 GEMM
// for W8A8, the dequantise-to-bf16 GEMM (the matmul_fp8_mma kernel) for weight-only activations. Needs scratch: M * K bytes.
int launch_f8(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, const ik_scratch* scratch,
              void* stream) {
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0 || K == 0) return launch(x, w, scales, out, stream, 0);
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (!scratch || !scratch->data || scratch->bytes < M * K) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    hipStream_t s = IK_STREAM(stream);
    uint8_t* xq = (uint8_t*)scratch->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    const int64_t n8 = M * K / 8;
    f8_actq_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((const ik_bf16*)x->data, xq, sp, n8);
    const int vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    DqArgs a;
    a.x = (const ik_bf16*)x->data;
    a.w = wp;
    a.bs = nullptr;
    a.sc = sp;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = vec_out;
    if (out->dtype == IK_DTYPE_F32) {
        f8_gemm_run<float>(xq, wp, sp, (float*)out->data, (int)M, N, (int)K, vec_out, s);
        dq_gemm_run<Fp8PolicyW16>(a, (float*)out->data, s);
    } else {
        f8_gemm_run<ik_bf16>(xq, wp, sp, (ik_bf16*)out->data, (int)M, N, (int)K, vec_out, s);
        dq_gemm_run<Fp8PolicyW16>(a, (ik_bf16*)out->data, s);
    }
    return ik_launch_status();
}

// matmul_fp8_mma_v2 (HIP only: ik_hip_abi.h): Fp8PolicyQ above. Needs scratch: M * K bytes (the quantised activations).
int launch_v2(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, const ik_scratch* scratch,
              void* stream) {
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0 || K == 0) return launch(x, w, scales, out, stream, 0);
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (!scratch || !scratch->data || (int64_t)scratch->bytes < M * K) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    hipStream_t s = IK_STREAM(stream);
    uint8_t* xq = (uint8_t*)scratch->data;
    const float* sp = (const float*)scales->data;
    const int64_t n8 = M * K / 8;
    f8_actq_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((const ik_bf16*)x->data, xq, sp, n8);
    DqArgs a;
    a.x = (const ik_bf16*)x->data;
    a.w = (const uint8_t*)w->data;
    a.bs = nullptr;
    a.sc = sp;
    a.xq = xq;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    if (out->dtype == IK_DTYPE_F32) dq_gemm_run<Fp8PolicyQ>(a, (float*)out->data, s);
    else dq_gemm_run<Fp8PolicyQ>(a, (ik_bf16*)out->data, s);
    return ik_launch_status();
}

// Block shape of the pair kernel (default staging): R rows per warp, W warps per block. One row per warp and eight warps
// per block: DRAM-cold on the 35B's pairs (qkv + z 12288 x 2048, k + v 1024 x 2048) it is 2.3 us faster than v2's two rows per
// warp on the large one and ties on the small one (docs/results/2026-10-17-hip-fp8-gemv.md). Each row is summed in the same
// order whatever the shape, so every (R, W) is bitwise equal to every other. INTARSIA_HIP_FP8_PAIR="R,W" (R 1, 2, 4;
// W 4, 8, 16) forces one; it is read at every launch (the sweep test changes it in-process).
struct PairCfg {
    int R, W;
};

PairCfg pair_cfg() {
    PairCfg c = {1, 8};
    if (const char* e = getenv("INTARSIA_HIP_FP8_PAIR")) {
        int r = 0, w = 0;
        if (sscanf(e, "%d,%d", &r, &w) == 2) c = {r, w};
    }
    return c;
}

template <int R, int W>
void pair_launch(const ik_bf16* xp, const uint8_t* p0, const float* s0, ik_bf16* q0, int64_t N0, const uint8_t* p1,
                 const float* s1, ik_bf16* q1, int64_t N1, int64_t K, hipStream_t s) {
    constexpr int per = W * R;
    const unsigned grid = (unsigned)((N0 + per - 1) / per + (N1 + per - 1) / per);
    fp8_gemv_pair_kernel<R, ik_bf16, 2, W><<<grid, W * IK_WARP, (size_t)K * 2, s>>>(xp, p0, s0, q0, (int)N0, p1, s1, q1, (int)N1, (int)K);
}

template <int R>
bool pair_w(int W, const ik_bf16* xp, const uint8_t* p0, const float* s0, ik_bf16* q0, int64_t N0, const uint8_t* p1,
            const float* s1, ik_bf16* q1, int64_t N1, int64_t K, hipStream_t s) {
    switch (W) {
        case 4: pair_launch<R, 4>(xp, p0, s0, q0, N0, p1, s1, q1, N1, K, s); return true;
        case 8: pair_launch<R, 8>(xp, p0, s0, q0, N0, p1, s1, q1, N1, K, s); return true;
        case 16: pair_launch<R, 16>(xp, p0, s0, q0, N0, p1, s1, q1, N1, K, s); return true;
        default: return false;
    }
}

bool pair_dispatch(PairCfg c, const ik_bf16* xp, const uint8_t* p0, const float* s0, ik_bf16* q0, int64_t N0, const uint8_t* p1,
                   const float* s1, ik_bf16* q1, int64_t N1, int64_t K, hipStream_t s) {
    switch (c.R) {
        case 1: return pair_w<1>(c.W, xp, p0, s0, q0, N0, p1, s1, q1, N1, K, s);
        case 2: return pair_w<2>(c.W, xp, p0, s0, q0, N0, p1, s1, q1, N1, K, s);
        case 4: return pair_w<4>(c.W, xp, p0, s0, q0, N0, p1, s1, q1, N1, K, s);
        default: return false;
    }
}

// M == 1, bf16 activations and outputs, staging variant as INTARSIA_HIP_FP8_Q (the other staging variants keep v2's two
// rows per warp and eight warps).
int pair(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* sc0, const ik_tensor* w1, const ik_tensor* sc1,
         ik_tensor* o0, ik_tensor* o1, void* stream, bool wmma = false) {
    IK_TRY(check(x, w0, sc0, o0));
    IK_TRY(check(x, w1, sc1, o1));
    if (o0->dtype != IK_DTYPE_BF16 || o1->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (M != 1 || M * K * 2 > 49152) return IK_ERR_SHAPE;
    if (N0 == 0 && N1 == 0) return IK_OK;
    if (K == 0) {
        hipStream_t z = IK_STREAM(stream);
        if (hipMemsetAsync(o0->data, 0, ik_numel(o0) * 2, z) != hipSuccess) return IK_ERR_LAUNCH;
        return hipMemsetAsync(o1->data, 0, ik_numel(o1) * 2, z) == hipSuccess ? IK_OK : IK_ERR_LAUNCH;
    }
    if (N0 > 0x7fffffffLL || N1 > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t *p0 = (const uint8_t*)w0->data, *p1 = (const uint8_t*)w1->data;
    const float *s0 = (const float*)sc0->data, *s1 = (const float*)sc1->data;
    ik_bf16 *q0 = (ik_bf16*)o0->data, *q1 = (ik_bf16*)o1->data;
    if (wmma) {
        F8rCfg c;
        if (!f8r_cfg(1, K, false, 0, false, 1, &c)) return IK_ERR_SHAPE;
        const unsigned grid = (unsigned)((N0 + 15) / 16 + (N1 + 15) / 16);
        if (c.nj == 16) f8r_pair_kernel<16><<<grid, kF8rThreads, c.smem, s>>>(xp, p0, s0, q0, (int)N0, p1, s1, q1, (int)N1, (int)K, c.kw);
        else f8r_pair_kernel<8><<<grid, kF8rThreads, c.smem, s>>>(xp, p0, s0, q0, (int)N0, p1, s1, q1, (int)N1, (int)K, c.kw);
        return ik_launch_status();
    }
    const int q = staging_q();
    if (q == 2) return pair_dispatch(pair_cfg(), xp, p0, s0, q0, N0, p1, s1, q1, N1, K, s) ? ik_launch_status() : IK_ERR_SHAPE;
    constexpr int R = 2, per = kWarps * R;
    const unsigned grid = (unsigned)((N0 + per - 1) / per + (N1 + per - 1) / per);
    const size_t smem = (size_t)K * 2;
    if (q == 3) fp8_gemv_pair_kernel<R, ik_bf16, 3><<<grid, kWarps * IK_WARP, smem, s>>>(xp, p0, s0, q0, (int)N0, p1, s1, q1, (int)N1, (int)K);
    else fp8_gemv_pair_kernel<R, ik_bf16, 1><<<grid, kWarps * IK_WARP, smem, s>>>(xp, p0, s0, q0, (int)N0, p1, s1, q1, (int)N1, (int)K);
    return ik_launch_status();
}

// ---- the GDN gated rmsnorm folded into the fp8 GEMV (out_proj) ------------------------------------------------------
// y = fp8_gemv_v2(gated_rmsnorm(core, z, w), W): every block normalises the 32 heads itself while it stages the activations
// (the weights of its first batch are already in flight), so the norm launch and its output round trip disappear. The staged
// row, and therefore the result, is bitwise the two launches it replaces.
template <int R, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_gated_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z, const ik_bf16* __restrict__ nw, float eps,
                      float woff, const uint8_t* __restrict__ w, const float* __restrict__ sc, OutT* __restrict__ out, int N,
                      int K) {
    fp8_gemv_body<1, R, OutT, 2, kWarps, 2>(x, w, sc, out, N, K, (int64_t)blockIdx.x, z, nw, eps, woff);
}

template <int R>
void gated_run(const ik_bf16* x, const ik_bf16* z, const ik_bf16* nw, float eps, float woff, const uint8_t* w, const float* sc,
               ik_bf16* out, int64_t N, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps * R - 1) / (kWarps * R));
    fp8_gemv_gated_kernel<R, ik_bf16><<<grid, kWarps * IK_WARP, (size_t)K * 2, s>>>(x, z, nw, eps, woff, w, sc, out, (int)N, (int)K);
}

// x / z [R, 128] (R * 128 == K), nw [128], w [N, K] fp8, out [1, N] bf16. Rows per warp as the single v2 GEMV chooses them.
int gated_matmul(const ik_tensor* x, const ik_tensor* z, const ik_tensor* nw, float eps, float woff, const ik_tensor* w,
                 const ik_tensor* scales, ik_tensor* out, void* stream, bool wmma = false) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(z, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(nw, IK_DTYPE_BF16, 1));
    const int64_t rows = x->shape[0], D = x->shape[1], K = rows * D, N = w->shape[0];
    if (z->shape[0] != rows || z->shape[1] != D || nw->shape[0] != D || D != 128) return IK_ERR_SHAPE;
    // the matmul sees the normalised rows as one activation row [1, K]
    ik_tensor xv = *x;
    xv.ndim = 2;
    xv.shape[0] = 1;
    xv.shape[1] = K;
    xv.stride[0] = K;
    xv.stride[1] = 1;
    IK_TRY(check(&xv, w, scales, out));
    if (out->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    if (K * 2 > 49152 || !ik_aligned16(x->data) || !ik_aligned16(z->data) || !ik_aligned16(nw->data)) return IK_ERR_SHAPE;
    if (N == 0 || K == 0) return IK_ERR_SHAPE;  // (nothing the model produces; the unfused ops handle empty shapes)
    if (N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16 *xp = (const ik_bf16*)x->data, *zp = (const ik_bf16*)z->data, *np = (const ik_bf16*)nw->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    ik_bf16* op = (ik_bf16*)out->data;
    if (wmma) {
        F8rCfg c;
        if (!f8r_cfg(1, K, true, 0, false, 1, &c)) return IK_ERR_SHAPE;
        const unsigned grid = (unsigned)((N + 15) / 16);
        if (c.nj == 16) f8r_gated_kernel<16><<<grid, kF8rThreads, c.smem, s>>>(xp, zp, np, eps, woff, wp, sp, op, (int)N, (int)K, c.kw);
        else f8r_gated_kernel<8><<<grid, kF8rThreads, c.smem, s>>>(xp, zp, np, eps, woff, wp, sp, op, (int)N, (int)K, c.kw);
        return ik_launch_status();
    }
    switch (gemv_rows(N, true)) {
        case 1: gated_run<1>(xp, zp, np, eps, woff, wp, sp, op, N, K, s); break;
        case 4: gated_run<4>(xp, zp, np, eps, woff, wp, sp, op, N, K, s); break;
        default: gated_run<2>(xp, zp, np, eps, woff, wp, sp, op, N, K, s); break;
    }
    return ik_launch_status();
}

// ---- core-op fusion phase 1 (docs/results/2026-10-19-core-op-fusion-p1.md): P5 and P4b -----------------------------------
// P5 rmsnorm_add_matmul_fp8_pair: the attention block's input norm (rmsnorm_add) and its k / v pair GEMV as one launch. Every
// block recomputes the norm of the one row in its prologue (norm_add_stage), block 0 alone stores `sum` and `normed`; the GEMV
// is fp8_gemv_pair_kernel's (Q = 2 staging, the default) on the normalised row in LDS. Each output row is one warp's
// reduction, independent of the rows per warp and of the block count, so the bits equal ik_rmsnorm_add_bf16 followed by
// ik_matmul_fp8_pair_gemv.
template <int R>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_norm_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res, const ik_bf16* __restrict__ nw,
                          float eps, float woff, ik_bf16* __restrict__ normed, ik_bf16* __restrict__ sum,
                          const uint8_t* __restrict__ w0, const float* __restrict__ sc0, ik_bf16* __restrict__ o0, int N0,
                          const uint8_t* __restrict__ w1, const float* __restrict__ sc1, ik_bf16* __restrict__ o1, int N1,
                          int K) {
    constexpr int per = kWarps * R;
    const int nb0 = (N0 + per - 1) / per;
    const bool writer = blockIdx.x == 0;
    if ((int)blockIdx.x < nb0)
        fp8_gemv_body<1, R, ik_bf16, 2, kWarps, 3>(x, w0, sc0, o0, N0, K, (int64_t)blockIdx.x, res, nw, eps, woff, normed, sum,
                                                   writer);
    else
        fp8_gemv_body<1, R, ik_bf16, 2, kWarps, 3>(x, w1, sc1, o1, N1, K, (int64_t)blockIdx.x - nb0, res, nw, eps, woff, normed,
                                                   sum, writer);
}

// x, res [1, K], nw [K], w0 [N0, K], w1 [N1, K] fp8 + scales, normed / sum [1, K], o0 [1, N0], o1 [1, N1]; bf16, 16-byte
// aligned.
int norm_pair(const ik_tensor* x, const ik_tensor* res, const ik_tensor* nw, float eps, float woff, const ik_tensor* w0,
              const ik_tensor* sc0, const ik_tensor* w1, const ik_tensor* sc1, ik_tensor* normed, ik_tensor* sum, ik_tensor* o0,
              ik_tensor* o1, void* stream, bool wmma = false) {
    IK_TRY(check(x, w0, sc0, o0));
    IK_TRY(check(x, w1, sc1, o1));
    IK_TRY(ik_check(res, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(nw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(normed, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(sum, IK_DTYPE_BF16, 2));
    if (o0->dtype != IK_DTYPE_BF16 || o1->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (M != 1 || K % 16 != 0 || K * 2 > 49152 || res->shape[0] != 1 || res->shape[1] != K || nw->shape[0] != K ||
        normed->shape[0] != 1 || normed->shape[1] != K || sum->shape[0] != 1 || sum->shape[1] != K)
        return IK_ERR_SHAPE;
    if (!ik_aligned16(res->data) || !ik_aligned16(nw->data) || !ik_aligned16(normed->data) || !ik_aligned16(sum->data))
        return IK_ERR_SHAPE;
    if (K == 0 || N0 == 0 || N1 == 0) return IK_ERR_SHAPE;  // (nothing the model produces; the unfused ops handle empty shapes)
    if (N0 > 0x7fffffffLL || N1 > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (wmma) {
        F8rCfg c;
        // the norm runs once, in one window (it writes `normed` / `sum`): size the window for the weight-only form, which stages the row as
    // bf16 (kw / 2 elements), i.e. ask for a window of 2 * K rounded up to the chunk (K <= 2048)
    if (!f8r_cfg(1, 2 * ((K + 255) / 256 * 256), true, (size_t)K * 2, false, 1, &c)) return IK_ERR_SHAPE;
        const unsigned grid = (unsigned)((N0 + 15) / 16 + (N1 + 15) / 16);
        if (c.nj == 16)
            f8r_norm_pair_kernel<16><<<grid, kF8rThreads, c.smem, IK_STREAM(stream)>>>(
                (const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)nw->data, eps, woff, (ik_bf16*)normed->data,
                (ik_bf16*)sum->data, (const uint8_t*)w0->data, (const float*)sc0->data, (ik_bf16*)o0->data, (int)N0,
                (const uint8_t*)w1->data, (const float*)sc1->data, (ik_bf16*)o1->data, (int)N1, (int)K, c.kw);
        else
            f8r_norm_pair_kernel<8><<<grid, kF8rThreads, c.smem, IK_STREAM(stream)>>>(
                (const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)nw->data, eps, woff, (ik_bf16*)normed->data,
                (ik_bf16*)sum->data, (const uint8_t*)w0->data, (const float*)sc0->data, (ik_bf16*)o0->data, (int)N0,
                (const uint8_t*)w1->data, (const float*)sc1->data, (ik_bf16*)o1->data, (int)N1, (int)K, c.kw);
        return ik_launch_status();
    }
    constexpr int R = 1, per = kWarps * R;  // fp8_gemv_pair_kernel's default shape (pair_cfg: one row per warp, eight warps)
    const unsigned grid = (unsigned)((N0 + per - 1) / per + (N1 + per - 1) / per);
    fp8_gemv_norm_pair_kernel<R><<<grid, kWarps * IK_WARP, (size_t)K * 2, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)nw->data, eps, woff, (ik_bf16*)normed->data,
        (ik_bf16*)sum->data, (const uint8_t*)w0->data, (const float*)sc0->data, (ik_bf16*)o0->data, (int)N0,
        (const uint8_t*)w1->data, (const float*)sc1->data, (ik_bf16*)o1->data, (int)N1, (int)K);
    return ik_launch_status();
}

// P4b sigmoid_mul_matmul_fp8: the attention output gate and o_proj as one launch. Every block computes sigmoid_mul of the row
// as it stages it (sigmoid_mul_chunk); the GEMV is matmul_fp8_gemv_v2's (Q = 2, rows per warp as gemv_rows picks them), so
// the bits equal ik_sigmoid_mul_bf16 followed by ik_matmul_fp8_gemv_v2.
template <int R>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_sigmoid_kernel(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ gate, const uint8_t* __restrict__ w,
                        const float* __restrict__ sc, ik_bf16* __restrict__ out, int N, int K) {
    fp8_gemv_body<1, R, ik_bf16, 2, kWarps, 4>(a, w, sc, out, N, K, (int64_t)blockIdx.x, gate);
}

template <int R>
void sigmoid_run(const ik_bf16* a, const ik_bf16* g, const uint8_t* w, const float* sc, ik_bf16* out, int64_t N, int64_t K,
                 hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps * R - 1) / (kWarps * R));
    fp8_gemv_sigmoid_kernel<R><<<grid, kWarps * IK_WARP, (size_t)K * 2, s>>>(a, g, w, sc, out, (int)N, (int)K);
}

// a, gate [1, K] bf16 (16-byte aligned), w [N, K] fp8 + scales, out [1, N] bf16.
int sigmoid_matmul(const ik_tensor* a, const ik_tensor* gate, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                   void* stream, bool wmma = false) {
    IK_TRY(check(a, w, scales, out));
    IK_TRY(ik_check(gate, IK_DTYPE_BF16, 2));
    if (out->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    const int64_t M = a->shape[0], K = a->shape[1], N = w->shape[0];
    if (M != 1 || K % 16 != 0 || K * 2 > 49152 || gate->shape[0] != M || gate->shape[1] != K) return IK_ERR_SHAPE;
    if (!ik_aligned16(gate->data)) return IK_ERR_SHAPE;
    if (N == 0 || K == 0) return IK_ERR_SHAPE;  // (nothing the model produces; the unfused ops handle empty shapes)
    if (N > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16 *ap = (const ik_bf16*)a->data, *gp = (const ik_bf16*)gate->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    ik_bf16* op = (ik_bf16*)out->data;
    if (wmma) {
        F8rCfg c;
        if (!f8r_cfg(1, K, true, 0, false, 1, &c)) return IK_ERR_SHAPE;
        const unsigned grid = (unsigned)((N + 15) / 16);
        if (c.nj == 16) f8r_sigmoid_kernel<16><<<grid, kF8rThreads, c.smem, s>>>(ap, gp, wp, sp, op, (int)N, (int)K, c.kw);
        else f8r_sigmoid_kernel<8><<<grid, kF8rThreads, c.smem, s>>>(ap, gp, wp, sp, op, (int)N, (int)K, c.kw);
        return ik_launch_status();
    }
    switch (gemv_rows(N, true)) {
        case 1: sigmoid_run<1>(ap, gp, wp, sp, op, N, K, s); break;
        case 4: sigmoid_run<4>(ap, gp, wp, sp, op, N, K, s); break;
        default: sigmoid_run<2>(ap, gp, wp, sp, op, N, K, s); break;
    }
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_rmsnorm_add_matmul_fp8_pair_gemv(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, float w_offset,
                                        const ik_tensor* w0, const ik_tensor* scales0, const ik_tensor* w1,
                                        const ik_tensor* scales1, ik_tensor* normed, ik_tensor* sum, ik_tensor* out0,
                                        ik_tensor* out1, void* stream) {
    return norm_pair(x, res, w, eps, w_offset, w0, scales0, w1, scales1, normed, sum, out0, out1, stream);
}

int ik_sigmoid_mul_matmul_fp8_gemv(const ik_tensor* a, const ik_tensor* gate, const ik_tensor* w, const ik_tensor* scales,
                                   ik_tensor* out, void* stream) {
    return sigmoid_matmul(a, gate, w, scales, out, stream);
}

int ik_gated_rmsnorm_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* z, const ik_tensor* nw, float eps, float w_offset,
                                     const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return gated_matmul(x, z, nw, eps, w_offset, w, scales, out, stream);
}

int ik_matmul_fp8_pair_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* scales0, const ik_tensor* w1,
                            const ik_tensor* scales1, ik_tensor* out0, ik_tensor* out1, void* stream) {
    return pair(x, w0, scales0, w1, scales1, out0, out1, stream);
}
int ik_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, 1);
}
int ik_matmul_fp8_gemv_v2(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, 2);
}
int ik_matmul_fp8_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch_rows(x, w, scales, out, stream);
}
int ik_matmul_fp8_wmma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, const ik_scratch* scratch,
                       void* stream) {
    return launch_wmma(x, w, scales, out, scratch, stream);
}
int ik_matmul_bf16_wmma(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return launch_bf16_wmma(x, w, out, stream);
}
int ik_matmul_bf16_wmma_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return launch_bf16_wmma(x, w, out, stream);  // the same launcher: rows_exact registration of one family
}
int ik_matmul_pair_bf16_wmma(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* out0, ik_tensor* out1, void* stream) {
    return bf16_pair_wmma(x, w0, w1, out0, out1, stream);
}
int ik_rmsnorm_add_matmul_pair_bf16_wmma(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, float w_offset,
                                         const ik_tensor* w0, const ik_tensor* w1, ik_tensor* normed, ik_tensor* sum, ik_tensor* out0,
                                         ik_tensor* out1, void* stream) {
    return bf16_norm_pair_wmma(x, res, w, eps, w_offset, w0, w1, normed, sum, out0, out1, stream);
}
int ik_matmul_fp8_wmma_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, const ik_scratch* scratch,
                            void* stream) {
    return launch_wmma(x, w, scales, out, scratch, stream);  // the same launcher: rows_exact registration of one family
}
int ik_matmul_fp8_pair_wmma(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* scales0, const ik_tensor* w1,
                            const ik_tensor* scales1, ik_tensor* out0, ik_tensor* out1, void* stream) {
    return pair(x, w0, scales0, w1, scales1, out0, out1, stream, true);
}
int ik_gated_rmsnorm_matmul_fp8_wmma(const ik_tensor* x, const ik_tensor* z, const ik_tensor* nw, float eps, float w_offset,
                                     const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return gated_matmul(x, z, nw, eps, w_offset, w, scales, out, stream, true);
}
int ik_rmsnorm_add_matmul_fp8_pair_wmma(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, float w_offset,
                                        const ik_tensor* w0, const ik_tensor* scales0, const ik_tensor* w1,
                                        const ik_tensor* scales1, ik_tensor* normed, ik_tensor* sum, ik_tensor* out0,
                                        ik_tensor* out1, void* stream) {
    return norm_pair(x, res, w, eps, w_offset, w0, scales0, w1, scales1, normed, sum, out0, out1, stream, true);
}
int ik_sigmoid_mul_matmul_fp8_wmma(const ik_tensor* a, const ik_tensor* gate, const ik_tensor* w, const ik_tensor* scales,
                                   ik_tensor* out, void* stream) {
    return sigmoid_matmul(a, gate, w, scales, out, stream, true);
}
int ik_matmul_fp8_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, 0);
}
int ik_matmul_fp8_mma_v2(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                         const ik_scratch* scratch, void* stream) {
    return launch_v2(x, w, scales, out, scratch, stream);
}
int ik_matmul_fp8_mma_f8(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                         const ik_scratch* scratch, void* stream) {
    return launch_f8(x, w, scales, out, scratch, stream);
}

}  // extern "C"
