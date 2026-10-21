// matmul_exl3: out[M,N] = x[M,K] @ W, W = diag(suh) H_K W_q H_N diag(svh) (EXL3 trellis weights, any codebook and width).
//
//   x       [M,K]            bf16 / f16 / f32
//   trellis i16 [K/16,N/16,8*k2]   one 16x16 tile of W_q per (k tile, n tile), 4*k2 32-bit words each (k2 = 2 * bits)
//   suh     f16 [K]          input-side scales;   svh f16 [N]: output-side scales
//   cb      i32 [1] (device) codebook: 0 = 3inst, 1 = mcg, 2 = mul1
//   out     [M,N]            bf16 / f16 / f32
//
// y = ((H (x * suh)) @ W_q) H * svh with H the 128-point Sylvester Hadamard / sqrt(128), blockwise. Two launches:
// rot_in (x * suh, H, rounded to fp16 into the scratch) and linear (one block per 128 output columns: its warps split
// K, every warp decodes its 16x16 tiles of W_q straight into mma.m16n8k16 B fragments, fp16 inputs, fp32 accumulators;
// the warps' sums are added in warp order and H, svh and the output rounding finish the rows). The codebook is read on
// the device at kernel entry and dispatched with one uniform branch into code templated on it, so the launch never
// synchronises; k2 comes from the trellis shape (a template parameter). Split-K is not used (SK = 1): a block's K range
// and the order of every sum depend on (K, N) only, so a row's bits do not depend on M or the other rows.
//
// The decode and linear kernels are ported from TensorFold (tensorfold/cuda/exl3/{decode.cuh,linear.cu}, MIT, see
// LICENSES/TensorFold-MIT.txt), which follows ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp). Changes: torch-free
// (ik ABI, ik dtype codes), the codebook is read on the device, no bias, no split-K, the trellis strides are the
// stored layout's.

#include "ik_launch.h"
#include "ik_quant.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

enum Codebook : int { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2 };

// ---- decode (tensorfold/cuda/exl3/decode.cuh) ------------------------------------------------------------------

// 32-bit words of a tile: 256 values of K2 / 2 bits.
template <int K2>
__host__ __device__ constexpr int tile_words() {
    return 4 * K2;
}

// E(p): one past the last stream bit of value p's 16-bit window.
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// Words a lane reads for its 8 values: the most any lane's windows span (2, or 3 for 3.5 bits and 5 to 8 bits).
template <int K2>
__host__ __device__ constexpr int lane_words() {
    int most = 0;
    for (int l = 0; l < 32; ++l) {
        const int first = (stream_end<K2>(8 * l) - 16 + 128 * K2) % 32;
        const int need = first + stream_end<K2>(8 * l + 7) - stream_end<K2>(8 * l) + 16;
        most = need > most ? need : most;
    }
    return (most + 31) / 32;
}

// The index of the first word lane `lane` reads, and the bit offset of its first window in that word.
template <int K2>
__device__ __forceinline__ void lane_start(int lane, int& word, int& offset) {
    const int first = 4 * lane * K2 + K2 / 2 - 16 + 128 * K2;  // E(8 lane) - 16, made non-negative
    word = (first >> 5) % tile_words<K2>();
    offset = first & 31;
}

// The same words from global memory through the read-only cache (tile: the tile's first 32-bit word).
template <int K2>
__device__ __forceinline__ void ldg_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = __ldg(tile + (word + i) % tile_words<K2>());
}

// Bits [d, d + 16) of the 96-bit big-endian stream hi:mid:lo (d a compile-time constant after unrolling).
__device__ __forceinline__ uint32_t window16(uint32_t hi, uint32_t mid, uint32_t lo, int d) {
    if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
    if (d < 32) return __funnelshift_l(mid, hi, d) >> 16;
    if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
    return __funnelshift_l(lo, mid, d - 32) >> 16;
}

// Two codebook values from two 16-bit states, as a half2 in a uint32 (the first state in the low half).
template <int CB>
__device__ __forceinline__ uint32_t decode2(uint32_t s0, uint32_t s1) {
    if constexpr (CB == CB_MUL1) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t h0 = __dp4a(x0, 0x01010101u, 0x6400u);  // fp16 bits of 1024 + the byte sum
        const uint32_t h1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hh = (h0 & 0xffffu) | (h1 << 16);
        const half2 r = __hfma2(*reinterpret_cast<const half2*>(&hh), __half2half2(__ushort_as_half(0x1eee)),
                                __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<const uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == CB_MCG) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const uint32_t lo = __byte_perm(x0, x1, 0x5410);
        const uint32_t hi = __byte_perm(x0, x1, 0x7632);
        const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
        return *reinterpret_cast<const uint32_t*>(&r);
    }
}

// This lane's 8 values of a tile as the B fragments of the tile's two n8 halves (columns 0-7 in b0, 8-15 in b1).
template <int K2, int CB>
__device__ __forceinline__ void decode_lane(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&b0)[2],
                                            uint32_t (&b1)[2]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const uint32_t c = lane_words<K2>() > 2 ? w[lane_words<K2>() > 2 ? 2 : 0] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], offset);  // the stream from the lane's first window on
    const uint32_t mid = __funnelshift_l(c, w[1], offset);
    const uint32_t lo = c << offset;
    uint32_t s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = window16(hi, mid, lo, stream_end<K2>(j) - stream_end<K2>(0));
    b0[0] = decode2<CB>(s[0], s[1]);
    b0[1] = decode2<CB>(s[2], s[3]);
    b1[0] = decode2<CB>(s[4], s[5]);
    b1[1] = decode2<CB>(s[6], s[7]);
}

// mma.m16n8k16, fp16 inputs, fp32 accumulators: d += a @ b.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Unscaled Walsh-Hadamard transform of 128 fp32 values, 4 a lane, in a fixed butterfly order (multiply by 1/sqrt(128)).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

constexpr float HAD_SCALE = 0.08838834764831845f;  // 1 / sqrt(128)

// ---- linear (tensorfold/cuda/exl3/linear.cu) ---------------------------------------------------------------------

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == IK_DTYPE_F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == IK_DTYPE_BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == IK_DTYPE_F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == IK_DTYPE_BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
}

// A lane's words of one k step: at 1 and 2 bits the warp loads the step together and each lane takes its words by shuffle.
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

// Tile j's lane words from a loaded step; prev: the lane's first window starts in the word before its own.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;  // a window inside one word never reads w[1]
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

// x * suh, rotated blockwise, rounded to fp16 (the mma's input): one warp per 128 values of a row.
__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                     const half* __restrict__ suh, half* __restrict__ xh, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4(suh, IK_DTYPE_F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, IK_DTYPE_F16, (size_t)row * K + k, v);
}

// One block's work for codebook CB: output columns [128 nb, 128 nb + 128) of every row. T: the block's tiles at
// k tile i, n block nb = T + i * stride_k + nb * stride_nb.
template <int K2, int CB, int WK>
__device__ __forceinline__ void linear_body(const half* __restrict__ xh, const uint32_t* __restrict__ T,
                                            long long stride_k, long long stride_nb, const half* __restrict__ svh,
                                            void* __restrict__ y, int y_dtype, int M, int K, int N, float* red) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    const int RH = min(M, 8);  // rows of red a warp

    const int nb = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / WK;
    const int kt0 = warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0; m0 < M; m0 += 16) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();  // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            for (int r = warp; r < rn; r += WK) {
                float v[4];
                const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                for (int w = 1; w < WK; ++w) {
                    const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                    v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                }
                finish(v, lane, svh, col0 + 4 * lane);
                store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
            }
        }
        __syncthreads();  // red is reused in the next pass
    }
}

// An unusable codebook (a value outside 0..2, or one that does not exist at this width): the block's outputs are NaN,
// so the mistake shows instead of reading uninitialised memory.
template <int WK>
__device__ __forceinline__ void poison_body(void* __restrict__ y, int y_dtype, int M, int N) {
    const float nan = __int_as_float(0x7fc00000);
    const float v[4] = {nan, nan, nan, nan};
    const int lane = threadIdx.x & 31;
    for (int r = threadIdx.x >> 5; r < M; r += WK) store4(y, y_dtype, (size_t)r * N + blockIdx.x * 128 + 4 * lane, v);
}

// One program per 128 output columns; the codebook (device data) selects the body once, uniformly.
template <int K2, int WK>
__global__ void __launch_bounds__(WK * 32) linear_kernel(const half* __restrict__ xh, const uint32_t* __restrict__ T,
                                                         long long stride_k, long long stride_nb,
                                                         const half* __restrict__ svh, void* __restrict__ y,
                                                         int y_dtype, const int* __restrict__ cbp, int M, int K, int N) {
    extern __shared__ __align__(16) float red[];  // WK * min(M, 8) * 128 floats
    const int cb = __ldg(cbp);
    if constexpr (K2 % 2 == 1) {
        if (cb == CB_MUL1) linear_body<K2, CB_MUL1, WK>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, red);
        else poison_body<WK>(y, y_dtype, M, N);
    } else {
        if (cb == CB_3INST) linear_body<K2, CB_3INST, WK>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, red);
        else if (cb == CB_MCG) linear_body<K2, CB_MCG, WK>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, red);
        else if (cb == CB_MUL1) linear_body<K2, CB_MUL1, WK>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, red);
        else poison_body<WK>(y, y_dtype, M, N);
    }
}

// ---- prefill GEMM ------------------------------------------------------------------------------------------------
//
// The GEMV above decodes every weight tile again for each 16-row pass. The GEMM decodes each tile once per block of
// 16 * MT rows: a block owns 128 output columns (one n block) and 16 * MT rows; its 8 warps own one 16-column tile each
// and walk the whole K range together. The rotated input of the block's rows is staged through shared memory
// (cp.async, 4 k tiles = 64 values per stage, 3 stages) and so are the block's trellis words (the 8 warps' tiles of the
// 4 k tiles of a stage are 4 runs of 8 * 16 * K2 contiguous bytes). A warp decodes its k tile from shared memory into B
// fragments once and runs the MT row groups against them (ldmatrix A fragments, 2 mma per group).
//
// What made it fast (GB10, 4096 x 4096 x 1024 rows: 1410 us -> 580 us; clock64 per phase and warp): with 2 warps per
// scheduler and a barrier every stage, the warps are in the same phase together, so the ~70 dependent integer operations
// of a decode, the burst of cp.async requests and the mma took turns instead of overlapping (each was ~450 clocks of a
// ~2200 clock k tile). So (1) the decode of k tile q + 1 is written ahead of the mma of k tile q, which lets the compiler
// schedule its chain into the gaps of the tensor pipe, (2) the requests of a stage are spread over the 4 k tiles of the
// stage before it, and (3) the A fragments are loaded one group ahead. Staging the trellis words through shared memory
// is not what removed the stall (a one-k-tile register prefetch measured the same); it only keeps those words out of the
// registers.
//
// The bits are the GEMV's: the GEMV's warp w sums the k tiles [w * K/128, (w + 1) * K/128) in one mma chain from zero
// and the block adds the 8 chains in warp order. The GEMM keeps one chain per K/128 run of k tiles, adds a finished
// chain to a running total in the same order, and finishes with the same H, svh and rounding on the same fp32 values.
// So the two kernels give identical bits, and so does every tiling (MT, slices): a row depends on nothing but itself.

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// A fragment of mma.m16n8k16 (rows 0-15, k 0-15) from shared memory: lane l addresses row l % 16, k offset 8 * (l / 16).
__device__ __forceinline__ void ldmatrix_a(uint32_t (&a)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr)
                 : "memory");
}

constexpr int kGemmThreads = 256;  // 8 warps: one 16-column tile each
constexpr int kGemmLda = 72;       // halfs per staged row: 64 values + 8 padding (144 bytes: ldmatrix without conflicts)
constexpr int kGemmStages = 3;
constexpr int kGemmPassRows = 64;  // rows the epilogue moves through shared memory at a time
constexpr int kGemmLdr = 136;      // floats per epilogue row: 128 + 8 padding

// Shared memory: the pipeline (A stages, then the trellis stages: 4 k tiles x 8 tiles x 4 K2 words each) or, after the
// loop, the epilogue's rows (they reuse the same bytes).
template <int K2, int MT>
constexpr size_t gemm_smem_bytes() {
    constexpr size_t pipe = (size_t)kGemmStages * (16 * MT * kGemmLda * 2 + 4 * 8 * tile_words<K2>() * 4);
    constexpr size_t epi = (size_t)(16 * MT < kGemmPassRows ? 16 * MT : kGemmPassRows) * kGemmLdr * 4;
    return pipe > epi ? pipe : epi;
}

// This lane's words of a tile held in shared memory (tile: its first word).
template <int K2>
__device__ __forceinline__ void lds_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
}

template <int K2, int CB, int MT>
__device__ __forceinline__ void gemm_body(const half* __restrict__ xh, const uint32_t* __restrict__ T,
                                          long long stride_k, long long stride_nb, const half* __restrict__ svh,
                                          void* __restrict__ y, int y_dtype, int M, int K, int N,
                                          unsigned char* smem) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int ROWS = 16 * MT;
    constexpr int STAGE_HALFS = ROWS * kGemmLda;
    constexpr int STAGE_WORDS = 4 * 8 * TW;
    constexpr int APIECES = ROWS * 8 / kGemmThreads;  // 16-byte pieces of a stage of the rotated input, per thread
    constexpr int BPIECES = 4 * 2 * TW;               // 16-byte pieces of a stage of trellis words, in all

    const int nb = blockIdx.x, row0 = blockIdx.y * ROWS, col0 = nb * 128;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, S = KT >> 2, C = KT >> 3;  // k tiles, stages of 4 k tiles, k tiles of one chain
    const uint32_t* tiles = T + (size_t)nb * stride_nb;  // this block's 8 tiles of k tile 0
    half* sA = reinterpret_cast<half*>(smem);
    uint32_t* sB = reinterpret_cast<uint32_t*>(smem + (size_t)kGemmStages * STAGE_HALFS * 2);

    // The requests of stage s are spread over the 4 k tiles of the stage before it is used (step q), so that they sit
    // between the mma of the warps instead of in a burst; the stage's group is committed after step 3.
    auto issue_part = [&](int s, int q) {
        if (s < S) {
            const uint32_t dsta = smem_u32(sA + (s % kGemmStages) * STAGE_HALFS);
#pragma unroll
            for (int j = q; j < APIECES; j += 4) {
                const int idx = j * kGemmThreads + tid;
                const int r = idx >> 3, c = idx & 7;
                const int gr = row0 + r < M ? row0 + r : M - 1;  // rows past M read inside the buffer, their outputs are dropped
                cp_async16(dsta + (uint32_t)(r * kGemmLda + c * 8) * 2, xh + (size_t)gr * K + s * 64 + c * 8);
            }
            if (q == 0) {
                const uint32_t dstb = smem_u32(sB + (s % kGemmStages) * STAGE_WORDS);
                for (int idx = tid; idx < BPIECES; idx += kGemmThreads) {  // 2 * TW pieces per k tile
                    const int k4 = idx / (2 * TW), c = idx % (2 * TW);
                    cp_async16(dstb + (uint32_t)(k4 * 8 * TW + c * 4) * 4, tiles + (size_t)(s * 4 + k4) * stride_k + c * 4);
                }
            }
        }
    };

    float acc[MT][2][4], tot[MT][2][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = tot[i][h][c] = 0.f;

#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int q = 0; q < 4; ++q) issue_part(s, q);
        cp_commit();
    }

    int kt = 0, ce = C;  // ce: the k tile count at which the running chain ends
#pragma unroll 1
    for (int s = 0; s < S; ++s) {
        cp_wait<1>();
        __syncthreads();  // stage s has landed; every warp is past stage s - 1, whose buffer the requests below overwrite
        const uint32_t abase =
            smem_u32(sA + (s % kGemmStages) * STAGE_HALFS) + (uint32_t)((lane & 15) * kGemmLda + (lane >> 4) * 8) * 2;
        const uint32_t* bstage = sB + (s % kGemmStages) * STAGE_WORDS + warp * TW;

        // The decode of k tile q + 1 is written ahead of the mma of k tile q: its long chain of dependent integer
        // operations then runs in the gaps of the tensor pipe instead of after it.
        uint32_t b0[2], b1[2];
        {
            uint32_t w[LW];
            lds_lane_words<K2>(bstage, lane, w);
            decode_lane<K2, CB>(w, lane, b0, b1);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t n0[2], n1[2];
            if (q < 3) {
                uint32_t w[LW];
                lds_lane_words<K2>(bstage + (q + 1) * 8 * TW, lane, w);
                decode_lane<K2, CB>(w, lane, n0, n1);
            }
            uint32_t a[2][4];
            ldmatrix_a(a[0], abase + (uint32_t)(q * 16) * 2);
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i + 1 < MT) ldmatrix_a(a[(i + 1) & 1], abase + (uint32_t)((i + 1) * 16 * kGemmLda + q * 16) * 2);
                mma16816(acc[i][0], a[i & 1], b0);
                mma16816(acc[i][1], a[i & 1], b1);
            }
            issue_part(s + 2, q);
            ++kt;
            if (kt == ce) {  // a chain is done: add it to the total (the GEMV's warp-order sum), start the next
                const bool first = ce == C;
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            tot[i][h][c] = first ? acc[i][h][c] : tot[i][h][c] + acc[i][h][c];
                            acc[i][h][c] = 0.f;
                        }
                ce += C;
            }
            if (q < 3) {
                b0[0] = n0[0]; b0[1] = n0[1]; b1[0] = n1[0]; b1[1] = n1[1];
            }
        }
        cp_commit();
    }
    cp_wait<0>();

    // H, svh and the output rounding, kGemmPassRows rows at a time through shared memory (the pipeline buffers are reused)
    constexpr int PR = ROWS < kGemmPassRows ? ROWS : kGemmPassRows;
    constexpr int GP = PR / 16;
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int hp = 0; hp < ROWS / PR; ++hp) {
        __syncthreads();  // the pipeline buffers / the previous pass's rows are free
#pragma unroll
        for (int i = 0; i < GP; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = warp * 16 + h * 8 + 2 * t;
                *reinterpret_cast<float2*>(red + (i * 16 + g) * kGemmLdr + col) =
                    make_float2(tot[hp * GP + i][h][0], tot[hp * GP + i][h][1]);
                *reinterpret_cast<float2*>(red + (i * 16 + g + 8) * kGemmLdr + col) =
                    make_float2(tot[hp * GP + i][h][2], tot[hp * GP + i][h][3]);
            }
        __syncthreads();
        for (int r = warp; r < PR; r += kGemmThreads / 32) {
            const int row = row0 + hp * PR + r;
            if (row < M) {
                float v[4];
                const float4 u = *reinterpret_cast<const float4*>(red + r * kGemmLdr + 4 * lane);
                v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
                finish(v, lane, svh, col0 + 4 * lane);
                store4(y, y_dtype, (size_t)row * N + col0 + 4 * lane, v);
            }
        }
    }
}

// An unusable codebook: the block's outputs are NaN (as the GEMV's).
template <int MT>
__device__ __forceinline__ void gemm_poison(void* __restrict__ y, int y_dtype, int M, int N) {
    const float nan = __int_as_float(0x7fc00000);
    const float v[4] = {nan, nan, nan, nan};
    const int lane = threadIdx.x & 31;
    for (int r = threadIdx.x >> 5; r < 16 * MT; r += kGemmThreads / 32) {
        const int row = blockIdx.y * 16 * MT + r;
        if (row < M) store4(y, y_dtype, (size_t)row * N + blockIdx.x * 128 + 4 * lane, v);
    }
}

template <int K2, int MT>
__global__ void __launch_bounds__(kGemmThreads, MT >= 8 ? 1 : 2)
    gemm_kernel(const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
                const half* __restrict__ svh, void* __restrict__ y, int y_dtype, const int* __restrict__ cbp, int M,
                int K, int N) {
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    const int cb = __ldg(cbp);
    if constexpr (K2 % 2 == 1) {
        if (cb == CB_MUL1) gemm_body<K2, CB_MUL1, MT>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, gemm_smem);
        else gemm_poison<MT>(y, y_dtype, M, N);
    } else {
        if (cb == CB_3INST) gemm_body<K2, CB_3INST, MT>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, gemm_smem);
        else if (cb == CB_MCG) gemm_body<K2, CB_MCG, MT>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, gemm_smem);
        else if (cb == CB_MUL1) gemm_body<K2, CB_MUL1, MT>(xh, T, stride_k, stride_nb, svh, y, y_dtype, M, K, N, gemm_smem);
        else gemm_poison<MT>(y, y_dtype, M, N);
    }
}

// ---- launcher ----------------------------------------------------------------------------------------------------

// Warps of a block: 8 (K / 16 is a multiple of 8 because K % 128 == 0, so they always divide the k tiles). A function of
// (K, N) only, so a row's bits do not depend on the batch. Measured on GB10 (M = 1, 4-bit): 4096 x 4096 41.4 us with 8
// warps vs 45.5 us with 4, 1024 x 4096 35.7 vs 38.9, 4096 x 12288 115.8 vs 129.2, 12288 x 4096 109.3 vs 104.6.
constexpr int kWarps = 8;

// Rows of one slice: the launcher takes any M and runs it as slices of at most this many rows (the rotated input of one
// slice is in the scratch at a time, slices follow each other on the stream).
constexpr int64_t kSliceRows = 128;

// The same for the GEMM: a longer slice keeps the grid full (N / 128 blocks per 16 * MT rows) at a larger scratch.
constexpr int64_t kGemmSliceRows = 512;

template <int K2>
int run(const half* xh, const uint32_t* T, long long stride_k, long long stride_nb, const half* svh, void* y,
        int y_dtype, const int* cbp, int M, int K, int N, cudaStream_t s) {
    const unsigned smem = (unsigned)(kWarps * (M < 8 ? M : 8) * 128 * sizeof(float));  // at most 32 KiB: no attribute needed
    linear_kernel<K2, kWarps><<<dim3((unsigned)(N / 128)), kWarps * 32, smem, s>>>(xh, T, stride_k, stride_nb, svh, y,
                                                                                   y_dtype, cbp, M, K, N);
    return ik_launch_status();
}

int launch_linear(int k2, const half* xh, const uint32_t* T, long long stride_k, long long stride_nb, const half* svh,
                  void* y, int y_dtype, const int* cbp, int M, int K, int N, cudaStream_t s) {
#define IK_EXL3_RUN(K2_) \
    case K2_:            \
        return run<K2_>(xh, T, stride_k, stride_nb, svh, y, y_dtype, cbp, M, K, N, s)
    switch (k2) {
        IK_EXL3_RUN(2);
        IK_EXL3_RUN(3);
        IK_EXL3_RUN(4);
        IK_EXL3_RUN(5);
        IK_EXL3_RUN(6);
        IK_EXL3_RUN(7);
        IK_EXL3_RUN(8);
        IK_EXL3_RUN(10);
        IK_EXL3_RUN(12);
        IK_EXL3_RUN(14);
        IK_EXL3_RUN(16);
        default:
            return IK_ERR_SHAPE;  // 9, 11, 13, 15 are no EXL3 width
    }
#undef IK_EXL3_RUN
}

template <int K2, int MT>
int run_gemm_mt(const half* xh, const uint32_t* T, long long stride_k, long long stride_nb, const half* svh, void* y,
                int y_dtype, const int* cbp, int M, int K, int N, cudaStream_t s) {
    constexpr size_t smem = gemm_smem_bytes<K2, MT>();
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_kernel<K2, MT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        return true;
    }();
    (void)attr_set;
    gemm_kernel<K2, MT><<<dim3((unsigned)(N / 128), (unsigned)((M + 16 * MT - 1) / (16 * MT))), kGemmThreads, smem, s>>>(
        xh, T, stride_k, stride_nb, svh, y, y_dtype, cbp, M, K, N);
    return ik_launch_status();
}

// Rows per block: 128 rows decode each tile once per 128 rows, 64 rows fill the SMs when the grid is small. The choice
// changes no bit of the output (see above), only the speed: 128-row blocks from two waves of blocks on (about 2 per SM).
template <int K2>
int run_gemm(const half* xh, const uint32_t* T, long long stride_k, long long stride_nb, const half* svh, void* y,
             int y_dtype, const int* cbp, int M, int K, int N, cudaStream_t s) {
    const long long blocks8 = (long long)(N / 128) * ((M + 127) / 128);
    const bool wide = blocks8 >= 96;
    return wide ? run_gemm_mt<K2, 8>(xh, T, stride_k, stride_nb, svh, y, y_dtype, cbp, M, K, N, s)
                : run_gemm_mt<K2, 4>(xh, T, stride_k, stride_nb, svh, y, y_dtype, cbp, M, K, N, s);
}

int launch_gemm(int k2, const half* xh, const uint32_t* T, long long stride_k, long long stride_nb, const half* svh,
                void* y, int y_dtype, const int* cbp, int M, int K, int N, cudaStream_t s) {
#define IK_EXL3_GEMM(K2_) \
    case K2_:             \
        return run_gemm<K2_>(xh, T, stride_k, stride_nb, svh, y, y_dtype, cbp, M, K, N, s)
    switch (k2) {
        IK_EXL3_GEMM(2);
        IK_EXL3_GEMM(3);
        IK_EXL3_GEMM(4);
        IK_EXL3_GEMM(5);
        IK_EXL3_GEMM(6);
        IK_EXL3_GEMM(7);
        IK_EXL3_GEMM(8);
        IK_EXL3_GEMM(10);
        IK_EXL3_GEMM(12);
        IK_EXL3_GEMM(14);
        IK_EXL3_GEMM(16);
        default:
            return IK_ERR_SHAPE;
    }
#undef IK_EXL3_GEMM
}

bool is_act(int dtype) { return dtype == IK_DTYPE_BF16 || dtype == IK_DTYPE_F16 || dtype == IK_DTYPE_F32; }

}  // namespace

// Both launchers: the GEMV runs the rows in slices of kSliceRows, the GEMM in slices of kGemmSliceRows; each slice is a
// rot_in and a linear / gemm launch.
static int exl3_launch(bool gemm, const ik_tensor* x, const ik_tensor* trellis, const ik_tensor* suh, const ik_tensor* svh,
                       const ik_tensor* cb, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    if (!x || !out) return IK_ERR_INVALID_ARG;
    if (!is_act(x->dtype) || !is_act(out->dtype)) return IK_ERR_DTYPE;
    IK_TRY(ik_check_a(x, x->dtype, 2, 16));
    IK_TRY(ik_check_a(out, out->dtype, 2, 16));
    IK_TRY(ik_check_a(trellis, IK_DTYPE_I16, 3, 4));
    IK_TRY(ik_check_a(suh, IK_DTYPE_F16, 1, 8));
    IK_TRY(ik_check_a(svh, IK_DTYPE_F16, 1, 8));
    IK_TRY(ik_check_a(cb, IK_DTYPE_I32, 1, 4));
    const int64_t M = x->shape[0], K = x->shape[1];
    const int64_t kt = trellis->shape[0], nt = trellis->shape[1], words = trellis->shape[2];
    const int64_t N = 16 * nt;
    // words = 8 * k2 (i16 per tile = 16 * k2 bytes / 2); k2 in 2..=16, or 3 / 5 / 7 (the mul1 codebook)
    if (words % 8 != 0 || words / 8 < 2 || words / 8 > 16) return IK_ERR_SHAPE;
    const int k2 = (int)(words / 8);
    if (16 * kt != K || K % 128 != 0 || N % 128 != 0 || suh->shape[0] != K || svh->shape[0] != N || cb->shape[0] != 1 ||
        out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0 || K == 0) return IK_OK;
    if (K > 0x3fffffffLL || N > 0x3fffffffLL) return IK_ERR_SHAPE;
    // the rotated input, fp16 [min(M, slice), K]
    // the GEMM stages the trellis with 16-byte cp.async; the GEMV gives the same bits from a trellis that is only 4-aligned
    if (gemm && !ik_aligned16(trellis->data)) gemm = false;
    const int64_t slice = gemm ? kGemmSliceRows : kSliceRows;
    const int64_t need = (M < slice ? M : slice) * K * 2;
    if (!scratch || !scratch->data || scratch->bytes < need || ((uintptr_t)scratch->data & 15u) != 0) return IK_ERR_INVALID_ARG;

    cudaStream_t s = IK_STREAM(stream);
    half* xh = (half*)scratch->data;
    const uint32_t* T = (const uint32_t*)trellis->data;
    const long long stride_k = (long long)nt * 4 * k2, stride_nb = 8LL * 4 * k2;
    const half* sv = (const half*)svh->data;
    const int* cbp = (const int*)cb->data;
    const int64_t xsize = x->dtype == IK_DTYPE_F32 ? 4 : 2, osize = out->dtype == IK_DTYPE_F32 ? 4 : 2;
    for (int64_t r0 = 0; r0 < M; r0 += slice) {
        const int rows = (int)(M - r0 < slice ? M - r0 : slice);
        rot_in_kernel<<<dim3((unsigned)((K / 128 + 3) / 4), (unsigned)rows), 128, 0, s>>>(
            (const char*)x->data + r0 * K * xsize, x->dtype, (const half*)suh->data, xh, (int)K);
        IK_TRY(ik_launch_status());
        char* yp = (char*)out->data + r0 * N * osize;
        IK_TRY(gemm ? launch_gemm(k2, xh, T, stride_k, stride_nb, sv, yp, out->dtype, cbp, rows, (int)K, (int)N, s)
                    : launch_linear(k2, xh, T, stride_k, stride_nb, sv, yp, out->dtype, cbp, rows, (int)K, (int)N, s));
    }
    return IK_OK;
}

extern "C" {

int ik_matmul_exl3_gemv(const ik_tensor* x, const ik_tensor* trellis, const ik_tensor* suh, const ik_tensor* svh,
                        const ik_tensor* cb, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return exl3_launch(false, x, trellis, suh, svh, cb, out, scratch, stream);
}

int ik_matmul_exl3_gemm(const ik_tensor* x, const ik_tensor* trellis, const ik_tensor* suh, const ik_tensor* svh,
                        const ik_tensor* cb, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return exl3_launch(true, x, trellis, suh, svh, cb, out, scratch, stream);
}

}  // extern "C"
