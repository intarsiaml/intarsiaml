// NVFP4 W4A4 grouped tensor-core MoE experts for prefill: the expert GEMMs of moe_grouped.cu on the block-scaled FP4
// tensor core of sm_120 / sm_121 (docs/results/2026-10-17-cuda-moe-nvfp4-mma.md).
//
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3  (SASS OMMA.SF.16864)
//
// The checkpoint's weights are already in the operand format (e2m1 + one ue4m3 scale per 16 k), so the weight tile goes
// from global memory through shared memory into the B fragments with no dequantisation. The activations are quantised
// to the same format:
//
//   quant     per row of x [T,H] (and of the sorted h [P,I]): g = 2688 / amax(row), per 16-wide block
//             s = e4m3(amax(block) * g / 6) and e2m1(x * g / s) (cvt.rn.satfinite, the vLLM/FlashInfer recipe with a
//             per-row instead of per-tensor global scale). Dequantised value = e2m1 * s / g.
//   gate_up   same tiling as moe_grouped.cu (64 gate + 64 up columns per block, K = H); the epilogue multiplies the
//             accumulator by amax/2688 (undo g) and the expert's global weight scale, then SiLU-GLU with the reference
//             rounding points and writes h in sorted pair order (bf16).
//   quant     h -> e2m1 + ue4m3 + per-row g.
//   down      K = I, epilogue scales and scatters y[pair]; the reduce kernel is moe_grouped.cu's.
//
// Needs H % 128 == 0 and I % 128 == 0. Built for sm_121a / sm_120a only (build.rs: files named *_sm120a.cu); when it is
// not (IK_SM120A == 0) the launcher is a stub returning IK_ERR_UNSUPPORTED and the Rust side does not register it.
//
// Scratch layout (must match quantmoe.rs `moe_experts_nvfp4_scratch`; regions rounded to 256 bytes): the layout of
// moe_grouped.cu (pairs, offsets, ntiles, tiles, h bf16 [P,I], y bf16 [P,H]) followed by
//   xq u8 [T,H/2] | xs e4m3 [T,H/16] | xg f32 [T] | hq u8 [P,I/2] | hs e4m3 [P,I/16] | hg f32 [P]   (g stored as 1/g)
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "moe_group.cuh"
#include <stdlib.h>

#if IK_SM120A && defined(__CUDA_ARCH__) && !(defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#error "IK_SM120A=1 needs -arch=sm_120a or sm_121a"
#endif

#if IK_SM120A
namespace {

constexpr int kWRows = 128;      // weight rows per block (64 gate + 64 up, or 128 down rows)
constexpr int kUnit = 128;       // k per pipeline unit (64 bytes of fp4 per row: the DRAM-efficient run length)
constexpr int kRawStride = 80;   // staged row stride in bytes: 64 payload + 16 pad (conflict-free 32-bit fragment loads)

// ---- block-scaled FP4 MMA --------------------------------------------------------------------------------------------

// D (16x8 f32) = A (16x64 e2m1) * B (64x8 e2m1) with per-16-k ue4m3 scales: sfa holds the 4 scales of one row of A,
// sfb the 4 of one column of B. Operand layout: see kernels/cuda_checks/omma_tile.cu (checked against a cpu reference).
IK_DEVI void omma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sfa,
                  uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

IK_DEVI void cp_async8(void* smem_dst, const void* gmem_src, int src_bytes) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(gmem_src), "r"(src_bytes) : "memory");
}

// ---- activation quantisation ---------------------------------------------------------------------------------------

// 8 floats -> 4 bytes of e2m1 pairs (element 2i in the low nibble of byte i), round to nearest even, saturating.
IK_DEVI uint32_t pack_e2m1x8(const float* f) {
    uint32_t r;
    asm("{\n.reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n}\n"
        : "=r"(r)
        : "f"(f[0]), "f"(f[1]), "f"(f[2]), "f"(f[3]), "f"(f[4]), "f"(f[5]), "f"(f[6]), "f"(f[7]));
    return r;
}

constexpr float kGlobalDen = 2688.f;  // 448 * 6: block scales then span the whole e4m3 range

// One warp per row of x [rows, K] bf16 -> q u8 [rows, K/2], s e4m3 [rows, K/16], ginv f32 [rows] (= amax / 2688).
// A row of zeros (or with amax <= 1e-30) gets zero data, zero scales and ginv 0. Deterministic.
__global__ void __launch_bounds__(128) act_quant_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ q,
                                                        uint8_t* __restrict__ s, float* __restrict__ ginv, int rows,
                                                        int K) {
    const int warp = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const ik_bf16* xr = x + (int64_t)warp * K;
    float amax = 0.f;
    for (int b = lane; b < K / 16; b += 32) {
        const uint4 u0 = *reinterpret_cast<const uint4*>(xr + b * 16);
        const uint4 u1 = *reinterpret_cast<const uint4*>(xr + b * 16 + 8);
        const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            amax = fmaxf(amax, fabsf(__uint_as_float(w[i] << 16)));
            amax = fmaxf(amax, fabsf(__uint_as_float(w[i] & 0xffff0000u)));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    // rows below 1e-30 count as zero: 2688 / amax would overflow for denormal amax and turn the row into NaNs
    const bool nz = amax > 1e-30f;
    const float gx = nz ? kGlobalDen / amax : 0.f;
    if (lane == 0) ginv[warp] = nz ? amax / kGlobalDen : 0.f;
    for (int b = lane; b < K / 16; b += 32) {
        const uint4 u0 = *reinterpret_cast<const uint4*>(xr + b * 16);
        const uint4 u1 = *reinterpret_cast<const uint4*>(xr + b * 16 + 8);
        const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
        float f[16];
        float bmax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            f[2 * i] = __uint_as_float(w[i] << 16);
            f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
            bmax = fmaxf(bmax, fmaxf(fabsf(f[2 * i]), fabsf(f[2 * i + 1])));
        }
        const uint8_t sb = ik_f_to_e4m3((bmax * gx) * (1.f / 6.f));
        const float sd = ik_e4m3_to_f(sb);
        const float inv = sd > 0.f ? __fdiv_rn(gx, sd) : 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) f[i] *= inv;
        uint2 o;
        o.x = pack_e2m1x8(f);
        o.y = pack_e2m1x8(f + 8);
        *reinterpret_cast<uint2*>(q + (int64_t)warp * (K / 2) + b * 8) = o;
        s[(int64_t)warp * (K / 16) + b] = sb;
    }
}

// ---- grouped GEMM --------------------------------------------------------------------------------------------------

// MODE 0: h_sorted[row, i] = silu-glu(xq[pair/K] . gate_i, xq[pair/K] . up_i)   (A = xq [T,H], out h [P,I] bf16)
// MODE 1: y[pair, n]       = hq[row] . down_n                                  (A = hq [P,I], out y [P,H] bf16)
// Block = 4 x WM warps: 4 along the 128 weight rows (32 rows each; in MODE 0 16 gate + the matching 16 up rows), WM
// along the pairs, each warp owns MT m16 tiles: BM = 16 * MT * WM pairs. A unit is 128 k: 64 bytes of fp4 + 8 bytes of
// scales per row, copied with cp.async into a ring of NS stages; one barrier per unit.
template <int MT, int WM, int MODE, int NS>
__global__ void __launch_bounds__(128 * WM, NS == 2 ? (WM == 1 ? 4 : 2) : 1)
    moe_nvfp4_kernel(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as, const float* __restrict__ ainv,
                     const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, const float* __restrict__ gscales,
                     const int* __restrict__ pairs, const int* __restrict__ offsets, const int2* __restrict__ tiles,
                     const int* __restrict__ ntiles, ik_bf16* __restrict__ out, int H, int I, int K) {
    constexpr int BM = 16 * MT * WM;
    constexpr int NTHREADS = 128 * WM;
    constexpr int RQ = 512 / NTHREADS;  // 16-byte weight chunks per thread and unit (128 rows x 4)
    constexpr int kWRaw = kWRows * kRawStride, kWSc = kWRows * 8, kARaw = BM * kRawStride, kASc = BM * 8;
    constexpr int kStage = kWRaw + kWSc + kARaw + kASc;
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    extern __shared__ __align__(16) unsigned char smem[];
    static_assert(NS >= 2 && NS <= 4, "stages");
    auto wraw = [&](int s) { return smem + s * kStage; };
    auto wsc = [&](int s) { return smem + s * kStage + kWRaw; };
    auto araw = [&](int s) { return smem + s * kStage + kWRaw + kWSc; };
    auto asc = [&](int s) { return smem + s * kStage + kWRaw + kWSc + kARaw; };

    const int2 td = tiles[tile];
    const int e = td.x, m0 = td.y;
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int rows = min(BM, cnt - m0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp & 3, wm = warp >> 2;
    const int g = lane >> 2, t = lane & 3;
    const int Kdim = MODE == 0 ? H : I;
    const int U = Kdim / kUnit;

    int n0;
    int64_t nrows;
    if (MODE == 0) {
        n0 = blockIdx.x * 64;
        nrows = 2 * (int64_t)I;
    } else {
        n0 = blockIdx.x * 128;
        nrows = H;
    }
    auto weight_row = [&](int r) { return MODE == 0 ? (r < 64 ? n0 + r : I + n0 + (r - 64)) : n0 + r; };

    // weight copies: chunk (tid + NTHREADS q) = row r, 16-byte part p of the unit's 64 bytes; scales: tid < 128 -> row tid
    const int rpart = tid & 3;
    const uint8_t* wsrc[RQ];
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
        const int r = (tid >> 2) + (NTHREADS / 4) * q;
        wsrc[q] = w + ((int64_t)e * nrows + weight_row(r)) * (Kdim / 2) + rpart * 16;
    }
    const uint8_t* ssrc = sc + ((int64_t)e * nrows + weight_row(tid < kWRows ? tid : 0)) * (Kdim / 16);
    // activation copies: chunk (tid + NTHREADS q) of the BM x 4 chunks = row (tid >> 2) + NTHREADS / 4 * q, part tid & 3;
    // scales: tid < BM -> row tid
    constexpr int AQ = (BM * 4 + NTHREADS - 1) / NTHREADS;
    const bool a_sc = tid < BM;
    int64_t arow[AQ], asrow = 0;
    bool a_ok[AQ], as_ok = false;
#pragma unroll
    for (int q = 0; q < AQ; ++q) {
        const int r = (tid >> 2) + (NTHREADS / 4) * q;
        arow[q] = 0;
        a_ok[q] = r < rows;
        if (a_ok[q]) arow[q] = MODE == 0 ? pairs[begin + m0 + r] / K : begin + m0 + r;
    }
    if (a_sc) {
        as_ok = tid < rows;
        if (as_ok) asrow = MODE == 0 ? pairs[begin + m0 + tid] / K : begin + m0 + tid;
    }
    auto issue = [&](int u, int stage) {
#pragma unroll
        for (int q = 0; q < RQ; ++q) {
            const int r = (tid >> 2) + (NTHREADS / 4) * q;
            ik_cp_async16(wraw(stage) + r * kRawStride + rpart * 16, wsrc[q] + u * 64, 16);
        }
        if (tid < kWRows) cp_async8(wsc(stage) + tid * 8, ssrc + u * 8, 8);
#pragma unroll
        for (int q = 0; q < AQ; ++q) {
            const int r = (tid >> 2) + (NTHREADS / 4) * q;
            if (r < BM)
                ik_cp_async16(araw(stage) + r * kRawStride + rpart * 16, aq + arow[q] * (Kdim / 2) + u * 64 + rpart * 16,
                              a_ok[q] ? 16 : 0);
        }
        if (a_sc) cp_async8(asc(stage) + tid * 8, as + asrow * (Kdim / 16) + u * 8, as_ok ? 8 : 0);
    };

    float acc[MT][4][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

    // smem row of the j-th n8 tile of this warp
    auto nrow = [&](int j) { return MODE == 0 ? (j < 2 ? wn * 16 + j * 8 : 64 + wn * 16 + (j - 2) * 8) : wn * 32 + j * 8; };

#pragma unroll
    for (int s0 = 0; s0 < NS - 1; ++s0) {
        if (s0 < U) issue(s0, s0);
        ik_cp_async_commit();
    }
    for (int u = 0; u < U; ++u) {
        ik_cp_async_wait_group<NS - 2>();  // units 0..u landed
        __syncthreads();                   // visible to all; everyone is done with the unit that slot (u + NS - 1) % NS held
        if (u + NS - 1 < U) issue(u + NS - 1, (u + NS - 1) % NS);
        ik_cp_async_commit();
        const unsigned char* wr = wraw(u % NS);
        const unsigned char* ws = wsc(u % NS);
        const unsigned char* ar_ = araw(u % NS);
        const unsigned char* as_ = asc(u % NS);
#pragma unroll
        for (int st = 0; st < 2; ++st) {  // two k64 steps per unit
            uint32_t b[4][2], sfb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = nrow(j) + g;
                b[j][0] = *reinterpret_cast<const uint32_t*>(wr + row * kRawStride + st * 32 + 4 * t);
                b[j][1] = *reinterpret_cast<const uint32_t*>(wr + row * kRawStride + st * 32 + 16 + 4 * t);
                sfb[j] = t == 0 ? *reinterpret_cast<const uint32_t*>(ws + row * 8 + st * 4) : 0u;
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r0 = wm * 16 * MT + i * 16 + g;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ar_ + r0 * kRawStride + st * 32 + 4 * t);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ar_ + (r0 + 8) * kRawStride + st * 32 + 4 * t);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ar_ + r0 * kRawStride + st * 32 + 16 + 4 * t);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ar_ + (r0 + 8) * kRawStride + st * 32 + 16 + 4 * t);
                const uint32_t sfa = t < 2 ? *reinterpret_cast<const uint32_t*>(as_ + (r0 + 8 * t) * 8 + st * 4) : 0u;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float c[4] = {0.f, 0.f, 0.f, 0.f};
                    omma(c, a0, a1, a2, a3, b[j][0], b[j][1], sfa, sfb[j]);
#pragma unroll
                    for (int v = 0; v < 4; ++v) acc[i][j][v] += c[v];
                }
            }
        }
    }

    // ---- epilogue: undo the activation scale, apply the expert's weight scale, (MODE 0) SiLU-GLU, stage, write ----
    constexpr int NCOL = MODE == 0 ? 64 : 128;
    constexpr int OS = NCOL + 8;
    __syncthreads();
    ik_bf16* const ob = reinterpret_cast<ik_bf16*>(smem);  // [BM][OS], aliases the ring
    float ri[MT][2];  // 1/g of the rows this thread's accumulators belong to
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int rl = wm * 16 * MT + i * 16 + g + hh * 8;
            float v = 0.f;
            if (rl < rows) v = ainv[MODE == 0 ? pairs[begin + m0 + rl] / K : begin + m0 + rl];
            ri[i][hh] = v;
        }
    if (MODE == 0) {
        const float gg = gscales[e * 3 + 0], gu = gscales[e * 3 + 1];
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int rl = wm * 16 * MT + i * 16 + g + hh * 8;
                    const int cl = wn * 16 + j * 8 + t * 2;
                    float hv[2];
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const float gt = round_bf16(acc[i][j][2 * hh + v] * (ri[i][hh] * gg));
                        const float u = round_bf16(acc[i][j + 2][2 * hh + v] * (ri[i][hh] * gu));
                        const float a = round_bf16(gt / (1.f + expf(-gt)));
                        hv[v] = a * u;
                    }
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) = __floats2bfloat162_rn(hv[0], hv[1]);
                }
    } else {
        const float gd = gscales[e * 3 + 2];
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int rl = wm * 16 * MT + i * 16 + g + hh * 8;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int cl = wn * 32 + j * 8 + t * 2;
                    const float sc2 = ri[i][hh] * gd;
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) =
                        __floats2bfloat162_rn(acc[i][j][2 * hh] * sc2, acc[i][j][2 * hh + 1] * sc2);
                }
            }
    }
    __syncthreads();
    constexpr int CPR = NCOL / 8;
    for (int c = tid; c < BM * CPR; c += NTHREADS) {
        const int rl = c / CPR, part = c % CPR;
        if (rl >= rows) continue;
        ik_bf16* dst;
        if (MODE == 0) dst = out + (int64_t)(begin + m0 + rl) * I + n0 + part * 8;
        else dst = out + (int64_t)pairs[begin + m0 + rl] * H + n0 + part * 8;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(ob + rl * OS + part * 8);
    }
}

// ---- launchers -----------------------------------------------------------------------------------------------------

struct ScratchLayout {
    int64_t pairs, offsets, ntiles, tiles, h, y, xq, xs, xg, hq, hs, hg, total;
};

ScratchLayout scratch_layout(int64_t T, int64_t P, int64_t E, int64_t H, int64_t I) {
    ScratchLayout l;
    const int64_t max_tiles = (P + 15) / 16 + (P < E ? P : E);
    l.pairs = 0;
    l.offsets = l.pairs + r256(P * 4);
    l.ntiles = l.offsets + r256((E + 1) * 4);
    l.tiles = l.ntiles + 256;
    l.h = l.tiles + r256(max_tiles * 8);
    l.y = l.h + r256(P * I * 2);
    l.xq = l.y + r256(P * H * 2);
    l.xs = l.xq + r256(T * (H / 2));
    l.xg = l.xs + r256(T * (H / 16));
    l.hq = l.xg + r256(T * 4);
    l.hs = l.hq + r256(P * (I / 2));
    l.hg = l.hs + r256(P * (I / 16));
    l.total = l.hg + r256(P * 4);
    return l;
}

int pick_bm(int64_t P, int64_t E) {
    if (const char* v = getenv("INTARSIA_MOE_BM")) {
        const int b = atoi(v);
        if (b == 16 || b == 32 || b == 64) return b;
    }
    const double mu = (double)P / (double)E;
    return mu <= 10.0 ? 16 : (mu <= 21.0 ? 32 : 64);
}

template <int MT, int WM, int MODE, int NS>
void launch_gemm_ns(const uint8_t* aq, const uint8_t* as, const float* ainv, const uint8_t* w, const uint8_t* sc,
                 const float* gs, const int* pairs, const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* out,
                 unsigned tiles_ub, int H, int I, int K, cudaStream_t s) {
    constexpr int BM = 16 * MT * WM;
    constexpr size_t smem = (size_t)NS * (kWRows * kRawStride + kWRows * 8 + BM * kRawStride + BM * 8);
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(moe_nvfp4_kernel<MT, WM, MODE, NS>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        return true;
    }();
    (void)attr_set;
    const unsigned ny = MODE == 0 ? (unsigned)(I / 64) : (unsigned)(H / 128);
    moe_nvfp4_kernel<MT, WM, MODE, NS><<<dim3(ny, tiles_ub), 128 * WM, smem, s>>>(aq, as, ainv, w, sc, gs, pairs, offsets,
                                                                              tiles, ntiles, out, H, I, K);
}

// Pipeline depth: 2 stages (two blocks per SM; 3 or 4 stages leave one) by default, INTARSIA_NVFP4_STAGES=2|3|4
// overrides it (A/B experiments, docs/results/2026-10-17-cuda-moe-nvfp4-mma.md).
int pick_stages() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_NVFP4_STAGES");
        const int n = e ? atoi(e) : 2;
        return n >= 2 && n <= 4 ? n : 2;
    }();
    return v;
}

template <int MT, int WM, int MODE>
void launch_gemm(const uint8_t* aq, const uint8_t* as, const float* ainv, const uint8_t* w, const uint8_t* sc,
                 const float* gs, const int* pairs, const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* out,
                 unsigned tiles_ub, int H, int I, int K, cudaStream_t s) {
    switch (pick_stages()) {
        case 2: launch_gemm_ns<MT, WM, MODE, 2>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, s); break;
        case 4: launch_gemm_ns<MT, WM, MODE, 4>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, s); break;
        case 3: launch_gemm_ns<MT, WM, MODE, 3>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, s); break;
        default: launch_gemm_ns<MT, WM, MODE, 2>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, s); break;
    }
}

}  // namespace
#endif  // IK_SM120A

extern "C" {

int ik_moe_experts_nvfp4_mma_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                  const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                  const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                  const ik_scratch* scratch, void* stream) {
#if !IK_SM120A
    (void)x; (void)ids; (void)weights; (void)gate_up_w; (void)gate_up_s; (void)down_w; (void)down_s; (void)gscales;
    (void)out; (void)scratch; (void)stream;
    return IK_ERR_UNSUPPORTED;
#else
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(ids, IK_DTYPE_I32, 2, 4));
    IK_TRY(ik_check_a(weights, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(gate_up_w, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(gate_up_s, IK_DTYPE_FP8_E4M3, 3, 8));
    IK_TRY(ik_check_a(down_w, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(down_s, IK_DTYPE_FP8_E4M3, 3, 8));
    IK_TRY(ik_check_a(gscales, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(out, IK_DTYPE_BF16, 2, 2));
    const int64_t T = x->shape[0], H = x->shape[1], K = ids->shape[1];
    const int64_t E = gate_up_w->shape[0], I = gate_up_w->shape[1] / 2;
    if (H % 128 != 0 || I % 128 != 0 || gate_up_w->shape[1] != 2 * I || gate_up_w->shape[2] != H / 2 ||
        gate_up_s->shape[0] != E || gate_up_s->shape[1] != 2 * I || gate_up_s->shape[2] != H / 16 ||
        down_w->shape[0] != E || down_w->shape[1] != H || down_w->shape[2] != I / 2 || down_s->shape[0] != E ||
        down_s->shape[1] != H || down_s->shape[2] != I / 16 || gscales->shape[0] != E || gscales->shape[1] != 3 ||
        ids->shape[0] != T || weights->shape[0] != T || weights->shape[1] != K || out->shape[0] != T ||
        out->shape[1] != H)
        return IK_ERR_SHAPE;
    if (E < 1 || E > kMaxExperts || K < 1 || K > E) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    const int64_t P = T * K;
    if (P > 0x7fffffffLL / 2 || H > 0x7fffffffLL || I > 0x7fffffffLL) return IK_ERR_SHAPE;
    const ScratchLayout L = scratch_layout(T, P, E, H, I);
    if (!scratch || !scratch->data || scratch->bytes < L.total) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    unsigned char* base = (unsigned char*)scratch->data;
    int* pairs = (int*)(base + L.pairs);
    int* offsets = (int*)(base + L.offsets);
    int* ntiles = (int*)(base + L.ntiles);
    int2* tiles = (int2*)(base + L.tiles);
    ik_bf16* hbuf = (ik_bf16*)(base + L.h);
    ik_bf16* ybuf = (ik_bf16*)(base + L.y);
    uint8_t* xq = base + L.xq;
    uint8_t* xs = base + L.xs;
    float* xg = (float*)(base + L.xg);
    uint8_t* hq = base + L.hq;
    uint8_t* hs = base + L.hs;
    float* hg = (float*)(base + L.hg);
    cudaStream_t s = IK_STREAM(stream);

    const int BM = pick_bm(P, E);
    const unsigned tiles_ub = (unsigned)((P + BM - 1) / BM + (P < E ? P : E));
    moe_group_tiles_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), s>>>((const int*)ids->data, (int)P, (int)E, BM, pairs,
                                                                        offsets, ntiles, tiles);
    IK_TRY(ik_launch_status());
    act_quant_kernel<<<(unsigned)((T + 3) / 4), 128, 0, s>>>((const ik_bf16*)x->data, xq, xs, xg, (int)T, (int)H);
    IK_TRY(ik_launch_status());
    const uint8_t* guw = (const uint8_t*)gate_up_w->data;
    const uint8_t* gus = (const uint8_t*)gate_up_s->data;
    const uint8_t* dnw = (const uint8_t*)down_w->data;
    const uint8_t* dns = (const uint8_t*)down_s->data;
    const float* gs = (const float*)gscales->data;
    const unsigned qblocks = (unsigned)((P + 3) / 4);
#define IK_RUN(MT, WM)                                                                                               \
    launch_gemm<MT, WM, 0>(xq, xs, xg, guw, gus, gs, pairs, offsets, tiles, ntiles, hbuf, tiles_ub, (int)H, (int)I, \
                           (int)K, s);                                                                                \
    IK_TRY(ik_launch_status());                                                                                       \
    act_quant_kernel<<<qblocks, 128, 0, s>>>(hbuf, hq, hs, hg, (int)P, (int)I);                                       \
    IK_TRY(ik_launch_status());                                                                                       \
    launch_gemm<MT, WM, 1>(hq, hs, hg, dnw, dns, gs, pairs, offsets, tiles, ntiles, ybuf, tiles_ub, (int)H, (int)I,  \
                           (int)K, s);
    switch (BM) {
        case 16: IK_RUN(1, 1) break;
        case 32: IK_RUN(1, 2) break;
        default: IK_RUN(2, 2) break;
    }
#undef IK_RUN
    IK_TRY(ik_launch_status());
    moe_reduce_kernel<ik_bf16><<<dim3((unsigned)T, (unsigned)((H + 255) / 256)), 256, 0, s>>>(
        ybuf, (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)K);
    return ik_launch_status();
#endif
}

}  // extern "C"
