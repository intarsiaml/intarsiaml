// Grouped tensor-core MoE experts prefill, v2 (W4A16: fp4 weights dequantised in registers, bf16 activations).
// docs/results/2026-10-18-cuda-moe-w4a16.md. Same op, inputs, scratch layout and BITWISE the same results as
// moe_experts_mma_bf16 (moe_grouped.cu); only the schedule differs:
//
//   * no dequantised bf16 weight tile in shared memory: every warp reads the staged packed bytes of its own weight
//     rows and builds its B fragments in registers (prmt lookup -> bf16 -> bf16x2 multiply by the block scale; exact,
//     e2m1 * e4m3 has few significant bits), straight before the mma. Shared memory per block drops from 85 KB to
//     48 KB (BM = 64), so two blocks share an SM and one block's prologue/epilogue hides behind the other's loop;
//   * one warp owns distinct weight rows across ALL pairs of the tile (no second warp along M that would repeat the
//     dequantisation), and skips the m16 tiles that lie past the expert's last pair;
//   * the e4m3 block scales travel with the weights through the cp.async ring (8 bytes per row and 128-k unit)
//     instead of per-thread global loads.
//
// Numerics are unchanged: every k16 product sum is formed from a zero accumulator by one mma and added to the running
// f32 sum with a normal fp32 add, in k order; B values are the same bf16 numbers; the epilogue is moe_grouped.cu's.
//
// Pipeline per k64 step s: wait for group G_s (A(s) and, for even s, the packed unit s/2 with its scales) -> barrier ->
// issue G_{s+2} -> mma step s. A ring 3 slots, packed ring 2 units.
//
// Scratch layout: identical to moe_grouped.cu (intarsia_core::quant::ops::moe_experts_mma_scratch).
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "moe_group.cuh"
#include <cuda_bf16.h>
#include <stdlib.h>

namespace {

constexpr int BK = 64;         // k per pipeline step
constexpr int LDS = BK + 8;    // padded smem row stride of the A tile (bf16 elements): conflict-free ldmatrix
constexpr int kRawStride = 80; // bytes per staged row: 64 payload + 8 scales + 8 pad (16-byte reads stay conflict-free)

IK_DEVI void cp_async8(void* smem_dst, const void* gmem_src) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(d), "l"(gmem_src) : "memory");
}

// 4 fp4 values (the low 16 bits of q: nibble j = element j, sign in bit 3) -> bf16x2 {e0, e1} and {e2, e3}, unscaled.
// bf16 of the e2m1 magnitudes 0 .5 1 1.5 2 3 4 6 = 0x0000 3F00 3F80 3FC0 4000 4040 4080 40C0: high bytes by one prmt,
// low bytes by another, interleaved by two more; the sign bits are moved to bit 15 / 31 by a multiply.
IK_DEVI void fp4x4_to_bf16x2x2(uint32_t q, uint32_t& r0, uint32_t& r1) {
    const uint32_t sel = q & 0x7777u;
    const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
    const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
    r0 = __byte_perm(lo, hi, 0x5140u) | (((q & 0x88u) * 0x1001000u) & 0x80008000u);
    r1 = __byte_perm(lo, hi, 0x7362u) | (((q & 0x8800u) * 0x10010u) & 0x80008000u);
}

IK_DEVI uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
    __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
    __nv_bfloat162 r = __hmul2(x, y);
    return *reinterpret_cast<uint32_t*>(&r);
}

// {s, s} as bf16x2 from an e4m3 byte (exact)
IK_DEVI uint32_t scale_pair(float f) {
    __nv_bfloat162 r = __float2bfloat162_rn(f);
    return *reinterpret_cast<uint32_t*>(&r);
}

// ---- grouped GEMM ----------------------------------------------------------------------------------------------------

// MODE 0: h_sorted[row, i] = silu-glu(x[pair/K] . gate_i, x[pair/K] . up_i)   (a_src = x [T,H], out = h [P,I])
// MODE 1: y[pair, n]       = h_sorted[row] . down_n                          (a_src = h [P,I], out = y [P,H])
// Block = R / (8 JW) warps, each owning 8 * JW weight rows (MODE 0: JW/2 gate n8 tiles and the matching JW/2 up tiles) and
// all BM = 16 * MT pairs of the tile; R weight rows per block (MODE 0: R/2 gate + the matching R/2 up rows).
// Pipeline lookahead: A(s + LA) and the packed unit of step s + LR are requested at iteration s (LR even, LR >= LA).
template <int R, int MT, int JW, int MODE, int MINB, int LA, int LR>
__global__ void __launch_bounds__(R * 4 / JW, MINB)
moe_w4a16_kernel(const ik_bf16* __restrict__ a_src, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                 const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                 const int2* __restrict__ tiles, const int* __restrict__ ntiles, ik_bf16* __restrict__ out, int H, int I,
                 int K) {
    constexpr int BM = 16 * MT;
    constexpr int NT = R * 4 / JW;  // R / (8 JW) warps
    constexpr int JH = JW / 2;
    constexpr int RQ = R * 4 / NT;  // 16-byte packed chunks per thread and unit (R rows x 4)
    constexpr int NA = LA + 1, NR = LR / 2 + 1;
    static_assert(LR % 2 == 0 && LR >= LA && LA >= 1 && R * 4 % NT == 0 && BM % (NT / 8) == 0, "pipeline shape");
    const int tile = blockIdx.y;  // blocks of one tile (all its weight column tiles) are adjacent: the A rows stay in L2
    if (tile >= *ntiles) return;  // block-uniform: the grid is an upper bound
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int XS = BM * LDS;
    ik_bf16* const xs0 = reinterpret_cast<ik_bf16*>(smem_raw);
    auto xs = [&](int slot) { return xs0 + slot * XS; };  // A tile [BM][LDS], ring of NA
    unsigned char* const raw_base = smem_raw + (size_t)(NA * XS) * 2;
    auto rawu = [&](int slot) { return raw_base + slot * (R * kRawStride); };  // packed weights + scales, ring of NR units

    const int2 td = tiles[tile];
    const int e = td.x, m0 = td.y;
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int rows = min(BM, cnt - m0);  // 1..BM
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp;
    const int g = lane >> 2, t4 = lane & 3;
    const int Kdim = MODE == 0 ? H : I;
    const int S = Kdim / BK;   // 64-wide steps
    const int U = Kdim / 128;  // 128-wide packed units (Kdim % 128 == 0)

    int n0;
    int64_t nrows;
    if (MODE == 0) {
        n0 = blockIdx.x * (R / 2);
        nrows = 2 * (int64_t)I;
    } else {
        n0 = blockIdx.x * R;
        nrows = H;
    }
    auto weight_row = [&](int r) { return MODE == 0 ? (r < R / 2 ? n0 + r : I + n0 + (r - R / 2)) : n0 + r; };

    // packed-weight copy: chunk (tid + NT q) = row r, 16-byte part p of the unit's 64 bytes; threads < 128 also copy the
    // 8 scale bytes of row `tid`
    const int rpart = tid & 3;
    const uint8_t* rsrc[RQ];
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
        const int r = (tid >> 2) + (NT / 4) * q;
        rsrc[q] = w + ((int64_t)e * nrows + weight_row(r)) * (Kdim / 2) + rpart * 16;
    }
    const uint8_t* ssrc = sc + ((int64_t)e * nrows + weight_row(tid < R ? tid : 0)) * (Kdim / 16);
    auto issue_raw = [&](int u, int slot) {
#pragma unroll
        for (int q = 0; q < RQ; ++q) {
            const int r = (tid >> 2) + (NT / 4) * q;
            ik_cp_async16(rawu(slot) + r * kRawStride + rpart * 16, rsrc[q] + u * 64, 16);
        }
        if (tid < R) cp_async8(rawu(slot) + tid * kRawStride + 64, ssrc + u * 8);
    };

    // this thread's A rows: chunk (16 B = 8 bf16) c8 of rows r = (tid >> 3) + (NT / 8) q, q < BM / (NT / 8)
    constexpr int AQ = BM / (NT / 8);
    const int c8 = tid & 7;
    const ik_bf16* arow[AQ];
    bool avalid[AQ];
#pragma unroll
    for (int q = 0; q < AQ; ++q) {
        const int r = (tid >> 3) + (NT / 8) * q;
        avalid[q] = r < rows;
        arow[q] = a_src;  // a valid address for the zero-filled rows
        if (avalid[q]) {
            if (MODE == 0) arow[q] = a_src + (int64_t)(pairs[begin + m0 + r] / K) * H;
            else arow[q] = a_src + (int64_t)(begin + m0 + r) * I;
        }
    }
    auto issue_a = [&](int st, int slot) {
#pragma unroll
        for (int q = 0; q < AQ; ++q) {
            const int r = (tid >> 3) + (NT / 8) * q;
            ik_cp_async16(xs(slot) + r * LDS + c8 * 8, arow[q] + st * BK + c8 * 8, avalid[q] ? 16 : 0);
        }
    };
    auto issue_group = [&](int v) {  // the group of iteration v (v < 0: prologue)
        if (v + LA >= 0 && v + LA < S) issue_a(v + LA, (v + LA) % NA);
        if (!((v + LR) & 1) && (v + LR) / 2 < U) issue_raw((v + LR) / 2, ((v + LR) / 2) % NR);
        ik_cp_async_commit();
    };

    float acc[MT][JW][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < JW; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

    // smem row (within the 128 packed rows) of the j-th n8 tile of this warp
    auto nrow = [&](int j) {
        return MODE == 0 ? (j < JH ? wn * 8 * JH + j * 8 : R / 2 + wn * 8 * JH + (j - JH) * 8) : wn * 8 * JW + j * 8;
    };
    const uint32_t selt = (uint32_t)t4 | ((uint32_t)(4 + t4) << 4);

    for (int v = -LR; v < 0; ++v) issue_group(v);
    for (int s = 0; s < S; ++s) {
        ik_cp_async_wait_group<LA - 1>();  // A(s) and the packed unit of step s have landed
        __syncthreads();                   // visible to all; everyone is done with step s-1's buffers
        issue_group(s);
        const ik_bf16* x = xs(s % NA);
        const unsigned char* rb = rawu((s >> 1) % NR) + (s & 1) * 32;
        // packed words of this thread's rows (g) for the whole step, and their four block scales
        uint32_t wd[JW][8];
        uint32_t scs[JW];
#pragma unroll
        for (int j = 0; j < JW; ++j) {
            const unsigned char* rp = rb + (nrow(j) + g) * kRawStride;
            const uint4 q0 = *reinterpret_cast<const uint4*>(rp);
            const uint4 q1 = *reinterpret_cast<const uint4*>(rp + 16);
            wd[j][0] = q0.x; wd[j][1] = q0.y; wd[j][2] = q0.z; wd[j][3] = q0.w;
            wd[j][4] = q1.x; wd[j][5] = q1.y; wd[j][6] = q1.z; wd[j][7] = q1.w;
        }
#pragma unroll
        for (int j = 0; j < JW; ++j) {
            const unsigned char* rsp = rawu((s >> 1) % NR) + (nrow(j) + g) * kRawStride + 64 + (s & 1) * 4;
            scs[j] = *reinterpret_cast<const uint32_t*>(rsp);
        }
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            uint32_t b[JW][2];
#pragma unroll
            for (int j = 0; j < JW; ++j) {
                const uint32_t q = __byte_perm(wd[j][2 * kk], wd[j][2 * kk + 1], selt);
                uint32_t r0, r1;
                fp4x4_to_bf16x2x2(q, r0, r1);
                const uint32_t sp = scale_pair(ik_e4m3_to_f((uint8_t)((scs[j] >> (8 * kk)) & 0xFFu)));
                b[j][0] = mul_bf16x2(r0, sp);
                b[j][1] = mul_bf16x2(r1, sp);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i * 16 >= rows) continue;  // block-uniform: this m16 tile holds no pair
                uint32_t a0, a1, a2, a3;
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ik_ldmatrix_x4(a0, a1, a2, a3, x + r * LDS + kk * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < JW; ++j) {
                    // each k16 product sum is formed from zero and added to the running sum with a normal fp32 add (the
                    // reason is in moe_grouped.cu; bitwise the same sums as that kernel)
                    float tt[4] = {0.f, 0.f, 0.f, 0.f};
                    ik_mma_bf16_16816(tt, a0, a1, a2, a3, b[j][0], b[j][1]);
#pragma unroll
                    for (int v = 0; v < 4; ++v) acc[i][j][v] += tt[v];
                }
            }
        }
    }

    // ---- epilogue (as moe_grouped.cu): scale, (MODE 0) SiLU-GLU, stage the tile in shared memory, write full rows ----
    constexpr int NCOL = MODE == 0 ? R / 2 : R;  // output columns of the block
    constexpr int OS = NCOL + 8;                // padded smem row (bf16): conflict-free fragment stores
    __syncthreads();                            // every warp is done reading the pipeline buffers
    ik_bf16* const ob = xs0;                    // [BM][OS], aliases the A ring
    if (MODE == 0) {
        const float gg = gscales[e * 3 + 0], gu = gscales[e * 3 + 1];
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < JH; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int rl = i * 16 + g + hh * 8;
                    const int cl = wn * 8 * JH + j * 8 + t4 * 2;
                    float hv[2];
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const float gt = round_bf16(acc[i][j][2 * hh + v] * gg);
                        const float u = round_bf16(acc[i][j + JH][2 * hh + v] * gu);
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
                const int rl = i * 16 + g + hh * 8;
#pragma unroll
                for (int j = 0; j < JW; ++j) {
                    const int cl = wn * 8 * JW + j * 8 + t4 * 2;
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) =
                        __floats2bfloat162_rn(acc[i][j][2 * hh] * gd, acc[i][j][2 * hh + 1] * gd);
                }
            }
    }
    __syncthreads();
    constexpr int CPR = NCOL / 8;  // 16-byte chunks per row
    for (int c = tid; c < BM * CPR; c += NT) {
        const int rl = c / CPR, part = c % CPR;
        if (rl >= rows) continue;
        ik_bf16* dst;
        if (MODE == 0) dst = out + (int64_t)(begin + m0 + rl) * I + n0 + part * 8;
        else dst = out + (int64_t)pairs[begin + m0 + rl] * H + n0 + part * 8;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(ob + rl * OS + part * 8);
    }
}

// ---- launchers -------------------------------------------------------------------------------------------------------

struct ScratchLayout {
    int64_t pairs, offsets, ntiles, tiles, h, y, total;
};

// Same as moe_grouped.cu (tile table capacity is sized for BM = 16, the most tiles any BM produces).
ScratchLayout scratch_layout(int64_t P, int64_t E, int64_t H, int64_t I) {
    ScratchLayout l;
    const int64_t max_tiles = (P + 15) / 16 + (P < E ? P : E);
    l.pairs = 0;
    l.offsets = l.pairs + r256(P * 4);
    l.ntiles = l.offsets + r256((E + 1) * 4);
    l.tiles = l.ntiles + 256;
    l.h = l.tiles + r256(max_tiles * 8);
    l.y = l.h + r256(P * I * 2);
    l.total = l.y + r256(P * H * 2);
    return l;
}

// Rows per tile from the mean pairs per expert: 32 up to 56 (T <= 1792 at 256 experts, top-8), else 64 (measured on spark1:
// 3080 vs 3350 us at T = 1536, 3610 vs 3540 us at 2048). INTARSIA_MOE_BM=32|64 overrides it (A/B experiments, tests).
int pick_bm(int64_t P, int64_t E) {
    if (const char* v = getenv("INTARSIA_MOE_BM")) {
        const int b = atoi(v);
        if (b == 32 || b == 64) return b;
    }
    return (double)P / (double)E <= 56.0 ? 32 : 64;
}

template <int R, int MT, int JW, int MODE, int MINB, int LA, int LR>
void launch_w4(const ik_bf16* a, const uint8_t* w, const uint8_t* sc, const float* gs, const int* pairs,
               const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* out, unsigned tiles_ub, int H, int I,
               int K, cudaStream_t s) {
    constexpr size_t smem = (size_t)((LA + 1) * 16 * MT) * LDS * 2 + (size_t)(LR / 2 + 1) * R * kRawStride;
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(moe_w4a16_kernel<R, MT, JW, MODE, MINB, LA, LR>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(reinterpret_cast<const void*>(moe_w4a16_kernel<R, MT, JW, MODE, MINB, LA, LR>),
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)attr_set;
    const unsigned ny = MODE == 0 ? (unsigned)(I / (R / 2)) : (unsigned)(H / R);
    moe_w4a16_kernel<R, MT, JW, MODE, MINB, LA, LR><<<dim3(ny, tiles_ub), R * 4 / JW, smem, s>>>(a, w, sc, gs, pairs, offsets, tiles, ntiles, out, H, I, K);
}

template <int R, int MT, int JW, int MINB, int LA, int LR>
void run_gemms(const ik_bf16* x, const uint8_t* guw, const uint8_t* gus, const uint8_t* dnw, const uint8_t* dns,
               const float* gs, const int* pairs, const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* hbuf,
               ik_bf16* ybuf, unsigned tiles_ub, int H, int I, int K, cudaStream_t s) {
    launch_w4<R, MT, JW, 0, MINB, LA, LR>(x, guw, gus, gs, pairs, offsets, tiles, ntiles, hbuf, tiles_ub, H, I, K, s);
    launch_w4<R, MT, JW, 1, MINB, LA, LR>(hbuf, dnw, dns, gs, pairs, offsets, tiles, ntiles, ybuf, tiles_ub, H, I, K, s);
}

}  // namespace

extern "C" {

int ik_moe_experts_w4a16_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                              const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                              const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                              void* stream) {
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
    const ScratchLayout L = scratch_layout(P, E, H, I);
    if (!scratch || !scratch->data || scratch->bytes < L.total) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    unsigned char* base = (unsigned char*)scratch->data;
    int* pairs = (int*)(base + L.pairs);
    int* offsets = (int*)(base + L.offsets);
    int* ntiles = (int*)(base + L.ntiles);
    int2* tiles = (int2*)(base + L.tiles);
    ik_bf16* hbuf = (ik_bf16*)(base + L.h);
    ik_bf16* ybuf = (ik_bf16*)(base + L.y);
    cudaStream_t s = IK_STREAM(stream);

    const int BM = pick_bm(P, E);
    const unsigned tiles_ub = (unsigned)((P + BM - 1) / BM + (P < E ? P : E));
    moe_group_tiles_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), s>>>((const int*)ids->data, (int)P, (int)E, BM, pairs,
                                                                        offsets, ntiles, tiles);
    IK_TRY(ik_launch_status());
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* guw = (const uint8_t*)gate_up_w->data;
    const uint8_t* gus = (const uint8_t*)gate_up_s->data;
    const uint8_t* dnw = (const uint8_t*)down_w->data;
    const uint8_t* dns = (const uint8_t*)down_s->data;
    const float* gs = (const float*)gscales->data;
    if (BM == 32) {
        run_gemms<128, 2, 4, 2, 2, 2>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s);
    } else {
        run_gemms<128, 4, 4, 2, 2, 2>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s);
    }
    IK_TRY(ik_launch_status());
    moe_reduce_kernel<ik_bf16><<<dim3((unsigned)T, (unsigned)((H + 255) / 256)), 256, 0, s>>>(
        ybuf, (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)K);
    return ik_launch_status();
}

}  // extern "C"
