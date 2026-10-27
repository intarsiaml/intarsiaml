// Grouped tensor-core MoE experts for prefill (docs/quant-moe.md). Same op and signature as moe_experts_bf16
// (moe.cu); this one is for batches where every active expert sees several tokens.
//
//   group    counting-sort the (token, slot) pairs by expert and emit a tile table of (expert, first row) work
//            items, `BM` rows each, so a block never mixes experts. One block, E <= 1024.
//   gate_up  block = one tile x 64 intermediate columns: 64 gate rows + the matching 64 up rows of the expert,
//            streamed over H in steps of 64. Each NVFP4 weight tile is loaded once and dequantised to bf16 in
//            shared memory (exact: e2m1 * e4m3 has <= 4 significant bits), then all `BM` pairs of the tile run
//            on tensor cores (mma.sync m16n8k16, bf16 in; every k16 partial sum is formed from zero and added to the
//            f32 accumulator with a normal fp32 add). The epilogue applies the global scales and SiLU-GLU with the
//            reference rounding points and writes h in SORTED pair order.
//   down     block = one tile x 128 output rows, K = I. A is the contiguous sorted h rows; the epilogue scales
//            and scatters y[pair] back to the original (token, slot) index.
//   reduce   out[t] = sum_k round(y[t,k] * w[t,k]) in fixed k order (same as moe.cu: no atomics on `out`).
//
// Pipelining (see the comment above moe_mma_kernel): packed weights move in 64-byte-per-row cp.async units into a
// 2-unit staging ring, the A tile through a 3-slot cp.async ring, each thread dequantises its slice of a row into one of
// two bf16 weight buffers while the previous step's mma runs. Every pair's arithmetic depends only on its own row, so
// results are independent of tiling and of the order the atomics produce in the grouping: bit-identical run to run.
//
// Needs H % 128 == 0 and I % 64 == 0 (Qwen3.6-35B-A3B: H=2048, I=512; Gemma-4-26B-A4B: H=2816, I=704, whose down projection
// has K = I = 5.5 raw units). Scratch layout (must match
// intarsia_core::quant::ops::moe_experts_mma_scratch, regions rounded to 256 bytes):
//   pairs int[T*K] | offsets int[E+1] | ntiles int[1] | tiles int2[ceil(P/16) + min(E,P)] | h bf16[P,I] | y bf16[P,H]
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "moe_group.cuh"
#include <stdlib.h>

namespace {

constexpr int BK = 64;         // k per pipeline step
constexpr int LDS = BK + 8;    // padded smem row stride (bf16 elements): conflict-free fragment loads
constexpr int kWRows = 128;    // weight rows per block (64 gate + 64 up, or 128 down rows): one per thread

// ---- grouped GEMM --------------------------------------------------------------------------------------------------

// MODE 0: h_sorted[row, i] = silu-glu(x[pair/K] . gate_i, x[pair/K] . up_i)   (a_src = x [T,H], out = h [P,I])
// MODE 1: y[pair, n]       = h_sorted[row] . down_n                          (a_src = h [P,I], out = y [P,H])
// `row` = position of the pair in the expert-sorted list. Weights are [E, rows, Kdim/2] fp4 + [E, rows, Kdim/16] e4m3.
//
// Block = 4 x WM warps: 4 along the 128 weight rows (32 rows each; in MODE 0 16 gate + the matching 16 up rows),
// WM along the pairs, each warp owns MT m16 tiles: BM = 16 * MT * WM pairs. 128 * WM threads.
//
// Memory pipeline (this is what limits the kernel, see docs/quant-moe.md): LPDDR reads of 32 B per weight row at a
// 1 KB row stride reach only ~50% of the bandwidth, runs of >= 64 B per row reach ~85%. So the packed weights are
// copied with cp.async in units of 64 contiguous bytes per row (128 k elements, 4 lanes per row), staged in shared
// memory, and each thread dequantises its slice of a row from there, 64 k elements per step. Per iteration s:
//   wait for step s's A tile and the raw unit of step s+1 -> barrier -> issue the copies of step s+2 (one commit
//   group per iteration: A(s+2), plus raw unit u when s == 2u-3) -> dequantise step s+1 into the bf16 tile -> mma step s.
// Ring sizes: A 3 slots, raw 2 units, bf16 weight tile 2 buffers.
constexpr int kRawStride = 80;                    // bytes per staged row: 64 payload + 16 pad (conflict-free 16 B reads)
constexpr int kRawUnit = kWRows * kRawStride;     // one unit: 128 rows x 64 B

template <int MT, int WM, int MODE, bool GELU>
__global__ void __launch_bounds__(128 * WM)
moe_mma_kernel(const ik_bf16* __restrict__ a_src, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
               const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
               const int2* __restrict__ tiles, const int* __restrict__ ntiles, ik_bf16* __restrict__ out, int H, int I,
               int K) {
    constexpr int BM = 16 * MT * WM;
    constexpr int NTHREADS = 128 * WM;
    constexpr int RQ = 512 / NTHREADS;  // 16-byte raw chunks per thread and unit (128 rows x 4)
    const int tile = blockIdx.y;  // blocks of one tile (all its weight column tiles) are adjacent: the A rows stay in L2
    if (tile >= *ntiles) return;  // block-uniform: the grid is an upper bound
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int XS = BM * LDS, WS = kWRows * LDS;
    ik_bf16* const xs0 = reinterpret_cast<ik_bf16*>(smem_raw);
    auto xs = [&](int slot) { return xs0 + slot * XS; };           // A tile [BM][LDS], ring of 3
    auto ws = [&](int buf) { return xs0 + 3 * XS + buf * WS; };    // dequantised weight tile [128][LDS], 2 buffers
    unsigned char* const raw_base = smem_raw + (size_t)(3 * XS + 2 * WS) * 2;
    auto rawu = [&](int slot) { return raw_base + slot * kRawUnit; };  // staged packed weights, ring of 2 units

    const int2 td = tiles[tile];
    const int e = td.x, m0 = td.y;
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int rows = min(BM, cnt - m0);  // 1..BM
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp & 3, wm = warp >> 2;
    const int Kdim = MODE == 0 ? H : I;
    const int S = Kdim / BK;     // 64-wide steps
    const int U = (Kdim + 127) / 128;  // 128-wide raw units; a Kdim of 64 mod 128 (Gemma 4: I = 704) ends in a half unit

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

    // packed-weight copy: chunk (tid + NTHREADS q) = row r, 16-byte part p of the unit's 64 bytes
    const int rpart = tid & 3;
    const uint8_t* rsrc[RQ];
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
        const int r = (tid >> 2) + (NTHREADS / 4) * q;
        rsrc[q] = w + ((int64_t)e * nrows + weight_row(r)) * (Kdim / 2) + rpart * 16;
    }
    auto issue_raw = [&](int u, int slot) {
#pragma unroll
        for (int q = 0; q < RQ; ++q) {
            const int r = (tid >> 2) + (NTHREADS / 4) * q;
            // the half unit at the end of a Kdim of 64 mod 128 holds 32 payload bytes per row: parts 2 and 3 are zero-filled
            const bool in_row = u * 64 + rpart * 16 < Kdim / 2;
            ik_cp_async16(rawu(slot) + r * kRawStride + rpart * 16, in_row ? rsrc[q] + u * 64 : rsrc[q], in_row ? 16 : 0);
        }
    };

    // this thread dequantises row `wr` of the tile (k offset `khalf * 32` of each step when WM == 2)
    const int wr = WM == 1 ? tid : tid >> 1;
    const int khalf = WM == 1 ? 0 : (tid & 1);
    const uint8_t* sp = sc + ((int64_t)e * nrows + weight_row(wr)) * (Kdim / 16) + khalf * 2;
    uint32_t rawsc;
    auto load_sc = [&](int st) {
        if (WM == 1) rawsc = *reinterpret_cast<const uint32_t*>(sp + st * (BK / 16));
        else rawsc = *reinterpret_cast<const uint16_t*>(sp + st * (BK / 16));
    };
    auto dequant = [&](int st, int buf) {
        const unsigned char* rp = rawu((st >> 1) & 1) + wr * kRawStride + (st & 1) * 32 + khalf * 16;
        const uint4 q0 = *reinterpret_cast<const uint4*>(rp);
        const uint4 q1 = WM == 1 ? *reinterpret_cast<const uint4*>(rp + 16) : q0;
        const uint32_t wd[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int b = 0; b < 4 / WM; ++b) {
            uint4 lo, hi;
            ik_fp4x16_scaled_to_bf16(make_uint2(wd[2 * b], wd[2 * b + 1]), ik_e4m3_to_f((uint8_t)((rawsc >> (8 * b)) & 0xFFu)),
                                     lo, hi);
            uint4* dst = reinterpret_cast<uint4*>(ws(buf) + wr * LDS + khalf * 32 + b * 16);
            dst[0] = lo;
            dst[1] = hi;
        }
    };

    // this thread's A rows: chunk (16 B = 8 bf16) c8 of rows r = (tid >> 3) + (NTHREADS / 8) q
    const int c8 = tid & 7;
    const ik_bf16* arow[MT];
    bool avalid[MT];
#pragma unroll
    for (int q = 0; q < MT; ++q) {
        const int r = (tid >> 3) + (NTHREADS / 8) * q;
        avalid[q] = r < rows;
        arow[q] = a_src;  // a valid address for the zero-filled rows
        if (avalid[q]) {
            if (MODE == 0) arow[q] = a_src + (int64_t)(pairs[begin + m0 + r] / K) * H;
            else arow[q] = a_src + (int64_t)(begin + m0 + r) * I;
        }
    }
    auto issue_a = [&](int st, int slot) {
#pragma unroll
        for (int q = 0; q < MT; ++q) {
            const int r = (tid >> 3) + (NTHREADS / 8) * q;
            ik_cp_async16(xs(slot) + r * LDS + c8 * 8, arow[q] + st * BK + c8 * 8, avalid[q] ? 16 : 0);
        }
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

    // prologue: groups 0 and 1 hold A(0) + unit 0 and A(1) + unit 1
    issue_a(0, 0);
    issue_raw(0, 0);
    ik_cp_async_commit();
    if (S > 1) issue_a(1, 1);
    if (U > 1) issue_raw(1, 1);
    ik_cp_async_commit();
    load_sc(0);
    ik_cp_async_wait_group<1>();
    __syncthreads();
    dequant(0, 0);
    if (S > 1) load_sc(1);
    for (int s = 0; s < S; ++s) {
        ik_cp_async_wait_group<1>();  // groups 0..s complete: A(s), and the raw unit of step s+1
        __syncthreads();              // visible to all; everyone is done with step s-1's buffers
        if (s + 2 < S) issue_a(s + 2, (s + 2) % 3);
        if ((s & 1) && (s + 3) / 2 < U) issue_raw((s + 3) / 2, ((s + 3) / 2) & 1);
        ik_cp_async_commit();
        if (s + 1 < S) dequant(s + 1, (s + 1) & 1);
        if (s + 2 < S) load_sc(s + 2);
        const ik_bf16* x = xs(s % 3);
        const ik_bf16* wt = ws(s & 1);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t b[4][2];
#pragma unroll
            for (int pj = 0; pj < 2; ++pj) {  // two n8 tiles per ldmatrix.x4
                const int mi = lane >> 3;
                const int nr = nrow(2 * pj + (mi >> 1)) + (lane & 7);
                ik_ldmatrix_x4(b[2 * pj][0], b[2 * pj][1], b[2 * pj + 1][0], b[2 * pj + 1][1],
                               wt + nr * LDS + kk + (mi & 1) * 8);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                uint32_t a0, a1, a2, a3;
                const int r = wm * 16 * MT + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ik_ldmatrix_x4(a0, a1, a2, a3, x + r * LDS + kk + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // Each k16 product sum is formed from zero and then added to the running sum with a normal fp32
                    // add. Accumulating the whole K inside the mma (acc as the C operand) left 0.6 to 1 % of the bf16
                    // outputs one step off the exact result (the warp-per-row kernel: about 0.1 %); the tensor core's
                    // accumulate step dropping low bits is the likely cause, inferred from the effect of this change.
                    // See tests/quantmoe.rs `moe_experts_mma_is_as_accurate_as_old_kernel`.
                    float t[4] = {0.f, 0.f, 0.f, 0.f};
                    ik_mma_bf16_16816(t, a0, a1, a2, a3, b[j][0], b[j][1]);
#pragma unroll
                    for (int v = 0; v < 4; ++v) acc[i][j][v] += t[v];
                }
            }
        }
    }

    // ---- epilogue: scale, (MODE 0) SiLU-GLU, stage the tile in shared memory, write full contiguous rows ----
    constexpr int NCOL = MODE == 0 ? 64 : 128;  // output columns of the block
    constexpr int OS = NCOL + 8;                // padded smem row (bf16): conflict-free fragment stores
    __syncthreads();                            // every warp is done reading the pipeline buffers
    ik_bf16* const ob = xs0;                    // [BM][OS], aliases the A ring
    if (MODE == 0) {
        const float gg = gscales[e * 3 + 0], gu = gscales[e * 3 + 1];
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int rl = wm * 16 * MT + i * 16 + (lane >> 2) + hh * 8;
                    const int cl = wn * 16 + j * 8 + (lane & 3) * 2;
                    float hv[2];
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const float g = round_bf16(acc[i][j][2 * hh + v] * gg);
                        const float u = round_bf16(acc[i][j + 2][2 * hh + v] * gu);
                        const float a = round_bf16(GELU ? ik_gelu_tanh(g) : g / (1.f + expf(-g)));
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
                const int rl = wm * 16 * MT + i * 16 + (lane >> 2) + hh * 8;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int cl = wn * 32 + j * 8 + (lane & 3) * 2;
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) =
                        __floats2bfloat162_rn(acc[i][j][2 * hh] * gd, acc[i][j][2 * hh + 1] * gd);
                }
            }
    }
    __syncthreads();
    constexpr int CPR = NCOL / 8;  // 16-byte chunks per row
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
    int64_t pairs, offsets, ntiles, tiles, h, y, total;
};

// Tile table capacity is sized for BM = 16 (the most tiles any BM produces).
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

// Rows per tile from the mean pairs per expert (a tile wastes mma rows when an expert has fewer than BM pairs, and
// dequantises its weights once more per extra tile). INTARSIA_MOE_BM=16|32|64 overrides it (A/B experiments).
int pick_bm(int64_t P, int64_t E) {
    if (const char* v = getenv("INTARSIA_MOE_BM")) {
        const int b = atoi(v);
        if (b == 16 || b == 32 || b == 64) return b;
    }
    const double mu = (double)P / (double)E;
    return mu <= 10.0 ? 16 : (mu <= 21.0 ? 32 : 64);
}

template <int MT, int WM, int MODE, bool GELU>
void launch_mma(const ik_bf16* a, const uint8_t* w, const uint8_t* sc, const float* gs, const int* pairs,
                const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* out, unsigned tiles_ub, int H, int I,
                int K, cudaStream_t s) {
    constexpr size_t smem = (size_t)(3 * 16 * MT * WM + 2 * kWRows) * LDS * 2 + 2 * kRawUnit;
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(moe_mma_kernel<MT, WM, MODE, GELU>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        return true;
    }();
    (void)attr_set;
    const unsigned ny = MODE == 0 ? (unsigned)(I / 64) : (unsigned)(H / 128);
    moe_mma_kernel<MT, WM, MODE, GELU><<<dim3(ny, tiles_ub), 128 * WM, smem, s>>>(a, w, sc, gs, pairs, offsets, tiles, ntiles, out, H, I, K);
}

template <int MT, int WM, bool GELU>
void run_gemms(const ik_bf16* x, const uint8_t* guw, const uint8_t* gus, const uint8_t* dnw, const uint8_t* dns,
               const float* gs, const int* pairs, const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* hbuf,
               ik_bf16* ybuf, unsigned tiles_ub, int H, int I, int K, cudaStream_t s) {
    launch_mma<MT, WM, 0, GELU>(x, guw, gus, gs, pairs, offsets, tiles, ntiles, hbuf, tiles_ub, H, I, K, s);
    launch_mma<MT, WM, 1, GELU>(hbuf, dnw, dns, gs, pairs, offsets, tiles, ntiles, ybuf, tiles_ub, H, I, K, s);
}

int moe_mma_launch(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                            const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                            const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                            void* stream, bool gelu) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(ids, IK_DTYPE_I32, 2, 4));
    IK_TRY(ik_check_a(weights, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(gate_up_w, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(gate_up_s, IK_DTYPE_FP8_E4M3, 3, 4));
    IK_TRY(ik_check_a(down_w, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(down_s, IK_DTYPE_FP8_E4M3, 3, 4));
    IK_TRY(ik_check_a(gscales, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(out, IK_DTYPE_BF16, 2, 2));
    const int64_t T = x->shape[0], H = x->shape[1], K = ids->shape[1];
    const int64_t E = gate_up_w->shape[0], I = gate_up_w->shape[1] / 2;
    if (H % 128 != 0 || I % 64 != 0 || gate_up_w->shape[1] != 2 * I || gate_up_w->shape[2] != H / 2 ||
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
    if (gelu) {
        switch (BM) {
            case 16: run_gemms<1, 1, true>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s); break;
            case 32: run_gemms<1, 2, true>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s); break;
            default: run_gemms<2, 2, true>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s); break;
        }
    } else {
        switch (BM) {
            case 16: run_gemms<1, 1, false>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s); break;
            case 32: run_gemms<1, 2, false>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s); break;
            default: run_gemms<2, 2, false>(xp, guw, gus, dnw, dns, gs, pairs, offsets, tiles, ntiles, hbuf, ybuf, tiles_ub, (int)H, (int)I, (int)K, s); break;
        }
    }
    IK_TRY(ik_launch_status());
    moe_reduce_kernel<ik_bf16><<<dim3((unsigned)T, (unsigned)((H + 255) / 256)), 256, 0, s>>>(
        ybuf, (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_moe_experts_mma_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                            const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                            const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                            void* stream) {
    return moe_mma_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, false);
}

// gelu_tanh in the expert activation (OpAttrs::gelu); otherwise the same bits as ik_moe_experts_mma_bf16.
int ik_moe_experts_mma_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                            const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                            const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                            void* stream) {
    return moe_mma_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true);
}

}  // extern "C"
