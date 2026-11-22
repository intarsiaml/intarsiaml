// Inkling-2-Mini routed experts, tensor-core prefill (lane ink-cuda-mixmoe): the grouped GEMM of moe_grouped.cu (Qwen3.5-35B
// NVFP4 prefill) templated on the weight format (FMT 0 NVFP4, 1 MXFP8: e4m3 and an E8M0 scale per 32 along k, staged in
// 128-byte units) and the activation (ACT 0 silu, 1 gelu_tanh, 2 situ with the routing weight folded into h). The weights are
// dequantised once per tile into bf16 (exact: an e2m1 x e4m3 or e4m3 x 2^k product fits bf16's mantissa), bf16 mma with
// per-k16 sums added in fp32. Launched by ink_moe.cu for chunks of more than kMmaMinPairs pairs (prefill); a row's value is
// independent of T within this path (chunk-invariant) but not bitwise the decode kernels (another k order), as the 35B prefill.
// Launches: tile grouping, gate+up, down, token sum (4).
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "moe_group.cuh"
#include "ink_common.cuh"
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
// (lane ink-cuda-mixmoe) FMT 0 NVFP4: a unit is 64 payload bytes per row (128 k); FMT 1 MXFP8: 128 bytes (128 k)
template <int FMT>
struct RawGeom {
    static constexpr int RB = FMT == 0 ? 64 : 128;  // payload bytes per row and unit
    static constexpr int RS = RB + 16;              // staged row stride (conflict-free 16 B reads)
    static constexpr int RU = kWRows * RS;          // one unit
};

template <int MT, int WM, int MODE, int FMT, int ACT>
__global__ void __launch_bounds__(128 * WM)
ink_moe_mma_kernel(const ik_bf16* __restrict__ a_src, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                   const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                   const int2* __restrict__ tiles, const int* __restrict__ ntiles, ik_bf16* __restrict__ out, int H, int I,
                   int K, const float* __restrict__ wts, float gcap, float ucap) {
    constexpr int kRawStride = RawGeom<FMT>::RS;
    constexpr int kRawUnit = RawGeom<FMT>::RU;
    constexpr int RCH = RawGeom<FMT>::RB / 16;  // 16-byte chunks per row and unit
    constexpr int BM = 16 * MT * WM;
    constexpr int NTHREADS = 128 * WM;
    constexpr int RQ = kWRows * RCH / NTHREADS;  // 16-byte raw chunks per thread and unit
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
    const int U = Kdim / 128;    // 128-wide raw units (Kdim % 128 == 0)

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

    // packed-weight copy: chunk (tid + NTHREADS q) = row r, 16-byte part p of the unit's RB bytes
    const int rpart = tid % RCH;
    const int64_t wrow_bytes = FMT == 0 ? Kdim / 2 : Kdim;
    const uint8_t* rsrc[RQ];
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
        const int r = tid / RCH + (NTHREADS / RCH) * q;
        rsrc[q] = w + ((int64_t)e * nrows + weight_row(r)) * wrow_bytes + rpart * 16;
    }
    auto issue_raw = [&](int u, int slot) {
#pragma unroll
        for (int q = 0; q < RQ; ++q) {
            const int r = tid / RCH + (NTHREADS / RCH) * q;
            ik_cp_async16(rawu(slot) + r * kRawStride + rpart * 16, rsrc[q] + u * RawGeom<FMT>::RB, 16);
        }
    };

    // this thread dequantises row `wr` of the tile (k offset `khalf * 32` of each step when WM == 2)
    const int wr = WM == 1 ? tid : tid >> 1;
    const int khalf = WM == 1 ? 0 : (tid & 1);
    const uint8_t* sp = FMT == 0 ? sc + ((int64_t)e * nrows + weight_row(wr)) * (Kdim / 16) + khalf * 2
                                 : sc + ((int64_t)e * nrows + weight_row(wr)) * (Kdim / 32) + khalf;
    uint32_t rawsc;
    auto load_sc = [&](int st) {
        if (FMT == 0) {
            if (WM == 1) rawsc = *reinterpret_cast<const uint32_t*>(sp + st * (BK / 16));
            else rawsc = *reinterpret_cast<const uint16_t*>(sp + st * (BK / 16));
        } else {
            if (WM == 1) rawsc = (uint32_t)sp[st * 2] | ((uint32_t)sp[st * 2 + 1] << 8);
            else rawsc = sp[st * 2];
        }
    };
    auto dequant = [&](int st, int buf) {
        if (FMT == 1) {
            // MXFP8: 64 (WM 1) or 32 (WM 2) e4m3 of this row per step, times 2^(E8M0 - 127): exact in bf16 within its range
            const unsigned char* rp = rawu((st >> 1) & 1) + wr * kRawStride + (st & 1) * 64 + khalf * 32;
#pragma unroll
            for (int qi = 0; qi < 4 / WM; ++qi) {
                const uint4 q = *reinterpret_cast<const uint4*>(rp + qi * 16);
                const uint32_t eb = (rawsc >> (8 * (qi >> 1))) & 0xFFu;
                const float s = eb == 0 ? __uint_as_float(0x00400000u) : (eb == 255 ? __uint_as_float(0x7fc00000u) : __uint_as_float(eb << 23));
                float f[16];
                ik_fp8x16_to_f32(q, f);
#pragma unroll
                for (int i = 0; i < 16; ++i) f[i] *= s;
                uint4* dst = reinterpret_cast<uint4*>(ws(buf) + wr * LDS + khalf * 32 + qi * 16);
                dst[0] = ik_pack8(f);
                dst[1] = ik_pack8(f + 8);
            }
            return;
        }
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
        const float gg = FMT == 0 ? gscales[e * 3 + 0] : 1.f, gu = FMT == 0 ? gscales[e * 3 + 1] : 1.f;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int rl = wm * 16 * MT + i * 16 + (lane >> 2) + hh * 8;
                    const int cl = wn * 16 + j * 8 + (lane & 3) * 2;
                    const float wp = ACT == 2 && rl < rows ? wts[pairs[begin + m0 + rl]] : 0.f;
                    float hv[2];
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const float g = round_bf16(acc[i][j][2 * hh + v] * gg);
                        const float u = round_bf16(acc[i][j + 2][2 * hh + v] * gu);
                        if (ACT == 2) {
                            hv[v] = gcap * tanhf(g / gcap) * (1.f / (1.f + expf(-g))) * ucap * tanhf(u / ucap) * wp;
                        } else {
                            const float a = round_bf16(ACT == 1 ? ik_gelu_tanh(g) : g / (1.f + expf(-g)));
                            hv[v] = a * u;
                        }
                    }
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) = __floats2bfloat162_rn(hv[0], hv[1]);
                }
    } else {
        const float gd = FMT == 0 ? gscales[e * 3 + 2] : 1.f;
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



int pick_bm_ink(int64_t P, int64_t E) {
    if (const char* v = getenv("INTARSIA_INK_MOE_BM")) {
        const int b = atoi(v);
        if (b == 16 || b == 32 || b == 64) return b;
    }
    const double mu = (double)P / (double)E;
    return mu <= 10.0 ? 16 : (mu <= 21.0 ? 32 : 64);
}

template <int MT, int WM, int MODE, int FMT, int ACT>
int launch_one(const ik_bf16* a, const uint8_t* w, const uint8_t* sc, const float* gs, const int* pairs, const int* offsets,
               const int2* tiles, const int* ntiles, ik_bf16* out, unsigned tiles_ub, int H, int I, int K, const float* wts, float gc,
               float uc, cudaStream_t s) {
    constexpr size_t smem = (size_t)(3 * 16 * MT * WM + 2 * kWRows) * LDS * 2 + 2 * RawGeom<FMT>::RU;
    static const cudaError_t attr = cudaFuncSetAttribute(reinterpret_cast<const void*>(ink_moe_mma_kernel<MT, WM, MODE, FMT, ACT>),
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (attr != cudaSuccess) return IK_ERR_LAUNCH;
    const unsigned nx = MODE == 0 ? (unsigned)(I / 64) : (unsigned)(H / 128);
    ink_moe_mma_kernel<MT, WM, MODE, FMT, ACT><<<dim3(nx, tiles_ub), 128 * WM, smem, s>>>(a, w, sc, gs, pairs, offsets, tiles, ntiles,
                                                                                         out, H, I, K, wts, gc, uc);
    return ik_launch_status();
}

template <int MT, int WM, int FMT, int ACT>
int gemms(const InkMmaArgs& a, const int* pairs, const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* h, ik_bf16* y,
          unsigned tiles_ub, cudaStream_t s) {
    IK_TRY((launch_one<MT, WM, 0, FMT, ACT>(a.x, a.guw, a.gus, a.gs, pairs, offsets, tiles, ntiles, h, tiles_ub, a.H, a.I, a.K, a.wts,
                                            a.gc, a.uc, s)));
    return launch_one<MT, WM, 1, FMT, ACT>(h, a.dnw, a.dns, a.gs, pairs, offsets, tiles, ntiles, y, tiles_ub, a.H, a.I, a.K, a.wts,
                                           a.gc, a.uc, s);
}

// out[t, r] = rA(sum_k term) over the valid slots in slot order; term = y (situ: the weight is in h) or rA(y * w)
template <typename OutT, bool MULW>
__global__ void __launch_bounds__(256) ink_mma_sum_kernel(const ik_bf16* __restrict__ y, const int* __restrict__ ids,
                                                          const float* __restrict__ w, OutT* __restrict__ out, int E, int H, int K) {
    const int t = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    if (c >= H) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int id = ids[(int64_t)t * K + k];
        if (id < 0 || id >= E) continue;
        const float v = ik_b2f(y[((int64_t)t * K + k) * H + c]);
        acc += MULW ? round_bf16(v * w[(int64_t)t * K + k]) : v;
    }
    ik_store(out + (int64_t)t * H + c, acc);
}

template <int FMT, int ACT>
int run_fa(const InkMmaArgs& a, int BM, const int* pairs, const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* h,
           ik_bf16* y, unsigned tiles_ub, cudaStream_t s) {
    switch (BM) {
        case 16: return gemms<1, 1, FMT, ACT>(a, pairs, offsets, tiles, ntiles, h, y, tiles_ub, s);
        case 32: return gemms<1, 2, FMT, ACT>(a, pairs, offsets, tiles, ntiles, h, y, tiles_ub, s);
        default: return gemms<2, 2, FMT, ACT>(a, pairs, offsets, tiles, ntiles, h, y, tiles_ub, s);
    }
}

}  // namespace

int64_t ink_moe_mma_scratch_bytes(int64_t P, int64_t E, int64_t H, int64_t I) {
    const int64_t max_tiles = (P + 15) / 16 + (P < E ? P : E);
    return r256(P * 4) + r256((E + 1) * 4) + 256 + r256(max_tiles * 8) + r256(P * I * 2) + r256(P * H * 2);
}

int ink_moe_mma(const InkMmaArgs& a, void* scratch, int out_f32, cudaStream_t s) {
    const int64_t P = (int64_t)a.T * a.K, E = a.E, H = a.H, I = a.I;
    unsigned char* base = (unsigned char*)scratch;
    int* pairs = (int*)base;
    int* offsets = (int*)(base + r256(P * 4));
    int* ntiles = (int*)(base + r256(P * 4) + r256((E + 1) * 4));
    int2* tiles = (int2*)((unsigned char*)ntiles + 256);
    const int64_t max_tiles = (P + 15) / 16 + (P < E ? P : E);
    ik_bf16* h = (ik_bf16*)((unsigned char*)tiles + r256(max_tiles * 8));
    ik_bf16* y = (ik_bf16*)((unsigned char*)h + r256(P * I * 2));
    const int BM = pick_bm_ink(P, E);
    const unsigned tiles_ub = (unsigned)((P + BM - 1) / BM + (P < E ? P : E));
    moe_group_tiles_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), s>>>(a.ids, (int)P, (int)E, BM, pairs, offsets, ntiles, tiles);
    IK_TRY(ik_launch_status());
    int rc;
    if (a.fmt == 0) rc = run_fa<0, 2>(a, BM, pairs, offsets, tiles, ntiles, h, y, tiles_ub, s);
    else if (a.act == 2) rc = run_fa<1, 2>(a, BM, pairs, offsets, tiles, ntiles, h, y, tiles_ub, s);
    else if (a.act == 1) rc = run_fa<1, 1>(a, BM, pairs, offsets, tiles, ntiles, h, y, tiles_ub, s);
    else rc = run_fa<1, 0>(a, BM, pairs, offsets, tiles, ntiles, h, y, tiles_ub, s);
    IK_TRY(rc);
    const dim3 g((unsigned)a.T, (unsigned)((H + 255) / 256));
    const bool mulw = a.act != 2;
    if (out_f32) {
        if (mulw) ink_mma_sum_kernel<float, true><<<g, 256, 0, s>>>(y, a.ids, a.wts, (float*)a.out, (int)E, (int)H, a.K);
        else ink_mma_sum_kernel<float, false><<<g, 256, 0, s>>>(y, a.ids, a.wts, (float*)a.out, (int)E, (int)H, a.K);
    } else {
        if (mulw) ink_mma_sum_kernel<ik_bf16, true><<<g, 256, 0, s>>>(y, a.ids, a.wts, (ik_bf16*)a.out, (int)E, (int)H, a.K);
        else ink_mma_sum_kernel<ik_bf16, false><<<g, 256, 0, s>>>(y, a.ids, a.wts, (ik_bf16*)a.out, (int)E, (int)H, a.K);
    }
    return ik_launch_status();
}
