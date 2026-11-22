// Inkling-2-Mini routed experts, W4A4 prefill on the block-scaled FP4 tensor core (ink.act_quant = 1, NVFP4 layers; lane
// ink-cuda-mixmoe): the grouped GEMM of moe_nvfp4_mma_sm120a.cu (Qwen3.5-35B, unchanged there) with
//   * the operands of ink_actq.cuh (the contract's per-token "fouroversix" quantiser, outer = amax / 1536) in place of that
//     file's 2688 rule: x -> codes / e4m3 scales / outer by ink_actq_nv, and h likewise between the GEMMs;
//   * the SiTU epilogue of ink_moe_mma.cu: g = rA((sum * outer) * gs), the routing weight folded into h; down y = (sum * outer) * gs.
// The tensor core forms the products of the e2m1 codes and their e4m3 scales exactly and sums them in f32, so the values are the
// contract's up to the f32 accumulation order (as every other W4A4 path). A row's value is independent of the chunk.
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// Built for sm_121a / sm_120a only (build.rs: *_sm120a.cu); otherwise ink_moe_fp4 returns IK_ERR_UNSUPPORTED and
// ink_moe.cu keeps the bf16 tensor-core path. Launches: tile grouping, x quant (by the caller), gate+up, h quant, down, sum (6).
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "moe_group.cuh"
#include "ink_common.cuh"
#include "ink_actq.cuh"
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

// ---- grouped GEMM --------------------------------------------------------------------------------------------------

// MODE 0: h_sorted[row, i] = situ(xq[pair/K] . gate_i, xq[pair/K] . up_i) * w   (A = xq [T,H], out h [P,I] bf16)
// MODE 1: y[pair, n]       = hq[row] . down_n                                  (A = hq [P,I], out y [P,H] bf16)
// Block = 4 x WM warps: 4 along the 128 weight rows (32 rows each; in MODE 0 16 gate + the matching 16 up rows), WM
// along the pairs, each warp owns MT m16 tiles: BM = 16 * MT * WM pairs. A unit is 128 k: 64 bytes of fp4 + 8 bytes of
// scales per row, copied with cp.async into a ring of NS stages; one barrier per unit.
template <int MT, int WM, int MODE, int NS>
__global__ void __launch_bounds__(128 * WM, NS == 2 ? (WM == 1 ? 4 : 2) : 1)
    ink_fp4_kernel(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as, const float* __restrict__ ainv,
                     const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, const float* __restrict__ gscales,
                     const int* __restrict__ pairs, const int* __restrict__ offsets, const int2* __restrict__ tiles,
                     const int* __restrict__ ntiles, ik_bf16* __restrict__ out, int H, int I, int K,
                     const float* __restrict__ wts, float gcap, float ucap) {
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
    float ri[MT][2];  // the row scale (outer) of the rows this thread's accumulators belong to
    float wp[MT][2];  // MODE 0: their routing weights
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int rl = wm * 16 * MT + i * 16 + g + hh * 8;
            float v = 0.f;
            if (rl < rows) v = ainv[MODE == 0 ? pairs[begin + m0 + rl] / K : begin + m0 + rl];
            ri[i][hh] = v;
            wp[i][hh] = MODE == 0 && rl < rows ? wts[pairs[begin + m0 + rl]] : 0.f;
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
                        // Inkling SiTU (ink_moe_mma.cu's epilogue): g = rA((sum * outer) * gs), the routing weight in h
                        const float g0 = round_bf16(__fmul_rn(acc[i][j][2 * hh + v], ri[i][hh]) * gg);
                        const float u0 = round_bf16(__fmul_rn(acc[i][j + 2][2 * hh + v], ri[i][hh]) * gu);
                        hv[v] = gcap * tanhf(g0 / gcap) * (1.f / (1.f + expf(-g0))) * ucap * tanhf(u0 / ucap) * wp[i][hh];
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
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) = __floats2bfloat162_rn(
                        __fmul_rn(acc[i][j][2 * hh], ri[i][hh]) * gd, __fmul_rn(acc[i][j][2 * hh + 1], ri[i][hh]) * gd);
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

int pick_bm_fp4(int64_t P, int64_t E) {
    if (const char* v = getenv("INTARSIA_INK_MOE_BM")) {
        const int b = atoi(v);
        if (b == 16 || b == 32 || b == 64) return b;
    }
    const double mu = (double)P / (double)E;
    return mu <= 10.0 ? 16 : (mu <= 21.0 ? 32 : 64);
}

template <int MT, int WM, int MODE, int NS>
void launch_gemm_ns(const uint8_t* aq, const uint8_t* as, const float* ainv, const uint8_t* w, const uint8_t* sc,
                 const float* gs, const int* pairs, const int* offsets, const int2* tiles, const int* ntiles, ik_bf16* out,
                 unsigned tiles_ub, int H, int I, int K, const float* wts, float gc, float uc, cudaStream_t s) {
    constexpr int BM = 16 * MT * WM;
    constexpr size_t smem = (size_t)NS * (kWRows * kRawStride + kWRows * 8 + BM * kRawStride + BM * 8);
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(ink_fp4_kernel<MT, WM, MODE, NS>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        return true;
    }();
    (void)attr_set;
    const unsigned ny = MODE == 0 ? (unsigned)(I / 64) : (unsigned)(H / 128);
    ink_fp4_kernel<MT, WM, MODE, NS><<<dim3(ny, tiles_ub), 128 * WM, smem, s>>>(aq, as, ainv, w, sc, gs, pairs, offsets,
                                                                              tiles, ntiles, out, H, I, K, wts, gc, uc);
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
                 unsigned tiles_ub, int H, int I, int K, const float* wts, float gc, float uc, cudaStream_t s) {
    switch (pick_stages()) {
        case 2: launch_gemm_ns<MT, WM, MODE, 2>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, wts, gc, uc, s); break;
        case 4: launch_gemm_ns<MT, WM, MODE, 4>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, wts, gc, uc, s); break;
        case 3: launch_gemm_ns<MT, WM, MODE, 3>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, wts, gc, uc, s); break;
        default: launch_gemm_ns<MT, WM, MODE, 2>(aq, as, ainv, w, sc, gs, pairs, offsets, tiles, ntiles, out, tiles_ub, H, I, K, wts, gc, uc, s); break;
    }
}


// out[t, r] = rA(sum_k y[t*K + k, r]) over the valid slots in slot order (situ: the routing weight is in h)
template <typename OutT>
__global__ void __launch_bounds__(256) ink_fp4_sum_kernel(const ik_bf16* __restrict__ y, const int* __restrict__ ids, OutT* __restrict__ out,
                                                          int E, int H, int K) {
    const int t = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    if (c >= H) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int id = ids[(int64_t)t * K + k];
        if (id < 0 || id >= E) continue;
        acc += ik_b2f(y[((int64_t)t * K + k) * H + c]);
    }
    ik_store(out + (int64_t)t * H + c, acc);
}

}  // namespace
#endif  // IK_SM120A

int ink_moe_fp4_available() { return IK_SM120A ? 1 : 0; }

int64_t ink_moe_fp4_scratch_bytes(int64_t P, int64_t E, int64_t H, int64_t I) {
    const int64_t max_tiles = (P + 15) / 16 + (P < E ? P : E);
    return r256(P * 4) + r256((E + 1) * 4) + 256 + r256(max_tiles * 8) + r256(P * I * 2) + r256(P * H * 2) + r256(P * (I / 2)) +
           r256(P * (I / 16)) + r256(P * 4);
}

int ink_moe_fp4(const InkMmaArgs& a, const uint8_t* xq, const uint8_t* xs, void* scratch, int out_f32, cudaStream_t s) {
#if !IK_SM120A
    (void)a; (void)xq; (void)xs; (void)scratch; (void)out_f32; (void)s;
    return IK_ERR_UNSUPPORTED;
#else
    if (a.fmt != 0 || a.act != 2 || !a.xo || a.H % 128 || a.I % 128 || a.E > kMaxExperts) return IK_ERR_INVALID_ARG;
    const int64_t P = (int64_t)a.T * a.K, E = a.E, H = a.H, I = a.I;
    unsigned char* base = (unsigned char*)scratch;
    const int64_t max_tiles = (P + 15) / 16 + (P < E ? P : E);
    int* pairs = (int*)base;
    int* offsets = (int*)(base + r256(P * 4));
    int* ntiles = (int*)(base + r256(P * 4) + r256((E + 1) * 4));
    int2* tiles = (int2*)((unsigned char*)ntiles + 256);
    ik_bf16* h = (ik_bf16*)((unsigned char*)tiles + r256(max_tiles * 8));
    ik_bf16* y = (ik_bf16*)((unsigned char*)h + r256(P * I * 2));
    uint8_t* hq = (uint8_t*)y + r256(P * H * 2);
    uint8_t* hs = hq + r256(P * (I / 2));
    float* hg = (float*)(hs + r256(P * (I / 16)));
    const int BM = pick_bm_fp4(P, E);
    const unsigned tiles_ub = (unsigned)((P + BM - 1) / BM + (P < E ? P : E));
    moe_group_tiles_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), s>>>(a.ids, (int)P, (int)E, BM, pairs, offsets, ntiles, tiles);
    IK_TRY(ik_launch_status());
#define IK_RUN(MT, WM)                                                                                                          \
    launch_gemm<MT, WM, 0>(xq, xs, a.xo, a.guw, a.gus, a.gs, pairs, offsets, tiles, ntiles, h, tiles_ub, (int)H, (int)I, a.K, a.wts, \
                           a.gc, a.uc, s);                                                                                       \
    IK_TRY(ik_launch_status());                                                                                                  \
    IK_TRY(ink_actq_nv(h, P, (int)I, nullptr, hg, hq, hs, s));                                                                    \
    launch_gemm<MT, WM, 1>(hq, hs, hg, a.dnw, a.dns, a.gs, pairs, offsets, tiles, ntiles, y, tiles_ub, (int)H, (int)I, a.K, a.wts,  \
                           a.gc, a.uc, s);
    switch (BM) {
        case 16: IK_RUN(1, 1) break;
        case 32: IK_RUN(1, 2) break;
        default: IK_RUN(2, 2) break;
    }
#undef IK_RUN
    IK_TRY(ik_launch_status());
    const dim3 g((unsigned)a.T, (unsigned)((H + 255) / 256));
    if (out_f32) ink_fp4_sum_kernel<float><<<g, 256, 0, s>>>(y, a.ids, (float*)a.out, (int)E, (int)H, a.K);
    else ink_fp4_sum_kernel<ik_bf16><<<g, 256, 0, s>>>(y, a.ids, (ik_bf16*)a.out, (int)E, (int)H, a.K);
    return ik_launch_status();
#endif
}
