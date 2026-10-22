// matmul_fp8_gemm: same op and numerics as matmul_fp8_mma (fp8.cu), with the memory schedule of wq_gemm.cuh.
// W8A8 (input_scale > 0): the activations are first quantised to e4m3-valued bf16 into scratch (one pass over x), the
// GEMM then reads those. Needs K % 64 == 0.
#include "wq_gemm.cuh"
#include "ik_internal.h"
#include <stdlib.h>
#include <map>
#include <mutex>
#include <utility>

namespace {

__global__ void __launch_bounds__(256)
fp8_actq_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ xq, const float* __restrict__ sc, int64_t n8) {
    const float s_in = sc[1];
    if (!(s_in > 0.f)) return;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    float f[8];
    ik_unpack8(reinterpret_cast<const uint4*>(x)[i], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
    reinterpret_cast<uint4*>(xq)[i] = ik_pack8(f);
}

// fp8_actq_kernel for the second projection of a pair: nothing to do when its input scale equals the first's (the first
// launch's xq is then this one's, bit for bit; the pair GEMM reads that one), else the same quantisation into xq.
__global__ void __launch_bounds__(256)
fp8_actq_cond_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ xq, const float* __restrict__ sc,
                     const float* __restrict__ sc_first, int64_t n8) {
    const float s_in = sc[1];
    if (!(s_in > 0.f) || s_in == sc_first[1]) return;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    float f[8];
    ik_unpack8(reinterpret_cast<const uint4*>(x)[i], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
    reinterpret_cast<uint4*>(xq)[i] = ik_pack8(f);
}

template <typename OutT>
int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const ik_bf16* xq, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    cudaStream_t s = IK_STREAM(stream);
    const int vec = (sizeof(OutT) == 2 && N % 8 == 0 && ik_aligned16(out->data)) ? 1 : 0;
    wq::gemm_by_m<wq::FMT_FP8, OutT>((const ik_bf16*)x->data, xq, (const uint8_t*)w->data, nullptr, (const float*)sc->data,
                                     (OutT*)out->data, (int)M, (int)N, (int)K, vec, s);
    return ik_launch_status();
}

// ---- matmul_fp8_gemm_reg ------------------------------------------------------------------------------------------------
// The GEMM of wq_gemm.cuh (FMT_FP8) with the weights dequantised in registers instead of a bf16 tile in shared memory, and
// a 128 x 128 block tile instead of 64 x 128 (docs/results/2026-10-19-cuda-prefill-3.md). Bitwise the results of
// `matmul_fp8_gemm` for every finite e4m3 weight: the A fragments are the same ldmatrix loads of the same bf16 activations
// (x, or the e4m3-valued bf16 xq of fp8_actq_kernel), each B fragment holds the same two bf16 values per register that
// ldmatrix of the dequantised tile gives (k = 2t, 2t+1 and 2t+8, 2t+9 of row n = lane / 4 for t = lane % 4), every output
// element's accumulator goes through the same chain of m16n8k16 mma (k16 steps in order, C = the accumulator), its
// position inside the m16n8 fragment is the same (tile origins are multiples of 16 rows and of 128 columns, warp column
// 32 * (warp % 4)), and the epilogue is the same scale and rounding. e4m3 -> bf16 is exact: the 7 magnitude bits go into
// bits 10..4 of the bf16 (exponent field 0..15, mantissa 3 bits), the sign into bit 15, and one bf16 multiply by 2^120
// (exact; e4m3 subnormals land on bf16 subnormals first, which bf16 arithmetic keeps) restores the value. The NaN codes
// 0x7F / 0xFF are the one difference: they become +-480 here (no checkpoint has NaN weights).
//
// Block = 8 warps, 2 along M x 4 along N, warp tile 64 rows x 32 columns (4 x 4 m16n8 tiles). Per 64-wide k step: the A
// tile (128 x 64 bf16) and the raw weight tile (128 rows x 64 bytes) in one 3-slot cp.async ring (86 KB of shared memory,
// one block per SM). Needs K % 128 == 0 (the rule of the other fp8 GEMMs; K % 64 would do).
namespace reg {

constexpr int BM = 128;

// The schedule (all bitwise the same results, lane S1 docs/results/2026-10-22-s1-spark-prefill.md): WNW warps along N
// (2 along M), JW n8 tiles per warp (warp tile 64 x 8 JW, block tile BM x 8 JW WNW), ST ring slots of BKT-wide k steps,
// MINB blocks per SM (launch bounds). Cfg<4, 3> is the round-3 kernel (128 x 128, k 64, 3 slots, 86 KB).
template <int WNW, int ST, int JW_ = 4, int BKT = 64, int MINB_ = 1, bool PIPE_ = false>
struct Cfg {
    static constexpr int WN = WNW, JW = JW_, BK = BKT, MINB = MINB_;
    static constexpr bool PIPE = PIPE_;  // fragment-pipelined main loop
    static constexpr int BN = 8 * JW * WNW, NT = 64 * WNW, STAGES = ST;
    static constexpr int LDA = BK + 8;   // bf16 elements per A smem row (k 64: 144 bytes, k 32: 80; ldmatrix rows hit distinct bank groups)
    static constexpr int LDW = BK + 16;  // bytes per weight smem row (80 / 48: the 8 rows a B load touches hit distinct banks)
    static constexpr int A_BYTES = BM * LDA * 2;
    static constexpr int W_BYTES = BN * LDW;
    static constexpr int STAGE_BYTES = A_BYTES + W_BYTES;
    static constexpr size_t SMEM = (size_t)STAGES * STAGE_BYTES;
    static constexpr int CPRA = BK / 8, CPRW = BK / 16;  // 16-byte chunks per A / weight row and stage
    static_assert(BK == 32 || BK == 64, "k step");
    static_assert(BM * CPRA % NT == 0 && BN * CPRW % NT == 0, "copy split");
    static_assert(BM * (BN + 8) * 2 <= STAGES * STAGE_BYTES, "epilogue staging must fit in the ring");
};

// two e4m3 bytes (bytes p and p + 1 of w, p = 0 or 2) -> bf16x2 (byte p in the low half), exact
IK_DEVI uint32_t e4m3x2_to_bf16x2(uint32_t w, uint32_t sel) {
    const uint32_t y = __byte_perm(w, 0u, sel);                   // [0, x_lo, 0, x_hi]
    const uint32_t f = (y & 0x80008000u) | ((y >> 4) & 0x07F007F0u); // sign, then exponent:mantissa at bits 10..4
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&f);
    const __nv_bfloat162 k = __halves2bfloat162(__ushort_as_bfloat16((unsigned short)0x7B80u),
                                                __ushort_as_bfloat16((unsigned short)0x7B80u));  // 2^120
    v = __hmul2(v, k);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// One 128 x BN output tile (block `bid` of the problem's grid). The body of gemm_reg_kernel, and of each half of
// gemm_reg_pair_kernel's grid: the same instructions per tile, so the pair's outputs are bitwise those of two calls.
template <class C, typename OutT>
IK_DEVI void gemm_reg_tile(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ xq, const uint8_t* __restrict__ w,
                           const float* __restrict__ gs, OutT* __restrict__ out, int M, int N, int K, int vec_out,
                           int nfast, int bid, unsigned char* smem) {
    constexpr int BN = C::BN, NT = C::NT, STAGES = C::STAGES, STAGE_BYTES = C::STAGE_BYTES;
    constexpr int BK = C::BK, LDA = C::LDA, LDW = C::LDW, A_BYTES = C::A_BYTES, JW = C::JW, WNW = C::WN;
    constexpr int CPRA = C::CPRA, CPRW = C::CPRW;
    constexpr int AQ = BM * CPRA / NT;  // 16-byte A chunks per thread and stage
    constexpr int WQ = BN * CPRW / NT;  // 16-byte weight chunks per thread and stage
    const int nmt = (M + BM - 1) / BM, nnt = (N + BN - 1) / BN;
    const int m0 = (nfast ? bid / nnt : bid % nmt) * BM;
    const int n0 = (nfast ? bid % nnt : bid / nmt) * BN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp % WNW, wm = warp / WNW;
    const int S = K / BK;
    const bool actq = gs[1] > 0.f;
    const ik_bf16* const a_src = actq ? xq : x;
    const float s_out = gs[0] * (actq ? gs[1] : 1.f);
    auto a_tile = [&](int slot) { return reinterpret_cast<ik_bf16*>(smem + slot * STAGE_BYTES); };
    auto w_tile = [&](int slot) { return smem + slot * STAGE_BYTES + A_BYTES; };

    // copies: A row tid / CPRA + (NT / CPRA) q, 16-byte chunk tid % CPRA (q < AQ); weight row tid / CPRW + (NT / CPRW) q,
    // chunk tid % CPRW (q < WQ)
    const ik_bf16* asrc[AQ];
    bool av[AQ];
#pragma unroll
    for (int q = 0; q < AQ; ++q) {
        const int r = tid / CPRA + (NT / CPRA) * q;
        av[q] = m0 + r < M;
        asrc[q] = a_src + (int64_t)(av[q] ? m0 + r : 0) * K + (tid % CPRA) * 8;
    }
    const uint8_t* wsrc[WQ];
    bool wv[WQ];
#pragma unroll
    for (int q = 0; q < WQ; ++q) {
        const int r = tid / CPRW + (NT / CPRW) * q;
        wv[q] = n0 + r < N;
        wsrc[q] = w + (int64_t)(wv[q] ? n0 + r : 0) * K + (tid % CPRW) * 16;
    }
    auto issue = [&](int st) {
        const int slot = st % STAGES;
        ik_bf16* a = a_tile(slot);
        unsigned char* b = w_tile(slot);
#pragma unroll
        for (int q = 0; q < AQ; ++q)
            ik_cp_async16(a + (tid / CPRA + (NT / CPRA) * q) * LDA + (tid % CPRA) * 8, asrc[q] + (int64_t)st * BK, av[q] ? 16 : 0);
#pragma unroll
        for (int q = 0; q < WQ; ++q)
            ik_cp_async16(b + (tid / CPRW + (NT / CPRW) * q) * LDW + (tid % CPRW) * 16, wsrc[q] + (int64_t)st * BK, wv[q] ? 16 : 0);
    };

    // B fragment source: row wn * 32 + j * 8 + (lane >> 2) of the weight tile, the 32-bit word holding k bytes 2t, 2t+1
    // (and +8 for the second register), t = lane & 3
    const int t = lane & 3;
    const int boff = (wn * 8 * JW + (lane >> 2)) * LDW + 4 * (t >> 1);
    const uint32_t p = 2u * (uint32_t)(t & 1);
    const uint32_t sel = 4u | (p << 4) | (4u << 8) | ((p + 1u) << 12);

    float acc[4][JW][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < JW; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < S) issue(st);
        ik_cp_async_commit();
    }
    if constexpr (C::PIPE) {
        // Fragment-pipelined main loop (lane S1): the A and B fragments of k16 step kk + 1 are loaded (and B dequantised)
        // while the mma of step kk issue, and the barrier that makes the next ring slot visible sits before the last k16
        // step of a stage, whose fragments are already in registers. Same mma per accumulator in the same order as the
        // loop below: bitwise the same results.
        constexpr int KS = BK / 16;
        static_assert(KS % 2 == 0, "fragment double buffer");
        uint32_t fa[2][4][4], fb[2][JW][2];
        auto load_frag = [&](int slot, int kk, uint32_t (&a)[4][4], uint32_t (&b)[JW][2]) {
            const ik_bf16* at = a_tile(slot);
            const unsigned char* bt = w_tile(slot) + boff + kk * 16;
#pragma unroll
            for (int j = 0; j < JW; ++j) {
                const uint32_t lo = *reinterpret_cast<const uint32_t*>(bt + j * 8 * LDW);
                const uint32_t hi = *reinterpret_cast<const uint32_t*>(bt + j * 8 * LDW + 8);
                b[j][0] = e4m3x2_to_bf16x2(lo, sel);
                b[j][1] = e4m3x2_to_bf16x2(hi, sel);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = wm * 64 + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ik_ldmatrix_x4(a[i][0], a[i][1], a[i][2], a[i][3], at + r * LDA + kk * 16 + (lane >> 4) * 8);
            }
        };
        ik_cp_async_wait_group<STAGES - 2>();  // stage 0 has landed
        __syncthreads();
        if (STAGES - 1 < S) issue(STAGES - 1);
        ik_cp_async_commit();
        load_frag(0, 0, fa[0], fb[0]);
        for (int s = 0; s < S; ++s) {
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                const int cur = kk & 1, nxt = cur ^ 1;
                if (kk < KS - 1) {
                    load_frag(s % STAGES, kk + 1, fa[nxt], fb[nxt]);
                } else if (s + 1 < S) {
                    // stage s + 1 landed and visible; every warp has read its last fragments of stage s, so its slot
                    // takes stage s + STAGES
                    ik_cp_async_wait_group<STAGES - 2>();
                    __syncthreads();
                    if (s + STAGES < S) issue(s + STAGES);
                    ik_cp_async_commit();
                    load_frag((s + 1) % STAGES, 0, fa[nxt], fb[nxt]);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < JW; ++j)
                        ik_mma_bf16_16816(acc[i][j], fa[cur][i][0], fa[cur][i][1], fa[cur][i][2], fa[cur][i][3], fb[cur][j][0],
                                          fb[cur][j][1]);
            }
        }
    } else {
    for (int s = 0; s < S; ++s) {
        ik_cp_async_wait_group<STAGES - 2>();  // stage s has landed
        __syncthreads();                       // visible to all; everyone is done with stage s-1's slot
        if (s + STAGES - 1 < S) issue(s + STAGES - 1);
        ik_cp_async_commit();
        const ik_bf16* at = a_tile(s % STAGES);
        const unsigned char* bt = w_tile(s % STAGES) + boff;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t b[JW][2];
#pragma unroll
            for (int j = 0; j < JW; ++j) {
                const uint32_t lo = *reinterpret_cast<const uint32_t*>(bt + j * 8 * LDW + kk);
                const uint32_t hi = *reinterpret_cast<const uint32_t*>(bt + j * 8 * LDW + kk + 8);
                b[j][0] = e4m3x2_to_bf16x2(lo, sel);
                b[j][1] = e4m3x2_to_bf16x2(hi, sel);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t a0, a1, a2, a3;
                const int r = wm * 64 + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ik_ldmatrix_x4(a0, a1, a2, a3, at + r * LDA + kk + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < JW; ++j) ik_mma_bf16_16816(acc[i][j], a0, a1, a2, a3, b[j][0], b[j][1]);
            }
        }
    }
    }
    ik_cp_async_wait_all();

    // ---- epilogue (as wq_gemm.cuh) ----
    if (sizeof(OutT) == 2 && vec_out) {
        constexpr int OS = BN + 8;
        __syncthreads();
        ik_bf16* const ob = reinterpret_cast<ik_bf16*>(smem);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int rl = wm * 64 + i * 16 + (lane >> 2) + hh * 8;
#pragma unroll
                for (int j = 0; j < JW; ++j) {
                    const int cl = wn * 8 * JW + j * 8 + (lane & 3) * 2;
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) =
                        __floats2bfloat162_rn(acc[i][j][2 * hh] * s_out, acc[i][j][2 * hh + 1] * s_out);
                }
            }
        __syncthreads();
        constexpr int CPR = BN / 8;
        for (int c = tid; c < BM * CPR; c += NT) {
            const int rl = c / CPR, part = c % CPR;
            if (m0 + rl >= M || n0 + part * 8 >= N) continue;  // N % 8 == 0 when vec_out
            *reinterpret_cast<uint4*>(reinterpret_cast<ik_bf16*>(out) + (int64_t)(m0 + rl) * N + n0 + part * 8) =
                *reinterpret_cast<const uint4*>(ob + rl * OS + part * 8);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < JW; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int row = m0 + wm * 64 + i * 16 + (lane >> 2) + hh * 8;
                    const int col = n0 + wn * 8 * JW + j * 8 + (lane & 3) * 2;
                    if (row >= M) continue;
                    if (col < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * hh] * s_out);
                    if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * hh + 1] * s_out);
                }
    }
}

template <class C, typename OutT>
__global__ void __launch_bounds__(C::NT, C::MINB)
gemm_reg_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ xq, const uint8_t* __restrict__ w,
                const float* __restrict__ gs, OutT* __restrict__ out, int M, int N, int K, int vec_out, int nfast) {
    extern __shared__ __align__(16) unsigned char smem[];
    gemm_reg_tile<C, OutT>(x, xq, w, gs, out, M, N, K, vec_out, nfast, blockIdx.x, smem);
}

// Two projections of one activation in one grid (Op::MatmulFp8Pair at prefill: GDN in_proj_qkv + in_proj_z, attention
// k + v): blocks [0, nblk0) are problem 0's tiles, the rest problem 1's, each tile exactly as gemm_reg_kernel computes it.
// Problem 1 reads problem 0's quantised activations when the two input scales are equal (the e4m3 activations are then
// the same bytes: fp8_actq_kernel is elementwise in x and input_scale), else its own (written by fp8_actq_cond_kernel).
template <class C, typename OutT>
__global__ void __launch_bounds__(C::NT, C::MINB)
gemm_reg_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ xq0, const ik_bf16* __restrict__ xq1,
                     const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1, const float* __restrict__ gs0,
                     const float* __restrict__ gs1, OutT* __restrict__ out0, OutT* __restrict__ out1, int M, int N0, int N1,
                     int K, int vec0, int vec1, int nfast0, int nfast1, int nblk0) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int bid = blockIdx.x;
    if (bid < nblk0) {
        gemm_reg_tile<C, OutT>(x, xq0, w0, gs0, out0, M, N0, K, vec0, nfast0, bid, smem);
    } else {
        const ik_bf16* const q = gs1[1] == gs0[1] ? xq0 : xq1;
        gemm_reg_tile<C, OutT>(x, q, w1, gs1, out1, M, N1, K, vec1, nfast1, bid - nblk0, smem);
    }
}

template <class C, typename OutT>
void launch_pair(const ik_bf16* x, const ik_bf16* xq0, const ik_bf16* xq1, const uint8_t* w0, const uint8_t* w1,
                 const float* sc0, const float* sc1, OutT* out0, OutT* out1, int M, int N0, int N1, int K, int vec0, int vec1,
                 cudaStream_t s) {
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_reg_pair_kernel<C, OutT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM);
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_reg_pair_kernel<C, OutT>),
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)attr_set;
    // the tile order of each half is launch()'s
    const int nfast0 = (double)M * 2.0 > (double)N0 ? 1 : 0, nfast1 = (double)M * 2.0 > (double)N1 ? 1 : 0;
    const int64_t nblk0 = (int64_t)((M + BM - 1) / BM) * ((N0 + C::BN - 1) / C::BN);
    const int64_t nblk1 = (int64_t)((M + BM - 1) / BM) * ((N1 + C::BN - 1) / C::BN);
    gemm_reg_pair_kernel<C, OutT><<<dim3((unsigned)(nblk0 + nblk1)), C::NT, C::SMEM, s>>>(
        x, xq0, xq1, w0, w1, sc0, sc1, out0, out1, M, N0, N1, K, vec0, vec1, nfast0, nfast1, (int)nblk0);
}

template <class C, typename OutT>
void launch(const ik_bf16* x, const ik_bf16* xq, const uint8_t* w, const float* sc, OutT* out, int M, int N, int K, int vec,
            cudaStream_t s) {
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_reg_kernel<C, OutT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM);
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_reg_kernel<C, OutT>),
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)attr_set;
    // activation bytes vs weight bytes: the larger operand is the one consecutive blocks share (as wq_gemm.cuh)
    const int nfast = (double)M * 2.0 > (double)N ? 1 : 0;
    const int64_t nblk = (int64_t)((M + BM - 1) / BM) * ((N + C::BN - 1) / C::BN);
    gemm_reg_kernel<C, OutT><<<dim3((unsigned)nblk), C::NT, C::SMEM, s>>>(x, xq, w, sc, out, M, N, K, vec, nfast);
}

// The schedules (every one bitwise the same results; the choice is speed only). INTARSIA_CUDA_FP8_REG_CFG=<id> forces one
// for every shape (microbenchmarks); unset, pick_cfg chooses by shape.
using Cfg0 = Cfg<4, 3>;              // 128 x 128, k 64, 3 slots (round 3), 86 KB, 1 block / SM
using Cfg1 = Cfg<4, 2, 8, 64, 1>;    // 128 x 256, k 64, 2 slots, 78 KB
using Cfg2 = Cfg<4, 3, 8, 32, 1>;    // 128 x 256, k 32, 3 slots, 68 KB
using Cfg3 = Cfg<4, 4, 8, 32, 1>;    // 128 x 256, k 32, 4 slots, 90 KB
using Cfg4 = Cfg<4, 3, 4, 32, 2>;    // 128 x 128, k 32, 3 slots, 49 KB, 2 blocks / SM
using Cfg5 = Cfg<4, 4, 4, 32, 1>;    // 128 x 128, k 32, 4 slots, 66 KB
using Cfg6 = Cfg<2, 3, 8, 32, 2>;    // 128 x 128 with 4 warps of 64 x 64, k 32, 3 slots, 49 KB, 2 blocks / SM
// the same with the fragment-pipelined main loop
using Cfg7 = Cfg<4, 3, 4, 64, 1, true>;   // as Cfg0
using Cfg8 = Cfg<4, 2, 8, 64, 1, true>;   // as Cfg1
using Cfg9 = Cfg<4, 4, 4, 32, 1, true>;   // as Cfg5
using Cfg10 = Cfg<4, 3, 8, 32, 1, true>;  // as Cfg2
using Cfg11 = Cfg<4, 3, 4, 32, 2, true>;  // as Cfg4
constexpr int kNumCfg = 12;

int forced_cfg() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_CUDA_FP8_REG_CFG");
        if (!e || !*e) return -1;
        const int c = atoi(e);
        return (c >= 0 && c < kNumCfg) ? c : -1;
    }();
    return v;
}

int pick_cfg(int M, int N, int K) {
    (void)M;
    (void)N;
    (void)K;
    const int f = forced_cfg();
    return f >= 0 ? f : 0;
}

template <typename OutT>
void launch_cfg(int cfg, const ik_bf16* x, const ik_bf16* xq, const uint8_t* w, const float* sc, OutT* out, int M, int N,
                int K, int vec, cudaStream_t s) {
    switch (cfg) {
        case 1: launch<Cfg1, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 2: launch<Cfg2, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 3: launch<Cfg3, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 4: launch<Cfg4, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 5: launch<Cfg5, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 6: launch<Cfg6, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 7: launch<Cfg7, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 8: launch<Cfg8, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 9: launch<Cfg9, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 10: launch<Cfg10, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        case 11: launch<Cfg11, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
        default: launch<Cfg0, OutT>(x, xq, w, sc, out, M, N, K, vec, s); break;
    }
}

template <typename OutT>
void launch_pair_cfg(int cfg, const ik_bf16* x, const ik_bf16* xq0, const ik_bf16* xq1, const uint8_t* w0, const uint8_t* w1,
                     const float* sc0, const float* sc1, OutT* out0, OutT* out1, int M, int N0, int N1, int K, int vec0,
                     int vec1, cudaStream_t s) {
#define IK_PAIR(C) launch_pair<C, OutT>(x, xq0, xq1, w0, w1, sc0, sc1, out0, out1, M, N0, N1, K, vec0, vec1, s)
    switch (cfg) {
        case 1: IK_PAIR(Cfg1); break;
        case 2: IK_PAIR(Cfg2); break;
        case 3: IK_PAIR(Cfg3); break;
        case 4: IK_PAIR(Cfg4); break;
        case 5: IK_PAIR(Cfg5); break;
        case 6: IK_PAIR(Cfg6); break;
        case 7: IK_PAIR(Cfg7); break;
        case 8: IK_PAIR(Cfg8); break;
        case 9: IK_PAIR(Cfg9); break;
        case 10: IK_PAIR(Cfg10); break;
        case 11: IK_PAIR(Cfg11); break;
        default: IK_PAIR(Cfg0); break;
    }
#undef IK_PAIR
}

template <typename OutT>
int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const ik_bf16* xq, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    cudaStream_t s = IK_STREAM(stream);
    const int vec = (sizeof(OutT) == 2 && N % 8 == 0 && ik_aligned16(out->data)) ? 1 : 0;
    // a grid of fewer than 24 128 x 128 tiles (small N at small M: the 35B k/v projections below 1024 tokens) leaves most
    // of the 48 SMs idle; the 64-row tiles of wq_gemm.cuh give the same bits there
    const int64_t tiles = ((M + BM - 1) / BM) * ((N + Cfg<4, 3>::BN - 1) / Cfg<4, 3>::BN);
    if (tiles < 24) {
        wq::gemm_by_m<wq::FMT_FP8, OutT>((const ik_bf16*)x->data, xq, (const uint8_t*)w->data, nullptr,
                                         (const float*)sc->data, (OutT*)out->data, (int)M, (int)N, (int)K, vec, s);
    } else {
        launch_cfg<OutT>(pick_cfg((int)M, (int)N, (int)K), (const ik_bf16*)x->data, xq, (const uint8_t*)w->data,
                         (const float*)sc->data, (OutT*)out->data, (int)M, (int)N, (int)K, vec, s);
    }
    return ik_launch_status();
}

}  // namespace reg

// shared argument checks of matmul_fp8_gemm and matmul_fp8_gemm_reg; launches the activation quantiser, returns xq
int check_args(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, bool* empty) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 128 != 0 || w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    *empty = M == 0 || N == 0;
    if (*empty) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    return IK_OK;
}

int prologue(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, const ik_scratch* scratch,
             void* stream, ik_bf16** xq_out, bool* empty) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 128 != 0 || w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    *empty = M == 0 || N == 0;
    if (*empty) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    // scratch: the quantised activations [M, K] bf16 (always reserved; used when input_scale > 0, known only on device)
    const int64_t xq_bytes = M * K * 2;
    if (!scratch || !scratch->data || scratch->bytes < xq_bytes) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    ik_bf16* xq = (ik_bf16*)scratch->data;
    cudaStream_t s = IK_STREAM(stream);
    const int64_t n8 = M * K / 8;
    fp8_actq_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((const ik_bf16*)x->data, xq, (const float*)scales->data, n8);
    IK_TRY(ik_launch_status());
    *xq_out = xq;
    return IK_OK;
}

}  // namespace

extern "C" {

int ik_fp8_gemm_weight_only_internal(const void* x, const void* w, const void* scales, void* out, int out_is_f32, int64_t M,
                                     int64_t N, int64_t K, void* stream) {
    cudaStream_t s = IK_STREAM(stream);
    const uint8_t* wp = (const uint8_t*)w;
    const float* sc = (const float*)scales;
    if (out_is_f32) {
        wq::gemm_by_m<wq::FMT_FP8, float>((const ik_bf16*)x, nullptr, wp, nullptr, sc, (float*)out, (int)M, (int)N, (int)K, 0, s, 1);
    } else {
        const int vec = (N % 8 == 0 && ik_aligned16(out)) ? 1 : 0;
        wq::gemm_by_m<wq::FMT_FP8, ik_bf16>((const ik_bf16*)x, nullptr, wp, nullptr, sc, (ik_bf16*)out, (int)M, (int)N, (int)K, vec, s, 1);
    }
    return ik_launch_status();
}

int ik_matmul_fp8_gemm(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                       const ik_scratch* scratch, void* stream) {
    ik_bf16* xq = nullptr;
    bool empty = false;
    IK_TRY(prologue(x, w, scales, out, scratch, stream, &xq, &empty));
    if (empty) return IK_OK;
    return out->dtype == IK_DTYPE_F32 ? run<float>(x, w, scales, out, xq, stream) : run<ik_bf16>(x, w, scales, out, xq, stream);
}

// ---- matmul_fp8_gemm_cublas (lane S1) -------------------------------------------------------------------------------
// The same GEMM on cuBLAS's bf16 tensor-core kernels (OPT-IN, INTARSIA_CUDA_FP8_CUBLAS=1): each fp8 weight is expanded
// once to its exact bf16 values (the B values of gemm_reg; NaN codes -> +-480 likewise) into a device copy kept for the
// life of the weight's allocation (2 bytes per weight: +2.6 GB on the 35B, outside the engine's memory plan; dropped by
// ik_rt_free), the activations are quantised as fp8_actq_kernel does (or copied, weight only), and cublasGemmEx (compute
// 32F) runs with alpha = gemm_reg's epilogue scale. On sm_121 / CUDA 13.0 at the 35B prefill shapes cuBLAS forms every
// output element by the same m16n8k16 chain in k order with no split-K, so the bytes equal matmul_fp8_gemm_reg's
// (tests/fp8_gemm_reg.rs fp8_gemm_cublas_is_bitwise_gemm_reg and quantmoe probe_cublas_bf16_vs_gemm_reg_bits check it on
// the device). Falls back to matmul_fp8_gemm_reg (same bytes) during graph capture, for weights outside an ik_rt_malloc
// allocation, and when the copy cannot be allocated. Scratch: 256 B ({alpha, beta}) + xq [M, K] bf16.
namespace cub {

__global__ void __launch_bounds__(256) e4m3_to_bf16_kernel(const uint8_t* __restrict__ w, ik_bf16* __restrict__ o, int64_t n8) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint2 v = reinterpret_cast<const uint2*>(w)[i];
    uint4 r;
    // bytes 0,1 / 2,3 of each word: the selector of gemm_reg's B decode (p = 0 / 2, byte p in the low half)
    r.x = reg::e4m3x2_to_bf16x2(v.x, 4u | (0u << 4) | (4u << 8) | (1u << 12));
    r.y = reg::e4m3x2_to_bf16x2(v.x, 4u | (2u << 4) | (4u << 8) | (3u << 12));
    r.z = reg::e4m3x2_to_bf16x2(v.y, 4u | (0u << 4) | (4u << 8) | (1u << 12));
    r.w = reg::e4m3x2_to_bf16x2(v.y, 4u | (2u << 4) | (4u << 8) | (3u << 12));
    reinterpret_cast<uint4*>(o)[i] = r;
}

// xq = fp8_actq_kernel's values when the input scale is positive, else x itself (cuBLAS takes a host pointer, and the scale
// lives on the device); thread 0 also writes {alpha, beta} = {gemm_reg's s_out, 0}. `first`: when non-null and its input
// scale equals this one, nothing is quantised (that launch's xq holds the same bytes) but alpha / beta are still written.
__global__ void __launch_bounds__(256)
actq_alpha_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ xq, const float* __restrict__ sc,
                  const float* __restrict__ first, float* __restrict__ ab, int64_t n8) {
    const float s_in = sc[1];
    const bool actq = s_in > 0.f;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        ab[0] = sc[0] * (actq ? s_in : 1.f);
        ab[1] = 0.f;
    }
    if (first && first[1] == s_in) return;
    if (i >= n8) return;
    const uint4 u = reinterpret_cast<const uint4*>(x)[i];
    if (!actq) {
        reinterpret_cast<uint4*>(xq)[i] = u;
        return;
    }
    float f[8];
    ik_unpack8(u, f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
    reinterpret_cast<uint4*>(xq)[i] = ik_pack8(f);
}

constexpr size_t kAbBytes = 256;

struct Copy {
    ik_bf16* bf;
    int64_t n, k;
    uintptr_t base;
};
std::mutex g_mu;
std::map<std::pair<int, uintptr_t>, Copy> g_copies;  // (device, weight address)
bool g_alloc_failed = false;

// the bf16 copy of w (made on stream s on first use), or nullptr: fall back
const ik_bf16* weight_copy(const ik_tensor* w, cudaStream_t s) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    const int64_t N = w->shape[0], K = w->shape[1];
    std::lock_guard<std::mutex> lock(g_mu);
    const auto key = std::make_pair(dev, (uintptr_t)w->data);
    auto it = g_copies.find(key);
    if (it != g_copies.end()) return (it->second.n == N && it->second.k == K) ? it->second.bf : nullptr;
    if (g_alloc_failed) return nullptr;
    uintptr_t base = 0;
    if (ik_rt_alloc_base(w->data, &base) != 0) return nullptr;
    void* p = nullptr;
    if (cudaMalloc(&p, (size_t)(N * K) * 2) != cudaSuccess) {
        cudaGetLastError();
        g_alloc_failed = true;
        return nullptr;
    }
    const int64_t n8 = N * K / 8;
    e4m3_to_bf16_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((const uint8_t*)w->data, (ik_bf16*)p, n8);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(p);
        return nullptr;
    }
    g_copies[key] = Copy{(ik_bf16*)p, N, K, base};
    return (const ik_bf16*)p;
}

}  // namespace cub

extern "C" void ik_fp8_bf16_cache_forget(const void* base) {
    std::lock_guard<std::mutex> lock(cub::g_mu);
    for (auto it = cub::g_copies.begin(); it != cub::g_copies.end();) {
        if (it->second.base == (uintptr_t)base) {
            cudaFree(it->second.bf);
            it = cub::g_copies.erase(it);
        } else {
            ++it;
        }
    }
}

int ik_matmul_fp8_gemm_reg(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                           const ik_scratch* scratch, void* stream);

int ik_matmul_fp8_gemm_cublas(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                              const ik_scratch* scratch, void* stream) {
    bool empty = false;
    IK_TRY(check_args(x, w, scales, out, &empty));
    if (empty) return IK_OK;
    const int64_t M = x->shape[0], K = x->shape[1];
    const size_t need = cub::kAbBytes + (size_t)(M * K) * 2;
    if (!scratch || !scratch->data || (size_t)scratch->bytes < need || (reinterpret_cast<uintptr_t>(scratch->data) & 255u))
        return IK_ERR_INVALID_ARG;
    cudaStream_t s = IK_STREAM(stream);
    const ik_scratch rest{(char*)scratch->data + cub::kAbBytes, scratch->bytes - (int64_t)cub::kAbBytes};
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (out->dtype != IK_DTYPE_BF16 || !ik_aligned16(out->data) || cudaStreamIsCapturing(s, &cs) != cudaSuccess ||
        cs != cudaStreamCaptureStatusNone)
        return ik_matmul_fp8_gemm_reg(x, w, scales, out, &rest, stream);
    const ik_bf16* const wbf = cub::weight_copy(w, s);
    if (!wbf) return ik_matmul_fp8_gemm_reg(x, w, scales, out, &rest, stream);
    float* const ab = (float*)scratch->data;
    ik_bf16* const xq = (ik_bf16*)rest.data;
    const int64_t n8 = M * K / 8;
    cub::actq_alpha_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((const ik_bf16*)x->data, xq,
                                                                         (const float*)scales->data, nullptr, ab, n8);
    IK_TRY(ik_launch_status());
    return ik_cublas_gemm_bf16_devalpha(xq, wbf, out->data, M, w->shape[0], K, ab, s);
}

int ik_matmul_fp8_gemm_reg(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                           const ik_scratch* scratch, void* stream) {
    ik_bf16* xq = nullptr;
    bool empty = false;
    IK_TRY(prologue(x, w, scales, out, scratch, stream, &xq, &empty));
    if (empty) return IK_OK;
    return out->dtype == IK_DTYPE_F32 ? reg::run<float>(x, w, scales, out, xq, stream)
                                      : reg::run<ik_bf16>(x, w, scales, out, xq, stream);
}

// Op::MatmulFp8Pair at prefill (M >= 256, registry): bitwise two ik_matmul_fp8_gemm_reg calls on the same x. The activation
// quantiser runs once when the two input scales are equal (checked on the device), and the two GEMMs share one grid when
// both take the 128 x 128 path (k / v at N = 512: 64 tiles each, 1.3 waves of 48 SMs alone, 2.7 together). Scratch: two
// quantised activations, [M, K] bf16 each.
int ik_matmul_fp8_pair_gemm_reg(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* sc0, const ik_tensor* w1,
                                const ik_tensor* sc1, ik_tensor* out0, ik_tensor* out1, const ik_scratch* scratch,
                                void* stream) {
    bool e0 = false, e1 = false;
    IK_TRY(check_args(x, w0, sc0, out0, &e0));
    IK_TRY(check_args(x, w1, sc1, out1, &e1));
    if (out0->dtype != out1->dtype) return IK_ERR_DTYPE;
    if (e0 || e1) {  // a degenerate half: the two single calls (scratch holds one xq)
        IK_TRY(ik_matmul_fp8_gemm_reg(x, w0, sc0, out0, scratch, stream));
        return ik_matmul_fp8_gemm_reg(x, w1, sc1, out1, scratch, stream);
    }
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    const int64_t xq_elems = M * K;
    if (!scratch || !scratch->data || scratch->bytes < 2 * xq_elems * 2) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    ik_bf16* const xq0 = (ik_bf16*)scratch->data;
    ik_bf16* const xq1 = xq0 + xq_elems;  // K % 128 == 0: 16-byte aligned
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* const xp = (const ik_bf16*)x->data;
    const float* const s0 = (const float*)sc0->data;
    const float* const s1 = (const float*)sc1->data;
    const int64_t n8 = xq_elems / 8;
    const unsigned qblocks = (unsigned)((n8 + 255) / 256);
    fp8_actq_kernel<<<qblocks, 256, 0, s>>>(xp, xq0, s0, n8);
    IK_TRY(ik_launch_status());
    using C = reg::Cfg0;  // the small-grid rule of reg::run (128 x 128 tile count)
    const int64_t mt = (M + reg::BM - 1) / reg::BM;
    const int64_t t0 = mt * ((N0 + C::BN - 1) / C::BN), t1 = mt * ((N1 + C::BN - 1) / C::BN);
    if (t0 < 24 || t1 < 24) {  // reg::run's small-grid path for a half: the single calls, own quantiser
        fp8_actq_kernel<<<qblocks, 256, 0, s>>>(xp, xq1, s1, n8);
        IK_TRY(ik_launch_status());
        if (out0->dtype == IK_DTYPE_F32) {
            IK_TRY(reg::run<float>(x, w0, sc0, out0, xq0, stream));
            return reg::run<float>(x, w1, sc1, out1, xq1, stream);
        }
        IK_TRY(reg::run<ik_bf16>(x, w0, sc0, out0, xq0, stream));
        return reg::run<ik_bf16>(x, w1, sc1, out1, xq1, stream);
    }
    fp8_actq_cond_kernel<<<qblocks, 256, 0, s>>>(xp, xq1, s1, s0, n8);
    IK_TRY(ik_launch_status());
    const uint8_t* const wp0 = (const uint8_t*)w0->data;
    const uint8_t* const wp1 = (const uint8_t*)w1->data;
    if (out0->dtype == IK_DTYPE_F32) {
        reg::launch_pair_cfg<float>(reg::pick_cfg((int)M, (int)(N0 > N1 ? N0 : N1), (int)K), xp, xq0, xq1, wp0, wp1, s0, s1,
                                    (float*)out0->data, (float*)out1->data, (int)M, (int)N0, (int)N1, (int)K, 0, 0, s);
    } else {
        const int v0 = (N0 % 8 == 0 && ik_aligned16(out0->data)) ? 1 : 0;
        const int v1 = (N1 % 8 == 0 && ik_aligned16(out1->data)) ? 1 : 0;
        reg::launch_pair_cfg<ik_bf16>(reg::pick_cfg((int)M, (int)(N0 > N1 ? N0 : N1), (int)K), xp, xq0, xq1, wp0, wp1, s0, s1,
                                      (ik_bf16*)out0->data, (ik_bf16*)out1->data, (int)M, (int)N0, (int)N1, (int)K, v0, v1, s);
    }
    return ik_launch_status();
}

}  // extern "C"
