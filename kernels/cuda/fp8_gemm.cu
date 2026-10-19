// matmul_fp8_gemm: same op and numerics as matmul_fp8_mma (fp8.cu), with the memory schedule of wq_gemm.cuh.
// W8A8 (input_scale > 0): the activations are first quantised to e4m3-valued bf16 into scratch (one pass over x), the
// GEMM then reads those. Needs K % 64 == 0.
#include "wq_gemm.cuh"

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

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int LDA = BK + 8;     // bf16 elements per A smem row (144 bytes: ldmatrix rows hit distinct bank groups)
constexpr int LDW = BK + 16;    // bytes per weight smem row (80: the 8 rows a B load touches hit distinct banks)
constexpr int NT = 256;
constexpr int STAGES = 3;
constexpr int A_BYTES = BM * LDA * 2;
constexpr int W_BYTES = BN * LDW;
constexpr int STAGE_BYTES = A_BYTES + W_BYTES;
constexpr size_t SMEM = (size_t)STAGES * STAGE_BYTES;
static_assert(BM * (BN + 8) * 2 <= STAGES * STAGE_BYTES, "epilogue staging must fit in the ring");

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

template <typename OutT>
__global__ void __launch_bounds__(NT)
gemm_reg_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ xq, const uint8_t* __restrict__ w,
                const float* __restrict__ gs, OutT* __restrict__ out, int M, int N, int K, int vec_out, int nfast) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int nmt = (M + BM - 1) / BM, nnt = (N + BN - 1) / BN;
    const int bid = blockIdx.x;
    const int m0 = (nfast ? bid / nnt : bid % nmt) * BM;
    const int n0 = (nfast ? bid % nnt : bid / nmt) * BN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp & 3, wm = warp >> 2;
    const int S = K / BK;
    const bool actq = gs[1] > 0.f;
    const ik_bf16* const a_src = actq ? xq : x;
    const float s_out = gs[0] * (actq ? gs[1] : 1.f);
    auto a_tile = [&](int slot) { return reinterpret_cast<ik_bf16*>(smem + slot * STAGE_BYTES); };
    auto w_tile = [&](int slot) { return smem + slot * STAGE_BYTES + A_BYTES; };

    // copies: A row (tid >> 3) + 32 q, 16-byte chunk tid & 7 (q < 4); weight row (tid >> 2) + 64 q, chunk tid & 3 (q < 2)
    const ik_bf16* asrc[4];
    bool av[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int r = (tid >> 3) + 32 * q;
        av[q] = m0 + r < M;
        asrc[q] = a_src + (int64_t)(av[q] ? m0 + r : 0) * K + (tid & 7) * 8;
    }
    const uint8_t* wsrc[2];
    bool wv[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int r = (tid >> 2) + 64 * q;
        wv[q] = n0 + r < N;
        wsrc[q] = w + (int64_t)(wv[q] ? n0 + r : 0) * K + (tid & 3) * 16;
    }
    auto issue = [&](int st) {
        const int slot = st % STAGES;
        ik_bf16* a = a_tile(slot);
        unsigned char* b = w_tile(slot);
#pragma unroll
        for (int q = 0; q < 4; ++q)
            ik_cp_async16(a + ((tid >> 3) + 32 * q) * LDA + (tid & 7) * 8, asrc[q] + (int64_t)st * BK, av[q] ? 16 : 0);
#pragma unroll
        for (int q = 0; q < 2; ++q)
            ik_cp_async16(b + ((tid >> 2) + 64 * q) * LDW + (tid & 3) * 16, wsrc[q] + (int64_t)st * BK, wv[q] ? 16 : 0);
    };

    // B fragment source: row wn * 32 + j * 8 + (lane >> 2) of the weight tile, the 32-bit word holding k bytes 2t, 2t+1
    // (and +8 for the second register), t = lane & 3
    const int t = lane & 3;
    const int boff = (wn * 32 + (lane >> 2)) * LDW + 4 * (t >> 1);
    const uint32_t p = 2u * (uint32_t)(t & 1);
    const uint32_t sel = 4u | (p << 4) | (4u << 8) | ((p + 1u) << 12);

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < S) issue(st);
        ik_cp_async_commit();
    }
    for (int s = 0; s < S; ++s) {
        ik_cp_async_wait_group<STAGES - 2>();  // stage s has landed
        __syncthreads();                       // visible to all; everyone is done with stage s-1's slot
        if (s + STAGES - 1 < S) issue(s + STAGES - 1);
        ik_cp_async_commit();
        const ik_bf16* at = a_tile(s % STAGES);
        const unsigned char* bt = w_tile(s % STAGES) + boff;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t b[4][2];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
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
                for (int j = 0; j < 4; ++j) ik_mma_bf16_16816(acc[i][j], a0, a1, a2, a3, b[j][0], b[j][1]);
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
                for (int j = 0; j < 4; ++j) {
                    const int cl = wn * 32 + j * 8 + (lane & 3) * 2;
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
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int row = m0 + wm * 64 + i * 16 + (lane >> 2) + hh * 8;
                    const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                    if (row >= M) continue;
                    if (col < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * hh] * s_out);
                    if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * hh + 1] * s_out);
                }
    }
}

template <typename OutT>
int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const ik_bf16* xq, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    cudaStream_t s = IK_STREAM(stream);
    const int vec = (sizeof(OutT) == 2 && N % 8 == 0 && ik_aligned16(out->data)) ? 1 : 0;
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gemm_reg_kernel<OutT>), cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)SMEM);
        return true;
    }();
    (void)attr_set;
    // activation bytes vs weight bytes: the larger operand is the one consecutive blocks share (as wq_gemm.cuh)
    const int nfast = (double)M * 2.0 > (double)N ? 1 : 0;
    const int64_t nblk = (int64_t)((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    gemm_reg_kernel<OutT><<<dim3((unsigned)nblk), NT, SMEM, s>>>((const ik_bf16*)x->data, xq, (const uint8_t*)w->data,
                                                                 (const float*)sc->data, (OutT*)out->data, (int)M, (int)N,
                                                                 (int)K, vec, nfast);
    return ik_launch_status();
}

}  // namespace reg

// shared argument checks of matmul_fp8_gemm and matmul_fp8_gemm_reg; launches the activation quantiser, returns xq
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

int ik_matmul_fp8_gemm_reg(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                           const ik_scratch* scratch, void* stream) {
    ik_bf16* xq = nullptr;
    bool empty = false;
    IK_TRY(prologue(x, w, scales, out, scratch, stream, &xq, &empty));
    if (empty) return IK_OK;
    return out->dtype == IK_DTYPE_F32 ? reg::run<float>(x, w, scales, out, xq, stream)
                                      : reg::run<ik_bf16>(x, w, scales, out, xq, stream);
}

}  // extern "C"
