// matmul_fp8_gemm_f8: W8A8 prefill GEMM on e4m3 tensor cores (mma.sync m16n8k32, e4m3 x e4m3 -> f32).
//
// Same op and numerics as matmul_fp8_gemm / matmul_fp8_mma: with a static input scale the activations are quantised to
// e4m3 (RNE, saturating, x / input_scale), every product e4m3 x e4m3 is exact in f32, and y = weight_scale * input_scale *
// sum. wq_gemm.cuh dequantises each weight tile to bf16 in shared memory once per M tile (BM <= 64 rows), which makes the
// ALUs, not the tensor cores, the limit at M >= 1024; here the e4m3 weights go from global memory straight to the mma
// fragments (cp.async -> smem -> ldmatrix), nothing is converted, and the tensor core does 2x the MACs per instruction
// of the bf16 mma. Only the order of the f32 additions differs: every k32 product sum is formed from zero and added to the
// f32 accumulator with a normal fp32 add (as moe_grouped.cu does), so the kernel is as accurate as the bf16 one.
//
// W8A16 (input_scale <= 0, known only on device) cannot use e4m3 activations: the launcher also runs the bf16 kernel of
// wq_gemm.cuh, and each of the two kernels exits at once when it is not the one for this scale.
//
// Tile: BM x 128 weight rows x 64 k bytes per stage, 4 stages of cp.async. Block = 4 x WM warps (4 along N, 32 columns
// each; WM along M, MT m16 tiles each). Needs K % 128 == 0 (as the other fp8 gemm kernels) and MT even.
#include "wq_gemm.cuh"
#include "ik_internal.h"
#include <stdlib.h>

namespace {

constexpr int BN = 128;         // weight rows per block
constexpr int BKB = 64;         // k (fp8 bytes) per stage
constexpr int LDB = BKB + 16;   // smem row stride in bytes: 8 consecutive rows hit 8 distinct 16-byte bank groups
constexpr int STAGES = 4;

IK_DEVI void ik_mma_e4m3_16832(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// xq = e4m3(x / input_scale) as bytes; same rounding as fp8_gemm.cu's bf16-valued quantiser
__global__ void __launch_bounds__(256)
fp8_actq8_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ xq, const float* __restrict__ sc, int64_t n8) {
    const float s_in = sc[1];
    if (!(s_in > 0.f)) return;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    float f[8];
    ik_unpack8(reinterpret_cast<const uint4*>(x)[i], f);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        lo |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[e], s_in)) << (8 * e);
        hi |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 + e], s_in)) << (8 * e);
    }
    reinterpret_cast<uint2*>(xq)[i] = make_uint2(lo, hi);
}

template <int MT, int WM, typename OutT>
__global__ void __launch_bounds__(128 * WM)
fp8_f8_gemm_kernel(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                   OutT* __restrict__ out, int M, int N, int K, int nfast) {
    if (!(sc[1] > 0.f)) return;  // weight-only activations: the bf16 kernel handles this launch
    static_assert(MT % 2 == 0, "A tile chunks must divide over the threads");
    constexpr int BM = 16 * MT * WM;
    constexpr int NT = 128 * WM;
    constexpr int ASZ = BM * LDB, BSZ = BN * LDB, SSZ = ASZ + BSZ;
    constexpr int AQ = BM * 4 / NT;  // 16-byte A chunks per thread and stage
    constexpr int BQ = BN * 4 / NT;
    extern __shared__ __align__(16) unsigned char smem[];

    const int nmt = (M + BM - 1) / BM, nnt = (N + BN - 1) / BN;
    const int bid = blockIdx.x;
    const int m0 = (nfast ? bid / nnt : bid % nmt) * BM;
    const int n0 = (nfast ? bid % nnt : bid / nmt) * BN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp & 3, wm = warp >> 2;
    const int S = K / BKB;
    const float s_out = sc[0] * sc[1];

    // chunk c of a stage: row c >> 2, 16-byte part c & 3 of the 64 k bytes
    const uint8_t* asrc[AQ];
    const uint8_t* bsrc[BQ];
    bool av[AQ], bv[BQ];
#pragma unroll
    for (int q = 0; q < AQ; ++q) {
        const int r = (tid >> 2) + (NT / 4) * q;
        av[q] = m0 + r < M;
        asrc[q] = xq + (int64_t)(av[q] ? m0 + r : 0) * K + (tid & 3) * 16;
    }
#pragma unroll
    for (int q = 0; q < BQ; ++q) {
        const int r = (tid >> 2) + (NT / 4) * q;
        bv[q] = n0 + r < N;
        bsrc[q] = w + (int64_t)(bv[q] ? n0 + r : 0) * K + (tid & 3) * 16;
    }
    auto issue = [&](int st) {
        unsigned char* a = smem + (st % STAGES) * SSZ;
        unsigned char* b = a + ASZ;
#pragma unroll
        for (int q = 0; q < AQ; ++q) {
            const int r = (tid >> 2) + (NT / 4) * q;
            ik_cp_async16(a + r * LDB + (tid & 3) * 16, asrc[q] + (int64_t)st * BKB, av[q] ? 16 : 0);
        }
#pragma unroll
        for (int q = 0; q < BQ; ++q) {
            const int r = (tid >> 2) + (NT / 4) * q;
            ik_cp_async16(b + r * LDB + (tid & 3) * 16, bsrc[q] + (int64_t)st * BKB, bv[q] ? 16 : 0);
        }
    };

    float acc[MT][4][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
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
        __syncthreads();                       // visible to all; everyone is done with stage s-1's buffer
        if (s + STAGES - 1 < S) issue(s + STAGES - 1);
        ik_cp_async_commit();
        const unsigned char* at = smem + (s % STAGES) * SSZ;
        const unsigned char* bt = at + ASZ;
#pragma unroll
        for (int kk = 0; kk < BKB; kk += 32) {
            uint32_t b[4][2];
#pragma unroll
            for (int pj = 0; pj < 2; ++pj) {  // two n8 tiles per ldmatrix.x4: (tile, k 0-15), (tile, k 16-31), ...
                const int mi = lane >> 3;
                const int nr = wn * 32 + (2 * pj + (mi >> 1)) * 8 + (lane & 7);
                ik_ldmatrix_x4(b[2 * pj][0], b[2 * pj][1], b[2 * pj + 1][0], b[2 * pj + 1][1],
                               bt + nr * LDB + kk + (mi & 1) * 16);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                uint32_t a0, a1, a2, a3;
                const int r = wm * 16 * MT + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ik_ldmatrix_x4(a0, a1, a2, a3, at + r * LDB + kk + (lane >> 4) * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float t[4] = {0.f, 0.f, 0.f, 0.f};
                    ik_mma_e4m3_16832(t, a0, a1, a2, a3, b[j][0], b[j][1]);
#pragma unroll
                    for (int v = 0; v < 4; ++v) acc[i][j][v] += t[v];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int row = m0 + wm * 16 * MT + i * 16 + (lane >> 2) + hh * 8;
                const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                if (col < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * hh] * s_out);
                if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * hh + 1] * s_out);
            }
}

template <int MT, int WM, typename OutT>
void launch_f8(const uint8_t* xq, const uint8_t* w, const float* sc, OutT* out, int M, int N, int K, cudaStream_t s) {
    constexpr int BM = 16 * MT * WM;
    constexpr size_t smem = (size_t)STAGES * (BM + BN) * LDB;
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(fp8_f8_gemm_kernel<MT, WM, OutT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        return true;
    }();
    (void)attr_set;
    const int nfast = M > N ? 1 : 0;  // the larger operand (activations M x K bytes vs weights N x K bytes) is the shared one
    const int64_t nblk = (int64_t)((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    fp8_f8_gemm_kernel<MT, WM, OutT><<<dim3((unsigned)nblk), 128 * WM, smem, s>>>(xq, w, sc, out, M, N, K, nfast);
}

// tile height: INTARSIA_F8_BM=64|128 overrides (A/B experiments); default 128 rows from M >= 1024
template <typename OutT>
int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const uint8_t* xq, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    cudaStream_t s = IK_STREAM(stream);
    int bm = M >= 1024 ? 128 : 64;
    if (const char* v = getenv("INTARSIA_F8_BM")) {
        const int b = atoi(v);
        if (b == 64 || b == 128) bm = b;
    }
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* scp = (const float*)sc->data;
    if (bm == 128) launch_f8<4, 2, OutT>(xq, wp, scp, (OutT*)out->data, (int)M, (int)N, (int)K, s);
    else launch_f8<2, 2, OutT>(xq, wp, scp, (OutT*)out->data, (int)M, (int)N, (int)K, s);
    IK_TRY(ik_launch_status());
    // weight-only activations (input_scale <= 0): the bf16 kernel, which exits when the scale is positive
    return ik_fp8_gemm_weight_only_internal(x->data, w->data, sc->data, out->data, sizeof(OutT) == 4, M, N, K, stream);
}

}  // namespace

extern "C" {

int ik_matmul_fp8_gemm_f8(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                          const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 128 != 0 || w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    // scratch: the quantised activations [M, K] bytes (the registry reserves M*K*2 like matmul_fp8_gemm)
    const int64_t xq_bytes = M * K;
    if (!scratch || !scratch->data || scratch->bytes < xq_bytes) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    uint8_t* xq = (uint8_t*)scratch->data;
    cudaStream_t s = IK_STREAM(stream);
    const int64_t n8 = M * K / 8;
    fp8_actq8_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((const ik_bf16*)x->data, xq, (const float*)scales->data, n8);
    IK_TRY(ik_launch_status());
    return out->dtype == IK_DTYPE_F32 ? run<float>(x, w, scales, out, xq, stream) : run<ik_bf16>(x, w, scales, out, xq, stream);
}

}  // extern "C"
