// matmul_fp8_gemm_lt: W8A8 prefill GEMM on cuBLASLt's native e4m3 path (OPT-IN, INTARSIA_CUDA_FP8_GEMM_LT=1).
//
// Same op and activation numerics as matmul_fp8_gemm / matmul_fp8_gemm_reg / matmul_fp8_gemm_f8: with a static input scale
// the activations are quantised to e4m3 with the same rounding (x / input_scale, RNE, saturating; the bytes are the
// e4m3 values the default path holds as bf16), every e4m3 x e4m3 product is exact, and y = weight_scale * input_scale * sum.
// This is vLLM's scheme for the checkpoint's fp8 layers (static per-tensor `input_scale`, `scaled_fp8_quant` +
// `cutlass_scaled_mm`). Only the accumulation differs from the default bf16-mma GEMM: cuBLASLt runs e4m3 tensor-core MMA
// and accumulates in its own order, so the results are not bitwise the default kernel's (docs/results/2026-10-21-cuda-round7.md).
//
// Determinism and row invariance: one cuBLASLt algorithm per (N, K, output dtype) is picked once (heuristic at M = 2048,
// no workspace, split-K forced to 1, no reduction scheme) and used for every M, so a row's result does not depend on how
// many rows share the call (tail chunks, prefix reuse). cuBLAS reads the scales from device memory (A/B scale pointers):
// no host synchronisation, capturable.
//
// W8A16 (input_scale <= 0, known only on device) cannot use e4m3 activations: as in matmul_fp8_gemm_f8 the launcher also
// runs the bf16 weight-only kernel, which exits at once when the scale is positive and overwrites the output otherwise.
#include "ik_launch.h"
#include "ik_internal.h"
#include "ik_quant.cuh"
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>

namespace {

// xq = e4m3(x / input_scale) as bytes; the same rounding as fp8_gemm.cu's bf16-valued quantiser (fp8_actq_kernel)
__global__ void __launch_bounds__(256)
lt_actq8_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ xq, const float* __restrict__ sc, int64_t n8) {
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

constexpr int kMaxDevices = 16;
constexpr int kRefM = 2048;  // the prefill chunk: the algorithm is chosen for it and used for every M

struct Plan {
    cublasLtMatmulAlgo_t algo;
    bool ok = false;
};

struct LtState {
    cublasLtHandle_t h = nullptr;
    std::mutex mu;
    std::map<std::tuple<int64_t, int64_t, int>, Plan> plans;  // (N, K, out_f32)
};
LtState g_lt[kMaxDevices];

bool lt_debug() {
    static const bool on = [] {
        const char* v = getenv("INTARSIA_LT_DEBUG");
        return v && v[0] == '1';
    }();
    return on;
}

bool fast_accum() {
    static const bool on = [] {
        const char* v = getenv("INTARSIA_LT_FAST_ACCUM");
        return v && v[0] == '1';
    }();
    return on;
}

// Descriptors on the stack (opaque storage: no host allocation per call). Column-major view of the row-major tensors:
// out[M,N] = x[M,K] w[N,K]^T  ==  D (N x M, ld N) = op(A) (N x K: A = w as K x N, transposed) * B (K x M = xq, ld K).
struct Desc {
    cublasLtMatmulDescOpaque_t op;
    cublasLtMatrixLayoutOpaque_t a, b, d;
};

int make_desc(Desc& ds, int64_t M, int64_t N, int64_t K, bool f32_out, const float* sc) {
    auto opd = reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op);
    auto la = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a);
    auto lb = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b);
    auto ld = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d);
    if (cublasLtMatmulDescInit(opd, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return IK_ERR_LAUNCH;
    const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    const int8_t fa = fast_accum() ? 1 : 0;
    const void* sa = sc;      // weight_scale (A = weights)
    const void* sb = sc + 1;  // input_scale (B = activations)
    if (cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof fa) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof sa) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof sb) != CUBLAS_STATUS_SUCCESS)
        return IK_ERR_LAUNCH;
    const cudaDataType_t dt = f32_out ? CUDA_R_32F : CUDA_R_16BF;
    if (cublasLtMatrixLayoutInit(la, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)N, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutInit(lb, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)M, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutInit(ld, dt, (uint64_t)N, (uint64_t)M, N) != CUBLAS_STATUS_SUCCESS)
        return IK_ERR_LAUNCH;
    return IK_OK;
}

// One algorithm per (N, K, out dtype): heuristic at M = kRefM without workspace, split-K pinned to 1. Caller holds the lock.
int get_plan(LtState& st, int64_t N, int64_t K, bool f32_out, const float* sc, Plan** out) {
    auto key = std::make_tuple(N, K, f32_out ? 1 : 0);
    auto it = st.plans.find(key);
    if (it != st.plans.end()) {
        *out = &it->second;
        return it->second.ok ? IK_OK : IK_ERR_UNSUPPORTED;
    }
    Plan& p = st.plans[key];
    *out = &p;
    Desc ds;
    IK_TRY(make_desc(ds, kRefM, N, K, f32_out, sc));
    auto opd = reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op);
    auto la = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a);
    auto lb = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b);
    auto ld = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d);
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return IK_ERR_LAUNCH;
    const size_t ws = 0;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    constexpr int kWant = 8;
    cublasLtMatmulHeuristicResult_t res[kWant] = {};
    int got = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(st.h, opd, la, lb, ld, ld, pref, kWant, res, &got);
    cublasLtMatmulPreferenceDestroy(pref);
    if (hs != CUBLAS_STATUS_SUCCESS || got == 0) {
        fprintf(stderr, "matmul_fp8_gemm_lt: no cuBLASLt e4m3 algorithm for N=%lld K=%lld (status %d)\n", (long long)N,
                (long long)K, (int)hs);
        return IK_ERR_UNSUPPORTED;
    }
    for (int i = 0; i < got; ++i) {
        if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize != 0) continue;
        cublasLtMatmulAlgo_t a = res[i].algo;
        int32_t splitk = 1;
        uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
        size_t w = 0;
        int32_t cur = 0;
        // pin split-K = 1 / no reduction when the algorithm has these knobs (attributes it lacks fail harmlessly)
        if (cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &cur, sizeof cur, &w) ==
                CUBLAS_STATUS_SUCCESS &&
            cur != 1) {
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk);
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red);
        }
        cublasLtMatmulHeuristicResult_t chk = {};
        if (cublasLtMatmulAlgoCheck(st.h, opd, la, lb, ld, ld, &a, &chk) != CUBLAS_STATUS_SUCCESS || chk.workspaceSize != 0)
            continue;
        p.algo = a;
        p.ok = true;
        if (lt_debug()) {
            int32_t id = -1, tile = -1, stages = -1;
            cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &w);
            fprintf(stderr, "matmul_fp8_gemm_lt: N=%lld K=%lld f32=%d -> heuristic #%d algo %d tile %d stages %d (of %d)\n",
                    (long long)N, (long long)K, (int)f32_out, i, id, tile, stages, got);
        }
        return IK_OK;
    }
    fprintf(stderr, "matmul_fp8_gemm_lt: no workspace-free split-K-1 algorithm for N=%lld K=%lld\n", (long long)N, (long long)K);
    return IK_ERR_UNSUPPORTED;
}

int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const uint8_t* xq, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    const bool f32_out = out->dtype == IK_DTYPE_F32;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) return IK_ERR_LAUNCH;
    LtState& st = g_lt[dev];
    const float* scp = (const float*)sc->data;
    {
        std::lock_guard<std::mutex> lock(st.mu);
        if (!st.h && cublasLtCreate(&st.h) != CUBLAS_STATUS_SUCCESS) {
            st.h = nullptr;
            return IK_ERR_LAUNCH;
        }
        Plan* p = nullptr;
        IK_TRY(get_plan(st, N, K, f32_out, scp, &p));
        Desc ds;
        IK_TRY(make_desc(ds, M, N, K, f32_out, scp));
        const float alpha = 1.f, beta = 0.f;
        const cublasStatus_t rs = cublasLtMatmul(
            st.h, reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op), &alpha, w->data,
            reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a), xq, reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b), &beta,
            out->data, reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d), out->data,
            reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d), &p->algo, nullptr, 0, IK_STREAM(stream));
        if (rs != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "matmul_fp8_gemm_lt: cublasLtMatmul M=%lld N=%lld K=%lld failed (%d)\n", (long long)M,
                    (long long)N, (long long)K, (int)rs);
            return IK_ERR_LAUNCH;
        }
    }
    // weight-only activations (input_scale <= 0): the bf16 kernel, which exits when the scale is positive
    return ik_fp8_gemm_weight_only_internal(x->data, w->data, sc->data, out->data, f32_out ? 1 : 0, M, N, K, stream);
}

}  // namespace

extern "C" {

int ik_matmul_fp8_gemm_lt(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                          const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check_a(out, out->dtype, 2, 16));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 128 != 0 || N % 16 != 0 || w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M ||
        out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    // scratch: the quantised activations [M, K] bytes (the registry reserves M*K*2 like matmul_fp8_gemm)
    const int64_t xq_bytes = M * K;
    if (!scratch || !scratch->data || scratch->bytes < xq_bytes) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    uint8_t* xq = (uint8_t*)scratch->data;
    const int64_t n8 = M * K / 8;
    lt_actq8_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, IK_STREAM(stream)>>>((const ik_bf16*)x->data, xq,
                                                                                 (const float*)scales->data, n8);
    IK_TRY(ik_launch_status());
    return run(x, w, scales, out, xq, stream);
}

}  // extern "C"
