// matmul_bf16_inv: out[M,N] = x[M,K] w[N,K]^T, bf16 in, f32 accumulate, bf16 out, with every row's bits independent of M
// (lane rip, row-invariant prefill: docs/results/2026-10-29-rip-row-invariant-prefill.md).
//
// matmul_bf16 calls cublasGemmEx with the default heuristic, which picks its algorithm (tile, split-K, kernel) from (M, N, K):
// the same row computed in a call of another row count can come out with other bits (a prefix-reuse resume, a 256-row busy
// chunk, a short tail). Here, as in fp8_gemm_lt.cu, ONE cuBLASLt algorithm per (N, K) is chosen once (heuristic at M = kRefM, the
// CUDA prefill chunk, INTARSIA_INV_REF_M; no workspace; split-K forced to 1, no reduction scheme) and run for every M, M = 1 included. With one
// algorithm and no split-K, each output element is one CTA's K loop in one fixed order whatever M is, so row m of an M-row call
// is bitwise row m of any other call over the same row (tests/row_invariance.rs).
//
// The prefill of a row-invariant pass selects it (OpCall::prefill_inv); ordinary calls keep matmul_bf16 / matmul_bf16_gemv.
// INTARSIA_LT_DEBUG=1 prints the chosen algorithm per shape.
#include "ik_launch.h"
#include "ik_internal.h"
#include "ik_quant.cuh"
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

namespace {

constexpr int kMaxDevices = 16;
// The row count the one algorithm per (N, K) is chosen for (heuristic at M = ref_m()): INTARSIA_INV_REF_M, default 4096 (the CUDA
// prefill chunk of the dense Qwen3 recipe). Read once per process, so every call of the process runs one algorithm per shape.
int ref_m() {
    static const int m = [] {
        const char* v = getenv("INTARSIA_INV_REF_M");
        const long x = v ? strtol(v, nullptr, 10) : 0;
        return (x >= 16 && x <= 65536) ? (int)x : 4096;
    }();
    return m;
}

struct Plan {
    cublasLtMatmulAlgo_t algo;
    bool ok = false;
};

struct LtState {
    cublasLtHandle_t h = nullptr;
    std::mutex mu;
    std::map<std::pair<int64_t, int64_t>, Plan> plans;  // (N, K)
};
LtState g_lt[kMaxDevices];

bool lt_debug() {
    static const bool on = [] {
        const char* v = getenv("INTARSIA_LT_DEBUG");
        return v && v[0] == '1';
    }();
    return on;
}

// Column-major view of the row-major tensors: out[M,N] = x[M,K] w[N,K]^T == D (N x M, ld N) = op(A) (A = w as K x N,
// transposed) * B (x as K x M, ld K).
struct Desc {
    cublasLtMatmulDescOpaque_t op;
    cublasLtMatrixLayoutOpaque_t a, b, d;
};

int make_desc(Desc& ds, int64_t M, int64_t N, int64_t K) {
    auto opd = reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op);
    auto la = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a);
    auto lb = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b);
    auto ld = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d);
    if (cublasLtMatmulDescInit(opd, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return IK_ERR_LAUNCH;
    const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    if (cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB) != CUBLAS_STATUS_SUCCESS)
        return IK_ERR_LAUNCH;
    if (cublasLtMatrixLayoutInit(la, CUDA_R_16BF, (uint64_t)K, (uint64_t)N, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutInit(lb, CUDA_R_16BF, (uint64_t)K, (uint64_t)M, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutInit(ld, CUDA_R_16BF, (uint64_t)N, (uint64_t)M, N) != CUBLAS_STATUS_SUCCESS)
        return IK_ERR_LAUNCH;
    return IK_OK;
}

// One algorithm per (N, K): heuristic at M = ref_m() without workspace, 16-byte aligned operands, split-K pinned to 1. The
// algorithm must also pass cublasLtMatmulAlgoCheck at M = 1 (the smallest call it will serve). Caller holds the lock.
int get_plan(LtState& st, int64_t N, int64_t K, Plan** out) {
    const auto key = std::make_pair(N, K);
    auto it = st.plans.find(key);
    if (it != st.plans.end()) {
        *out = &it->second;
        return it->second.ok ? IK_OK : IK_ERR_UNSUPPORTED;
    }
    Plan& p = st.plans[key];
    *out = &p;
    Desc ds, d1;
    IK_TRY(make_desc(ds, ref_m(), N, K));
    IK_TRY(make_desc(d1, 1, N, K));
    auto opd = reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op);
    auto la = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a);
    auto lb = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b);
    auto ld = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d);
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return IK_ERR_LAUNCH;
    const size_t ws = 0;
    const uint32_t align = 16;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align, sizeof align);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align, sizeof align);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align, sizeof align);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align, sizeof align);
    constexpr int kWant = 8;
    cublasLtMatmulHeuristicResult_t res[kWant] = {};
    int got = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(st.h, opd, la, lb, ld, ld, pref, kWant, res, &got);
    cublasLtMatmulPreferenceDestroy(pref);
    if (hs != CUBLAS_STATUS_SUCCESS || got == 0) {
        fprintf(stderr, "matmul_bf16_inv: no cuBLASLt bf16 algorithm for N=%lld K=%lld (status %d)\n", (long long)N, (long long)K,
                (int)hs);
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
        if (cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &cur, sizeof cur, &w) == CUBLAS_STATUS_SUCCESS &&
            cur != 1) {
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk);
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red);
        }
        cublasLtMatmulHeuristicResult_t chk = {};
        if (cublasLtMatmulAlgoCheck(st.h, opd, la, lb, ld, ld, &a, &chk) != CUBLAS_STATUS_SUCCESS || chk.workspaceSize != 0)
            continue;
        cublasLtMatmulHeuristicResult_t chk1 = {};
        auto o1 = reinterpret_cast<cublasLtMatmulDesc_t>(&d1.op);
        auto lb1 = reinterpret_cast<cublasLtMatrixLayout_t>(&d1.b);
        auto ld1 = reinterpret_cast<cublasLtMatrixLayout_t>(&d1.d);
        if (cublasLtMatmulAlgoCheck(st.h, o1, la, lb1, ld1, ld1, &a, &chk1) != CUBLAS_STATUS_SUCCESS || chk1.workspaceSize != 0)
            continue;
        p.algo = a;
        p.ok = true;
        if (lt_debug()) {
            int32_t id = -1, tile = -1, stages = -1;
            cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &w);
            fprintf(stderr, "matmul_bf16_inv: N=%lld K=%lld ref M %d -> heuristic #%d algo %d tile %d stages %d (of %d)\n",
                    (long long)N, (long long)K, ref_m(), i, id, tile, stages, got);
        }
        return IK_OK;
    }
    fprintf(stderr, "matmul_bf16_inv: no workspace-free split-K-1 algorithm for N=%lld K=%lld\n", (long long)N, (long long)K);
    return IK_ERR_UNSUPPORTED;
}

}  // namespace

extern "C" int ik_matmul_bf16_inv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(out, IK_DTYPE_BF16, 2, 16));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (w->shape[1] != K || out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    if (K % 8 != 0 || N % 8 != 0) return IK_ERR_SHAPE;  // 16-byte rows (the algorithm is chosen for 16-byte alignment)
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0) return cudaMemsetAsync(out->data, 0, (size_t)(M * N * 2), IK_STREAM(stream)) == cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) return IK_ERR_LAUNCH;
    LtState& st = g_lt[dev];
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.h && cublasLtCreate(&st.h) != CUBLAS_STATUS_SUCCESS) {
        st.h = nullptr;
        return IK_ERR_LAUNCH;
    }
    Plan* p = nullptr;
    IK_TRY(get_plan(st, N, K, &p));
    Desc ds;
    IK_TRY(make_desc(ds, M, N, K));
    const float alpha = 1.f, beta = 0.f;
    const cublasStatus_t rs = cublasLtMatmul(
        st.h, reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op), &alpha, w->data, reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a),
        x->data, reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b), &beta, out->data,
        reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d), out->data, reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d), &p->algo,
        nullptr, 0, IK_STREAM(stream));
    if (rs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "matmul_bf16_inv: cublasLtMatmul M=%lld N=%lld K=%lld failed (%d)\n", (long long)M, (long long)N,
                (long long)K, (int)rs);
        return IK_ERR_LAUNCH;
    }
    return IK_OK;
}
