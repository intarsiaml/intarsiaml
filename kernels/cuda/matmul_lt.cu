// matmul_bf16_inv: out[M,N] = x[M,K] w[N,K]^T, bf16 in, f32 accumulate, bf16 out, with every row's bits independent of M
// (lane rip, row-invariant prefill: docs/results/2026-10-29-rip-row-invariant-prefill.md; lane rip2, per-M algorithms:
// docs/results/2026-10-29-rip2-row-invariant-gemm.md).
//
// matmul_bf16 calls cublasGemmEx with the default heuristic, which picks its algorithm (tile, split-K, kernel) from (M, N, K):
// the same row computed in a call of another row count can come out with other bits (a prefix-reuse resume, a 256-row busy
// chunk, a short tail). Here, as in fp8_gemm_lt.cu, ONE cuBLASLt algorithm per (N, K), the primary, defines the bits: chosen once
// (heuristic at M = kRefM, the CUDA prefill chunk, INTARSIA_INV_REF_M; no workspace; split-K forced to 1, no reduction scheme).
// With one algorithm and no split-K, each output element is one CTA's K loop in one fixed order whatever M is, so row m of an
// M-row call is bitwise row m of any other call over the same row (tests/row_invariance.rs).
//
// The primary is slow at some row counts (~1k rows: 1.24x cuBLAS, its tile wastes a wave). Each call of fewer than ref_m() / 2
// rows therefore runs the algorithm cuBLASLt's heuristic picks for M (32 MB workspace preference, as cublasGemmEx) when that
// pick has split-K 1, no workspace, and is PROVEN bitwise equal to the primary: once per (N, K, algorithm), on a fixed random
// probe (kProbeRows rows of wide-magnitude bf16 against the call's own weights). Most split-K-1 cuBLASLt kernels on sm_121
// (mma.sync, one sequential K loop per output) are equal to each other whatever their tile; the others (sliced-K, a different
// MMA shape) differ in almost every output and fail the probe, and M then runs the primary. A row's bits are thus the primary's
// at every M. A call inside a stream capture, which cannot run the probe, runs the primary unless its M is already decided.
// INTARSIA_INV_SINGLE=1 runs the primary at every M (lane rip's behaviour).
//
// The prefill of a row-invariant pass selects it (OpCall::prefill_inv); ordinary calls keep matmul_bf16 / matmul_bf16_gemv.
// INTARSIA_LT_DEBUG=1 prints the primary per shape and each probed algorithm.
#include "ik_launch.h"
#include "ik_internal.h"
#include "ik_quant.cuh"
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <string.h>
#include <tuple>
#include <vector>
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

bool single() {
    static const bool on = [] {
        const char* v = getenv("INTARSIA_INV_SINGLE");
        return v && v[0] == '1';
    }();
    return on;
}

constexpr int kProbeRows = 256;
constexpr size_t kHeurWorkspace = 32u << 20;  // matmul.cu kWorkspaceBytes: the heuristic is asked as cublasGemmEx's would be

// An algorithm's configuration (id, tile, stages, split-K, reduction, swizzle, custom option, inner / cluster shape).
using AlgoKey = std::tuple<int32_t, uint32_t, uint32_t, int32_t, uint32_t, uint32_t, uint32_t, uint16_t, uint16_t>;

template <class T>
T algo_cfg(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t at, T dflt) {
    T v = dflt;
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&a, at, &v, sizeof v, &w);
    return v;
}

AlgoKey algo_key(const cublasLtMatmulAlgo_t& a) {
    return AlgoKey(algo_cfg<int32_t>(a, CUBLASLT_ALGO_CONFIG_ID, -1), algo_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_TILE_ID, 0),
                   algo_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_STAGES_ID, 0), algo_cfg<int32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1),
                   algo_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0),
                   algo_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0), algo_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0),
                   algo_cfg<uint16_t>(a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 0),
                   algo_cfg<uint16_t>(a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 0));
}

struct Plan {
    cublasLtMatmulAlgo_t algo;  // the primary
    bool ok = false;
    std::vector<cublasLtMatmulAlgo_t> equal;  // algorithms proven bitwise equal to the primary (the primary first)
    std::map<AlgoKey, int> probed;            // algorithm -> index in `equal`, or -1 (differs / unusable)
    std::map<int64_t, int> by_m;              // M -> index in `equal`
};

struct LtState {
    cublasLtHandle_t h = nullptr;
    std::mutex mu;
    std::map<std::pair<int64_t, int64_t>, Plan> plans;  // (N, K)
    std::map<int64_t, void*> probe_x;                    // K -> kProbeRows x K random bf16
    void* probe_out = nullptr;                           // 2 x kProbeRows x probe_n bf16
    int64_t probe_n = 0;
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
        p.equal.push_back(a);
        p.probed[algo_key(a)] = 0;
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

int lt_run(LtState& st, const cublasLtMatmulAlgo_t& a, int64_t M, int64_t N, int64_t K, const void* x, const void* w, void* out,
           cudaStream_t s) {
    Desc ds;
    IK_TRY(make_desc(ds, M, N, K));
    const float alpha = 1.f, beta = 0.f;
    const cublasStatus_t rs = cublasLtMatmul(st.h, reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op), &alpha, w,
                                             reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a), x,
                                             reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b), &beta, out,
                                             reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d), out,
                                             reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d), &a, nullptr, 0, s);
    if (rs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "matmul_bf16_inv: cublasLtMatmul M=%lld N=%lld K=%lld failed (%d)\n", (long long)M, (long long)N, (long long)K,
                (int)rs);
        return IK_ERR_LAUNCH;
    }
    return IK_OK;
}

// Is `a` bitwise the primary? Both run over the probe rows (fixed-seed wide-magnitude bf16: normal * 2^U(-6, 6), so another
// summation order shows in nearly every output) against the call's weights; every output compared. Synchronizes `s`.
bool probe_equal(LtState& st, Plan& p, const cublasLtMatmulAlgo_t& a, int64_t N, int64_t K, const void* w, cudaStream_t s) {
    void*& px = st.probe_x[K];
    if (!px) {
        std::vector<uint16_t> h((size_t)kProbeRows * K);
        uint64_t z = 0x9e3779b97f4a7c15ull ^ (uint64_t)K;
        auto next = [&] {  // splitmix64
            z += 0x9e3779b97f4a7c15ull;
            uint64_t r = z;
            r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ull;
            r = (r ^ (r >> 27)) * 0x94d049bb133111ebull;
            return r ^ (r >> 31);
        };
        for (auto& e : h) {
            const uint64_t r = next();
            // sign, exponent 2^(-6..6) around 1, 7 random mantissa bits: bf16 bit pattern
            const uint32_t sign = (uint32_t)(r & 1), ex = 127 - 6 + (uint32_t)((r >> 1) % 13), man = (uint32_t)((r >> 8) & 0x7f);
            e = (uint16_t)((sign << 15) | (ex << 7) | man);
        }
        if (cudaMalloc(&px, h.size() * 2) != cudaSuccess) {
            px = nullptr;
            return false;
        }
        if (cudaMemcpy(px, h.data(), h.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    }
    if (st.probe_n < N) {
        if (st.probe_out) cudaFree(st.probe_out);
        st.probe_out = nullptr;
        st.probe_n = 0;
        if (cudaMalloc(&st.probe_out, (size_t)2 * kProbeRows * N * 2) != cudaSuccess) {
            st.probe_out = nullptr;
            return false;
        }
        st.probe_n = N;
    }
    const size_t n = (size_t)kProbeRows * N;
    uint16_t* o0 = (uint16_t*)st.probe_out;
    uint16_t* o1 = o0 + n;
    if (lt_run(st, p.algo, kProbeRows, N, K, px, w, o0, s) != IK_OK || lt_run(st, a, kProbeRows, N, K, px, w, o1, s) != IK_OK) return false;
    std::vector<uint16_t> h(2 * n);
    if (cudaMemcpyAsync(h.data(), o0, 2 * n * 2, cudaMemcpyDeviceToHost, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess)
        return false;
    return memcmp(h.data(), h.data() + n, n * 2) == 0;
}

// The algorithm an M-row call runs (index in p.equal): the heuristic's first pick for M that is split-K 1, workspace-free,
// checks at M and is proven equal to the primary; else the primary. Caller holds the lock.
int pick(LtState& st, Plan& p, int64_t M, int64_t N, int64_t K, const void* w, cudaStream_t s) {
    // From ref_m() / 2 rows up the primary itself runs: it is the heuristic's workspace-free choice at ref_m(), and there it beat
    // the per-M picks (8B layer at 2048 / 3000 / 4096 rows: 0.956 / 0.973 / 0.959 of cuBLAS vs 1.011 / 0.993 / 0.993; with the
    // picks, 4k-token prefill was -3.5%, docs/results/2026-10-29-rip2-row-invariant-gemm.md).
    if (single() || M >= ref_m() / 2) return 0;
    auto it = p.by_m.find(M);
    if (it != p.by_m.end()) return it->second;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) return 0;  // cannot probe; not cached
    int chosen = 0;
    Desc ds;
    if (make_desc(ds, M, N, K) == IK_OK) {
        auto opd = reinterpret_cast<cublasLtMatmulDesc_t>(&ds.op);
        auto la = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.a);
        auto lb = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.b);
        auto ld = reinterpret_cast<cublasLtMatrixLayout_t>(&ds.d);
        cublasLtMatmulPreference_t pref = nullptr;
        constexpr int kWant = 8;
        cublasLtMatmulHeuristicResult_t res[kWant] = {};
        int got = 0;
        if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
            const size_t ws = kHeurWorkspace;
            const uint32_t align = 16;
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align, sizeof align);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align, sizeof align);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align, sizeof align);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align, sizeof align);
            if (cublasLtMatmulAlgoGetHeuristic(st.h, opd, la, lb, ld, ld, pref, kWant, res, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
            cublasLtMatmulPreferenceDestroy(pref);
        }
        for (int i = 0; i < got; ++i) {
            const cublasLtMatmulAlgo_t& a = res[i].algo;
            if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize != 0) continue;
            if (algo_cfg<int32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1) != 1) continue;
            const AlgoKey key = algo_key(a);
            auto pr = p.probed.find(key);
            int idx;
            if (pr != p.probed.end()) {
                idx = pr->second;
            } else {
                // the probe's row count must also be one the algorithm serves
                Desc dp;
                cublasLtMatmulHeuristicResult_t chk = {};
                const bool runs = make_desc(dp, kProbeRows, N, K) == IK_OK &&
                                  cublasLtMatmulAlgoCheck(st.h, reinterpret_cast<cublasLtMatmulDesc_t>(&dp.op),
                                                          reinterpret_cast<cublasLtMatrixLayout_t>(&dp.a),
                                                          reinterpret_cast<cublasLtMatrixLayout_t>(&dp.b),
                                                          reinterpret_cast<cublasLtMatrixLayout_t>(&dp.d),
                                                          reinterpret_cast<cublasLtMatrixLayout_t>(&dp.d), &a, &chk) == CUBLAS_STATUS_SUCCESS &&
                                  chk.workspaceSize == 0;
                const bool eq = runs && probe_equal(st, p, a, N, K, w, s);
                idx = eq ? (int)p.equal.size() : -1;
                if (eq) p.equal.push_back(a);
                p.probed[key] = idx;
                if (lt_debug())
                    fprintf(stderr, "matmul_bf16_inv: N=%lld K=%lld probe algo %d tile %u stages %u cust %u (first at M %lld): %s\n",
                            (long long)N, (long long)K, std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<6>(key),
                            (long long)M, eq ? "equal to the primary" : (runs ? "DIFFERS" : "unusable"));
            }
            if (idx < 0) continue;
            cublasLtMatmulHeuristicResult_t chk = {};
            if (cublasLtMatmulAlgoCheck(st.h, opd, la, lb, ld, ld, &p.equal[idx], &chk) != CUBLAS_STATUS_SUCCESS || chk.workspaceSize != 0)
                continue;
            chosen = idx;
            break;
        }
    }
    p.by_m[M] = chosen;
    return chosen;
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
    cudaStream_t s = IK_STREAM(stream);
    const int idx = pick(st, *p, M, N, K, w->data, s);
    return lt_run(st, p->equal[idx], M, N, K, x->data, w->data, out->data, s);
}
