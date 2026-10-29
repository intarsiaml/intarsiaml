// Lane rip2 probe: which cuBLASLt bf16 algorithms are bitwise equal to the pinned matmul_bf16_inv algorithm, and how fast is each
// at the Qwen3-8B projection shapes. Standalone (links cublasLt / cublas only):
//   nvcc -O2 -std=c++17 -o lt_probe lt_probe.cu -lcublasLt -lcublas
//   ./lt_probe [heur|enum|all] [M list for timing, default 991,1024,2048,4096]
// heur: cuBLAS's own heuristic picks (32 MB workspace) per shape and M, with their split-K / reduction, plus cublasGemmEx time.
// enum: every algorithm id x tile x stages x swizzle x custom option x split-K {1,2,4}, checked at M = 4352 and M = 1; its M = 4352
// output compared bitwise to the reference (the heuristic pick at M = 4096 without workspace, split-K pinned to 1: matmul_lt.cu's
// get_plan), then timed. Lines "CAND" are machine-readable.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } \
    } while (0)

static cublasLtHandle_t g_lt;
static cublasHandle_t g_cb;
static void* g_ws;
constexpr size_t kWs = 32u << 20;
constexpr int kMaxM = 4352;

struct Desc {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    ~Desc() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (d) cublasLtMatrixLayoutDestroy(d);
    }
};

static void make_desc(Desc& ds, int64_t M, int64_t N, int64_t K) {
    cublasLtMatmulDescCreate(&ds.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(ds.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA);
    cublasLtMatmulDescSetAttribute(ds.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB);
    cublasLtMatrixLayoutCreate(&ds.a, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&ds.b, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&ds.d, CUDA_R_16BF, N, M, N);
}

template <class T>
static T cfg(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t at) {
    T v{};
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&a, at, &v, sizeof v, &w);
    return v;
}

static std::string algo_str(const cublasLtMatmulAlgo_t& a) {
    char buf[256];
    snprintf(buf, sizeof buf, "id %d tile %d stages %d splitk %d red %u swz %u cust %u inner %u cluster %u",
             cfg<int32_t>(a, CUBLASLT_ALGO_CONFIG_ID), cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_TILE_ID),
             cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_STAGES_ID), cfg<int32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
             cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME), cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
             cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION), cfg<uint16_t>(a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
             cfg<uint16_t>(a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID));
    return buf;
}

static bool run(const cublasLtMatmulAlgo_t& a, int64_t M, int64_t N, int64_t K, const void* x, const void* w, void* out) {
    Desc ds;
    make_desc(ds, M, N, K);
    const float alpha = 1.f, beta = 0.f;
    return cublasLtMatmul(g_lt, ds.op, &alpha, w, ds.a, x, ds.b, &beta, out, ds.d, out, ds.d, &a, g_ws, kWs, 0) ==
           CUBLAS_STATUS_SUCCESS;
}

static bool check(const cublasLtMatmulAlgo_t& a, int64_t M, int64_t N, int64_t K, size_t* ws) {
    Desc ds;
    make_desc(ds, M, N, K);
    cublasLtMatmulHeuristicResult_t r = {};
    if (cublasLtMatmulAlgoCheck(g_lt, ds.op, ds.a, ds.b, ds.d, ds.d, &a, &r) != CUBLAS_STATUS_SUCCESS) return false;
    if (ws) *ws = r.workspaceSize;
    return r.workspaceSize <= kWs;
}

// median microseconds over `it` launches (after 2 warm-ups); <0 on failure
template <class F>
static float timeit(F f, int it = 9) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 2; ++i)
        if (!f()) return -1;
    std::vector<float> v;
    for (int i = 0; i < it; ++i) {
        cudaEventRecord(e0);
        if (!f()) return -1;
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        v.push_back(ms * 1000.f);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static float time_gemmex(int64_t M, int64_t N, int64_t K, const void* x, const void* w, void* out) {
    const float alpha = 1.f, beta = 0.f;
    return timeit([&] {
        return cublasGemmEx(g_cb, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, w, CUDA_R_16BF, (int)K, x, CUDA_R_16BF,
                            (int)K, &beta, out, CUDA_R_16BF, (int)N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) ==
               CUBLAS_STATUS_SUCCESS;
    });
}

static int heuristic(int64_t M, int64_t N, int64_t K, size_t ws, cublasLtMatmulHeuristicResult_t* res, int want) {
    Desc ds;
    make_desc(ds, M, N, K);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    const uint32_t align = 16;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align, sizeof align);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align, sizeof align);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align, sizeof align);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align, sizeof align);
    int got = 0;
    cublasLtMatmulAlgoGetHeuristic(g_lt, ds.op, ds.a, ds.b, ds.d, ds.d, pref, want, res, &got);
    cublasLtMatmulPreferenceDestroy(pref);
    return got;
}

// matmul_lt.cu get_plan: first heuristic result at M = 4096 without workspace that takes split-K 1 and checks at M = 1
static bool ref_algo(int64_t N, int64_t K, cublasLtMatmulAlgo_t* out) {
    cublasLtMatmulHeuristicResult_t res[8] = {};
    const int got = heuristic(4096, N, K, 0, res, 8);
    for (int i = 0; i < got; ++i) {
        if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize != 0) continue;
        cublasLtMatmulAlgo_t a = res[i].algo;
        int32_t sk = 1, cur = cfg<int32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
        if (cur != 1) {
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk);
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red);
        }
        size_t w = 1;
        if (!check(a, 4096, N, K, &w) || w != 0 || !check(a, 1, N, K, &w) || w != 0) continue;
        *out = a;
        return true;
    }
    return false;
}

// wide-magnitude random bf16: normal * 2^U(-6, 6), so a changed summation order shows in the low bits
static std::vector<__nv_bfloat16> rnd(size_t n, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-6, 6);
    std::vector<__nv_bfloat16> v(n);
    for (auto& e : v) e = __float2bfloat16(ldexpf(nd(g), ex(g)));
    return v;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    std::vector<int> Ms = {991, 1024, 2048, 4096};
    if (argc > 2) {
        Ms.clear();
        for (char* s = strtok(argv[2], ","); s; s = strtok(nullptr, ",")) Ms.push_back(atoi(s));
    }
    cublasLtCreate(&g_lt);
    cublasCreate(&g_cb);
    CK(cudaMalloc(&g_ws, kWs));
    cublasSetWorkspace(g_cb, g_ws, kWs);
    cublasSetMathMode(g_cb, (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    int ver = (int)cublasLtGetVersion();
    printf("cublasLt version %d\n", ver);
    const std::pair<int64_t, int64_t> shapes[] = {{4096, 4096}, {1024, 4096}, {12288, 4096}, {4096, 12288}};
    for (auto [N, K] : shapes) {
        const auto hx = rnd((size_t)kMaxM * K, 1 + (uint32_t)N), hw = rnd((size_t)N * K, 7 + (uint32_t)K);
        void *x, *w, *o, *r;
        CK(cudaMalloc(&x, hx.size() * 2));
        CK(cudaMalloc(&w, hw.size() * 2));
        CK(cudaMalloc(&o, (size_t)kMaxM * N * 2));
        CK(cudaMalloc(&r, (size_t)kMaxM * N * 2));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        printf("== shape N %lld K %lld\n", (long long)N, (long long)K);
        if (mode == "heur" || mode == "all") {
            for (int M : Ms) {
                printf("GEMMEX N %lld K %lld M %d us %.1f\n", (long long)N, (long long)K, M, time_gemmex(M, N, K, x, w, o));
                cublasLtMatmulHeuristicResult_t res[8] = {};
                const int got = heuristic(M, N, K, kWs, res, 8);
                for (int i = 0; i < got && i < 4; ++i)
                    printf("HEUR M %d #%d %s ws %zu us %.1f\n", M, i, algo_str(res[i].algo).c_str(), res[i].workspaceSize,
                           timeit([&] { return run(res[i].algo, M, N, K, x, w, o); }));
            }
        }
        cublasLtMatmulAlgo_t ref;
        if (!ref_algo(N, K, &ref)) {
            printf("no reference algorithm\n");
            continue;
        }
        printf("REF %s\n", algo_str(ref).c_str());
        run(ref, kMaxM, N, K, x, w, r);
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> href((size_t)kMaxM * N), hout((size_t)kMaxM * N);
        CK(cudaMemcpy(href.data(), r, href.size() * 2, cudaMemcpyDeviceToHost));
        if (mode == "pool") {
            std::vector<cublasLtMatmulAlgo_t> pool;
            auto same = [](const cublasLtMatmulAlgo_t& p, const cublasLtMatmulAlgo_t& q) { return algo_str(p) == algo_str(q); };
            for (int M : Ms) {
                cublasLtMatmulHeuristicResult_t res[16] = {};
                const int got = heuristic(M, N, K, kWs, res, 16);
                for (int i = 0; i < got; ++i) {
                    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
                    bool dup = false;
                    for (auto& q : pool) dup = dup || same(q, res[i].algo);
                    if (!dup) pool.push_back(res[i].algo);
                }
            }
            for (auto& a : pool) {
                size_t ws1 = 0, ws2 = 0;
                const bool ok1 = check(a, kMaxM, N, K, &ws1), ok2 = check(a, 1, N, K, &ws2);
                int cls = -1;
                if (ok1 && run(a, kMaxM, N, K, x, w, o) && cudaDeviceSynchronize() == cudaSuccess) {
                    CK(cudaMemcpy(hout.data(), o, hout.size() * 2, cudaMemcpyDeviceToHost));
                    cls = hout == href ? 0 : 1;
                }
                printf("POOL N %lld K %lld %s ws %zu/%zu m1 %d class %d us", (long long)N, (long long)K, algo_str(a).c_str(), ws1, ws2,
                       (int)ok2, cls);
                for (int M : Ms) printf(" %d:%.1f", M, check(a, M, N, K, nullptr) ? timeit([&] { return run(a, M, N, K, x, w, o); }, 7) : -1.f);
                printf("\n");
                fflush(stdout);
            }
            printf("GEMMEX N %lld K %lld us", (long long)N, (long long)K);
            for (int M : Ms) printf(" %d:%.1f", M, time_gemmex(M, N, K, x, w, o));
            printf("\n");
            continue;
        }
        if (mode != "enum" && mode != "all") continue;
        int ids[256], nid = 0;
        cublasLtMatmulAlgoGetIds(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 256, ids, &nid);
        std::map<uint64_t, int> classes;  // output hash -> class number (0 = the reference)
        for (int ii = 0; ii < nid; ++ii) {
            cublasLtMatmulAlgo_t a0;
            if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[ii],
                                       &a0) != CUBLAS_STATUS_SUCCESS)
                continue;
            size_t sz = 0;
            std::vector<uint32_t> tiles(1, 0), stages(1, 0);
            cublasLtMatmulAlgoCapGetAttribute(&a0, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz);
            if (sz) {
                tiles.resize(sz / 4);
                cublasLtMatmulAlgoCapGetAttribute(&a0, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
            }
            sz = 0;
            cublasLtMatmulAlgoCapGetAttribute(&a0, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz);
            if (sz) {
                stages.resize(sz / 4);
                cublasLtMatmulAlgoCapGetAttribute(&a0, CUBLASLT_ALGO_CAP_STAGES_IDS, stages.data(), sz, &sz);
            }
            int32_t swz_ok = 0, split_ok = 0, cust_max = 0;
            cublasLtMatmulAlgoCapGetAttribute(&a0, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swz_ok, sizeof swz_ok, &sz);
            cublasLtMatmulAlgoCapGetAttribute(&a0, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &split_ok, sizeof split_ok, &sz);
            cublasLtMatmulAlgoCapGetAttribute(&a0, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &cust_max, sizeof cust_max, &sz);
            printf("ALGO id %d tiles %zu stages %zu swz %d splitk %d custmax %d\n", ids[ii], tiles.size(), stages.size(), swz_ok,
                   split_ok, cust_max);
            const bool full = getenv("FULL") && getenv("FULL")[0] == '1';  // also swizzle, custom options, split-K 2 / 4
            for (uint32_t t : tiles)
                for (uint32_t s : stages)
                    for (uint32_t swz = 0; swz <= (swz_ok && full ? 1u : 0u); ++swz)
                        for (uint32_t cu = 0; cu <= (uint32_t)(full ? std::min(cust_max, 3) : 0); ++cu)
                            for (int32_t sk : {1, 2, 4}) {
                                if (sk > 1 && (!split_ok || !full)) continue;
                                cublasLtMatmulAlgo_t a = a0;
                                const uint32_t red = sk == 1 ? CUBLASLT_REDUCTION_SCHEME_NONE : CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, sizeof t);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &s, sizeof s);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cu, sizeof cu);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red);
                                size_t ws1 = 0, ws2 = 0;
                                if (!check(a, kMaxM, N, K, &ws1) || !check(a, 1, N, K, &ws2)) continue;
                                CK(cudaMemset(o, 0xff, (size_t)kMaxM * N * 2));
                                if (!run(a, kMaxM, N, K, x, w, o)) continue;
                                if (cudaDeviceSynchronize() != cudaSuccess) {
                                    printf("CAND launch error %s\n", algo_str(a).c_str());
                                    exit(2);
                                }
                                CK(cudaMemcpy(hout.data(), o, hout.size() * 2, cudaMemcpyDeviceToHost));
                                int cls;
                                if (hout == href) cls = 0;
                                else {
                                    uint64_t hsh = 1469598103934665603ull;
                                    for (uint16_t v : hout) hsh = (hsh ^ v) * 1099511628211ull;
                                    auto it = classes.find(hsh);
                                    cls = it != classes.end() ? it->second : (classes[hsh] = (int)classes.size() + 1);
                                }
                                printf("CAND N %lld K %lld %s ws %zu class %d us", (long long)N, (long long)K, algo_str(a).c_str(),
                                       std::max(ws1, ws2), cls);
                                for (int M : Ms) printf(" %d:%.1f", M, timeit([&] { return run(a, M, N, K, x, w, o); }, 5));
                                printf("\n");
                                fflush(stdout);
                            }
        }
        cudaFree(x);
        cudaFree(w);
        cudaFree(o);
        cudaFree(r);
    }
    return 0;
}
