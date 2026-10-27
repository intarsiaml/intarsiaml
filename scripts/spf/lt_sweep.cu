// Lane spf: time cublasGemmEx (the engine's call: bf16 in/out, f32 compute, DISALLOW_REDUCED_PRECISION_REDUCTION) against every
// cublasLt heuristic algorithm for the Qwen3-8B prefill GEMM shapes. out[M,N] = x[M,K] w[N,K]^T.
//   nvcc -O2 -o lt_sweep lt_sweep.cu -lcublas -lcublasLt && ./lt_sweep [M]
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda err %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
#define CB(x) do { int e = (int)(x); if (e != 0) { printf("cublas err %d line %d\n", e, __LINE__); exit(1);} } while (0)


int main(int argc, char** argv) {
    std::vector<int> Ms; for (int i = 1; i < argc; ++i) Ms.push_back(atoi(argv[i])); if (Ms.empty()) Ms.push_back(4096);
    const bool quiet = getenv("QUIET") != nullptr;
    struct Shape { const char* name; int N, K; } shapes[] = {
        {"q/o  N=4096 K=4096", 4096, 4096}, {"k/v  N=1024 K=4096", 1024, 4096},
        {"gate/up N=12288 K=4096", 12288, 4096}, {"down N=4096 K=12288", 4096, 12288}};
    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
    cudaStream_t s; CK(cudaStreamCreate(&s)); CB(cublasSetStream(h, s));
    const size_t wsz = 32u << 20; void* ws; CK(cudaMalloc(&ws, wsz));
    CB(cublasSetWorkspace(h, ws, wsz));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int M : Ms) for (auto& sh : shapes) {
        const int N = sh.N, K = sh.K;
        __nv_bfloat16 *x, *w, *o, *o2;
        const size_t offw = getenv("OFFW") ? atoi(getenv("OFFW")) : 0, offx = getenv("OFFX") ? atoi(getenv("OFFX")) : 0, offo = getenv("OFFO") ? atoi(getenv("OFFO")) : 0;
        char *bx, *bw, *bo, *bo2;
        CK(cudaMalloc(&bx, (size_t)M * K * 2 + 4096)); CK(cudaMalloc(&bw, (size_t)N * K * 2 + 4096));
        CK(cudaMalloc(&bo, (size_t)M * N * 2 + 4096)); CK(cudaMalloc(&bo2, (size_t)M * N * 2 + 4096));
        x = (__nv_bfloat16*)(bx + offx); w = (__nv_bfloat16*)(bw + offw); o = (__nv_bfloat16*)(bo + offo); o2 = (__nv_bfloat16*)(bo2 + offo);
        std::vector<__nv_bfloat16> hx((size_t)M * K), hw((size_t)N * K);
        srand(1);
        const bool gauss = getenv("GAUSS") != nullptr;
        auto g = [&]() { float u1 = (rand() + 1.f) / (RAND_MAX + 2.f), u2 = rand() / (float)RAND_MAX; return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2); };
        for (auto& v : hx) v = gauss ? __float2bfloat16(g()) : __float2bfloat16((rand() / (float)RAND_MAX - 0.5f));
        for (auto& v : hw) v = gauss ? __float2bfloat16(g() * 0.02f) : __float2bfloat16((rand() / (float)RAND_MAX - 0.5f) * 0.05f);
        CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        const float alpha = 1.f, beta = 0.f;
        auto run_ex = [&]() {
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &beta, o,
                            CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        auto time = [&](auto&& f, int iters) {
            for (int i = 0; i < 3; ++i) f();
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < iters; ++i) f();
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / iters;
        };
        const double fl = 2.0 * M * N * K;
        float t_ex = time(run_ex, 20);
        printf("%s M=%d: cublasGemmEx %.3f ms = %.1f TFLOP/s\n", sh.name, M, t_ex, fl / t_ex / 1e9);
        {   // cold weights: cycle over NW distinct copies (each call streams a weight not read for NW-1 calls)
            const int NW = 24; std::vector<__nv_bfloat16*> wc(NW);
            for (int i = 0; i < NW; ++i) { CK(cudaMalloc(&wc[i], (size_t)N * K * 2)); CK(cudaMemcpy(wc[i], w, (size_t)N * K * 2, cudaMemcpyDeviceToDevice)); }
            int ci = 0;
            auto f = [&]() { CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, wc[ci++ % NW], CUDA_R_16BF, K, x, CUDA_R_16BF, K, &beta, o, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); };
            float t_cold = time(f, 48);
            if (!quiet) printf("   cold weights (%d copies): %.3f ms = %.1f TFLOP/s\n", NW, t_cold, fl / t_cold / 1e9);
            for (auto p : wc) cudaFree(p);
        }
        // Lt
        cublasLtMatmulDesc_t d; CB(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
        CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
        cublasLtMatrixLayout_t la, lb, lc;
        CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
        CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K));
        CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
        cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
        uint32_t mask = CUBLASLT_REDUCTION_SCHEME_INPLACE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_NONE;
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof mask));
        cublasLtMatmulHeuristicResult_t res[64]; int n = 0;
        CB(cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, pref, 64, res, &n));
        float best = 1e9; int bi = -1;
        for (int i = 0; i < n; ++i) {
            auto f = [&]() {
                cublasStatus_t st = cublasLtMatmul(lt, d, &alpha, w, la, x, lb, &beta, o2, lc, o2, lc, &res[i].algo, ws, wsz, s);
                if (st != 0) { printf("lt err %d\n", (int)st); exit(1); }
            };
            float t = time(f, 20);
            int id = -1, tile = -1, stages = -1, splitk = -1, red = -1;
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, nullptr);
            if (!quiet) printf("   lt[%2d] id=%d tile=%d stages=%d splitk=%d red=%d ws=%zu: %.3f ms = %.1f TFLOP/s\n", i, id, tile, stages,
                   splitk, red, res[i].workspaceSize, t, fl / t / 1e9);
            if (t < best) { best = t; bi = i; }
        }
        printf("   best lt[%d] %.3f ms = %.1f TFLOP/s (%+.1f%% vs GemmEx)\n", bi, best, fl / best / 1e9, 100.0 * (t_ex / best - 1));
        // are the outputs equal bitwise?
        std::vector<__nv_bfloat16> a((size_t)M * N), b((size_t)M * N);
        run_ex(); CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(a.data(), o, a.size() * 2, cudaMemcpyDeviceToHost));
        if (bi >= 0) {
            CB(cublasLtMatmul(lt, d, &alpha, w, la, x, lb, &beta, o2, lc, o2, lc, &res[bi].algo, ws, wsz, s));
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(b.data(), o2, b.size() * 2, cudaMemcpyDeviceToHost));
            size_t diff = 0; for (size_t i = 0; i < a.size(); ++i) diff += (*(uint16_t*)&a[i] != *(uint16_t*)&b[i]);
            printf("   best-lt vs GemmEx output: %zu / %zu elements differ\n", diff, a.size());
        }
        cudaFree(bx); cudaFree(bw); cudaFree(bo); cudaFree(bo2);
    }
    return 0;
}
