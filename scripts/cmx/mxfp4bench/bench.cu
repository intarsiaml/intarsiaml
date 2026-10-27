// cmx: standalone MXFP4 W4A16 microbench: GEMV (kernels/cuda/mxfp4.cu) against the tensor-core GEMM (kernels/cuda/mxfp4_gemm.cu)
// on the Qwen3.8-27B shapes. Build/run on spark1 inside a fleetq slot:
//   nvcc -O3 -arch=sm_121 -I kernels -I kernels/cuda -o bench scripts/cmx/mxfp4bench/bench.cu kernels/cuda/mxfp4.cu kernels/cuda/mxfp4_gemm.cu
//   ./bench [reps]
// Per shape and M: GEMV us (M <= 64 only), GEMM us, GB/s of the weight bytes, TFLOP/s, max |gemm - gemv| relative to max |gemv|
// (M <= 64), and the row-invariance check: the first 16 rows of the GEMM at M against the GEMM at M = 16 (bitwise).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "ik_launch.h"
#include "ik_cuda_abi.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static ik_tensor T(void* p, int dt, int64_t a, int64_t b, int nd = 2) {
    ik_tensor t; memset(&t, 0, sizeof t);
    t.data = p; t.dtype = dt; t.ndim = nd; t.shape[0] = a; t.shape[1] = b; t.stride[0] = b; t.stride[1] = 1;
    return t;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    struct Sh { const char* name; int64_t N, K; };
    const Sh shapes[] = {{"gate/up", 17408, 5120}, {"down", 5120, 17408}, {"o_proj", 5120, 6144},
                         {"qkv-ish", 10240, 5120}, {"q", 6144, 5120}, {"gdn-in", 12288, 5120}};
    const int Ms[] = {2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 32, 33, 64, 128, 256, 1024, 4096};
    cudaStream_t s; cudaEvent_t e0, e1;
    CK(cudaStreamCreate(&s)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float* act; CK(cudaMalloc(&act, 4)); CK(cudaMemset(act, 0, 4));
    const int NB = 4;
    for (const Sh& sh : shapes) {
        uint8_t *w[NB], *sc[NB];
        std::vector<uint8_t> hw((size_t)sh.N * sh.K / 2), hs((size_t)sh.N * sh.K / 32);
        srand(7 + (int)sh.N);
        for (auto& b : hw) b = (uint8_t)rand();
        for (auto& b : hs) b = (uint8_t)(120 + rand() % 8);
        for (int b = 0; b < NB; ++b) {
            CK(cudaMalloc(&w[b], hw.size())); CK(cudaMalloc(&sc[b], hs.size()));
            CK(cudaMemcpy(w[b], hw.data(), hw.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(sc[b], hs.data(), hs.size(), cudaMemcpyHostToDevice));
        }
        const int64_t MM = 4096;
        std::vector<ik_bf16> hx((size_t)MM * sh.K);
        for (auto& v : hx) { float f = (float)(rand() % 2001 - 1000) / 500.f; v = __float2bfloat16(f); }
        ik_bf16 *x, *o0, *o1, *o16; 
        CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&o0, (size_t)MM * sh.N * 2)); CK(cudaMalloc(&o1, (size_t)MM * sh.N * 2)); CK(cudaMalloc(&o16, (size_t)16 * sh.N * 2));
        auto at = T(act, IK_DTYPE_F32, 1, 1, 1);
        for (int M : Ms) {
            auto xt = T(x, IK_DTYPE_BF16, M, sh.K);
            auto time = [&](auto fn, int r) {
                for (int i = 0; i < 2; ++i) fn(i % NB);
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < r; ++i) fn(i % NB);
                CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms * 1000.f / r;
            };
            auto gemv = [&](int b) {
                auto wt = T(w[b], IK_DTYPE_U8, sh.N, sh.K / 2); auto st = T(sc[b], IK_DTYPE_U8, sh.N, sh.K / 32); auto ot = T(o0, IK_DTYPE_BF16, M, sh.N);
                int r = ik_matmul_mxfp4_gemv(&xt, &wt, &st, &at, &ot, s); if (r) { printf("gemv err %d\n", r); exit(1); }
            };
            auto gemm = [&](int b) {
                auto wt = T(w[b], IK_DTYPE_U8, sh.N, sh.K / 2); auto st = T(sc[b], IK_DTYPE_U8, sh.N, sh.K / 32); auto ot = T(o1, IK_DTYPE_BF16, M, sh.N);
                int r = ik_matmul_mxfp4_mma(&xt, &wt, &st, &at, &ot, s); if (r) { printf("gemm err %d\n", r); exit(1); }
            };
            const int r = M >= 1024 ? 3 : reps;
            float tg = M <= 64 ? time(gemv, r) : -1.f;
            float tm = time(gemm, r);
            double maxd = -1, maxref = 0; int rowinv = -1;
            if (M <= 64) {
                std::vector<ik_bf16> a((size_t)M * sh.N), b((size_t)M * sh.N);
                gemv(0); gemm(0); CK(cudaStreamSynchronize(s));
                CK(cudaMemcpy(a.data(), o0, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), o1, b.size() * 2, cudaMemcpyDeviceToHost));
                maxd = 0;
                for (size_t i = 0; i < a.size(); ++i) { double x0 = __bfloat162float(a[i]), x1 = __bfloat162float(b[i]); maxd = fmax(maxd, fabs(x0 - x1)); maxref = fmax(maxref, fabs(x0)); }
            }
            {   // row invariance: rows 0..min(M,16)-1 of the GEMM at M against the GEMM at M = 16 on the same rows
                const int R = M < 16 ? M : 16;
                auto x16 = T(x, IK_DTYPE_BF16, 16, sh.K); auto wt = T(w[0], IK_DTYPE_U8, sh.N, sh.K / 2); auto st = T(sc[0], IK_DTYPE_U8, sh.N, sh.K / 32); auto ot = T(o16, IK_DTYPE_BF16, 16, sh.N);
                gemm(0); CK(cudaStreamSynchronize(s));
                int rc = ik_matmul_mxfp4_mma(&x16, &wt, &st, &at, &ot, s); if (rc) { printf("gemm16 err %d\n", rc); exit(1); }
                CK(cudaStreamSynchronize(s));
                std::vector<ik_bf16> a((size_t)R * sh.N), b((size_t)R * sh.N);
                CK(cudaMemcpy(a.data(), o1, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), o16, b.size() * 2, cudaMemcpyDeviceToHost));
                rowinv = memcmp(a.data(), b.data(), a.size() * 2) == 0;
            }
            const double wb = (double)sh.N * sh.K / 2 + (double)sh.N * sh.K / 32;
            printf("%-8s N=%-5ld K=%-5ld M=%-4d gemv %8.1f us  gemm %8.1f us (%6.1f GB/s %6.2f TFLOP/s)  maxdiff/maxref %.2e  rowinv %d\n", sh.name, (long)sh.N, (long)sh.K, M,
                   tg, tm, wb / (tm * 1e3), 2.0 * M * sh.N * sh.K / (tm * 1e6), maxd < 0 ? -1.0 : maxd / maxref, rowinv);
            fflush(stdout);
        }
        for (int b = 0; b < NB; ++b) { cudaFree(w[b]); cudaFree(sc[b]); }
        cudaFree(x); cudaFree(o0); cudaFree(o1); cudaFree(o16);
    }
    return 0;
}
