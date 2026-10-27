// s27d: standalone MXFP4 GEMV microbench. nvcc -O3 -arch=sm_121 -I../../../kernels/cuda -o bench bench.cu
// Usage: ./bench [reps]  Prints per 27B shape, M in {1,3,4}: baseline and pipe<B> times, GB/s, bitwise equality.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../../kernels/cuda/mxfp4.cu"   // baseline (anonymous namespace + extern "C" launcher)
#include "variants.cuh"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
static ik_tensor T(void* p, int dt, int64_t a, int64_t b) { ik_tensor t; memset(&t, 0, sizeof t); t.data = p; t.dtype = dt; t.ndim = 2; t.shape[0] = a; t.shape[1] = b; t.stride[0] = b; t.stride[1] = 1; return t; }
template <int B> static void run_pipe(const ik_bf16* x, const uint8_t* w, const uint8_t* sc, const float* act, ik_bf16* out, int64_t M, int64_t N, int K, cudaStream_t s) {
    dim3 grid((unsigned)((N + kWarps - 1) / kWarps), (unsigned)((M + kMB - 1) / kMB));
    mxfp4_gemv_pipe_kernel<ik_bf16, B><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, sc, act, out, M, N, K);
}
int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 50;
    struct Sh { const char* name; int64_t N; int K; } shapes[] = {{"gate/up", 17408, 5120}, {"down", 5120, 17408}, {"out/o", 5120, 6144}, {"qkv", 10240, 5120}, {"z", 6144, 5120}, {"q", 12288, 5120}};
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float* act; CK(cudaMalloc(&act, 4)); CK(cudaMemset(act, 0, 4));
    for (auto sh : shapes) {
        const int NB = 4;  // rotating weight copies (> L2)
        uint8_t *w[NB], *sc[NB];
        size_t wb = (size_t)sh.N * sh.K / 2, sb = (size_t)sh.N * sh.K / 32;
        std::vector<uint8_t> hw(wb), hs(sb);
        for (size_t i = 0; i < wb; ++i) hw[i] = (uint8_t)(rand() & 0xff);
        for (size_t i = 0; i < sb; ++i) hs[i] = (uint8_t)(120 + rand() % 8);
        for (int b = 0; b < NB; ++b) { CK(cudaMalloc(&w[b], wb)); CK(cudaMalloc(&sc[b], sb)); CK(cudaMemcpy(w[b], hw.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sc[b], hs.data(), sb, cudaMemcpyHostToDevice)); }
        for (int M : {1, 3, 4}) {
            std::vector<ik_bf16> hx((size_t)M * sh.K); for (auto& v : hx) { float f = (rand() % 2001 - 1000) / 500.f; v = __float2bfloat16(f); }
            ik_bf16 *x, *o0, *o1; CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&o0, (size_t)M * sh.N * 2)); CK(cudaMalloc(&o1, (size_t)M * sh.N * 2));
            auto xt = T(x, IK_DTYPE_BF16, M, sh.K); auto at = T(act, IK_DTYPE_F32, 1, 1); at.ndim = 1; at.shape[0] = 1;
            auto time = [&](auto fn) {
                for (int i = 0; i < 5; ++i) fn(i % NB);
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < reps; ++i) fn(i % NB);
                CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms * 1000.f / reps;
            };
            auto base = [&](int b) { auto wt = T(w[b], IK_DTYPE_U8, sh.N, sh.K / 2); auto st = T(sc[b], IK_DTYPE_U8, sh.N, sh.K / 32); auto ot = T(o0, IK_DTYPE_BF16, M, sh.N);
                                     int r = ik_matmul_mxfp4_gemv(&xt, &wt, &st, &at, &ot, s); if (r) { printf("base err %d\n", r); exit(1); } };
            float tb = time(base);
            std::vector<ik_bf16> r0((size_t)M * sh.N), r1((size_t)M * sh.N);
            CK(cudaMemcpy(r0.data(), o0, r0.size() * 2, cudaMemcpyDeviceToHost));
            double gb = (double)(wb + sb) / 1e3;  // MB -> per us gives GB/s when /us * 1e-3... bytes/us/1e3
            printf("%-8s N=%-6lld K=%-6d M=%d base %7.1f us %6.1f GB/s", sh.name, (long long)sh.N, sh.K, M, tb, (wb + sb) / tb / 1e3);
            auto pipe = [&](auto fnB, const char* nm) {
                float t = time([&](int b) { fnB(w[b], sc[b]); });
                CK(cudaMemcpy(r1.data(), o1, r1.size() * 2, cudaMemcpyDeviceToHost));
                printf(" | %s %7.1f us %6.1f GB/s %s", nm, t, (wb + sb) / t / 1e3, memcmp(r0.data(), r1.data(), r0.size() * 2) == 0 ? "bitwise" : "DIFF");
            };
            pipe([&](const uint8_t* wp, const uint8_t* sp) { run_pipe<1>(x, wp, sp, act, o1, M, sh.N, sh.K, s); }, "p1");
            pipe([&](const uint8_t* wp, const uint8_t* sp) { run_pipe<2>(x, wp, sp, act, o1, M, sh.N, sh.K, s); }, "p2");
            pipe([&](const uint8_t* wp, const uint8_t* sp) { run_pipe<4>(x, wp, sp, act, o1, M, sh.N, sh.K, s); }, "p4");
            printf("\n");
            cudaFree(x); cudaFree(o0); cudaFree(o1);
        }
        for (int b = 0; b < NB; ++b) { cudaFree(w[b]); cudaFree(sc[b]); }
    }
    return 0;
}
