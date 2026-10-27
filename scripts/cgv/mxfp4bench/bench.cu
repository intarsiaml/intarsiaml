// cgv: standalone MXFP4 GEMV microbench, baseline `mxfp4_gemv_kernel` vs the multi-row kernel `mxfp4_gemv_mr_kernel<MT, R>` of kernels/cuda/mxfp4.cu.
//   nvcc -O3 -arch=sm_121 -I../../../kernels/cuda -o bench bench.cu      Usage: ./bench [reps]
// Per 27B shape and M in {1,2,3,4}: baseline us / GB/s, then each R in {1,2,4,8}: us, GB/s, bitwise-equal to the baseline.
// Weights rotate over 4 copies (> L2) so every launch streams from DRAM.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../../kernels/cuda/mxfp4.cu"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
static ik_tensor T(void* p, int dt, int64_t a, int64_t b) { ik_tensor t; memset(&t, 0, sizeof t); t.data = p; t.dtype = dt; t.ndim = 2; t.shape[0] = a; t.shape[1] = b; t.stride[0] = b; t.stride[1] = 1; return t; }
template <int MT, int R> static void run_mr(const ik_bf16* x, const uint8_t* w, const uint8_t* sc, ik_bf16* out, int64_t N, int K, cudaStream_t s) {
    dim3 grid((unsigned)((N / R + kWarps - 1) / kWarps));
    mxfp4_gemv_mr_kernel<ik_bf16, MT, R><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, sc, out, N, K);
}
template <int R> static void run_mr_m(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, ik_bf16* out, int64_t N, int K, cudaStream_t s) {
    switch (M) { case 1: run_mr<1, R>(x, w, sc, out, N, K, s); break; case 2: run_mr<2, R>(x, w, sc, out, N, K, s); break;
                 case 3: run_mr<3, R>(x, w, sc, out, N, K, s); break; default: run_mr<4, R>(x, w, sc, out, N, K, s); }
}
// DRAM ceiling probe: plain coalesced 16-byte streaming read of `n16` uint4 (one per thread per iteration, grid-stride), the xor of everything stored so the loads stay.
__global__ void stream_kernel(const uint4* __restrict__ p, int64_t n16, unsigned* out) {
    unsigned acc = 0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (int64_t)gridDim.x * blockDim.x) { const uint4 v = __ldcs(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}
int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 100;
    struct Sh { const char* name; int64_t N; int K; } shapes[] = {{"gate/up", 17408, 5120}, {"down", 5120, 17408}, {"out/o", 5120, 6144}, {"qkv", 10240, 5120}, {"z", 6144, 5120}, {"q", 12288, 5120}};
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float* act; CK(cudaMalloc(&act, 4)); CK(cudaMemset(act, 0, 4));
    if (getenv("STREAM")) {  // STREAM=1 NB=64 ./bench: the ceiling of a plain read over 47 MB buffers rotated NB deep
        const int NBs = getenv("NB") ? atoi(getenv("NB")) : 4; const size_t bytes = (size_t)17408 * 5120 * 17 / 32;
        std::vector<uint8_t*> b(NBs); unsigned* o; CK(cudaMalloc(&o, 4));
        for (auto& p : b) { CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 1, bytes)); }
        for (int blocks : {96, 192, 384, 768, 2176}) {
            for (int i = 0; i < 5; ++i) stream_kernel<<<blocks, 256, 0, s>>>((const uint4*)b[i % NBs], bytes / 16, o);
            float best = 1e30f;
            for (int blk = 0; blk < 5; ++blk) { CK(cudaEventRecord(e0, s)); for (int i = 0; i < reps; ++i) stream_kernel<<<blocks, 256, 0, s>>>((const uint4*)b[i % NBs], bytes / 16, o);
                CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = ms * 1000.f / reps < best ? ms * 1000.f / reps : best; }
            printf("stream NB=%d blocks=%d: %zu bytes %.1f us %.1f GB/s\n", NBs, blocks, bytes, best, bytes / best / 1e3);
        }
        return 0;
    }
    for (auto sh : shapes) {
        const int NB = getenv("NB") ? atoi(getenv("NB")) : 4;  // rotating weight copies; the engine streams 18 GB per token, so try NB >= 64
        std::vector<uint8_t*> w(NB), sc(NB);
        size_t wb = (size_t)sh.N * sh.K / 2, sb = (size_t)sh.N * sh.K / 32;
        std::vector<uint8_t> hw(wb), hs(sb);
        for (size_t i = 0; i < wb; ++i) hw[i] = (uint8_t)(rand() & 0xff);
        for (size_t i = 0; i < sb; ++i) hs[i] = (uint8_t)(120 + rand() % 8);
        for (int b = 0; b < NB; ++b) { CK(cudaMalloc(&w[b], wb)); CK(cudaMalloc(&sc[b], sb)); CK(cudaMemcpy(w[b], hw.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sc[b], hs.data(), sb, cudaMemcpyHostToDevice)); }
        for (int M : {1, 2, 3, 4}) {
            std::vector<ik_bf16> hx((size_t)M * sh.K); for (auto& v : hx) { float f = (rand() % 2001 - 1000) / 500.f; v = __float2bfloat16(f); }
            ik_bf16 *x, *o0, *o1; CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&o0, (size_t)M * sh.N * 2)); CK(cudaMalloc(&o1, (size_t)M * sh.N * 2));
            auto xt = T(x, IK_DTYPE_BF16, M, sh.K); auto at = T(act, IK_DTYPE_F32, 1, 1); at.ndim = 1; at.shape[0] = 1;
            auto time = [&](auto fn) {
                for (int i = 0; i < 5; ++i) fn(i % NB);
                float best = 1e30f;  // best of 5 blocks of `reps` launches (the box's clock and neighbours move +-3 % between blocks)
                for (int blk = 0; blk < 5; ++blk) {
                    CK(cudaEventRecord(e0, s));
                    for (int i = 0; i < reps; ++i) fn(i % NB);
                    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                    best = ms * 1000.f / reps < best ? ms * 1000.f / reps : best;
                }
                return best;
            };
            auto base = [&](int b) { auto wt = T(w[b], IK_DTYPE_U8, sh.N, sh.K / 2); auto st = T(sc[b], IK_DTYPE_U8, sh.N, sh.K / 32); auto ot = T(o0, IK_DTYPE_BF16, M, sh.N);
                                     int r = ik_matmul_mxfp4_gemv(&xt, &wt, &st, &at, &ot, s); if (r) { printf("base err %d\n", r); exit(1); } };
            float tb = time(base);
            std::vector<ik_bf16> r0((size_t)M * sh.N), r1((size_t)M * sh.N);
            CK(cudaMemcpy(r0.data(), o0, r0.size() * 2, cudaMemcpyDeviceToHost));
            printf("%-8s N=%-6lld K=%-6d M=%d base %7.1f us %6.1f GB/s", sh.name, (long long)sh.N, sh.K, M, tb, (wb + sb) / tb / 1e3);
            auto mr = [&](auto fnB, const char* nm) {
                CK(cudaMemset(o1, 0xff, r1.size() * 2));
                float t = time([&](int b) { fnB(w[b], sc[b]); });
                CK(cudaMemcpy(r1.data(), o1, r1.size() * 2, cudaMemcpyDeviceToHost));
                printf(" | %s %7.1f us %6.1f GB/s %s", nm, t, (wb + sb) / t / 1e3, memcmp(r0.data(), r1.data(), r0.size() * 2) == 0 ? "bitwise" : "DIFF");
            };
            mr([&](const uint8_t* wp, const uint8_t* sp) { run_mr_m<1>(M, x, wp, sp, o1, sh.N, sh.K, s); }, "R1");
            mr([&](const uint8_t* wp, const uint8_t* sp) { run_mr_m<2>(M, x, wp, sp, o1, sh.N, sh.K, s); }, "R2");
            mr([&](const uint8_t* wp, const uint8_t* sp) { run_mr_m<4>(M, x, wp, sp, o1, sh.N, sh.K, s); }, "R4");
            mr([&](const uint8_t* wp, const uint8_t* sp) { run_mr_m<8>(M, x, wp, sp, o1, sh.N, sh.K, s); }, "R8");
            mr([&](const uint8_t* wp, const uint8_t* sp) {  // the production launcher (R by M)
                auto wt = T((void*)wp, IK_DTYPE_U8, sh.N, sh.K / 2); auto st = T((void*)sp, IK_DTYPE_U8, sh.N, sh.K / 32); auto ot = T(o1, IK_DTYPE_BF16, M, sh.N);
                int r = ik_matmul_mxfp4_gemv_mr(&xt, &wt, &st, &at, &ot, s); if (r) { printf("mr err %d\n", r); exit(1); } }, "prod");
            printf("\n");
            cudaFree(x); cudaFree(o0); cudaFree(o1);
        }
        for (int b = 0; b < NB; ++b) { cudaFree(w[b]); cudaFree(sc[b]); }
    }
    return 0;
}
