// rope_ip_bench: hash + time of ik_rope_tail on the rope shapes (same shapes, seeds and generators as the rope section of
// ../../../2026-11-03-dsv41-attn-fusion/raw/bench/hc_rope_bench.cu, so the hashes are comparable with its bench-vec0.txt).
// usage: rope_ip_bench [iters] [mode]
//   mode 0 (default): out of place, x -> y, as the engine calls it today.
//   mode 1: in place: y = a copy of x, then ik_rope_tail(y -> y); only valid against an object built with -DROPE_INPLACE or -DROPE_INPLACE_SHFL
//           (rope-inplace.patch). The hashed result is the first application on the fresh copy; the timing reps then re-rotate the same buffer
//           (the values stay finite and the kernel's time does not depend on them).
// Every hash must equal the mode-0 hash of the unmodified object. Output: one `rope <name> hash <h> min <ms> med <ms> ms` line per shape.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include "../../../../../kernels/abi/ik_abi.h"
extern "C" {
int ik_rope_tail(const ik_tensor* x, const ik_tensor* inv_freq, int64_t pos0, int64_t pos_stride, int64_t ratio, int32_t inverse, ik_tensor* out,
                 const ik_tensor* pos, void* stream);
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)
#define IKC(x) do { int e = (x); if (e != 0) { printf("ik status %d at %s:%d\n", e, __FILE__, __LINE__); exit(1);} } while (0)

static ik_tensor mk(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t; memset(&t, 0, sizeof t);
    t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1;
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}
static uint64_t fnv(const void* d, size_t n) { const uint8_t* p = (const uint8_t*)d; uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }
static uint64_t hash_dev(const void* d, size_t n) { std::vector<uint8_t> h(n); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return fnv(h.data(), n); }
static void* dev_rand_bf16(size_t n, float scale, uint32_t seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<__nv_bfloat16> h(n); for (auto& v : h) v = __float2bfloat16_rn(u(g));
    void* d; CK(cudaMalloc(&d, n * 2)); CK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice)); return d;
}
static void* dev_rand_f32(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> h(n); for (auto& v : h) v = u(g);
    void* d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice)); return d;
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 20;
    const int mode = argc > 2 ? atoi(argv[2]) : 0;
    const int reps = 9;
    printf("rope_ip_bench mode %d (%s) iters %d reps %d\n", mode, mode ? "in place" : "out of place", iters, reps);
    struct RC { const char* name; int64_t T, N, Dh, half, pos0, stride; int inv; };
    RC rcs[] = {{"q T512 H32 hd512", 512, 32, 512, 32, 1000, 1, 0}, {"o^-1 T512 H32 hd512", 512, 32, 512, 32, 1000, 1, 1}, {"q T2048 H32 hd512", 2048, 32, 512, 32, 0, 1, 0},
                {"o^-1 T2048 H32 hd512", 2048, 32, 512, 32, 2048, 1, 1}, {"kv T512 H1 hd512", 512, 1, 512, 32, 1536, 1, 0}, {"idx_q T512 H64 hd128", 512, 64, 128, 32, 1536, 1, 0},
                {"cmp T128 H1 hd512 stride4", 128, 1, 512, 32, 1024, 4, 0}, {"T63 H32 hd512 (old path)", 63, 32, 512, 32, 7, 1, 0}, {"T500 H32 hd512 pos 40000", 500, 32, 512, 32, 40000, 1, 0}};
    for (auto& c : rcs) {
        size_t n = c.T * c.N * c.Dh;
        void* x = dev_rand_bf16(n, 4.f, 11); void* y; CK(cudaMalloc(&y, n * 2)); CK(cudaMemset(y, 0, n * 2));
        void* f = dev_rand_f32(c.half, 1e-4f, 1.0f, 5);
        ik_tensor tx = mk(x, IK_DTYPE_BF16, {c.T, c.N, c.Dh}), tf = mk(f, IK_DTYPE_F32, {c.half}), ty = mk(y, IK_DTYPE_BF16, {c.T, c.N, c.Dh});
        auto run = [&] {
            if (mode == 0) IKC(ik_rope_tail(&tx, &tf, c.pos0, c.stride, 1, c.inv, &ty, nullptr, nullptr));
            else IKC(ik_rope_tail(&ty, &tf, c.pos0, c.stride, 1, c.inv, &ty, nullptr, nullptr));
        };
        if (mode == 1) CK(cudaMemcpy(y, x, n * 2, cudaMemcpyDeviceToDevice));
        run();
        CK(cudaDeviceSynchronize());
        const uint64_t h = hash_dev(y, n * 2);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 3 * iters; ++i) run();
        CK(cudaDeviceSynchronize());
        std::vector<float> r;
        for (int rep = 0; rep < reps; ++rep) {
            cudaEventRecord(a);
            for (int i = 0; i < iters; ++i) run();
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); r.push_back(ms / iters);
        }
        std::sort(r.begin(), r.end());
        printf("rope %-28s hash %016llx min %.4f med %.4f ms\n", c.name, (unsigned long long)h, r.front(), r[r.size() / 2]);
        cudaFree(x); cudaFree(y); cudaFree(f);
    }
    return 0;
}
