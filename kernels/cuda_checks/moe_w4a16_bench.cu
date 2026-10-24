// Standalone microbench + bitwise check for the CUDA routed-expert prefill kernels (lane mp).
//   nvcc -O3 -std=c++17 -arch=sm_121 -I kernels/cuda -I kernels kernels/cuda_checks/moe_w4a16_bench.cu kernels/cuda/moe_w4a16.cu \
//        [kernels/cuda/moe_w4a16_v3.cu -DHAVE_V3] -o moe_w4a16_bench
//   ./moe_w4a16_bench [T,T,...] [reps] [skew]      T tokens (default 1024,2048,4096), skew 0 = uniform routing, 1 = zipf-like
// Real layer shape (256 experts, top-8, H 2048, I 512). Times the production launcher (ik_moe_experts_w4a16_bf16) and, with
// HAVE_V3, the candidate (ik_moe_experts_w4a16_v3_bf16); the candidate's output is compared byte for byte with the production
// kernel's (both must be bit-identical: same accumulation order).
#include "abi/ik_abi.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

extern "C" {
int ik_moe_experts_w4a16_bf16(const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*,
                              const ik_tensor*, const ik_tensor*, const ik_tensor*, ik_tensor*, const ik_scratch*, void*);
#ifdef HAVE_V3
int ik_moe_experts_w4a16_v3_bf16(const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*,
                                 const ik_tensor*, const ik_tensor*, const ik_tensor*, ik_tensor*, const ik_scratch*, void*);
#endif
}

#define CK(x)                                                                                         \
    do {                                                                                              \
        cudaError_t e_ = (x);                                                                         \
        if (e_ != cudaSuccess) {                                                                      \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));                \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

static ik_tensor mk(void* p, int dt, std::vector<int64_t> shape) {
    ik_tensor t;
    memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = dt;
    t.ndim = (int)shape.size();
    int64_t st = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}

static int64_t r256(int64_t v) { return (v + 255) & ~int64_t(255); }

template <class T>
static T* dev_copy(const std::vector<T>& h) {
    T* d;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    std::vector<int> ts = {1024, 2048, 4096};
    if (argc > 1) {
        ts.clear();
        for (char* s = strtok(argv[1], ","); s; s = strtok(nullptr, ",")) ts.push_back(atoi(s));
    }
    const int reps = argc > 2 ? atoi(argv[2]) : 10;
    const int skew = argc > 3 ? atoi(argv[3]) : 0;
    const int E = 256, K = 8, H = 2048, I = 512;
    const int Tmax = *std::max_element(ts.begin(), ts.end());
    std::mt19937_64 rng(5);
    // weights: random packed fp4 bytes, e4m3 scales in a sane positive range
    std::vector<uint8_t> guw((size_t)E * 2 * I * (H / 2)), dnw((size_t)E * H * (I / 2));
    std::vector<uint8_t> gus((size_t)E * 2 * I * (H / 16)), dns((size_t)E * H * (I / 16));
    for (auto& b : guw) b = (uint8_t)rng();
    for (auto& b : dnw) b = (uint8_t)rng();
    for (auto& b : gus) b = (uint8_t)(0x30 + rng() % 0x14);
    for (auto& b : dns) b = (uint8_t)(0x30 + rng() % 0x14);
    std::vector<float> gsc((size_t)E * 3);
    for (auto& g : gsc) g = 1e-3f * (1.f + (rng() % 1000) / 1000.f);
    std::vector<uint16_t> x((size_t)Tmax * H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) {
        float f = nd(rng);
        uint32_t u;
        memcpy(&u, &f, 4);
        v = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    std::vector<int> ids((size_t)Tmax * K);
    std::vector<float> wts((size_t)Tmax * K, 0.125f);
    // zipf-like skew: expert popularity ~ 1/(rank+1)^0.7 (real routing is far from uniform)
    std::vector<double> pop(E);
    for (int e = 0; e < E; ++e) pop[e] = skew ? 1.0 / std::pow(e + 1.0, 0.7) : 1.0;
    std::discrete_distribution<int> dist(pop.begin(), pop.end());
    for (int t = 0; t < Tmax; ++t) {
        int got[K];
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = dist(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= got[j] == e;
            } while (dup);
            got[k] = e;
            ids[(size_t)t * K + k] = e;
        }
    }
    uint8_t *d_guw = dev_copy(guw), *d_dnw = dev_copy(dnw), *d_gus = dev_copy(gus), *d_dns = dev_copy(dns);
    float* d_gsc = dev_copy(gsc);
    uint16_t* d_x = dev_copy(x);
    int* d_ids = dev_copy(ids);
    float* d_w = dev_copy(wts);
    const int64_t Pmax = (int64_t)Tmax * K;
    const int64_t max_tiles = (Pmax + 15) / 16 + std::min<int64_t>(Pmax, E);
    const int64_t scratch_bytes =
        r256(Pmax * 4) + r256((E + 1) * 4) + 256 + r256(max_tiles * 8) + r256(Pmax * I * 2) + r256(Pmax * H * 2) + 4096;
    void* d_scr;
    CK(cudaMalloc(&d_scr, scratch_bytes));
    ik_scratch scr{d_scr, scratch_bytes};
    uint16_t *d_out, *d_out2;
    CK(cudaMalloc(&d_out, (size_t)Tmax * H * 2));
    CK(cudaMalloc(&d_out2, (size_t)Tmax * H * 2));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    printf("%-8s %-10s %10s %10s %8s\n", "T", "kernel", "us(med)", "TFLOPS", "bitwise");
    for (int T : ts) {
        ik_tensor tx = mk(d_x, IK_DTYPE_BF16, {T, H}), tids = mk(d_ids, IK_DTYPE_I32, {T, K}), tw = mk(d_w, IK_DTYPE_F32, {T, K});
        ik_tensor tguw = mk(d_guw, IK_DTYPE_U8, {E, 2 * I, H / 2}), tgus = mk(d_gus, IK_DTYPE_FP8_E4M3, {E, 2 * I, H / 16});
        ik_tensor tdnw = mk(d_dnw, IK_DTYPE_U8, {E, H, I / 2}), tdns = mk(d_dns, IK_DTYPE_FP8_E4M3, {E, H, I / 16});
        ik_tensor tgs = mk(d_gsc, IK_DTYPE_F32, {E, 3});
        ik_tensor tout = mk(d_out, IK_DTYPE_BF16, {T, H}), tout2 = mk(d_out2, IK_DTYPE_BF16, {T, H});
        const double tflop = 2.0 * (double)T * K * ((double)H * 2 * I + (double)I * H) / 1e12;
        struct Cand {
            const char* name;
            int (*fn)(const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*,
                      const ik_tensor*, const ik_tensor*, ik_tensor*, const ik_scratch*, void*);
            ik_tensor* out;
        } cands[] = {{"prod", ik_moe_experts_w4a16_bf16, &tout},
#ifdef HAVE_V3
                     {"v3", ik_moe_experts_w4a16_v3_bf16, &tout2},
#endif
        };
        std::vector<uint16_t> ref((size_t)T * H);
        const bool only = getenv("ONLY") != nullptr && getenv("ONLY")[0] != 0;  // ONLY=v3: skip the production kernel (profiling one kernel with nsys)
        for (auto& c : cands) {
            if (only && strcmp(c.name, getenv("ONLY")) != 0) continue;
            CK(cudaMemset(c.out->data, 0xAB, (size_t)T * H * 2));
            int st = c.fn(&tx, &tids, &tw, &tguw, &tgus, &tdnw, &tdns, &tgs, c.out, &scr, nullptr);
            CK(cudaDeviceSynchronize());
            if (st != 0) {
                printf("%-8d %-10s status %d\n", T, c.name, st);
                continue;
            }
            std::vector<uint16_t> got((size_t)T * H);
            CK(cudaMemcpy(got.data(), c.out->data, got.size() * 2, cudaMemcpyDeviceToHost));
            const char* bw = "ref";
            if (c.out == &tout) ref = got;
            else bw = got == ref ? "BITWISE" : "DIFF";
            size_t nd = 0;
            if (c.out != &tout)
                for (size_t i = 0; i < got.size(); ++i) nd += got[i] != ref[i];
            std::vector<float> us;
            for (int r = 0; r < reps + 2; ++r) {
                CK(cudaEventRecord(e0));
                c.fn(&tx, &tids, &tw, &tguw, &tgus, &tdnw, &tdns, &tgs, c.out, &scr, nullptr);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                if (r >= 2) us.push_back(ms * 1e3f);
            }
            std::sort(us.begin(), us.end());
            const float med = us[us.size() / 2];
            printf("%-8d %-10s %10.1f %10.1f %8s", T, c.name, med, tflop / (med * 1e-6), bw);
            if (nd) printf("  (%zu of %zu differ)", nd, got.size());
            printf("  [min %.1f max %.1f]\n", us.front(), us.back());
        }
    }
    return 0;
}
