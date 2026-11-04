// attn_ab_bench: attention-only A/B bench (same inputs as hc_rope_bench.cu's attention section; rep list for the spread). usage: attn_ab_bench [iters] [reps]
// derived from: Standalone check + timing of the DSv4.1 prefill kernels touched by lane attnfuse: rope_tail (element-wise vs row-vector kernel) and
// hc_sublayer / hc_post (element-wise vs vector staging; separate HcPost + unfolded op vs folded op). Linked against the repo's own
// objects (dsv41_misc.cu, dsv41_hc.cu, dsv41_hc_sub.cu). Run twice, INTARSIA_DSV41_ROPE_VEC / INTARSIA_DSV41_HC_VEC = 0 and 1 (or unset):
// every hash line must be equal across the two runs AND, for hc, across the sep / fold variants.
// usage: hc_rope_bench [iters]
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "../../../../../kernels/abi/ik_abi.h"
extern "C" {
int ik_rope_tail(const ik_tensor* x, const ik_tensor* inv_freq, int64_t pos0, int64_t pos_stride, int64_t ratio, int32_t inverse, ik_tensor* out,
                 const ik_tensor* pos, void* stream);
int ik_attn_sparse(const ik_tensor* q, const ik_tensor* win, const ik_tensor* cmp, const ik_tensor* idx, const ik_tensor* sink, int64_t window, int64_t pos0,
                   int64_t ratio, float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream);
int ik_attn_mma_probe(int64_t D);
int ik_hc_post(const ik_tensor* x, const ik_tensor* res, const ik_tensor* post, const ik_tensor* comb, ik_tensor* y, void* stream);
int ik_hc_sublayer(const ik_tensor* x, const ik_tensor* fn, const ik_tensor* scale, const ik_tensor* base, const ik_tensor* pre_in,
                   const ik_tensor* norm_w, const ik_tensor* post_x, const ik_tensor* post_mix, const ik_tensor* post_comb, float eps, float w_offset,
                   int64_t iters, float hc_eps, ik_tensor* n, ik_tensor* pre, ik_tensor* post, ik_tensor* comb, ik_tensor* h_out,
                   const ik_scratch* scratch, void* stream);
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
template <typename F> static float time_ms(F f, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) f();
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) f();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
        if (ms < best) best = ms;
    }
    return best;
}
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
    const int reps = argc > 2 ? atoi(argv[2]) : 15;
    ik_attn_mma_probe(512);
    // same shapes, seeds and generators as hc_rope_bench.cu (so the hashes are comparable with bench-vec0.txt), plus T11 (split path)
    struct AC { const char* name; int64_t T, Nc, Kw, pos0, ratio; };
    AC acs[] = {{"CSA T512 Nc1024 Kw512", 512, 1024, 512, 3072, 4}, {"CSA T2048 Nc1024 Kw512", 2048, 1024, 512, 2048, 4}, {"HCA T512 Nc32 Kw32", 512, 32, 32, 3072, 128},
                {"CSA T37 Nc1024 Kw512", 37, 1024, 512, 4000, 4}, {"CSA T11 Nc1024 Kw512", 11, 1024, 512, 4000, 4}};
    for (auto& c : acs) {
        const int64_t H = 32, D = 512, W = 128;
        void* q = dev_rand_bf16(c.T * H * D, 1.f, 41); void* win = dev_rand_bf16((W + c.T) * D, 1.f, 42); void* cmp = dev_rand_bf16(c.Nc * D, 1.f, 43);
        std::mt19937 g(44); std::vector<int> hi(c.T * c.Kw); for (auto& v : hi) { v = (int)(g() % (c.Nc + c.Nc / 16)); if (v >= c.Nc) v = -1; }
        void* idx; CK(cudaMalloc(&idx, hi.size() * 4)); CK(cudaMemcpy(idx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
        void* sink = dev_rand_f32(H, -1.f, 1.f, 45); void* o; CK(cudaMalloc(&o, c.T * H * D * 2)); CK(cudaMemset(o, 0, c.T * H * D * 2));
        ik_tensor tq = mk(q, IK_DTYPE_BF16, {c.T, H, D}), tw = mk(win, IK_DTYPE_BF16, {W + c.T, D}), tc = mk(cmp, IK_DTYPE_BF16, {c.Nc, D}), ti = mk(idx, IK_DTYPE_I32, {c.T, c.Kw}),
                  ts = mk(sink, IK_DTYPE_F32, {H}), to = mk(o, IK_DTYPE_BF16, {c.T, H, D});
        const float sc = 1.f / sqrtf((float)D);
        // T <= 16 splits the key list: the merge needs scratch
        void* scr = nullptr; ik_scratch sk; memset(&sk, 0, sizeof sk);
        if (c.T <= 16) { sk.bytes = (size_t)c.T * H * 32 * (D + 2) * 4; CK(cudaMalloc(&scr, sk.bytes)); sk.data = scr; }
        const ik_scratch* skp = scr ? &sk : nullptr;
        IKC(ik_attn_sparse(&tq, &tw, &tc, &ti, &ts, W, c.pos0, c.ratio, sc, &to, skp, nullptr, nullptr));
        CK(cudaDeviceSynchronize());
        uint64_t h = hash_dev(o, c.T * H * D * 2);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 3 * iters; ++i) ik_attn_sparse(&tq, &tw, &tc, &ti, &ts, W, c.pos0, c.ratio, sc, &to, skp, nullptr, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> r;
        for (int rep = 0; rep < reps; ++rep) {
            cudaEventRecord(a);
            for (int i = 0; i < iters; ++i) ik_attn_sparse(&tq, &tw, &tc, &ti, &ts, W, c.pos0, c.ratio, sc, &to, skp, nullptr, nullptr);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); r.push_back(ms / iters);
        }
        std::vector<float> s = r; std::sort(s.begin(), s.end());
        printf("attn %-24s hash %016llx min %.4f med %.4f max %.4f ms | reps:", c.name, (unsigned long long)h, s.front(), s[s.size() / 2], s.back());
        for (float v : r) printf(" %.4f", v);
        printf("\n");
        for (void* p : {q, win, cmp, idx, sink, o}) cudaFree(p);
        if (scr) cudaFree(scr);
    }
    return 0;
}
