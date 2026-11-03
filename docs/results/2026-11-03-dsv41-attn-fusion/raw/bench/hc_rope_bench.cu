// Standalone check + timing of the DSv4.1 prefill kernels touched by lane attnfuse: rope_tail (element-wise vs row-vector kernel) and
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
#include "../../../../../kernels/abi/ik_abi.h"
extern "C" {
int ik_rope_tail(const ik_tensor* x, const ik_tensor* inv_freq, int64_t pos0, int64_t pos_stride, int64_t ratio, int32_t inverse, ik_tensor* out,
                 const ik_tensor* pos, void* stream);
int ik_attn_sparse(const ik_tensor* q, const ik_tensor* win, const ik_tensor* cmp, const ik_tensor* idx, const ik_tensor* sink, int64_t window, int64_t pos0,
                   int64_t ratio, float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream);
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
    printf("env ROPE_VEC=%s HC_VEC=%s\n", getenv("INTARSIA_DSV41_ROPE_VEC") ? getenv("INTARSIA_DSV41_ROPE_VEC") : "-", getenv("INTARSIA_DSV41_HC_VEC") ? getenv("INTARSIA_DSV41_HC_VEC") : "-");
    // ---- rope_tail ----
    struct RC { const char* name; int64_t T, N, Dh, half, pos0, stride; int inv; };
    RC rcs[] = {{"q T512 H32 hd512", 512, 32, 512, 32, 1000, 1, 0}, {"o^-1 T512 H32 hd512", 512, 32, 512, 32, 1000, 1, 1}, {"q T2048 H32 hd512", 2048, 32, 512, 32, 0, 1, 0},
                {"o^-1 T2048 H32 hd512", 2048, 32, 512, 32, 2048, 1, 1}, {"kv T512 H1 hd512", 512, 1, 512, 32, 1536, 1, 0}, {"idx_q T512 H64 hd128", 512, 64, 128, 32, 1536, 1, 0},
                {"cmp T128 H1 hd512 stride4", 128, 1, 512, 32, 1024, 4, 0}, {"T63 H32 hd512 (old path)", 63, 32, 512, 32, 7, 1, 0}, {"T500 H32 hd512 pos 40000", 500, 32, 512, 32, 40000, 1, 0}};
    for (auto& c : rcs) {
        size_t n = c.T * c.N * c.Dh;
        void* x = dev_rand_bf16(n, 4.f, 11); void* y; CK(cudaMalloc(&y, n * 2)); CK(cudaMemset(y, 0, n * 2));
        void* f = dev_rand_f32(c.half, 1e-4f, 1.0f, 5);
        ik_tensor tx = mk(x, IK_DTYPE_BF16, {c.T, c.N, c.Dh}), tf = mk(f, IK_DTYPE_F32, {c.half}), ty = mk(y, IK_DTYPE_BF16, {c.T, c.N, c.Dh});
        IKC(ik_rope_tail(&tx, &tf, c.pos0, c.stride, 1, c.inv, &ty, nullptr, nullptr));
        CK(cudaDeviceSynchronize());
        uint64_t h = hash_dev(y, n * 2);
        float ms = time_ms([&] { ik_rope_tail(&tx, &tf, c.pos0, c.stride, 1, c.inv, &ty, nullptr, nullptr); }, iters);
        printf("rope %-28s hash %016llx  %8.4f ms  (%.0f GB/s)\n", c.name, (unsigned long long)h, ms, 2.0 * n * 2 / (ms * 1e6));
        cudaFree(x); cudaFree(y); cudaFree(f);
    }
    // ---- attention (prompt chunk: tensor-core kernel, one block per query) ----
    struct AC { const char* name; int64_t T, Nc, Kw, pos0, ratio; };
    AC acs[] = {{"CSA T512 Nc1024 Kw512", 512, 1024, 512, 3072, 4}, {"CSA T2048 Nc1024 Kw512", 2048, 1024, 512, 2048, 4}, {"HCA T512 Nc32 Kw32", 512, 32, 32, 3072, 128}, {"CSA T37 Nc1024 Kw512", 37, 1024, 512, 4000, 4}};
    for (auto& c : acs) {
        const int64_t H = 32, D = 512, W = 128;
        void* q = dev_rand_bf16(c.T * H * D, 1.f, 41); void* win = dev_rand_bf16((W + c.T) * D, 1.f, 42); void* cmp = dev_rand_bf16(c.Nc * D, 1.f, 43);
        std::mt19937 g(44); std::vector<int> hi(c.T * c.Kw); for (auto& v : hi) { v = (int)(g() % (c.Nc + c.Nc / 16)); if (v >= c.Nc) v = -1; }
        void* idx; CK(cudaMalloc(&idx, hi.size() * 4)); CK(cudaMemcpy(idx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
        void* sink = dev_rand_f32(H, -1.f, 1.f, 45); void* o; CK(cudaMalloc(&o, c.T * H * D * 2)); CK(cudaMemset(o, 0, c.T * H * D * 2));
        ik_tensor tq = mk(q, IK_DTYPE_BF16, {c.T, H, D}), tw = mk(win, IK_DTYPE_BF16, {W + c.T, D}), tc = mk(cmp, IK_DTYPE_BF16, {c.Nc, D}), ti = mk(idx, IK_DTYPE_I32, {c.T, c.Kw}),
                  ts = mk(sink, IK_DTYPE_F32, {H}), to = mk(o, IK_DTYPE_BF16, {c.T, H, D});
        const float sc = 1.f / sqrtf((float)D);
        IKC(ik_attn_sparse(&tq, &tw, &tc, &ti, &ts, W, c.pos0, c.ratio, sc, &to, nullptr, nullptr, nullptr));
        CK(cudaDeviceSynchronize());
        uint64_t h = hash_dev(o, c.T * H * D * 2);
        float ms = time_ms([&] { ik_attn_sparse(&tq, &tw, &tc, &ti, &ts, W, c.pos0, c.ratio, sc, &to, nullptr, nullptr, nullptr); }, iters);
        printf("attn %-26s hash %016llx  %8.4f ms\n", c.name, (unsigned long long)h, ms);
        for (void* p : {q, win, cmp, idx, sink, o}) cudaFree(p);
    }
    // ---- hc ----
    struct HS { int S; int64_t D, T; };
    HS hss[] = {{4, 5120, 512}, {4, 5120, 2048}, {4, 5120, 11}, {4, 5120, 37}, {3, 2048, 70}, {2, 2048, 70}, {1, 2048, 33}, {4, 5000, 70}};
    for (auto& hs : hss) {
        const int S = hs.S; const int64_t D = hs.D, T = hs.T; const int m = (2 + S) * S;
        void* x = dev_rand_bf16(T * S * D, 2.f, 21);            // previous stream
        void* ao = dev_rand_bf16(T * D, 1.f, 22);               // sublayer output
        void* fn = dev_rand_f32((size_t)m * S * D, -0.02f, 0.02f, 23);
        float sc[3] = {0.7f, 0.6f, 0.5f}; void* scale; CK(cudaMalloc(&scale, 12)); CK(cudaMemcpy(scale, sc, 12, cudaMemcpyHostToDevice));
        void* base = dev_rand_f32(m, -0.5f, 0.5f, 24);
        void* pre_in = dev_rand_f32(T * S, 0.f, 1.f, 25);
        void* nw = dev_rand_bf16(D, 1.f, 26);
        void* pmix = dev_rand_f32(T * S, 0.f, 2.f, 27); void* pcomb = dev_rand_f32(T * S * S, 0.f, 1.f, 28);
        void *h, *h2, *nn, *pre, *post, *comb; size_t hb = (size_t)T * S * D * 2;
        CK(cudaMalloc(&h, hb)); CK(cudaMalloc(&h2, hb)); CK(cudaMalloc(&nn, T * D * 2)); CK(cudaMalloc(&pre, T * S * 4)); CK(cudaMalloc(&post, T * S * 4)); CK(cudaMalloc(&comb, T * S * S * 4));
        void* scr; size_t scrb = (size_t)T * 1536 * (m + 1) * 4 + 4096; CK(cudaMalloc(&scr, scrb)); ik_scratch sk; memset(&sk, 0, sizeof sk); sk.data = scr; sk.bytes = scrb;
        ik_tensor tx = mk(x, IK_DTYPE_BF16, {T, S, D}), tao = mk(ao, IK_DTYPE_BF16, {T, D}), tfn = mk(fn, IK_DTYPE_F32, {m, S * D}), tsc = mk(scale, IK_DTYPE_F32, {3}),
                  tb = mk(base, IK_DTYPE_F32, {m}), tpi = mk(pre_in, IK_DTYPE_F32, {T, S}), tnw = mk(nw, IK_DTYPE_BF16, {D}), tpm = mk(pmix, IK_DTYPE_F32, {T, S}),
                  tpc = mk(pcomb, IK_DTYPE_F32, {T, S, S}), th = mk(h, IK_DTYPE_BF16, {T, S, D}), th2 = mk(h2, IK_DTYPE_BF16, {T, S, D}), tn = mk(nn, IK_DTYPE_BF16, {T, D}),
                  tpre = mk(pre, IK_DTYPE_F32, {T, S}), tpost = mk(post, IK_DTYPE_F32, {T, S}), tcomb = mk(comb, IK_DTYPE_F32, {T, S, S});
        ik_tensor e0 = mk(nullptr, IK_DTYPE_BF16, {0}), e1 = mk(nullptr, IK_DTYPE_F32, {0}), e2 = mk(nullptr, IK_DTYPE_BF16, {0});
        auto sep = [&] {  // HcPost then the unfolded op (what prefill runs today)
            ik_hc_post(&tao, &tx, &tpm, &tpc, &th, nullptr);
            ik_hc_sublayer(&th, &tfn, &tsc, &tb, &tpi, &tnw, &e0, &e1, &e1, 1e-6f, 0.f, 20, 1e-6f, &tn, &tpre, &tpost, &tcomb, &e2, &sk, nullptr);
        };
        auto fold = [&] {
            ik_hc_sublayer(&tx, &tfn, &tsc, &tb, &tpi, &tnw, &tao, &tpm, &tpc, 1e-6f, 0.f, 20, 1e-6f, &tn, &tpre, &tpost, &tcomb, &th2, &sk, nullptr);
        };
        auto unf = [&] {  // the unfolded op alone (h = x)
            ik_hc_sublayer(&tx, &tfn, &tsc, &tb, &tpi, &tnw, &e0, &e1, &e1, 1e-6f, 0.f, 20, 1e-6f, &tn, &tpre, &tpost, &tcomb, &e2, &sk, nullptr);
        };
        auto post_only = [&] { ik_hc_post(&tao, &tx, &tpm, &tpc, &th, nullptr); };
        auto hashes = [&](const char* tag, void* hh) {
            CK(cudaDeviceSynchronize());
            printf("hc S=%d D=%-5lld T=%-5lld %-6s h %016llx n %016llx pre %016llx post %016llx comb %016llx\n", S, (long long)D, (long long)T, tag, (unsigned long long)hash_dev(hh, hb),
                   (unsigned long long)hash_dev(nn, T * D * 2), (unsigned long long)hash_dev(pre, T * S * 4), (unsigned long long)hash_dev(post, T * S * 4),
                   (unsigned long long)hash_dev(comb, T * S * S * 4));
        };
        CK(cudaMemset(nn, 0, T * D * 2)); sep(); hashes("sep", h);
        CK(cudaMemset(nn, 0, T * D * 2)); fold(); hashes("fold", h2);
        float t_sep = time_ms(sep, iters), t_fold = time_ms(fold, iters), t_unf = time_ms(unf, iters), t_post = time_ms(post_only, iters);
        printf("hc S=%d D=%-5lld T=%-5lld time ms: sep (post+unfolded) %.4f = post %.4f + unfolded %.4f | fold %.4f\n", S, (long long)D, (long long)T, t_sep, t_post, t_unf, t_fold);
        for (void* p : {x, ao, fn, scale, base, pre_in, nw, pmix, pcomb, h, h2, nn, pre, post, comb, scr}) cudaFree(p);
    }
    return 0;
}
