// Standalone nvcc check of the device-side position launchers and CUDA graph replay (no Rust, no cargo needed).
// It mirrors crates/intarsia-backend-cuda/tests/graph_pos.rs and adds a bitwise comparison of the split-KV decode /
// verify launchers against the pre-change implementation (compiled from an old copy with `-D` symbol renames).
//
//   OLD=<dir holding the old kernels/ tree>   (optional; without it the old-vs-new check is skipped)
//   nvcc -O2 --std=c++17 -arch=sm_121 --expt-relaxed-constexpr -I kernels kernels/cuda_checks/graph_pos_check.cu \
//        kernels/cuda/*.cu [old_splitkv.o] -lcublas -o graph_pos_check
//   old_splitkv.o: nvcc -c -O3 --std=c++17 -arch=sm_121 --expt-relaxed-constexpr -I $OLD \
//        -Dik_attn_decode_bf16_splitkv=old_ik_attn_decode_bf16_splitkv \
//        -Dik_attn_prefill_bf16_splitkv=old_ik_attn_prefill_bf16_splitkv $OLD/kernels/cuda/attention_splitkv.cu \
//        -o old_splitkv.o        (compile with -DHAVE_OLD for this file)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <vector>
#include <functional>
#include "abi/ik_abi.h"
#include "cuda/ik_runtime.h"

extern "C" {
int ik_attn_prefill_bf16(const ik_tensor*, const ik_tensor*, const ik_tensor*, int64_t, float, ik_tensor*, void*);
#ifdef HAVE_OLD
int old_ik_attn_decode_bf16_splitkv(const ik_tensor*, const ik_tensor*, const ik_tensor*, int64_t, float, ik_tensor*,
                                    const ik_scratch*, void*);
int old_ik_attn_prefill_bf16_splitkv(const ik_tensor*, const ik_tensor*, const ik_tensor*, int64_t, float, ik_tensor*,
                                     const ik_scratch*, void*);
#endif
}

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(2);                                                                            \
        }                                                                                       \
    } while (0)
#define IKOK(x)                                                              \
    do {                                                                     \
        int r_ = (x);                                                        \
        if (r_ != 0) {                                                       \
            printf("launcher %s returned %d at %s:%d\n", #x, r_, __FILE__, __LINE__); \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

static int g_fail = 0, g_pass = 0;
static void expect(bool ok, const char* fmt, ...) {
    if (ok) { ++g_pass; return; }
    ++g_fail;
    va_list ap;
    va_start(ap, fmt);
    printf("FAIL: ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

static uint64_t rng_state = 88172645463325252ull;
static float frand() {  // xorshift, normal-ish via sum of 4 uniforms
    float s = 0;
    for (int i = 0; i < 4; ++i) {
        rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
        s += (float)(rng_state >> 40) / (float)(1 << 24) - 0.5f;
    }
    return s * 1.7f;
}

static std::vector<__nv_bfloat16> rnd(size_t n, float scale) {
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = __float2bfloat16(frand() * scale);
    return v;
}
template <class T> static T* to_dev(const std::vector<T>& h) {
    T* d; CK(cudaMalloc(&d, h.size() * sizeof(T) + 16)); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static void* dev_zero(size_t bytes) { void* d; CK(cudaMalloc(&d, bytes + 16)); CK(cudaMemset(d, 0, bytes + 16)); return d; }
static std::vector<uint8_t> fetch(const void* d, size_t n) {
    std::vector<uint8_t> h(n); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return h;
}
static ik_tensor mk(void* p, int dtype, std::vector<int64_t> shape) {
    ik_tensor t; memset(&t, 0, sizeof t);
    t.data = p; t.dtype = dtype; t.ndim = (int)shape.size();
    int64_t acc = 1;
    for (int i = 0; i < IK_MAX_DIMS; ++i) { t.shape[i] = 1; t.stride[i] = 0; }
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = acc; acc *= shape[i]; }
    return t;
}
static int32_t* dev_pos(int p) { int32_t v = p; int32_t* d; CK(cudaMalloc(&d, 16)); CK(cudaMemcpy(d, &v, 4, cudaMemcpyHostToDevice)); return d; }
static void set_pos(int32_t* d, int p) { int32_t v = p; CK(cudaMemcpy(d, &v, 4, cudaMemcpyHostToDevice)); }

static int nsplit_max(int nkv, int rows) {
    int by_fill = (256 + nkv * rows - 1) / (nkv * rows);
    return by_fill > 64 ? 64 : (by_fill < 1 ? 1 : by_fill);
}
static ik_scratch mk_scratch(int T, int N, int D, int nkv) {
    int64_t bytes = (int64_t)T * N * nsplit_max(nkv, T) * (D + 2) * 4;
    ik_scratch s; s.data = dev_zero(bytes); s.bytes = bytes; return s;
}

static const int kPositions[] = {0, 1, 2, 31, 62, 63, 64, 65, 127, 128, 129, 255, 1000, 2047, 2048, 4090};

static void test_rope_kv() {
    for (auto cfg : std::vector<std::vector<int>>{{16, 256, 64}, {8, 128, 0}, {6, 64, 0}}) {
        int n = cfg[0], d = cfg[1], rot = cfg[2];
        auto x = rnd((size_t)n * d, 1.f);
        __nv_bfloat16* dx = to_dev(x);
        for (int p : kPositions) {
            auto run = [&](bool dev) {
                void* out = dev_zero((size_t)n * d * 2);
                ik_tensor tx = mk(dx, IK_DTYPE_BF16, {1, n, d}), to = mk(out, IK_DTYPE_BF16, {1, n, d});
                int32_t* dp = dev_pos(p);
                ik_tensor tp = mk(dp, IK_DTYPE_I32, {1});
                IKOK(ik_rope_bf16(&tx, dev ? 0 : p, 1e6f, rot, &to, dev ? &tp : nullptr, nullptr));
                auto h = fetch(out, (size_t)n * d * 2);
                cudaFree(out); cudaFree(dp);
                return h;
            };
            expect(run(false) == run(true), "rope n=%d d=%d rot=%d p=%d: device position differs", n, d, rot, p);
        }
    }
    for (auto cfg : std::vector<std::vector<int>>{{2, 256}, {8, 128}}) {
        int nkv = cfg[0], d = cfg[1], cap = 4096 + 8;
        auto nw = rnd((size_t)nkv * d, 1.f), c0 = rnd((size_t)nkv * cap * d, 1.f);
        __nv_bfloat16* dn = to_dev(nw);
        for (int p : kPositions) {
            auto run = [&](bool dev) {
                __nv_bfloat16* cache = to_dev(c0);
                ik_tensor tn = mk(dn, IK_DTYPE_BF16, {1, nkv, d}), tc = mk(cache, IK_DTYPE_BF16, {nkv, cap, d});
                int32_t* dp = dev_pos(p);
                ik_tensor tp = mk(dp, IK_DTYPE_I32, {1});
                IKOK(ik_kv_append_bf16(&tn, dev ? 0 : p, &tc, dev ? &tp : nullptr, nullptr));
                auto h = fetch(cache, c0.size() * 2);
                cudaFree(cache); cudaFree(dp);
                return h;
            };
            expect(run(false) == run(true), "kv_append nkv=%d d=%d p=%d: device position differs", nkv, d, p);
            // the row really landed at p
            __nv_bfloat16* cache = to_dev(c0);
            ik_tensor tn = mk(dn, IK_DTYPE_BF16, {1, nkv, d}), tc = mk(cache, IK_DTYPE_BF16, {nkv, cap, d});
            int32_t* dp = dev_pos(p);
            ik_tensor tp = mk(dp, IK_DTYPE_I32, {1});
            IKOK(ik_kv_append_bf16(&tn, 0, &tc, &tp, nullptr));
            auto h = fetch(cache, c0.size() * 2);
            bool landed = memcmp(&h[((size_t)(nkv - 1) * cap + p) * d * 2], &nw[(size_t)(nkv - 1) * d], d * 2) == 0;
            expect(landed, "kv_append nkv=%d d=%d: row %d not written by the device position", nkv, d, p);
            cudaFree(cache); cudaFree(dp);
        }
    }
}

struct AttnCase { int n, nkv, d; };

static void test_attn_host_vs_device() {
    for (AttnCase c : {AttnCase{16, 2, 256}, AttnCase{32, 8, 128}, AttnCase{10, 2, 64}}) {
        int cap = 4096 + 8;
        auto q = rnd((size_t)c.n * c.d, 2.f), kc = rnd((size_t)c.nkv * cap * c.d, 1.f), vc = rnd((size_t)c.nkv * cap * c.d, 1.f);
        __nv_bfloat16 *dq = to_dev(q), *dk = to_dev(kc), *dv = to_dev(vc);
        float scale = 1.f / sqrtf((float)c.d);
        ik_scratch sc = mk_scratch(1, c.n, c.d, c.nkv);
        for (int split = 0; split < 2; ++split) {
            for (int p : kPositions) {
                auto run = [&](bool dev) {
                    void* out = dev_zero((size_t)c.n * c.d * 2);
                    ik_tensor tq = mk(dq, IK_DTYPE_BF16, {1, c.n, c.d}), tk = mk(dk, IK_DTYPE_BF16, {c.nkv, cap, c.d}),
                              tv = mk(dv, IK_DTYPE_BF16, {c.nkv, cap, c.d}), to = mk(out, IK_DTYPE_BF16, {1, c.n, c.d});
                    int32_t* dp = dev_pos(p);
                    ik_tensor tp = mk(dp, IK_DTYPE_I32, {1});
                    const ik_tensor* pp = dev ? &tp : nullptr;
                    int64_t p0 = dev ? 0 : p;
                    if (split) IKOK(ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, p0, scale, &to, &sc, pp, nullptr));
                    else IKOK(ik_attn_decode_bf16(&tq, &tk, &tv, p0, scale, &to, pp, nullptr));
                    auto h = fetch(out, (size_t)c.n * c.d * 2);
                    cudaFree(out); cudaFree(dp);
                    return h;
                };
                expect(run(false) == run(true), "%s n=%d nkv=%d d=%d p=%d: device position differs",
                       split ? "attn_decode_splitkv" : "attn_decode", c.n, c.nkv, c.d, p);
            }
        }
        // split-KV against the SIMT kernel (different algorithm: tolerance)
        for (int p : kPositions) {
            void *o1 = dev_zero((size_t)c.n * c.d * 2), *o2 = dev_zero((size_t)c.n * c.d * 2);
            ik_tensor tq = mk(dq, IK_DTYPE_BF16, {1, c.n, c.d}), tk = mk(dk, IK_DTYPE_BF16, {c.nkv, cap, c.d}),
                      tv = mk(dv, IK_DTYPE_BF16, {c.nkv, cap, c.d}), t1 = mk(o1, IK_DTYPE_BF16, {1, c.n, c.d}),
                      t2 = mk(o2, IK_DTYPE_BF16, {1, c.n, c.d});
            IKOK(ik_attn_decode_bf16(&tq, &tk, &tv, p, scale, &t1, nullptr, nullptr));
            IKOK(ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, p, scale, &t2, &sc, nullptr, nullptr));
            auto a = fetch(o1, (size_t)c.n * c.d * 2), b = fetch(o2, (size_t)c.n * c.d * 2);
            float mx = 0;
            for (size_t i = 0; i < (size_t)c.n * c.d; ++i)
                mx = fmaxf(mx, fabsf(__bfloat162float(((__nv_bfloat16*)a.data())[i]) - __bfloat162float(((__nv_bfloat16*)b.data())[i])));
            expect(mx < 2e-2f, "splitkv vs SIMT n=%d nkv=%d d=%d p=%d: max diff %g", c.n, c.nkv, c.d, p, mx);
            cudaFree(o1); cudaFree(o2);
        }
    }
}

#ifdef HAVE_OLD
// the max-grid / in-kernel split count must reproduce the old host-sized launch bit for bit
static void test_splitkv_old_vs_new() {
    int checked = 0;
    for (AttnCase c : {AttnCase{16, 2, 256}, AttnCase{32, 8, 128}, AttnCase{6, 2, 32}, AttnCase{10, 2, 64}}) {
        int cap = 4096 + 40;
        auto kc = rnd((size_t)c.nkv * cap * c.d, 1.f), vc = rnd((size_t)c.nkv * cap * c.d, 1.f);
        __nv_bfloat16 *dk = to_dev(kc), *dv = to_dev(vc);
        float scale = 1.f / sqrtf((float)c.d);
        for (int T : {1, 2, 5, 16}) {
            auto q = rnd((size_t)T * c.n * c.d, 2.f);
            __nv_bfloat16* dq = to_dev(q);
            ik_scratch sc = mk_scratch(T, c.n, c.d, c.nkv);
            for (int p = 0; p < 4090; p += (p < 300 ? 1 : 97)) {
                if (p + T > cap) break;
                void *o1 = dev_zero((size_t)T * c.n * c.d * 2), *o2 = dev_zero((size_t)T * c.n * c.d * 2);
                ik_tensor tq = mk(dq, IK_DTYPE_BF16, {T, c.n, c.d}), tk = mk(dk, IK_DTYPE_BF16, {c.nkv, cap, c.d}),
                          tv = mk(dv, IK_DTYPE_BF16, {c.nkv, cap, c.d}), t1 = mk(o1, IK_DTYPE_BF16, {T, c.n, c.d}),
                          t2 = mk(o2, IK_DTYPE_BF16, {T, c.n, c.d});
                if (T == 1) {
                    IKOK(old_ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, p, scale, &t1, &sc, nullptr));
                    IKOK(ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, p, scale, &t2, &sc, nullptr, nullptr));
                } else {
                    IKOK(old_ik_attn_prefill_bf16_splitkv(&tq, &tk, &tv, p, scale, &t1, &sc, nullptr));
                    IKOK(ik_attn_prefill_bf16_splitkv(&tq, &tk, &tv, p, scale, &t2, &sc, nullptr));
                }
                expect(fetch(o1, (size_t)T * c.n * c.d * 2) == fetch(o2, (size_t)T * c.n * c.d * 2),
                       "splitkv old vs new differ: n=%d nkv=%d d=%d T=%d p=%d", c.n, c.nkv, c.d, T, p);
                ++checked;
                cudaFree(o1); cudaFree(o2);
            }
            cudaFree(dq);
        }
        cudaFree(dk); cudaFree(dv);
    }
    printf("splitkv old-vs-new: %d launches compared bitwise\n", checked);
}
#endif

static void test_graph_replay() {
    int n = 16, nkv = 2, d = 256, cap = 4096 + 8;
    auto q = rnd((size_t)n * d, 2.f), kn = rnd((size_t)nkv * d, 1.f), vn = rnd((size_t)nkv * d, 1.f);
    auto kc = rnd((size_t)nkv * cap * d, 1.f), vc = rnd((size_t)nkv * cap * d, 1.f);
    __nv_bfloat16 *dq = to_dev(q), *dkn = to_dev(kn), *dvn = to_dev(vn), *dk = to_dev(kc), *dv = to_dev(vc);
    void *out = dev_zero((size_t)n * d * 2), *out_ref = dev_zero((size_t)n * d * 2);
    ik_scratch sc = mk_scratch(1, n, d, nkv);
    int32_t* dp = dev_pos(0);
    ik_tensor tq = mk(dq, IK_DTYPE_BF16, {1, n, d}), tkn = mk(dkn, IK_DTYPE_BF16, {1, nkv, d}), tvn = mk(dvn, IK_DTYPE_BF16, {1, nkv, d}),
              tk = mk(dk, IK_DTYPE_BF16, {nkv, cap, d}), tv = mk(dv, IK_DTYPE_BF16, {nkv, cap, d}),
              to = mk(out, IK_DTYPE_BF16, {1, n, d}), tr = mk(out_ref, IK_DTYPE_BF16, {1, n, d}), tp = mk(dp, IK_DTYPE_I32, {1});
    float scale = 1.f / sqrtf((float)d);
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    IKOK(ik_rt_graph_begin(s));
    IKOK(ik_kv_append_bf16(&tkn, 0, &tk, &tp, s));
    IKOK(ik_kv_append_bf16(&tvn, 0, &tv, &tp, s));
    IKOK(ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, 0, scale, &to, &sc, &tp, s));
    void* exec = nullptr;
    IKOK(ik_rt_graph_end(s, &exec));
    expect(exec != nullptr, "graph exec");
    // capture executes nothing
    CK(cudaStreamSynchronize(s));
    for (int p : kPositions) {
        set_pos(dp, p);
        IKOK(ik_rt_graph_launch(exec, s));
        CK(cudaStreamSynchronize(s));
        auto got = fetch(out, (size_t)n * d * 2);
        IKOK(ik_kv_append_bf16(&tkn, p, &tk, nullptr, s));
        IKOK(ik_kv_append_bf16(&tvn, p, &tv, nullptr, s));
        IKOK(ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, p, scale, &tr, &sc, nullptr, s));
        CK(cudaStreamSynchronize(s));
        expect(got == fetch(out_ref, (size_t)n * d * 2), "graph replay at p=%d differs from eager", p);
    }
    set_pos(dp, 3);  // backwards
    IKOK(ik_rt_graph_launch(exec, s)); CK(cudaStreamSynchronize(s));
    auto got = fetch(out, (size_t)n * d * 2);
    IKOK(ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, 3, scale, &tr, &sc, nullptr, s)); CK(cudaStreamSynchronize(s));
    expect(got == fetch(out_ref, (size_t)n * d * 2), "graph replay backwards to p=3");
    // replay speed vs eager launches of the same three kernels
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    set_pos(dp, 2000);
    const int reps = 200;
    cudaEventRecord(e0, s);
    for (int i = 0; i < reps; ++i) ik_rt_graph_launch(exec, s);
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    float g_ms; cudaEventElapsedTime(&g_ms, e0, e1);
    cudaEventRecord(e0, s);
    for (int i = 0; i < reps; ++i) {
        ik_kv_append_bf16(&tkn, 0, &tk, &tp, s); ik_kv_append_bf16(&tvn, 0, &tv, &tp, s);
        ik_attn_decode_bf16_splitkv(&tq, &tk, &tv, 0, scale, &to, &sc, &tp, s);
    }
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    float e_ms; cudaEventElapsedTime(&e_ms, e0, e1);
    printf("graph replay %.1f us/step vs eager launches %.1f us/step (3 kernels, p=2000)\n", g_ms * 1e3 / reps, e_ms * 1e3 / reps);
    IKOK(ik_rt_graph_destroy(exec));

    // a capture that is invalidated (synchronous call inside) fails cleanly and leaves the stream usable
    IKOK(ik_rt_graph_begin(s));
    cudaError_t bad = cudaStreamSynchronize(s);
    expect(bad != cudaSuccess, "synchronize during capture should fail");
    void* ex2 = nullptr;
    int rc = ik_rt_graph_end(s, &ex2);
    expect(rc != 0 && ex2 == nullptr, "invalidated capture must report an error (rc=%d)", rc);
    IKOK(ik_kv_append_bf16(&tkn, 5, &tk, nullptr, s));
    CK(cudaStreamSynchronize(s));
    expect(cudaGetLastError() == cudaSuccess, "stream usable after a failed capture");
}

int main() {
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s sm_%d%d\n", pr.name, pr.major, pr.minor);
    test_rope_kv();
    test_attn_host_vs_device();
#ifdef HAVE_OLD
    test_splitkv_old_vs_new();
#else
    printf("splitkv old-vs-new: SKIPPED (built without HAVE_OLD)\n");
#endif
    test_graph_replay();
    printf("%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
