// Lane wg standalone microbench + correctness of the fp8 WMMA rows family vs the shipping GEMV kernels. Built in the HIP dev
// container (no cargo): hipcc -O3 --offload-arch=gfx1201 -I kernels bench_f8.cpp kernels/hip/fp8.hip ...  (see scripts/wg/bench.sh)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "hip/ik_hip_abi.h"

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

typedef int (*Fn)(const ik_tensor*, const ik_tensor*, const ik_tensor*, ik_tensor*, void*);
extern "C" int ik_matmul_fp8_gemv_v2(const ik_tensor*, const ik_tensor*, const ik_tensor*, ik_tensor*, void*);
extern "C" int ik_matmul_fp8_gemv_rows(const ik_tensor*, const ik_tensor*, const ik_tensor*, ik_tensor*, void*);

static ik_scratch g_scr;
static int wmma_staged(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* o, void* s) { return ik_matmul_fp8_wmma(x, w, sc, o, nullptr, s); }
static int wmma_direct(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* o, void* s) { return ik_matmul_fp8_wmma(x, w, sc, o, &g_scr, s); }
static ik_tensor T(void* p, int dt, int64_t a, int64_t b = 0) {
    ik_tensor t; memset(&t, 0, sizeof t);
    t.data = p; t.dtype = dt; t.ndim = b ? 2 : 1; t.shape[0] = a; t.stride[0] = b ? b : 1;
    if (b) { t.shape[1] = b; t.stride[1] = 1; }
    return t;
}
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16; }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

extern "C" int ik_matmul_bf16_gemv(const ik_tensor*, const ik_tensor*, ik_tensor*, void*);
extern "C" int ik_matmul_bf16_gemv_rows(const ik_tensor*, const ik_tensor*, ik_tensor*, void*);

static void bench_bf16(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Shape { int N, K; } shapes[] = {{256, 2048}, {32, 2048}, {1, 2048}, {4096, 4096}, {1024, 2048}};
    int ms[] = {1, 2, 4, 8, 12, 16};
    for (auto sh : shapes) {
        int N = sh.N, K = sh.K;
        size_t wb = (size_t)N * K * 2;
        int copies = (int)std::max<size_t>(3, std::min<size_t>(256, (size_t)(300 << 20) / wb + 1));
        std::vector<uint16_t> hw((size_t)N * K);
        for (auto& v : hw) v = f2bf(nd(rng) * 0.05f);
        std::vector<uint16_t*> dw(copies);
        for (auto& p : dw) { CK(hipMalloc(&p, wb)); CK(hipMemcpy(p, hw.data(), wb, hipMemcpyHostToDevice)); }
        std::vector<uint16_t> hx((size_t)64 * K);
        for (auto& v : hx) v = f2bf(nd(rng));
        uint16_t *dx, *dout; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        CK(hipMalloc(&dout, (size_t)64 * N * 2));
        typedef int (*F)(const ik_tensor*, const ik_tensor*, ik_tensor*, void*);
        struct KK { const char* name; F f; int maxm; } ks[] = {{"gemv", ik_matmul_bf16_gemv, 4}, {"gemv_rows", ik_matmul_bf16_gemv_rows, 16}, {"wmma", ik_matmul_bf16_wmma, 64}};
        for (int m : ms) {
            printf("bf16 %5dx%-5d M=%-2d     :", N, K, m);
            for (auto& k : ks) {
                if (m > k.maxm) { printf("  %-9s    n/a ", k.name); continue; }
                ik_tensor xt = T(dx, IK_DTYPE_BF16, m, K), ot = T(dout, IK_DTYPE_BF16, m, N);
                std::vector<ik_tensor> wt(copies);
                for (int i = 0; i < copies; ++i) wt[i] = T(dw[i], IK_DTYPE_BF16, N, K);
                int rc = k.f(&xt, &wt[0], &ot, nullptr);
                if (rc) { printf("  %-9s rc=%d ", k.name, rc); continue; }
                CK(hipDeviceSynchronize());
                int rounds = 20;
                hipEvent_t a, b; (void)hipEventCreate(&a); (void)hipEventCreate(&b);
                for (int i = 0; i < copies; ++i) k.f(&xt, &wt[i], &ot, nullptr);
                (void)hipEventRecord(a);
                for (int r = 0; r < rounds; ++r) for (int i = 0; i < copies; ++i) k.f(&xt, &wt[i], &ot, nullptr);
                (void)hipEventRecord(b); (void)hipEventSynchronize(b);
                float ms_ = 0; (void)hipEventElapsedTime(&ms_, a, b);
                double us = ms_ * 1e3 / (rounds * copies);
                printf("  %-9s %7.2f us %6.1f GB/s", k.name, us, wb / us / 1e3);
            }
            printf("\n");
        }
        {   // row invariance + agreement
            int mc = 16;
            std::vector<uint16_t> ref((size_t)mc * N), got((size_t)mc * N);
            ik_tensor xt = T(dx, IK_DTYPE_BF16, mc, K), ot = T(dout, IK_DTYPE_BF16, mc, N), wt = T(dw[0], IK_DTYPE_BF16, N, K);
            ik_matmul_bf16_gemv_rows(&xt, &wt, &ot, nullptr); CK(hipDeviceSynchronize());
            CK(hipMemcpy(ref.data(), dout, ref.size() * 2, hipMemcpyDeviceToHost));
            ik_matmul_bf16_wmma(&xt, &wt, &ot, nullptr); CK(hipDeviceSynchronize());
            CK(hipMemcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost));
            double num = 0, den = 0; for (size_t i = 0; i < ref.size(); ++i) { double d = bf2f(got[i]) - bf2f(ref[i]); num += d * d; den += (double)bf2f(ref[i]) * bf2f(ref[i]); }
            long bad = 0;
            for (int r = 0; r < mc; ++r) {
                ik_tensor x1 = T(dx + (size_t)r * K, IK_DTYPE_BF16, 1, K), o1 = T(dout, IK_DTYPE_BF16, 1, N);
                ik_matmul_bf16_wmma(&x1, &wt, &o1, nullptr); CK(hipDeviceSynchronize());
                std::vector<uint16_t> one(N); CK(hipMemcpy(one.data(), dout, N * 2, hipMemcpyDeviceToHost));
                for (int n = 0; n < N; ++n) bad += one[n] != got[(size_t)r * N + n];
            }
            printf("  check: rel_l2(wmma vs gemv_rows)=%.3e  rows16-vs-rows1 mismatches=%ld\n", std::sqrt(num / den), bad);
        }
        for (auto p : dw) (void)hipFree(p);
        (void)hipFree(dx); (void)hipFree(dout);
    }
}

int main(int argc, char** argv) {
    setenv("INTARSIA_HIP_WMMA_DIRECT_MIN", "1", 1);
    int dev = getenv("DEV") ? atoi(getenv("DEV")) : 0;
    CK(hipSetDevice(dev));
    bool w8a16 = getenv("A16") != nullptr;
    if (getenv("BF16")) { std::mt19937 r(5); bench_bf16(r); return 0; }
    struct Shape { int N, K; } shapes[] = {{2048, 2048}, {2048, 4096}, {4096, 2048}, {8192, 2048}, {512, 2048}, {12288, 2048}, {1024, 2048}};
    int ms[] = {1, 2, 4, 8, 12, 16};
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    { void* p; CK(hipMalloc(&p, 64 << 20)); g_scr.data = p; g_scr.bytes = 64 << 20; }
    for (auto sh : shapes) {
        int N = sh.N, K = sh.K;
        size_t wb = (size_t)N * K;
        int copies = (int)std::max<size_t>(3, std::min<size_t>(64, (size_t)(300 << 20) / wb + 1));
        std::vector<uint8_t> hw(wb);
        for (auto& b : hw) { b = (uint8_t)(rng() & 0xff); if ((b & 0x7f) == 0x7f) b &= 0xf7; if (((b >> 3) & 15) > 11) b = (b & 0x87) | (8 << 3); }
        std::vector<uint8_t*> dw(copies);
        for (auto& p : dw) { CK(hipMalloc(&p, wb)); CK(hipMemcpy(p, hw.data(), wb, hipMemcpyHostToDevice)); }
        int MM = 64;
        std::vector<uint16_t> hx((size_t)MM * K);
        for (auto& v : hx) v = f2bf(nd(rng));
        uint16_t* dx; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        float hs[2] = {0.02f, w8a16 ? 0.f : 0.15f};
        float* dsc; CK(hipMalloc(&dsc, 8)); CK(hipMemcpy(dsc, hs, 8, hipMemcpyHostToDevice));
        uint16_t* dout; CK(hipMalloc(&dout, (size_t)MM * N * 2));
        ik_tensor sct = T(dsc, IK_DTYPE_F32, 2);
        struct K_ { const char* name; Fn f; int maxm; } kernels[] = {
            {"gemv_v2", ik_matmul_fp8_gemv_v2, 4}, {"gemv_rows", ik_matmul_fp8_gemv_rows, 16}, {"wmma_s", wmma_staged, 64}, {"wmma_d", wmma_direct, 64}};
        for (int m : ms) {
            printf("fp8 %5dx%-5d M=%-2d %s:", N, K, m, w8a16 ? "A16" : "A8 ");
            for (auto& k : kernels) {
                if (m > k.maxm) { printf("  %-9s    n/a ", k.name); continue; }
                ik_tensor xt = T(dx, IK_DTYPE_BF16, m, K), ot = T(dout, IK_DTYPE_BF16, m, N);
                std::vector<ik_tensor> wt(copies);
                for (int i = 0; i < copies; ++i) wt[i] = T(dw[i], IK_DTYPE_FP8_E4M3, N, K);
                int rc = k.f(&xt, &wt[0], &sct, &ot, nullptr);
                if (rc) { printf("  %-9s rc=%d ", k.name, rc); continue; }
                CK(hipDeviceSynchronize());
                int rounds = std::max(3, 400 / copies * 0 + 20);
                hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
                for (int i = 0; i < copies; ++i) k.f(&xt, &wt[i], &sct, &ot, nullptr);
                hipEventRecord(a);
                for (int r = 0; r < rounds; ++r) for (int i = 0; i < copies; ++i) k.f(&xt, &wt[i], &sct, &ot, nullptr);
                hipEventRecord(b); hipEventSynchronize(b);
                float ms_ = 0; hipEventElapsedTime(&ms_, a, b);
                double us = ms_ * 1e3 / (rounds * copies);
                printf("  %-9s %7.2f us %6.1f GB/s", k.name, us, wb / us / 1e3);
            }
            printf("\n");
        }
        // correctness: wmma vs gemv_rows (different order, expect ~1e-3 rel) and row invariance across M
        {
            int mcheck = 16;
            std::vector<uint16_t> ref((size_t)mcheck * N), got((size_t)mcheck * N);
            ik_tensor xt = T(dx, IK_DTYPE_BF16, mcheck, K), ot = T(dout, IK_DTYPE_BF16, mcheck, N), wt = T(dw[0], IK_DTYPE_FP8_E4M3, N, K);
            ik_matmul_fp8_gemv_rows(&xt, &wt, &sct, &ot, nullptr); CK(hipDeviceSynchronize());
            CK(hipMemcpy(ref.data(), dout, ref.size() * 2, hipMemcpyDeviceToHost));
            wmma_staged(&xt, &wt, &sct, &ot, nullptr); CK(hipDeviceSynchronize());
            CK(hipMemcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost));
            double num = 0, den = 0; for (size_t i = 0; i < ref.size(); ++i) { double d = bf2f(got[i]) - bf2f(ref[i]); num += d * d; den += (double)bf2f(ref[i]) * bf2f(ref[i]); }
            // row invariance: M=1 on each row vs the 16-row call
            long bad = 0;
            for (int r = 0; r < mcheck; ++r) {
                ik_tensor x1 = T(dx + (size_t)r * K, IK_DTYPE_BF16, 1, K), o1 = T(dout, IK_DTYPE_BF16, 1, N);
                wmma_staged(&x1, &wt, &sct, &o1, nullptr); CK(hipDeviceSynchronize());
                std::vector<uint16_t> one(N); CK(hipMemcpy(one.data(), dout, N * 2, hipMemcpyDeviceToHost));
                for (int n = 0; n < N; ++n) bad += one[n] != got[(size_t)r * N + n];
            }
            // staged vs direct: same bits
            std::vector<uint16_t> dgot((size_t)mcheck * N);
            wmma_direct(&xt, &wt, &sct, &ot, nullptr); CK(hipDeviceSynchronize());
            CK(hipMemcpy(dgot.data(), dout, dgot.size() * 2, hipMemcpyDeviceToHost));
            long sd = 0; for (size_t i = 0; i < dgot.size(); ++i) sd += dgot[i] != got[i];
            printf("  check: rel_l2(wmma vs gemv_rows)=%.3e  rows16-vs-rows1 mismatches=%ld  staged-vs-direct mismatches=%ld\n", std::sqrt(num / den), bad, sd);
        }
        for (auto p : dw) hipFree(p);
        hipFree(dx); hipFree(dsc); hipFree(dout);
    }
    return 0;
}
