// Lane mx: isolated timing of the MXFP4 prefill GEMMs at the 27B's shapes on one R9700: matmul_mxfp4_mma (bf16-dequant
// WMMA, today) vs matmul_mxfp4_mma_f8 (fp8 WMMA, lane mx), W4A4, bf16 out, plus a bitwise check of the two outputs.
// Weights: 4 rotating copies per shape so the timed loop does not run out of the Infinity Cache.
// Build (dev image, repo root): hipcc -O3 --offload-arch=gfx1201 -I kernels/hip -I kernels/abi scripts/mx/mxbench.hip \
//   kernels/hip/mxfp4.hip kernels/hip/ik_runtime.hip -o mx-out/mxbench
// Run: mxbench [M=4096] [reps=20]
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "ik_hip_abi.h"

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static ik_tensor mk(void* p, int dt, int64_t a, int64_t b) {
    ik_tensor t;
    memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = dt;
    t.ndim = b ? 2 : 1;
    t.shape[0] = a;
    t.shape[1] = b;
    if (b) {
        t.stride[0] = b;
        t.stride[1] = 1;
    } else {
        t.stride[0] = 1;
    }
    return t;
}

static uint32_t rs = 1234567u;
static uint32_t rnd() {
    rs ^= rs << 13;
    rs ^= rs >> 17;
    rs ^= rs << 5;
    return rs;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 20;
    struct S { const char* what; int N, K; } shapes[] = {
        {"mlp gate / up", 17408, 5120}, {"mlp down", 5120, 17408}, {"gdn in_proj_qkv", 10240, 5120},
        {"gdn in_proj_z", 6144, 5120},  {"gdn out / attn o", 5120, 6144}, {"attn q + gate", 12288, 5120},
        {"attn k / v", 1024, 5120},     {"gdn in_proj_a / b", 48, 5120},
    };
    const int NC = 4;
    size_t maxx = 0, maxw = 0, maxo = 0;
    for (auto& s : shapes) {
        maxx = std::max(maxx, (size_t)M * s.K * 2);
        maxw = std::max(maxw, (size_t)s.N * s.K / 2);
        maxo = std::max(maxo, (size_t)M * s.N * 2);
    }
    // activations: bf16 normal-ish with some outlier groups
    std::vector<uint16_t> hx(maxx / 2);
    for (size_t i = 0; i < hx.size(); ++i) {
        float v = ((int)(rnd() % 2001) - 1000) / 500.f;
        if ((i / 32) % 17 == 3) v *= 64.f;
        uint32_t u;
        memcpy(&u, &v, 4);
        hx[i] = (uint16_t)(u >> 16);
    }
    std::vector<uint8_t> hw(maxw), hs(maxw / 16);
    for (auto& b : hw) b = (uint8_t)rnd();
    for (auto& b : hs) b = (uint8_t)(118 + rnd() % 4);
    void *dx, *dout, *dout2, *dscr, *dact;
    void *dw[NC], *ds[NC];
    CK(hipMalloc(&dx, maxx));
    CK(hipMalloc(&dout, maxo));
    CK(hipMalloc(&dout2, maxo));
    CK(hipMalloc(&dscr, maxx + 16));
    CK(hipMalloc(&dact, 4));
    for (int c = 0; c < NC; ++c) {
        CK(hipMalloc(&dw[c], maxw));
        CK(hipMalloc(&ds[c], maxw / 16));
        CK(hipMemcpy(dw[c], hw.data(), maxw, hipMemcpyHostToDevice));
        CK(hipMemcpy(ds[c], hs.data(), maxw / 16, hipMemcpyHostToDevice));
    }
    CK(hipMemcpy(dx, hx.data(), maxx, hipMemcpyHostToDevice));
    const float one = 1.f;
    CK(hipMemcpy(dact, &one, 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    printf("M = %d, W4A4, bf16 out, %d reps over %d weight copies\n", M, reps, NC);
    printf("%-20s %6s %6s %12s %12s %8s %s\n", "shape", "N", "K", "mma TFLOP/s", "f8 TFLOP/s", "speedup", "bitwise");
    double tm = 0, tf = 0, fl_all = 0;
    for (auto& s : shapes) {
        ik_tensor x = mk(dx, IK_DTYPE_BF16, M, s.K), act = mk(dact, IK_DTYPE_F32, 1, 0);
        ik_tensor o = mk(dout, IK_DTYPE_BF16, M, s.N), o2 = mk(dout2, IK_DTYPE_BF16, M, s.N);
        ik_scratch sc = {dscr, (int64_t)M * s.K * 2 + 16};
        double fl = 2.0 * M * s.N * s.K;
        float ms[2];
        for (int which = 0; which < 2; ++which) {
            auto call = [&](int c, ik_tensor* out) {
                ik_tensor w = mk(dw[c], IK_DTYPE_U8, s.N, s.K / 2), sv = mk(ds[c], IK_DTYPE_U8, s.N, s.K / 32);
                int r = which ? ik_matmul_mxfp4_mma_f8(&x, &w, &sv, &act, out, &sc, nullptr)
                              : ik_matmul_mxfp4_mma(&x, &w, &sv, &act, out, &sc, nullptr);
                if (r) {
                    fprintf(stderr, "launch %d failed: %d\n", which, r);
                    exit(1);
                }
            };
            call(0, which ? &o2 : &o);
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(e0, nullptr));
            for (int i = 0; i < reps; ++i) call(i % NC, which ? &o2 : &o);
            CK(hipEventRecord(e1, nullptr));
            CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms[which], e0, e1));
            ms[which] /= reps;
        }
        // outputs of the last calls used copy (reps - 1) % NC for both: compare
        std::vector<uint16_t> a((size_t)M * s.N), b((size_t)M * s.N);
        CK(hipMemcpy(a.data(), dout, a.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), dout2, b.size() * 2, hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
        printf("%-20s %6d %6d %12.1f %12.1f %7.2fx %s (%zu differ)\n", s.what, s.N, s.K, fl / ms[0] / 1e9, fl / ms[1] / 1e9,
               ms[0] / ms[1], bad ? "DIFF" : "same", bad);
        int mult = (!strcmp(s.what, "mlp gate / up") || !strcmp(s.what, "attn k / v") || !strcmp(s.what, "gdn in_proj_a / b")) ? 2 : 1;
        int layers = (!strncmp(s.what, "gdn", 3)) ? 48 : (!strncmp(s.what, "attn", 4) ? 16 : 64);
        if (!strcmp(s.what, "gdn out / attn o")) layers = 64;
        tm += ms[0] * mult * layers;
        tf += ms[1] * mult * layers;
        fl_all += fl * mult * layers;
    }
    printf("whole-model GEMM time per %d-token chunk: mma %.1f ms (%.1f TFLOP/s), f8 %.1f ms (%.1f TFLOP/s)\n", M, tm,
           fl_all / tm / 1e9, tf, fl_all / tf / 1e9);
    return 0;
}
