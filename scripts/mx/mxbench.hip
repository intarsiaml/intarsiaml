// Lane mx: isolated timing of the MXFP4 prefill GEMMs at the 27B's shapes on one R9700: matmul_mxfp4_mma v1 (main) vs v2
// (fast dequant) with its tile table and every tile of the override list, W4A4, bf16 out, each checked bitwise against v1.
// Weights: 4 rotating copies per shape so the timed loop does not run out of the Infinity Cache.
// Build (dev image, repo root): hipcc -O3 --offload-arch=gfx1201 -DIK_MX_BENCH -I kernels/hip -I kernels/abi scripts/mx/mxbench.hip \
//   kernels/hip/mxfp4.hip kernels/hip/ik_runtime.hip -o mx-out/mxbench
// Run: mxbench [M=4096] [reps=20]
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "ik_hip_abi.h"
extern "C" void ik_mxfp4_bench_set(int wn, int wm, int tn, int tm, int v1);

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
    void *dx, *dout, *dscr, *dact;
    void *dw[NC], *ds[NC];
    CK(hipMalloc(&dx, maxx));
    CK(hipMalloc(&dout, maxo));
    CK(hipMalloc(&dscr, maxx + M + 32768 + 16));
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
    const int cfgs[][5] = {{0, 0, 0, 0, 1}, {0, 0, 0, 0, 0}, {4, 2, 2, 4, 0}, {4, 4, 4, 2, 0}, {4, 4, 2, 4, 0}, {2, 2, 4, 4, 0},
                           {4, 2, 4, 4, 0}, {2, 4, 4, 4, 0}, {4, 4, 2, 2, 0}, {2, 4, 4, 2, 0}, {2, 2, 2, 2, 0}, {2, 2, 4, 2, 0},
                           {1, 4, 1, 1, 0}, {1, 1, 1, 1, 0}, {1, 4, 1, 2, 0}, {0, 0, 0, 0, 2}};
    const int NCF = sizeof(cfgs) / sizeof(cfgs[0]);
    printf("M = %d, W4A4, bf16 out, %d reps over %d weight copies; TFLOP/s per config, '!' = differs from v1 (expected for fold only)\n", M, reps, NC);
    printf("%-18s %6s %6s", "shape", "N", "K");
    for (int c = 0; c < NCF; ++c) {
        char b[32];
        if (cfgs[c][4] == 2) snprintf(b, sizeof b, "fold");
        else if (cfgs[c][4]) snprintf(b, sizeof b, "v1");
        else if (!cfgs[c][0]) snprintf(b, sizeof b, "v2-table");
        else snprintf(b, sizeof b, "v2-%d%d%d%d", cfgs[c][0], cfgs[c][1], cfgs[c][2], cfgs[c][3]);
        printf(" %10s", b);
    }
    printf("\n");
    std::vector<double> tot(NCF, 0.0);
    double fl_all = 0;
    std::vector<uint16_t> ref, cur;
    for (auto& s : shapes) {
        ik_tensor x = mk(dx, IK_DTYPE_BF16, M, s.K), act = mk(dact, IK_DTYPE_F32, 1, 0);
        ik_tensor o = mk(dout, IK_DTYPE_BF16, M, s.N);
        ik_scratch sc = {dscr, (int64_t)M * s.K * 2 + M + s.N + 16};
        const double fl = 2.0 * M * s.N * s.K;
        int mult = (!strcmp(s.what, "mlp gate / up") || !strcmp(s.what, "attn k / v") || !strcmp(s.what, "gdn in_proj_a / b")) ? 2 : 1;
        int layers = !strncmp(s.what, "gdn", 3) ? 48 : (!strncmp(s.what, "attn", 4) ? 16 : 64);
        if (!strcmp(s.what, "gdn out / attn o")) layers = 64;
        printf("%-18s %6d %6d", s.what, s.N, s.K);
        for (int c = 0; c < NCF; ++c) {
            ik_mxfp4_bench_set(cfgs[c][0], cfgs[c][1], cfgs[c][2], cfgs[c][3], cfgs[c][4] == 1);
            const bool fold = cfgs[c][4] == 2;
            auto call = [&](int cp) {
                ik_tensor w = mk(dw[cp], IK_DTYPE_U8, s.N, s.K / 2), sv = mk(ds[cp], IK_DTYPE_U8, s.N, s.K / 32);
                int r = fold ? ik_matmul_mxfp4_mma_fold(&x, &w, &sv, &act, &o, &sc, nullptr)
                             : ik_matmul_mxfp4_mma(&x, &w, &sv, &act, &o, &sc, nullptr);
                if (r) {
                    fprintf(stderr, "launch failed: %d\n", r);
                    exit(1);
                }
            };
            call(0);
            CK(hipDeviceSynchronize());
            cur.resize((size_t)M * s.N);
            CK(hipMemcpy(cur.data(), dout, cur.size() * 2, hipMemcpyDeviceToHost));
            if (c == 0) ref = cur;
            size_t bad = 0;
            for (size_t i = 0; i < cur.size(); ++i) bad += cur[i] != ref[i];
            float ms;
            CK(hipEventRecord(e0, nullptr));
            for (int i = 0; i < reps; ++i) call(i % NC);
            CK(hipEventRecord(e1, nullptr));
            CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms, e0, e1));
            ms /= reps;
            printf(" %9.1f%s", fl / ms / 1e9, bad ? "!" : " ");
            tot[c] += (double)ms * mult * layers;
            if (bad) fprintf(stderr, "%s cfg %d: %zu outputs differ from v1\n", s.what, c, bad);
        }
        printf("\n");
        fl_all += fl * mult * layers;
    }
    printf("whole-model GEMM ms per %d-token chunk:", M);
    for (int c = 0; c < NCF; ++c) printf(" %10.1f", tot[c]);
    printf("\nwhole-model TFLOP/s:                       ");
    for (int c = 0; c < NCF; ++c) printf(" %10.1f", fl_all / tot[c] / 1e9);
    printf("\n");
    return 0;
}
