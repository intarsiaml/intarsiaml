// Lane hp: tile sweep of f8_gemm_kernel (kernels/hip/ik_f8_gemm.h) at the 35B's dense fp8 projection shapes. Each tile is
// timed and its output compared bitwise with the default tile's (f8_gemm_run): the k order of the kernel does not depend
// on the tile, so every tile must give the same bits.
// Build (dev image): hipcc -O3 --offload-arch=gfx1201 -I kernels/hip -I kernels f8sweep.hip kernels/hip/ik_runtime.hip -o f8sweep
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "ik_f8_gemm.h"

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

typedef void (*Launch)(const uint8_t*, const uint8_t*, const float*, ik_bf16*, int, int64_t, int, hipStream_t);

template <int WN, int WM, int TN, int TM>
void launch(const uint8_t* xq, const uint8_t* w, const float* sc, ik_bf16* out, int M, int64_t N, int K, hipStream_t s) {
    dim3 grid((unsigned)(((int64_t)M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((N + WN * TN * 16 - 1) / (WN * TN * 16)));
    f8_gemm_kernel<WN, WM, TN, TM, ik_bf16><<<grid, WN * WM * 32, 0, s>>>(xq, w, sc, out, M, N, K, 1);
}
void launch_default(const uint8_t* xq, const uint8_t* w, const float* sc, ik_bf16* out, int M, int64_t N, int K, hipStream_t s) {
    f8_gemm_run<ik_bf16>(xq, w, sc, out, M, N, K, 1, s);
}

struct V { const char* name; Launch f; };
#define T(a, b, c, d) {#a "," #b "," #c "," #d, launch<a, b, c, d>}
static const V variants[] = {
    {"default", launch_default},
    T(4, 2, 2, 4), T(2, 4, 4, 2), T(2, 2, 4, 4), T(4, 2, 4, 2), T(2, 4, 2, 4),
    T(4, 2, 4, 4), T(2, 4, 4, 4), T(4, 4, 2, 4), T(4, 4, 4, 2), T(4, 4, 2, 2),
    T(2, 2, 4, 8), T(2, 2, 8, 4), T(4, 2, 2, 8), T(2, 4, 8, 2), T(8, 1, 2, 8), T(1, 8, 8, 2),
    T(4, 1, 4, 8), T(1, 4, 8, 4), T(2, 1, 8, 8), T(1, 2, 8, 8),
};

int main() {
    struct S { const char* what; int M, N, K; } shapes[] = {
        {"qkv / q+gate", 4096, 8192, 2048}, {"z", 4096, 4096, 2048}, {"o / out_proj", 4096, 2048, 4096},
        {"k / v", 4096, 512, 2048}, {"tail 312", 312, 8192, 2048},
    };
    size_t maxw = 0, maxx = 0, maxy = 0;
    for (auto& s : shapes) {
        maxw = std::max(maxw, (size_t)s.N * s.K);
        maxx = std::max(maxx, (size_t)s.M * s.K);
        maxy = std::max(maxy, (size_t)s.M * s.N * 2);
    }
    uint8_t *w, *x;
    ik_bf16 *y0, *y1;
    float* sc;
    CK(hipMalloc(&w, maxw));
    CK(hipMalloc(&x, maxx));
    CK(hipMalloc(&y0, maxy));
    CK(hipMalloc(&y1, maxy));
    CK(hipMalloc(&sc, 8));
    {
        std::vector<uint8_t> h(std::max(maxw, maxx));
        uint64_t r = 0x9e3779b97f4a7c15ull;
        for (auto& b : h) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            b = (uint8_t)(r >> 24);
            if ((b & 0x7f) == 0x7f) b ^= 1;  // no NaN
            if ((b & 0x78) > 0x40) b &= 0xc7;  // magnitudes below 2^1 or so (no bf16 overflow)
        }
        CK(hipMemcpy(w, h.data(), maxw, hipMemcpyHostToDevice));
        CK(hipMemcpy(x, h.data() + 7, maxx - 7, hipMemcpyHostToDevice));
        float s2[2] = {0.01f, 0.5f};
        CK(hipMemcpy(sc, s2, 8, hipMemcpyHostToDevice));
    }
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    std::vector<uint16_t> ref(maxy / 2), got(maxy / 2);
    for (auto& s : shapes) {
        printf("\n%s M=%d N=%d K=%d\n", s.what, s.M, s.N, s.K);
        const size_t ob = (size_t)s.M * s.N * 2;
        for (const V& v : variants) {
            ik_bf16* y = (&v == variants) ? y0 : y1;
            CK(hipMemset(y, 0, ob));
            v.f(x, w, sc, y, s.M, s.N, s.K, 0);
            CK(hipGetLastError());
            CK(hipDeviceSynchronize());
            const int reps = 30;
            float best = 1e30f;
            for (int t = 0; t < 3; ++t) {
                CK(hipEventRecord(e0));
                for (int r = 0; r < reps; ++r) v.f(x, w, sc, y, s.M, s.N, s.K, 0);
                CK(hipEventRecord(e1));
                CK(hipEventSynchronize(e1));
                float ms;
                CK(hipEventElapsedTime(&ms, e0, e1));
                best = fminf(best, ms / reps);
            }
            bool same = true;
            if (&v == variants) {
                CK(hipMemcpy(ref.data(), y0, ob, hipMemcpyDeviceToHost));
            } else {
                CK(hipMemcpy(got.data(), y1, ob, hipMemcpyDeviceToHost));
                same = memcmp(ref.data(), got.data(), ob) == 0;
            }
            printf("  %-10s %8.1f us %7.1f TFLOPS %s\n", v.name, best * 1e3, 2.0 * s.M * s.N * s.K / (best * 1e-3) / 1e12,
                   same ? "bitwise" : "DIFFERS");
        }
    }
    return 0;
}
