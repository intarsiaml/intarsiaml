// Lane l8: standalone microbenchmark of candidate fp8 x fp8 WMMA GEMMs against the current f8_gemm_run
// (kernels/hip/ik_f8_gemm.h). Every candidate output is compared BITWISE with the reference's (per-output k order must not
// change); timing is best of 3 bursts of REPS launches (hipEvent) at the 35B's dense fp8 projection shapes.
// Build: scripts/l8/build.sh f8bench   Run: .l8bin/f8bench [shape-filter-substring] [variant-filter-substring]
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "ik_f8_gemm.h"
#include "f8v.h"  // candidates: L8_VARIANTS(X) with X(name, launch_fn)

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

typedef void (*Launch)(const uint8_t*, const uint8_t*, const float*, ik_bf16*, int, int64_t, int, hipStream_t);

static void launch_ref(const uint8_t* xq, const uint8_t* w, const float* sc, ik_bf16* out, int M, int64_t N, int K,
                       hipStream_t s) {
    f8_gemm_run<ik_bf16>(xq, w, sc, out, M, N, K, 1, s);
}

struct V { const char* name; Launch f; };
#define X(name, fn) {name, fn},
static const V variants[] = {
    {"ref", launch_ref},
    L8_VARIANTS(X)
};
#undef X

int main(int argc, char** argv) {
    const char* sf = argc > 1 ? argv[1] : "";
    const char* vf = argc > 2 ? argv[2] : "";
    struct S { const char* what; int M, N, K; } shapes[] = {
        {"qkv M4096", 4096, 8192, 2048}, {"z M4096", 4096, 4096, 2048}, {"o M4096", 4096, 2048, 4096},
        {"kv M4096", 4096, 512, 2048},
        // partial M and N (edge handling of every tile), bitwise vs ref
        {"qkv M2049", 2049, 8192, 2048}, {"qkv M4089", 4089, 8192, 2048}, {"z M3000", 3000, 4096, 2048},
        {"o M4089", 4089, 2048, 4096}, {"kv M2049", 2049, 512, 2048}, {"o M300", 300, 2048, 4096},
        {"odd N 1000", 2100, 1000, 2048}, {"K 1056", 2100, 1280, 1056},
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
    CK(hipMalloc(&w, maxw + 4096));
    CK(hipMalloc(&x, maxx + 4096));
    CK(hipMalloc(&y0, maxy));
    CK(hipMalloc(&y1, maxy));
    CK(hipMalloc(&sc, 8));
    {
        std::vector<uint8_t> h(std::max(maxw, maxx) + 64);
        uint64_t r = 0x9e3779b97f4a7c15ull;
        for (auto& b : h) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            b = (uint8_t)(r >> 24);
            if ((b & 0x7f) == 0x7f) b ^= 1;    // no NaN
            if ((b & 0x78) > 0x40) b &= 0xc7;  // magnitudes below 2^1 or so (no bf16 overflow)
        }
        CK(hipMemcpy(w, h.data(), maxw, hipMemcpyHostToDevice));
        CK(hipMemcpy(x, h.data() + 7, maxx, hipMemcpyHostToDevice));
        float s2[2] = {0.01f, 0.5f};
        CK(hipMemcpy(sc, s2, 8, hipMemcpyHostToDevice));
    }
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    std::vector<uint16_t> ref(maxy / 2), got(maxy / 2);
    int bad = 0;
    for (auto& s : shapes) {
        if (!strstr(s.what, sf)) continue;
        printf("\n%s M=%d N=%d K=%d\n", s.what, s.M, s.N, s.K);
        const size_t ob = (size_t)s.M * s.N * 2;
        const bool timed = s.M >= 2048 && s.N >= 512 && !strstr(s.what, "odd") && !strstr(s.what, "K 1056");
        for (const V& v : variants) {
            if (&v != variants && !strstr(v.name, vf)) continue;
            ik_bf16* y = (&v == variants) ? y0 : y1;
            CK(hipMemset(y, 0, ob));
            v.f(x, w, sc, y, s.M, s.N, s.K, 0);
            CK(hipGetLastError());
            CK(hipDeviceSynchronize());
            float best = 1e30f;
            if (timed || &v == variants) {
                const int reps = 30;
                for (int t = 0; t < 3; ++t) {
                    CK(hipEventRecord(e0));
                    for (int r = 0; r < reps; ++r) v.f(x, w, sc, y, s.M, s.N, s.K, 0);
                    CK(hipEventRecord(e1));
                    CK(hipEventSynchronize(e1));
                    float ms;
                    CK(hipEventElapsedTime(&ms, e0, e1));
                    best = fminf(best, ms / reps);
                }
            }
            bool same = true;
            if (&v == variants) {
                CK(hipMemcpy(ref.data(), y0, ob, hipMemcpyDeviceToHost));
            } else {
                CK(hipMemcpy(got.data(), y1, ob, hipMemcpyDeviceToHost));
                same = memcmp(ref.data(), got.data(), ob) == 0;
                if (!same) ++bad;
            }
            printf("  %-18s %8.1f us %7.1f TFLOPS %s\n", v.name, best * 1e3,
                   2.0 * s.M * s.N * s.K / (best * 1e-3) / 1e12, same ? "bitwise" : "DIFFERS");
        }
    }
    printf("\n%s (%d candidate outputs differ from ref)\n", bad ? "FAIL" : "ALL BITWISE", bad);
    return bad ? 1 : 0;
}
