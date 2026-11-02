// Lane hfp8: standalone microbench + bitwise check of the HIP fp8 row-exact GEMV (kernels/hip/fp8.hip) at M = 1..16 rows.
// Includes fp8.hip itself, so the numbers are the kernels in the tree. Build and run through scripts/hfp8/bench.sh:
//   BENCH_SHAPES="8192x2048,4096x2048,2048x4096,512x2048" BENCH_MS=1,8,12,16 BENCH_MODE=time|check|both ./bench_fp8
// time: us per launch of ik_matmul_fp8_gemv_rows, weights rotated over >= 400 MB of copies (defeats the 64 MB infinity cache).
// check: row m of an M-row call equals ik_matmul_fp8_gemv_v2 on row m alone, bit for bit (W8A8 and W8A16, bf16 and f32 out).
#include "fp8.hip"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(2); } } while (0)

static ik_tensor T(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t{};
    t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1;
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}
static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; size_t a = 0;
    while (a <= s.size()) { size_t b = s.find(',', a); if (b == std::string::npos) b = s.size(); if (b > a) v.push_back(s.substr(a, b - a)); a = b + 1; }
    return v;
}
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

int main() {
    const char* e;
    std::string shapes_s = (e = getenv("BENCH_SHAPES")) ? e : "8192x2048,4096x2048,2048x4096,512x2048";
    std::string ms_s = (e = getenv("BENCH_MS")) ? e : "1,4,8,9,12,13,16";
    std::string mode = (e = getenv("BENCH_MODE")) ? e : "both";
    const int reps = (e = getenv("BENCH_REPS")) ? atoi(e) : 5;
    const int aq = (e = getenv("BENCH_ACTQ")) ? atoi(e) : 1;
    std::mt19937 rng(1234);
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int bad = 0, nchk = 0;
    for (auto& sh : split(shapes_s)) {
        int64_t N = atoll(sh.c_str()), K = atoll(sh.substr(sh.find('x') + 1).c_str());
        const int64_t wb = N * K;
        const int copies = (int)std::max<int64_t>(2, (400ll << 20) / wb + 1);
        std::vector<uint8_t> hw(wb);
        for (auto& b : hw) { b = (uint8_t)rng(); if ((b & 0x7f) == 0x7f) b ^= 0x08; }
        std::vector<uint8_t*> dw(copies);
        for (auto& p : dw) { CK(hipMalloc(&p, wb)); CK(hipMemcpy(p, hw.data(), wb, hipMemcpyHostToDevice)); }
        std::vector<uint16_t> hx(16 * K);
        for (auto& v : hx) v = f2bf(((int)(rng() % 20001) - 10000) * 1e-4f);
        ik_bf16* dx; CK(hipMalloc(&dx, 16 * K * 2)); CK(hipMemcpy(dx, hx.data(), 16 * K * 2, hipMemcpyHostToDevice));
        float hs[2] = {0.01f, aq ? 0.125f : 0.f};
        float* dsc; CK(hipMalloc(&dsc, 8)); CK(hipMemcpy(dsc, hs, 8, hipMemcpyHostToDevice));
        void* dout; CK(hipMalloc(&dout, 16 * N * 4));
        void* dref; CK(hipMalloc(&dref, 16 * N * 4));
        double base = 0;
        for (auto& ms : split(ms_s)) {
            const int M = atoi(ms.c_str());
            ik_tensor tx = T(dx, IK_DTYPE_BF16, {M, K}), ts = T(dsc, IK_DTYPE_F32, {2});
            if (mode == "time" || mode == "both") {
                ik_tensor tw = T(dw[0], IK_DTYPE_FP8_E4M3, {N, K}), to = T(dout, IK_DTYPE_BF16, {M, N});
                for (int c = 0; c < 2; ++c) { tw.data = dw[c]; if (ik_matmul_fp8_gemv_rows(&tx, &tw, &ts, &to, st)) { fprintf(stderr, "launch failed\n"); return 3; } }
                CK(hipStreamSynchronize(st));
                double best = 1e30;
                for (int rep = 0; rep < reps; ++rep) {
                    CK(hipEventRecord(e0, st));
                    for (int c = 0; c < copies; ++c) { tw.data = dw[c]; ik_matmul_fp8_gemv_rows(&tx, &tw, &ts, &to, st); }
                    CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                    float ms_ = 0; CK(hipEventElapsedTime(&ms_, e0, e1));
                    best = std::min(best, ms_ * 1e3 / copies);
                }
                if (M == 1) base = best;
                printf("TIME N=%-5ld K=%-5ld M=%-2d %8.2f us  %6.0f GB/s(weights once)  x%.2f of M=1\n", (long)N, (long)K, M, best, wb / best / 1e3, best / base);
                fflush(stdout);
            }
            if (mode == "check" || mode == "both") {
                for (int f32o = 0; f32o < 2; ++f32o) {
                    const size_t osz = f32o ? 4 : 2;
                    ik_tensor tw = T(dw[0], IK_DTYPE_FP8_E4M3, {N, K}), to = T(dout, f32o ? IK_DTYPE_F32 : IK_DTYPE_BF16, {M, N});
                    CK(hipMemset(dout, 0xA5, 16 * N * 4));
                    if (ik_matmul_fp8_gemv_rows(&tx, &tw, &ts, &to, st)) { fprintf(stderr, "launch failed\n"); return 3; }
                    CK(hipStreamSynchronize(st));
                    std::vector<uint8_t> got(M * N * osz), want(osz * N);
                    CK(hipMemcpy(got.data(), dout, got.size(), hipMemcpyDeviceToHost));
                    int badrows = 0;
                    for (int m = 0; m < M; ++m) {
                        ik_tensor x1 = T(dx + m * K, IK_DTYPE_BF16, {1, K}), o1 = T(dref, f32o ? IK_DTYPE_F32 : IK_DTYPE_BF16, {1, N});
                        if (ik_matmul_fp8_gemv_v2(&x1, &tw, &ts, &o1, st)) { fprintf(stderr, "ref launch failed\n"); return 3; }
                        CK(hipStreamSynchronize(st));
                        CK(hipMemcpy(want.data(), dref, osz * N, hipMemcpyDeviceToHost));
                        if (memcmp(want.data(), got.data() + (size_t)m * N * osz, osz * N)) ++badrows;
                    }
                    ++nchk; bad += badrows != 0;
                    printf("CHECK N=%ld K=%ld M=%d actq=%d out=%s: %s\n", (long)N, (long)K, M, aq, f32o ? "f32" : "bf16", badrows ? "MISMATCH" : "bitwise ok");
                    if (badrows) printf("  %d of %d rows differ\n", badrows, M);
                }
            }
        }
        for (auto p : dw) CK(hipFree(p));
        CK(hipFree(dx)); CK(hipFree(dsc)); CK(hipFree(dout)); CK(hipFree(dref));
    }
    if (nchk) printf("SUMMARY %d checks, %d failed\n", nchk, bad);
    return bad != 0;
}
