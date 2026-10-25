// Lane mr: standalone harness for the MXFP4 row-exact GEMV (kernels/hip/mxfp4.hip) without a cargo build.
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 -I kernels/hip scripts/mr/mxbench.hip -o mxbench
//   ./mxbench N K [iters] [kernel: gemv|rows]      (timing per M = 1..16 on a rotating set of weight copies, plus the
//                                                    bitwise check: row r of the M-row call == the M = 1 call on row r)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include "../../kernels/hip/mxfp4.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float nrm() { float s = 0; for (int i = 0; i < 12; ++i) s += (rnd() & 0xffff) / 65536.f; return s - 6.f; }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

static ik_tensor T(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t; memset(&t, 0, sizeof t); t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1; for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}

int main(int argc, char** argv) {
    const int64_t N = argc > 1 ? atoll(argv[1]) : 17408, K = argc > 2 ? atoll(argv[2]) : 5120;
    const int iters = argc > 3 ? atoi(argv[3]) : 200;
    const bool rows = argc > 4 && !strcmp(argv[4], "rows");
    const int MMAX = 16, COPIES = 3;
    const int64_t wb = N * K / 2, sb = N * K / 32;
    std::vector<uint8_t> hw(wb), hs(sb);
    for (auto& b : hw) b = (uint8_t)rnd();
    for (auto& b : hs) b = (uint8_t)(120 + rnd() % 15);
    std::vector<uint16_t> hx((size_t)MMAX * K);
    for (auto& v : hx) v = bf16(nrm());
    uint8_t *dw[COPIES], *ds[COPIES]; uint16_t* dx; float* dact; float* dout; float* dref; void* dscr;
    for (int c = 0; c < COPIES; ++c) { HC(hipMalloc(&dw[c], wb)); HC(hipMalloc(&ds[c], sb)); HC(hipMemcpy(dw[c], hw.data(), wb, hipMemcpyHostToDevice)); HC(hipMemcpy(ds[c], hs.data(), sb, hipMemcpyHostToDevice)); }
    HC(hipMalloc(&dx, hx.size() * 2)); HC(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dact, 4)); float one = 1.f; HC(hipMemcpy(dact, &one, 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dout, (size_t)MMAX * N * 4)); HC(hipMalloc(&dref, (size_t)MMAX * N * 4));
    HC(hipMalloc(&dscr, (size_t)MMAX * K * 2 + 4096));
    ik_scratch scr{dscr, (int64_t)MMAX * K * 2 + 4096};
    bool skip = false;
    auto run = [&](int M, int c, uint16_t* x, float* out) {
        ik_tensor tx = T(x, IK_DTYPE_BF16, {M, K}), tw = T(dw[c], IK_DTYPE_U8, {N, K / 2}), ts = T(ds[c], IK_DTYPE_U8, {N, K / 32});
        ik_tensor ta = T(dact, IK_DTYPE_F32, {1}), to = T(out, IK_DTYPE_F32, {M, N});
        int r = rows ? ik_matmul_mxfp4_gemv_rows(&tx, &tw, &ts, &ta, &to, &scr, nullptr)
                     : ik_matmul_mxfp4_gemv(&tx, &tw, &ts, &ta, &to, &scr, nullptr);
        if (r == 3) { skip = true; return; }
        if (r != 0) { printf("launcher status %d (M=%d)\n", r, M); exit(3); }
    };
    // reference: each row alone through the M = 1 launch (the production decode path)
    for (int m = 0; m < MMAX; ++m) run(1, 0, dx + (size_t)m * K, dref + (size_t)m * N);
    HC(hipDeviceSynchronize());
    std::vector<float> ref((size_t)MMAX * N), got((size_t)MMAX * N);
    HC(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    double t1 = 0;
    printf("%s %ldx%ld\n", rows ? "rows" : "gemv", (long)N, (long)K);
    for (int M = 1; M <= MMAX; ++M) {
        if (!rows && M > 8) break;
        skip = false; run(M, 0, dx, dout); if (skip) continue;
        for (int i = 0; i < 20; ++i) run(M, i % COPIES, dx, dout);
        HC(hipEventRecord(e0, nullptr));
        for (int i = 0; i < iters; ++i) run(M, i % COPIES, dx, dout);
        HC(hipEventRecord(e1, nullptr)); HC(hipEventSynchronize(e1));
        float ms; HC(hipEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1e3 / iters; if (M == 1) t1 = us;
        run(M, 0, dx, dout); HC(hipDeviceSynchronize());
        HC(hipMemcpy(got.data(), dout, (size_t)M * N * 4, hipMemcpyDeviceToHost));
        long bad = 0; for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) if (memcmp(&got[(size_t)m * N + n], &ref[(size_t)m * N + n], 4)) ++bad;
        printf("M=%2d %8.2f us  x%.2f  %s%s\n", M, us, us / t1, bad ? "MISMATCH rows=" : "bitwise", bad ? std::to_string(bad).c_str() : "");
    }
    return 0;
}
