// Lane mr: standalone harness for logits_bf16_gemv_rows (kernels/hip/matmul.hip) without a cargo build.
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 -I kernels/hip scripts/mr/lgbench.hip -o lgbench
//   ./lgbench N K [iters]    time per M = 1..16 of the row-exact launcher and the bitwise check: row m of the M-row call
//                            == the M = 1 launcher (ik_logits_bf16_gemv, the production decode path) on row m alone
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include "../../kernels/hip/matmul.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float nrm() { float s = 0; for (int i = 0; i < 12; ++i) s += (rnd() & 0xffff) / 65536.f; return s - 6.f; }

static ik_tensor T(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t; memset(&t, 0, sizeof t); t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1; for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}

int main(int argc, char** argv) {
    const int64_t N = argc > 1 ? atoll(argv[1]) : 248320, K = argc > 2 ? atoll(argv[2]) : 5120;
    const int iters = argc > 3 ? atoi(argv[3]) : 50;
    const int MMAX = 16;
    std::vector<uint16_t> hw((size_t)N * K), hx((size_t)MMAX * K);
    const float sc = 1.f / sqrtf((float)K);
    for (auto& v : hw) v = bf16(nrm() * sc * 0.5f);
    for (auto& v : hx) v = bf16(nrm());
    uint16_t *dw, *dx; float *dout, *dref;
    HC(hipMalloc(&dw, hw.size() * 2)); HC(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dx, hx.size() * 2)); HC(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dout, (size_t)MMAX * N * 4)); HC(hipMalloc(&dref, (size_t)MMAX * N * 4));
    auto rows = [&](int M, float* out) {
        ik_tensor tx = T(dx, IK_DTYPE_BF16, {M, K}), tw = T(dw, IK_DTYPE_BF16, {N, K}), to = T(out, IK_DTYPE_F32, {M, N});
        int r = ik_logits_bf16_gemv_rows(&tx, &tw, &to, nullptr);
        if (r != 0) { printf("rows status %d (M=%d)\n", r, M); exit(3); }
    };
    for (int m = 0; m < MMAX; ++m) {
        ik_tensor tx = T(dx + (size_t)m * K, IK_DTYPE_BF16, {1, K}), tw = T(dw, IK_DTYPE_BF16, {N, K}), to = T(dref + (size_t)m * N, IK_DTYPE_F32, {1, N});
        if (ik_logits_bf16_gemv(&tx, &tw, &to, nullptr) != 0) { printf("solo failed\n"); return 3; }
    }
    HC(hipDeviceSynchronize());
    std::vector<float> ref((size_t)MMAX * N), got((size_t)MMAX * N);
    HC(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    printf("logits rows %ldx%ld variant %s\n", (long)N, (long)K, getenv("LGR_VARIANT") ? getenv("LGR_VARIANT") : "default");
    for (int M = 1; M <= MMAX; ++M) {
        rows(M, dout);
        for (int i = 0; i < 5; ++i) rows(M, dout);
        HC(hipEventRecord(e0, nullptr));
        for (int i = 0; i < iters; ++i) rows(M, dout);
        HC(hipEventRecord(e1, nullptr)); HC(hipEventSynchronize(e1));
        float ms; HC(hipEventElapsedTime(&ms, e0, e1));
        HC(hipMemcpy(got.data(), dout, (size_t)M * N * 4, hipMemcpyDeviceToHost));
        long bad = 0; for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) if (memcmp(&got[(size_t)m * N + n], &ref[(size_t)m * N + n], 4)) ++bad;
        printf("M=%2d %8.1f us  %s%s\n", M, ms * 1e3 / iters, bad ? "MISMATCH rows=" : "bitwise", bad ? std::to_string(bad).c_str() : "");
    }
    return 0;
}
