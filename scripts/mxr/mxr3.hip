// Lane mxr, Phase 1 timing harness (window w4): the PRODUCTION int8 rows path of kernels/hip/mxfp4.hip (i8_rows) against the
// bf16 rows kernel it replaces (mxfp4_xprep_kernel + rows_by_m, what INTARSIA_HIP_MXFP4_I8=0 runs) and the M = 1 launch.
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 -I kernels/hip scripts/mxr/mxr3.hip -o mxr3
//   ./mxr3 N K [iters] [rounds] [act 1|0] [Ms...]      act 0 = W4A16 (every int8 call takes the flagged fp-chain path)
// Every arm is checked bitwise against the production M = 1 launch of each row (prod1). Bitwise edge coverage is in
// crates/intarsia-backend-hip/tests/rows_exact.rs; this file times.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include "../../kernels/hip/mxfp4.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float nrm() { float s = 0; for (int i = 0; i < 12; ++i) s += (rnd() & 0xffff) / 65536.f; return s - 6.f; }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x40); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static ik_tensor T(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t; memset(&t, 0, sizeof t); t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1; for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    for (int i = 1; i < argc; ++i) {
        char* end; strtoll(argv[i], &end, 10);
        if (*end) { printf("argument %d is not a number: '%s'\n", i, argv[i]); return 1; }
    }
    const int64_t N = argc > 1 ? atoll(argv[1]) : 17408, K = argc > 2 ? atoll(argv[2]) : 5120;
    const int iters = argc > 3 ? atoi(argv[3]) : 200, rounds = argc > 4 ? atoi(argv[4]) : 3;
    const float actv = argc > 5 ? (float)atoi(argv[5]) : 1.f;
    std::vector<int> Ms;
    for (int i = 6; i < argc; ++i) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {1, 2, 4, 8, 12, 16};
    if (N <= 0 || K <= 0 || K % 32 || iters < 1 || iters > 2000 || rounds < 1 || rounds > 20) { printf("bad N/K/iters/rounds\n"); return 1; }
    for (int M : Ms) if (M < 1 || M > 16) { printf("bad M %d\n", M); return 1; }
    const int MMAX = 16, COPIES = 3;
    const int64_t G = K / 32, wb = N * K / 2, sb = N * G;
    std::vector<uint8_t> hw(wb), hs(sb);
    for (auto& b : hw) b = (uint8_t)rnd();
    for (auto& b : hs) b = (uint8_t)(120 + rnd() % 15);
    std::vector<uint16_t> hx((size_t)MMAX * K);
    for (auto& v : hx) v = bf16(nrm());
    for (int64_t e = 0; e < K; ++e) hx[(size_t)3 * K + e] = 0;  // row 3: all zeros (a padded / finished row)
    uint8_t *dw[COPIES], *ds[COPIES]; uint16_t* dx; float* dact; float* dout; float* dref; void* dscr;
    for (int c = 0; c < COPIES; ++c) {
        HC(hipMalloc(&dw[c], wb)); HC(hipMalloc(&ds[c], sb));
        HC(hipMemcpy(dw[c], hw.data(), wb, hipMemcpyHostToDevice)); HC(hipMemcpy(ds[c], hs.data(), sb, hipMemcpyHostToDevice));
    }
    HC(hipMalloc(&dx, hx.size() * 2)); HC(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dact, 4)); HC(hipMemcpy(dact, &actv, 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dout, (size_t)MMAX * N * 4)); HC(hipMalloc(&dref, (size_t)MMAX * N * 4));
    const int64_t scrb = mx_i8_scratch(MMAX, K) + 64;
    HC(hipMalloc(&dscr, scrb));
    ik_scratch scr{dscr, scrb};
    const ik_bf16* xb = reinterpret_cast<const ik_bf16*>(dx);
    auto prod1 = [&](int m, int c, float* o) {
        ik_tensor tx = T(dx + (size_t)m * K, IK_DTYPE_BF16, {1, K}), tw = T(dw[c], IK_DTYPE_U8, {N, K / 2}), ts = T(ds[c], IK_DTYPE_U8, {N, G});
        ik_tensor ta = T(dact, IK_DTYPE_F32, {1}), to = T(o, IK_DTYPE_F32, {1, N});
        if (ik_matmul_mxfp4_gemv(&tx, &tw, &ts, &ta, &to, &scr, nullptr)) { printf("prod1 failed\n"); exit(3); }
    };
    auto bf16rows = [&](int M, int c, float* o) {  // what INTARSIA_HIP_MXFP4_I8=0 runs
        ik_bf16* xs = (ik_bf16*)dscr;
        const int64_t groups = (int64_t)M * K / kG;
        mxfp4_xprep_kernel<<<(unsigned)((groups + 255) / 256), 256>>>(xb, dact, xs, groups);
        if (rows_by_m(M, xs, dw[c], ds[c], o, N, K, nullptr)) { printf("rows failed\n"); exit(3); }
    };
    auto i8 = [&](int M, int xp, int c, float* o) {
        if (i8_rows(M, xp, xb, dact, dw[c], ds[c], o, N, (const uint8_t*)nullptr, (const uint8_t*)nullptr, (float*)nullptr, 0, K, &scr, nullptr)) {
            printf("i8 failed\n"); exit(3);
        }
    };
    for (int m = 0; m < MMAX; ++m) prod1(m, 0, dref + (size_t)m * N);
    HC(hipDeviceSynchronize());
    std::vector<float> ref((size_t)MMAX * N), got((size_t)MMAX * N);
    HC(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const int rule = (N <= 5120 || K > 8192) ? 4 : 2;
    printf("## mxr3 %ldx%ld act=%g iters=%d rounds=%d (shape rule XP=%d); us per call incl. pre-pass, min over rounds [max]; x = ratio to prod1 M=1; bitwise vs prod1 per row\n",
           (long)N, (long)K, actv, iters, rounds, rule);
    const char* names[] = {"prod1", "bf16 rows", "i8 xp2", "i8 xp4"};
    double t1 = 0;
    long total_bad = 0;
    for (int a = 0; a < 4; ++a) {
        printf("%-10s", names[a]);
        for (int M : Ms) {
            if ((a == 0) != (M == 1)) { printf(" |"); continue; }
            auto run = [&](int c) {
                if (a == 0) prod1(0, c, dout);
                else if (a == 1) bf16rows(M, c, dout);
                else i8(M, a == 2 ? 2 : 4, c, dout);
            };
            double best = 1e30, worst = 0;
            for (int rd = 0; rd < rounds; ++rd) {
                for (int i = 0; i < 10; ++i) run(i % COPIES);
                HC(hipEventRecord(e0, nullptr));
                for (int i = 0; i < iters; ++i) run(i % COPIES);
                HC(hipEventRecord(e1, nullptr)); HC(hipEventSynchronize(e1));
                float ms; HC(hipEventElapsedTime(&ms, e0, e1));
                best = std::min(best, ms * 1e3 / iters); worst = std::max(worst, ms * 1e3 / iters);
            }
            if (a == 0) t1 = best;
            HC(hipMemset(dout, 0xCD, (size_t)MMAX * N * 4));
            run(0); HC(hipDeviceSynchronize());
            HC(hipMemcpy(got.data(), dout, (size_t)M * N * 4, hipMemcpyDeviceToHost));
            long bad = 0;
            for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) if (memcmp(&got[(size_t)m * N + n], &ref[(size_t)m * N + n], 4)) ++bad;
            total_bad += bad;
            printf(" | M=%2d %7.1f [%7.1f] x%.2f %s", M, best, worst, t1 > 0 ? best / t1 : 0.0, bad ? ("MISMATCH " + std::to_string(bad)).c_str() : "ok");
        }
        printf("\n");
    }
    printf("## total mismatches %ld\n", total_bad);
    return total_bad ? 4 : 0;
}
