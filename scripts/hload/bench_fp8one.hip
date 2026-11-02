// Lane hload: standalone microbench + output dump of the HIP one-row fp8 GEMV family (kernels/hip/fp8.hip, fp8_gemv_body: the
// M = 1, 2 rows of the decode step and the M = 1 folds) at the shapes of Qwen3.6-35B-A3B NVFP4. Includes fp8.hip itself.
//   BENCH_CASES="v2:8192:2048,rows2:8192:2048,pair:8192:2048:4096,gated:2048:4096,sig:2048:4096,npair:512:2048:512" \
//   BENCH_MODE=time|dump|both BENCH_REPS=5 BENCH_ACTQ=1 BENCH_DUMP=<dir> ./bench_fp8one
// Case kinds (N x K weights; N1 the second weight of a pair):
//   v2       ik_matmul_fp8_gemv_v2, one row                       (attention q_proj at M = 1)
//   rowsM    ik_matmul_fp8_gemv_rows, M rows (M = 1..4 run fp8_gemv_kernel<M>; the batched step at 2 rows)
//   pair     ik_matmul_fp8_pair_gemv (GDN in_proj_qkv + in_proj_z at M = 1)
//   gated    ik_gated_rmsnorm_matmul_fp8_gemv (GDN out_proj with the gated norm folded, K = rows * 128)
//   sig      ik_sigmoid_mul_matmul_fp8_gemv (attention o_proj with the output gate folded)
//   npair    ik_rmsnorm_add_matmul_fp8_pair_gemv (attention input norm + k / v)
// time: us per launch, weights rotated over >= 400 MB of copies, best of BENCH_REPS. dump: every output tensor of copy 0, bf16
// and (v2 / rows) f32 out, to <dir>/<case>-a<actq>.bin, for a cmp between builds.
#include "fp8.hip"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int ik_rt_stamp(void*, const char*, int) { return 0; }

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(2); } } while (0)

static ik_tensor T(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t{};
    t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1;
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}
static std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> v; size_t a = 0;
    while (a <= s.size()) { size_t b = s.find(c, a); if (b == std::string::npos) b = s.size(); if (b > a) v.push_back(s.substr(a, b - a)); a = b + 1; }
    return v;
}
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

int main() {
    const char* e;
    std::string cases = (e = getenv("BENCH_CASES")) ? e : "v2:8192:2048,rows2:8192:2048,pair:8192:2048:4096,gated:2048:4096,sig:2048:4096,npair:512:2048:512";
    std::string mode = (e = getenv("BENCH_MODE")) ? e : "both";
    const int reps = (e = getenv("BENCH_REPS")) ? atoi(e) : 5;
    const int aq = (e = getenv("BENCH_ACTQ")) ? atoi(e) : 1;
    const char* dump = getenv("BENCH_DUMP");
    if (dump) { std::string c = std::string("mkdir -p ") + dump; if (system(c.c_str()) != 0) return 2; }
    std::mt19937 rng(99);
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto bfvec = [&](size_t n, float amp) {
        std::vector<uint16_t> v(n);
        for (auto& x : v) x = f2bf(((int)(rng() % 20001) - 10000) * 1e-4f * amp);
        return v;
    };
    auto up = [&](const void* h, size_t bytes) { void* d; CK(hipMalloc(&d, std::max<size_t>(bytes, 16))); CK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice)); return d; };
    for (auto& cs : split(cases, ',')) {
        auto f = split(cs, ':');
        const std::string kind = f[0];
        const int64_t N = atoll(f[1].c_str()), K = atoll(f[2].c_str()), N1 = f.size() > 3 ? atoll(f[3].c_str()) : 0;
        const int M = kind.rfind("rows", 0) == 0 ? atoi(kind.c_str() + 4) : 1;
        const bool two = kind == "pair" || kind == "npair";
        const int64_t wb = (N + (two ? N1 : 0)) * K;
        const int copies = mode == "dump" ? 1 : (int)std::max<int64_t>(2, (400ll << 20) / wb + 1);
        std::vector<uint8_t> hw((size_t)wb);
        for (auto& b : hw) { b = (uint8_t)rng(); if ((b & 0x7f) == 0x7f) b ^= 0x08; }
        std::vector<uint8_t*> dw(copies);
        for (auto& p : dw) { CK(hipMalloc(&p, wb)); CK(hipMemcpy(p, hw.data(), wb, hipMemcpyHostToDevice)); }
        auto hx = bfvec((size_t)M * K, 1.f), hz = bfvec((size_t)K, 1.f), hres = bfvec((size_t)K, 1.f), hnw = bfvec((size_t)K, 0.5f);
        void *dx = up(hx.data(), hx.size() * 2), *dz = up(hz.data(), hz.size() * 2), *dres = up(hres.data(), hres.size() * 2);
        void* dnw = up(hnw.data(), hnw.size() * 2);
        float hs[4] = {0.01f, aq ? 0.125f : 0.f, 0.02f, aq ? 0.0625f : 0.f};
        void *dsc0 = up(hs, 8), *dsc1 = up(hs + 2, 8);
        void *dout0, *dout1, *dnormed, *dsum;
        CK(hipMalloc(&dout0, (size_t)M * N * 4)); CK(hipMalloc(&dout1, std::max<int64_t>(N1, 1) * 4));
        CK(hipMalloc(&dnormed, K * 2)); CK(hipMalloc(&dsum, K * 2));
        auto run = [&](int c, int f32o) -> int {
            uint8_t* w0 = dw[c];
            uint8_t* w1 = w0 + N * K;
            ik_tensor ts0 = T(dsc0, IK_DTYPE_F32, {2}), ts1 = T(dsc1, IK_DTYPE_F32, {2});
            ik_tensor tw0 = T(w0, IK_DTYPE_FP8_E4M3, {N, K}), tw1 = T(w1, IK_DTYPE_FP8_E4M3, {N1, K});
            ik_tensor to0 = T(dout0, f32o ? IK_DTYPE_F32 : IK_DTYPE_BF16, {M, N}), to1 = T(dout1, IK_DTYPE_BF16, {1, N1});
            if (kind == "v2") { ik_tensor tx = T(dx, IK_DTYPE_BF16, {1, K}); return ik_matmul_fp8_gemv_v2(&tx, &tw0, &ts0, &to0, st); }
            if (M > 0 && kind.rfind("rows", 0) == 0) { ik_tensor tx = T(dx, IK_DTYPE_BF16, {M, K}); return ik_matmul_fp8_gemv_rows(&tx, &tw0, &ts0, &to0, st); }
            if (kind == "pair") { ik_tensor tx = T(dx, IK_DTYPE_BF16, {1, K}); return ik_matmul_fp8_pair_gemv(&tx, &tw0, &ts0, &tw1, &ts1, &to0, &to1, st); }
            if (kind == "gated") {
                ik_tensor tx = T(dx, IK_DTYPE_BF16, {K / 128, 128}), tz = T(dz, IK_DTYPE_BF16, {K / 128, 128}), tn = T(dnw, IK_DTYPE_BF16, {128});
                return ik_gated_rmsnorm_matmul_fp8_gemv(&tx, &tz, &tn, 1e-6f, 0.f, &tw0, &ts0, &to0, st);
            }
            if (kind == "sig") {
                ik_tensor ta = T(dx, IK_DTYPE_BF16, {1, K}), tg = T(dz, IK_DTYPE_BF16, {1, K});
                return ik_sigmoid_mul_matmul_fp8_gemv(&ta, &tg, &tw0, &ts0, &to0, st);
            }
            if (kind == "npair") {
                ik_tensor tx = T(dx, IK_DTYPE_BF16, {1, K}), tr = T(dres, IK_DTYPE_BF16, {1, K}), tn = T(dnw, IK_DTYPE_BF16, {K});
                ik_tensor tnm = T(dnormed, IK_DTYPE_BF16, {1, K}), tsm = T(dsum, IK_DTYPE_BF16, {1, K});
                return ik_rmsnorm_add_matmul_fp8_pair_gemv(&tx, &tr, &tn, 1e-6f, 1.f, &tw0, &ts0, &tw1, &ts1, &tnm, &tsm, &to0, &to1, st);
            }
            return -1;
        };
        if (mode == "time" || mode == "both") {
            for (int c = 0; c < 2; ++c) if (int rc = run(c, 0)) { fprintf(stderr, "%s: launch rc %d\n", cs.c_str(), rc); return 3; }
            CK(hipStreamSynchronize(st));
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                CK(hipEventRecord(e0, st));
                for (int c = 0; c < copies; ++c) run(c, 0);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                best = std::min(best, ms * 1e3 / copies);
            }
            printf("TIME %-22s actq=%d %8.2f us %6.0f GB/s\n", cs.c_str(), aq, best, wb / best / 1e3);
            fflush(stdout);
        }
        if ((mode == "dump" || mode == "both") && dump) {
            std::string fn = std::string(dump) + "/" + kind + "-" + f[1] + "-" + f[2] + (N1 ? "-" + f[3] : "") + "-a" + std::to_string(aq) + ".bin";
            FILE* fo = fopen(fn.c_str(), "wb");
            if (!fo) { fprintf(stderr, "cannot write %s\n", fn.c_str()); return 2; }
            const bool f32ok = kind == "v2" || kind.rfind("rows", 0) == 0;
            for (int f32o = 0; f32o < (f32ok ? 2 : 1); ++f32o) {
                CK(hipMemset(dout0, 0xA5, (size_t)M * N * 4)); CK(hipMemset(dout1, 0xA5, std::max<int64_t>(N1, 1) * 4));
                CK(hipMemset(dnormed, 0xA5, K * 2)); CK(hipMemset(dsum, 0xA5, K * 2));
                if (int rc = run(0, f32o)) { fprintf(stderr, "%s: launch rc %d\n", cs.c_str(), rc); return 3; }
                CK(hipStreamSynchronize(st));
                std::vector<uint8_t> o((size_t)M * N * (f32o ? 4 : 2));
                CK(hipMemcpy(o.data(), dout0, o.size(), hipMemcpyDeviceToHost));
                fwrite(o.data(), 1, o.size(), fo);
                if (two) { std::vector<uint8_t> o1((size_t)N1 * 2); CK(hipMemcpy(o1.data(), dout1, o1.size(), hipMemcpyDeviceToHost)); fwrite(o1.data(), 1, o1.size(), fo); }
                if (kind == "npair") {
                    std::vector<uint8_t> o2((size_t)K * 2);
                    CK(hipMemcpy(o2.data(), dnormed, o2.size(), hipMemcpyDeviceToHost)); fwrite(o2.data(), 1, o2.size(), fo);
                    CK(hipMemcpy(o2.data(), dsum, o2.size(), hipMemcpyDeviceToHost)); fwrite(o2.data(), 1, o2.size(), fo);
                }
            }
            fclose(fo);
            printf("DUMP %s\n", fn.c_str());
        }
        for (auto p : dw) CK(hipFree(p));
        for (void* p : {dx, dz, dres, dnw, dsc0, dsc1, dout0, dout1, dnormed, dsum}) CK(hipFree(p));
    }
    return 0;
}
