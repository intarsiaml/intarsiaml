// Lane hload: standalone microbench + output dump of the batched shared expert of the grouped HIP MoE block
// (kernels/hip/moe_block_grouped.hip: shared_glu_rows_kernel, shared_down_combine_rows_kernel; 3..16 rows) at the shapes of
// Qwen3.6-35B-A3B NVFP4 (H 2048, shared IS 512; TP=2 shard IS 256). Includes moe_block_grouped.hip; launches the two kernels
// as ik_moe_block_bf16_grouped does.
//   BENCH_TS=3,4,8,16 BENCH_I=512,256 BENCH_REPS=5 BENCH_LAYERS=96 BENCH_DUMP=<dir> ./bench_shared
// The shared weights of BENCH_LAYERS copies rotate (about 140 MB at IS 512, beyond the 64 MB infinity cache). us per launch of
// each kernel: best of BENCH_REPS reps of the mean over the copies. Dump: glu output and combined output of copy 0.
#include "moe_block_grouped.hip"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// the grouped launcher's other ops live in moe.hip (not linked here; the harness launches the two shared kernels directly)
extern "C" int ik_rt_stamp(void*, const char*, int) { return 0; }
extern "C" int ik_moe_experts_bf16_rows(const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*, const ik_tensor*,
                                        const ik_tensor*, const ik_tensor*, const ik_tensor*, ik_tensor*, const ik_scratch*, void*) { return -1; }
extern "C" int ik_moe_topk_bf16_v2(const ik_tensor*, int, int, ik_tensor*, ik_tensor*, void*) { return -1; }

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(2); } } while (0)

static std::vector<int> csv(const char* s, const char* def) {
    std::string v = s ? s : def;
    std::vector<int> r;
    size_t a = 0;
    while (a < v.size()) { size_t b = v.find(',', a); if (b == std::string::npos) b = v.size(); r.push_back(atoi(v.substr(a, b - a).c_str())); a = b + 1; }
    return r;
}
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

int main() {
    const char* e;
    const auto Ts = csv(getenv("BENCH_TS"), "3,4,8,16");
    const auto Is = csv(getenv("BENCH_I"), "512");
    const int reps = (e = getenv("BENCH_REPS")) ? atoi(e) : 5;
    const int L = (e = getenv("BENCH_LAYERS")) ? atoi(e) : 96;
    const char* dump = getenv("BENCH_DUMP");
    if (dump) { std::string c = std::string("mkdir -p ") + dump; if (system(c.c_str()) != 0) return 2; }
    const int H = 2048;
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    std::mt19937 rng(555);
    std::vector<uint8_t> hb;
    auto dev = [&](size_t n, int kind) {  // 0 fp4 bytes, 1 e4m3 scales, 2 f32 global scale, 3 bf16 values (n elements)
        void* d;
        if (kind == 2) {
            std::vector<float> f(n);
            for (auto& v : f) v = 0.5f + (rng() % 1000) * 1e-3f;
            CK(hipMalloc(&d, n * 4)); CK(hipMemcpy(d, f.data(), n * 4, hipMemcpyHostToDevice));
        } else if (kind == 3) {
            std::vector<uint16_t> f(n);
            for (auto& v : f) v = f2bf(((int)(rng() % 20001) - 10000) * 1e-4f);
            CK(hipMalloc(&d, n * 2)); CK(hipMemcpy(d, f.data(), n * 2, hipMemcpyHostToDevice));
        } else {
            hb.resize(n);
            for (auto& b : hb) b = kind ? (uint8_t)(0x30 + rng() % 16) : (uint8_t)rng();
            CK(hipMalloc(&d, n)); CK(hipMemcpy(d, hb.data(), n, hipMemcpyHostToDevice));
        }
        return d;
    };
    for (int IS : Is) {
        struct Layer { void *sgw, *sgs, *sgg, *suw, *sus, *sug, *sdw, *sds, *sdg; };
        std::vector<Layer> ly(L);
        for (auto& l : ly) {
            l.sgw = dev((size_t)IS * H / 2, 0); l.sgs = dev((size_t)IS * H / 16, 1); l.sgg = dev(1, 2);
            l.suw = dev((size_t)IS * H / 2, 0); l.sus = dev((size_t)IS * H / 16, 1); l.sug = dev(1, 2);
            l.sdw = dev((size_t)H * IS / 2, 0); l.sds = dev((size_t)H * IS / 16, 1); l.sdg = dev(1, 2);
        }
        void* dx = dev((size_t)16 * H, 3);
        void* droute = dev((size_t)16 * H, 3);
        void* dlogit = dev(16, 3);
        void *dsa, *dout;
        CK(hipMalloc(&dsa, (size_t)16 * IS * 2)); CK(hipMalloc(&dout, (size_t)16 * H * 2));
        for (int T : Ts) {
            const unsigned tgroups = (unsigned)((T + kRowsGrp - 1) / kRowsGrp);
            auto glu = [&](int l) {
                const Layer& y = ly[l];
                shared_glu_rows_kernel<<<dim3((unsigned)((IS + kWarps - 1) / kWarps), tgroups), kWarps * IK_WARP, 0, st>>>(
                    (const ik_bf16*)dx, (const uint8_t*)y.sgw, (const uint8_t*)y.sgs, (const float*)y.sgg, (const uint8_t*)y.suw,
                    (const uint8_t*)y.sus, (const float*)y.sug, (ik_bf16*)dsa, T, IS, H);
            };
            auto down = [&](int l) {
                const Layer& y = ly[l];
                shared_down_combine_rows_kernel<<<dim3((unsigned)((H + kWarps - 1) / kWarps), (unsigned)((T + kRowsGrpDown - 1) / kRowsGrpDown)),
                                                  kWarps * IK_WARP, 0, st>>>(
                    (const ik_bf16*)dsa, (const uint8_t*)y.sdw, (const uint8_t*)y.sds, (const float*)y.sdg, (const ik_bf16*)droute,
                    (const ik_bf16*)dlogit, (ik_bf16*)dout, T, H, IS);
            };
            if (dump) {
                glu(0); down(0);
                CK(hipStreamSynchronize(st));
                std::vector<uint16_t> a((size_t)T * IS), o((size_t)T * H);
                CK(hipMemcpy(a.data(), dsa, a.size() * 2, hipMemcpyDeviceToHost));
                CK(hipMemcpy(o.data(), dout, o.size() * 2, hipMemcpyDeviceToHost));
                char fn[512];
                snprintf(fn, sizeof fn, "%s/shared-T%d-I%d.bin", dump, T, IS);
                FILE* f = fopen(fn, "wb");
                if (!f) { fprintf(stderr, "cannot write %s\n", fn); return 2; }
                fwrite(a.data(), 2, a.size(), f); fwrite(o.data(), 2, o.size(), f); fclose(f);
            }
            double bg = 1e30, bd = 1e30;
            for (int r = 0; r < reps; ++r) {
                for (int l = 0; l < L; ++l) glu(l);
                CK(hipEventRecord(e0, st));
                for (int l = 0; l < L; ++l) glu(l);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                bg = std::min(bg, ms * 1e3 / L);
                CK(hipEventRecord(e0, st));
                for (int l = 0; l < L; ++l) down(l);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                CK(hipEventElapsedTime(&ms, e0, e1));
                bd = std::min(bd, ms * 1e3 / L);
            }
            CK(hipGetLastError());
            printf("shared IS=%d T=%2d glu %7.2f us  down_combine %7.2f us\n", IS, T, bg, bd);
            fflush(stdout);
        }
        for (auto& l : ly)
            for (void* p : {l.sgw, l.sgs, l.sgg, l.suw, l.sus, l.sug, l.sdw, l.sds, l.sdg}) CK(hipFree(p));
        for (void* p : {dx, droute, dlogit, dsa, dout}) CK(hipFree(p));
    }
    return 0;
}
