// Lane hload: standalone microbench + output dump of the HIP row-exact MoE expert projections (kernels/hip/moe.hip):
// moe_gate_up_g2_kernel<RW> and moe_down_g2_kernel<RW> at the decode shapes of Qwen3.6-35B-A3B NVFP4 (E 256, top-8, H 2048,
// I 512; TP=2 shard I 256). Includes moe.hip itself, so the numbers are the kernels of the tree it is compiled against.
//   BENCH_TS=1,2,4,8,16 BENCH_I=512,256 BENCH_GU=2 BENCH_DN=2 BENCH_REPS=5 BENCH_DUMP=<dir> [BENCH_OLD=1] ./bench_moe
// A tree with the lane's launch helpers (IK_G2_HAVE_LAUNCH) launches through them (the fast twins where they apply), unless BENCH_OLD=1.
// Routing: per token, top-8 distinct experts drawn from a Zipf-like popularity (exponent BENCH_ZIPF, default 0.9) over a fixed
// random expert permutation, one draw per layer copy (BENCH_LAYERS copies, default 12, about 5.4 GB of expert weights rotate, so
// every launch reads its experts from DRAM). The script prints the distinct experts and the most pairs on one expert per T
// (HPROF measured 42 distinct at 8 rows and 69 at 16, max 8-16 pairs on one expert).
// Timing: us per launch (best of BENCH_REPS reps of the mean over all layer copies), gate_up and down separately.
// Dump: h (gate_up output) and y (down output) of layer copy 0 for each (T, I) as raw bf16 files, for a cmp between builds.
#include "moe.hip"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int ik_rt_stamp(void*, const char*, int) { return 0; }  // the runtime stamp (INTARSIA_HIP_STAMP) is not linked here

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
    const auto Ts = csv(getenv("BENCH_TS"), "1,2,4,8,16");
    const auto Is = csv(getenv("BENCH_I"), "512");
    const int GU = (e = getenv("BENCH_GU")) ? atoi(e) : 2, DN = (e = getenv("BENCH_DN")) ? atoi(e) : 2;
    const int reps = (e = getenv("BENCH_REPS")) ? atoi(e) : 5;
    const int L = (e = getenv("BENCH_LAYERS")) ? atoi(e) : 12;
    const double zipf = (e = getenv("BENCH_ZIPF")) ? atof(e) : 0.9;
    const char* dump = getenv("BENCH_DUMP");
    const int old = (e = getenv("BENCH_OLD")) ? atoi(e) : 0;  // 1: the moe_*_g2_kernel launches even where the tree has the fast twins
    if (dump) { std::string c = std::string("mkdir -p ") + dump; if (system(c.c_str()) != 0) return 2; }
    const int E = 256, K = 8, H = 2048;
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int I : Is) {
        std::mt19937 rng(4321);
        const size_t guw_b = (size_t)E * 2 * I * H / 2, gus_b = (size_t)E * 2 * I * H / 16;
        const size_t dnw_b = (size_t)E * H * I / 2, dns_b = (size_t)E * H * I / 16;
        std::vector<uint8_t> hb(guw_b);
        std::vector<uint8_t*> guw(L), gus(L), dnw(L), dns(L);
        auto fill = [&](uint8_t* d, size_t n, bool scale) {
            hb.resize(n);
            // fp4 nibbles: any; e4m3 block scales: small positive normals (0x30..0x3f: 2^-1.. ~1), never NaN
            for (auto& b : hb) b = scale ? (uint8_t)(0x30 + rng() % 16) : (uint8_t)rng();
            CK(hipMemcpy(d, hb.data(), n, hipMemcpyHostToDevice));
        };
        for (int l = 0; l < L; ++l) {
            CK(hipMalloc(&guw[l], guw_b)); fill(guw[l], guw_b, false);
            CK(hipMalloc(&gus[l], gus_b)); fill(gus[l], gus_b, true);
            CK(hipMalloc(&dnw[l], dnw_b)); fill(dnw[l], dnw_b, false);
            CK(hipMalloc(&dns[l], dns_b)); fill(dns[l], dns_b, true);
        }
        std::vector<float> gs(E * 3);
        for (auto& v : gs) v = 0.5f + (rng() % 1000) * 1e-3f;
        float* dgs; CK(hipMalloc(&dgs, gs.size() * 4)); CK(hipMemcpy(dgs, gs.data(), gs.size() * 4, hipMemcpyHostToDevice));
        std::vector<uint16_t> hx(16 * H);
        for (auto& v : hx) v = f2bf(((int)(rng() % 20001) - 10000) * 1e-4f);
        ik_bf16* dx; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        ik_bf16 *dh, *dy; CK(hipMalloc(&dh, (size_t)16 * K * I * 2)); CK(hipMalloc(&dy, (size_t)16 * K * H * 2));
        int *dids, *dpairs, *doff, *dact;
        CK(hipMalloc(&dids, (size_t)L * 16 * K * 4)); CK(hipMalloc(&dpairs, 16 * K * 4));
        CK(hipMalloc(&doff, (size_t)L * (E + 1) * 4)); CK(hipMalloc(&dact, (size_t)L * (E + 1) * 4));
        int* dpairsL; CK(hipMalloc(&dpairsL, (size_t)L * 16 * K * 4));
        // expert popularity: Zipf over a random permutation
        std::vector<int> perm(E);
        for (int i = 0; i < E; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<double> cdf(E);
        double acc = 0;
        for (int i = 0; i < E; ++i) { acc += 1.0 / pow(i + 1.0, zipf); cdf[i] = acc; }
        for (auto& c : cdf) c /= acc;
        std::uniform_real_distribution<double> U(0, 1);
        for (int T : Ts) {
            const int P = T * K;
            const int slots = std::min(P, E);
            double distinct = 0; int maxp = 0;
            for (int l = 0; l < L; ++l) {
                std::vector<int> ids(P);
                for (int t = 0; t < T; ++t) {
                    std::vector<int> got;
                    while ((int)got.size() < K) {
                        const int r = (int)(std::lower_bound(cdf.begin(), cdf.end(), U(rng)) - cdf.begin());
                        const int ex = perm[std::min(r, E - 1)];
                        if (std::find(got.begin(), got.end(), ex) == got.end()) got.push_back(ex);
                    }
                    for (int k = 0; k < K; ++k) ids[t * K + k] = got[k];
                }
                std::vector<int> cnt(E, 0);
                for (int v : ids) cnt[v]++;
                int d = 0;
                for (int c : cnt) { d += c > 0; maxp = std::max(maxp, c); }
                distinct += d;
                CK(hipMemcpy(dids + (size_t)l * 16 * K, ids.data(), P * 4, hipMemcpyHostToDevice));
                moe_group_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), st>>>(dids + (size_t)l * 16 * K, P, E, dpairsL + (size_t)l * 16 * K,
                                                                           doff + (size_t)l * (E + 1), dact + (size_t)l * (E + 1));
            }
            CK(hipStreamSynchronize(st));
            auto gate_up = [&](int l) {
                const int* pr = dpairsL + (size_t)l * 16 * K; const int* of = doff + (size_t)l * (E + 1); const int* ac = dact + (size_t)l * (E + 1);
#define GU_L(RW) moe_gate_up_g2_kernel<RW><<<dim3(slots, (unsigned)((I + kRowWarps * RW - 1) / (kRowWarps * RW))), kRowWarps * IK_WARP, 0, st>>>( \
                    dx, guw[l], gus[l], dgs, pr, of, ac, dh, E, H, I, K)
#ifdef IK_G2_HAVE_LAUNCH
                if (!old) { g2_gate_up_launch(GU, slots, dx, guw[l], gus[l], dgs, pr, of, ac, dh, E, H, I, K, st); return; }
#endif
                if (GU == 1) GU_L(1); else GU_L(2);
            };
            auto down = [&](int l) {
                const int* pr = dpairsL + (size_t)l * 16 * K; const int* of = doff + (size_t)l * (E + 1); const int* ac = dact + (size_t)l * (E + 1);
#define DN_L(RW) moe_down_g2_kernel<RW><<<dim3(slots, (unsigned)((H + kRowWarps * RW - 1) / (kRowWarps * RW))), kRowWarps * IK_WARP, 0, st>>>( \
                    dh, dnw[l], dns[l], dgs, pr, of, ac, dy, E, H, I)
#ifdef IK_G2_HAVE_LAUNCH
                if (!old) { g2_down_launch(DN, slots, dh, dnw[l], dns[l], dgs, pr, of, ac, dy, E, H, I, st); return; }
#endif
                if (DN == 1) DN_L(1); else if (DN == 2) DN_L(2); else DN_L(4);
            };
            if (dump) {
                gate_up(0); down(0);
                CK(hipStreamSynchronize(st));
                std::vector<uint16_t> oh((size_t)P * I), oy((size_t)P * H);
                CK(hipMemcpy(oh.data(), dh, oh.size() * 2, hipMemcpyDeviceToHost));
                CK(hipMemcpy(oy.data(), dy, oy.size() * 2, hipMemcpyDeviceToHost));
                char fn[512];
                snprintf(fn, sizeof fn, "%s/moe-T%d-I%d.bin", dump, T, I);
                FILE* f = fopen(fn, "wb");
                if (!f) { fprintf(stderr, "cannot write %s\n", fn); return 2; }
                fwrite(oh.data(), 2, oh.size(), f); fwrite(oy.data(), 2, oy.size(), f); fclose(f);
            }
            double best_gu = 1e30, best_dn = 1e30;
            for (int r = 0; r < reps; ++r) {
                for (int l = 0; l < L; ++l) gate_up(l);  // warm-up / evict
                CK(hipEventRecord(e0, st));
                for (int l = 0; l < L; ++l) gate_up(l);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                best_gu = std::min(best_gu, ms * 1e3 / L);
                CK(hipEventRecord(e0, st));
                for (int l = 0; l < L; ++l) down(l);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                CK(hipEventElapsedTime(&ms, e0, e1));
                best_dn = std::min(best_dn, ms * 1e3 / L);
            }
            printf("I=%d T=%2d distinct %.1f maxpairs %d  gate_up<%d> %7.2f us  down<%d> %7.2f us\n", I, T, distinct / L, maxp, GU, best_gu, DN, best_dn);
            fflush(stdout);
        }
        for (int l = 0; l < L; ++l) { CK(hipFree(guw[l])); CK(hipFree(gus[l])); CK(hipFree(dnw[l])); CK(hipFree(dns[l])); }
        CK(hipFree(dgs)); CK(hipFree(dx)); CK(hipFree(dh)); CK(hipFree(dy)); CK(hipFree(dids)); CK(hipFree(dpairs));
        CK(hipFree(doff)); CK(hipFree(dact)); CK(hipFree(dpairsL));
    }
    return 0;
}
