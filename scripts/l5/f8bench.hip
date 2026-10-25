// Lane l5: standalone harness of the fp8-WMMA routed experts (kernels/hip/moe_wmma_fp8.h) against the bf16 r7 kernels (moe_wmma_r7.h) at the
// Qwen3.6-35B-A3B shapes (E = 256, top-8, H = 2048, I = 512) on one R9700, in ONE binary.
//   Build (dev image, repo root): hipcc -O3 -std=c++17 --offload-arch=gfx1201 -I kernels/hip -I kernels/abi scripts/l5/f8bench.hip \
//         kernels/hip/ik_runtime.hip -ldl -o <out>/f8bench
//   Run: f8bench [T=4096] [reps=10] [skew=0|1] [check=1] [ablate=0]    (T = tokens of the call; pairs = 8 T)
// Prints per projection ms and TFLOP/s of r7 and of the fp8 kernels (variants), the quantiser passes, and the reference check:
//   * the quantiser bytes and scales against a host RNE e4m3 quantiser (exact);
//   * the fp8 gate_up / down outputs of three experts against an f64 reference over the SAME e4m3 inputs read back from the device (what
//     the kernel must agree with up to f32 accumulation: reported as max |err| in units of the bf16 ulp of the reference and as the number of bf16 outputs that differ);
//   * the fp8 outputs against the bf16 r7 outputs (relative Frobenius error: the cost of quantising the activations).
#define IK_F8_ABLATE 1
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "../../kernels/hip/moe.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float urand() { return (rnd() & 0xffffff) / 16777216.f; }
static float nrm() { float s = 0; for (int i = 0; i < 12; ++i) s += urand(); return s - 6.f; }
static uint16_t bf16bits(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static float rbf(float f) { return bf2f(bf16bits(f)); }

// OCP e4m3fn
static float e4m3f(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = ldexpf((float)m, -9);
    else v = ldexpf(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}
// RNE, saturating at 448 (host twin of ik_f_to_e4m3)
static uint8_t f2e4m3(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    const uint8_t sign = (u >> 24) & 0x80u;
    const float a = fabsf(f);
    if (a >= 448.f) return sign | 0x7e;
    if (a < 0.015625f) return sign | (uint8_t)lrintf(a * 512.f);
    uint32_t ua; memcpy(&ua, &a, 4);
    int e = (int)(ua >> 23) - 127;
    uint32_t m = (ua & 0x7fffffu) >> 20;
    const uint32_t rem = ua & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    if (m == 8u) { m = 0; ++e; }
    if (e > 8 || (e == 8 && m == 7u)) return sign | 0x7e;
    return sign | (uint8_t)(((e + 7) << 3) | m);
}
static const int kPerm[8] = {0, 2, 4, 6, 1, 3, 5, 7};  // stored byte i of a group of 8 = element kPerm[i]
static const float kE2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static float e2m1(int c) { return (c & 8 ? -1.f : 1.f) * kE2m1[c & 7]; }

static std::vector<float> g_e4;  // e4m3 -> f32

struct Ev {
    hipEvent_t a, b;
    Ev() { HC(hipEventCreate(&a)); HC(hipEventCreate(&b)); }
};
template <class F> static double time_ms(F&& f, int reps) {
    Ev ev;
    f();
    HC(hipDeviceSynchronize());
    HC(hipEventRecord(ev.a, nullptr));
    for (int i = 0; i < reps; ++i) f();
    HC(hipEventRecord(ev.b, nullptr));
    HC(hipEventSynchronize(ev.b));
    float ms;
    HC(hipEventElapsedTime(&ms, ev.a, ev.b));
    return ms / reps;
}

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? atoll(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 10;
    const int skew = argc > 3 ? atoi(argv[3]) : 0;
    const int check = argc > 4 ? atoi(argv[4]) : 1;
    const int ablate = argc > 5 ? atoi(argv[5]) : 0;
    const int64_t E = 256, H = 2048, I = 512, K = 8, P = T * K;
    g_e4.resize(256);
    for (int i = 0; i < 256; ++i) g_e4[i] = e4m3f((uint8_t)i);

    // ---- host data
    std::vector<uint16_t> hx(T * H);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < H; ++k) {
            float v = nrm();
            if (k % 256 == 17) v *= 8.f;  // outlier channels
            hx[t * H + k] = bf16bits(v);
        }
        if (t % 97 == 5) for (int64_t k = 0; k < H; ++k) hx[t * H + k] = bf16bits(bf2f(hx[t * H + k]) * 30.f);  // a loud token
    }
    std::vector<uint8_t> hguw((size_t)E * 2 * I * (H / 2)), hgus((size_t)E * 2 * I * (H / 16)), hdw((size_t)E * H * (I / 2)), hds((size_t)E * H * (I / 16));
    for (auto& b : hguw) b = (uint8_t)rnd();
    for (auto& b : hdw) b = (uint8_t)rnd();
    auto rs_byte = [] { return (uint8_t)((3 + rnd() % 4) << 3 | (rnd() & 7)); };  // e4m3 scales 2^-4 .. 2^-1 * (1 + m/8)
    for (auto& b : hgus) b = rs_byte();
    for (auto& b : hds) b = rs_byte();
    std::vector<float> hgs(E * 3);
    for (auto& g : hgs) g = 0.5f + urand();
    // routing: K distinct experts per token; skew = 1: a few hot experts
    std::vector<int> hids(P);
    std::vector<float> hw(P);
    {
        std::vector<double> pw(E);
        for (int e = 0; e < E; ++e) pw[e] = skew ? 1.0 / pow(1.0 + (e * 7919 % E), 0.9) : 1.0;
        for (int64_t t = 0; t < T; ++t) {
            double tot = 0;
            for (int e = 0; e < E; ++e) tot += pw[e];
            std::vector<double> w = pw;
            for (int k = 0; k < K; ++k) {
                double r = urand() * tot, acc = 0;
                int pick = E - 1;
                for (int e = 0; e < E; ++e) { if (w[e] <= 0) continue; acc += w[e]; if (acc >= r) { pick = e; break; } }
                hids[t * K + k] = pick;
                tot -= w[pick];
                w[pick] = 0;
                hw[t * K + k] = 1.f / K;
            }
        }
    }

    // ---- device
    ik_bf16 *dx, *dhb, *dyb, *dhf, *dyf;
    uint8_t *dguw, *dgus, *ddw, *dds, *dqx, *dqh;
    float *dgs, *dqxs, *dqhs;
    int *dids, *dpairs, *doffs, *dact;
    HC(hipMalloc(&dx, T * H * 2));
    HC(hipMalloc(&dhb, P * I * 2)); HC(hipMalloc(&dyb, P * H * 2));
    HC(hipMalloc(&dhf, P * I * 2)); HC(hipMalloc(&dyf, P * H * 2));
    HC(hipMalloc(&dguw, hguw.size())); HC(hipMalloc(&dgus, hgus.size())); HC(hipMalloc(&ddw, hdw.size())); HC(hipMalloc(&dds, hds.size()));
    HC(hipMalloc(&dqx, T * H)); HC(hipMalloc(&dqxs, T * 4)); HC(hipMalloc(&dqh, P * I)); HC(hipMalloc(&dqhs, P * 4));
    HC(hipMalloc(&dgs, E * 3 * 4)); HC(hipMalloc(&dids, P * 4));
    HC(hipMalloc(&dpairs, P * 4)); HC(hipMalloc(&doffs, (E + 1) * 4)); HC(hipMalloc(&dact, (E + 1) * 4));
    HC(hipMemcpy(dx, hx.data(), T * H * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dguw, hguw.data(), hguw.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dgus, hgus.data(), hgus.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(ddw, hdw.data(), hdw.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dds, hds.data(), hds.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dgs, hgs.data(), E * 3 * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dids, hids.data(), P * 4, hipMemcpyHostToDevice));
    hipStream_t s = nullptr;
    const unsigned slots = (unsigned)std::min<int64_t>(P, E);
    moe_group_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), s>>>(dids, (int)P, (int)E, dpairs, doffs, dact);
    HC(hipDeviceSynchronize());
    std::vector<int> hoffs(E + 1), hpairs(P);
    HC(hipMemcpy(hoffs.data(), doffs, (E + 1) * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hpairs.data(), dpairs, P * 4, hipMemcpyDeviceToHost));
    int maxc = 0, nact = 0;
    for (int e = 0; e < E; ++e) { maxc = std::max(maxc, hoffs[e + 1] - hoffs[e]); nact += hoffs[e + 1] > hoffs[e]; }
    printf("T = %lld  pairs = %lld  active experts = %d  avg %.1f max %d pairs/expert  skew=%d  reps=%d\n", (long long)T, (long long)P, nact, (double)P / nact, maxc, skew, reps);

    const double fl_gu = 2.0 * P * (2.0 * I) * H, fl_dn = 2.0 * P * H * I;
    unsigned zt64 = (unsigned)((T + 63) / 64);
    {
        const int64_t avg = (P + E - 1) / E;
        const unsigned cap = (unsigned)std::max<int64_t>(1, ((int64_t)moe_zmul() * avg + 63) / 64);
        if (cap < zt64) zt64 = cap;
    }
    // ---- r7 (the defaults of the 64-pair tile)
    auto r7_gu = [&] { moe_wmma_launch_p<true, 64, kMwDefault64>(I, P, slots, zt64, dx, dguw, dgus, dgs, dpairs, doffs, dact, dhb, E, H, K, s); };
    auto r7_dn = [&] { moe_wmma_launch_p<false, 64, kMwDefault64Dn>(H, P, slots, zt64, dhb, ddw, dds, dgs, dpairs, doffs, dact, dyb, E, I, K, s); };
    const double t_r7_gu = time_ms(r7_gu, reps), t_r7_dn = time_ms(r7_dn, reps);
    printf("r7 bf16     gate_up %8.3f ms %7.1f TFLOP/s | down %8.3f ms %7.1f TFLOP/s | total %8.3f ms %7.1f TFLOP/s\n", t_r7_gu, fl_gu / t_r7_gu / 1e9,
           t_r7_dn, fl_dn / t_r7_dn / 1e9, t_r7_gu + t_r7_dn, (fl_gu + fl_dn) / (t_r7_gu + t_r7_dn) / 1e9);

    // ---- fp8: quantisers + kernels per variant
    auto q_x = [&] { moe_f8_quant_launch(dx, dqx, dqxs, T, H, s); };
    auto q_h = [&] { moe_f8_quant_launch(dhf, dqh, dqhs, P, I, s); };
    const double t_qx = time_ms(q_x, reps);
    // first fill dhf with a real h (the fp8 gate_up with the default variant) so that q_h quantises realistic rows
    moe_f8_launch_p<true, 64, kMf8Skip>(I, slots, zt64, dqx, dqxs, dguw, dgus, dgs, dpairs, doffs, dact, dhf, E, H, K, s);
    HC(hipDeviceSynchronize());
    const double t_qh = time_ms(q_h, reps);
    printf("quant       x %.3f ms (%lld rows)   h %.3f ms (%lld rows)\n", t_qx, (long long)T, t_qh, (long long)P);
    struct Var { const char* name; int pairs, v; };
    double best_total = 1e30;
    const char* best_name = "";
#define IK_VAR(NAME, PAIRS_, V_)                                                                                                               \
    {                                                                                                                                          \
        unsigned zt = (unsigned)((T + PAIRS_ - 1) / PAIRS_);                                                                                   \
        const int64_t avg = (P + E - 1) / E;                                                                                                   \
        const unsigned cap = (unsigned)std::max<int64_t>(1, ((int64_t)moe_zmul() * avg + PAIRS_ - 1) / PAIRS_);                                \
        if (cap < zt) zt = cap;                                                                                                                \
        auto gu = [&] { moe_f8_launch_p<true, PAIRS_, V_>(I, slots, zt, dqx, dqxs, dguw, dgus, dgs, dpairs, doffs, dact, dhf, E, H, K, s); };   \
        auto dn = [&] { moe_f8_launch_p<false, PAIRS_, V_>(H, slots, zt, dqh, dqhs, ddw, dds, dgs, dpairs, doffs, dact, dyf, E, I, K, s); };    \
        const double tg = time_ms(gu, reps), td = time_ms(dn, reps);                                                                          \
        const double tot = t_qx + tg + t_qh + td;                                                                                              \
        printf("fp8 %-9s gate_up %8.3f ms %7.1f TFLOP/s | down %8.3f ms %7.1f TFLOP/s | +quant total %8.3f ms %7.1f TFLOP/s  x%.2f vs r7\n", NAME, tg,  \
               fl_gu / tg / 1e9, td, fl_dn / td / 1e9, tot, (fl_gu + fl_dn) / tot / 1e9, (t_r7_gu + t_r7_dn) / tot);                           \
        if (tot < best_total) { best_total = tot; best_name = NAME; }                                                                          \
    }
    IK_VAR("p64 v0", 64, 0)
    IK_VAR("p64 skip", 64, kMf8Skip)
    IK_VAR("p64 sk+dp", 64, kMf8Skip | kMf8Deep)
    IK_VAR("p64 sk+zf", 64, kMf8Skip | kMf8ZFast)
    IK_VAR("p64 s+d+zf", 64, kMf8Skip | kMf8Deep | kMf8ZFast)
    IK_VAR("p128 skip", 128, kMf8Skip)
    IK_VAR("p128 sk+zf", 128, kMf8Skip | kMf8ZFast)
    if (ablate) {
        IK_VAR("sz NoW", 64, kMf8Skip | kMf8ZFast | kMf8NoW)
        IK_VAR("sz NoFma", 64, kMf8Skip | kMf8ZFast | kMf8NoFma)
        IK_VAR("sz NoLds", 64, kMf8Skip | kMf8ZFast | kMf8NoLds)
        IK_VAR("sz NoDec", 64, kMf8Skip | kMf8ZFast | kMf8NoDec)
        IK_VAR("sz NoW+Fma", 64, kMf8Skip | kMf8ZFast | kMf8NoW | kMf8NoFma)
        IK_VAR("sz all-off", 64, kMf8Skip | kMf8ZFast | kMf8NoW | kMf8NoFma | kMf8NoLds | kMf8NoDec)
    }
#undef IK_VAR
    printf("best fp8 variant: %s  %.3f ms (r7 %.3f ms, x%.2f)\n", best_name, best_total, t_r7_gu + t_r7_dn, (t_r7_gu + t_r7_dn) / best_total);

    if (!check) return 0;
    // ---- outputs of the default variant (p64 skip) for the checks
    HC(hipMemset(dhf, 0, P * I * 2));
    HC(hipMemset(dyf, 0, P * H * 2));
    q_x();
    moe_f8_launch_p<true, 64, kMf8Skip>(I, slots, zt64, dqx, dqxs, dguw, dgus, dgs, dpairs, doffs, dact, dhf, E, H, K, s);
    q_h();
    moe_f8_launch_p<false, 64, kMf8Skip>(H, slots, zt64, dqh, dqhs, ddw, dds, dgs, dpairs, doffs, dact, dyf, E, I, K, s);
    r7_gu();
    r7_dn();
    HC(hipDeviceSynchronize());
    std::vector<uint8_t> hqx(T * H), hqh(P * I);
    std::vector<float> hqxs(T), hqhs(P);
    std::vector<uint16_t> hh(P * I), hy(P * H), hhb(P * I), hyb(P * H);
    HC(hipMemcpy(hqx.data(), dqx, T * H, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hqxs.data(), dqxs, T * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hqh.data(), dqh, P * I, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hqhs.data(), dqhs, P * 4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hh.data(), dhf, P * I * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hy.data(), dyf, P * H * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hhb.data(), dhb, P * I * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hyb.data(), dyb, P * H * 2, hipMemcpyDeviceToHost));

    // (1) quantiser vs host RNE: x rows
    {
        size_t bad = 0, badsc = 0;
        for (int64_t t = 0; t < T; ++t) {
            float amax = 0;
            for (int64_t k = 0; k < H; ++k) amax = std::max(amax, fabsf(bf2f(hx[t * H + k])));
            const float sc = std::max(amax / 448.f, 1e-20f);
            if (sc != hqxs[t]) ++badsc;
            for (int64_t g = 0; g < H / 8; ++g)
                for (int i = 0; i < 8; ++i) {
                    const float y = std::min(std::max(bf2f(hx[t * H + g * 8 + kPerm[i]]) / sc, -448.f), 448.f);
                    if (f2e4m3(y) != hqx[t * H + g * 8 + i]) ++bad;
                }
        }
        printf("quant x check: %zu of %lld bytes differ, %zu of %lld scales differ (host RNE e4m3)\n", bad, (long long)(T * H), badsc, (long long)T);
    }
    // (2) f64 reference of 3 experts over the device's own e4m3 inputs
    auto wval = [&](const std::vector<uint8_t>& w, const std::vector<uint8_t>& sw, int64_t row, int64_t k, int64_t Kd) {
        const uint8_t b = w[(size_t)row * (Kd / 2) + k / 2];
        const int c = (k & 1) ? (b >> 4) : (b & 15);
        return (double)e2m1(c) * g_e4[sw[(size_t)row * (Kd / 16) + k / 16]];
    };
    // element k of an activation row stored permuted: byte index in the row
    auto qidx = [&](int64_t k) { int g = (int)(k / 8), j = (int)(k % 8); int i = 0; while (kPerm[i] != j) ++i; return g * 8 + i; };
    int experts[3] = {0, 100 % (int)E, (int)E - 1};
    {
        // the busiest expert instead of 100
        int best = 0;
        for (int e = 0; e < E; ++e) if (hoffs[e + 1] - hoffs[e] > hoffs[best + 1] - hoffs[best]) best = e;
        experts[1] = best;
    }
    double max_ulp_gu = 0, max_ulp_dn = 0, max_rel_gu = 0, max_rel_dn = 0;
    size_t ndiff_gu = 0, nout_gu = 0, ndiff_dn = 0, nout_dn = 0;
    std::vector<int> qix(std::max(H, I));
    for (int64_t k = 0; k < std::max(H, I); ++k) qix[k] = qidx(k);
    for (int ei = 0; ei < 3; ++ei) {
        const int e = experts[ei];
        const int b = hoffs[e], c = hoffs[e + 1] - hoffs[e];
        const int nsamp = std::min(c, 48);
        for (int pi = 0; pi < nsamp; ++pi) {
            const int pair = hpairs[b + pi * c / nsamp];
            const int64_t tok = pair / K;
            // gate_up
            for (int64_t n = 0; n < I; n += 1) {
                if ((n % 7) != (pair % 7) && n > 31) continue;  // sparse sample of features
                double g = 0, u = 0;
                for (int64_t k = 0; k < H; ++k) {
                    const double xv = (double)g_e4[hqx[tok * H + qix[k]]];
                    g += xv * wval(hguw, hgus, ((int64_t)e * 2 + 0) * I + n, k, H);
                    u += xv * wval(hguw, hgus, ((int64_t)e * 2 + 1) * I + n, k, H);
                }
                const float a = hqxs[tok];
                const float gf = rbf((float)(g * a) * hgs[e * 3 + 0]), uf = rbf((float)(u * a) * hgs[e * 3 + 1]);
                const float sf = rbf(gf / (1.f + expf(-gf)));
                const float ref = sf * uf;
                const float got = bf2f(hh[(size_t)pair * I + n]);
                const double ulp = fabs(ref) > 0 ? ldexp(1.0, ilogb(fabs(ref)) - 7) : 1e-30;
                max_ulp_gu = std::max(max_ulp_gu, fabs(got - ref) / ulp);
                max_rel_gu = std::max(max_rel_gu, fabs(got - ref) / std::max(fabs((double)ref), 1e-3));
                ndiff_gu += bf16bits(got) != bf16bits(rbf(ref)); ++nout_gu;
            }
            // down: input is the device's quantised h
            for (int64_t n = 0; n < H; n += 1) {
                if ((n % 11) != (pair % 11) && n > 31) continue;
                double d = 0;
                for (int64_t k = 0; k < I; ++k) d += (double)g_e4[hqh[(size_t)pair * I + qix[k]]] * wval(hdw, hds, (int64_t)e * H + n, k, I);
                const float ref = rbf((float)(d * hqhs[pair]) * hgs[e * 3 + 2]);
                const float got = bf2f(hy[(size_t)pair * H + n]);
                const double ulp = fabs(ref) > 0 ? ldexp(1.0, ilogb(fabs(ref)) - 7) : 1e-30;
                max_ulp_dn = std::max(max_ulp_dn, fabs(got - ref) / ulp);
                max_rel_dn = std::max(max_rel_dn, fabs(got - ref) / std::max(fabs((double)ref), 1e-3));
                ndiff_dn += bf16bits(got) != bf16bits(ref); ++nout_dn;
            }
        }
    }
    printf("reference (f64 over the same e4m3 inputs; experts %d %d %d): gate_up %zu of %zu bf16 outputs differ, max %.2f bf16 ulp | down %zu of %zu differ, max %.2f ulp\n",
           experts[0], experts[1], experts[2], ndiff_gu, nout_gu, max_ulp_gu, ndiff_dn, nout_dn, max_ulp_dn);
    // (3) fp8 vs bf16 r7 (relative Frobenius error over everything)
    {
        double n1 = 0, d1 = 0, n2 = 0, d2 = 0;
        for (size_t i = 0; i < hh.size(); ++i) { const double a = bf2f(hh[i]), b = bf2f(hhb[i]); n1 += (a - b) * (a - b); d1 += b * b; }
        for (size_t i = 0; i < hy.size(); ++i) { const double a = bf2f(hy[i]), b = bf2f(hyb[i]); n2 += (a - b) * (a - b); d2 += b * b; }
        printf("fp8 vs bf16 r7 (relative Frobenius): gate_up h %.4f   down y (own h) %.4f  [down compares y of different h inputs]\n", sqrt(n1 / d1), sqrt(n2 / d2));
    }
    // (4) the down projection fed the SAME h: r7 down on the fp8 gate_up output vs fp8 down
    {
        HC(hipMemcpy(dhb, dhf, P * I * 2, hipMemcpyDeviceToDevice));
        r7_dn();
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(hyb.data(), dyb, P * H * 2, hipMemcpyDeviceToHost));
        double n2 = 0, d2 = 0;
        for (size_t i = 0; i < hy.size(); ++i) { const double a = bf2f(hy[i]), b = bf2f(hyb[i]); n2 += (a - b) * (a - b); d2 += b * b; }
        printf("down on the same h: fp8 vs bf16 r7 relative Frobenius %.4f (h quantisation alone)\n", sqrt(n2 / d2));
    }
    return 0;
}
