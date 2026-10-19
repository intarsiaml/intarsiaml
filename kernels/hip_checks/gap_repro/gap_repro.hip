// Standalone repro: GPU idle gap after a long (>= ~1 ms) kernel on gfx1201 (Radeon AI PRO R9700). No dependencies beyond HIP.
// M launches of a spin kernel (spins until a wall-clock deadline, so the duration is exact) go to ONE stream with no host sync between them (the queue is full, the next
// dispatch was enqueued long before). Each launch writes its own start/end with wall_clock64() (min block start, max block end);
// gap[i] = start[i+1] - end[i]. Modes: alu (FMA chains), mem (GAP_MB=128 streaming read), wmma (WMMA + 32 KB LDS, like the engine's GEMMs).
// Grids: 1 block, one block per CU, two per CU. Also a mixed 2 ms / 0.5 ms sequence. GAP_PLAN=file dumps the launch plan so that
// rocprofv3 --kernel-trace output can be matched to it (run.sh).   hipcc -O3 --offload-arch=gfx1201 gap_repro.hip -o gap_repro
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

template <int MODE>
__global__ void __launch_bounds__(256) spin_k(unsigned long long* ts, float* sink, const uint4* buf, unsigned mask, unsigned long long dur) {
    __shared__ half8 lds[2048];  // 32 KB
    const unsigned long long t0 = wall_clock64();
    const int t = threadIdx.x;
    float acc = (float)t, b = acc + 1, c = acc + 2, d = acc + 3;
    unsigned x = 0, i = 0;
    half8 ha = {}, hb = {};
    float8 wc = {};
    if (MODE == 2) {
        for (int j = 0; j < 8; ++j) lds[t + 256 * j] = half8{(_Float16)t, 1, 1, 1, 1, 1, 1, 1};
        __syncthreads();
        ha = lds[t]; hb = lds[255 - t];
    }
    do {  // spin until `dur` wall-clock ticks after this block's start, doing the mode's work in between clock reads
        for (int u = 0; u < 16; ++u, ++i) {
            if (MODE == 0) { acc = fmaf(acc, 0.9999f, 0.5f); b = fmaf(b, 0.9998f, 0.4f); c = fmaf(c, 0.9997f, 0.3f); d = fmaf(d, 0.9996f, 0.2f); }
            else if (MODE == 1) { uint4 v = buf[((blockIdx.x * 256 + t) + i * (gridDim.x * 256)) & mask]; x ^= v.x ^ v.y ^ v.z ^ v.w; }
            else {
                wc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ha, hb, wc);
                ha = lds[(t + i * 37) & 2047]; lds[(t * 5 + i) & 2047] = hb; hb = lds[(t * 3 + i * 11) & 2047];
            }
        }
    } while (wall_clock64() - t0 < dur);
    acc += b + c + d + (float)x + wc[0] + wc[7];
    if (acc == 1234.5f) sink[0] = acc;  // keep the work alive
    __syncthreads();
    if (t == 0) { atomicMin(&ts[0], t0); atomicMax(&ts[1], wall_clock64()); }
}

static double wc_mhz; static float* sink; static uint4* buf; static unsigned mask;
static void launch(int mode, int grid, unsigned long long dur, unsigned long long* ts) {
    if (mode == 0) spin_k<0><<<grid, 256>>>(ts, sink, buf, mask, dur);
    else if (mode == 1) spin_k<1><<<grid, 256>>>(ts, sink, buf, mask, dur);
    else spin_k<2><<<grid, 256>>>(ts, sink, buf, mask, dur);
}
static double pct(std::vector<double>& v, double p) { std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; }

int main() {
    int clk = 100000; (void)hipDeviceGetAttribute(&clk, hipDeviceAttributeWallClockRate, 0); wc_mhz = clk / 1000.0;
    hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0)); const int ncu = pr.multiProcessorCount;
    printf("device %s %s, %d CUs, wall clock %.1f MHz\n", pr.name, pr.gcnArchName, ncu, wc_mhz);
    const size_t bytes = (size_t)(getenv("GAP_MB") ? atoi(getenv("GAP_MB")) : 128) << 20; mask = bytes / 16 - 1;  // GAP_MB: a power of two, > the 64 MB infinity cache
    CK(hipMalloc(&buf, bytes)); CK(hipMemset(buf, 1, bytes)); CK(hipMalloc(&sink, 4));
    const int M = 100, NTS = 2 * M;
    unsigned long long* ts; CK(hipMalloc(&ts, NTS * 8));
    FILE* plan = getenv("GAP_PLAN") ? fopen(getenv("GAP_PLAN"), "w") : nullptr;
    const char* mn[] = {"alu", "mem", "wmma"}; const int grids[] = {1, ncu, 2 * ncu};
    const double durs[] = {0.5, 0.9, 1.0, 1.1, 1.5, 2, 3};
    std::map<std::string, std::vector<double>> gaps, kd; std::vector<std::string> order; int exp = 0;
    std::vector<unsigned long long> h(NTS);
    auto fetch = [&] { CK(hipMemcpy(h.data(), ts, NTS * 8, hipMemcpyDeviceToHost)); };
    auto reset = [&] { for (int i = 0; i < M; ++i) { h[2 * i] = ~0ull; h[2 * i + 1] = 0; } CK(hipMemcpy(ts, h.data(), NTS * 8, hipMemcpyHostToDevice)); };
    auto dur_ms = [&](int i) { return (double)(h[2 * i + 1] - h[2 * i]) / wc_mhz / 1000.0; };
    for (int mode = 0; mode < 3; ++mode) for (int grid : grids) {
        for (int seq = 0; seq < 2; ++seq) for (double d : durs) {
            if (seq == 1 && d != 2) continue;  // mixed sequence once: 2 ms, 0.5 ms, 2 ms, ...
            std::vector<double> dd(M); for (int i = 0; i < M; ++i) dd[i] = seq ? (i % 2 ? 0.5 : 2.0) : d;
            reset();
            for (int i = 0; i < M; ++i) {  // no host sync between launches
                launch(mode, grid, (unsigned long long)(dd[i] * 1000 * wc_mhz), ts + 2 * i);
                if (plan) fprintf(plan, "%d %s %d %s%.1f\n", exp, mn[mode], grid, seq ? "mix:" : "", dd[i]);
            }
            CK(hipStreamSynchronize(0)); ++exp; fetch();
            for (int i = 0; i + 1 < M; ++i) {
                char k[64]; snprintf(k, sizeof k, "%-5s %5d %-8s", mn[mode], grid, (std::string(seq ? "mix:" : "") + std::to_string(dd[i]).substr(0, 3)).c_str());
                if (!gaps.count(k)) order.push_back(k);
                gaps[k].push_back(((double)h[2 * i + 2] - (double)h[2 * i + 1]) / wc_mhz);  // us
                kd[k].push_back(dur_ms(i));
            }
        }
    }
    if (plan) fclose(plan);
    printf("# device timestamps; gap = start[i+1] - end[i] in us, back-to-back launches on one stream, no host sync\n");
    printf("%-5s %5s %-8s %5s %8s %9s %9s %9s %9s %7s\n", "mode", "grid", "dur_ms", "n", "kdur_ms", "gap_mean", "gap_p50", "gap_p90", "gap_max", ">100us");
    for (auto& k : order) {
        auto& g = gaps[k]; double m = 0, kdm = 0; int big = 0;
        for (double x : g) { m += x; big += x > 100; }
        for (double x : kd[k]) kdm += x;
        printf("%s %5zu %8.3f %9.1f %9.1f %9.1f %9.1f %6.0f%%\n", k.c_str(), g.size(), kdm / g.size(), m / g.size(), pct(g, .5), pct(g, .9), pct(g, 1), 100.0 * big / g.size());
    }
    return 0;
}
