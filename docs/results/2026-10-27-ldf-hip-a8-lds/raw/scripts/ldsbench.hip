// Lane LDF micro-benchmark (one R9700, gfx1201): the LDS fragment-read + fp8 WMMA inner loop of matmul_mxfp4_a8_mma without the global loads.
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1201 -o ldsbench ldsbench.hip      Run: ./ldsbench [filter]
// Each row is one variant: tile WN x WM waves, TN x TM fragments per wave, LDS row stride LD (bytes), SW (XOR swizzle of the 16 B chunks, LD 64),
// PIPE (0: read, then WMMA, per 32 k; 1: both 32 k halves are read before the first WMMA), BAR (a block barrier per 64 k stage), ST (LDS stores of
// the next stage's tile, as the kernel's sstore), RD (0: one ds_read_b128 per fragment, 1: two ds_read_b64), KS (k per sub-step reads: 32).
// Prints TFLOP/s of the WMMAs (peak 366) and, for the read-only rows, GB/s of LDS reads.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v8f wmma(uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi, v8f c) {
    const uint2 a = make_uint2(alo, ahi), b = make_uint2(blo, bhi);
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(v2i, a), __builtin_bit_cast(v2i, b), c);
}

template <int LD, int SW>
__device__ __forceinline__ int fo(int r, int k) {  // byte offset of row r, k (a multiple of 16) in a tile
    if constexpr (SW == 0) return r * LD + k;
    else return r * 64 + ((((k >> 4) & 3) ^ ((r >> 1) & 3)) << 4);
}

template <int RD>
__device__ __forceinline__ uint4 rdfrag(const uint8_t* p0, int o) {
    if constexpr (RD == 0 || RD == 2) return *reinterpret_cast<const uint4*>(p0 + o);
    else {
        const uint2 lo = *reinterpret_cast<const uint2*>(p0 + o), hi = *reinterpret_cast<const uint2*>(p0 + o + 8);
        return make_uint4(lo.x, lo.y, hi.x, hi.y);
    }
}

template <int WN, int WM, int TN, int TM, int LD, int SW, int PIPE, int BAR, int ST, int RD, int WM_ON>
__global__ __launch_bounds__(WN * WM * 32) void inner(float* out, int iters, int rot) {
    constexpr int NT = WN * WM * 32, BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int SWB = SW ? 64 : LD;
    __shared__ __align__(16) uint8_t sw[2][BN * SWB];
    __shared__ __align__(16) uint8_t sx[2][BM * SWB];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, wn = wave % WN, wm = wave / WN;
    for (int i = tid; i < (int)(sizeof(sw) / 16); i += NT) reinterpret_cast<uint4*>(sw)[i] = make_uint4(i * 0x01010101u & 0x77777777u, 0x30303030u, i, 0x38383838u);
    for (int i = tid; i < (int)(sizeof(sx) / 16); i += NT) reinterpret_cast<uint4*>(sx)[i] = make_uint4(0x30303030u, i * 0x03030303u & 0x77777777u, 0x34343434u, i);
    __syncthreads();
    v8f acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j) acc[i][j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    const int lr = lane & 15, lh = (lane >> 4) * 16;
    uint4 af0[TN];  // RD 2: the weight fragments live in registers (no LDS reads for them), to bound a design that bypasses LDS for one operand
#pragma unroll
    for (int i = 0; i < TN; ++i) af0[i] = rdfrag<0>(sw[0], fo<LD, SW>(wn * TN * 16 + i * 16 + lr, lh));
    for (int it = 0; it < iters; ++it) {
        const int buf = (it + rot) & 1;
        const uint8_t* pw = sw[buf];
        const uint8_t* px = sx[buf];
        uint4 af[2][TN], bf[2][TM];
        auto rd = [&](int h) {
#pragma unroll
            for (int i = 0; i < TN; ++i) {
                if constexpr (RD == 2) af[h][i] = af0[i];
                else af[h][i] = rdfrag<RD>(pw, fo<LD, SW>(wn * TN * 16 + i * 16 + lr, h * 32 + lh));
            }
#pragma unroll
            for (int j = 0; j < TM; ++j) bf[h][j] = rdfrag<RD>(px, fo<LD, SW>(wm * TM * 16 + j * 16 + lr, h * 32 + lh));
        };
        auto mm = [&](int h) {
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = wmma(af[h][i].x, af[h][i].y, bf[h][j].x, bf[h][j].y, acc[i][j]);
                    acc[i][j] = wmma(af[h][i].z, af[h][i].w, bf[h][j].z, bf[h][j].w, acc[i][j]);
                }
        };
        auto stores = [&]() {
            if constexpr (ST) {  // the next stage's tile into the other buffer, 16 B pieces (what sstore writes)
                constexpr int CHN = BN * 4, CHM = BM * 4;
                for (int c = tid; c < CHN; c += NT) *reinterpret_cast<uint4*>(&sw[buf ^ 1][fo<LD, SW>(c >> 2, (c & 3) * 16)]) = make_uint4(c, it, c, it);
                for (int c = tid; c < CHM; c += NT) *reinterpret_cast<uint4*>(&sx[buf ^ 1][fo<LD, SW>(c >> 2, (c & 3) * 16)]) = make_uint4(it, c, it, c);
            }
        };
        if constexpr (PIPE == 2) {  // split barrier: both halves read, WMMAs of half 0, the stores, signal, WMMAs of half 1, wait
            rd(0);
            rd(1);
            mm(0);
            stores();
            asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
            mm(1);
            asm volatile("s_barrier_wait -1" ::: "memory");
            continue;
        }
        if constexpr (WM_ON) {
            if constexpr (PIPE == 1) {
                rd(0);
                rd(1);
                mm(0);
                mm(1);
            } else {
                rd(0);
                mm(0);
                rd(1);
                mm(1);
            }
        } else {  // reads only: fold them into acc[0][0] with xor so they are not removed
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                rd(h);
                uint32_t x = 0;
#pragma unroll
                for (int i = 0; i < TN; ++i) x ^= af[h][i].x ^ af[h][i].y ^ af[h][i].z ^ af[h][i].w;
#pragma unroll
                for (int j = 0; j < TM; ++j) x ^= bf[h][j].x ^ bf[h][j].y ^ bf[h][j].z ^ bf[h][j].w;
                acc[0][0][h] += __uint_as_float(x & 0x7fffffu);
            }
        }
        if constexpr (ST) {  // the next stage's tile into the other buffer, 16 B pieces (what sstore writes)
            constexpr int CHN = BN * 4, CHM = BM * 4;
            for (int c = tid; c < CHN; c += NT) *reinterpret_cast<uint4*>(&sw[buf ^ 1][fo<LD, SW>(c >> 2, (c & 3) * 16)]) = make_uint4(c, it, c, it);
            for (int c = tid; c < CHM; c += NT) *reinterpret_cast<uint4*>(&sx[buf ^ 1][fo<LD, SW>(c >> 2, (c & 3) * 16)]) = make_uint4(it, c, it, c);
        }
        if constexpr (BAR) __syncthreads();
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) s += acc[i][j][e];
    if (s == 12345.f) out[0] = s;  // never true in practice
}

struct Row {
    std::string name;
    void (*fn)(float*, int, int);
    int nt, wmmaPerIter, readB, ldsBytes, wmmaOn;
};

#define ROW(NAME, WN, WM, TN, TM, LD, SW, PIPE, BAR, ST, RD, ON)                                                                           \
    rows.push_back({NAME, inner<WN, WM, TN, TM, LD, SW, PIPE, BAR, ST, RD, ON>, WN* WM * 32, WN* WM * 2 * TN * TM * 2,                    \
                    WN* WM * 2 * (TN + TM) * 512, 2 * ((WN * TN * 16 + WM * TM * 16) * (SW ? 64 : LD)), ON})

int main(int argc, char** argv) {
    std::string filter = argc > 1 ? argv[1] : "";
    std::vector<Row> rows;
    // name: WNxWM tiles TNxTM  LD  flags
    ROW("base 4x4w 2x4f LD80 b128 bar st", 4, 4, 2, 4, 80, 0, 0, 1, 1, 0, 1);
    ROW("pipe 4x4w 2x4f LD80 b128 bar st", 4, 4, 2, 4, 80, 0, 1, 1, 1, 0, 1);
    ROW("SPLIT 4x4w 2x4f LD80 b128 st", 4, 4, 2, 4, 80, 0, 2, 1, 1, 0, 1);
    ROW("SPLIT swz 4x4w 2x4f st", 4, 4, 2, 4, 64, 1, 2, 1, 1, 0, 1);
    ROW("SPLIT b64 4x4w 2x4f LD80 st", 4, 4, 2, 4, 80, 0, 2, 1, 1, 1, 1);
    ROW("SPLIT swz b64 4x4w 2x4f st", 4, 4, 2, 4, 64, 1, 2, 1, 1, 1, 1);
    ROW("swz b64 4x4w 2x4f bar st", 4, 4, 2, 4, 64, 1, 0, 1, 1, 1, 1);
    ROW("SPLIT 2x4w 4x4f LD80 st", 2, 4, 4, 4, 80, 0, 2, 1, 1, 0, 1);
    ROW("SPLIT swz 2x4w 4x4f st", 2, 4, 4, 4, 64, 1, 2, 1, 1, 0, 1);
    ROW("nobar 4x4w 2x4f LD80 b128 st", 4, 4, 2, 4, 80, 0, 0, 0, 1, 0, 1);
    ROW("nostore 4x4w 2x4f LD80 b128 bar", 4, 4, 2, 4, 80, 0, 0, 1, 0, 0, 1);
    ROW("pure 4x4w 2x4f LD80 b128 (nobar nost)", 4, 4, 2, 4, 80, 0, 0, 0, 0, 0, 1);
    ROW("pure-pipe 4x4w 2x4f LD80 b128", 4, 4, 2, 4, 80, 0, 1, 0, 0, 0, 1);
    ROW("b64 4x4w 2x4f LD80 bar st", 4, 4, 2, 4, 80, 0, 0, 1, 1, 1, 1);
    ROW("ld64sw 4x4w 2x4f swz bar st", 4, 4, 2, 4, 64, 1, 0, 1, 1, 0, 1);
    ROW("ld64sw-pure 4x4w 2x4f swz", 4, 4, 2, 4, 64, 1, 0, 0, 0, 0, 1);
    ROW("ld64 4x4w 2x4f LD64 nopad pure", 4, 4, 2, 4, 64, 0, 0, 0, 0, 0, 1);
    ROW("ld72 4x4w 2x4f LD72 pure", 4, 4, 2, 4, 72, 0, 0, 0, 0, 0, 1);
    ROW("2x4w 4x4f LD80 bar st", 2, 4, 4, 4, 80, 0, 0, 1, 1, 0, 1);
    ROW("2x4w 4x4f LD80 pure", 2, 4, 4, 4, 80, 0, 0, 0, 0, 0, 1);
    ROW("2x4w 4x4f LD80 pure-pipe", 2, 4, 4, 4, 80, 0, 1, 0, 0, 0, 1);
    ROW("4x2w 2x4f(128x128) LD80 bar st", 4, 2, 2, 4, 80, 0, 0, 1, 1, 0, 1);
    ROW("4x4w 2x2f LD80 bar st", 4, 4, 2, 2, 80, 0, 0, 1, 1, 0, 1);
    ROW("Areg 4x4w 2x4f LD80 bar st (weights from regs)", 4, 4, 2, 4, 80, 0, 0, 1, 1, 2, 1);
    ROW("Areg 4x4w 2x4f LD80 pure", 4, 4, 2, 4, 80, 0, 0, 0, 0, 2, 1);
    ROW("Areg-pipe 4x4w 2x4f LD80 pure", 4, 4, 2, 4, 80, 0, 1, 0, 0, 2, 1);
    // LDS read bandwidth only (no WMMA)
    ROW("READ 4x4w 2x4f LD80 b128", 4, 4, 2, 4, 80, 0, 0, 0, 0, 0, 0);
    ROW("READ 4x4w 2x4f LD80 b64", 4, 4, 2, 4, 80, 0, 0, 0, 0, 1, 0);
    ROW("READ 4x4w 2x4f swz b128", 4, 4, 2, 4, 64, 1, 0, 0, 0, 0, 0);
    ROW("READ 4x4w 2x4f LD64 b128", 4, 4, 2, 4, 64, 0, 0, 0, 0, 0, 0);
    ROW("READ 4x4w 2x4f LD72 b128", 4, 4, 2, 4, 72, 0, 0, 0, 0, 0, 0);

    hipDeviceProp_t pr;
    hipGetDeviceProperties(&pr, 0);
    printf("# %s  CUs=%d  clock=%d kHz  ldsPerBlockMax=%zu\n", pr.name, pr.multiProcessorCount, pr.clockRate, (size_t)pr.sharedMemPerBlock);
    float* out;
    hipMalloc(&out, 64);
    const int blocks = 960, iters = 600;
    for (auto& r : rows) {
        if (!filter.empty() && r.name.find(filter) == std::string::npos) continue;
        r.fn<<<blocks, r.nt>>>(out, 4, 0);  // warm
        hipDeviceSynchronize();
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {
            hipEventRecord(a);
            r.fn<<<blocks, r.nt>>>(out, iters, 0);
            hipEventRecord(b);
            hipEventSynchronize(b);
            float ms;
            hipEventElapsedTime(&ms, a, b);
            if (ms < best) best = ms;
        }
        if (hipGetLastError() != hipSuccess) {
            printf("%-44s LAUNCH FAILED (lds %d)\n", r.name.c_str(), r.ldsBytes);
            continue;
        }
        const double wm = (double)blocks * iters * r.wmmaPerIter, rb = (double)blocks * iters * r.readB;
        printf("%-44s lds=%6d  %8.3f ms  %s%7.1f TFLOP/s  %7.1f GB/s LDS read\n", r.name.c_str(), r.ldsBytes, best, r.wmmaOn ? "" : "(no WMMA) ", wm * 8192 / (best * 1e-3) / 1e12 * (r.wmmaOn ? 1 : 0),
               rb / (best * 1e-3) / 1e9);
    }
    return 0;
}
