// standalone microbench of the down combine kernel: the repo's combine_kernel (baseline) against variants (bitwise check)
#include "cuda/exl3_moe.cu"
namespace ik_exl3x {
int launch_cp(const GroupedArgs&, cudaStream_t) { return IK_ERR_SHAPE; }
int launch_rows(const GroupedArgs&, bool, cudaStream_t) { return IK_ERR_SHAPE; }
int launch_dense_cp(const half*, const uint32_t*, float*, const int*, int, int, int, int, int, cudaStream_t) { return IK_ERR_SHAPE; }
}  // namespace ik_exl3x
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <functional>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)
using namespace ik_exl3x;

// variant A: a warp owns one (row, 128 columns) and runs all K slots in registers: no shared memory, no barrier
template <int K, int WPB, typename TOUT>
__global__ void __launch_bounds__(WPB * 32) combine_w(const float* __restrict__ Zd, const int* __restrict__ ids, const float* __restrict__ wts, Experts ex, TOUT* __restrict__ out) {
    const int H = ex.H, r = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int cb = blockIdx.y * WPB + warp;
    if (cb * 128 >= H) return;
    const int n = cb * 128 + 4 * lane;
    int e[K];
    float w[K];
    float4 z[K];
#pragma unroll
    for (int q = 0; q < K; ++q) {
        const int p = r * K + q;
        e[q] = ids[p];
        w[q] = wts[p];
        z[q] = (e[q] >= 0 && e[q] < ex.E) ? __ldg(reinterpret_cast<const float4*>(Zd + (size_t)p * H + n)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < K; ++q) {
        float o[4] = {0.f, 0.f, 0.f, 0.f};
        if (e[q] >= 0 && e[q] < ex.E) {
            float v[4] = {z[q].x, z[q].y, z[q].z, z[q].w};
            fwht128(v, lane);
            const half* sv = ex.svh_d(e[q]) + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(sv[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = fmaf(w[q], o[j], acc[j]);
    }
    TOUT* orow = out + (size_t)r * H + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) orow[j] = from_f<TOUT>(acc[j]);
}

__global__ void fill_f(float* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u + seed; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
        p[i] = ((h & 0xffffff) / 16777215.f - 0.5f) * 8.f;
    }
}
__global__ void fill_h(half* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u + seed; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
        p[i] = __float2half(((h & 0xffff) / 65535.f) + 0.5f);
    }
}

int main(int argc, char** argv) {
    const int R = argc > 1 ? atoi(argv[1]) : 2048, K = 7, H = 5120, I = 1152, E = 385;
    const int reps = argc > 2 ? atoi(argv[2]) : 15;
    const int P = R * K, C = 3 * H + 3 * I;
    std::vector<int> ids((size_t)P);
    uint64_t s = 88172645463325252ull;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int r = 0; r < R; ++r) { for (int q = 0; q < 6; ++q) ids[(size_t)r * K + q] = (int)(rnd() % 384); ids[(size_t)r * K + 6] = 384; }
    ids[5] = -1; ids[100] = 999;
    std::vector<float> wts((size_t)P);
    for (auto& v : wts) v = (rnd() % 1000) / 700.f;
    float *d_z, *d_w, *d_o0, *d_o1; int* d_ids; half* d_sc;
    CK(cudaMalloc(&d_z, (size_t)P * H * 4)); CK(cudaMalloc(&d_w, wts.size() * 4)); CK(cudaMalloc(&d_ids, ids.size() * 4));
    CK(cudaMalloc(&d_o0, (size_t)R * H * 4)); CK(cudaMalloc(&d_o1, (size_t)R * H * 4)); CK(cudaMalloc(&d_sc, (size_t)E * C * 2));
    fill_f<<<2048, 256>>>(d_z, (size_t)P * H, 7u); fill_h<<<1024, 256>>>(d_sc, (size_t)E * C, 9u);
    CK(cudaMemcpy(d_w, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    Experts ex{}; ex.sc = d_sc; ex.C = C; ex.E = E; ex.H = H; ex.I = I;
    struct V { const char* name; std::function<void()> run; };
    std::vector<V> vs = {
        {"base combine_kernel<float>", [&]() { combine_kernel<float><<<dim3(R, H / 128), 32 * K>>>(d_z, d_ids, d_w, ex, d_o1, K); }},
        {"warp-per-item WPB=8", [&]() { combine_w<7, 8, float><<<dim3(R, (H / 128 + 7) / 8), 256>>>(d_z, d_ids, d_w, ex, d_o1); }},
        {"warp-per-item WPB=4", [&]() { combine_w<7, 4, float><<<dim3(R, (H / 128 + 3) / 4), 128>>>(d_z, d_ids, d_w, ex, d_o1); }},
        {"warp-per-item WPB=2", [&]() { combine_w<7, 2, float><<<dim3(R, (H / 128 + 1) / 2), 64>>>(d_z, d_ids, d_w, ex, d_o1); }},
        {"warp-per-item WPB=1", [&]() { combine_w<7, 1, float><<<dim3(R, H / 128), 32>>>(d_z, d_ids, d_w, ex, d_o1); }},
    };
    std::vector<float> ref((size_t)R * H), got((size_t)R * H);
    CK(cudaMemset(d_o0, 0xff, (size_t)R * H * 4));
    combine_kernel<float><<<dim3(R, H / 128), 32 * K>>>(d_z, d_ids, d_w, ex, d_o0, K); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), d_o0, ref.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t vi = 1; vi < vs.size(); ++vi) {
        CK(cudaMemset(d_o1, 0xff, (size_t)R * H * 4)); vs[vi].run(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), d_o1, got.size() * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0; for (size_t i = 0; i < got.size(); ++i) if (memcmp(&got[i], &ref[i], 4) != 0) ++bad;
        printf("%-34s bitwise vs base: %s (%zu differ)\n", vs[vi].name, bad ? "NO" : "yes", bad);
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<std::vector<float>> t(vs.size());
    // after = 0: combine alone (clean caches); after = 1: right after a kernel that wrote all of Zd (as the down GEMM does)
    std::vector<std::vector<float>> t2(vs.size());
    for (int rep = 0; rep < reps; ++rep) for (size_t vi = 0; vi < vs.size(); ++vi) {
        cudaEventRecord(e0); vs[vi].run(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); t[vi].push_back(ms);
        fill_f<<<2048, 256>>>(d_z, (size_t)P * H, 7u + rep);  // dirty
        cudaEventRecord(e0); vs[vi].run(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); t2[vi].push_back(ms);
    }
    for (size_t vi = 0; vi < vs.size(); ++vi) { auto a = t2[vi]; std::sort(a.begin(), a.end()); printf("AFTER A 293 MB WRITE: %-34s min %.3f median %.3f ms\n", vs[vi].name, a[0], a[a.size() / 2]); }
    double bytes = (double)P * H * 4 + (double)R * H * 4;
    for (size_t vi = 0; vi < vs.size(); ++vi) { auto a = t[vi]; std::sort(a.begin(), a.end()); printf("%-34s min %.3f median %.3f ms  (%.0f GB/s at median; DRAM floor at 254 GB/s %.2f ms)\n", vs[vi].name, a[0], a[a.size() / 2], bytes / (a[a.size() / 2] * 1e-3) / 1e9, bytes / 254e9 * 1e3); }
    return 0;
}
