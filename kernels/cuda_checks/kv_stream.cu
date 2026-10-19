// Standalone DRAM read-bandwidth probe for split-KV attention decode (docs/results/2026-10-19-cuda-round5.md).
// Ten "layers" of a bf16 KV cache at the 35B shape (2 kv heads x CAP positions x 256 dims, K and V: 33.5 MB per layer at
// CAP 16384, 335 MB in all, far above L2) are read once per launch, layer after layer, as decode does. Modes:
//   stream     grid-stride 16 B loads, 4 in flight per thread, 4 blocks of 256 per SM: the plain read rate of the box
//   pattern    the split-KV kernel's own grid and loads (64 splits x 2 kv heads, 8 warps; warp g reads keys g, g+8, ...;
//              KB keys of K and V per thread per iteration) with no arithmetic: the ceiling of that access pattern
//   pattern+N  the same, N iterations of loads in flight per thread (prefetch depth)
// Prints us per layer and GB/s for each mode.
//   nvcc -O3 -gencode arch=compute_121,code=sm_121 kernels/cuda_checks/kv_stream.cu -o kv_stream && ./kv_stream [cap]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int D = 256, NKV = 2, LAYERS = 10;

__global__ void stream_rd(const uint4* __restrict__ k, const uint4* __restrict__ v, size_t n16, uint32_t* sink) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n16; i += 4 * stride) {
        uint4 a[4], b[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { a[u] = __ldg(k + i + u * stride); b[u] = __ldg(v + i + u * stride); }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= a[u].x ^ a[u].w ^ b[u].y ^ b[u].z;
    }
    for (; i < n16; i += stride) { uint4 a = __ldg(k + i), b = __ldg(v + i); acc ^= a.x ^ b.y; }
    if (acc == 0x9e3779b9u) atomicAdd(sink, 1u);
}

// grid (nsplit, NKV), 256 threads: warp gk of split sp reads keys start + it*8*KB + u*8 + gk, 16 B per thread of K and V,
// DEPTH iterations of loads issued before the first is consumed (DEPTH 1 = the kernel's own loop)
template <int KB, int DEPTH>
__global__ void __launch_bounds__(256) pattern_rd(const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc, int64_t cap,
                                                  int64_t kv_len, int64_t chunk, uint32_t* sink) {
    const int sp = blockIdx.x, kvh = blockIdx.y, c = threadIdx.x % 32, gk = threadIdx.x / 32;
    const int64_t start = (int64_t)sp * chunk;
    int64_t end = start + chunk;
    if (end > kv_len) end = kv_len;
    if (start >= end) return;
    const uint16_t* kb = kc + (int64_t)kvh * cap * D + c * 8;
    const uint16_t* vb = vc + (int64_t)kvh * cap * D + c * 8;
    uint32_t acc = 0;
    constexpr int STEP = 8 * KB;
    uint4 kk[DEPTH][KB], vv[DEPTH][KB];
    auto load = [&](int slot, int64_t base) {
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            const int64_t j = base + u * 8 + gk;
            kk[slot][u] = make_uint4(0, 0, 0, 0); vv[slot][u] = make_uint4(0, 0, 0, 0);
            if (j < end) { kk[slot][u] = __ldg(reinterpret_cast<const uint4*>(kb + j * D)); vv[slot][u] = __ldg(reinterpret_cast<const uint4*>(vb + j * D)); }
        }
    };
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) load(d, start + (int64_t)d * STEP);
    int slot = 0;
    for (int64_t base = start; base < end; base += STEP) {
#pragma unroll
        for (int u = 0; u < KB; ++u) acc ^= kk[0][u].x ^ kk[0][u].w ^ vv[0][u].y ^ vv[0][u].z;
#pragma unroll
        for (int d = 0; d + 1 < DEPTH; ++d)
#pragma unroll
            for (int u = 0; u < KB; ++u) { kk[d][u] = kk[d + 1][u]; vv[d][u] = vv[d + 1][u]; }
        load(DEPTH - 1, base + (int64_t)DEPTH * STEP);
        (void)slot;
    }
    if (acc == 0x9e3779b9u) atomicAdd(sink, 1u);
}

int main(int argc, char** argv) {
    const int64_t cap = argc > 1 ? atoll(argv[1]) : 16384;
    const size_t per = (size_t)NKV * cap * D * 2;  // bytes of K (or V) per layer
    uint16_t *k[LAYERS], *v[LAYERS];
    for (int l = 0; l < LAYERS; ++l) {
        cudaMalloc(&k[l], per); cudaMalloc(&v[l], per);
        cudaMemset(k[l], 0x11, per); cudaMemset(v[l], 0x22, per);
    }
    uint32_t* sink; cudaMalloc(&sink, 4);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int64_t kv_len = cap;
    // the split policy of kernels/cuda/attention_splitkv.cu at T = 1
    int64_t nsplit = (kv_len + 63) / 64, fill = (256 + NKV - 1) / NKV;
    if (nsplit > fill) nsplit = fill;
    if (nsplit > 64) nsplit = 64;
    int64_t chunk = (kv_len + nsplit - 1) / nsplit; chunk = (chunk + 63) / 64 * 64;
    printf("cap %lld: %d SMs, %.1f MB per layer (K+V), nsplit %lld, chunk %lld\n", (long long)cap, sms, 2.0 * per / 1e6,
           (long long)nsplit, (long long)chunk);
    auto timeit = [&](const char* name, auto launch) {
        for (int l = 0; l < LAYERS; ++l) launch(l);  // warm
        const int reps = 20;
        cudaEventRecord(a);
        for (int r = 0; r < reps; ++r)
            for (int l = 0; l < LAYERS; ++l) launch(l);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        const double us = ms * 1e3 / (reps * LAYERS);
        printf("%-12s %8.1f us per layer  %6.1f GB/s\n", name, us, 2.0 * per / us / 1e3);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(e)); exit(1); }
    };
    const size_t n16 = per / 16;
    for (int bps : {2, 4, 8})
        timeit(bps == 2 ? "stream x2" : bps == 4 ? "stream x4" : "stream x8",
               [&](int l) { stream_rd<<<sms * bps, 256>>>((const uint4*)k[l], (const uint4*)v[l], n16, sink); });
    dim3 g((unsigned)nsplit, NKV);
    timeit("pattern", [&](int l) { pattern_rd<2, 1><<<g, 256>>>(k[l], v[l], cap, kv_len, chunk, sink); });
    timeit("pattern+2", [&](int l) { pattern_rd<2, 2><<<g, 256>>>(k[l], v[l], cap, kv_len, chunk, sink); });
    timeit("pattern+4", [&](int l) { pattern_rd<2, 4><<<g, 256>>>(k[l], v[l], cap, kv_len, chunk, sink); });
    timeit("pattern+8", [&](int l) { pattern_rd<2, 8><<<g, 256>>>(k[l], v[l], cap, kv_len, chunk, sink); });
    return 0;
}
