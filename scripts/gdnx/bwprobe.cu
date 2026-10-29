// Lane gdnx: read+write bandwidth probe on spark1 (GB10) at the GDN decode footprint: 30 "layers" x 32 MB f32, every float4
// read, scaled and written back in place. Variants: float4s per thread (PER, one warp per block, so a warp covers PER * 512
// contiguous bytes), load flavour (HINT 0 plain, 1 __ldcs/__stcs, 2 __ldg, 3 ld.L2::256B prefetch hint, 4 L2 evict_first policy).
// nvcc -O3 -arch=sm_121 bwprobe.cu -o bwprobe && ./bwprobe
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ float4 ld256(const float4* p) {
    float4 v;
    asm volatile("ld.global.L2::256B.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float4 ldef(const float4* p, uint64_t pol) {
    float4 v;
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p), "l"(pol));
    return v;
}
__device__ __forceinline__ void stef(float4* p, float4 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "l"(pol));
}

template <int HINT, int PER>
__global__ void __launch_bounds__(32) rw(float4* __restrict__ s, float f) {
    const long base = (long)blockIdx.x * 32 * PER + threadIdx.x;
    uint64_t pol = 0;
    if (HINT == 4) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    float4 v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const float4* p = s + base + 32l * i;
        v[i] = HINT == 1 ? __ldcs(p) : HINT == 2 ? __ldg(p) : HINT == 3 ? ld256(p) : HINT == 4 ? ldef(p, pol) : *p;
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        float4 x = v[i];
        x.x *= f; x.y *= f; x.z *= f; x.w *= f;
        float4* p = s + base + 32l * i;
        if (HINT == 1) __stcs(p, x);
        else if (HINT == 4) stef(p, x, pol);
        else *p = x;
    }
}

template <int HINT, int PER>
float run(float4** bufs, int layers, long n) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(a);
        for (int l = 0; l < layers; ++l) rw<HINT, PER><<<(unsigned)(n / 32 / PER), 32>>>(bufs[l], 1.0f);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (rep && ms < best) best = ms;
    }
    return best * 1000.f / layers;
}

#define ROW(P) { float u0 = run<0, P>(bufs, layers, n), u1 = run<1, P>(bufs, layers, n), u2 = run<2, P>(bufs, layers, n), \
    u3 = run<3, P>(bufs, layers, n), u4 = run<4, P>(bufs, layers, n); \
    printf("BW per=%2d (%5d B/warp)  plain %.1f  cs %.1f  ldg %.1f  L2::256B %.1f  evict_first %.1f us  (plain %.0f GB/s)\n", P, P * 512, u0, u1, u2, u3, u4, 2 * bytes / u0 / 1e3); }

int main() {
    const int layers = 30;
    const long bytes = 32l << 20, n = bytes / 16;
    float4* bufs[layers];
    for (int l = 0; l < layers; ++l) { cudaMalloc(&bufs[l], bytes); cudaMemset(bufs[l], 0, bytes); }
    ROW(1) ROW(2) ROW(4) ROW(8) ROW(16) ROW(32)
    return 0;
}
