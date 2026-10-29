// Lane attn16: read-only bandwidth probe on spark2 (GB10) at the B = 16 decode attention footprint: 10 "layers" x 38 MB of
// bf16, every uint4 read once and folded into a per-thread xor (written once per thread), event-timed. Variants: uint4 loads
// in flight per thread (PER) and threads per block. Probe for the read floor the attention kernel can reach.
// nvcc -O3 -arch=sm_121 rdprobe.cu -o rdprobe && ./rdprobe
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

template <int PER>
__global__ void rd(const uint4* __restrict__ s, long n, uint32_t* out) {
    const long stride = (long)gridDim.x * blockDim.x * PER;
    uint32_t x = 0;
    for (long i = ((long)blockIdx.x * blockDim.x * PER) + threadIdx.x; i < n; i += stride) {
        uint4 v[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p) v[p] = (i + (long)p * blockDim.x < n) ? __ldg(s + i + (long)p * blockDim.x) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int p = 0; p < PER; ++p) x ^= v[p].x ^ v[p].y ^ v[p].z ^ v[p].w;
    }
    out[(long)blockIdx.x * blockDim.x + threadIdx.x] = x;
}

template <int PER>
void run(const uint4* buf, long n16, int layers, uint32_t* out, int threads, int bps) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int grid = sms * bps;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e9;
    for (int rep = 0; rep < 6; ++rep) {
        cudaEventRecord(a);
        for (int l = 0; l < layers; ++l) rd<PER><<<grid, threads>>>(buf + (long)l * n16, n16, out);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (rep > 0 && ms < best) best = ms;
    }
    const double us = best * 1e3 / layers;
    printf("RDPROBE PER=%d threads=%d blocks/SM=%d: %.1f us per %.1f MB, %.0f GB/s\n", PER, threads, bps, us, n16 * 16 / 1e6, n16 * 16 / us / 1e3);
}

// the attention's pattern: `regions` (row, kv head) pairs, each a K and a V area of `keys` rows of 512 B in its own allocation,
// cut into items of 64 keys dealt round-robin to the blocks; warp w of a block reads keys base + 8 u + w (u < 2) per iteration,
// a 16 B piece per lane of the K row and of the V row
__global__ void pat(const uint4* const* __restrict__ kp, const uint4* const* __restrict__ vp, int regions, int keys, uint32_t* out) {
    const int per = (keys + 63) / 64, items = regions * per;
    const int c = threadIdx.x % 32, gk = threadIdx.x / 32;
    uint32_t x = 0;
    for (int it = blockIdx.x; it < items; it += gridDim.x) {
        const int r = it / per, sp = it % per;
        const uint4* kb = kp[r] + c;
        const uint4* vb = vp[r] + c;
        const int start = sp * 64, end = min(start + 64, keys);
        for (int base = start; base < end; base += 16) {
            uint4 a[2], b[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int j = base + u * 8 + gk;
                a[u] = j < end ? __ldg(kb + (long)j * 32) : make_uint4(0, 0, 0, 0);
                b[u] = j < end ? __ldg(vb + (long)j * 32) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) x ^= a[u].x ^ a[u].w ^ b[u].y ^ b[u].z;
        }
    }
    out[(long)blockIdx.x * blockDim.x + threadIdx.x] = x;
}

int main() {
    {
        const int layers = 10, regions = 32, keys = 1190, cap = 1400;
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        const uint4** hk = new const uint4*[layers * regions];
        const uint4** hv = new const uint4*[layers * regions];
        for (int i = 0; i < layers * regions; ++i) {
            void *a, *b;
            cudaMalloc(&a, (size_t)cap * 512);
            cudaMalloc(&b, (size_t)cap * 512);
            cudaMemset(a, 1, (size_t)cap * 512);
            cudaMemset(b, 1, (size_t)cap * 512);
            hk[i] = (const uint4*)a;
            hv[i] = (const uint4*)b;
        }
        const uint4 **dk, **dv;
        cudaMalloc(&dk, sizeof(void*) * layers * regions);
        cudaMalloc(&dv, sizeof(void*) * layers * regions);
        cudaMemcpy(dk, hk, sizeof(void*) * layers * regions, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, hv, sizeof(void*) * layers * regions, cudaMemcpyHostToDevice);
        uint32_t* out;
        cudaMalloc(&out, 64 << 20);
        for (int bps : {1, 2, 4}) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            float best = 1e9;
            for (int rep = 0; rep < 6; ++rep) {
                cudaEventRecord(a);
                for (int l = 0; l < layers; ++l) pat<<<sms * bps, 256>>>(dk + l * regions, dv + l * regions, regions, keys, out);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                if (rep > 0 && ms < best) best = ms;
            }
            const double us = best * 1e3 / layers, mb = 2.0 * regions * keys * 512 / 1e6;
            printf("PATPROBE blocks/SM=%d: %.1f us per %.1f MB, %.0f GB/s\n", bps, us, mb, mb / us * 1e3);
        }
    }
    const long bytes = 38L << 20;
    const int layers = 10;
    const long n16 = bytes / 16;
    uint4* buf;
    uint32_t* out;
    cudaMalloc(&buf, bytes * layers);
    cudaMemset(buf, 1, bytes * layers);
    cudaMalloc(&out, 64 << 20);
    for (int bps : {1, 2, 4, 8})
        for (int threads : {256}) {
            run<1>(buf, n16, layers, out, threads, bps);
            run<2>(buf, n16, layers, out, threads, bps);
            run<4>(buf, n16, layers, out, threads, bps);
            run<8>(buf, n16, layers, out, threads, bps);
        }
    return 0;
}
