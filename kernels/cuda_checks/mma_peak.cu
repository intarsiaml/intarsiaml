// Standalone tensor-core throughput probe for the dense prefill projections (docs/results/2026-10-19-cuda-prefill-3.md):
// register-only loops of mma.sync m16n8k16 bf16 (f32 accumulate, C = D chained as wq_gemm.cuh does) and m16n8k32 e4m3
// (f32 accumulate), 8 independent accumulator chains per warp, no memory traffic. Prints TFLOPS per warps-per-block
// configuration plus the device's shared-memory limits (per block opt-in, per SM) and SM count / clock.
//   nvcc -O2 -gencode arch=compute_121,code=sm_121 kernels/cuda_checks/mma_peak.cu -o mma_peak && ./mma_peak
#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int CH = 8;      // independent accumulator chains per warp
constexpr int ITERS = 4096;

__global__ void bf16_loop(float* sink, uint32_t seed) {
    float acc[CH][4] = {};
    uint32_t a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u, b1 = a0 * 13u;
    a0 &= 0x3F003F00u; a1 &= 0x3F003F00u; a2 &= 0x3F003F00u; a3 &= 0x3F003F00u; b0 &= 0x3F003F00u; b1 &= 0x3F003F00u;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int c = 0; c < CH; ++c)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
    if (s == 1.2345f) sink[0] = s;
}

__global__ void e4m3_loop(float* sink, uint32_t seed) {
    float acc[CH][4] = {};
    uint32_t a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u, b1 = a0 * 13u;
    a0 &= 0x30303030u; a1 &= 0x30303030u; a2 &= 0x30303030u; a3 &= 0x30303030u; b0 &= 0x30303030u; b1 &= 0x30303030u;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int c = 0; c < CH; ++c)
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
    if (s == 1.2345f) sink[0] = s;
}

int main() {
    int dev = 0, sms = 0, smem_blk = 0, smem_sm = 0, clk = 0, regs_sm = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&smem_blk, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    printf("SMs %d, max smem per block (opt-in) %d B, per SM %d B, regs per SM %d, clock %d kHz\n", sms, smem_blk, smem_sm,
           regs_sm, clk);
    float* sink;
    cudaMalloc(&sink, 4);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int kind = 0; kind < 2; ++kind) {
        for (int warps : {4, 8, 16}) {
            const int blocks = sms * 8;
            auto launch = [&] {
                if (kind == 0) bf16_loop<<<blocks, 32 * warps>>>(sink, 7u);
                else e4m3_loop<<<blocks, 32 * warps>>>(sink, 7u);
            };
            launch();
            cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int r = 0; r < 5; ++r) launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            const double k = kind == 0 ? 16.0 : 32.0;
            const double flop = 5.0 * blocks * warps * (double)ITERS * CH * 2.0 * 16 * 8 * k;
            printf("%s m16n8k%d, %2d warps/block, %d blocks: %.1f TFLOPS\n", kind == 0 ? "bf16" : "e4m3", (int)k, warps, blocks,
                   flop / (ms * 1e-3) / 1e12);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("error: %s\n", cudaGetErrorString(err));
    return 0;
}
