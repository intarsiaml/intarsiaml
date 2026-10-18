// Standalone DRAM read-bandwidth probe for the MoE weight stream (docs/results/2026-10-18-cuda-moe-w4a16.md): a buffer of
// ROWS rows of ROWB bytes (the gate_up weights of one layer: 256 experts x 1024 rows x 1024 B) is read once by blocks that
// each own 128 consecutive rows and walk along the rows in units of RUN bytes (RUN contiguous bytes per row per unit, as the
// MoE kernels' cp.async units do). Prints GB/s per run length.
//   nvcc -O2 -gencode arch=compute_121,code=sm_121 kernels/cuda_checks/stream_runs.cu -o stream_runs && ./stream_runs
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void rd(const uint4* __restrict__ p, size_t rowb16, int run16, int units, uint64_t* sink) {
    // block = 128 rows; thread t of 128 reads, per unit, the chunk (t % run16) of row (t / run16) + k * (128 / run16)
    const int tid = threadIdx.x;
    const size_t row0 = (size_t)blockIdx.x * 128;
    uint4 acc = make_uint4(0, 0, 0, 0);
    const int rows_per_pass = 128 / run16 > 0 ? 128 / run16 : 1;
    for (int u = 0; u < units; ++u) {
#pragma unroll 4
        for (int k = 0; k < (128 * run16) / 128; ++k) {  // each pass covers 128 chunks
            const int c = tid + 128 * k;
            const int r = c / run16, off = c % run16;
            const uint4 v = __ldcg(p + (row0 + r) * rowb16 + (size_t)u * run16 + off);
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    (void)rows_per_pass;
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) atomicAdd((unsigned long long*)sink, 1ull);
}

int main() {
    const size_t rows = 256ull * 1024, rowb = 1024;
    const size_t bytes = rows * rowb;
    uint4* p;
    uint64_t* sink;
    cudaMalloc(&p, bytes);
    cudaMalloc(&sink, 8);
    cudaMemset(p, 1, bytes);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int run : {32, 64, 128, 256, 512, 1024}) {
        const int run16 = run / 16, units = (int)(rowb / run);
        const int blocks = (int)(rows / 128);
        for (int rep = 0; rep < 2; ++rep) {
            cudaEventRecord(a);
            rd<<<blocks, 128>>>(p, rowb / 16, run16, units, sink);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            if (rep) printf("run %4d B per row per unit: %7.3f ms  %6.1f GB/s\n", run, ms, bytes / ms / 1e6);
        }
    }
    return 0;
}
