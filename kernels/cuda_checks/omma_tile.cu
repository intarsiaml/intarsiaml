// Standalone nvcc check of ONE block-scaled FP4 tensor-core tile (no Rust, no cargo):
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// against a double-precision cpu reference. Random e2m1 data and a distinct random ue4m3 scale for every
// (row, k16 block) of A and every (column, k16 block) of B, so any wrong assumption about which register, nibble, lane
// or scale byte feeds which element changes the result.
//
//   nvcc -O2 --std=c++17 -gencode arch=compute_121a,code=sm_121a kernels/cuda_checks/omma_tile.cu -o omma_tile && ./omma_tile
//
// Layout under test (CUTLASS cute/atom/mma_traits_sm120.hpp; g = lane / 4, t = lane % 4; element i of a register is
// the nibble at bits [4i+3:4i]):
//   A  a0: row g,   k = 8t..8t+7      a1: row g+8, k = 8t..8t+7      a2: row g,   k = 32+8t..    a3: row g+8, k = 32+8t..
//   B  b0: col g,   k = 8t..8t+7      b1: col g,   k = 32+8t..32+8t+7
//   C  c0,c1: row g, cols 2t, 2t+1    c2,c3: row g+8, cols 2t, 2t+1
//   scale A: lane with t == 0 supplies row g, lane with t == 1 supplies row g+8 (4 ue4m3 bytes, byte j = k16 block j)
//   scale B: lane with t == 0 supplies column g
// Prints PASS and the largest relative error, or FAIL and the first mismatches.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__CUDA_ARCH__) && !(defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#error "needs -gencode arch=compute_121a,code=sm_121a (plain -arch=sm_121a also builds compute_121 PTX)"
#endif

__device__ __forceinline__ void omma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

// A: [16][32] bytes (64 nibbles per row, element k = nibble k%2 of byte k/2), B: [8][32] bytes, sfa: [16][4], sfb: [8][4]
__global__ void tile_kernel(const uint8_t* A, const uint8_t* B, const uint8_t* sfa, const uint8_t* sfb, float* D) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    auto w32 = [](const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); };
    uint32_t a[4], b[2];
    a[0] = w32(A + g * 32 + 4 * t);
    a[1] = w32(A + (g + 8) * 32 + 4 * t);
    a[2] = w32(A + g * 32 + 16 + 4 * t);
    a[3] = w32(A + (g + 8) * 32 + 16 + 4 * t);
    b[0] = w32(B + g * 32 + 4 * t);
    b[1] = w32(B + g * 32 + 16 + 4 * t);
    const uint32_t sa = t == 0 ? w32(sfa + g * 4) : (t == 1 ? w32(sfa + (g + 8) * 4) : 0u);
    const uint32_t sb = t == 0 ? w32(sfb + g * 4) : 0u;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    omma(d, a, b[0], b[1], sa, sb);
    D[g * 8 + 2 * t] = d[0];
    D[g * 8 + 2 * t + 1] = d[1];
    D[(g + 8) * 8 + 2 * t] = d[2];
    D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

static double e2m1(int c) {
    static const double v[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return (c & 8 ? -1.0 : 1.0) * v[c & 7];
}
static double ue4m3(int c) {  // bias 7, 3 mantissa bits, 0x7F = NaN
    const int e = (c >> 3) & 15, m = c & 7;
    return e == 0 ? ldexp(m / 8.0, -6) : ldexp(1.0 + m / 8.0, e - 7);
}
static int nib(const uint8_t* row, int k) { return (row[k >> 1] >> (4 * (k & 1))) & 15; }

int main() {
    uint8_t A[16 * 32], B[8 * 32], sfa[16 * 4], sfb[8 * 4];
    int bad_total = 0;
    for (int trial = 0; trial < 20; ++trial) {
        srand(1234 + trial);
        for (auto& x : A) x = rand() & 255;
        for (auto& x : B) x = rand() & 255;
        for (auto& x : sfa) x = 0x28 + rand() % 24;  // 2^-3 .. 2^2 range, no NaN
        for (auto& x : sfb) x = 0x28 + rand() % 24;
        uint8_t *dA, *dB, *dsa, *dsb;
        float* dD;
        cudaMalloc(&dA, sizeof A);
        cudaMalloc(&dB, sizeof B);
        cudaMalloc(&dsa, sizeof sfa);
        cudaMalloc(&dsb, sizeof sfb);
        cudaMalloc(&dD, 128 * 4);
        cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice);
        cudaMemcpy(dsa, sfa, sizeof sfa, cudaMemcpyHostToDevice);
        cudaMemcpy(dsb, sfb, sizeof sfb, cudaMemcpyHostToDevice);
        tile_kernel<<<1, 32>>>(dA, dB, dsa, dsb, dD);
        float D[128];
        cudaError_t err = cudaMemcpy(D, dD, sizeof D, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("FAIL cuda error %s\n", cudaGetErrorString(err));
            return 1;
        }
        double worst = 0;
        int bad = 0;
        for (int r = 0; r < 16; ++r)
            for (int n = 0; n < 8; ++n) {
                double ref = 0, mag = 0;
                for (int k = 0; k < 64; ++k) {
                    const double p = e2m1(nib(A + r * 32, k)) * ue4m3(sfa[r * 4 + k / 16]) * e2m1(nib(B + n * 32, k)) *
                                     ue4m3(sfb[n * 4 + k / 16]);
                    ref += p;
                    mag += fabs(p);
                }
                const double e = fabs(D[r * 8 + n] - ref) / (mag + 1e-30);
                if (e > worst) worst = e;
                if (e > 1e-6 && bad++ < 4 && trial == 0) printf("  mismatch r=%d n=%d got %g want %g\n", r, n, D[r * 8 + n], ref);
            }
        printf("trial %2d: worst error / sum|terms| = %.3g  %s\n", trial, worst, bad ? "MISMATCH" : "ok");
        bad_total += bad;
        cudaFree(dA); cudaFree(dB); cudaFree(dsa); cudaFree(dsb); cudaFree(dD);
    }
    printf(bad_total ? "FAIL\n" : "PASS\n");
    return bad_total != 0;
}
