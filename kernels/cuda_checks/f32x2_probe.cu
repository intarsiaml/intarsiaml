// Probe: can sm_121 (GB10) issue packed fp32 adds (PTX add.rn.f32x2, FADD2 in SASS), and what does the
// "k16 product sum from zero + separate fp32 add" accumulation of the MoE kernels cost against plain mma
// accumulation (C = acc)? Lane S1 (docs/results/2026-10-22-s1-spark-prefill.md).
//
//   nvcc -O3 -arch=sm_121 -o f32x2_probe kernels/cuda_checks/f32x2_probe.cu && ./f32x2_probe
//   cuobjdump -sass f32x2_probe | grep -cE 'FADD2|FADD.*F32x2'
//
// Modes (each warp: 4 x 4 independent m16n8k16 bf16 tiles per k16 step, like moe_w4a16 MT = 4, JW = 4):
//   0  mma with C = acc (the tensor-core accumulate; the cuBLAS / mma_peak reference, NOT the MoE numerics)
//   1  t = mma(0); acc += t with four scalar fp32 adds (the MoE kernels today)
//   2  t = mma(0); acc += t with two add.rn.f32x2 (bitwise the same as mode 1: IEEE fp32 RN per lane)
// It also checks mode 1 == mode 2 bit for bit.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b, const float* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void add2(float& x0, float& x1, float y0, float y1) {
    asm("{\n\t.reg .b64 pa, pb, pd;\n\t"
        "mov.b64 pa, {%0, %1};\n\t"
        "mov.b64 pb, {%2, %3};\n\t"
        "add.rn.f32x2 pd, pa, pb;\n\t"
        "mov.b64 {%0, %1}, pd;\n\t}"
        : "+f"(x0), "+f"(x1)
        : "f"(y0), "f"(y1));
}

// fp32 elementwise throughput: 8 independent chains of s = fma(s, a, b) per thread, scalar (FX2 = 0) or packed (1).
template <int FX2>
__global__ void __launch_bounds__(256) fma_probe(const float* __restrict__ src, float* __restrict__ out, int iters) {
    float s[16];
    const float a = src[threadIdx.x & 31] * 1e-3f + 0.999f, b = src[(threadIdx.x + 7) & 31] * 1e-3f;
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = src[(threadIdx.x + i) & 31];
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 16; i += 2) {
            if (FX2) {
                asm("{\n\t.reg .b64 ps, pa, pb;\n\t"
                    "mov.b64 ps, {%0, %1};\n\t"
                    "mov.b64 pa, {%2, %2};\n\t"
                    "mov.b64 pb, {%3, %3};\n\t"
                    "fma.rn.f32x2 ps, ps, pa, pb;\n\t"
                    "mov.b64 {%0, %1}, ps;\n\t}"
                    : "+f"(s[i]), "+f"(s[i + 1])
                    : "f"(a), "f"(b));
            } else {
                s[i] = __fmaf_rn(s[i], a, b);
                s[i + 1] = __fmaf_rn(s[i + 1], a, b);
            }
        }
    }
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) r += s[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}

template <int MODE>
__global__ void __launch_bounds__(128, 2) probe(const uint32_t* __restrict__ src, float* __restrict__ out, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t a[4][4], b[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int v = 0; v < 4; ++v) a[i][v] = src[(i * 4 + v) * 32 + lane];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int v = 0; v < 2; ++v) b[j][v] = src[512 + (j * 2 + v) * 32 + lane];
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (MODE == 0) {
                    mma16816(acc[i][j], a[i], b[j], acc[i][j]);
                } else {
                    const float z[4] = {0.f, 0.f, 0.f, 0.f};
                    float t[4];
                    mma16816(t, a[i], b[j], z);
                    if (MODE == 1) {
#pragma unroll
                        for (int v = 0; v < 4; ++v) acc[i][j][v] += t[v];
                    } else {
                        add2(acc[i][j][0], acc[i][j][1], t[0], t[1]);
                        add2(acc[i][j][2], acc[i][j][3], t[2], t[3]);
                    }
                }
            }
        // perturb the operands a little so the compiler cannot hoist the mma out of the loop
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j][0] ^= (uint32_t)(it & 1) << 3;
    }
    float* o = out + ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 64;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) o[(i * 4 + j) * 4 + v] = acc[i][j][v];
}

template <int MODE>
double run(const uint32_t* src, float* out, int blocks, int iters) {
    probe<MODE><<<blocks, 128>>>(src, out, 8);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    probe<MODE><<<blocks, 128>>>(src, out, iters);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double flop = (double)blocks * 4 /*warps*/ * iters * 16 /*mma*/ * 16 * 8 * 16 * 2;
    return flop / (ms * 1e-3) / 1e12;
}

int main() {
    int dev = 0, sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int blocks = sms * 2, iters = 4096;
    std::vector<uint32_t> h(1024);
    uint32_t s = 12345;
    for (auto& x : h) {  // random bf16 pairs in [-2, 2)
        s = s * 1664525u + 1013904223u;
        const float f0 = ((s >> 8) & 0xFFFF) / 16384.f - 2.f, f1 = ((s >> 16) & 0xFFFF) / 16384.f - 2.f;
        __nv_bfloat162 p = __floats2bfloat162_rn(f0, f1);
        memcpy(&x, &p, 4);
    }
    uint32_t* src;
    float *o1, *o2;
    cudaMalloc(&src, 4096);
    const size_t nout = (size_t)blocks * 128 * 64;
    cudaMalloc(&o1, nout * 4);
    cudaMalloc(&o2, nout * 4);
    cudaMemcpy(src, h.data(), 4096, cudaMemcpyHostToDevice);
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = run<0>(src, o1, blocks, iters);
        const double t1 = run<1>(src, o1, blocks, iters);
        const double t2 = run<2>(src, o2, blocks, iters);
        printf("rep %d: mma C=acc %.1f TFLOPS | zero+4 FADD %.1f | zero+2 FADD2 %.1f\n", rep, t0, t1, t2);
    }
    {
        float* fs;
        float *f1, *f2;
        cudaMalloc(&fs, 128);
        cudaMalloc(&f1, (size_t)sms * 8 * 256 * 4);
        cudaMalloc(&f2, (size_t)sms * 8 * 256 * 4);
        std::vector<float> hf(32);
        for (int i = 0; i < 32; ++i) hf[i] = 0.5f + i * 0.01f;
        cudaMemcpy(fs, hf.data(), 128, cudaMemcpyHostToDevice);
        for (int rep = 0; rep < 3; ++rep) {
            float ms[2];
            for (int m = 0; m < 2; ++m) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                if (m == 0) fma_probe<0><<<sms * 8, 256>>>(fs, f1, 16);
                else fma_probe<1><<<sms * 8, 256>>>(fs, f2, 16);
                cudaEventRecord(e0);
                if (m == 0) fma_probe<0><<<sms * 8, 256>>>(fs, f1, 8192);
                else fma_probe<1><<<sms * 8, 256>>>(fs, f2, 8192);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms[m], e0, e1);
            }
            const double fl = (double)sms * 8 * 256 * 8192 * 16 * 2;
            printf("rep %d: scalar FFMA %.2f TFLOPS | fma.rn.f32x2 %.2f TFLOPS\n", rep, fl / (ms[0] * 1e9), fl / (ms[1] * 1e9));
        }
        std::vector<float> x1((size_t)sms * 8 * 256), x2(x1.size());
        cudaMemcpy(x1.data(), f1, x1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(x2.data(), f2, x2.size() * 4, cudaMemcpyDeviceToHost);
        size_t d = 0;
        for (size_t i = 0; i < x1.size(); ++i) d += memcmp(&x1[i], &x2[i], 4) != 0;
        printf("fma scalar vs f32x2 bitwise: %zu of %zu differ\n", d, x1.size());
    }
    std::vector<float> a(nout), b(nout);
    cudaMemcpy(a.data(), o1, nout * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), o2, nout * 4, cudaMemcpyDeviceToHost);
    size_t diff = 0;
    for (size_t i = 0; i < nout; ++i) diff += memcmp(&a[i], &b[i], 4) != 0;
    printf("mode1 vs mode2 bitwise: %zu of %zu differ; err=%s\n", diff, nout, cudaGetErrorString(cudaGetLastError()));
    return diff != 0;
}
