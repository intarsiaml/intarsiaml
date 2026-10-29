// lane moe16c: does fma.rn.f32.{f16,bf16} (FHFMA) give the bits of fmaf on the f32 images? exhaustive over the inputs the rows
// kernels feed it: f16 = every e4m3 x every e4m3; bf16 = every bf16 x every e2m1 (signed); c from a set incl. subnormals / inf
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>
__device__ float mix_h(unsigned short a, unsigned short b, float c) { float d; asm("fma.rn.f32.f16 %0, %1, %2, %3;" : "=f"(d) : "h"(a), "h"(b), "f"(c)); return d; }
__device__ float mix_b(unsigned short a, unsigned short b, float c) { float d; asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(d) : "h"(a), "h"(b), "f"(c)); return d; }
__constant__ uint32_t kC[16] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807fffffu, 0x00800000u, 0x3f800000u, 0xbf800001u, 0x7f7fffffu,
                                0xff7fffffu, 0x7f800000u, 0x3a800000u, 0xb9000000u, 0x4b000001u, 0x33800000u, 0x0c000000u, 0x47c35000u};
__global__ void chk(unsigned long long* bad, unsigned long long* tot) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nb = 0, nt = 0;
    if (idx < 65536) {  // f16: a = e4m3 code idx & 255, b = e4m3 code idx >> 8
        const uint8_t ca = idx & 255, cb = idx >> 8;
        if ((ca & 0x7F) != 0x7F && (cb & 0x7F) != 0x7F) {
            __half_raw ha = __nv_cvt_fp8_to_halfraw(ca, __NV_E4M3), hb = __nv_cvt_fp8_to_halfraw(cb, __NV_E4M3);
            const float fa = __half2float(*reinterpret_cast<__half*>(&ha)), fb = __half2float(*reinterpret_cast<__half*>(&hb));
            for (int k = 0; k < 16; ++k) {
                const float c = __uint_as_float(kC[k]);
                const float r0 = fmaf(fa, fb, c), r1 = mix_h(ha.x, hb.x, c);
                nb += __float_as_uint(r0) != __float_as_uint(r1) && !(r0 != r0 && r1 != r1); ++nt;
                // chained: as in the 16-long dot
                const float c2 = fmaf(fa, fb, fmaf(fb, fa, c)), m2 = mix_h(ha.x, hb.x, mix_h(hb.x, ha.x, c));
                nb += __float_as_uint(c2) != __float_as_uint(m2) && !(c2 != c2 && m2 != m2); ++nt;
            }
        }
    } else if (idx < 65536 * 2) {  // bf16 a = idx - 65536, b = each signed e2m1 value
        const unsigned short a = idx - 65536;
        const float fa = __uint_as_float((uint32_t)a << 16);
        const float e2[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        for (int s = 0; s < 16; ++s) {
            const float fb = (s & 8) ? -e2[s & 7] : e2[s & 7];
            const unsigned short b = (unsigned short)(__float_as_uint(fb) >> 16);
            for (int k = 0; k < 16; ++k) {
                const float c = __uint_as_float(kC[k]);
                const float r0 = fmaf(fa, fb, c), r1 = mix_b(a, b, c);
                nb += __float_as_uint(r0) != __float_as_uint(r1) && !(r0 != r0 && r1 != r1); ++nt;
            }
        }
    }
    atomicAdd(bad, nb); atomicAdd(tot, nt);
}
int main() {
    unsigned long long *d, h[2];
    cudaMalloc(&d, 16); cudaMemset(d, 0, 16);
    chk<<<512, 256>>>(d, d + 1);
    cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
    printf("MIXED-FMA check: %llu of %llu results differ from fmaf (%s)\n", h[0], h[1], cudaGetErrorString(cudaGetLastError()));
    return h[0] != 0;
}
