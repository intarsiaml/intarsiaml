#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
__device__ __forceinline__ float fma_f32_f16(unsigned short a, unsigned short b, float c) {
    float d;
    asm("fma.rn.f32.f16 %0, %1, %2, %3;" : "=f"(d) : "h"(a), "h"(b), "f"(c));
    return d;
}
__device__ __forceinline__ uint32_t cvt2(unsigned short v) {
    __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
    return *reinterpret_cast<uint32_t*>(&r);
}
extern "C" __global__ void mixed(const uint32_t* w, const uint32_t* x, float* out) {
    uint32_t a = cvt2(w[threadIdx.x] & 0xFFFF), b = cvt2(x[threadIdx.x] & 0xFFFF);
    float s = out[threadIdx.x];
    s = fma_f32_f16((unsigned short)(a & 0xFFFF), (unsigned short)(b & 0xFFFF), s);
    s = fma_f32_f16((unsigned short)(a >> 16), (unsigned short)(b >> 16), s);
    out[threadIdx.x] = s;
}
extern "C" __global__ void plain(const uint32_t* w, const uint32_t* x, float* out) {
    uint32_t a = cvt2(w[threadIdx.x] & 0xFFFF), b = cvt2(x[threadIdx.x] & 0xFFFF);
    float2 fa = __half22float2(*reinterpret_cast<__half2*>(&a)), fb = __half22float2(*reinterpret_cast<__half2*>(&b));
    float s = out[threadIdx.x];
    s = fmaf(fa.x, fb.x, s);
    s = fmaf(fa.y, fb.y, s);
    out[threadIdx.x] = s;
}
