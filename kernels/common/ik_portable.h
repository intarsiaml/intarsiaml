/*
 * CUDA/HIP shims shared by the simple kernels (kernels/cuda today, kernels/hip later).
 * Only things the kernels actually use: bf16 conversion, 8-wide bf16 vectors,
 * warp/block reductions. Device code only; launcher plumbing lives in ik_launch.h.
 */
#ifndef IK_PORTABLE_H
#define IK_PORTABLE_H

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 ik_bf16;
typedef __hip_bfloat162 ik_bf162;
#define IK_WARP 64
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
typedef __nv_bfloat16 ik_bf16;
typedef __nv_bfloat162 ik_bf162;
#define IK_WARP 32
#endif

#define IK_DEVI static __device__ __forceinline__

IK_DEVI float ik_b2f(ik_bf16 v) { return __bfloat162float(v); }
IK_DEVI ik_bf16 ik_f2b(float v) { return __float2bfloat16_rn(v); }

/* 8 x bf16 = 16 bytes, moved with one vector load/store */
IK_DEVI void ik_unpack8(const uint4& u, float* f) {
    const ik_bf162* p = reinterpret_cast<const ik_bf162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 t = __bfloat1622float2(p[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

IK_DEVI uint4 ik_pack8(const float* f) {
    uint4 u;
    ik_bf162* p = reinterpret_cast<ik_bf162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return u;
}

IK_DEVI float ik_warp_sum(float v) {
#pragma unroll
    for (int o = IK_WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

IK_DEVI float ik_warp_max(float v) {
#pragma unroll
    for (int o = IK_WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Block-wide sum / max, result broadcast to every thread. `red` is a shared
 * array of at least blockDim.x / IK_WARP floats. Every thread must call it. */
IK_DEVI float ik_block_sum(float v, float* red) {
    const int lane = threadIdx.x % IK_WARP, w = threadIdx.x / IK_WARP, nw = (blockDim.x + IK_WARP - 1) / IK_WARP;
    v = ik_warp_sum(v);
    __syncthreads();  // `red` may still be in use by a previous reduction
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < nw; ++i) r += red[i];  // fixed order: deterministic
    return r;
}

IK_DEVI float ik_block_max(float v, float* red) {
    const int lane = threadIdx.x % IK_WARP, w = threadIdx.x / IK_WARP, nw = (blockDim.x + IK_WARP - 1) / IK_WARP;
    v = ik_warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = red[0];
    for (int i = 1; i < nw; ++i) r = fmaxf(r, red[i]);
    return r;
}

#endif /* IK_PORTABLE_H */
