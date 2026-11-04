#pragma once
struct __nv_bfloat16 { unsigned short x; __host__ __device__ __nv_bfloat16() {} __host__ __device__ __nv_bfloat16(float) {} };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
__host__ __device__ float __bfloat162float(__nv_bfloat16);
__host__ __device__ __nv_bfloat16 __float2bfloat16_rn(float);
__host__ __device__ __nv_bfloat162 __floats2bfloat162_rn(float, float);
__host__ __device__ float2 __bfloat1622float2(__nv_bfloat162);
