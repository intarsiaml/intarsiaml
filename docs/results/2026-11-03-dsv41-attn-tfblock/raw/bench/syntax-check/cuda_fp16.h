#pragma once
#include <cuda_runtime.h>
struct __half_raw { unsigned short x; };
struct __half { unsigned short x; __host__ __device__ __half() {} __host__ __device__ __half(float) {} __host__ __device__ __half(__half_raw) {} };
__host__ __device__ float __half2float(__half);
__host__ __device__ __half __float2half_rn(float);
