// minimal stand-in for the CUDA runtime headers, for a clang -fsyntax-only / NVPTX device-only parse check on a machine without the CUDA SDK
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#include "__clang_cuda_builtin_vars.h"
struct dim3 { unsigned x, y, z; __host__ __device__ dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint4 { unsigned x, y, z, w; };
struct float2 { float x, y; };
static __host__ __device__ uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return uint4{a, b, c, d}; }
typedef struct CUstream_st* cudaStream_t;
enum cudaError_t { cudaSuccess = 0 };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8, cudaFuncAttributePreferredSharedMemoryCarveout = 9 };
enum cudaDeviceAttr { cudaDevAttrMaxSharedMemoryPerMultiprocessor, cudaDevAttrReservedSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                      cudaDevAttrMaxSharedMemoryPerBlock, cudaDevAttrMaxThreadsPerMultiProcessor, cudaDevAttrMultiProcessorCount,
                      cudaDevAttrMaxRegistersPerMultiprocessor };
struct cudaFuncAttributes { size_t sharedSizeBytes, localSizeBytes; int numRegs; };
__host__ cudaError_t cudaGetLastError();
__host__ cudaError_t cudaGetDevice(int*);
__host__ cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
template <class T> __host__ cudaError_t cudaFuncSetAttribute(T*, cudaFuncAttribute, int);
template <class T> __host__ cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, T*);
template <class T> __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*, T*, int, size_t);
extern "C" cudaError_t cudaConfigureCall(dim3, dim3, size_t = 0, cudaStream_t = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void* stream = 0);
// device builtins
static __device__ void __syncwarp(unsigned m = 0xffffffffu) { __nvvm_bar_warp_sync(m); }
static __device__ float __shfl_xor_sync(unsigned m, float v, int o, int w = 32) { return __nvvm_shfl_sync_bfly_f32(m, v, o, 0x1f); }
static __device__ uint4 __ldg(const uint4* p) { return *p; }
static __device__ unsigned __ldg(const unsigned* p) { return *p; }
static __device__ size_t __cvta_generic_to_shared(const void* p) { return (size_t)(uintptr_t)p; }
__device__ float expf(float x);
__device__ float tanhf(float);
__device__ float fmaf(float, float, float);
__device__ float sqrtf(float);
__device__ float rsqrtf(float);
__device__ float fabsf(float);
__device__ float fminf(float, float);
__device__ float fmaxf(float a, float b);
__device__ void sincosf(float, float*, float*);
static __device__ int64_t min(int64_t a, int64_t b) { return a < b ? a : b; }
__device__ float log1pf(float);
static __device__ unsigned __float_as_uint(float f) { return __builtin_bit_cast(unsigned, f); }
static __device__ float __uint_as_float(unsigned u) { return __builtin_bit_cast(float, u); }
static __device__ int max(int a, int b) { return a > b ? a : b; }
static __device__ int64_t max(int64_t a, int64_t b) { return a > b ? a : b; }
#include "cuda_bf16.h"
__host__ const char* cudaGetErrorString(cudaError_t);
__host__ cudaError_t cudaMalloc(void**, size_t);
__host__ cudaError_t cudaFree(void*);
__host__ cudaError_t cudaMemcpy(void*, const void*, size_t, int);
__host__ cudaError_t cudaMemset(void*, int, size_t);
__host__ cudaError_t cudaDeviceSynchronize();
typedef struct CUevent_st* cudaEvent_t;
__host__ cudaError_t cudaEventCreate(cudaEvent_t*);
__host__ cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
__host__ cudaError_t cudaEventSynchronize(cudaEvent_t);
__host__ cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
