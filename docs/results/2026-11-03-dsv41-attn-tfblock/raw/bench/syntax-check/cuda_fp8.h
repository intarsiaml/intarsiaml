#pragma once
typedef unsigned char __nv_fp8_storage_t;
enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE };
enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 };
#include <cuda_fp16.h>
__device__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(float, __nv_saturation_t, __nv_fp8_interpretation_t);
__device__ __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t, __nv_fp8_interpretation_t);
