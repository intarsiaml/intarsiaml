/* Declarations shared between the launcher files and ik_runtime.cu (not part of the public ABI). */
#ifndef IK_INTERNAL_H
#define IK_INTERNAL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* cuBLAS handle + workspace for the current device; created once, outside any launcher. */
int ik_cublas_init(int device);
void ik_cublas_shutdown(int device);
/* The weight-only (input_scale <= 0) fallback of ik_matmul_fp8_gemm_f8: the bf16 dequantise-once GEMM of fp8_gemm.cu, launched
 * from that translation unit so its shared-memory attribute is set on the instance that runs (a template instantiated in two
 * files has two device copies and one static "attribute set" guard). Exits on the device when the scale is positive. */
int ik_fp8_gemm_weight_only_internal(const void* x, const void* w, const void* scales, void* out, int out_is_f32, int64_t M,
                                     int64_t N, int64_t K, void* stream);
/* matmul.cu: row-major bf16 out[M,N] = alpha * x[M,K] @ w[N,K]^T on the shared cuBLAS handle; alpha_beta_dev = {alpha, 0}
 * in device memory. */
int ik_cublas_gemm_bf16_devalpha(const void* x, const void* w, void* out, int64_t M, int64_t N, int64_t K,
                                 const float* alpha_beta_dev, void* stream);
/* ik_runtime.cu: the ik_rt_malloc allocation holding device address p (0 = found). fp8_gemm.cu: drop the bf16 weight copies
 * of matmul_fp8_gemm_cublas made from weights inside the allocation at `base` (called by ik_rt_free before cudaFree). */
int ik_rt_alloc_base(const void* p, uintptr_t* base);
void ik_fp8_bf16_cache_forget(const void* base);
#ifdef __cplusplus
}
#endif
#endif
