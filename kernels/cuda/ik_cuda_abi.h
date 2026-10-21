/* CUDA-only launchers: declared here instead of the shared kernels/abi/ik_abi.h (the HIP backend does the same with
 * kernels/hip/ik_hip_abi.h). The CUDA backend's `every_launcher_is_declared_in_the_abi_header` test reads both. */
#ifndef IK_CUDA_ABI_H
#define IK_CUDA_ABI_H

#include "abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* moe_experts_w4a16_bf16: moe_experts_mma_bf16 (same inputs, same scratch, bitwise the same outputs) with the fp4 weights
 * dequantised in registers instead of a shared-memory bf16 tile, see kernels/cuda/moe_w4a16.cu. Needs H % 128 == 0,
 * I % 128 == 0 and 8-byte aligned scale tensors. */
int ik_moe_experts_w4a16_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                              const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                              const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                              const ik_scratch* scratch, void* stream);

/* moe_experts_nvfp4_mma_bf16: moe_experts_mma_bf16 (same inputs, scratch = the grouped layout plus the quantised
 * activations, see kernels/cuda/moe_nvfp4_mma_sm120a.cu) with the expert GEMMs on the block-scaled FP4 tensor core:
 * activations are quantised to e2m1 + one ue4m3 scale per 16 values + one f32 scale per row, so the result differs from
 * the bf16-activation kernels by the activation quantisation error. Needs H % 128 == 0, I % 128 == 0. Built for sm_120a /
 * sm_121a only; elsewhere the symbol exists and returns IK_ERR_UNSUPPORTED. */
int ik_moe_experts_nvfp4_mma_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                  const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                  const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                  const ik_scratch* scratch, void* stream);

/* matmul_fp8_gemm_reg: matmul_fp8_gemm (same arguments, same scratch, bitwise the same outputs for finite e4m3 weights)
 * with the weights dequantised in registers instead of a bf16 shared-memory tile and a 128 x 128 block tile, see
 * kernels/cuda/fp8_gemm.cu. Needs K % 128 == 0. */
int ik_matmul_fp8_gemm_reg(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                           const ik_scratch* scratch, void* stream);

/* matmul_fp8_gemm_lt: matmul_fp8_gemm (same arguments, same scratch) with the W8A8 products on cuBLASLt's e4m3 tensor-core
 * GEMM: the same e4m3 activations, a different accumulation (not bitwise the default kernels), one pinned algorithm per
 * (N, K) so rows do not depend on M; see kernels/cuda/fp8_gemm_lt.cu. OPT-IN (INTARSIA_CUDA_FP8_GEMM_LT=1). Needs
 * K % 128 == 0, N % 16 == 0. */
int ik_matmul_fp8_gemm_lt(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                          const ik_scratch* scratch, void* stream);

/* gdn_delta_rule_bf16_split: ik_gdn_delta_rule_bf16 (same arguments, bitwise the same outputs and state) with each state
 * row's element-wise work split over several lanes, see kernels/cuda/gdn.cu. dk in {64, 128}; state 16-byte aligned. */
int ik_gdn_delta_rule_bf16_split(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                                 const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream);

/* attn_decode_f8kv_splitkv / attn_prefill_f8kv_splitkv / attn_prefill_f8kv_flash: ik_attn_decode_bf16_splitkv /
 * ik_attn_prefill_bf16_splitkv / ik_attn_prefill_bf16_flash (same arguments and scratch, bf16 q and out) over an fp8_e4m3 KV
 * cache [Nkv, cap, D]; bitwise the bf16 kernels on the cache upcast to bf16, see kernels/cuda/attention_splitkv.cu and
 * attention_flash.cu. Selected only when the cache is fp8_e4m3 (`--kv-dtype fp8_e4m3`). */
int ik_attn_decode_f8kv_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                                ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream);
int ik_attn_prefill_f8kv_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                                 ik_tensor* out, const ik_scratch* scratch, void* stream);
int ik_attn_prefill_f8kv_flash(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                               ik_tensor* out, void* stream);

/* matmul_mxfp4 (M1 drill, kernels/cuda/mxfp4.cu): Op::MatmulMxfp4, OCP MXFP4 weights (u8 e2m1 pairs [N, K/2], u8 e8m0
 * scales [N, K/32]) and act f32 [1] (0: bf16 activations, 1: dynamic MXFP4 QDQ, Quark "even"). Any M (GEMV per 8 rows). */
int ik_matmul_mxfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                         ik_tensor* out, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* IK_CUDA_ABI_H */
