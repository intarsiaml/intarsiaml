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

/* moe_experts_gelu_bf16 / moe_experts_decode_gelu_bf16 / moe_experts_decode_v2_gelu_bf16 / moe_experts_mma_gelu_bf16 /
 * moe_experts_w4a16_gelu_bf16: the same launchers without the _gelu (same inputs, scratch and shape limits), with gelu_tanh
 * instead of SiLU in the expert activation: h = bf16(round_bf16(gelu_tanh(g)) * u) (OpAttrs::gelu, Gemma-4). Registered with
 * `.ext()`; the plain launchers are unchanged. kernels/cuda/moe.cu, moe_decode.cu, moe_grouped.cu, moe_w4a16.cu. */
int ik_moe_experts_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                             const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                             const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                             const ik_scratch* scratch, void* stream);
int ik_moe_experts_mma_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                 const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                 const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                 const ik_scratch* scratch, void* stream);
int ik_moe_experts_w4a16_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                   const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                   const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                   const ik_scratch* scratch, void* stream);
int ik_moe_experts_decode_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                    const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                    const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                    const ik_scratch* scratch, void* stream);
int ik_moe_experts_decode_v2_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
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
/* matmul_fp8_gemm_cublas: the GEMM of matmul_fp8_gemm_reg on cuBLAS's bf16 kernels over exactly expanded e4m3 operands
   (alpha = the gemm_reg epilogue scale, read on the device); bf16 output, K % 128 == 0. Scratch: 256 B + (M*K + N*K) bf16.
   kernels/cuda/fp8_gemm.cu. */
int ik_matmul_fp8_gemm_cublas(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                              const ik_scratch* scratch, void* stream);
/* matmul_fp8_pair_gemm_reg: Op::MatmulFp8Pair at prefill, bitwise two ik_matmul_fp8_gemm_reg calls on the same x: the
   activation quantiser runs once when the two input scales are equal (device-side check), and both GEMMs share one grid.
   Scratch: 2 * M * K bf16. kernels/cuda/fp8_gemm.cu. Needs K % 128 == 0. */
int ik_matmul_fp8_pair_gemm_reg(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* sc0, const ik_tensor* w1,
                                const ik_tensor* sc1, ik_tensor* out0, ik_tensor* out1, const ik_scratch* scratch,
                                void* stream);
/* matmul_nvfp4_gemm_reg / matmul_nvfp4_glu_gemm: dense NVFP4 prefill GEMMs with the weights dequantised in registers,
   bitwise matmul_nvfp4_gemm (and, for the GLU, two of them followed by silu_glu_bf16). kernels/cuda/nvfp4_gemm_reg.cu.
   Need K % 128 == 0 and N % 128 == 0 (GEMM) / N % 64 == 0 (GLU, bf16 out). */
int ik_matmul_nvfp4_gemm_reg(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                             const ik_tensor* global_scale, ik_tensor* out, void* stream);
int ik_matmul_nvfp4_glu_gemm(const ik_tensor* x, const ik_tensor* w_gate, const ik_tensor* block_scales_gate,
                             const ik_tensor* global_scale_gate, const ik_tensor* w_up, const ik_tensor* block_scales_up,
                             const ik_tensor* global_scale_up, ik_tensor* out, void* stream);

/* matmul_fp8_gemm_lt: matmul_fp8_gemm (same arguments, same scratch) with the W8A8 products on cuBLASLt's e4m3 tensor-core
 * GEMM: the same e4m3 activations, a different accumulation (not bitwise the default kernels), one pinned algorithm per
 * (N, K) so rows do not depend on M; see kernels/cuda/fp8_gemm_lt.cu. DEFAULT ON since 2026-10-23 (INTARSIA_CUDA_FP8_GEMM_LT=0 switches it off). Needs
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

/* Row-table launches (batched decode, `OpCall::rows`, R3): one launch for the T = 1 decode call of every row of a batched step.
 * `tab` is u8 [rows, stride, 8] (little-endian u64 device addresses, 8-byte aligned); the per-sequence state tensor arguments
 * (`state0`, `cache0`, `kc0`/`vc0`) are row 0's buffers, shape / dtype carriers only. Sequence r's state is the buffer at entry
 * [r][index] of the table (`byte_off` bytes in for the delta-rule state); the attention kernel reads the K cache at that entry and
 * the V cache at the next one. `pos` is i32 [rows]: the position of each row's token (the cache length before the append).
 * Row r is BITWISE the one-row launch it twins (tests/row_tables.rs): ik_gdn_conv_bf16 (conv + state update in one launch),
 * ik_gdn_delta_rule_bf16_split, ik_kv_append_bf16, ik_attn_decode_bf16_splitkv (same scratch, `rows` rows of it). Rope with a
 * device position of one entry per token (ik_rope_bf16's `pos` of numel T) is the same table-free extension. */
int ik_gdn_conv_bf16_rows(const ik_tensor* x, const ik_tensor* taps, ik_tensor* y, const ik_tensor* state0, const ik_tensor* tab,
                          int64_t stride, int64_t index, void* stream);
int ik_gdn_delta_rule_bf16_split_rows(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                                      const ik_tensor* dt_bias, float eps, ik_tensor* o, const ik_tensor* state0,
                                      const ik_tensor* tab, int64_t stride, int64_t index, int64_t byte_off, void* stream);
int ik_kv_append_bf16_rows(const ik_tensor* new_kv, const ik_tensor* cache0, const ik_tensor* tab, int64_t stride,
                           int64_t index, const ik_tensor* pos, void* stream);
int ik_attn_decode_bf16_splitkv_rows(const ik_tensor* q, const ik_tensor* kc0, const ik_tensor* vc0, float scale, ik_tensor* out,
                                     const ik_scratch* scratch, const ik_tensor* tab, int64_t stride, int64_t index,
                                     const ik_tensor* pos, void* stream);

/* topk_f32 (R3b, kernels/cuda/topk.cu): the k largest values of each f32 row [R, V] in the sampler's order (value descending, ties by
 * ascending index, -0.0 == +0.0), with their indices: ids i32 [R, k], vals f32 [R, k]; nan i32 [R] is 1 for a row that holds a NaN (its
 * entries are not meaningful). 1 <= k <= 64. Scratch: R * 32 * k * 8 bytes. Not an op: the model calls it for device-side sampling. */
int ik_topk_f32(const ik_tensor* logits, int64_t k, ik_tensor* ids, ik_tensor* vals, ik_tensor* nan, const ik_scratch* scratch,
                void* stream);

/* matmul_mxfp4 (M1 drill, kernels/cuda/mxfp4.cu): Op::MatmulMxfp4, OCP MXFP4 weights (u8 e2m1 pairs [N, K/2], u8 e8m0
 * scales [N, K/32]) and act f32 [1] (0: bf16 activations, 1: dynamic MXFP4 QDQ, Quark "even"). Any M (GEMV per 8 rows). */
int ik_matmul_mxfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                         ik_tensor* out, void* stream);

/* matmul_exl3_gemv: x[M,K] (bf16 / f16 / f32), trellis i16 [K/16,N/16,8*k2] (k2 = 2 * bits: 2, 4, .., 16 with any codebook,
 * 3 / 5 / 7 with mul1), suh f16 [K], svh f16 [N], codebook i32 [1] ON THE DEVICE (0 = 3inst, 1 = mcg, 2 = mul1; any other
 * value, or a codebook that does not exist at the width, fills out with NaN) -> out[M,N] (bf16 / f16 / f32) =
 * x @ (diag(suh) H_K W_q H_N diag(svh)), H the 128-point Hadamard / sqrt(128), see kernels/cuda/exl3.cu. K, N multiples of
 * 128; x, out, suh, svh 16- / 8-byte aligned. Scratch: the rotated input, fp16 [min(M,128),K] (16-byte aligned). The
 * rotation is rounded to fp16 before the tensor-core multiply. Any M: rows go through in slices of 128, so the scratch is
 * min(M,128)*K*2 bytes; every 16 rows decode the weight tiles again (a decode kernel, not a prefill GEMM). */
int ik_matmul_exl3_gemv(const ik_tensor* x, const ik_tensor* trellis, const ik_tensor* suh, const ik_tensor* svh,
                        const ik_tensor* codebook, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* matmul_exl3_gemm: the same op and arguments, a prefill GEMM: each block decodes a weight tile once for 64 or 128 rows
 * (ldmatrix + mma against shared-memory slices of the rotated input), and its output is bit for bit the GEMV's. Scratch:
 * the rotated input, fp16 [min(M,512),K] (16-byte aligned); rows go through in slices of 512. */
int ik_matmul_exl3_gemm(const ik_tensor* x, const ik_tensor* trellis, const ik_tensor* suh, const ik_tensor* svh,
                        const ik_tensor* codebook, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* Row-exact multi-row twins of the decode GEMVs (rows_exact kernels, `OpCall::exact_rows`; hybrid continuous batching, stage R0):
 * the same arguments and launch contract as the 1-row-GEMV launcher named in each comment, any M in 1..=16, and row m of the
 * output is BITWISE the M = 1 call on row m alone
 * (tests/rows_exact.rs: every M, W8A8 and W8A16, the 35B and 8B shapes). */
/* = ik_matmul_fp8_gemv */
int ik_matmul_fp8_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream);
/* = ik_matmul_nvfp4_gemv */
int ik_matmul_nvfp4_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                              const ik_tensor* global_scale, ik_tensor* out, void* stream);
/* = ik_matmul_bf16_gemv / ik_logits_bf16_gemv */
int ik_matmul_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
int ik_logits_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
/* = ik_moe_experts_decode_v2_bf16 / ik_moe_experts_decode_v2_gelu_bf16 with T <= 16 tokens; same scratch
 * (moe_experts_decode_scratch). */
int ik_moe_experts_decode_v2_rows_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                       const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                       const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                       const ik_scratch* scratch, void* stream);
int ik_moe_experts_decode_v2_gelu_rows_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                            const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                            const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                            const ik_scratch* scratch, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* IK_CUDA_ABI_H */
