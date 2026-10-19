/* Launchers that only the HIP backend defines (declared here, not in kernels/abi/ik_abi.h, which the other backends share;
 * crates/intarsia-backend-hip/src/kernels.rs checks this file as well as ik_abi.h). */
#ifndef IK_HIP_ABI_H
#define IK_HIP_ABI_H

#include "../abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* matmul_fp8_mma_f8 (docs/results/2026-10-17-hip-prefill.md): the W8A8 prefill GEMM on the gfx12 fp8 x fp8 WMMA
 * (ik_f8_gemm.h). Same op as matmul_fp8_mma; scratch: M * K bytes (the quantised activations, 16-byte aligned);
 * weight-only activations (input_scale <= 0) run the dequantise-to-bf16 kernel of matmul_fp8_mma inside. */
int ik_matmul_fp8_mma_f8(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                         const ik_scratch* scratch, void* stream);

/* matmul_fp8_mma_v2 (docs/results/2026-10-19-hip-round3.md): matmul_fp8_mma with the W8A8 activations quantised once per
 * call into scratch (M * K bytes, 16-byte aligned) and the weights decoded by the gfx12 fp8 conversion instruction; same
 * bf16 tiles and WMMA order, so bitwise matmul_fp8_mma for every shape and both activation modes. */
int ik_matmul_fp8_mma_v2(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                         const ik_scratch* scratch, void* stream);

/* qk_norm_rope_kv (docs/results/2026-10-19-hip-round4.md): Op::QkNormRopeKv, the attention prologue in one launch:
 * q_out = rope(qk_norm(q, qw)), k cache rows pos0.. = rope(qk_norm(k, kw)), v cache rows pos0.. = v. Bitwise the unfused
 * ik_qk_norm_bf16 -> ik_rope_bf16 -> ik_kv_append_bf16 chain. head_dim % 8 == 0 and <= 256, N / Nkv <= 8, 16-byte
 * aligned q / k / weights / q_out; caches bf16 or fp8_e4m3; optional device position `pos` as ik_rope_bf16. */
int ik_qk_norm_rope_kv_bf16(const ik_tensor* q, const ik_tensor* k, const ik_tensor* v, const ik_tensor* qw,
                            const ik_tensor* kw, float eps, float w_offset, int64_t pos0, float theta, int64_t rotary_dim,
                            ik_tensor* q_out, ik_tensor* kcache, ik_tensor* vcache, const ik_tensor* pos, void* stream);

/* gdn_delta_rule_bf16_v2: ik_gdn_delta_rule_bf16 (dk = 128) without its stalls (raw next-token inputs in a register ring,
 * LDS-only barriers, DPP warp sums, gates 32 tokens at a time); the same operations in the same order, so bitwise
 * ik_gdn_delta_rule_bf16. */
int ik_gdn_delta_rule_bf16_v2(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                              const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream);

/* attn_prefill_bf16_flash_v2 (docs/results/2026-10-19-hip-round5.md): ik_attn_prefill_bf16_flash with INTARSIA_HIP_FLASH_HPB
 * (default 2) query heads of one kv head per block, so a K / V tile is staged once for all of them; every wave runs the
 * v1 kernel's instructions for its rows, so bitwise v1. */
int ik_attn_prefill_bf16_flash_v2(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                                  ik_tensor* out, void* stream);

/* moe_experts_bf16_v3 (docs/results/2026-10-19-hip-round5.md): ik_moe_experts_bf16_v2 with the decode path (T * K <= 16)
 * in one launch (gate_up and down + reduce blocks in one grid, the down blocks wait on a per-stream counter); the same
 * expressions in the same order, so bitwise v2 for every shape. */
int ik_moe_experts_bf16_v3(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                           const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                           const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* IK_HIP_ABI_H */
