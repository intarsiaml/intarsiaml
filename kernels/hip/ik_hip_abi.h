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
/* gdn_delta_rule_bf16_v3 (lane hp, docs/results/2026-10-22-hp-hip-prefill-roofline.md): v2's operations in v2's order,
 * rescheduled (the output dot of token t and the k dot of t + 1 in one pass over the state; q / k normalised and staged
 * two tokens ahead in a four-buffer LDS ring with no barrier), so bitwise ik_gdn_delta_rule_bf16_v2; dk = 128. */
int ik_gdn_delta_rule_bf16_v3(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                              const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream);

/* attn_prefill_bf16_flash_v2 (docs/results/2026-10-19-hip-round5.md): ik_attn_prefill_bf16_flash with INTARSIA_HIP_FLASH_HPB
 * (default 4) query heads of one kv head per block, so a K / V tile is staged once for all of them; every wave runs the
 * v1 kernel's instructions for its rows, so bitwise v1. */
int ik_attn_prefill_bf16_flash_v2(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                                  ik_tensor* out, void* stream);
/* Kernel-variant selection inside one op (docs/results/2026-10-19-hip-round6.md, -round7.md), for the registry pins: the query
 * heads per block flash v2 starts from (INTARSIA_HIP_FLASH_HPB, default 4), and the grouped MoE projections' variant bits for a
 * pair tile of `pairs` (INTARSIA_HIP_MOE_MW), gate_up and down (round 7 defaults: 45 / 25 at 64 pairs, 0 / 41 at 32).
 * Host-only, no device needed. */
int ik_hip_flash_hpb(void);
/* The staging / math variant flash v2 runs (INTARSIA_HIP_FLASH_VP, a bit mask, default 5; docs/results/2026-10-22-pf-prefill-ctx.md):
 * every value gives the same bits. Host-only. */
int ik_hip_flash_vp(void);
int ik_hip_moe_mw(int pairs);
int ik_hip_moe_mw_down(int pairs);
/* round 9 (docs/results/2026-10-19-hip-round9.md): the number of launch pieces ik_split_pieces (ik_hip.h) gives a launch
 * estimated at `est_us` with at most `max_pieces` pieces, under the current INTARSIA_HIP_SPLIT_US / _SPLIT_MIN_US. Host-only. */
int ik_hip_split_pieces(double est_us, int64_t max_pieces);
/* round 9: 1 when split-KV stage 1 runs at 8 waves per SIMD (splitkv_kernel_o8; opt-in, INTARSIA_HIP_SPLITKV_OCC=1). Host-only. */
int ik_hip_splitkv_occ(void);

/* moe_experts_bf16_v3 (docs/results/2026-10-19-hip-round5.md): ik_moe_experts_bf16_v2 with the decode path (T * K <= 16)
 * in one launch (gate_up and down + reduce blocks in one grid, the down blocks wait on a per-stream counter); the same
 * expressions in the same order, so bitwise v2 for every shape. */
int ik_moe_experts_bf16_v3(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                           const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                           const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* Row-exact multi-row twins of the decode kernels (lane hb, hybrid continuous batching on HIP, docs/results/2026-10-23-hb-hip-hybrid-batching.md):
 * row m (token t of the MoE) of an M-row call is BITWISE the one-row kernel on row m alone, for every M from 1 to 16
 * (G-B, crates/intarsia-backend-hip/tests/rows_exact.rs). Registered `rows_exact`: only a batched step's calls pick them.
 *   matmul_fp8_gemv_rows    = ik_matmul_fp8_gemv_v2 per row (K % 16 == 0, K * 2 <= 48 KiB; M > 4 in groups of at most 48 KiB / (K * 2)
 *                             and 12 rows)
 *   matmul_nvfp4_gemv_rows  = ik_matmul_nvfp4_gemv_v2 per row (M > 4 in groups of at most 4: the M = 8 instantiation is 4x slower than M = 4)
 *   matmul_bf16_gemv_rows / logits_bf16_gemv_rows = ik_matmul_bf16_gemv / ik_logits_bf16_gemv per row (any K, M rows in one launch)
 *   moe_experts_bf16_rows   = ik_moe_experts_bf16_v2 per token, never the tensor-core path (T <= 16) */
int ik_matmul_fp8_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream);
int ik_matmul_nvfp4_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                              const ik_tensor* global_scale, ik_tensor* out, void* stream);
int ik_matmul_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
int ik_logits_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
int ik_moe_experts_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                             const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                             const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* Core-op fusion phase 1 (docs/results/2026-10-19-core-op-fusion-p1.md): three row-local folds, no cross-block wait, each
 * bitwise the two launches it replaces. M == 1 (decode) for the fp8 ones, 1 <= M <= 8 for the nvfp4 one; bf16 activations.
 *   rmsnorm_add_matmul_fp8_pair_gemv (P5, fp8.hip): ik_rmsnorm_add_bf16 then ik_matmul_fp8_pair_gemv on the normed row
 *     (the attention block's input norm and its k / v projections); also stores normed and sum.
 *   sigmoid_mul_matmul_fp8_gemv (P4b, fp8.hip): ik_sigmoid_mul_bf16 then ik_matmul_fp8_gemv_v2 (the attention output gate
 *     and o_proj).
 *   shared_down_combine_nvfp4_gemv (P2a, moe_shared.hip): ik_matmul_nvfp4_gemv_v2 then ik_moe_combine_bf16 (the MoE
 *     shared-expert down projection and the combine). */
int ik_rmsnorm_add_matmul_fp8_pair_gemv(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, float w_offset,
                                        const ik_tensor* w0, const ik_tensor* scales0, const ik_tensor* w1,
                                        const ik_tensor* scales1, ik_tensor* normed, ik_tensor* sum, ik_tensor* out0,
                                        ik_tensor* out1, void* stream);
int ik_sigmoid_mul_matmul_fp8_gemv(const ik_tensor* a, const ik_tensor* gate, const ik_tensor* w, const ik_tensor* scales,
                                   ik_tensor* out, void* stream);
int ik_shared_down_combine_nvfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs,
                                      const ik_tensor* routed, const ik_tensor* logit, ik_tensor* out, void* stream);

/* P2' (docs/results/2026-10-19-p2prime-moeblock.md), moe_block.hip: Op::MoeBlock in one launch, bitwise moe_topk_bf16_v2 ->
 * moe_experts_bf16_v2 -> matmul_nvfp4_glu_gemv (gate / up gemv + silu_glu at T > 1) -> shared_down_combine_nvfp4_gemv.
 * `in` = the op's 17 inputs as one contiguous array; T * K <= 16, K <= 8, E <= 512; scratch ik_moe_block_scratch_bytes. */
int ik_moe_block_bf16(const ik_tensor* in, int top_k, int norm, ik_tensor* out, const ik_scratch* scratch, void* stream);
int64_t ik_moe_block_scratch_bytes(int64_t T, int64_t K, int64_t I, int64_t IS);

/* Core-op fusion phase 2 (docs/results/2026-10-19-core-op-fusion-p2.md), each bitwise the two launches it replaces:
 *   gate_split_qk_norm_rope_kv_bf16 (P4a, gate_qkr.hip): ik_gate_split_bf16 then ik_qk_norm_rope_kv_bf16 of the split-off
 *     query; qg [T,N,2D], gate out [T,N,D]; the prologue's constraints.
 *   gdn_conv_delta_bf16 (P3, gdn_fused.hip): ik_gdn_conv_bf16 then ik_gdn_delta_rule_bf16_v2 of its output, both states in
 *     place; dk = 128, K = 4, T <= 32. The q / k conv state is stored by the last block of each key head (a per-stream
 *     ticket counter, no wait); without a free counter slot, a second launch stores it. */
int ik_gate_split_qk_norm_rope_kv_bf16(const ik_tensor* qg, const ik_tensor* k, const ik_tensor* v, const ik_tensor* qw,
                                       const ik_tensor* kw, float eps, float w_offset, int64_t pos0, float theta,
                                       int64_t rotary_dim, ik_tensor* q_out, ik_tensor* gate, ik_tensor* kcache,
                                       ik_tensor* vcache, const ik_tensor* pos, void* stream);
int ik_gdn_conv_delta_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* a, const ik_tensor* b,
                           const ik_tensor* a_log, const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state,
                           ik_tensor* cstate, void* stream);

/* matmul_mxfp4 (M1 drill, kernels/hip/mxfp4.hip): Op::MatmulMxfp4, OCP MXFP4 weights (u8 e2m1 pairs [N, K/2], u8 e8m0
 * scales [N, K/32]) and act f32 [1] (0: bf16 activations, 1: dynamic MXFP4 QDQ of the activations, Quark "even").
 * gemv: 1 <= M <= 8, scratch optional (with it the activations are QDQ'd once, not once per output row). mma: any M.
 * scratch >= M * K * 2 bytes, 16-byte aligned (the activations, QDQ'd or copied). */
int ik_matmul_mxfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                         ik_tensor* out, const ik_scratch* scratch, void* stream);
int ik_matmul_mxfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                        ik_tensor* out, const ik_scratch* scratch, void* stream);
/* matmul_mxfp4_pair_gemv (lane P1): Op::MatmulMxfp4Pair, two MXFP4 projections of one x with one act input; M == 1 one launch
 * (x staged once), else the two single gemvs (scratch as matmul_mxfp4_gemv). */
int ik_matmul_mxfp4_pair_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1,
                              const ik_tensor* s1, const ik_tensor* act, ik_tensor* out0, ik_tensor* out1,
                              const ik_scratch* scratch, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* IK_HIP_ABI_H */
