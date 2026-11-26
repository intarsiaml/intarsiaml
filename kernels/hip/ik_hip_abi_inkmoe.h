/* Launchers of the Inkling-2-Mini routed experts: MoeExperts with the SiTU activation (NVFP4, .situ()), MoeExpertsMxfp8 (72), and the router MoeRoute (DSv4.1's op; the fused InkRoute (76) is optional) on HIP
 * (lane ink-hip-moe; crates/intarsia-backend-hip/src/inkmoe.rs holds the table, kernels.rs checks that every launcher of that table is
 * declared here). Contract: crates/intarsia-core/src/inkling2/ops.rs. Every launcher returns an ik_status, never synchronises
 * and never allocates. */
#ifndef IK_HIP_ABI_INKMOE_H
#define IK_HIP_ABI_INKMOE_H

#include "../abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MoeExperts with ink.situ (NVFP4): x bf16 [T,H], ids i32 [T,K], weights f32 [T,K], gate_up e2m1 [E,2I,H/2] + e4m3 scales
 * [E,2I,H/16], down [E,H,I/2] + [E,H,I/16], global scales f32 [E,3] -> out bf16 | f32 [T,H]. aq: bit 0 = ink.act_quant, bit 1 =
 * ink.moe_layout; both must be 0 (IK_ERR_UNSUPPORTED otherwise). Scratch: quant::ops::moe_experts_scratch. The _rows twin is the
 * same launcher (every path is rows-exact). */
int ik_moe_experts_situ_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                             const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                             float gate_cap, float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream);
int ik_moe_experts_situ_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                                  const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                                  float gate_cap, float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* MoeExpertsMxfp8 (72): w13 e4m3 [E,2I,H] + E8M0 scales u8 [E,2I,H/32], w2 [E,H,I] + [E,H,I/32]; act 0 silu, 1 gelu_tanh,
 * 2 situ; aq must be 0. */
int ik_moe_experts_mxfp8_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                              const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                              float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream);
int ik_moe_experts_mxfp8_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                                   const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                                   float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* benchmarks / tests: 1 = only gate_up, 2 = only down, 3 = both launches returning at once, 4 = returning after the grouping,
 * 0 = both; the load windows of gate_up and down (-1 = the env
 * INTARSIA_HIP_INK_MOE_PRE / the default); the step form (1 lean, 0 the first form, -1 = INTARSIA_HIP_INK_MOE_LEAN / lean) */
void ik_ink_moe_probe(int v);
void ik_ink_moe_set_pre(int gu, int dn);
void ik_ink_moe_set_lean(int v);
/* the smallest LDS size class a call may take (0 = by the call; 1; 2 = the largest; 4 = by the call, with up to 32 pairs grouped
 * inside the wave instead of LDS tables); bitwise the same output */
void ik_ink_moe_set_sz(int v);
/* pairs per accumulator batch below the pair table (4 | 8; anything else = the default rule); bitwise the same output */
void ik_ink_moe_set_nb(int v);
/* the tensor-core prefill of the ordinary names: 0 off, 1 the rule (64-pair tiles from 16 pairs per expert), 32 | 64 every
 * pair-table call with that tile, -1 = INTARSIA_HIP_INK_MOE_WMMA / the default (off) */
void ik_ink_moe_set_wmma(int v);
/* the one-wave grouping of size class 0 (at most 32 pairs; lane ink-hip-dec2): 1 on, 0 the shared-memory grouping, -1 =
 * INTARSIA_HIP_INK_DEC_FAST bit 1 / the default (on); bitwise the same output */
void ik_ink_moe_set_dec(int v);

#ifdef __cplusplus
}
#endif

#endif
