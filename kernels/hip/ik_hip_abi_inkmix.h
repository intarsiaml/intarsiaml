/* Launchers of the Inkling-2-Mini in-projection and sublayer-boundary ops: InkInProj (73), ConvMix (74), AttnResMix (67), SconvRes (69), LoopBlend (68) on HIP
 * (lane ink-hip-mix; crates/intarsia-backend-hip/src/inkmix.rs holds the table, kernels.rs checks that every launcher of that table is
 * declared here). Contract: crates/intarsia-core/src/inkling2/ops.rs. Every launcher returns an ik_status, never synchronises
 * and never allocates. */
#ifndef IK_HIP_ABI_INKMIX_H
#define IK_HIP_ABI_INKMIX_H

#include "../abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* kernels/hip/ink_proj.hip. x [S,T,D] bf16 (S = 1 or 3: segment j reads plane min(j, S-1)), w [Nq+Nk+Nv, D] bf16 -> pq [T,Nq], pk [T,Nk],
 * pv [T,Nv] bf16. _gemv: T <= 16, D % 16 == 0, 16-byte aligned x / w (one wave per weight row; rows-exact); _gemv_rows: the same launcher (the
 * `rows_exact` registration); _matmul: any T, three ik_matmul_bf16 GEMMs (prefill). */
int ik_ink_in_proj_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);
int ik_ink_in_proj_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);
int ik_ink_in_proj_bf16_matmul(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);

/* kernels/hip/ink_mix.hip (contract: crates/intarsia-core/src/inkling2/ops.rs; one block per row, rows-exact). pos0 / pos as the other
 * positioned ops (pos null: the host pos0). ring is updated in place. ConvMix / LoopBlend take the blend's scratch: ik_ink_blend_scratch_bytes. */
int ik_sconv_res_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend, int64_t pos0,
                      const ik_tensor* pos, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, void* stream);
int ik_attn_res_mix_bf16(const ik_tensor* src, const ik_tensor* last, const ik_tensor* addend, const ik_tensor* queries,
                         const ik_tensor* norm, float eps, ik_tensor* out, void* stream);
int ik_conv_mix_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend, const ik_tensor* src,
                     const ik_tensor* queries, const ik_tensor* norm, const ik_tensor* msg, const ik_tensor* down, const ik_tensor* gain,
                     const ik_tensor* forget, int64_t pos0, const ik_tensor* pos, float eps, ik_tensor* y, ik_tensor* ring, ik_tensor* sum,
                     ik_tensor* mix, const ik_scratch* scratch, void* stream);
int ik_loop_blend_bf16(const ik_tensor* dest, const ik_tensor* msg, const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget,
                       const ik_tensor* norm, float eps, ik_tensor* out, const ik_scratch* scratch, void* stream);
/* kernels/hip/ink_route_fused.hip: InkRoute (76). x bf16 [T,D], w bf16 [E,D], bias f32 [E] -> weights f32 [T,K], ids i32 [T,K], logits f32 [T,E] or an
 * empty tensor (then the scratch holds them: T * E * 4 bytes). T <= 16, E <= 256, K <= 32, D % 8 == 0, 16-byte aligned x / w. The logits are
 * bitwise ik_logits_bf16_gemv's, the routing bitwise ik_moe_route's. Same attributes as MoeRoute. */
int ik_ink_route_bf16(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bias, int64_t top_k, float scale, int32_t norm_topk, float temperature,
                      int32_t score_func, ik_tensor* weights, ik_tensor* ids, ik_tensor* logits, const ik_scratch* scratch, void* stream);
/* tests / benchmarks: bit mask of the fast forms of the mix kernels (1 row kernel loads, 2 fused blend launches, 4 gate kernel loads; 0 = the first forms; default 2; same bits) */
void ik_ink_mix_mode(int v);
/* the phase timestamps of the last row-kernel launch (a build with INK_PROBE defined; IK_ERR_UNSUPPORTED otherwise), tests/ink_mix_probe.rs */
int ik_ink_probe(unsigned long long* out, int n);
/* scratch bytes of the blend for s streams, r rows, width d, controller rank g (mirrored by blend_bytes in src/inkmix.rs) */
int64_t ik_ink_blend_scratch_bytes(int64_t s, int64_t r, int64_t d, int64_t g);

#ifdef __cplusplus
}
#endif

#endif
