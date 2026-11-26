/* The Inkling-2-Mini router launcher on HIP (lane ink-hip; crates/intarsia-backend-hip/src/inkroute.rs). */
#ifndef IK_HIP_ABI_INKROUTE_H
#define IK_HIP_ABI_INKROUTE_H

#include "../abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* moe_route (kernels/hip/ink_route.hip): logits f32 [T,E], bias f32 [E] -> weights f32 [T,K], ids i32 [T,K]. score_func 0 softmax,
 * 1 sigmoid, 2 sqrt-softplus; the top K of score + bias (ties to the lower id), weights = the scores of the picks, divided by
 * their sum with norm_topk, times scale. E <= 1024, K <= 32. Same contract as the CUDA launcher (ik_cuda_abi_dsv41.h). */
int ik_moe_route(const ik_tensor* logits, const ik_tensor* bias, int64_t top_k, float scale, int32_t norm_topk, float temperature,
                 int32_t score_func, ik_tensor* weights, ik_tensor* ids, void* stream);
/* tests: moe_route's fast tail (lane ink-hip-dec2): 1 on, 0 lane 0's loop, -1 = INTARSIA_HIP_INK_DEC_FAST bit 4 / the default (on); same bits */
void ik_moe_route_set_dec(int v);

#ifdef __cplusplus
}
#endif

#endif
