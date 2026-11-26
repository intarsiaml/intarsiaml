/* The DSpark draft's launchers on HIP that the Inkling-2 lanes had not ported (lane ink-hip-spec; crates/intarsia-backend-hip/src/inkspec.rs). */
#ifndef IK_HIP_ABI_INKSPEC_H
#define IK_HIP_ABI_INKSPEC_H
#include "../abi/ik_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
/* hc_pre_bf16 (kernels/hip/ink_spec.hip, Op::HcPre): x [T,S,D] f32 or bf16, mix f32 [T,S] -> y [T,D] f32 or bf16,
 * y[t,d] = sum_c mix[t,c] x[t,c,d] (fmaf from 0 in c order, as CUDA's ik_hc_pre in kernels/cuda/dsv41_hc.cu), 1 <= S <= 8. */
int ik_hc_pre_bf16(const ik_tensor* x, const ik_tensor* mix, ik_tensor* y, void* stream);
#ifdef __cplusplus
}
#endif
#endif
