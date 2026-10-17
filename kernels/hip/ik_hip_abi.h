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

#ifdef __cplusplus
}
#endif

#endif /* IK_HIP_ABI_H */
