/* Lossless 12-bit bf16 weights (P12) on CUDA (lane ink-cuda-p12; kernels/cuda/inkling2_p12.cu). Format:
 * crates/intarsia-core/src/quant/p12.rs: codes u8 [N, 3K/2], aux (u32 held as i32) [N + 1 + E]. Every launcher decodes each
 * weight to its exact bf16 bits in registers and runs its bf16 twin's arithmetic in that twin's order, so its output is the
 * twin's on the decoded weight, bit for bit; rows-exact (row m of an M-row call is bitwise the one-row call). K % 16 == 0,
 * x bf16 and 16-byte aligned, codes 16-byte aligned. */
#ifndef IK_CUDA_ABI_INKP12_H
#define IK_CUDA_ABI_INKP12_H

#include "abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* matmul_p12: x [M,K] (M 1..16) -> out [M,N] bf16 or f32 (the output's dtype); == matmul_bf16_gemv[_rows] (bf16) and
 * logits_bf16_gemv[_rows] (f32) on the decoded weight. The four names are one launcher (the registrations differ). */
int ik_matmul_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_logits_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_logits_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);

/* ink_in_proj over P12 codes of the stacked [Nq+Nk+Nv, D] weight: x [S,T,D] -> pq, pk, pv; == ink_in_proj_bf16_mma[_rows]. */
int ik_ink_in_proj_p12_mma(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                           ik_tensor* pv, void* stream);
int ik_ink_in_proj_p12_mma_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                                ik_tensor* pv, void* stream);
/* InkInProj in the GEMV form (opt-in INTARSIA_INK_IN_PROJ_GEMV=1): one warp per stacked weight row, 1..=16 rows, rows-exact;
 * the P12 launcher is bitwise the bf16 one. */
int ik_ink_in_proj_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);
int ik_ink_in_proj_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                            ik_tensor* pv, void* stream);

#ifdef __cplusplus
}
#endif

#endif
