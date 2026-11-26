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
/* the screened greedy head's HeadScreen and ScreenArgmax (ink_screen.cu; contract intarsia_core::screen) */
int ik_head_screen_f32(const ik_tensor* a, const ik_tensor* h, const ik_tensor* beta, ik_tensor* cand, ik_tensor* cnt, void* stream);
int ik_screen_argmax_f32(const ik_tensor* cand, const ik_tensor* cnt, const ik_tensor* y, ik_tensor* tok, void* stream);
/* LogitsGather over a P12 head (screened greedy head): the serving head kernel in its index mode (inkling2_p12.cu) */
int ik_logits_gather_p12(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* cand, const ik_tensor* cnt,
                         const ik_tensor* aux, ik_tensor* y, void* stream);
int ik_matmul_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_logits_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_logits_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);

/* ink_in_proj over P12 codes of the stacked [Nq+Nk+Nv, D] weight: x [S,T,D] -> pq, pk, pv; == ink_in_proj_bf16_mma[_rows]. */
int ik_ink_in_proj_p12_mma(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                           ik_tensor* pv, void* stream);
int ik_ink_in_proj_p12_mma_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                                ik_tensor* pv, void* stream);
/* InkInProj in the GEMV form (default on, INTARSIA_INK_IN_PROJ_GEMV=0 off): one warp per stacked weight row, 1..=16 rows, rows-exact;
 * the P12 launcher is bitwise the bf16 one. */
int ik_ink_in_proj_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);
int ik_ink_in_proj_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                            ik_tensor* pv, void* stream);
/* the rows-exact registry names of the two (each row's sum order is the same whatever M): the same launchers */
int ik_ink_in_proj_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);
int ik_ink_in_proj_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                                 ik_tensor* pv, void* stream);

#ifdef __cplusplus
}
#endif

#endif
