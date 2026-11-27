/* Int8 weights with one f32 scale per output row on HIP (lane ink-hip-i8; kernels/hip/ink_i8.hip, crates/intarsia-backend-hip/src/inki8.rs).
 * Format: crates/intarsia-core/src/quant/i8row.rs: q i8 [N, K], scales f32 [N, 1]; out[m, n] = rA(scales[n] * sum_k x[m, k] * q[n, k]).
 * W8A16, lossy, opt-in. Rows-exact (row m of an M-row call is the one-row call's, up to a signed zero). K % 16 == 0, x bf16 and 16-byte
 * aligned, q 16-byte aligned, scales 4-byte aligned. */
#ifndef IK_HIP_ABI_INKI8_H
#define IK_HIP_ABI_INKI8_H
#include "../abi/ik_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
/* matmul_i8: x [M,K] (M 1..16) -> out [M,N] bf16 or f32 (the output's dtype). The four names are one launcher (the registrations
 * differ: ordinary calls of 1..4 rows, the rows-exact registrations of 1..16 rows). */
int ik_matmul_i8_gemv(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream);
int ik_matmul_i8_gemv_rows(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream);
int ik_matmul_i8_logits_gemv(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream);
int ik_matmul_i8_logits_gemv_rows(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream);
/* ink_in_proj over i8 codes of the stacked [Nq+Nk+Nv, D] weight (scales [Nq+Nk+Nv, 1]): x [S,T,D] (T 1..16) -> pq, pk, pv bf16. */
int ik_ink_in_proj_i8_gemv(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv,
                           void* stream);
int ik_ink_in_proj_i8_gemv_rows(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* pq, ik_tensor* pk,
                                ik_tensor* pv, void* stream);
/* Test / bench hook (not a launcher): the launch shape. gemv_r / gemv_u: rows per wave and chunk loads in flight per lane of the
 * MatmulI8 kernel (M 1..4: R 1..2, U 2/4/8; M 5..16: R 1..2, U 1/2; anything else, or < 0, the default: 1,8 at one row, 1,4 at 2..4,
 * 2,2 at 5..16); inproj_u: 1..5 chunk loads in flight per lane of the in-projection kernel, <= 0 the default (5 at 1..4 rows, else 2). */
void ik_hip_i8_set(int gemv_r, int gemv_u, int inproj_u);
#ifdef __cplusplus
}
#endif
#endif
