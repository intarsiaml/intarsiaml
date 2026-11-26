/* Lossless 12-bit bf16 weights (P12) on HIP (lane ink-hip-p12; kernels/hip/ink_p12.hip, crates/intarsia-backend-hip/src/inkp12.rs).
 * Format: crates/intarsia-core/src/quant/p12.rs: codes u8 [N, 3K/2], aux (u32 held as i32) [N + 1 + E]. Every launcher decodes each
 * weight to its exact bf16 bits in registers and runs its HIP bf16 twin's arithmetic in that twin's order (lane L sums chunks
 * L, L + 32, ... ascending, eight fmas in element order, then ik_warp_sum), so its output is the twin's on the decoded weight, bit
 * for bit; rows-exact (row m of an M-row call is bitwise the one-row call). K % 16 == 0, x bf16 and 16-byte aligned, codes
 * 16-byte aligned. */
#ifndef IK_HIP_ABI_INKP12_H
#define IK_HIP_ABI_INKP12_H
#include "../abi/ik_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
/* matmul_p12: x [M,K] (M 1..16) -> out [M,N] bf16 or f32 (the output's dtype); == matmul_bf16_gemv[_rows] (bf16) and
 * logits_bf16_gemv[_rows] (f32) of matmul.hip on the decoded weight. The four names are one launcher (the registrations differ). */
int ik_matmul_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_logits_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
int ik_matmul_p12_logits_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream);
/* ink_in_proj over P12 codes of the stacked [Nq+Nk+Nv, D] weight: x [S,T,D] (T 1..16) -> pq, pk, pv; == ink_in_proj_bf16_gemv
 * (ink_proj.hip). The _rows name is the same launcher (rows-exact registration). */
int ik_ink_in_proj_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                            ik_tensor* pv, void* stream);
int ik_ink_in_proj_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                                 ik_tensor* pv, void* stream);
/* Test / bench hook (not a launcher): the launch shape of the P12 GEMVs, every value the same bits. gemv_r / gemv_u: rows per
 * wave and chunk loads in flight per lane of the 1..4-row MatmulP12 kernel (instantiated set in ink_p12.hip; anything else, or
 * < 0, the default 1,8 at one row and 2,4 above, the bf16 twins' shapes); inproj_u: 0 = the bf16 twin's stride loop (default),
 * 4 / 8 / 9 = that many chunk loads in flight per lane; < 0 the default. */
void ik_hip_p12_set(int gemv_r, int gemv_u, int inproj_u);
#ifdef __cplusplus
}
#endif
#endif
