// Inkling-2 W4A4 activation quantisers (ink.act_quant = 1; lane ink-cuda-mixmoe). Bit for bit the f32 operations of
// crates/intarsia-core/src/inkling2/actq.rs (contract: crates/intarsia-core/src/inkling2/ops.rs, "ink.act_quant = 1"): every
// product, quotient and sum is an explicit round-to-nearest intrinsic, so nvcc cannot contract them into FMAs, and no -ftz.
//   NVFP4  per row: amax, enc = min(1536 / amax, FLT_MAX), dec = 1 / enc, outer = amax * (1/1536); per block of 16 the 6- and
//          4-based e4m3 scales, codes e2m1(v * inv), the pairwise |(c * s) * outer - v| sums; 4 wins only when strictly
//          smaller. xq = c * s (exact in bf16); the matmul multiplies its f32 sum by outer.
//   MXFP8  per block of 32: the E8M0 power of two of bmax / 448 rounded up, xq = rA(e4m3(clamp(v / 2^e, +-448)) * 2^e).
#ifndef INK_ACTQ_CUH
#define INK_ACTQ_CUH

#include "ik_launch.h"
#include "ik_quant.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

IK_DEVI float aq_e4m3r(float v) { return ik_e4m3_to_f(ik_f_to_e4m3(v)); }

/* e2m1 code (sign in bit 3) of v: nearest, ties to the even code, saturating at 6, sign of a negative nonzero v (crate::quant::nvfp4::f32_to_fp4) */
IK_DEVI uint32_t aq_e2m1(float v) {
    const float a = fabsf(v);
    uint32_t c = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.f ? 6u : 7u;
    if (!(a == a)) c = 0u;
    return c | (v < 0.f ? 8u : 0u);
}

IK_DEVI float aq_e2m1_val(uint32_t c) {
    const float m[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = m[c & 7u];
    return (c & 8u) ? -v : v;
}

/* one candidate block scale s: the 16 codes and the pairwise error sum */
IK_DEVI float aq_nv_candidate(const float* v, float s, float dec, float outer, uint32_t* code) {
    const float inv = fminf(__fdiv_rn(1.f, __fmul_rn(s, dec)), FLT_MAX);
    float e[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        code[j] = aq_e2m1(__fmul_rn(v[j], inv));
        e[j] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(aq_e2m1_val(code[j]), s), outer), v[j]));
    }
#pragma unroll
    for (int w = 8; w >= 1; w >>= 1)
#pragma unroll
        for (int j = 0; j < w; ++j) e[j] = __fadd_rn(e[2 * j], e[2 * j + 1]);
    return e[0];
}

/* one NVFP4 block of 16 values: xq (exact), and its e4m3 scale byte; the 16 codes in `code` */
IK_DEVI float aq_nv_block(const float* v, float enc, float dec, float outer, uint32_t* code) {
    float bmax = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) bmax = fmaxf(bmax, fabsf(v[j]));
    const float s6 = aq_e4m3r(fminf(__fmul_rn(bmax, __fmul_rn(enc, 1.f / 6.f)), 448.f));
    const float s4 = aq_e4m3r(fminf(__fmul_rn(bmax, __fmul_rn(enc, 0.25f)), 448.f));
    uint32_t c6[16], c4[16];
    const float e6 = aq_nv_candidate(v, s6, dec, outer, c6);
    const float e4 = aq_nv_candidate(v, s4, dec, outer, c4);
    const bool four = e4 < e6;
#pragma unroll
    for (int j = 0; j < 16; ++j) code[j] = four ? c4[j] : c6[j];
    return four ? s4 : s6;
}

/* the row scales of an NVFP4 row with max |v| = amax */
IK_DEVI void aq_nv_row_scales(float amax, float* enc, float* dec, float* outer) {
    *enc = fminf(__fdiv_rn(1536.f, amax), FLT_MAX);
    *dec = __fdiv_rn(1.f, *enc);
    *outer = __fmul_rn(amax, 1.f / 1536.f);
}

/* one MXFP8 block of 32 values -> xq (as f32, the caller rounds to bf16) */
IK_DEVI void aq_mx_block(const float* v, float* q) {
    float bmax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) bmax = fmaxf(bmax, fabsf(v[j]));
    const float min_scale = __uint_as_float(0x00400000u);
    const float raw = __fmul_rn(bmax, 1.f / 448.f);
    uint32_t e = ((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23;
    if (raw <= min_scale) e = 0u;
    e = e < 254u ? e : 254u;
    const float scale = fmaxf(__uint_as_float(e << 23), min_scale);
    const float inv = e == 0u ? __uint_as_float(254u << 23) : __fdiv_rn(1.f, scale);
#pragma unroll
    for (int j = 0; j < 32; ++j) q[j] = __fmul_rn(aq_e4m3r(fminf(fmaxf(__fmul_rn(v[j], inv), -448.f), 448.f)), scale);
}

/* ink_actq.cu launchers. rows of C bf16 values (row stride C), in place allowed (xq == x).
 * NVFP4: C % 16 == 0, outer [rows] f32; codes [rows, C/2] / scales [rows, C/16] (e4m3) optional (null).
 * MXFP8: C % 32 == 0. */
int ink_actq_nv(const ik_bf16* x, int64_t rows, int C, ik_bf16* xq, float* outer, uint8_t* codes, uint8_t* scales, cudaStream_t s);
int ink_actq_mx(const ik_bf16* x, int64_t rows, int C, ik_bf16* xq, cudaStream_t s);

#endif
