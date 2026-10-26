/* HIP device helpers shared by the quantised-weight and MoE kernels (fp8.hip, nvfp4.hip, moe.hip).
 * Counterpart of kernels/cuda/ik_quant.cuh; the numerics contract is the same (docs/quant-moe.md). The "dequantise the
 * weight to bf16, bf16 MMA with exact products" scheme and the fp4 -> e4m3 lookup idea follow TensorFold's NVFP4
 * kernels (MIT, Copyright (c) 2026 TensorFold contributors, LICENSES/TensorFold-MIT.txt); the code here is
 * written for the torch-free C ABI and RDNA4:
 *   - an e4m3 or e2m1*e4m3 weight value is exact in bf16 and in f32, so every product with a bf16/fp8-valued
 *     activation is exact; the global/tensor scale is applied once, after the f32 accumulation;
 *   - everything accumulates in f32 in a fixed order (deterministic run to run).
 *
 * RDNA4 has no fp4 conversion instructions (fp8 ones it has: ik_e4m3x8_to_bf16), so both formats are decoded in registers with integer/float
 * bit arithmetic (e4m3) or a 16-entry shared-memory table (e2m1). The e4m3 decode is OCP e4m3fn (not fnuz).
 */
#ifndef IK_HIP_QUANT_H
#define IK_HIP_QUANT_H

#include "ik_hip.h"

/* ---- e4m3 (OCP fn: bias 7, no infinities, 0x7f / 0xff = NaN) ---------------------------------------------- */

/* e4m3 byte -> float, exact. Normals: shift the 7 magnitude bits into the f32 exponent / top mantissa and rebias
 * by 2^120 in the exponent field; subnormals (exp field 0) are m * 2^-9, built from an integer so the result
 * does not depend on the f32 denormal mode. */
IK_DEVI float ik_e4m3_to_f(uint32_t b) {
    const uint32_t mag = b & 0x7fu;
    const float normal = __uint_as_float((mag << 20) + 0x3C000000u);
    const float sub = (float)(mag & 7u) * (1.f / 512.f);
    float v = mag >= 8u ? normal : sub;
    v = mag == 0x7fu ? __uint_as_float(0x7fc00000u) : v;
    return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}

/* round-to-nearest-even, saturating at +-448 (NaN -> 0x7F), same as intarsia_core::dtype::f32_to_fp8_e4m3 */
IK_DEVI uint8_t ik_f_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint8_t sign = (u >> 24) & 0x80u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fu;  // NaN
    const float a = fabsf(f);
    if (a >= 448.f) return sign | 0x7eu;  // saturate (also covers inf and the round-up past 448)
    if (a < 0.015625f) {                  // below the smallest normal 2^-6: subnormals are multiples of 2^-9
        const uint32_t q = (uint32_t)__float2int_rn(a * 512.f);  // 0..8; 8 is the smallest normal (0x08)
        return sign | (uint8_t)q;
    }
    const uint32_t ua = __float_as_uint(a);
    int e = (int)(ua >> 23) - 127;  // -6 .. 8
    uint32_t m = (ua & 0x7fffffu) >> 20;
    const uint32_t rem = ua & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    if (m == 8u) {
        m = 0;
        ++e;
    }
    if (e > 8 || (e == 8 && m == 7u)) return sign | 0x7eu;
    return sign | (uint8_t)(((e + 7) << 3) | (int)m);
}

/* 16 e4m3 bytes -> 16 floats */
IK_DEVI void ik_fp8x16_to_f32(const uint4& u, float* f) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) f[4 * i + j] = ik_e4m3_to_f((w[i] >> (8 * j)) & 0xffu);
    }
}

/* 8 e4m3 bytes (two words, element 0 in the low byte of `lo`) -> 8 bf16 (one uint4, element 0 in the low half of .x).
 * gfx12 does have fp8 -> f32 conversion instructions (v_cvt_pk_f32_fp8, OCP e4m3): every e4m3 value is exact in f32 and in
 * bf16, so the bf16 is the top half of the converted f32 (one byte permute per pair): ~2 ALU ops per pair instead of ~25 for
 * ik_e4m3_to_f + ik_pack8. Bitwise that pair for every non-NaN code (tests/fp8_mma_v2.rs runs all 254 of them). The software
 * decode is the fallback of a compiler without the builtin. */
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
typedef float ik_q_v2f __attribute__((ext_vector_type(2)));
IK_DEVI uint32_t ik_e4m3x2_to_bf16x2(uint32_t w, bool hi_word) {
    const ik_q_v2f f = hi_word ? __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true) : __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false);
    return __builtin_amdgcn_perm(__float_as_uint(f[1]), __float_as_uint(f[0]), 0x07060302u);
}
IK_DEVI uint4 ik_e4m3x8_to_bf16(uint32_t lo, uint32_t hi) {
    return make_uint4(ik_e4m3x2_to_bf16x2(lo, false), ik_e4m3x2_to_bf16x2(lo, true), ik_e4m3x2_to_bf16x2(hi, false),
                      ik_e4m3x2_to_bf16x2(hi, true));
}
#else
IK_DEVI uint4 ik_e4m3x8_to_bf16(uint32_t lo, uint32_t hi) {
    float f[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        f[j] = ik_e4m3_to_f((lo >> (8 * j)) & 0xffu);
        f[4 + j] = ik_e4m3_to_f((hi >> (8 * j)) & 0xffu);
    }
    return ik_pack8(f);
}
#endif

/* ---- e2m1 (fp4): 0 .5 1 1.5 2 3 4 6, sign in bit 3 ------------------------------------------------------- */

IK_DEVI float ik_fp4_value(uint32_t c) {
    const uint32_t m = c & 7u;
    // m >= 2: exponent (m >> 1) + 126, mantissa top bit (m & 1) == (m << 22) + (126 << 23); m = 0 -> 0, m = 1 -> 0.5
    float v = __uint_as_float(0x3F000000u + (m << 22));
    v = m == 1u ? 0.5f : v;
    v = m == 0u ? 0.f : v;
    return __uint_as_float(__float_as_uint(v) | ((c & 8u) << 28));
}

/* One 16-entry table in shared memory (a conflict-free lookup: every entry sits in its own bank). Declare and fill
 * it before any early return; it contains a block barrier (IK_LDS_BARRIER), so every thread of the block must reach it. */
#define IK_FP4_LUT(name)                                               \
    __shared__ float name[16];                                         \
    if (threadIdx.x < 16) name[threadIdx.x] = ik_fp4_value(threadIdx.x); \
    IK_LDS_BARRIER()

/* 8 fp4 nibbles of a 32-bit word (element j = nibble j = byte j/2, low nibble first) -> 8 floats (unscaled) */
IK_DEVI void ik_fp4x8_to_f32(uint32_t w, const float* lut, float* f) {
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = lut[(w >> (4 * j)) & 15u];
}

/* ---- loads / stores -------------------------------------------------------------------------------------- */

typedef unsigned int ik_u32x2 __attribute__((ext_vector_type(2)));
/* streaming 8 B load of weights read once */
IK_DEVI uint2 ik_ldcs(const uint2* p) {
    const ik_u32x2 v = __builtin_nontemporal_load(reinterpret_cast<const ik_u32x2*>(p));
    return make_uint2(v[0], v[1]);
}

IK_DEVI void ik_store(float* p, float v) { *p = v; }
IK_DEVI void ik_store(ik_bf16* p, float v) { *p = ik_f2b(v); }

IK_DEVI float ik_round_bf16(float v) { return ik_b2f(ik_f2b(v)); }

/* ---- validation ------------------------------------------------------------------------------------------ */

/* dense rank-`ndim` tensor of `dtype`, data aligned to `align` bytes */
static inline int ik_check_a(const ik_tensor* t, int dtype, int ndim, uintptr_t align) {
    IK_TRY(ik_check(t, dtype, ndim));
    if (ik_numel(t) > 0 && (reinterpret_cast<uintptr_t>(t->data) & (align - 1)) != 0) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

#endif /* IK_HIP_QUANT_H */
