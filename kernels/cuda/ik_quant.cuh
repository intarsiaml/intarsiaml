/*
 * Device helpers shared by the quantised-weight and MoE kernels (fp8.cu, nvfp4.cu, moe.cu).
 *
 * The fp4 -> e4m3 byte lookup through `prmt` and the "dequantise a weight tile to bf16, then bf16 MMA with
 * exact products" scheme follow the ideas of TensorFold's NVFP4 kernels (MIT, Copyright (c) 2026 TensorFold
 * contributors, see LICENSES/TensorFold-MIT.txt); the code here is written for the torch-free C ABI.
 *
 * Numerics contract (docs/quant-moe.md):
 *   - an e4m3 or e2m1*e4m3 weight value is exact in bf16 and in f32, so every product with a bf16/fp8-valued
 *     activation is exact; the global/tensor scale is applied once, after the f32 accumulation.
 *   - everything accumulates in f32 in a fixed order (deterministic run to run).
 */
#ifndef IK_QUANT_CUH
#define IK_QUANT_CUH

#include "ik_launch.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

/* ---- decode --------------------------------------------------------------------------------------------- */

/* two e4m3 bytes (low byte first) -> two floats, exact */
IK_DEVI float2 ik_e4m3x2_to_f2(unsigned short v) {
    __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
    return __half22float2(*reinterpret_cast<__half2*>(&r));
}

IK_DEVI float ik_e4m3_to_f(uint8_t b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}

/* round-to-nearest-even, saturating at +-448 (NaN -> 0x7F), same as intarsia_core::dtype::f32_to_fp8_e4m3 */
IK_DEVI uint8_t ik_f_to_e4m3(float v) { return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)); }

/* 16 e4m3 bytes -> 16 floats */
IK_DEVI void ik_fp8x16_to_f32(const uint4& u, float* f) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] & 0xFFFFu));
        const float2 b = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] >> 16));
        f[4 * i + 0] = a.x;
        f[4 * i + 1] = a.y;
        f[4 * i + 2] = b.x;
        f[4 * i + 3] = b.y;
    }
}

/* e2m1 magnitudes 0 .5 1 1.5 2 3 4 6 as e4m3 bytes, for `prmt` (selector nibble = 3-bit magnitude code) */
#define IK_FP4_LUT_LO 0x3C383000u
#define IK_FP4_LUT_HI 0x4C484440u

/* the 4 low nibbles of `w` (elements 0..3, element j = nibble j) -> 4 e4m3 bytes (byte j = element j) */
IK_DEVI uint32_t ik_fp4x4_to_e4m3x4(uint32_t sel16) {
    const uint32_t mag = __byte_perm(IK_FP4_LUT_LO, IK_FP4_LUT_HI, sel16 & 0x7777u);
    const uint32_t s = sel16 & 0x8888u;
    const uint32_t sign = ((s & 0x8u) << 4) | ((s & 0x80u) << 8) | ((s & 0x800u) << 12) | ((s & 0x8000u) << 16);
    return mag | sign;
}

/* 8 fp4 nibbles of a 32-bit word (element j = nibble j = byte j/2, low nibble first) -> 8 floats (unscaled) */
IK_DEVI void ik_fp4x8_to_f32(uint32_t w, float* f) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t e = ik_fp4x4_to_e4m3x4(h ? (w >> 16) : (w & 0xFFFFu));
        const float2 a = ik_e4m3x2_to_f2(static_cast<unsigned short>(e & 0xFFFFu));
        const float2 b = ik_e4m3x2_to_f2(static_cast<unsigned short>(e >> 16));
        f[4 * h + 0] = a.x;
        f[4 * h + 1] = a.y;
        f[4 * h + 2] = b.x;
        f[4 * h + 3] = b.y;
    }
}

/* ---- mixed-precision products (lane moe16c) ------------------------------------------------------------------
 * The row-exact kernels at M = 16 were instruction-bound on unpacking: every staged e4m3 activation went e4m3 -> f16 (F2FP, two
 * per instruction) -> f32 (HADD2.F32, one per instruction), and every bf16 activation took a shift per element, once per output
 * tile. `fma.rn.f32.f16` / `fma.rn.f32.bf16` (PTX 8.6, sm_100 and later; FHFMA on sm_121) multiply two f16 / bf16 operands and add
 * an f32, rounding once. An f16 / bf16 value is exact in f32, so the result is bitwise fmaf(float(a), float(b), c): the f32
 * conversions fold into the multiply. (Checked exhaustively on spark1 against fmaf: every e4m3 x e4m3 as f16 and every bf16 x
 * signed e2m1 as bf16, with 16 addends including subnormals, infinities and the largest finite values: 0 of 18.8 M differ.)
 * Elsewhere the fallbacks run the f32 fmaf of the same images. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define IK_MIXED_FMA 1
#endif

/* c = fma(lo(a), lo(b), c), then fma(hi(a), hi(b), c): two f16 pairs, low half (the lower element index) first */
IK_DEVI float ik_fma2_f16(uint32_t a, uint32_t b, float c) {
#ifdef IK_MIXED_FMA
    asm("{\n\t.reg .b16 al, ah, bl, bh;\n\tmov.b32 {al, ah}, %1;\n\tmov.b32 {bl, bh}, %2;\n\t"
        "fma.rn.f32.f16 %0, al, bl, %0;\n\tfma.rn.f32.f16 %0, ah, bh, %0;\n\t}"
        : "+f"(c)
        : "r"(a), "r"(b));
    return c;
#else
    const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
    const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
    return fmaf(fa.y, fb.y, fmaf(fa.x, fb.x, c));
#endif
}

/* the same for two bf16 pairs */
IK_DEVI float ik_fma2_bf16(uint32_t a, uint32_t b, float c) {
#ifdef IK_MIXED_FMA
    asm("{\n\t.reg .b16 al, ah, bl, bh;\n\tmov.b32 {al, ah}, %1;\n\tmov.b32 {bl, bh}, %2;\n\t"
        "fma.rn.f32.bf16 %0, al, bl, %0;\n\tfma.rn.f32.bf16 %0, ah, bh, %0;\n\t}"
        : "+f"(c)
        : "r"(a), "r"(b));
    return c;
#else
    c = fmaf(__uint_as_float(a << 16), __uint_as_float(b << 16), c);
    return fmaf(__uint_as_float(a & 0xFFFF0000u), __uint_as_float(b & 0xFFFF0000u), c);
#endif
}

/* 16 e4m3 bytes -> 8 f16 pairs, exact: h[p] holds elements 2p (low half) and 2p + 1 */
IK_DEVI void ik_fp8x16_to_h2(const uint4& u, uint32_t* h) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __half2_raw a = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] & 0xFFFFu), __NV_E4M3);
        __half2_raw b = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] >> 16), __NV_E4M3);
        h[2 * i] = *reinterpret_cast<uint32_t*>(&a);
        h[2 * i + 1] = *reinterpret_cast<uint32_t*>(&b);
    }
}

/* e2m1 magnitudes 0 .5 1 1.5 2 3 4 6 as bf16: high bytes and low bytes, for `prmt` (selector nibble = 3-bit magnitude code) */
#define IK_FP4_BF_HI_LO 0x3F3F3F00u
#define IK_FP4_BF_HI_HI 0x40404040u
#define IK_FP4_BF_LO_LO 0xC0800000u
#define IK_FP4_BF_LO_HI 0xC0804000u

/* 8 fp4 nibbles of a 32-bit word (element j = nibble j, as ik_fp4x8_to_f32) -> 4 bf16 pairs (unscaled, exact): h[p] holds
 * elements 2p (low half) and 2p + 1 */
IK_DEVI void ik_fp4x8_to_bf2(uint32_t w, uint32_t* h) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const uint32_t sel16 = q ? (w >> 16) : (w & 0xFFFFu);
        const uint32_t s = sel16 & 0x8888u;
        const uint32_t sign = ((s & 0x8u) << 4) | ((s & 0x80u) << 8) | ((s & 0x800u) << 12) | ((s & 0x8000u) << 16);
        const uint32_t hi = __byte_perm(IK_FP4_BF_HI_LO, IK_FP4_BF_HI_HI, sel16 & 0x7777u) | sign;  // byte j: element j's high byte
        const uint32_t lo = __byte_perm(IK_FP4_BF_LO_LO, IK_FP4_BF_LO_HI, sel16 & 0x7777u);         // byte j: its low byte
        h[2 * q] = __byte_perm(lo, hi, 0x5140u);      // elements 0, 1 of this group of 4
        h[2 * q + 1] = __byte_perm(lo, hi, 0x7362u);  // elements 2, 3
    }
}

/* 8 bf16 / 8 half in a uint4 -> floats */
IK_DEVI void ik_half8_to_f32(const uint4& u, float* f) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

/* ---- output ------------------------------------------------------------------------------------------- */

IK_DEVI void ik_store(float* p, float v) { *p = v; }
IK_DEVI void ik_store(ik_bf16* p, float v) { *p = ik_f2b(v); }

/* ---- tensor-core fragments (mma.sync m16n8k16, bf16 inputs, f32 accumulate) ------------------------------ */

IK_DEVI void ik_mma_bf16_16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* ---- dequantise-to-bf16 tiles and async copies (moe_grouped.cu) ------------------------------------------ */

/* 16 fp4 values (8 bytes, element j = nibble j) times a block scale -> 16 bf16 in two 16-byte vectors. Exact: e2m1 *
 * e4m3 has at most 4 significant bits. */
IK_DEVI void ik_fp4x16_scaled_to_bf16(uint2 w, float sc, uint4& lo, uint4& hi) {
    float f[16];
    ik_fp4x8_to_f32(w.x, f);
    ik_fp4x8_to_f32(w.y, f + 8);
#pragma unroll
    for (int e = 0; e < 16; ++e) f[e] *= sc;
    lo = ik_pack8(f);
    hi = ik_pack8(f + 8);
}

/* 16-byte global -> shared copy through L2 (bypasses L1); `src_bytes` < 16 zero-fills the rest (0 = all zero, the
 * source is then not read but must still be a valid address). */
IK_DEVI void ik_cp_async16(void* smem_dst, const void* gmem_src, int src_bytes) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem_src), "r"(src_bytes) : "memory");
}
IK_DEVI void ik_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
IK_DEVI void ik_cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
/* wait until at most N of the most recently committed groups are still in flight */
template <int N>
IK_DEVI void ik_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

/* ldmatrix.x4: four 8x8 b16 matrices from shared memory; lane l supplies the address of row (l & 7) of matrix (l >> 3),
 * result register i holds, for lane l, elements (row l >> 2, columns 2 * (l & 3) + {0, 1}) of matrix i. */
IK_DEVI void ik_ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_row) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a)
                 : "memory");
}

/* ---- validation ----------------------------------------------------------------------------------------- */

/* dense rank-`ndim` tensor of `dtype`, data aligned to `align` bytes */
static inline int ik_check_a(const ik_tensor* t, int dtype, int ndim, uintptr_t align) {
    IK_TRY(ik_check(t, dtype, ndim));
    if (ik_numel(t) > 0 && (reinterpret_cast<uintptr_t>(t->data) & (align - 1)) != 0) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

#endif /* IK_QUANT_CUH */
