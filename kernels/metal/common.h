// Shared by the dense-op shaders (dense_ops.metal, gemm_bf16.metal, gemv_f32out.metal). Not compiled on its own.
//
// Layout contract (crates/intarsia-backend-metal/src/dense.rs): every bf16 / f32 / i32 pointer is bound at a byte
// offset rounded down to 4 bytes; the 0 or 1 leftover bf16 element of a 2-byte-aligned tensor arrives as a `s*` skip
// (in elements) that the kernel adds to the pointer. f32 / i32 tensors must be 4-byte aligned (skip is 0).

#pragma once
#include <metal_stdlib>
using namespace metal;

// Kernel arguments, mirrored by `OpP` in dense.rs (64 bytes). Field use per kernel is documented at each kernel.
struct OpP {
    uint a, b, c, d, e, f, g, h;  // dimensions / flags
    uint s0, s1, s2, s3, s4;      // leading elements to skip per bound tensor, in binding order
    float f0, f1;                 // eps / w_offset / theta
    uint pad;
};

inline float bf2f(ushort v) { return as_type<float>((uint)v << 16); }
inline ushort f2bf(float v) { return as_type<ushort>(bfloat(v)); }  // round to nearest even

// output element stores, so a template can write bf16 (ushort) or f32
inline void store_out(device ushort* y, ulong i, float v) { y[i] = f2bf(v); }
inline void store_out(device float* y, ulong i, float v) { y[i] = v; }

// ---- e4m3 (shared by fp8.metal, whose GEMVs quantise the activations, and the producers that write q(x) for them) -------
// two e4m3 bytes at bits 0..7 and 16..23 of t (the other bits are ignored) -> value / 256
inline float2 fp8_x2(uint t) {
    const uint h = ((t & 0x007F007Fu) << 7) | ((t & 0x00800080u) << 8);
    return float2(as_type<half2>(h));
}

// exact value of an e4m3 code
inline float fp8_dec(uint b) {
    if ((b & 0x7Fu) == 0x7Fu) return as_type<float>(0x7FC00000u);
    return fp8_x2(b).x * 256.0f;
}

// intarsia_core::dtype::f32_to_fp8_e4m3: round to nearest, ties to the even code, saturating at +-448, NaN -> 0x7F
inline uint fp8_enc(float v) {
    const uint s = (as_type<uint>(v) >> 24) & 0x80u;
    const float a = fabs(v);
    if (!(a == a)) return 0x7Fu;
    if (a >= 448.0f) return s | 0x7Eu;
    uint code;
    if (a < 0.015625f) {
        code = (uint)rint(a * 512.0f);  // subnormal grid 2^-9, rint = ties to even
    } else {
        uint b = as_type<uint>(a);  // 3 mantissa bits kept: round to nearest even on bit 20
        b += 0x7FFFFu + ((b >> 20) & 1u);
        code = (b >> 20) - 960u;  // re-bias the exponent: (127 - 7) << 3
    }
    return s | code;
}

// q(x) for one element: static W8A8 (is > 0) or the activation itself
inline float fp8_qx(float x, float is) {
    return is > 0.0f ? fp8_dec(fp8_enc(precise::divide(x, is))) : x;
}

// q(x) of one bf16 activation as the bf16 the fp8 GEMV stages (lane mnq: the producers write exactly this): static W8A8 (is > 0)
// quantises the stored bf16 value, otherwise the value is its own q(x)
inline ushort fp8_q_bf16(ushort v, float is) { return is > 0.0f ? f2bf(fp8_qx(bf2f(v), is)) : v; }
