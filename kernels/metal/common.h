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

// ---------------------------------------------------------------------------------------------------------------------
// OCP FP8 E4M3 ("fn": no infinity, 0x7F / 0xFF NaN, max 448), the KV cache element of `--kv-dtype fp8_e4m3`. Bit-exact
// counterparts of `fp8_e4m3_to_f32` / `f32_to_fp8_e4m3` in crates/intarsia-core/src/dtype.rs (the cpu oracle), checked
// exhaustively by tests/kv_fp8.rs. Every e4m3 value is exactly a bf16 (and f32) value, so a kernel that decodes fp8 to f32
// and then runs the bf16 kernel's arithmetic gives the same bits as the bf16 kernel on the decoded cache.
// ---------------------------------------------------------------------------------------------------------------------

// Decode through half: the magnitude bits of an e4m3 byte placed under the half exponent field (mag << 7) are the half
// whose value is the e4m3 value times 2^-8 (e4m3 bias 7, half bias 15; an e4m3 subnormal is a half subnormal with the same
// scaling), the sign moves from bit 7 to bit 15. The half -> f32 conversion is exact for subnormal halves too. An f32
// multiply on the f32 pattern of a subnormal e4m3 value would not do: the GPU flushes f32 denormal operands (measured,
// tests/kv_fp8.rs). Four bytes are handled as two half2 per 32-bit word with 5 integer ops each.
//
// `unpack4_e4m3_s8` returns value * 2^-8 (the attention kernels fold the 2^8 into q and into the final accumulator, both
// exact power-of-two scalings, so the products and sums are bit for bit those of the true values); `e4m3_to_f32` and
// `unpack8_e4m3` return the true values.
inline void unpack4_e4m3_s8(uint w, thread float* f) {
    const uint a = ((w << 7) & 0x3f803f80u) | ((w << 8) & 0x80008000u);  // bytes 0 and 2 as half2
    const uint w1 = w >> 8;
    const uint b = ((w1 << 7) & 0x3f803f80u) | ((w1 << 8) & 0x80008000u);  // bytes 1 and 3
    const half2 ha = as_type<half2>(a), hb = as_type<half2>(b);
    f[0] = float(ha.x);
    f[1] = float(hb.x);
    f[2] = float(ha.y);
    f[3] = float(hb.y);
}

inline float e4m3_to_f32(uint b) {
    return float(as_type<half>((ushort)(((b & 0x7fu) << 7) | ((b & 0x80u) << 8)))) * 256.0f;
}

// Encode f32 -> e4m3: round to nearest even, saturating at +-448 (NaN -> 0x7F), sign of zero kept.
inline uint f32_to_e4m3(float x) {
    const uint bits = as_type<uint>(x);
    const uint sign = (bits >> 24) & 0x80u;
    const uint a = bits & 0x7fffffffu;
    if (a > 0x7f800000u) return 0x7fu;  // NaN
    const float af = as_type<float>(a);
    // below 2^-6 the code is the subnormal mantissa: a * 2^9 rounded to even (exact scaling), 8 is the smallest normal
    if (a < 0x3c800000u) return sign | (uint)rint(af * 512.0f);
    if (a >= 0x43e00000u) return sign | 0x7eu;  // >= 448 saturates
    // normal: drop 20 mantissa bits with round to nearest even (a carry moves into the exponent, as it must), rebias
    // the exponent from 127 to 7
    const uint r = a + 0x7ffffu + ((a >> 20) & 1u);
    return sign | ((r >> 20) - (120u << 3));
}

// 8 e4m3 bytes (two registers) to 8 floats holding value * 2^-8, element i of byte i (little endian)
inline void unpack8_e4m3_s8(uint2 w, thread float* f) {
    unpack4_e4m3_s8(w.x, f);
    unpack4_e4m3_s8(w.y, f + 4);
}

// the same, true values
inline void unpack8_e4m3(uint2 w, thread float* f) {
    unpack8_e4m3_s8(w, f);
    for (uint i = 0; i < 8; ++i) f[i] *= 256.0f;
}

// ---------------------------------------------------------------------------------------------------------------------
// int8 KV cache quantiser (block scales), shared by every append kernel: a block of KVQ_B consecutive elements of one head
// row holds q = clamp(rint(x / s), -127, 127) with s = bf16(absmax / 127) (the bf16-ROUNDED scale is the one used to
// quantise and to dequantise). The block is KVQ_B lanes of a simdgroup (element i of the head row on lane i % 32).
// ---------------------------------------------------------------------------------------------------------------------
constant constexpr uint KVQ_B = 32;

inline float kvq_scale(float x, uint blk, uint lane) {
    float a = fabs(x);
    for (uint m = 1; m < blk; m <<= 1) a = max(a, simd_shuffle_xor(a, (ushort)m));
    return bf2f(f2bf(a / 127.0f));
}
inline float kvq_q(float x, float s) { return s == 0.0f ? 0.0f : clamp(rint(x / s), -127.0f, 127.0f); }
