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
