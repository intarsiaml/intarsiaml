// fp8 e4m3 KV cache helpers (`--kv-dtype fp8_e4m3`): the bf16 staging pass of the fp8 prefill attention, and the two
// conversion test kernels of tests/kv_fp8.rs (exhaustive checks of common.h's e4m3_to_f32 / f32_to_e4m3 against the cpu
// oracle). The conversions themselves are in common.h so every kernel shares them.

#include "common.h"

inline void unpack8_i8x(uint2 w, thread float* f) {
    const char4 a = as_type<char4>(w.x), b = as_type<char4>(w.y);
    f[0] = float(a.x); f[1] = float(a.y); f[2] = float(a.z); f[3] = float(a.w);
    f[4] = float(b.x); f[5] = float(b.y); f[6] = float(b.z); f[7] = float(b.w);
}

// Decode the first `kv_len` rows of every kv head of an fp8 K and V cache into bf16 caches of the same [Nkv, cap, D]
// layout (rows past kv_len are not written and not read by the flash kernels). Exact: every e4m3 value is a bf16 value.
// bind: 0 k cache (e4m3), 1 v cache, 2 k out (bf16), 3 v out, 4 OpP. a = Nkv, b = cap, c = D, d = kv_len.
// grid: x = row * (D / 8) + chunk of 8 elements (kv_len * D / 8), y = kv head, z = 0 (K) or 1 (V).
kernel void kv_dequant_f8(device const uchar* ks [[buffer(0)]],
                          device const uchar* vs [[buffer(1)]],
                          device ushort* kd [[buffer(2)]],
                          device ushort* vd [[buffer(3)]],
                          constant OpP& p [[buffer(4)]],
                          uint3 gid [[thread_position_in_grid]]) {
    const uint chunks = p.d * (p.c / 8);
    if (gid.x >= chunks || gid.y >= p.a) return;
    const ulong off = ((ulong)gid.y * p.b) * p.c + (ulong)gid.x * 8;
    device const uchar* src = gid.z == 0 ? ks : vs;
    device ushort* dst = gid.z == 0 ? kd : vd;
    const uint2 w = *(device const uint2*)(src + off);
    float f[8];
    unpack8_e4m3(w, f);
    uint4 r;
    r.x = (uint)f2bf(f[0]) | ((uint)f2bf(f[1]) << 16);
    r.y = (uint)f2bf(f[2]) | ((uint)f2bf(f[3]) << 16);
    r.z = (uint)f2bf(f[4]) | ((uint)f2bf(f[5]) << 16);
    r.w = (uint)f2bf(f[6]) | ((uint)f2bf(f[7]) << 16);
    *(device uint4*)(dst + off) = r;
}

// out[i] = e4m3_to_f32(in[i]) for i in 0..256
kernel void fp8_test_decode(device const uchar* in [[buffer(0)]], device float* out [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = e4m3_to_f32(in[i]);
}

// out[i] = f32_to_e4m3(in[i]); the host passes bf16 values widened to f32 and raw f32 bit patterns
kernel void fp8_test_encode(device const float* in [[buffer(0)]], device uchar* out [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    out[i] = (uchar)f32_to_e4m3(in[i]);
}

// Decode the first `kv_len` rows of every kv head of an int8 block-scaled K and V cache into bf16 caches [Nkv, cap, D] (the
// staging copy of the int8 prefill attention): bf16(q * s). Same arguments / grid as kv_dequant_f8; the scales are read from
// the area after each plane's data.
kernel void kv_dequant_i8(device const uchar* ks [[buffer(0)]],
                          device const uchar* vs [[buffer(1)]],
                          device ushort* kd [[buffer(2)]],
                          device ushort* vd [[buffer(3)]],
                          constant OpP& p [[buffer(4)]],
                          uint3 gid [[thread_position_in_grid]]) {
    const uint chunks = p.d * (p.c / 8);
    if (gid.x >= chunks || gid.y >= p.a) return;
    const ulong row = (ulong)gid.y * p.b + gid.x / (p.c / 8);
    const ulong off = row * p.c + (ulong)(gid.x % (p.c / 8)) * 8;
    device const uchar* src = gid.z == 0 ? ks : vs;
    device ushort* dst = gid.z == 0 ? kd : vd;
    const float sc = bf2f(((device const ushort*)(src + (ulong)p.a * p.b * p.c))[row * (p.c / KVQ_B) + (gid.x % (p.c / 8)) / 4]);
    const uint2 w = *(device const uint2*)(src + off);
    float f[8];
    unpack8_i8x(w, f);
    uint4 r;
    r.x = (uint)f2bf(f[0] * sc) | ((uint)f2bf(f[1] * sc) << 16);
    r.y = (uint)f2bf(f[2] * sc) | ((uint)f2bf(f[3] * sc) << 16);
    r.z = (uint)f2bf(f[4] * sc) | ((uint)f2bf(f[5] * sc) << 16);
    r.w = (uint)f2bf(f[6] * sc) | ((uint)f2bf(f[7] * sc) << 16);
    *(device uint4*)(dst + off) = r;
}
