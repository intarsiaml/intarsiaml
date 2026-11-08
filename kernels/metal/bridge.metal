// DSpine draft's code-bridge row ops on Metal (semantics: crates/intarsia-core/src/ops.rs, the cpu oracle
// crates/intarsia-backend-cpu/src/bridge_ref.rs; docs/plans/2026-11-07-dspine-port.md section 5).
//
// All four are one threadgroup per block row (rows <= 16), bf16 storage, f32 math with the oracle's rounding points. Reductions
// have a fixed order (thread-strided partial sums, simd_sum, serial sum over the simdgroup partials): run-to-run identical.
//
// Kernel arguments (`OpP`): a = row width (D / C / R / H), b = rows, f0 = eps, f1 = scale. s0.. = leading elements skip per
// bound tensor, in binding order.
#include "common.h"

constant float FLT_TINY = 1.1754943508222875e-38f;

inline float br_tg_sum(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint j = 0; j < nsg; ++j) t += part[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// ---- BridgeNorm: rms = sqrt(max(mean(x^2), FLT_MIN)); y = bf16(x / (rms + eps)) --------------------------------------------
// bind: 0 x [T,D] bf16, 1 y [T,D] bf16, 2 rms [T] f32. a = D, f0 = eps.
kernel void bridge_norm_bf16(device const ushort* x [[buffer(0)]],
                             device ushort* y [[buffer(1)]],
                             device float* rms [[buffer(2)]],
                             constant OpP& p [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tgs [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint d = p.a;
    device const ushort* xr = x + p.s0 + (ulong)row * d;
    device ushort* yr = y + p.s1 + (ulong)row * d;
    float ss = 0.0f;
    for (uint i = tid; i < d; i += tgs) {
        const float v = bf2f(xr[i]);
        ss = fma(v, v, ss);
    }
    const float tot = br_tg_sum(ss, part, lane, sg, nsg);
    const float r = precise::sqrt(max(tot / (float)d, FLT_TINY));
    for (uint i = tid; i < d; i += tgs) yr[i] = f2bf(bf2f(xr[i]) / (r + p.f0));
    if (tid == 0) rms[p.s2 + row] = r;
}

// ---- BridgeCode: y = bf16(scale * x / (sqrt(sum x^2) + eps)) ------------------------------------------------------------
// bind: 0 x [T,C] bf16, 1 y [T,C] bf16. a = C, f0 = eps, f1 = scale.
kernel void bridge_code_bf16(device const ushort* x [[buffer(0)]],
                             device ushort* y [[buffer(1)]],
                             constant OpP& p [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tgs [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint d = p.a;
    device const ushort* xr = x + p.s0 + (ulong)row * d;
    device ushort* yr = y + p.s1 + (ulong)row * d;
    float ss = 0.0f;
    for (uint i = tid; i < d; i += tgs) {
        const float v = bf2f(xr[i]);
        ss = fma(v, v, ss);
    }
    const float tot = br_tg_sum(ss, part, lane, sg, nsg);
    const float n = precise::sqrt(tot);
    for (uint i = tid; i < d; i += tgs) yr[i] = f2bf((p.f1 * bf2f(xr[i])) / (n + p.f0));
}

// ---- BridgeGate: msg[0] = 0; msg[t] = bf16(sigmoid((q[t] + s[t-1]) + bias) * v[t-1]) --------------------------------------
// packed rows are [q | s | v] (3R); the sender of row 1 is s0 = [s | v] (2R). bind: 0 packed [T,3R], 1 bias [R], 2 s0 [2R],
// 3 msg [T,R], all bf16. a = R. Grid: one thread per (row, column) of msg.
kernel void bridge_gate_bf16(device const ushort* packed [[buffer(0)]],
                             device const ushort* bias [[buffer(1)]],
                             device const ushort* s0 [[buffer(2)]],
                             device ushort* msg [[buffer(3)]],
                             constant OpP& p [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    const uint r = p.a;
    if (gid >= p.b * r) return;
    const uint row = gid / r, c = gid % r;
    if (row == 0) {
        msg[p.s3 + gid] = (ushort)0;
        return;
    }
    const ulong r3 = 3ul * r;
    const float q = bf2f(packed[p.s0 + (ulong)row * r3 + c]);
    float s, v;
    if (row == 1) {
        s = bf2f(s0[p.s2 + c]);
        v = bf2f(s0[p.s2 + r + c]);
    } else {
        s = bf2f(packed[p.s0 + (ulong)(row - 1) * r3 + r + c]);
        v = bf2f(packed[p.s0 + (ulong)(row - 1) * r3 + 2ul * r + c]);
    }
    const float z = (q + s) + bf2f(bias[p.s1 + c]);
    const float g = 1.0f / (1.0f + precise::exp(-z));
    msg[p.s3 + gid] = f2bf(g * v);
}

// ---- BridgeWrite: out = bf16(raw + (rms * delta) * (row != 0)); normed = bf16(bf16(out * rsqrt(mean(out^2) + eps)) * w) ------
// bind: 0 raw [T,H], 1 rms [T] f32, 2 delta [T,H], 3 w [H], 4 out [T,H], 5 normed [T,H]. a = H, f0 = eps; the skip of
// `normed` (the sixth binding, `OpP` has five skips) is in h. The second pass re-reads `out` at the elements the same thread wrote.
kernel void bridge_write_bf16(device const ushort* raw [[buffer(0)]],
                              device const float* rms [[buffer(1)]],
                              device const ushort* delta [[buffer(2)]],
                              device const ushort* w [[buffer(3)]],
                              device ushort* out [[buffer(4)]],
                              device ushort* normed [[buffer(5)]],
                              constant OpP& p [[buffer(6)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tgs [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint h = p.a;
    const ulong base = (ulong)row * h;
    const float rr = rms[p.s1 + row];
    const float na = row == 0 ? 0.0f : 1.0f;
    float ss = 0.0f;
    for (uint i = tid; i < h; i += tgs) {
        const ushort o = f2bf(bf2f(raw[p.s0 + base + i]) + (rr * bf2f(delta[p.s2 + base + i])) * na);
        out[p.s4 + base + i] = o;
        const float v = bf2f(o);
        ss = fma(v, v, ss);
    }
    const float tot = br_tg_sum(ss, part, lane, sg, nsg);
    const float inv = 1.0f / precise::sqrt(tot / (float)h + p.f0);
    for (uint i = tid; i < h; i += tgs) {
        const float y = bf2f(f2bf(bf2f(out[p.s4 + base + i]) * inv));
        normed[p.h + base + i] = f2bf(y * bf2f(w[p.s3 + i]));
    }
}
