// Phase B4: the attention prologue of a layer in one dispatch (Op::QkNormRopeKv). Replaces, per layer, QkNorm(q), Rope(q),
// QkNorm(k), Rope(k), KvAppend(k) and KvAppend(v): six dispatches (about 1 us of serial-encoder cost each plus the
// dependent-launch tails) become one.
//
// Bitwise the same results as the unfused sequence of `rmsnorm_bf16_ex1` (what `qk_norm_bf16` runs at head_dim <= 256),
// `rope_bf16` and `kv_append_bf16`: the same element-to-thread assignment (element i is thread i, threads = roundup32(D)),
// the same reduction (fma per thread, simd_sum, serial sum over the simdgroup partials), the same expressions, and the same
// rounding of the normed value to bf16 before the rotation (the unfused norm stores its output in a bf16 tensor that rope
// reads back). Only the intermediate round trip through memory is gone: the rotation partner comes from threadgroup memory.
//
// One threadgroup per (token, head): heads 0..N are q heads (norm, rope, store to q_out), N..N+Nkv are k heads (norm, rope,
// store into the k cache row pos0 + t) and N+Nkv..N+2*Nkv are v heads (copy into the v cache row). Head dim <= 256, even.
//
// bind: 0 q [T,N,D], 1 k [T,Nkv,D], 2 v [T,Nkv,D], 3 q_w [D], 4 k_w [D], 5 q_out [T,N,D], 6 k cache [Nkv,cap,D],
// 7 v cache, 8 pos (i32 [1], read when use_dev != 0), 9 params.

#include "common.h"

// Mirrored by `QkRopeKvP` in dense.rs (80 bytes). Skips are leading bf16 elements of each tensor, in binding order.
struct QkRopeKvP {
    uint t, n, nkv, d;       // tokens, q heads, kv heads, head dim
    uint rd, cap, use_dev, pos0;  // rotary dim (resolved, even, <= d), cache rows, device position flag, host pos0
    uint s_q, s_k, s_v, s_wq, s_wk, s_qo, s_kc, s_vc;
    float eps, w_offset, theta;
    uint pad;
};

// Threadgroup sum with a single barrier (part[] used once), none for one simdgroup: the same as `tg_sum1` of
// dense_ops.metal, which `rmsnorm_bf16_ex1` uses.
inline float qkr_tg_sum(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (nsg == 1) return v;
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint j = 0; j < nsg; ++j) t += part[j];
    return t;
}

kernel void qk_norm_rope_kv_bf16(device const ushort* q [[buffer(0)]],
                                 device const ushort* k [[buffer(1)]],
                                 device const ushort* v [[buffer(2)]],
                                 device const ushort* wq [[buffer(3)]],
                                 device const ushort* wk [[buffer(4)]],
                                 device ushort* qo [[buffer(5)]],
                                 device ushort* kc [[buffer(6)]],
                                 device ushort* vc [[buffer(7)]],
                                 device const int* posbuf [[buffer(8)]],
                                 constant QkRopeKvP& p [[buffer(9)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    threadgroup ushort nb[256];  // the bf16-rounded normed head, for the rotation partner
    const uint d = p.d;
    const uint heads = p.n + 2 * p.nkv;
    const uint t = p.t == 1 ? 0u : tg / heads;  // decode (one token): no integer division on the ramp
    const uint s = tg - t * heads;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    const uint i = tid;

    if (s >= p.n + p.nkv) {  // v head: cache[h, pos0 + t, :] = v[t, h, :]
        if (pos0 + t >= p.cap) return;  // a device-side position past the cache is dropped, as in kv_append_bf16
        const uint h = s - p.n - p.nkv;
        if (i < d) vc[p.s_vc + ((ulong)h * p.cap + pos0 + t) * d + i] = v[p.s_v + ((ulong)t * p.nkv + h) * d + i];
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;  // uniform over the threadgroup, so the barriers below stay safe

    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf = 0.0f, wf = 0.0f;
    if (i < d) {
        xf = bf2f(x[i]);
        wf = bf2f(w[i]);
    }
    // The rotation's angle depends only on the position and the lane, so it is computed while the loads and the
    // reduction are in flight instead of after the second barrier (same expressions, same values).
    const uint rd = p.rd;
    const uint half_ = rd / 2;
    const uint kk = i < half_ ? i : i - half_;
    float sn = 0.0f, cs = 0.0f;
    if (i < rd) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)kk / (float)rd);
        sn = precise::sin(ang);
        cs = precise::cos(ang);
    }
    float ss = 0.0f;
    if (i < d) ss = fma(xf, xf, ss);
    const float tot = qkr_tg_sum(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    if (i < d) nb[i] = f2bf(xf * r * (wf + p.w_offset));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= d) return;

    ushort res;
    if (i >= rd) {
        res = nb[i];  // beyond the rotary dim: the normed value is copied
    } else {
        const float x1 = bf2f(nb[kk]), x2 = bf2f(nb[kk + half_]);
        res = f2bf(i < half_ ? x1 * cs - x2 * sn : x2 * cs + x1 * sn);
    }
    if (isq) {
        qo[p.s_qo + ((ulong)t * p.n + h) * d + i] = res;
    } else {
        kc[p.s_kc + ((ulong)h * p.cap + pos0 + t) * d + i] = res;
    }
}

// Prefill form (T > 1, head_dim = rotary_dim = 128): bitwise `qk_norm_rope_kv_bf16`, built for T * (N + 2 Nkv) heads. The
// decode form spends most of its time in precise::pow / sin / cos, which every one of its 128 threads evaluates for
// every (token, head) although the angle depends only on the position and the rotation pair: 4096 sincos per token, 64
// distinct. Here one threadgroup is PF_HPT simdgroups = PF_HPT heads of one token, a table of the 64 (cos, sin) pairs is
// built once per threadgroup (the same expressions, so the same values), and a simdgroup owns one head: lane l holds
// elements l, 32 + l, 64 + l, 96 + l, so the four 32-element partial sums are four simd_sums (the sums of the decode
// form's four simdgroups), added serially in the same order, and the rotation partner of element i (i +- 64) sits in
// the same lane: no threadgroup memory and no barrier per head.
constant constexpr uint PF_HPT = 12;

kernel void qk_norm_rope_kv_bf16_pf(device const ushort* q [[buffer(0)]],
                                    device const ushort* k [[buffer(1)]],
                                    device const ushort* v [[buffer(2)]],
                                    device const ushort* wq [[buffer(3)]],
                                    device const ushort* wk [[buffer(4)]],
                                    device ushort* qo [[buffer(5)]],
                                    device ushort* kc [[buffer(6)]],
                                    device ushort* vc [[buffer(7)]],
                                    device const int* posbuf [[buffer(8)]],
                                    constant QkRopeKvP& p [[buffer(9)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 cs_sn[64];
    const uint d = 128;
    const uint heads = p.n + 2 * p.nkv;
    const uint groups = (heads + PF_HPT - 1) / PF_HPT;
    const uint t = tg / groups;
    const uint s = (tg - t * groups) * PF_HPT + sg;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    if (tid < 64) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)tid / (float)p.rd);
        cs_sn[tid] = float2(precise::cos(ang), precise::sin(ang));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s >= heads) return;

    if (s >= p.n + p.nkv) {  // v head
        if (pos0 + t >= p.cap) return;
        const uint h = s - p.n - p.nkv;
        for (uint g = 0; g < 4; ++g) {
            vc[p.s_vc + ((ulong)h * p.cap + pos0 + t) * d + g * 32 + lane] = v[p.s_v + ((ulong)t * p.nkv + h) * d + g * 32 + lane];
        }
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;
    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf[4], wf[4];
    float tot = 0.0f;
    for (uint g = 0; g < 4; ++g) {
        xf[g] = bf2f(x[g * 32 + lane]);
        wf[g] = bf2f(w[g * 32 + lane]);
        tot += simd_sum(fma(xf[g], xf[g], 0.0f));
    }
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    ushort nb[4];
    for (uint g = 0; g < 4; ++g) nb[g] = f2bf(xf[g] * r * (wf[g] + p.w_offset));
    ushort res[4];
    for (uint g = 0; g < 2; ++g) {
        const float cs = cs_sn[g * 32 + lane].x, sn = cs_sn[g * 32 + lane].y;
        const float x1 = bf2f(nb[g]), x2 = bf2f(nb[g + 2]);
        res[g] = f2bf(x1 * cs - x2 * sn);
        res[g + 2] = f2bf(x2 * cs + x1 * sn);
    }
    if (isq) {
        for (uint g = 0; g < 4; ++g) qo[p.s_qo + ((ulong)t * p.n + h) * d + g * 32 + lane] = res[g];
    } else {
        for (uint g = 0; g < 4; ++g) kc[p.s_kc + ((ulong)h * p.cap + pos0 + t) * d + g * 32 + lane] = res[g];
    }
}

// fp8 e4m3 KV cache form (`--kv-dtype fp8_e4m3`): exactly `qk_norm_rope_kv_bf16` up to the cache stores, which write the
// e4m3 code of the bf16-rounded value (the value the bf16 cache would hold, i.e. the unfused Rope -> KvAppend order of the
// cpu oracle), caches are one byte per element, the cache skips (s_kc, s_vc) are in bytes. v heads are encoded too.
kernel void qk_norm_rope_kv_f8(device const ushort* q [[buffer(0)]],
                                 device const ushort* k [[buffer(1)]],
                                 device const ushort* v [[buffer(2)]],
                                 device const ushort* wq [[buffer(3)]],
                                 device const ushort* wk [[buffer(4)]],
                                 device ushort* qo [[buffer(5)]],
                                 device uchar* kc [[buffer(6)]],
                                 device uchar* vc [[buffer(7)]],
                                 device const int* posbuf [[buffer(8)]],
                                 constant QkRopeKvP& p [[buffer(9)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    threadgroup ushort nb[256];  // the bf16-rounded normed head, for the rotation partner
    const uint d = p.d;
    const uint heads = p.n + 2 * p.nkv;
    const uint t = p.t == 1 ? 0u : tg / heads;  // decode (one token): no integer division on the ramp
    const uint s = tg - t * heads;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    const uint i = tid;

    if (s >= p.n + p.nkv) {  // v head: cache[h, pos0 + t, :] = v[t, h, :]
        if (pos0 + t >= p.cap) return;  // a device-side position past the cache is dropped, as in kv_append_bf16
        const uint h = s - p.n - p.nkv;
        if (i < d) vc[p.s_vc + ((ulong)h * p.cap + pos0 + t) * d + i] = (uchar)f32_to_e4m3(bf2f(v[p.s_v + ((ulong)t * p.nkv + h) * d + i]));
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;  // uniform over the threadgroup, so the barriers below stay safe

    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf = 0.0f, wf = 0.0f;
    if (i < d) {
        xf = bf2f(x[i]);
        wf = bf2f(w[i]);
    }
    // The rotation's angle depends only on the position and the lane, so it is computed while the loads and the
    // reduction are in flight instead of after the second barrier (same expressions, same values).
    const uint rd = p.rd;
    const uint half_ = rd / 2;
    const uint kk = i < half_ ? i : i - half_;
    float sn = 0.0f, cs = 0.0f;
    if (i < rd) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)kk / (float)rd);
        sn = precise::sin(ang);
        cs = precise::cos(ang);
    }
    float ss = 0.0f;
    if (i < d) ss = fma(xf, xf, ss);
    const float tot = qkr_tg_sum(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    if (i < d) nb[i] = f2bf(xf * r * (wf + p.w_offset));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= d) return;

    ushort res;
    if (i >= rd) {
        res = nb[i];  // beyond the rotary dim: the normed value is copied
    } else {
        const float x1 = bf2f(nb[kk]), x2 = bf2f(nb[kk + half_]);
        res = f2bf(i < half_ ? x1 * cs - x2 * sn : x2 * cs + x1 * sn);
    }
    if (isq) {
        qo[p.s_qo + ((ulong)t * p.n + h) * d + i] = res;
    } else {
        kc[p.s_kc + ((ulong)h * p.cap + pos0 + t) * d + i] = (uchar)f32_to_e4m3(bf2f(res));
    }
}

kernel void qk_norm_rope_kv_f8_pf(device const ushort* q [[buffer(0)]],
                                    device const ushort* k [[buffer(1)]],
                                    device const ushort* v [[buffer(2)]],
                                    device const ushort* wq [[buffer(3)]],
                                    device const ushort* wk [[buffer(4)]],
                                    device ushort* qo [[buffer(5)]],
                                    device uchar* kc [[buffer(6)]],
                                    device uchar* vc [[buffer(7)]],
                                    device const int* posbuf [[buffer(8)]],
                                    constant QkRopeKvP& p [[buffer(9)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 cs_sn[64];
    const uint d = 128;
    const uint heads = p.n + 2 * p.nkv;
    const uint groups = (heads + PF_HPT - 1) / PF_HPT;
    const uint t = tg / groups;
    const uint s = (tg - t * groups) * PF_HPT + sg;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    if (tid < 64) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)tid / (float)p.rd);
        cs_sn[tid] = float2(precise::cos(ang), precise::sin(ang));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s >= heads) return;

    if (s >= p.n + p.nkv) {  // v head
        if (pos0 + t >= p.cap) return;
        const uint h = s - p.n - p.nkv;
        for (uint g = 0; g < 4; ++g) {
            vc[p.s_vc + ((ulong)h * p.cap + pos0 + t) * d + g * 32 + lane] =
                (uchar)f32_to_e4m3(bf2f(v[p.s_v + ((ulong)t * p.nkv + h) * d + g * 32 + lane]));
        }
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;
    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf[4], wf[4];
    float tot = 0.0f;
    for (uint g = 0; g < 4; ++g) {
        xf[g] = bf2f(x[g * 32 + lane]);
        wf[g] = bf2f(w[g * 32 + lane]);
        tot += simd_sum(fma(xf[g], xf[g], 0.0f));
    }
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    ushort nb[4];
    for (uint g = 0; g < 4; ++g) nb[g] = f2bf(xf[g] * r * (wf[g] + p.w_offset));
    ushort res[4];
    for (uint g = 0; g < 2; ++g) {
        const float cs = cs_sn[g * 32 + lane].x, sn = cs_sn[g * 32 + lane].y;
        const float x1 = bf2f(nb[g]), x2 = bf2f(nb[g + 2]);
        res[g] = f2bf(x1 * cs - x2 * sn);
        res[g + 2] = f2bf(x2 * cs + x1 * sn);
    }
    if (isq) {
        for (uint g = 0; g < 4; ++g) qo[p.s_qo + ((ulong)t * p.n + h) * d + g * 32 + lane] = res[g];
    } else {
        for (uint g = 0; g < 4; ++g) kc[p.s_kc + ((ulong)h * p.cap + pos0 + t) * d + g * 32 + lane] = (uchar)f32_to_e4m3(bf2f(res[g]));
    }
}

// PROBE (not a serving kernel): bf16 cache holding bf16(dequant(quant(x))) per block; INTARSIA_METAL_KV_FAKEQ=<block>,<mask>
// mask bit0 = K, bit1 = V. Measures the accuracy an int8 block-scaled cache would have before building one.
kernel void qk_norm_rope_kv_fq(device const ushort* q [[buffer(0)]],
                                 device const ushort* k [[buffer(1)]],
                                 device const ushort* v [[buffer(2)]],
                                 device const ushort* wq [[buffer(3)]],
                                 device const ushort* wk [[buffer(4)]],
                                 device ushort* qo [[buffer(5)]],
                                 device ushort* kc [[buffer(6)]],
                                 device ushort* vc [[buffer(7)]],
                                 device const int* posbuf [[buffer(8)]],
                                 constant QkRopeKvP& p [[buffer(9)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    threadgroup ushort nb[256];  // the bf16-rounded normed head, for the rotation partner
    const uint d = p.d;
    const uint heads = p.n + 2 * p.nkv;
    const uint t = p.t == 1 ? 0u : tg / heads;  // decode (one token): no integer division on the ramp
    const uint s = tg - t * heads;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    const uint i = tid;

    if (s >= p.n + p.nkv) {  // v head: cache[h, pos0 + t, :] = v[t, h, :]
        if (pos0 + t >= p.cap) return;  // a device-side position past the cache is dropped, as in kv_append_bf16
        const uint h = s - p.n - p.nkv;
        {
            const float xv = i < d ? bf2f(v[p.s_v + ((ulong)t * p.nkv + h) * d + i]) : 0.0f;
            const uint blk = p.pad & 0xffu;
            const float sv = kvq_scale(xv, blk, lane);
            const float dq = (p.pad & 0x200u) ? kvq_q(xv, sv) * sv : xv;
            if (i < d) vc[p.s_vc + ((ulong)h * p.cap + pos0 + t) * d + i] = f2bf(dq);
        }
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;  // uniform over the threadgroup, so the barriers below stay safe

    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf = 0.0f, wf = 0.0f;
    if (i < d) {
        xf = bf2f(x[i]);
        wf = bf2f(w[i]);
    }
    // The rotation's angle depends only on the position and the lane, so it is computed while the loads and the
    // reduction are in flight instead of after the second barrier (same expressions, same values).
    const uint rd = p.rd;
    const uint half_ = rd / 2;
    const uint kk = i < half_ ? i : i - half_;
    float sn = 0.0f, cs = 0.0f;
    if (i < rd) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)kk / (float)rd);
        sn = precise::sin(ang);
        cs = precise::cos(ang);
    }
    float ss = 0.0f;
    if (i < d) ss = fma(xf, xf, ss);
    const float tot = qkr_tg_sum(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    if (i < d) nb[i] = f2bf(xf * r * (wf + p.w_offset));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= d) return;

    ushort res;
    if (i >= rd) {
        res = nb[i];  // beyond the rotary dim: the normed value is copied
    } else {
        const float x1 = bf2f(nb[kk]), x2 = bf2f(nb[kk + half_]);
        res = f2bf(i < half_ ? x1 * cs - x2 * sn : x2 * cs + x1 * sn);
    }
    if (isq) {
        qo[p.s_qo + ((ulong)t * p.n + h) * d + i] = res;
    } else {
        {
        const uint blk = p.pad & 0xffu;
        const float xr = bf2f(res);
        const float sk = kvq_scale(xr, blk, lane);
        const float dq = (p.pad & 0x100u) ? kvq_q(xr, sk) * sk : xr;
        kc[p.s_kc + ((ulong)h * p.cap + pos0 + t) * d + i] = f2bf(dq);
    }
    }
}

kernel void qk_norm_rope_kv_fq_pf(device const ushort* q [[buffer(0)]],
                                    device const ushort* k [[buffer(1)]],
                                    device const ushort* v [[buffer(2)]],
                                    device const ushort* wq [[buffer(3)]],
                                    device const ushort* wk [[buffer(4)]],
                                    device ushort* qo [[buffer(5)]],
                                    device ushort* kc [[buffer(6)]],
                                    device ushort* vc [[buffer(7)]],
                                    device const int* posbuf [[buffer(8)]],
                                    constant QkRopeKvP& p [[buffer(9)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 cs_sn[64];
    const uint d = 128;
    const uint heads = p.n + 2 * p.nkv;
    const uint groups = (heads + PF_HPT - 1) / PF_HPT;
    const uint t = tg / groups;
    const uint s = (tg - t * groups) * PF_HPT + sg;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    if (tid < 64) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)tid / (float)p.rd);
        cs_sn[tid] = float2(precise::cos(ang), precise::sin(ang));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s >= heads) return;

    if (s >= p.n + p.nkv) {  // v head
        if (pos0 + t >= p.cap) return;
        const uint h = s - p.n - p.nkv;
        for (uint g = 0; g < 4; ++g) {
            {
                const float xv = bf2f(v[p.s_v + ((ulong)t * p.nkv + h) * d + g * 32 + lane]);
                const float sv = kvq_scale(xv, p.pad & 0xffu, lane);
                const float dq = (p.pad & 0x200u) ? kvq_q(xv, sv) * sv : xv;
                vc[p.s_vc + ((ulong)h * p.cap + pos0 + t) * d + g * 32 + lane] = f2bf(dq);
            }
        }
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;
    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf[4], wf[4];
    float tot = 0.0f;
    for (uint g = 0; g < 4; ++g) {
        xf[g] = bf2f(x[g * 32 + lane]);
        wf[g] = bf2f(w[g * 32 + lane]);
        tot += simd_sum(fma(xf[g], xf[g], 0.0f));
    }
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    ushort nb[4];
    for (uint g = 0; g < 4; ++g) nb[g] = f2bf(xf[g] * r * (wf[g] + p.w_offset));
    ushort res[4];
    for (uint g = 0; g < 2; ++g) {
        const float cs = cs_sn[g * 32 + lane].x, sn = cs_sn[g * 32 + lane].y;
        const float x1 = bf2f(nb[g]), x2 = bf2f(nb[g + 2]);
        res[g] = f2bf(x1 * cs - x2 * sn);
        res[g + 2] = f2bf(x2 * cs + x1 * sn);
    }
    if (isq) {
        for (uint g = 0; g < 4; ++g) qo[p.s_qo + ((ulong)t * p.n + h) * d + g * 32 + lane] = res[g];
    } else {
        for (uint g = 0; g < 4; ++g) {
            const float xr = bf2f(res[g]);
            const float sk = kvq_scale(xr, p.pad & 0xffu, lane);
            const float dq = (p.pad & 0x100u) ? kvq_q(xr, sk) * sk : xr;
            kc[p.s_kc + ((ulong)h * p.cap + pos0 + t) * d + g * 32 + lane] = f2bf(dq);
        }
    }
}

// int8 block-scaled KV cache form (`--kv-dtype int8`): the same prologue as `qk_norm_rope_kv_f8`, the cache stores write
// q = clamp(rint(x / s), -127, 127) of the bf16-rounded value with the block scale s = bf16(absmax / 127) of the 32 elements of
// the simdgroup (kvq_scale / kvq_q, common.h) and the scale itself into the plane's scale area after the data
// ([Nkv, cap, D / 32] bf16). Both forms write identical cache bytes. s_kc / s_vc are bytes; head_dim must be a multiple of 32.
kernel void qk_norm_rope_kv_i8(device const ushort* q [[buffer(0)]],
                                 device const ushort* k [[buffer(1)]],
                                 device const ushort* v [[buffer(2)]],
                                 device const ushort* wq [[buffer(3)]],
                                 device const ushort* wk [[buffer(4)]],
                                 device ushort* qo [[buffer(5)]],
                                 device uchar* kc [[buffer(6)]],
                                 device uchar* vc [[buffer(7)]],
                                 device const int* posbuf [[buffer(8)]],
                                 constant QkRopeKvP& p [[buffer(9)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    threadgroup ushort nb[256];  // the bf16-rounded normed head, for the rotation partner
    const uint d = p.d;
    const uint heads = p.n + 2 * p.nkv;
    const uint t = p.t == 1 ? 0u : tg / heads;  // decode (one token): no integer division on the ramp
    const uint s = tg - t * heads;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    const uint i = tid;

    if (s >= p.n + p.nkv) {  // v head: cache[h, pos0 + t, :] = v[t, h, :]
        if (pos0 + t >= p.cap) return;  // a device-side position past the cache is dropped, as in kv_append_bf16
        const uint h = s - p.n - p.nkv;
        {
            const float xv = i < d ? bf2f(v[p.s_v + ((ulong)t * p.nkv + h) * d + i]) : 0.0f;
            const float sv = kvq_scale(xv, KVQ_B, lane);
            const ulong row = (ulong)h * p.cap + pos0 + t;
            if (i < d) vc[p.s_vc + row * d + i] = (uchar)((int)kvq_q(xv, sv) & 0xff);
            if (i < d && (i % KVQ_B) == 0)
                ((device ushort*)(vc + p.s_vc + (ulong)p.nkv * p.cap * d))[row * (d / KVQ_B) + i / KVQ_B] = f2bf(sv);
        }
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;  // uniform over the threadgroup, so the barriers below stay safe

    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf = 0.0f, wf = 0.0f;
    if (i < d) {
        xf = bf2f(x[i]);
        wf = bf2f(w[i]);
    }
    // The rotation's angle depends only on the position and the lane, so it is computed while the loads and the
    // reduction are in flight instead of after the second barrier (same expressions, same values).
    const uint rd = p.rd;
    const uint half_ = rd / 2;
    const uint kk = i < half_ ? i : i - half_;
    float sn = 0.0f, cs = 0.0f;
    if (i < rd) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)kk / (float)rd);
        sn = precise::sin(ang);
        cs = precise::cos(ang);
    }
    float ss = 0.0f;
    if (i < d) ss = fma(xf, xf, ss);
    const float tot = qkr_tg_sum(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    if (i < d) nb[i] = f2bf(xf * r * (wf + p.w_offset));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= d) return;

    ushort res;
    if (i >= rd) {
        res = nb[i];  // beyond the rotary dim: the normed value is copied
    } else {
        const float x1 = bf2f(nb[kk]), x2 = bf2f(nb[kk + half_]);
        res = f2bf(i < half_ ? x1 * cs - x2 * sn : x2 * cs + x1 * sn);
    }
    if (isq) {
        qo[p.s_qo + ((ulong)t * p.n + h) * d + i] = res;
    } else {
        {
        const float xr = bf2f(res);
        const float sk = kvq_scale(xr, KVQ_B, lane);
        const ulong row = (ulong)h * p.cap + pos0 + t;
        kc[p.s_kc + row * d + i] = (uchar)((int)kvq_q(xr, sk) & 0xff);
        if ((i % KVQ_B) == 0) ((device ushort*)(kc + p.s_kc + (ulong)p.nkv * p.cap * d))[row * (d / KVQ_B) + i / KVQ_B] = f2bf(sk);
    }
    }
}

kernel void qk_norm_rope_kv_i8_pf(device const ushort* q [[buffer(0)]],
                                    device const ushort* k [[buffer(1)]],
                                    device const ushort* v [[buffer(2)]],
                                    device const ushort* wq [[buffer(3)]],
                                    device const ushort* wk [[buffer(4)]],
                                    device ushort* qo [[buffer(5)]],
                                    device uchar* kc [[buffer(6)]],
                                    device uchar* vc [[buffer(7)]],
                                    device const int* posbuf [[buffer(8)]],
                                    constant QkRopeKvP& p [[buffer(9)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 cs_sn[64];
    const uint d = 128;
    const uint heads = p.n + 2 * p.nkv;
    const uint groups = (heads + PF_HPT - 1) / PF_HPT;
    const uint t = tg / groups;
    const uint s = (tg - t * groups) * PF_HPT + sg;
    const uint pos0 = p.use_dev != 0 ? (uint)posbuf[0] : p.pos0;
    if (tid < 64) {
        const float ang = (float)(pos0 + t) * precise::pow(p.theta, -2.0f * (float)tid / (float)p.rd);
        cs_sn[tid] = float2(precise::cos(ang), precise::sin(ang));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s >= heads) return;

    if (s >= p.n + p.nkv) {  // v head
        if (pos0 + t >= p.cap) return;
        const uint h = s - p.n - p.nkv;
        for (uint g = 0; g < 4; ++g) {
            {
                const float xv = bf2f(v[p.s_v + ((ulong)t * p.nkv + h) * d + g * 32 + lane]);
                const float sv = kvq_scale(xv, KVQ_B, lane);
                const ulong row = (ulong)h * p.cap + pos0 + t;
                vc[p.s_vc + row * d + g * 32 + lane] = (uchar)((int)kvq_q(xv, sv) & 0xff);
                if (lane == 0) ((device ushort*)(vc + p.s_vc + (ulong)p.nkv * p.cap * d))[row * (d / KVQ_B) + g] = f2bf(sv);
            }
        }
        return;
    }
    const bool isq = s < p.n;
    const uint h = isq ? s : s - p.n;
    if (!isq && pos0 + t >= p.cap) return;
    device const ushort* x = isq ? q + p.s_q + ((ulong)t * p.n + h) * d : k + p.s_k + ((ulong)t * p.nkv + h) * d;
    device const ushort* w = isq ? wq + p.s_wq : wk + p.s_wk;

    float xf[4], wf[4];
    float tot = 0.0f;
    for (uint g = 0; g < 4; ++g) {
        xf[g] = bf2f(x[g * 32 + lane]);
        wf[g] = bf2f(w[g * 32 + lane]);
        tot += simd_sum(fma(xf[g], xf[g], 0.0f));
    }
    const float r = 1.0f / precise::sqrt(tot / (float)d + p.eps);
    ushort nb[4];
    for (uint g = 0; g < 4; ++g) nb[g] = f2bf(xf[g] * r * (wf[g] + p.w_offset));
    ushort res[4];
    for (uint g = 0; g < 2; ++g) {
        const float cs = cs_sn[g * 32 + lane].x, sn = cs_sn[g * 32 + lane].y;
        const float x1 = bf2f(nb[g]), x2 = bf2f(nb[g + 2]);
        res[g] = f2bf(x1 * cs - x2 * sn);
        res[g + 2] = f2bf(x2 * cs + x1 * sn);
    }
    if (isq) {
        for (uint g = 0; g < 4; ++g) qo[p.s_qo + ((ulong)t * p.n + h) * d + g * 32 + lane] = res[g];
    } else {
        for (uint g = 0; g < 4; ++g) {
            const float xr = bf2f(res[g]);
            const float sk = kvq_scale(xr, KVQ_B, lane);
            const ulong row = (ulong)h * p.cap + pos0 + t;
            kc[p.s_kc + row * d + g * 32 + lane] = (uchar)((int)kvq_q(xr, sk) & 0xff);
            if (lane == 0) ((device ushort*)(kc + p.s_kc + (ulong)p.nkv * p.cap * d))[row * (d / KVQ_B) + g] = f2bf(sk);
        }
    }
}
