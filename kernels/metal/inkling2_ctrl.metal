// Inkling-2-Mini decode controller fusion on Metal (lane ink-dfuse; launcher crates/intarsia-backend-metal/src/inkctrl.rs).
// Contracts: crates/intarsia-core/src/inkling2/ops.rs (`InkRoute` 76, `ConvMixRoute` 77). Compiled -std=metal4.0 into
// intarsia_metal4.metallib (build.rs METAL4_SOURCES) for the MSL 3.2 device-scope fences of the cross-threadgroup hand-off.
//
// Hand-off ("last threadgroup"): a dispatch whose tail needs a result of the whole grid lets the LAST threadgroup to finish a
// stage run the tail. Every thread that stored fences (seq_cst, device scope), the threadgroup barriers, thread 0 increments a
// device counter (one per row, in a slot of the backend's sync arena, sync.rs); the threadgroup that sees count == n - 1 resets
// the counter to 0 and fences again before every thread reads the other threadgroups' stores. No threadgroup ever waits for
// another, so no forward-progress guarantee is needed, and every call leaves its slot zero (no host memset).
//
// Bits: `ink_route` computes each logit exactly as `gemv_f32o_v` (16-byte operands) or `gemv_f32o_scalar` (2-byte operands)
// does (one simdgroup per weight row, the same chunk order and fma chain, simd_sum), and the routing exactly as `ink_moe_route`
// (inkling2_moe.metal: one simdgroup, the same score / selection / renorm code), so ids and weights are bitwise those of
// `Logits` + `MoeRoute`. A row never depends on the other rows of the call (rows-exact).

#include "common.h"
inline float rt_lo(uint u) { return as_type<float>(u << 16); }
inline float rt_hi(uint u) { return as_type<float>(u & 0xffff0000u); }

// gemv_f32out.metal `dot8`
inline float rt_dot8(uint4 w, uint4 x) {
    float a = rt_lo(w.x) * rt_lo(x.x);
    a = fma(rt_hi(w.x), rt_hi(x.x), a);
    a = fma(rt_lo(w.y), rt_lo(x.y), a);
    a = fma(rt_hi(w.y), rt_hi(x.y), a);
    a = fma(rt_lo(w.z), rt_lo(x.z), a);
    a = fma(rt_hi(w.z), rt_hi(x.z), a);
    a = fma(rt_lo(w.w), rt_lo(x.w), a);
    a = fma(rt_hi(w.w), rt_hi(x.w), a);
    return a;
}

// One logit (weight row `row`) of activation row X in one simdgroup: `gemv_f32o_v`'s loop (K % 8 == 0, 16-byte operands).
inline float rt_logit_v(device const uint4* W, device const uint4* X, uint row, uint k8, uint lane) {
    device const uint4* wr = W + (ulong)row * k8;
    float acc = 0.0f;
    const uint block = 32 * 4;
    const uint nblk = k8 / block;
    uint i = lane;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 xv[4], wv[4];
        for (int u = 0; u < 4; ++u) xv[u] = X[i + 32 * u];
        for (int u = 0; u < 4; ++u) wv[u] = wr[i + 32 * u];
        for (int u = 0; u < 4; ++u) acc += rt_dot8(wv[u], xv[u]);
    }
    for (i = nblk * block + lane; i < k8; i += 32) acc += rt_dot8(wr[i], X[i]);
    return simd_sum(acc);
}

// `gemv_f32o_scalar`'s loop (any K, 2-byte operands).
inline float rt_logit_s(device const ushort* W, device const ushort* x, uint row, uint k, uint lane) {
    device const ushort* w = W + (ulong)row * k;
    float acc = 0.0f;
    for (uint i = lane; i < k; i += 32) acc = fma(bf2f(w[i]), bf2f(x[i]), acc);
    return simd_sum(acc);
}

// ---- the routing of one row in one simdgroup: `ink_moe_route` (inkling2_moe.metal), unchanged ------------------------------

inline bool rt_better(float v1, uint i1, float v2, uint i2) {
    if (i2 == 0xFFFFFFFFu) return i1 != 0xFFFFFFFFu;
    if (i1 == 0xFFFFFFFFu) return false;
    return v1 > v2 || (v1 == v2 && i1 < i2);
}

inline float rt_log1p(float z) {
    const float u = 1.f + z;
    return u == 1.f ? z : precise::log(u) * (z / (u - 1.f));
}

// E <= 1024, K <= 32; sf: 0 softmax, 1 sigmoid, 2 sqrt-softplus.
inline void rt_route(coherent(device) device const float* row, device const float* bias, device float* wout, device int* iout, uint E, uint K,
                     uint norm_topk, uint sf, float scale, float temp, uint lane) {
    const uint NJ = (E + 31u) / 32u;
    float sc[32], v[32];
    for (uint j = 0; j < NJ; ++j) {
        const uint e = lane + 32u * j;
        const float l = e < E ? row[e] / temp : 0.f;
        sc[j] = l;
    }
    if (sf == 0u) {
        float mx = -INFINITY;
        for (uint j = 0; j < NJ; ++j)
            if (lane + 32u * j < E) mx = max(mx, sc[j]);
        mx = simd_max(mx);
        float s = 0.f;
        for (uint j = 0; j < NJ; ++j) {
            const bool ok = lane + 32u * j < E;
            sc[j] = ok ? precise::exp(sc[j] - mx) : 0.f;
            s += sc[j];
        }
        s = simd_sum(s);
        for (uint j = 0; j < NJ; ++j) sc[j] = precise::divide(sc[j], s);
    } else if (sf == 1u) {
        for (uint j = 0; j < NJ; ++j) sc[j] = precise::divide(1.f, 1.f + precise::exp(-sc[j]));
    } else {
        for (uint j = 0; j < NJ; ++j) sc[j] = precise::sqrt(max(sc[j], 0.f) + rt_log1p(precise::exp(-fabs(sc[j]))));
    }
    for (uint j = 0; j < NJ; ++j) {
        const uint e = lane + 32u * j;
        v[j] = e < E ? sc[j] + bias[e] : 0.f;
    }
    uint taken = 0u;
    float pick = 0.f, den = 0.f;
    for (uint k = 0; k < K; ++k) {
        float bv = 0.f, bs = 0.f;
        uint bi = 0xFFFFFFFFu;
        for (uint j = 0; j < NJ; ++j) {
            const uint e = lane + 32u * j;
            if (e < E && !(taken & (1u << j)) && rt_better(v[j], e, bv, bi)) {
                bv = v[j];
                bs = sc[j];
                bi = e;
            }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o), os = simd_shuffle_xor(bs, o);
            const uint oi = simd_shuffle_xor(bi, o);
            if (rt_better(ov, oi, bv, bi)) {
                bv = ov;
                bs = os;
                bi = oi;
            }
        }
        if (bi % 32u == lane) taken |= 1u << (bi / 32u);
        den += bs;
        if (lane == k) pick = bs;
        if (lane == k) iout[k] = (int)bi;
    }
    if (lane < K) {
        const float w = (norm_topk != 0u && K > 1u) ? precise::divide(pick, den + 1e-20f) : pick;
        wout[lane] = w * scale;
    }
}

// ---- the last-threadgroup hand-off -------------------------------------------------------------------------------------------

// Every thread calls this after its stores of the stage; returns (in every thread) whether this threadgroup is the last of
// `n` to arrive at `ctr`. The last one has reset the counter and fenced; its threads may read the other threadgroups' stores.
inline bool rt_last(device atomic_uint* ctr, uint n, uint tid, threadgroup uint* flag) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        const bool last = old == n - 1u;
        if (last) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
        *flag = last ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool last = *flag != 0u;
    if (last) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    return last;
}

// ================================================================================================================
// InkRoute (op 76): logits = x . W^T (f32), then MoeRoute. Grid (ceil(E / RT_SG), T) threadgroups of RT_SG simdgroups; the
// simdgroup s of threadgroup (j, t) computes logit j * RT_SG + s of row t; the last threadgroup of row t routes it.
// ================================================================================================================

constant constexpr uint RT_SG = 4;  // logits (simdgroups) per threadgroup, as gemv_f32o_v

struct InkRouteP {
    uint t, e, k, d;               // rows, experts, top-k, hidden (the GEMV's K)
    uint norm_topk, sfunc, vec, ngr;  // renorm, score function, 16-byte form, threadgroups per row (ceil(E / RT_SG))
    uint sx, sw, pad0, pad1;       // leading bf16 elements of x / W (scalar form; 0 in the 16-byte form)
    float scale, temp, fpad0, fpad1;
};

// bind: 0 x [T, D] bf16, 1 W [E, D] bf16, 2 bias [E] f32, 3 weights [T, K] f32, 4 ids [T, K] i32, 5 logits [T, E] f32 (the
// call's or scratch), 6 counters (sync slot, u32 per row), 7 InkRouteP.
kernel void ink_route(device const ushort* x [[buffer(0)]],
                      device const ushort* w [[buffer(1)]],
                      device const float* bias [[buffer(2)]],
                      device float* wout [[buffer(3)]],
                      device int* iout [[buffer(4)]],
                      coherent(device) device float* logits [[buffer(5)]],
                      device atomic_uint* ctr [[buffer(6)]],
                      constant InkRouteP& p [[buffer(7)]],
                      uint2 tg2 [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint flag;
    const uint j = tg2.x, t = tg2.y, E = p.e;
    const uint e = j * RT_SG + sg;
    const uint row = min(e, E - 1u);  // simdgroups past E read a valid row and do not store
    float s;
    if (p.vec != 0u) {
        const uint k8 = p.d / 8u;
        s = rt_logit_v((device const uint4*)w, (device const uint4*)x + (ulong)t * k8, row, k8, lane);
    } else {
        s = rt_logit_s(w + p.sw, x + p.sx + (ulong)t * p.d, row, p.d, lane);
    }
    coherent(device) device float* lrow = logits + (ulong)t * E;
    if (lane == 0 && e < E) lrow[e] = s;
    if (!rt_last(ctr + t, p.ngr, tid, &flag) || sg != 0) return;
    rt_route(lrow, bias, wout + (ulong)t * p.k, iout + (ulong)t * p.k, E, p.k, p.norm_topk, p.sfunc, p.scale, p.temp, lane);
}
