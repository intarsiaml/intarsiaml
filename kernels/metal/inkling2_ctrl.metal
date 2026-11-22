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

// ================================================================================================================
// The blend tail (lane ink-dfuse): `h` GEMV + gate + norm of the decode blend (ConvMix with msg, LoopBlend) in one dispatch,
// for chunks of up to 16 rows, after the pair (the row kernel or `ink_blend_pair`). Bitwise the three dispatches it replaces
// (inkling2_mix.metal: `ink_in_proj_gemv_r<R>p1s8u2b<R>` over the stacked `down`, `ink_blend_gate_g<GC>`, `ink_blend_norm`):
//   H items (S * G, tickets first): weight row `w` of stream s for every row, `ip_rows<1, 1, 2, 8>` per row (a row's sum does
//     not depend on how many rows a launch takes: the per-row bits of the R-row GEMV), h stored rA; then +1 on hcnt[s].
//   C items (S * ceil(D / 256)): 256 channels of stream s (simdgroup k = channels 32k.., the partial index of the 64-channel
//     slices of `ink_blend_gate`); load the channel's gain / forget rows, then wait until hcnt[s] == G (every H item has a
//     lower ticket, so it has started and finishes without waiting: no deadlock without a forward-progress guarantee), then
//     the gate for every row as `ink_blend_gate`; the last C item of stream s (gcnt[s]) runs `ink_blend_norm` for its rows.
// p.nsl here is the tail's own partial count, ceil(D / 256) * 8 (one per 32 channels, as the gate's; the trailing zeros past D
// do not change the norm's sum). Tickets come from an atomic at threadgroup start, so the roles follow the start order, not
// the grid index. Counters
// (sync arena, 1 + 2S): [0] tickets, [1 + s] hcnt, [1 + S + s] gcnt; all back to 0 at the end of the dispatch.
// ================================================================================================================

inline float ip_lo(uint u) { return as_type<float>(u << 16); }
inline float ip_hi(uint u) { return as_type<float>(u & 0xffff0000u); }

// inkling2_mix.metal `ip_rows` with RB = 1, P = 1 (the same chunk order, fma chain, simd_sum and sg-ordered partial sum).
template <int U, int SG>
inline float bt_row(device const uint4* W, uint row, uint nvalid, device const uint4* X, uint k8, uint sg, uint lane,
                    threadgroup float* part) {
    constexpr uint lanes = 32 * SG;
    constexpr uint block = lanes * U;
    device const uint4* wr = W + (ulong)min(row, nvalid - 1) * k8;
    const uint nblk = k8 / block;
    uint i = sg * 32 + lane;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 wv[U];
        for (int u = 0; u < U; ++u) wv[u] = wr[i + lanes * u];
        uint4 xv[U];
        for (int u = 0; u < U; ++u) xv[u] = X[i + lanes * u];
        for (int u = 0; u < U; ++u) acc += rt_dot8(wv[u], xv[u]);
    }
    for (i = nblk * block + sg * 32 + lane; i < k8; i += lanes) {
        const uint4 wv = wr[i];
        const uint4 xv = X[i];
        acc += rt_dot8(wv, xv);
    }
    if (SG == 1) return simd_sum(acc);
    const float s = simd_sum(acc);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = part[0];
    for (int q = 1; q < SG; ++q) t += part[q];
    return t;
}

// InkBlendP of inkling2_mix.metal (the same struct; the tail reads d, rows, s, sd, g, nsl, t0, ocount, dcount, mcount, dt0,
// mt0, eps, sk[0..4]).
struct InkBlendP {
    uint d, rows, s, sd;
    uint g, nsl, cs, t0;
    uint ocount, dcount, mcount, pad0;
    uint dt0, mt0, pad1, pad2;
    float eps, fpad0, fpad1, fpad2;
    uint sk[8];
};

constant constexpr uint BT_THREADS = 256;  // SG 8 of the h GEMV, 256 channels per C item, the norm's 256 threads

// bind: 0 down [S * G, 2D] (16-byte aligned), 1 pair bf16 [S, rows, 2D] (scratch), 2 dest f32, 3 msg f32, 4 gain [S, D, G],
// 5 forget [S, D, G], 6 norm [S, D], 7 out bf16 [S, ocount, D], 8 h bf16 [S, rows, G] (scratch), 9 y f32 [S, rows, D]
// (scratch), 10 partial f32 [S, rows, nsl] (scratch), 11 counters, 12 InkBlendP (sk[0] gain, sk[1] forget, sk[2] norm,
// sk[3] out). Grid: S * G + S * ceil(D / 256) threadgroups of 256 threads.
template <int GC>
kernel void ink_blend_tail(device const uint4* down [[buffer(0)]],
                           device const uint4* pair [[buffer(1)]],
                           device const float* dest [[buffer(2)]],
                           device const float* msg [[buffer(3)]],
                           device const ushort* gain [[buffer(4)]],
                           device const ushort* forget [[buffer(5)]],
                           device const ushort* nrm [[buffer(6)]],
                           device ushort* out [[buffer(7)]],
                           coherent(device) device ushort* hb [[buffer(8)]],
                           coherent(device) device float* yo [[buffer(9)]],
                           coherent(device) device float* partial [[buffer(10)]],
                           device atomic_uint* ctr [[buffer(11)]],
                           constant InkBlendP& p [[buffer(12)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint tk_s;
    threadgroup float part[8];
    threadgroup uint flag;
    const uint ntg = p.s * p.g + p.s * ((p.d + BT_THREADS - 1) / BT_THREADS);
    if (tid == 0) {
        const uint t = atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
        if (t == ntg - 1) atomic_store_explicit(&ctr[0], 0u, memory_order_relaxed);  // every ticket is taken
        tk_s = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tk = tk_s, d = p.d, g = p.g, S = p.s, rows = p.rows;
    const uint k8 = 2 * d / 8;  // the pair rows are 2D wide
    if (tk < S * g) {
        // ---- H item: weight row w of stream s, every row ----
        const uint s = tk / g, w = tk % g;
        device const uint4* Ws = down + (ulong)s * g * k8;
        for (uint r = 0; r < rows; ++r) {
            const float t = bt_row<2, 8>(Ws, w, g, pair + ((ulong)s * rows + r) * k8, k8, sg, lane, part);
            if (sg == 0 && lane == 0) hb[((ulong)s * rows + r) * g + w] = f2bf(t);
            threadgroup_barrier(mem_flags::mem_threadgroup);  // `part` is rewritten by the next row
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_fetch_add_explicit(&ctr[1 + s], 1u, memory_order_relaxed);
        }
        return;
    }
    // ---- C item: 256 channels of stream s ----
    const uint nq = (d + BT_THREADS - 1) / BT_THREADS;
    const uint ci = tk - S * g, s = ci / nq, q = ci % nq, g8 = g / 8;
    const uint c = q * BT_THREADS + tid;
    const bool on = c < d;
    const uint cc = min(c, d - 1);
    device const uint4* gr = (device const uint4*)(gain + p.sk[0] + ((ulong)s * d + cc) * g);
    device const uint4* fr = (device const uint4*)(forget + p.sk[1] + ((ulong)s * d + cc) * g);
    uint4 gw[GC], fw[GC];
    for (int j = 0; j < GC; ++j) {
        gw[j] = (uint(j) < g8) ? gr[j] : uint4(0);
        fw[j] = (uint(j) < g8) ? fr[j] : uint4(0);
    }
    if (tid == 0) {
        while (atomic_load_explicit(&ctr[1 + s], memory_order_relaxed) < g) {
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const uint pk = q * (BT_THREADS / 32) + sg;  // this simdgroup's partial: the 32-channel half-slice index
    for (uint r = 0; r < rows; ++r) {
        coherent(device) device const uint4* hr = (coherent(device) device const uint4*)(hb + ((ulong)s * rows + r) * g);
        float gn = 0.0f, fg = 0.0f;
        for (int j = 0; j < GC; ++j) {
            if (uint(j) >= g8) break;
            const uint4 hv = hr[j];
            gn += rt_dot8(gw[j], hv);
            fg += rt_dot8(fw[j], hv);
        }
        float yv = 0.0f;
        if (on) {
            const float dv = dest[((ulong)(p.sd == 1 ? 0 : s) * p.dcount + p.dt0 + r) * d + c];
            const float mv = msg[((ulong)s * p.mcount + p.mt0 + r) * d + c];
            yv = (1.0f - precise::tanh(fg)) * dv + precise::tanh(gn) * mv;
            yo[((ulong)s * rows + r) * d + c] = yv;
        }
        const float ss = simd_sum(yv * yv);
        if (lane == 0 && pk < p.nsl) partial[((ulong)s * rows + r) * p.nsl + pk] = ss;
    }
    if (!rt_last(&ctr[1 + S + s], nq, tid, &flag)) return;
    if (tid == 0) atomic_store_explicit(&ctr[1 + s], 0u, memory_order_relaxed);  // every C item of s is past its wait
    // ---- the norm of stream s (`ink_blend_norm`, 256 threads) ----
    for (uint r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (uint j = 0; j < p.nsl; ++j) ss += partial[((ulong)s * rows + r) * p.nsl + j];
        const float inv = precise::rsqrt(ss / float(d) + p.eps);
        coherent(device) device const float* yr = yo + ((ulong)s * rows + r) * d;
        for (uint cx = tid; cx < d; cx += BT_THREADS)
            out[p.sk[3] + ((ulong)s * p.ocount + p.t0 + r) * d + cx] = f2bf(yr[cx] * inv * bf2f(nrm[p.sk[2] + (ulong)s * d + cx]));
    }
}

#define BT_ARGS device const uint4*, device const uint4*, device const float*, device const float*, device const ushort*, \
    device const ushort*, device const ushort*, device ushort*, coherent(device) device ushort*, coherent(device) device float*, \
    coherent(device) device float*, device atomic_uint*, constant InkBlendP&, uint, uint, uint
template [[host_name("ink_blend_tail_g1")]] kernel void ink_blend_tail<1>(BT_ARGS);
template [[host_name("ink_blend_tail_g8")]] kernel void ink_blend_tail<8>(BT_ARGS);
template [[host_name("ink_blend_tail_g16")]] kernel void ink_blend_tail<16>(BT_ARGS);
