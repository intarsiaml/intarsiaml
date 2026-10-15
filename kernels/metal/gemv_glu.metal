// Decode D1: the dense FFN's gate / up half in one dispatch (Op::MatmulGlu, Op::RmsNormAddMatmulGlu).
//
//   MatmulGlu:            a = bf16( silu(g) * u ),   g = bf16(x . Wg_row), u = bf16(x . Wu_row)
//   RmsNormAddMatmulGlu:  the same with x := RmsNormAdd(blk, res, w) computed in a prologue (and `sum` stored by threadgroup 0)
//
// The unfused decode sequence is `rmsnorm_add_bf16_ex*` (normed, sum), `gemv_pair_bf16_rt_*` (g, u) and `silu_glu_bf16`
// (a): three dispatches and two round trips of g / u / normed through memory. Here one threadgroup of 256 threads takes
// output row `tg` of BOTH matrices, so g and u of a row meet in the threadgroup and the GLU is the epilogue.
//
// Bitwise the unfused sequence by construction:
//  * each row's dot product keeps the lane / step / simd_sum / serial-simdgroup-partials order of `gemv_bf16_rt<2, 8>`
//    (16-byte loads, UNROLL 2) or `gemv_bf16_rt_a8<4, 8>` (8-byte loads, UNROLL 4): lane `tid` of the 256 reads chunk
//    (step * UNROLL + u) * 256 + tid of the weight row and of x, accumulates `acc += dot(w, x)` over u and steps, and the
//    eight simdgroup sums are added in order;
//  * the prologue is `rmsnorm_add_bf16_ex16` (256 threads, element i belongs to thread i % 256, 16 elements per thread,
//    fma sum of squares, simd_sum, serial sum over the simdgroup partials, then the same expressions), with the normed bf16
//    value kept in threadgroup memory instead of a device tensor;
//  * the epilogue is `silu_glu_bf16`'s expression on the bf16-rounded g and u.
// The third variant (2-byte loads, any alignment / K) is the correctness fallback for operands the vector loads cannot
// take; its accumulation order is its own (within tolerance of, not bitwise, the unfused sequence).
//
// bind: 0 Wg, 1 Wu, 2 X (x for MatmulGlu; blk for the norm form, read as bf16), 3 RES, 4 NW, 5 Y (a), 6 SUM, 7 params.
// Unused buffers of the plain form are bound to Y. Skips are leading bf16 elements of each tensor (the host binds at 4-byte
// aligned offsets); W / X skips are used by the 2-byte variants only.

#include "common.h"

struct GluP {
    uint k;       // reduction dim (the hidden size)
    uint n;       // output rows (the intermediate size); one threadgroup each
    uint wg_skip;
    uint wu_skip;
    uint x_skip;
    uint y_skip;
    uint res_skip;
    uint nw_skip;
    uint sum_skip;
    float eps;
    float w_off;
    uint pad;
};

inline float g_lo(uint u) { return as_type<float>(u << 16); }
inline float g_hi(uint u) { return as_type<float>(u & 0xffff0000u); }

// the dot products of gemv_bf16.metal (dot8 / dot4), same fma chain
inline float gdot(uint4 w, uint4 x) {
    float a = g_lo(w.x) * g_lo(x.x);
    a = fma(g_hi(w.x), g_hi(x.x), a);
    a = fma(g_lo(w.y), g_lo(x.y), a);
    a = fma(g_hi(w.y), g_hi(x.y), a);
    a = fma(g_lo(w.z), g_lo(x.z), a);
    a = fma(g_hi(w.z), g_hi(x.z), a);
    a = fma(g_lo(w.w), g_lo(x.w), a);
    a = fma(g_hi(w.w), g_hi(x.w), a);
    return a;
}
inline float gdot(uint2 w, uint2 x) {
    float a = g_lo(w.x) * g_lo(x.x);
    a = fma(g_hi(w.x), g_hi(x.x), a);
    a = fma(g_lo(w.y), g_lo(x.y), a);
    a = fma(g_hi(w.y), g_hi(x.y), a);
    return a;
}
inline float gdot(ushort w, ushort x) { return bf2f(w) * bf2f(x); }

// a pointer plus the element skip, for the 2-byte variants only (the vector variants are bound exactly aligned)
template <typename V>
inline device const V* g_adv(device const V* p, uint) { return p; }
inline device const ushort* g_adv(device const ushort* p, uint s) { return p + s; }

inline ushort g_bf16(float v) { return as_type<ushort>(bfloat(v)); }

template <typename V, int VEL, int UNROLL, bool NORM>
kernel void gemv_glu_t(device const V* Wg [[buffer(0)]],
                       device const V* Wu [[buffer(1)]],
                       device const V* X [[buffer(2)]],
                       device const ushort* RES [[buffer(3)]],
                       device const ushort* NW [[buffer(4)]],
                       device ushort* Y [[buffer(5)]],
                       device ushort* SUM [[buffer(6)]],
                       constant GluP& p [[buffer(7)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint SG = 8;
    constexpr uint lanes = 256;
    constexpr uint block = lanes * UNROLL;
    threadgroup float part[2 * SG];
    threadgroup float npart[SG];
    threadgroup uint4 xt[NORM ? 512 : 1];  // the normed x (4096 bf16 at most)

    const uint kv = p.k / VEL;
    device const V* wg = g_adv(Wg, p.wg_skip) + (ulong)tg * kv;
    device const V* wu = g_adv(Wu, p.wu_skip) + (ulong)tg * kv;
    const uint nblk = kv / block;

    // the first block of both weight rows goes out before the prologue: its latency hides behind the norm
    V gw[UNROLL], uw[UNROLL];
    for (int u = 0; u < UNROLL; ++u) { gw[u] = V(0); uw[u] = V(0); }
    if (nblk > 0) {
        for (int u = 0; u < UNROLL; ++u) {
            gw[u] = wg[tid + lanes * u];
            uw[u] = wu[tid + lanes * u];
        }
    }

    threadgroup const V* xs = (threadgroup const V*)xt;
    device const V* xd = g_adv(X, p.x_skip);
    if (NORM) {
        // rmsnorm_add_bf16_ex16: sum = bf16(x + res), normed = bf16(sum * r * (w + off))
        constexpr int NIT = 16;
        const uint h = p.k;
        device const ushort* xr = (device const ushort*)X + p.x_skip;
        device const ushort* rr = RES + p.res_skip;
        device const ushort* wr = NW + p.nw_skip;
        float sb[NIT];
        float ss = 0.0f;
        for (int j = 0; j < NIT; ++j) {
            const uint i = tid + (uint)j * lanes;
            if (i < h) {
                sb[j] = bf2f(f2bf(bf2f(xr[i]) + bf2f(rr[i])));
                ss = fma(sb[j], sb[j], ss);
            }
        }
        ss = simd_sum(ss);
        if (lane == 0) npart[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint j = 0; j < SG; ++j) tot += npart[j];
        const float r = 1.0f / precise::sqrt(tot / (float)h + p.eps);
        threadgroup ushort* xs16 = (threadgroup ushort*)xt;
        for (int j = 0; j < NIT; ++j) {
            const uint i = tid + (uint)j * lanes;
            if (i < h) {
                xs16[i] = f2bf(sb[j] * r * (bf2f(wr[i]) + p.w_off));
                if (tg == 0) SUM[p.sum_skip + i] = f2bf(sb[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float ag = 0.0f, au = 0.0f;
    uint i = tid;
    for (uint b = 0; b < nblk; ++b, i += block) {
        V xv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) xv[u] = NORM ? xs[i + lanes * u] : xd[i + lanes * u];
        for (int u = 0; u < UNROLL; ++u) ag += gdot(gw[u], xv[u]);
        for (int u = 0; u < UNROLL; ++u) au += gdot(uw[u], xv[u]);
        if (b + 1 < nblk) {
            for (int u = 0; u < UNROLL; ++u) {
                gw[u] = wg[i + block + lanes * u];
                uw[u] = wu[i + block + lanes * u];
            }
        }
    }
    for (i = nblk * block + tid; i < kv; i += lanes) {
        const V xv = NORM ? xs[i] : xd[i];
        ag += gdot(wg[i], xv);
        au += gdot(wu[i], xv);
    }

    const float sgg = simd_sum(ag);
    const float sgu = simd_sum(au);
    if (lane == 0) {
        part[sg] = sgg;
        part[SG + sg] = sgu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tgg = part[0], tgu = part[SG];
        for (uint j = 1; j < SG; ++j) {
            tgg += part[j];
            tgu += part[SG + j];
        }
        // g and u as the unfused tensors hold them (bf16), then silu_glu_bf16's expression
        const float gv = bf2f(g_bf16(tgg));
        const float uv = bf2f(g_bf16(tgu));
        Y[p.y_skip + tg] = f2bf(gv / (1.0f + precise::exp(-gv)) * uv);
    }
}

#define INST_GLU(NAME, V, VEL, UNROLL, NORM)                                                                          \
    template [[host_name(NAME)]] kernel void gemv_glu_t<V, VEL, UNROLL, NORM>(                                        \
        device const V*, device const V*, device const V*, device const ushort*, device const ushort*,               \
        device ushort*, device ushort*, constant GluP&, uint, uint, uint, uint);

INST_GLU("gemv_glu_v16", uint4, 8, 2, false)
INST_GLU("gemv_glu_v8", uint2, 4, 4, false)
INST_GLU("gemv_glu_v2", ushort, 1, 8, false)
INST_GLU("gemv_glu_norm_v16", uint4, 8, 2, true)
INST_GLU("gemv_glu_norm_v8", uint2, 4, 4, true)
INST_GLU("gemv_glu_norm_v2", ushort, 1, 8, true)

// ---- RmsNormAddMatmulQkv: the attention input norm and the q, k, v projections in one dispatch --------------------------
// The unfused decode sequence is `rmsnorm_add_bf16_ex*` (normed, sum), `gemv_bf16_rt_*` (q) and `gemv_pair_bf16_rt_*` (k, v):
// three dispatches. One threadgroup per output row of the concatenation [q; k; v]; each recomputes the (bitwise identical)
// norm prologue of the GLU kernel above and keeps the normed bf16 row in threadgroup memory. Every output row keeps the
// `gemv_bf16_rt<2, 8>` / `gemv_bf16_rt_a8<4, 8>` accumulation order (see the top of this file). Threadgroup 0 stores `sum`.
// bind: 0 Wq, 1 Wk, 2 Wv, 3 X (blk), 4 RES, 5 NW, 6 Yq, 7 Yk, 8 Yv, 9 SUM, 10 params.
struct QkvP {
    uint k;
    uint nq;
    uint nk;
    uint nv;
    uint wq_skip;
    uint wk_skip;
    uint wv_skip;
    uint x_skip;
    uint res_skip;
    uint nw_skip;
    uint yq_skip;
    uint yk_skip;
    uint yv_skip;
    uint sum_skip;
    float eps;
    float w_off;
};

template <typename V, int VEL, int UNROLL>
kernel void gemv_qkv_norm_t(device const V* Wq [[buffer(0)]],
                            device const V* Wk [[buffer(1)]],
                            device const V* Wv [[buffer(2)]],
                            device const V* X [[buffer(3)]],
                            device const ushort* RES [[buffer(4)]],
                            device const ushort* NW [[buffer(5)]],
                            device ushort* Yq [[buffer(6)]],
                            device ushort* Yk [[buffer(7)]],
                            device ushort* Yv [[buffer(8)]],
                            device ushort* SUM [[buffer(9)]],
                            constant QkvP& p [[buffer(10)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint SG = 8;
    constexpr uint lanes = 256;
    constexpr uint block = lanes * UNROLL;
    threadgroup float part[SG];
    threadgroup float npart[SG];
    threadgroup uint4 xt[512];

    const uint kv = p.k / VEL;
    const bool isq = tg < p.nq;
    const bool isk = !isq && tg < p.nq + p.nk;
    device const V* W = isq ? g_adv(Wq, p.wq_skip) : (isk ? g_adv(Wk, p.wk_skip) : g_adv(Wv, p.wv_skip));
    const uint row = isq ? tg : (isk ? tg - p.nq : tg - p.nq - p.nk);
    device ushort* Y = (isq ? Yq + p.yq_skip : (isk ? Yk + p.yk_skip : Yv + p.yv_skip)) + row;
    device const V* wr = W + (ulong)row * kv;
    const uint nblk = kv / block;

    V gw[UNROLL];
    for (int u = 0; u < UNROLL; ++u) gw[u] = V(0);
    if (nblk > 0) {
        for (int u = 0; u < UNROLL; ++u) gw[u] = wr[tid + lanes * u];
    }

    constexpr int NIT = 16;
    const uint h = p.k;
    device const ushort* xr = (device const ushort*)X + p.x_skip;
    device const ushort* rr = RES + p.res_skip;
    device const ushort* wn = NW + p.nw_skip;
    float sb[NIT];
    float ss = 0.0f;
    for (int j = 0; j < NIT; ++j) {
        const uint i = tid + (uint)j * lanes;
        if (i < h) {
            sb[j] = bf2f(f2bf(bf2f(xr[i]) + bf2f(rr[i])));
            ss = fma(sb[j], sb[j], ss);
        }
    }
    ss = simd_sum(ss);
    if (lane == 0) npart[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint j = 0; j < SG; ++j) tot += npart[j];
    const float r = 1.0f / precise::sqrt(tot / (float)h + p.eps);
    threadgroup ushort* xs16 = (threadgroup ushort*)xt;
    for (int j = 0; j < NIT; ++j) {
        const uint i = tid + (uint)j * lanes;
        if (i < h) {
            xs16[i] = f2bf(sb[j] * r * (bf2f(wn[i]) + p.w_off));
            if (tg == 0) SUM[p.sum_skip + i] = f2bf(sb[j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const V* xs = (threadgroup const V*)xt;
    float acc = 0.0f;
    uint i = tid;
    for (uint b = 0; b < nblk; ++b, i += block) {
        V xv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) xv[u] = xs[i + lanes * u];
        for (int u = 0; u < UNROLL; ++u) acc += gdot(gw[u], xv[u]);
        if (b + 1 < nblk) {
            for (int u = 0; u < UNROLL; ++u) gw[u] = wr[i + block + lanes * u];
        }
    }
    for (i = nblk * block + tid; i < kv; i += lanes) acc += gdot(wr[i], xs[i]);

    const float s = simd_sum(acc);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = part[0];
        for (uint j = 1; j < SG; ++j) t += part[j];
        *Y = g_bf16(t);
    }
}

#define INST_QKV(NAME, V, VEL, UNROLL)                                                                                \
    template [[host_name(NAME)]] kernel void gemv_qkv_norm_t<V, VEL, UNROLL>(                                         \
        device const V*, device const V*, device const V*, device const V*, device const ushort*,                    \
        device const ushort*, device ushort*, device ushort*, device ushort*, device ushort*, constant QkvP&,        \
        uint, uint, uint, uint);

INST_QKV("gemv_qkv_norm_v16", uint4, 8, 2)
INST_QKV("gemv_qkv_norm_v8", uint2, 4, 4)
INST_QKV("gemv_qkv_norm_v2", ushort, 1, 8)

// ---- MatmulGlu, 512-thread form: the gate row on simdgroups 0..7 and the up row on 8..15 --------------------------------
// Each half is exactly the 256-thread row kernel of the unfused sequence (same lanes, steps and simd_sum / serial order),
// so the per-thread load structure and register footprint are those of `gemv_pair_bf16_rt_*` and the two partial sums meet
// in threadgroup memory for the GLU epilogue. bind as `gemv_glu_t` (X is the GEMV x, the norm buffers are unused).
template <typename V, int VEL, int UNROLL>
kernel void gemv_glu2_t(device const V* Wg [[buffer(0)]],
                        device const V* Wu [[buffer(1)]],
                        device const V* X [[buffer(2)]],
                        device const ushort* RES [[buffer(3)]],
                        device const ushort* NW [[buffer(4)]],
                        device ushort* Y [[buffer(5)]],
                        device ushort* SUM [[buffer(6)]],
                        constant GluP& p [[buffer(7)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint lanes = 256;
    constexpr uint block = lanes * UNROLL;
    threadgroup float part[16];
    const bool up = tid >= lanes;
    const uint t = tid & (lanes - 1);
    const uint kv = p.k / VEL;
    device const V* wr = (up ? g_adv(Wu, p.wu_skip) : g_adv(Wg, p.wg_skip)) + (ulong)tg * kv;
    device const V* xd = g_adv(X, p.x_skip);
    const uint nblk = kv / block;
    float acc = 0.0f;
    uint i = t;
    for (uint b = 0; b < nblk; ++b, i += block) {
        V xv[UNROLL];
        V wv[UNROLL];
        for (int u = 0; u < UNROLL; ++u) xv[u] = xd[i + lanes * u];
        for (int u = 0; u < UNROLL; ++u) wv[u] = wr[i + lanes * u];
        for (int u = 0; u < UNROLL; ++u) acc += gdot(wv[u], xv[u]);
    }
    for (i = nblk * block + t; i < kv; i += lanes) acc += gdot(wr[i], xd[i]);
    const float s = simd_sum(acc);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tgg = part[0], tgu = part[8];
        for (uint j = 1; j < 8; ++j) {
            tgg += part[j];
            tgu += part[8 + j];
        }
        const float gv = bf2f(g_bf16(tgg));
        const float uv = bf2f(g_bf16(tgu));
        Y[p.y_skip + tg] = f2bf(gv / (1.0f + precise::exp(-gv)) * uv);
    }
}

#define INST_GLU2(NAME, V, VEL, UNROLL)                                                                               \
    template [[host_name(NAME)]] kernel void gemv_glu2_t<V, VEL, UNROLL>(                                             \
        device const V*, device const V*, device const V*, device const ushort*, device const ushort*,               \
        device ushort*, device ushort*, constant GluP&, uint, uint, uint, uint);

INST_GLU2("gemv_glu2_v16", uint4, 8, 2)
INST_GLU2("gemv_glu2_v8", uint2, 4, 4)
INST_GLU2("gemv_glu2_v2", ushort, 1, 8)
