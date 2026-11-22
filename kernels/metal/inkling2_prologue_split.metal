// InkKvPrologue of prefill chunks as two kernels (lane ink-prefill; launcher src/inkprefill.rs, hooked from inkattn.rs
// `launch_ink_prologue` when INTARSIA_INK_PROLOGUE_SPLIT=1). Bitwise `ink_kv_prologue_bf16` of inkling2_attn.metal:
//   ink_kv_prologue_norel_bf16  the prologue without the rel product: K / V conv, k clip, v unit RMS, q scale + clip (+ tau),
//                               the conv-ring write. Grid (Hk, T), threads a multiple of 32 >= D (no r lanes needed).
//   ink_rel_proj_bf16           rel = r . relproj for a block of PS_RB rows of one kv head: r recomputed with the prologue's
//                               arithmetic (lanes 0..dr-1 of one simdgroup, simd_sum of the squares, rel clip, tau, rA), the
//                               d_rel x 4 relproj words of a column quad loaded once per block instead of once per row, the
//                               same fma chain per element (j ascending from 0). Grid (Hk, ceil(T / PS_RB)), 128 threads.
// Same binding table and InkP as `ink_kv_prologue_bf16` (both kernels take the full table; each reads what it needs).
#include "common.h"

constant constexpr float PS_SRC_EPS = 1e-6f;
constant constexpr uint PS_RB = 8;   // rows per rel threadgroup

// identical layout to `InkP` of inkling2_attn.metal / inkattn.rs
struct PsP {
    uint t, h, hk, d;
    uint dr, e, k, rr;
    uint cr, pos0, use_dev, flags;
    uint tau_floor, window, nq, pw;
    float qk_clip, rel_clip, eps, tau_alpha;
    float scale, cscale, f2, f3;
    uint sk[16];
};

static inline float ps_rA(float x) { return bf2f(f2bf(x)); }
static inline uint ps_pos0(constant PsP& p, device const int* pos) { return p.use_dev ? (uint)max(pos[0], 0) : p.pos0; }
static inline float ps_clipf(float ss, float th) { return min(precise::divide(th, precise::sqrt(ss) + PS_SRC_EPS), 1.0f); }
static inline float ps_tau(constant PsP& p, uint pp) {
    if (p.tau_floor == 0) return 1.0f;
    const float r = max(precise::divide((float)(pp + 1), (float)p.tau_floor), 1.0f);
    return 1.0f + p.tau_alpha * precise::log(r);
}
static inline float ps_cx(device const ushort* x, device const ushort* ring, long q, uint pos0, uint pw, uint rr, uint ckv, uint ch) {
    if (q < 0) return 0.0f;
    if ((uint)q >= pos0) return bf2f(x[(ulong)((uint)q - pos0) * pw + ch]);
    return bf2f(ring[(ulong)((uint)q % rr) * ckv + ch]);
}
static inline float ps_conv(device const ushort* x, device const ushort* ring, device const ushort* taps, uint t, uint pos0,
                            constant PsP& p, uint ch, uint kvh) {
    const uint ckv = p.hk * p.d;
    const long pp = (long)(pos0 + t);
    const bool dyn = (p.flags & 1u) != 0;
    float acc = 0.0f;
    for (uint l = 0; l < p.k; ++l) {
        float w = bf2f(taps[(ulong)l * ckv + ch]);
        if (dyn) w += bf2f(x[(ulong)t * p.pw + ckv + kvh * p.k + l]);
        acc = fma(w, ps_cx(x, ring, pp - (long)l, pos0, p.pw, p.rr, ckv, ch), acc);
    }
    return ps_rA((acc + bf2f(x[(ulong)t * p.pw + ch])) * p.cscale);
}

// bind: 0 pq, 1 pk, 2 pv, 3 k taps, 4 v taps, 5 q scale, 6 r scale, 7 rel proj, 8 q, 9 rel, 10 k, 11 v, 12 k ring, 13 v ring,
// 14 pos, 15 PsP.
kernel void ink_kv_prologue_norel_bf16(device const ushort* pq_ [[buffer(0)]],
                                       device const ushort* pk_ [[buffer(1)]],
                                       device const ushort* pv_ [[buffer(2)]],
                                       device const ushort* kt_ [[buffer(3)]],
                                       device const ushort* vt_ [[buffer(4)]],
                                       device const ushort* qs_ [[buffer(5)]],
                                       device const ushort* rs_ [[buffer(6)]],
                                       device const ushort* rp_ [[buffer(7)]],
                                       device ushort* qo_ [[buffer(8)]],
                                       device ushort* relo_ [[buffer(9)]],
                                       device ushort* ko_ [[buffer(10)]],
                                       device ushort* vo_ [[buffer(11)]],
                                       device ushort* kr_ [[buffer(12)]],
                                       device ushort* vr_ [[buffer(13)]],
                                       device const int* pos [[buffer(14)]],
                                       constant PsP& p [[buffer(15)]],
                                       uint3 tg [[threadgroup_position_in_grid]],
                                       uint3 tid3 [[thread_position_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[8][10];
    const uint tid = tid3.x;
    device const ushort* pq = pq_ + p.sk[0];
    device const ushort* pk = pk_ + p.sk[1];
    device const ushort* pv = pv_ + p.sk[2];
    device const ushort* kt = kt_ + p.sk[3];
    device const ushort* vt = vt_ + p.sk[4];
    device const ushort* qs = qs_ + p.sk[5];
    device ushort* qo = qo_ + p.sk[8];
    device ushort* ko = ko_ + p.sk[10];
    device ushort* vo = vo_ + p.sk[11];
    device ushort* kr = kr_ + p.sk[12];
    device ushort* vr = vr_ + p.sk[13];

    const uint kvh = tg.x, t = tg.y;
    const uint D = p.d, H = p.h, G = p.h / p.hk;
    const uint ckv = p.hk * D;
    const uint pos0 = ps_pos0(p, pos);
    const uint pp = pos0 + t;
    const uint c = tid, ch = kvh * D + c;
    const ulong qrow = (ulong)t * p.nq;

    // vals as in ink_kv_prologue_bf16 (the r entries 2 + G .. stay out: their sums are separate)
    float vals[6];
    for (uint i = 0; i < 6; ++i) vals[i] = 0.0f;
    float kc = 0.0f, vc = 0.0f;
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (c < D) {
        kc = ps_conv(pk, kr, kt, t, pos0, p, ch, kvh);
        vc = ps_conv(pv, vr, vt, t, pos0, p, ch, kvh);
        vals[0] = kc * kc;
        vals[1] = vc * vc;
        for (uint g = 0; g < G; ++g) {
            const uint h = kvh * G + g;
            qv[g] = bf2f(pq[qrow + h * D + c]) * bf2f(qs[h * D + c]);
            vals[2 + g] = qv[g] * qv[g];
        }
    }
    const uint nv = 2 + G;
    for (uint i = 0; i < nv; ++i) {
        const float s = simd_sum(vals[i]);
        if (lane == 0) part[sg][i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[6];
    for (uint i = 0; i < nv; ++i) {
        float s = 0.0f;
        for (uint j = 0; j < nsg; ++j) s += part[j][i];
        tot[i] = s;
    }
    const float tau = ps_tau(p, pp);
    const bool use_tau = p.tau_floor > 0;
    if (c < D) {
        const ulong kvo = ((ulong)t * p.hk + kvh) * D + c;
        ko[kvo] = f2bf(kc * ps_clipf(tot[0], p.qk_clip));
        const float inv = precise::divide(1.0f, precise::sqrt(tot[1] / (float)D + p.eps));
        vo[kvo] = f2bf(vc * inv);
        for (uint g = 0; g < G; ++g) {
            const uint h = kvh * G + g;
            float x = ps_rA(qv[g] * ps_clipf(tot[2 + g], p.qk_clip));
            if (use_tau) x = ps_rA(x * tau);
            qo[((ulong)t * H + h) * D + c] = f2bf(x);
        }
    }
    if ((p.flags & 4u) != 0 && c < D && t + p.rr >= p.t) {
        const ulong ro = (ulong)(pp % p.rr) * ckv + ch;
        kr[ro] = pk[(ulong)t * p.pw + ch];
        vr[ro] = pv[(ulong)t * p.pw + ch];
    }
}

kernel void ink_rel_proj_bf16(device const ushort* pq_ [[buffer(0)]],
                              device const ushort* pk_ [[buffer(1)]],
                              device const ushort* pv_ [[buffer(2)]],
                              device const ushort* kt_ [[buffer(3)]],
                              device const ushort* vt_ [[buffer(4)]],
                              device const ushort* qs_ [[buffer(5)]],
                              device const ushort* rs_ [[buffer(6)]],
                              device const ushort* rp_ [[buffer(7)]],
                              device ushort* qo_ [[buffer(8)]],
                              device ushort* relo_ [[buffer(9)]],
                              device ushort* ko_ [[buffer(10)]],
                              device ushort* vo_ [[buffer(11)]],
                              device ushort* kr_ [[buffer(12)]],
                              device ushort* vr_ [[buffer(13)]],
                              device const int* pos [[buffer(14)]],
                              constant PsP& p [[buffer(15)]],
                              uint3 tg [[threadgroup_position_in_grid]],
                              uint3 tid3 [[thread_position_in_threadgroup]],
                              uint3 tpg3 [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float rsh[PS_RB][4][32];
    const uint tid = tid3.x, tpg = tpg3.x;
    device const ushort* pq = pq_ + p.sk[0];
    device const ushort* rs = rs_ + p.sk[6];
    device const ushort* rp = rp_ + p.sk[7];
    device ushort* relo = relo_ + p.sk[9];

    const uint kvh = tg.x, t0 = tg.y * PS_RB;
    const uint D = p.d, H = p.h, G = p.h / p.hk, dr = p.dr, E = p.e;
    const uint pos0 = ps_pos0(p, pos);
    const bool use_tau = p.tau_floor > 0;
    const uint nr = min(PS_RB, p.t - t0);

    // r of the block's rows: simdgroup sg takes rows sg, sg + nsg, ...; lane c < dr holds channel c (as the prologue's
    // simdgroup 0, whose lanes c >= dr add zeros to the same simd_sum)
    for (uint i = sg; i < nr; i += nsg) {
        const uint t = t0 + i;
        const ulong qrow = (ulong)t * p.nq;
        const float tau = ps_tau(p, pos0 + t);
        for (uint g = 0; g < G; ++g) {
            const uint h = kvh * G + g;
            float rv = 0.0f;
            if (lane < dr) rv = bf2f(pq[qrow + H * D + h * dr + lane]) * bf2f(rs[h * dr + lane]);
            const float ss = simd_sum(rv * rv);
            float s = 0.0f;
            s += ss;
            if (lane < dr) {
                float x = ps_rA(rv * ps_clipf(s, p.rel_clip));
                if (use_tau) x = ps_rA(x * tau);
                rsh[i][g][lane] = x;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool vec4 = (E % 4) == 0 && (p.flags & 8u) != 0 && dr == 16;
    const uint e4n = vec4 ? E / 4 : 0;
    for (uint e4 = tid; e4 < e4n; e4 += tpg) {
        uint2 w[16];
#pragma unroll
        for (uint j = 0; j < 16; ++j) w[j] = *(device const uint2*)(rp + (ulong)j * E + e4 * 4);
        for (uint i = 0; i < nr; ++i) {
            const uint t = t0 + i;
            for (uint g = 0; g < G; ++g) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint j = 0; j < 16; ++j) {
                    const float r = rsh[i][g][j];
                    a0 = fma(r, as_type<float>(w[j].x << 16), a0);
                    a1 = fma(r, as_type<float>(w[j].x & 0xffff0000u), a1);
                    a2 = fma(r, as_type<float>(w[j].y << 16), a2);
                    a3 = fma(r, as_type<float>(w[j].y & 0xffff0000u), a3);
                }
                device uint2* dst = (device uint2*)(relo + ((ulong)t * H + kvh * G + g) * E + e4 * 4);
                *dst = uint2((uint)f2bf(a0) | ((uint)f2bf(a1) << 16), (uint)f2bf(a2) | ((uint)f2bf(a3) << 16));
            }
        }
    }
    for (uint e = e4n * 4 + tid; e < E; e += tpg) {
        for (uint i = 0; i < nr; ++i) {
            const uint t = t0 + i;
            for (uint g = 0; g < G; ++g) {
                float a = 0.0f;
                for (uint j = 0; j < dr; ++j) a = fma(rsh[i][g][j], bf2f(rp[(ulong)j * E + e]), a);
                relo[((ulong)t * H + kvh * G + g) * E + e] = f2bf(a);
            }
        }
    }
}
