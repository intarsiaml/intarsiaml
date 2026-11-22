// Inkling-2-Mini depth-mix, conv and input-projection ops on Metal (lane ink-m-mix). Contract:
// crates/intarsia-core/src/inkling2/ops.rs (module doc); f64 reference crates/intarsia-backend-cpu/src/ink_ref.rs; launchers
// crates/intarsia-backend-metal/src/inkmix.rs.
//
// Rows-exact (G0): every kernel here computes output row t with a fixed per-row reduction order that does not depend on the
// number of rows T in the call. The multi-row InkInProj GEMV keeps the lane -> chunk mapping and the reduction tree of its
// R = 1 instance (R only adds independent accumulators, P only shares an x load between weight rows), so row r of an R-row
// launch is bitwise the 1-row launch of row r. The row kernels (conv / mix / blend) run one threadgroup per row with the
// same code for every T.
#include "common.h"

// ================================================================================================================
// InkInProj: y_j = x[min(j, S-1)] . W_j^T for the three stacked weight segments, bf16, one dispatch.
// ================================================================================================================

struct InkProjP {
    uint k;              // D (K % 8 == 0)
    uint n0, n1, n2;     // rows per segment (pq, pk, pv)
    uint xo1, xo2;       // element offset of the x block of segments 1 and 2 (min(j, S-1) * T * D)
    uint ys0, ys1, ys2;  // leading bf16 elements to skip in Y0 / Y1 / Y2 (4-byte aligned binding)
    uint g0, g1;         // threadgroups of segment 0 and 1 (ceil(n_j / P))
    uint gtot;           // all virtual threadgroups (g0 + g1 + g2); the grid may be smaller and loop over them
};

inline float ip_lo(uint u) { return as_type<float>(u << 16); }
inline float ip_hi(uint u) { return as_type<float>(u & 0xffff0000u); }

// The fma chain of gemv_multirow.metal dot8.
inline float ip_dot8(uint4 w, uint4 x) {
    float a = ip_lo(w.x) * ip_lo(x.x);
    a = fma(ip_hi(w.x), ip_hi(x.x), a);
    a = fma(ip_lo(w.y), ip_lo(x.y), a);
    a = fma(ip_hi(w.y), ip_hi(x.y), a);
    a = fma(ip_lo(w.z), ip_lo(x.z), a);
    a = fma(ip_hi(w.z), ip_hi(x.z), a);
    a = fma(ip_lo(w.w), ip_lo(x.w), a);
    a = fma(ip_hi(w.w), ip_hi(x.w), a);
    return a;
}

// P weight rows (row0.., clamped to nvalid - 1 for the loads) x RB activation rows (X, row stride k8). Lane L of the SG*32
// lanes reads chunk (b * U + u) * lanes + L of each row (16 bytes), then a K tail of one chunk per lane per step. Per
// (weight row, activation row): one accumulator in that chunk order, simd_sum, then the SG partials summed in sg order.
// `part` holds P * RB * SG floats; the totals land in out[] on every thread.
template <int RB, int P, int U, int SG>
inline void ip_rows(device const uint4* W, uint row0, uint nvalid, device const uint4* X, uint k8, uint sg, uint lane,
                    threadgroup float* part, thread float* out) {
    constexpr uint lanes = 32 * SG;
    constexpr uint block = lanes * U;
    device const uint4* wr[P];
    for (int q = 0; q < P; ++q) wr[q] = W + (ulong)min(row0 + q, nvalid - 1) * k8;
    const uint nblk = k8 / block;
    uint i = sg * 32 + lane;
    float acc[P * RB];
    for (int j = 0; j < P * RB; ++j) acc[j] = 0.0f;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 wv[P][U];
        for (int q = 0; q < P; ++q)
            for (int u = 0; u < U; ++u) wv[q][u] = wr[q][i + lanes * u];
        for (int r = 0; r < RB; ++r) {
            uint4 xv[U];
            for (int u = 0; u < U; ++u) xv[u] = X[(ulong)r * k8 + i + lanes * u];
            for (int q = 0; q < P; ++q)
                for (int u = 0; u < U; ++u) acc[q * RB + r] += ip_dot8(wv[q][u], xv[u]);
        }
    }
    for (i = nblk * block + sg * 32 + lane; i < k8; i += lanes) {
        uint4 wv[P];
        for (int q = 0; q < P; ++q) wv[q] = wr[q][i];
        for (int r = 0; r < RB; ++r) {
            const uint4 xv = X[(ulong)r * k8 + i];
            for (int q = 0; q < P; ++q) acc[q * RB + r] += ip_dot8(wv[q], xv);
        }
    }
    if (SG == 1) {
        for (int j = 0; j < P * RB; ++j) out[j] = simd_sum(acc[j]);
        return;
    }
    for (int j = 0; j < P * RB; ++j) {
        const float s = simd_sum(acc[j]);
        if (lane == 0) part[j * SG + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < P * RB; ++j) {
        float t = part[j * SG];
        for (int q = 1; q < SG; ++q) t += part[j * SG + q];
        out[j] = t;
    }
}

// bind: 0 W [n0+n1+n2, K] (16-byte aligned), 1 X [S, T, K] (16-byte aligned), 2 Y0 [R, n0], 3 Y1 [R, n1], 4 Y2 [R, n2],
// 5 InkProjP. Grid: g0 + g1 + ceil(n2 / P) threadgroups of SG * 32 threads. R rows of x per launch, in passes of RB.
template <int R, int P, int SG, int U, int RB>
kernel void ink_in_proj_gemv(device const uint4* W [[buffer(0)]],
                             device const uint4* X [[buffer(1)]],
                             device ushort* Y0 [[buffer(2)]],
                             device ushort* Y1 [[buffer(3)]],
                             device ushort* Y2 [[buffer(4)]],
                             constant InkProjP& p [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint ngrid [[threadgroups_per_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    static_assert(R % RB == 0, "R must be a multiple of RB");
    threadgroup float part[P * R * SG];
    const uint k8 = p.k / 8;
    // grid-stride over the virtual threadgroups: every in-flight threadgroup works on neighbouring rows (the bits of a
    // row do not depend on which threadgroup computes it)
    for (uint vt = tg; vt < p.gtot; vt += ngrid) {
        uint lt, n, wrow, xo, ys;
        device ushort* Y;
        if (vt < p.g0) {
            lt = vt; n = p.n0; wrow = 0; xo = 0; Y = Y0; ys = p.ys0;
        } else if (vt < p.g0 + p.g1) {
            lt = vt - p.g0; n = p.n1; wrow = p.n0; xo = p.xo1; Y = Y1; ys = p.ys1;
        } else {
            lt = vt - p.g0 - p.g1; n = p.n2; wrow = p.n0 + p.n1; xo = p.xo2; Y = Y2; ys = p.ys2;
        }
        const uint row0 = lt * P;
        device const uint4* Ws = W + (ulong)wrow * k8;
        device const uint4* Xs = X + xo / 8;
        float t[P * R];
        for (int rb = 0; rb < R; rb += RB) {
            float tb[P * RB];
            ip_rows<RB, P, U, SG>(Ws, row0, n, Xs + (ulong)rb * k8, k8, sg, lane, part + rb * P * SG, tb);
            for (int q = 0; q < P; ++q)
                for (int r = 0; r < RB; ++r) t[q * R + rb + r] = tb[q * RB + r];
        }
        if (sg == 0 && lane == 0) {
            for (int q = 0; q < P; ++q)
                if (row0 + q < n)
                    for (int r = 0; r < R; ++r) Y[ys + (ulong)r * n + row0 + q] = f2bf(t[q * R + r]);
        }
        if (SG > 1) threadgroup_barrier(mem_flags::mem_threadgroup);  // `part` is rewritten by the next row
    }
}

#define IP_ARGS device const uint4*, device const uint4*, device ushort*, device ushort*, device ushort*, \
    constant InkProjP&, uint, uint, uint, uint
#define IP_INST(R, P, SG, U, RB)                                                                                      \
    template [[host_name("ink_in_proj_gemv_r" #R "p" #P "s" #SG "u" #U "b" #RB)]] kernel void                        \
    ink_in_proj_gemv<R, P, SG, U, RB>(IP_ARGS);

// The exploration set (benchmarked in docs/results/2026-11-21-inkling2-metal-mix.md); the launcher's table in inkmix.rs
// lists what it uses.
#define IP_CFG(SG, U)                                                                                                 \
    IP_INST(1, 1, SG, U, 1) IP_INST(1, 2, SG, U, 1) IP_INST(1, 4, SG, U, 1)                                           \
    IP_INST(2, 1, SG, U, 2) IP_INST(2, 2, SG, U, 2) IP_INST(3, 1, SG, U, 3) IP_INST(3, 2, SG, U, 3)                   \
    IP_INST(4, 1, SG, U, 4) IP_INST(4, 2, SG, U, 4) IP_INST(5, 1, SG, U, 5) IP_INST(6, 1, SG, U, 6)                   \
    IP_INST(7, 1, SG, U, 7) IP_INST(8, 1, SG, U, 8) IP_INST(8, 2, SG, U, 4) IP_INST(9, 1, SG, U, 9)                   \
    IP_INST(10, 1, SG, U, 10) IP_INST(11, 1, SG, U, 11) IP_INST(12, 1, SG, U, 12) IP_INST(13, 1, SG, U, 13)           \
    IP_INST(14, 1, SG, U, 14) IP_INST(15, 1, SG, U, 15) IP_INST(16, 1, SG, U, 16) IP_INST(16, 1, SG, U, 8)            \
    IP_INST(16, 2, SG, U, 8)

IP_CFG(8, 2)
IP_CFG(9, 1)
IP_CFG(3, 3)
IP_CFG(1, 9)

// ================================================================================================================
// Row kernels: SconvRes, ConvMix (conv + residual + depth mix + norm), AttnResMix. One threadgroup per row, TPG threads,
// thread `tid` owns elements tid + j * TPG (j < NE) of the D-wide row. bf16 activations, f32 math.
// ================================================================================================================

constant uint MX_CONV = 1u;   // conv present (SconvRes / ConvMix); else AttnResMix (binding 0 is `last`)
constant uint MX_RES = 2u;
constant uint MX_ADD = 4u;
constant uint MX_SUM = 8u;
constant uint MX_NORM = 16u;  // mix output normed (bf16), else the f32 mix
constant uint MX_RING = 32u;  // write the ring rows of this launch's rows here (no row of the launch reads a row written)
constant uint MX_PAIR = 64u;
// row-table form (lane ink-batch, `OpCall::rows`): row t is the T = 1 conv of sequence t at posb[t], on the ring whose address
// is table entry [t, rti] (+ roff bytes); no in-call history, the ring row always written here
constant uint MX_ROWTAB = 128u;  // blend path: also write the controller pair rA([d w(d), m w(m)]) of each stream (bf16 scratch)

constant float SRC_EPS = 1e-6f;
constant int MX_NMAX = 8;     // stored sources + last
constant int MX_QB = 4;       // queries per reduction batch

struct InkMixP {
    uint d, t, t0, pad0;        // D, rows T of the call, first row of this launch (grid = rows of the launch)
    uint k, rr, pos0, use_dev;  // taps K, ring rows, host position, position from the device buffer
    uint n, q, flags, orow0;    // stored sources N, queries Q, MX_*, output row base (out row = t - orow0)
    uint ocount, nstream, pad2, pad3; // output rows per query block (row stride of the [Q, rows, D] output); blend streams S
    float eps, cscale, fpad0, fpad1;
    // leading bf16 elements per binding: x, taps, res, add, y, ring, sum, src, norm, out
    uint sk[12];
    uint rts, rti, tab_lo, tab_hi;  // MX_ROWTAB: table stride, entry, the table's GPU address
    uint roff, rpad0, rpad1, rpad2; // MX_ROWTAB: the ring carrier's byte offset
};

// v[j] (j < nr) summed over the threadgroup: simd_sum, then the simdgroup partials in sg order. All threads get red[].
inline void mx_reduce(thread float* v, int nr, threadgroup float* part, threadgroup float* red, uint tid, uint lane,
                      uint sg, uint nsg) {
    for (int j = 0; j < nr; ++j) {
        const float s = simd_sum(v[j]);
        if (lane == 0) part[j * 32 + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)nr) {
        float t = 0.0f;
        for (uint s = 0; s < nsg; ++s) t += part[tid * 32 + s];
        red[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The same with the count known at compile time (the thread array stays in registers). `part` holds NR * 32 floats, `red` NR.
template <int NR>
inline void mx_reduce_t(thread float (&v)[NR], threadgroup float* part, threadgroup float* red, uint tid, uint lane, uint sg,
                        uint nsg) {
    for (int j = 0; j < NR; ++j) {
        const float s = simd_sum(v[j]);
        if (lane == 0) part[j * 32 + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(NR)) {
        float t = 0.0f;
        for (uint s = 0; s < nsg; ++s) t += part[tid * 32 + s];
        red[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float rbf(float v) { return bf2f(f2bf(v)); }

// bind: 0 x [T,C] (ConvMix / SconvRes) or last [R,D] (AttnResMix), 1 taps [K,C], 2 res, 3 addend, 4 y, 5 ring [Rr,C] (rw),
// 6 sum, 7 src [N,T,D], 8 queries [Q,D] f32, 9 norm [Q,D], 10 out bf16, 11 out f32, 12 pos i32, 13 InkMixP.
// NE elements per thread, NS = N + 1 sources (stored + last), QB queries per reduction batch: all compile-time, so the
// per-thread arrays stay in registers and every load of a row is issued unconditionally (clamped addresses, masked values)
// before the first use (the loads of one thread overlap instead of serializing behind branches).
template <int NE, int NS, int QB>
kernel void ink_mix_row(device const ushort* x [[buffer(0)]],
                        device const ushort* taps [[buffer(1)]],
                        device const ushort* res [[buffer(2)]],
                        device const ushort* add [[buffer(3)]],
                        device ushort* y [[buffer(4)]],
                        device ushort* ring [[buffer(5)]],
                        device ushort* sum [[buffer(6)]],
                        device const ushort* src [[buffer(7)]],
                        device const float* qry [[buffer(8)]],
                        device const ushort* nrm [[buffer(9)]],
                        device ushort* outb [[buffer(10)]],
                        device float* outf [[buffer(11)]],
                        device const int* posb [[buffer(12)]],
                        device const float* msg [[buffer(13)]],
                        device ushort* pair [[buffer(14)]],
                        constant InkMixP& p [[buffer(15)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]]) {
    static_assert(NS <= MX_NMAX && QB <= MX_QB, "mix instance");
    constexpr int NV = NS + QB * NS;  // reduced values per batch: source squares, then the dots [qi][n]
    threadgroup float part[(MX_NMAX + MX_QB * MX_NMAX) * 32];
    threadgroup float red[32];
    const uint d = p.d;
    const uint t = p.t0 + row;
    const bool conv = (p.flags & MX_CONV) != 0;
    const bool hres = (p.flags & MX_RES) != 0, hadd = (p.flags & MX_ADD) != 0, hsum = (p.flags & MX_SUM) != 0;
    uint cc[NE];
    bool on[NE];
    for (int j = 0; j < NE; ++j) {
        const uint c = tid + uint(j) * tpg;
        on[j] = c < d;
        cc[j] = min(c, d - 1);
    }
    // the stored sources first (independent of the conv: their loads overlap it)
    float xs[NS][NE];
    for (int n = 0; n < NS - 1; ++n) {
        device const ushort* sr = src + p.sk[7] + ((ulong)n * p.t + t) * d;
        for (int j = 0; j < NE; ++j) xs[n][j] = on[j] ? bf2f(sr[cc[j]]) : 0.0f;
    }
    float last[NE];
    if (conv) {
        const bool rt = (p.flags & MX_ROWTAB) != 0;
        // the row-table form: row t is a T = 1 call at posb[t] (in-call row index 0, its x row at t)
        const int pos = rt ? posb[t] : p.use_dev ? posb[0] : int(p.pos0);
        const int pp = pos + (rt ? 0 : int(t));
        const int xoff = rt ? int(t) : 0;
        device const ushort* xr = x + p.sk[0];
        device const ushort* tr = taps + p.sk[1];
        device ushort* rg = ring + p.sk[5];
        if (rt) {
            device const ulong* tab = reinterpret_cast<device const ulong*>(as_type<ulong>(uint2(p.tab_lo, p.tab_hi)));
            rg = reinterpret_cast<device ushort*>(tab[(ulong)t * p.rts + p.rti] + p.roff);
        }
        float acc[NE], xc[NE], rv[NE], av[NE];
        for (int j = 0; j < NE; ++j) {
            acc[j] = 0.0f;
            xc[j] = bf2f(xr[(ulong)t * d + cc[j]]);
            rv[j] = hres ? bf2f(res[p.sk[2] + (ulong)t * d + cc[j]]) : 0.0f;
            av[j] = hadd ? bf2f(add[p.sk[3] + (ulong)t * d + cc[j]]) : 0.0f;
        }
        for (uint l = 0; l < p.k; ++l) {
            const int qq = pp - int(l);
            device const ushort* bp = qq >= pos ? xr + (ulong)(qq - pos + xoff) * d : rg + (ulong)(uint(max(qq, 0)) % p.rr) * d;
            device const ushort* tl = tr + (ulong)l * d;
            for (int j = 0; j < NE; ++j) {
                const float xv = qq >= 0 ? bf2f(bp[cc[j]]) : 0.0f;
                acc[j] = fma(bf2f(tl[cc[j]]), xv, acc[j]);
            }
        }
        for (int j = 0; j < NE; ++j) {
            const ulong o = (ulong)t * d + cc[j];
            const float yv = rbf((acc[j] + xc[j]) * p.cscale);
            float lv = yv;
            if (hres) lv = hsum ? rbf((rv[j] + av[j]) + yv) : rv[j] + yv;
            last[j] = on[j] ? lv : 0.0f;
            if (on[j]) {
                y[p.sk[4] + o] = f2bf(yv);
                if (hres && hsum) sum[p.sk[6] + o] = f2bf(lv);
            }
        }
        if ((p.flags & MX_RING) != 0 && (rt || int(t) >= int(p.t) - int(p.rr))) {
            device ushort* wr = rg + (ulong)(uint(pp) % p.rr) * d;
            for (int j = 0; j < NE; ++j)
                if (on[j]) wr[cc[j]] = xr[(ulong)t * d + cc[j]];
        }
    } else {
        for (int j = 0; j < NE; ++j) {
            float lv = bf2f(x[p.sk[0] + (ulong)t * d + cc[j]]);
            if (hadd) lv = lv + bf2f(add[p.sk[3] + (ulong)t * d + cc[j]]);
            last[j] = on[j] ? lv : 0.0f;
        }
    }
    if (p.q == 0) return;
    for (int j = 0; j < NE; ++j) xs[NS - 1][j] = last[j];

    // ---- depth mix over the N stored sources and `last` ----
    const bool norm = (p.flags & MX_NORM) != 0;
    for (uint q0 = 0; q0 < p.q; q0 += QB) {
        const int nq = int(min(uint(QB), p.q - q0));
        float v[NV];
        for (int n = 0; n < NS; ++n) {
            float a = 0.0f;
            for (int j = 0; j < NE; ++j) a = fma(xs[n][j], xs[n][j], a);
            v[n] = a;
        }
        for (int qi = 0; qi < QB; ++qi) {
            device const float* qr = qry + (ulong)min(q0 + uint(qi), p.q - 1) * d;
            float qv[NE];
            for (int j = 0; j < NE; ++j) qv[j] = (qi < nq && on[j]) ? qr[cc[j]] : 0.0f;
            for (int n = 0; n < NS; ++n) {
                float a = 0.0f;
                for (int j = 0; j < NE; ++j) a = fma(qv[j], xs[n][j], a);
                v[NS + qi * NS + n] = a;
            }
        }
        // partials per simdgroup, then simdgroup 0 finishes: lane L = (query qi = L / 8, source n = L % 8) sums its dot
        // and source n's squares over the simdgroups (in sg order), and the softmax over n runs across the 8 lanes of a
        // query (butterfly max / sum). The coefficients p[qi,n] * w[n] go to `red` for every thread.
        for (int j = 0; j < NV; ++j) {
            const float sj = simd_sum(v[j]);
            if (lane == 0) part[j * 32 + sg] = sj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const int qi = int(lane) / MX_NMAX, n = int(lane) % MX_NMAX;
            const bool ok = qi < nq && n < NS;
            const int jd = ok ? NS + qi * NS + n : 0, js = n < NS ? n : 0;
            float dt = 0.0f, sq = 0.0f;
            for (uint k = 0; k < nsg; ++k) {
                dt += part[jd * 32 + k];
                sq += part[js * 32 + k];
            }
            const float wn = precise::rsqrt(sq / float(d) + SRC_EPS);
            const float l = ok ? dt * wn : -INFINITY;
            float m = max(l, simd_shuffle_xor(l, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(2)));
            m = max(m, simd_shuffle_xor(m, ushort(4)));
            const float e = ok ? precise::exp(l - m) : 0.0f;
            float se = e + simd_shuffle_xor(e, ushort(1));
            se = se + simd_shuffle_xor(se, ushort(2));
            se = se + simd_shuffle_xor(se, ushort(4));
            red[lane] = ok ? e / se * wn : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mix[QB][NE];
        for (int qi = 0; qi < QB; ++qi) {
            float coef[NS];
            for (int n = 0; n < NS; ++n) coef[n] = red[qi * MX_NMAX + n];
            for (int j = 0; j < NE; ++j) {
                float a = 0.0f;
                for (int n = 0; n < NS; ++n) a = fma(coef[n], xs[n][j], a);
                mix[qi][j] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // red is reused below
        if (norm) {
            float ss[QB];
            for (int qi = 0; qi < QB; ++qi) {
                float a = 0.0f;
                for (int j = 0; j < NE; ++j) a = fma(mix[qi][j], mix[qi][j], a);
                ss[qi] = a;
            }
            mx_reduce_t(ss, part, red, tid, lane, sg, nsg);
            for (int qi = 0; qi < QB; ++qi) {
                if (qi >= nq) continue;
                const float inv = precise::rsqrt(red[qi] / float(d) + p.eps);
                const uint q = q0 + uint(qi);
                device const ushort* nr = nrm + p.sk[8] + (ulong)q * d;
                device ushort* orow = outb + p.sk[9] + ((ulong)q * p.ocount + (t - p.orow0)) * d;
                for (int j = 0; j < NE; ++j)
                    if (on[j]) orow[cc[j]] = f2bf(mix[qi][j] * inv * bf2f(nr[cc[j]]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);  // red is reused by the next batch
        } else {
            for (int qi = 0; qi < QB; ++qi) {
                if (qi >= nq) continue;
                device float* orow = outf + ((ulong)(q0 + uint(qi)) * p.ocount + (t - p.orow0)) * d;
                for (int j = 0; j < NE; ++j)
                    if (on[j]) orow[cc[j]] = mix[qi][j];
            }
            if ((p.flags & MX_PAIR) != 0) {
                // the controller pair of each stream: dest = mix[Q == 1 ? 0 : s] (unnormed, f32), m = msg[s, t]
                constexpr int SB = MX_QB;
                float mv[SB][NE];
                float v2[2 * SB];
                for (int si = 0; si < SB; ++si) {
                    device const float* mr = msg + ((ulong)min(uint(si), max(p.nstream, 1u) - 1) * p.t + t) * d;
                    float a = 0.0f, b = 0.0f;
                    const int qi = QB == 1 ? 0 : min(si, QB - 1);
                    for (int j = 0; j < NE; ++j) {
                        mv[si][j] = (uint(si) < p.nstream && on[j]) ? mr[cc[j]] : 0.0f;
                        a = fma(mix[qi][j], mix[qi][j], a);
                        b = fma(mv[si][j], mv[si][j], b);
                    }
                    v2[si] = a;
                    v2[SB + si] = b;
                }
                mx_reduce_t(v2, part, red, tid, lane, sg, nsg);
                for (int si = 0; si < SB; ++si) {
                    if (uint(si) >= p.nstream) continue;
                    const int qi = (QB == 1 || p.q == 1) ? 0 : min(si, QB - 1);
                    const float wd = precise::rsqrt(red[p.q == 1 ? 0 : si] / float(d) + SRC_EPS);
                    const float wm = precise::rsqrt(red[SB + si] / float(d) + SRC_EPS);
                    device ushort* pr = pair + ((ulong)si * p.ocount + (t - p.orow0)) * 2 * d;
                    for (int j = 0; j < NE; ++j) {
                        if (!on[j]) continue;
                        pr[cc[j]] = f2bf(mix[qi][j] * wd);
                        pr[d + cc[j]] = f2bf(mv[si][j] * wm);
                    }
                }
            }
        }
    }
}

#define MX_ARGS device const ushort*, device const ushort*, device const ushort*, device const ushort*, device ushort*,   \
    device ushort*, device ushort*, device const ushort*, device const float*, device const ushort*, device ushort*,     \
    device float*, device const int*, device const float*, device ushort*, constant InkMixP&, uint, uint, uint, uint, uint, uint
#define MX_INST(NE, NS, QB)                                                                                           \
    template [[host_name("ink_mix_row_e" #NE "n" #NS "q" #QB)]] kernel void ink_mix_row<NE, NS, QB>(MX_ARGS);
#define MX_NS(NE, QB) MX_INST(NE, 1, QB) MX_INST(NE, 2, QB) MX_INST(NE, 3, QB) MX_INST(NE, 4, QB) MX_INST(NE, 5, QB)     \
    MX_INST(NE, 6, QB) MX_INST(NE, 7, QB) MX_INST(NE, 8, QB)
MX_NS(1, 1) MX_NS(1, 4)
MX_NS(2, 1) MX_NS(2, 4)
MX_NS(3, 1) MX_NS(3, 4)
MX_NS(4, 1) MX_NS(4, 4)

// ---- deferred ring write: ring row (pos + t) % Rr = x[t] for the rows t in [max(t0, T - Rr), t0 + nrows) of a launch whose
// rows also read the ring. bind: 0 x [T,C], 1 ring [Rr,C], 2 pos i32, 3 InkMixP (d, t, t0, pad0 = nrows, rr, pos0, use_dev,
// sk[0] x, sk[5] ring). Grid: Threads(nrows * C).
kernel void ink_ring_write(device const ushort* x [[buffer(0)]],
                           device ushort* ring [[buffer(1)]],
                           device const int* posb [[buffer(2)]],
                           constant InkMixP& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.pad0 * p.d) return;
    const uint t = p.t0 + gid / p.d, c = gid % p.d;
    if (int(t) < int(p.t) - int(p.rr)) return;
    const int pos = p.use_dev ? posb[0] : int(p.pos0);
    ring[p.sk[5] + (ulong)(uint(pos + int(t)) % p.rr) * p.d + c] = x[p.sk[0] + (ulong)t * p.d + c];
}

// ================================================================================================================
// LoopBlend in three row stages around the h GEMV (ink_in_proj_gemv over the stacked `down`):
//   pair:  pair[s,t] = rA([d * w(d), m * w(m)])                     (threadgroup per (s, t))
//   gate:  y[s,t,c] = (1 - tanh(forget . h)) * d + tanh(gain . h) * m (threadgroup per (s, channel slice), all rows)
//   norm:  out[s,t] = rA(y * inv(y) * norm[s])                      (threadgroup per (s, t); the slice partials in order)
// ================================================================================================================

struct InkBlendP {
    uint d, rows, s, sd;     // D, rows of this chunk, streams, dest streams (1 broadcasts)
    uint g, nsl, cs, t0;     // controller rank G, channel slices, channels per slice, first row of the chunk (out row)
    uint ocount, dcount, mcount, pad0; // row counts (strides) of out, dest, msg tensors
    uint dt0, mt0, pad1, pad2;  // row base of dest / msg reads
    float eps, fpad0, fpad1, fpad2;
    uint sk[8];              // leading bf16 elements: gain, forget, norm, out
};

// bind: 0 dest f32 [Sd, dcount, D], 1 msg f32 [S, mcount, D], 2 pair bf16 [S, rows, 2D] (scratch), 3 InkBlendP.
// Grid: S * rows threadgroups.
kernel void ink_blend_pair(device const float* dest [[buffer(0)]],
                           device const float* msg [[buffer(1)]],
                           device ushort* pair [[buffer(2)]],
                           constant InkBlendP& p [[buffer(3)]],
                           uint gidx [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32 * 32];
    threadgroup float red[32];
    const uint s = gidx / p.rows, r = gidx % p.rows, d = p.d;
    device const float* dr = dest + ((ulong)(p.sd == 1 ? 0 : s) * p.dcount + p.dt0 + r) * d;
    device const float* mr = msg + ((ulong)s * p.mcount + p.mt0 + r) * d;
    float v[2] = {0.0f, 0.0f};
    for (uint c = tid; c < d; c += tpg) {
        v[0] = fma(dr[c], dr[c], v[0]);
        v[1] = fma(mr[c], mr[c], v[1]);
    }
    mx_reduce_t(v, part, red, tid, lane, sg, nsg);
    const float wd = precise::rsqrt(red[0] / float(d) + SRC_EPS), wm = precise::rsqrt(red[1] / float(d) + SRC_EPS);
    device ushort* pr = pair + ((ulong)s * p.rows + r) * 2 * d;
    for (uint c = tid; c < d; c += tpg) {
        pr[c] = f2bf(dr[c] * wd);
        pr[d + c] = f2bf(mr[c] * wm);
    }
}

// bind: 0 dest f32, 1 msg f32, 2 h bf16 [S, rows, G] (scratch), 3 gain [S, D, G], 4 forget [S, D, G], 5 y f32 [S, rows, D]
// (scratch), 6 partial f32 [S, rows, nsl] (scratch; nsl = 2 per channel slice, one per simdgroup), 7 InkBlendP. Grid: S *
// slices threadgroups of `cs` = 64 threads (one channel each). The channel's gain / forget rows stay in registers for all
// rows of the chunk; G % 8 == 0, G <= 8 * GC.
inline float bl_dot8(uint4 w, uint4 x) {
    float a = ip_lo(w.x) * ip_lo(x.x);
    a = fma(ip_hi(w.x), ip_hi(x.x), a);
    a = fma(ip_lo(w.y), ip_lo(x.y), a);
    a = fma(ip_hi(w.y), ip_hi(x.y), a);
    a = fma(ip_lo(w.z), ip_lo(x.z), a);
    a = fma(ip_hi(w.z), ip_hi(x.z), a);
    a = fma(ip_lo(w.w), ip_lo(x.w), a);
    a = fma(ip_hi(w.w), ip_hi(x.w), a);
    return a;
}

template <int GC>
kernel void ink_blend_gate(device const float* dest [[buffer(0)]],
                           device const float* msg [[buffer(1)]],
                           device const uint4* h [[buffer(2)]],
                           device const ushort* gain [[buffer(3)]],
                           device const ushort* forget [[buffer(4)]],
                           device float* yo [[buffer(5)]],
                           device float* partial [[buffer(6)]],
                           constant InkBlendP& p [[buffer(7)]],
                           uint gidx [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint nsl = p.nsl / 2;  // channel slices
    const uint s = gidx / nsl, sl = gidx % nsl, d = p.d, g8 = p.g / 8;
    const uint c = sl * p.cs + tid;
    const bool on = c < d;
    const uint cc = min(c, d - 1);
    // the rows are 16-byte aligned when the tensors are (G % 8 == 0); the skip is folded in by the host (sk 0 / 1 == 0)
    device const uint4* gr = (device const uint4*)(gain + p.sk[0] + ((ulong)s * d + cc) * p.g);
    device const uint4* fr = (device const uint4*)(forget + p.sk[1] + ((ulong)s * d + cc) * p.g);
    uint4 gw[GC], fw[GC];
    for (int j = 0; j < GC; ++j) {
        gw[j] = (uint(j) < g8) ? gr[j] : uint4(0);
        fw[j] = (uint(j) < g8) ? fr[j] : uint4(0);
    }
    for (uint r = 0; r < p.rows; ++r) {
        device const uint4* hr = h + ((ulong)s * p.rows + r) * g8;
        float gn = 0.0f, fg = 0.0f;
        for (int j = 0; j < GC; ++j) {
            if (uint(j) >= g8) break;
            const uint4 hv = hr[j];
            gn += bl_dot8(gw[j], hv);
            fg += bl_dot8(fw[j], hv);
        }
        float yv = 0.0f;
        if (on) {
            const float dv = dest[((ulong)(p.sd == 1 ? 0 : s) * p.dcount + p.dt0 + r) * d + c];
            const float mv = msg[((ulong)s * p.mcount + p.mt0 + r) * d + c];
            yv = (1.0f - precise::tanh(fg)) * dv + precise::tanh(gn) * mv;
            yo[((ulong)s * p.rows + r) * d + c] = yv;
        }
        const float ss = simd_sum(yv * yv);
        if (lane == 0) partial[((ulong)s * p.rows + r) * p.nsl + sl * 2 + sg] = ss;
    }
}
template [[host_name("ink_blend_gate_g1")]] kernel void ink_blend_gate<1>(device const float*, device const float*,
    device const uint4*, device const ushort*, device const ushort*, device float*, device float*, constant InkBlendP&,
    uint, uint, uint, uint);
template [[host_name("ink_blend_gate_g8")]] kernel void ink_blend_gate<8>(device const float*, device const float*,
    device const uint4*, device const ushort*, device const ushort*, device float*, device float*, constant InkBlendP&,
    uint, uint, uint, uint);
template [[host_name("ink_blend_gate_g16")]] kernel void ink_blend_gate<16>(device const float*, device const float*,
    device const uint4*, device const ushort*, device const ushort*, device float*, device float*, constant InkBlendP&,
    uint, uint, uint, uint);

// bind: 0 y f32 [S, rows, D] (scratch), 1 partial f32 [S, rows, nsl], 2 norm [S, D], 3 out bf16 [S, ocount, D], 4 InkBlendP.
// Grid: S * rows threadgroups.
kernel void ink_blend_norm(device const float* yi [[buffer(0)]],
                           device const float* partial [[buffer(1)]],
                           device const ushort* nrm [[buffer(2)]],
                           device ushort* out [[buffer(3)]],
                           constant InkBlendP& p [[buffer(4)]],
                           uint gidx [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]]) {
    const uint s = gidx / p.rows, r = gidx % p.rows, d = p.d;
    float ss = 0.0f;
    for (uint j = 0; j < p.nsl; ++j) ss += partial[((ulong)s * p.rows + r) * p.nsl + j];
    const float inv = precise::rsqrt(ss / float(d) + p.eps);
    device const float* yr = yi + ((ulong)s * p.rows + r) * d;
    for (uint c = tid; c < d; c += tpg)
        out[p.sk[3] + ((ulong)s * p.ocount + p.t0 + r) * d + c] = f2bf(yr[c] * inv * bf2f(nrm[p.sk[2] + (ulong)s * d + c]));
}
