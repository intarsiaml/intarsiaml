// Inkling-2-Mini attention ops on Metal (lane ink-m-attn; contract: crates/intarsia-core/src/inkling2/ops.rs, f64 reference
// crates/intarsia-backend-cpu/src/ink_ref.rs; launcher crates/intarsia-backend-metal/src/inkattn.rs).
//
//   ink_kv_prologue_bf16   InkKvPrologue: one threadgroup per (kv head, row). K / V short conv (static or dynamic taps), k clip,
//                          v unit RMS, q / r scale + clip (+ tau), rel = r . relproj, the conv-ring write when it cannot race
//                          the conv reads of other rows (T <= Rr - K + 1), else ink_ring_write_bf16 runs after it.
//   ink_attn_s1_g{1,2,3,4} RelSinkAttn stage 1: one threadgroup per (kv head, split, row). The split count and the key -> (split,
//                          simdgroup, slot, step) map depend only on the row's position (and the window), never on T, pos0 or
//                          the grid: an R-row call is bitwise R one-row calls (rows-exact), host or device position alike. With
//                          one split it writes the output, else a partial (max, sum, accumulator) into the scratch.
//   ink_attn_comb_bf16     stage 2: merges the partials of a (row, query head) in split order, then the sink and the gate.
//   ink_kv_write_bf16      the cache write after the attention when it could race the reads (ring with T > Cr - window + 1);
//                          otherwise split 0 of stage 1 writes its row.
//
// Scores live in the log2 domain: q is pre-multiplied by scale * log2(e), the rel bias and the sink by log2(e) (the bias is
// not scaled by `scale`; zero for distances >= E). The sink enters once, in the final normalisation. Every reduction has a
// fixed order; precise:: sqrt / divide where the reference clips or normalises.
#include "common.h"

constant constexpr uint INK_MAXS = 32;        // most splits (scratch stride)
constant constexpr uint INK_ALIGN = 32;       // split chunks are a multiple of this
constant constexpr uint INK_NSG = 4;          // simdgroups of a stage-1 threadgroup
constant constexpr float INK_LOG2E = 1.4426950408889634f;
constant constexpr float INK_SRC_EPS = 1e-6f;

// mirrored by `InkP` in inkattn.rs (repr(C))
struct InkP {
    uint t, h, hk, d;
    uint dr, e, k, rr;
    uint cr, pos0, use_dev, flags;  // flags: 1 dyn_conv, 2 kv ring, 4 fused cache / ring write, 8 rel 8-byte aligned
    uint tau_floor, window, nq, pw;
    float qk_clip, rel_clip, eps, tau_alpha;
    float scale, cscale, f2, f3;
    uint sk[16];  // leading element skip per binding index
    // row-table form (lane ink-batch, `OpCall::rows`): table stride / this call's first entry, and each state tensor's byte
    // offset in its row's buffer (the carrier's offset); unused by the other kernels
    uint rts, rti, tab_lo, tab_hi;  // the table's GPU address (lo, hi): resident through the indirect set
    uint roff[4];
};

// Row-table entry j of row r: the state buffer of sequence r (plus the carrier's byte offset).
inline device ushort* ink_rt(constant InkP& p, uint r, uint j) {
    device const ulong* tab = reinterpret_cast<device const ulong*>(as_type<ulong>(uint2(p.tab_lo, p.tab_hi)));
    return reinterpret_cast<device ushort*>(tab[(ulong)r * p.rts + p.rti + j] + p.roff[j]);
}

inline float rA(float x) { return bf2f(f2bf(x)); }

inline void ink_unpack8(uint4 w, thread float* f) {
    f[0] = as_type<float>(w.x << 16);
    f[1] = as_type<float>(w.x & 0xffff0000u);
    f[2] = as_type<float>(w.y << 16);
    f[3] = as_type<float>(w.y & 0xffff0000u);
    f[4] = as_type<float>(w.z << 16);
    f[5] = as_type<float>(w.z & 0xffff0000u);
    f[6] = as_type<float>(w.w << 16);
    f[7] = as_type<float>(w.w & 0xffff0000u);
}

inline uint ink_pos0(constant InkP& p, device const int* pos) { return p.use_dev ? (uint)max(pos[0], 0) : p.pos0; }

inline uint ink_lo(uint pp, uint window) { return (window > 0 && pp + 1 > window) ? pp + 1 - window : 0u; }

// a split holds at least `sk` keys (InkP.f3 bits: the launcher's split policy, one value per process)
inline uint ink_nsplit(uint vis, uint sk) { return clamp((vis + sk - 1) / sk, 1u, INK_MAXS); }

inline uint ink_chunk(uint vis, uint ns) {
    const uint c = (vis + ns - 1) / ns;
    return (c + INK_ALIGN - 1) / INK_ALIGN * INK_ALIGN;
}

inline float ink_sigmoid(float z) { return precise::divide(1.0f, 1.0f + precise::exp(-z)); }

// ------------------------------------------------------------------------------------------------------------------------
// InkKvPrologue
// ------------------------------------------------------------------------------------------------------------------------

// conv input of channel `ch` at position q (row stride `pw` in the call's input, ring rows `rr` of width `ckv`)
inline float ink_cx(device const ushort* x, device const ushort* ring, long q, uint pos0, uint pw, uint rr, uint ckv, uint ch) {
    if (q < 0) return 0.0f;
    if ((uint)q >= pos0) return bf2f(x[(ulong)((uint)q - pos0) * pw + ch]);
    return bf2f(ring[(ulong)((uint)q % rr) * ckv + ch]);
}

// one conv output (rounded): (sum_l (taps[l] + dyn[l]) * x(p - l) + x(p)) * s
inline float ink_conv(device const ushort* x, device const ushort* ring, device const ushort* taps, uint t, uint pos0,
                      constant InkP& p, uint ch, uint kvh) {
    const uint ckv = p.hk * p.d;
    const long pp = (long)(pos0 + t);
    const bool dyn = (p.flags & 1u) != 0;
    float acc = 0.0f;
    for (uint l = 0; l < p.k; ++l) {
        float w = bf2f(taps[(ulong)l * ckv + ch]);
        if (dyn) w += bf2f(x[(ulong)t * p.pw + ckv + kvh * p.k + l]);
        acc = fma(w, ink_cx(x, ring, pp - (long)l, pos0, p.pw, p.rr, ckv, ch), acc);
    }
    return rA((acc + bf2f(x[(ulong)t * p.pw + ch])) * p.cscale);
}

inline float ink_clipf(float ss, float th) { return min(precise::divide(th, precise::sqrt(ss) + INK_SRC_EPS), 1.0f); }

inline float ink_tau(constant InkP& p, uint pp) {
    if (p.tau_floor == 0) return 1.0f;
    const float r = max(precise::divide((float)(pp + 1), (float)p.tau_floor), 1.0f);
    return 1.0f + p.tau_alpha * precise::log(r);
}

// bind: 0 pq, 1 pk, 2 pv, 3 k taps, 4 v taps, 5 q scale, 6 r scale, 7 rel proj, 8 q, 9 rel, 10 k, 11 v, 12 k ring, 13 v ring,
// 14 pos, 15 InkP. Grid (Hk, T), threads a multiple of 32 >= max(D, dr), at most 256.
kernel void ink_kv_prologue_bf16(device const ushort* pq_ [[buffer(0)]],
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
                                 constant InkP& p [[buffer(15)]],
                                 uint3 tg [[threadgroup_position_in_grid]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 tpg3 [[threads_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[8][10];
    threadgroup float rsh[4][32];
    const uint tid = tid3.x, tpg = tpg3.x;
    device const ushort* pq = pq_ + p.sk[0];
    device const ushort* pk = pk_ + p.sk[1];
    device const ushort* pv = pv_ + p.sk[2];
    device const ushort* kt = kt_ + p.sk[3];
    device const ushort* vt = vt_ + p.sk[4];
    device const ushort* qs = qs_ + p.sk[5];
    device const ushort* rs = rs_ + p.sk[6];
    device const ushort* rp = rp_ + p.sk[7];
    device ushort* qo = qo_ + p.sk[8];
    device ushort* relo = relo_ + p.sk[9];
    device ushort* ko = ko_ + p.sk[10];
    device ushort* vo = vo_ + p.sk[11];
    device ushort* kr = kr_ + p.sk[12];
    device ushort* vr = vr_ + p.sk[13];

    const uint kvh = tg.x, t = tg.y;
    const uint D = p.d, H = p.h, G = p.h / p.hk, dr = p.dr, E = p.e;
    const uint ckv = p.hk * D;
    const uint pos0 = ink_pos0(p, pos);
    const uint pp = pos0 + t;
    const uint c = tid, ch = kvh * D + c;
    const ulong qrow = (ulong)t * p.nq;

    float vals[10];
    for (uint i = 0; i < 10; ++i) vals[i] = 0.0f;
    float kc = 0.0f, vc = 0.0f;
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f}, rv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (c < D) {
        kc = ink_conv(pk, kr, kt, t, pos0, p, ch, kvh);
        vc = ink_conv(pv, vr, vt, t, pos0, p, ch, kvh);
        vals[0] = kc * kc;
        vals[1] = vc * vc;
        for (uint g = 0; g < G; ++g) {
            const uint h = kvh * G + g;
            qv[g] = bf2f(pq[qrow + h * D + c]) * bf2f(qs[h * D + c]);
            vals[2 + g] = qv[g] * qv[g];
        }
    }
    if (c < dr) {
        for (uint g = 0; g < G; ++g) {
            const uint h = kvh * G + g;
            rv[g] = bf2f(pq[qrow + H * D + h * dr + c]) * bf2f(rs[h * dr + c]);
            vals[2 + G + g] = rv[g] * rv[g];
        }
    }
    const uint nv = 2 + 2 * G;
    for (uint i = 0; i < nv; ++i) {
        const float s = simd_sum(vals[i]);
        if (lane == 0) part[sg][i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[10];
    for (uint i = 0; i < nv; ++i) {
        float s = 0.0f;
        for (uint j = 0; j < nsg; ++j) s += part[j][i];
        tot[i] = s;
    }
    const float tau = ink_tau(p, pp);
    const bool use_tau = p.tau_floor > 0;
    if (c < D) {
        const ulong kvo = ((ulong)t * p.hk + kvh) * D + c;
        ko[kvo] = f2bf(kc * ink_clipf(tot[0], p.qk_clip));
        const float inv = precise::divide(1.0f, precise::sqrt(tot[1] / (float)D + p.eps));
        vo[kvo] = f2bf(vc * inv);
        for (uint g = 0; g < G; ++g) {
            const uint h = kvh * G + g;
            float x = rA(qv[g] * ink_clipf(tot[2 + g], p.qk_clip));
            if (use_tau) x = rA(x * tau);
            qo[((ulong)t * H + h) * D + c] = f2bf(x);
        }
    }
    if (c < dr) {
        for (uint g = 0; g < G; ++g) {
            float x = rA(rv[g] * ink_clipf(tot[2 + G + g], p.rel_clip));
            if (use_tau) x = rA(x * tau);
            rsh[g][c] = x;
        }
    }
    // the conv rings: only when no other row of the call still reads the row written (launcher: T <= Rr - K + 1)
    if ((p.flags & 4u) != 0 && c < D && t + p.rr >= p.t) {
        const ulong ro = (ulong)(pp % p.rr) * ckv + ch;
        kr[ro] = pk[(ulong)t * p.pw + ch];
        vr[ro] = pv[(ulong)t * p.pw + ch];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // rel = r . relproj: 4 consecutive columns per thread, the d_rel = 16 rows of relproj loaded up front into registers
    // (independent loads); other widths and misaligned operands take the scalar loop. Same per-element order (j ascending).
    const bool vec4 = (E % 4) == 0 && (p.flags & 8u) != 0 && dr == 16;  // launcher: rel proj and rel 8-byte aligned
    const uint e4n = vec4 ? E / 4 : 0;
    for (uint e4 = tid; e4 < e4n; e4 += tpg) {
        uint2 w[16];
#pragma unroll
        for (uint j = 0; j < 16; ++j) w[j] = *(device const uint2*)(rp + (ulong)j * E + e4 * 4);
        for (uint g = 0; g < G; ++g) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (uint j = 0; j < 16; ++j) {
                const float r = rsh[g][j];
                a0 = fma(r, as_type<float>(w[j].x << 16), a0);
                a1 = fma(r, as_type<float>(w[j].x & 0xffff0000u), a1);
                a2 = fma(r, as_type<float>(w[j].y << 16), a2);
                a3 = fma(r, as_type<float>(w[j].y & 0xffff0000u), a3);
            }
            device uint2* dst = (device uint2*)(relo + ((ulong)t * H + kvh * G + g) * E + e4 * 4);
            *dst = uint2((uint)f2bf(a0) | ((uint)f2bf(a1) << 16), (uint)f2bf(a2) | ((uint)f2bf(a3) << 16));
        }
    }
    for (uint e = e4n * 4 + tid; e < E; e += tpg) {
        for (uint g = 0; g < G; ++g) {
            float a = 0.0f;
            for (uint j = 0; j < dr; ++j) a = fma(rsh[g][j], bf2f(rp[(ulong)j * E + e]), a);
            relo[((ulong)t * H + kvh * G + g) * E + e] = f2bf(a);
        }
    }
}

// the conv-ring write as its own pass (rings whose rows a long chunk would overwrite while other rows still read them).
// bind: 0 pk, 1 pv, 2 k ring, 3 v ring, 4 pos, 5 InkP. One thread per (row, channel).
kernel void ink_ring_write_bf16(device const ushort* pk_ [[buffer(0)]],
                                device const ushort* pv_ [[buffer(1)]],
                                device ushort* kr_ [[buffer(2)]],
                                device ushort* vr_ [[buffer(3)]],
                                device const int* pos [[buffer(4)]],
                                constant InkP& p [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint ckv = p.hk * p.d;
    if (gid >= p.t * ckv) return;
    const uint t = gid / ckv, ch = gid % ckv;
    if (t + p.rr < p.t) return;
    device const ushort* pk = pk_ + p.sk[1];
    device const ushort* pv = pv_ + p.sk[2];
    device ushort* kr = kr_ + p.sk[12];
    device ushort* vr = vr_ + p.sk[13];
    const uint pp = ink_pos0(p, pos) + t;
    const ulong ro = (ulong)(pp % p.rr) * ckv + ch;
    kr[ro] = pk[(ulong)t * p.pw + ch];
    vr[ro] = pv[(ulong)t * p.pw + ch];
}

// ------------------------------------------------------------------------------------------------------------------------
// RelSinkAttn
// ------------------------------------------------------------------------------------------------------------------------

// final normalisation of one output element: accumulator `a` relative to max `m` (log2 domain), sum `l`, sink, gate
inline ushort ink_finish(float a, float m, float l, float sink2, float gate) {
    const float mt = max(m, sink2);
    const float sc = exp2(m - mt);
    const float lt = l * sc + exp2(sink2 - mt);
    return f2bf(precise::divide(a * sc, lt) * ink_sigmoid(gate));
}

// Threadgroup state of a stage-1 threadgroup: q of the GP heads (rounded, before the softmax scale) and the merge buffers. The
// rel bias of the split's distances below E (log2 domain, [GP][nb]) and, in the fused kernel, the K / V rows of the call's
// own keys in the split ([2][T][D] bf16) live in dynamic threadgroup memory sized by the launcher (threadgroup(0), (1)).
template <uint GP>
struct InkTg {
    float q[GP][128];
    float acc[INK_NSG][GP][128];
    float ml[INK_NSG][GP][2];
};


// Split geometry of row t (function of its position and the window only).
struct InkSplit {
    uint pp, lo, ns, ks, ke, dlo, nb;
};

inline InkSplit ink_split(constant InkP& p, uint pos0, uint t, uint split) {
    InkSplit g;
    g.pp = pos0 + t;
    g.lo = ink_lo(g.pp, p.window);
    const uint vis = g.pp - g.lo + 1;
    g.ns = ink_nsplit(vis, as_type<uint>(p.f3));
    const uint chunk = ink_chunk(vis, g.ns);
    g.ks = min(g.lo + split * chunk, g.pp + 1);
    g.ke = min(g.ks + chunk, g.pp + 1);
    // biased distances of the split: pp - (ke - 1) .. pp - ks, those below E
    g.dlo = g.ke > g.ks ? g.pp - (g.ke - 1) : 0u;
    g.nb = (g.ke > g.ks && g.dlo < p.e) ? min(g.ke - g.ks, p.e - g.dlo) : 0u;
    return g;
}

// The attention of one (kv head, split, row) over keys ks .. ke given the staged q and bias; writes the output (one split) or
// the partial. KV_TG: the call's own keys (j >= pos0) come from threadgroup rows `kvt` instead of the k / v inputs.
template <uint GP, bool KV_TG>
inline void ink_attn_body(threadgroup InkTg<GP>& S, threadgroup const float* bias, threadgroup const ushort* kvt, uint kvoff, InkSplit g, uint kvh, uint split, uint t,
                          uint PT, uint pos0, device const ushort* kn, device const ushort* vn, device const ushort* kc,
                          device const ushort* vc, device const ushort* pq, device ushort* out, device float* scr,
                          constant InkP& p, uint tid, uint lane, uint sg) {
    const uint D = p.d, H = p.h, Hk = p.hk, Cr = p.cr;
    const bool ring = (p.flags & 2u) != 0;
    const uint lpk = D / 8;     // lanes per key row
    const uint kps = 32 / lpk;  // keys per simdgroup step
    const uint slot = lane / lpk, le = lane % lpk;
    const float qsc = p.scale * INK_LOG2E;
    float qv[GP][8];
    for (uint h = 0; h < GP; ++h)
        for (uint i = 0; i < 8; ++i) qv[h][i] = S.q[h][le * 8 + i] * qsc;
    float m[GP], l[GP], acc[GP][8];
    for (uint h = 0; h < GP; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        for (uint i = 0; i < 8; ++i) acc[h][i] = 0.0f;
    }
    const uint rbase = ring ? g.lo % Cr : 0u;
    device const ushort* kcb = kc + (ulong)kvh * Cr * D;
    device const ushort* vcb = vc + (ulong)kvh * Cr * D;
    for (uint base = g.ks + sg * kps; base < g.ke; base += INK_NSG * kps) {
        const uint j = base + slot;
        const bool ok = j < g.ke;
        uint4 kw = uint4(0), vw = uint4(0);
        if (ok) {
            if (j >= pos0) {
                if (KV_TG) {
                    const uint ro = (j - pos0 - kvoff) * D + le * 8;
                    kw = *(threadgroup const uint4*)(kvt + ro);
                    vw = *(threadgroup const uint4*)(kvt + PT * D + ro);
                } else {
                    const ulong o = ((ulong)(j - pos0) * Hk + kvh) * D + le * 8;
                    kw = *(device const uint4*)(kn + o);
                    vw = *(device const uint4*)(vn + o);
                }
            } else {
                uint row = j;
                if (ring) {
                    row = (j - g.lo) + rbase;
                    if (row >= Cr) row -= Cr;
                }
                const ulong o = (ulong)row * D + le * 8;
                kw = *(device const uint4*)(kcb + o);
                vw = *(device const uint4*)(vcb + o);
            }
        }
        float kf[8], vf[8];
        ink_unpack8(kw, kf);
        ink_unpack8(vw, vf);
        const uint dist = g.pp - j;
        const uint bi = dist - g.dlo;
        for (uint h = 0; h < GP; ++h) {
            float s = 0.0f;
            for (uint i = 0; i < 8; ++i) s = fma(qv[h][i], kf[i], s);
            for (uint o = 1; o < lpk; o <<= 1) s += simd_shuffle_xor(s, (ushort)o);
            if (!ok) continue;
            if (bi < g.nb) s += bias[h * g.nb + bi];
            const float mn = max(m[h], s);
            const float corr = exp2(m[h] - mn);
            const float pr = exp2(s - mn);
            l[h] = l[h] * corr + pr;
            for (uint i = 0; i < 8; ++i) acc[h][i] = fma(pr, vf[i], acc[h][i] * corr);
            m[h] = mn;
        }
    }
    // merge the slots of the simdgroup (xor tree over the slot bits; both partners compute the same value)
    for (uint o = lpk; o < 32; o <<= 1) {
        for (uint h = 0; h < GP; ++h) {
            const float m2 = simd_shuffle_xor(m[h], (ushort)o);
            const float l2 = simd_shuffle_xor(l[h], (ushort)o);
            float a2[8];
            for (uint i = 0; i < 8; ++i) a2[i] = simd_shuffle_xor(acc[h][i], (ushort)o);
            const float mn = max(m[h], m2);
            if (mn == -INFINITY) continue;
            const float c1 = exp2(m[h] - mn), c2 = exp2(m2 - mn);
            l[h] = l[h] * c1 + l2 * c2;
            for (uint i = 0; i < 8; ++i) acc[h][i] = acc[h][i] * c1 + a2[i] * c2;
            m[h] = mn;
        }
    }
    if (slot == 0) {
        for (uint h = 0; h < GP; ++h) {
            for (uint i = 0; i < 8; ++i) S.acc[sg][h][le * 8 + i] = acc[h][i];
            if (le == 0) {
                S.ml[sg][h][0] = m[h];
                S.ml[sg][h][1] = l[h];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= D) return;
    const uint i = tid;
    for (uint hh = 0; hh < GP; ++hh) {
        const uint h = kvh * GP + hh;
        float mm = -INFINITY;
        for (uint s = 0; s < INK_NSG; ++s) mm = max(mm, S.ml[s][hh][0]);
        float ll = 0.0f, aa = 0.0f;
        for (uint s = 0; s < INK_NSG; ++s) {
            const float ms = S.ml[s][hh][0];
            const float c = ms == -INFINITY ? 0.0f : exp2(ms - mm);
            ll += S.ml[s][hh][1] * c;
            aa += S.acc[s][hh][i] * c;
        }
        const ulong pr = (ulong)t * p.nq;
        if (g.ns == 1) {
            const float sink2 = bf2f(pq[pr + 2 * H * D + H * p.dr + h]) * INK_LOG2E;
            const float gate = bf2f(pq[pr + H * D + H * p.dr + h * D + i]);
            out[((ulong)t * H + h) * D + i] = ink_finish(aa, mm, ll, sink2, gate);
        } else {
            const ulong slotp = ((ulong)t * H + h) * INK_MAXS + split;
            scr[slotp * D + i] = aa;
            if (i == 0) {
                device float* ml = scr + (ulong)PT * H * INK_MAXS * D;
                ml[slotp * 2] = mm;
                ml[slotp * 2 + 1] = ll;
            }
        }
    }
}

// RelSinkAttn stage 1. bind: 0 q, 1 k (chunk), 2 v (chunk), 3 k cache, 4 v cache, 5 rel, 6 pq, 7 out, 8 pos, 9 scratch,
// 10 InkP. Grid (Hk, max splits, T), INK_NSG simdgroups. q, k, v and the caches 16-byte aligned (launcher checks), D in
// {8..128}, E <= 1024.
template <uint GP>
[[kernel, max_total_threads_per_threadgroup(INK_NSG * 32)]] void ink_attn_s1(
    device const ushort* q_ [[buffer(0)]],
    device const ushort* kn_ [[buffer(1)]],
    device const ushort* vn_ [[buffer(2)]],
    device ushort* kc_ [[buffer(3)]],
    device ushort* vc_ [[buffer(4)]],
    device const ushort* rel_ [[buffer(5)]],
    device const ushort* pq_ [[buffer(6)]],
    device ushort* out_ [[buffer(7)]],
    device const int* pos [[buffer(8)]],
    device float* scr [[buffer(9)]],
    constant InkP& p [[buffer(10)]],
    threadgroup float* tb [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup InkTg<GP> S;
    const uint tid = tid3.x;
    device const ushort* q = q_ + p.sk[0];
    device const ushort* kn = kn_ + p.sk[1];
    device const ushort* vn = vn_ + p.sk[2];
    device ushort* kc = kc_ + p.sk[3];
    device ushort* vc = vc_ + p.sk[4];
    device const ushort* rel = rel_ + p.sk[5];
    device const ushort* pq = pq_ + p.sk[6];
    device ushort* out = out_ + p.sk[7];

    const uint kvh = tg.x, split = tg.y, t = tg.z;
    const uint D = p.d, H = p.h, Hk = p.hk, E = p.e, Cr = p.cr;
    const bool ring = (p.flags & 2u) != 0;
    const uint pos0 = ink_pos0(p, pos);
    const InkSplit g = ink_split(p, pos0, t, split);

    // the cache write of this row when it cannot race the reads (launcher: linear, or a ring with T <= Cr - window + 1)
    if ((p.flags & 4u) != 0 && split == 0 && (!ring || t + Cr >= p.t)) {
        const uint row = ring ? g.pp % Cr : g.pp;
        const ulong src = ((ulong)t * Hk + kvh) * D, dst = ((ulong)kvh * Cr + row) * D;
        for (uint i = tid; i < D; i += INK_NSG * 32) {
            kc[dst + i] = kn[src + i];
            vc[dst + i] = vn[src + i];
        }
    }
    if (split >= g.ns) return;
    for (uint x = tid; x < GP * D; x += INK_NSG * 32) S.q[x / D][x % D] = bf2f(q[((ulong)t * H + kvh * GP) * D + x]);
    for (uint x = tid; x < GP * g.nb; x += INK_NSG * 32) {
        const uint h = x / g.nb, b = x % g.nb;
        tb[h * g.nb + b] = bf2f(rel[((ulong)t * H + kvh * GP + h) * E + g.dlo + b]) * INK_LOG2E;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ink_attn_body<GP, false>(S, tb, nullptr, 0, g, kvh, split, t, p.t, pos0, kn, vn, kc, vc, pq, out, scr, p, tid, lane, sg);
}

#define INK_S1(GP)                                                                                                         \
    template [[host_name("ink_attn_s1_g" #GP)]] [[kernel]] void ink_attn_s1<GP>(                                          \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, device ushort*,                   \
        device const ushort*, device const ushort*, device ushort*, device const int*, device float*, constant InkP&,       \
        threadgroup float*, uint3, uint3, uint, uint);
INK_S1(1)
INK_S1(2)
INK_S1(3)

// ------------------------------------------------------------------------------------------------------------------------
// InkAttnDecode: InkKvPrologue + RelSinkAttn in one stage-1 threadgroup (1..=16 rows). Each (kv head, split, row) threadgroup
// recomputes what it needs of the prologue with the prologue's own arithmetic (same thread -> channel map, same reduction
// order, the same rounding points): q and r of its row's GP heads, the rel bias of its split's distances (the prologue's
// r . relproj chain per element), and the conv'd, clipped / normalised K / V rows of the call's keys inside its split. So the
// result is bitwise the unfused pair on this backend (tests/inkling2_attn.rs).
// ------------------------------------------------------------------------------------------------------------------------

// bind: 0 pq, 1 pk, 2 pv, 3 k taps, 4 v taps, 5 q scale, 6 r scale, 7 rel proj, 8 k cache, 9 v cache, 10 out, 11 k ring,
// 12 v ring, 13 pos, 14 scratch, 15 InkP. Grid (Hk, max splits, T), INK_NSG simdgroups; D in {8..128}, dr <= 32, E <= 1024.
// flags: 4 = cache write in-kernel (else into the scratch's staging rows for ink_kv_write_bf16), 16 = conv-ring write in-kernel.
// RT (`ink_attn_dec_rows_g{GP}`, lane ink-batch): the row-table form. Threadgroup row r runs the T = 1 call of sequence r at
// pos[r]: its row-indexed operands shifted by r, its caches / rings from table entries 0..4, its own T = 1 scratch block.
template <uint GP, bool RT>
[[kernel, max_total_threads_per_threadgroup(INK_NSG * 32)]] void ink_attn_dec(
    device const ushort* pq_ [[buffer(0)]],
    device const ushort* pk_ [[buffer(1)]],
    device const ushort* pv_ [[buffer(2)]],
    device const ushort* kt_ [[buffer(3)]],
    device const ushort* vt_ [[buffer(4)]],
    device const ushort* qs_ [[buffer(5)]],
    device const ushort* rs_ [[buffer(6)]],
    device const ushort* rp_ [[buffer(7)]],
    device ushort* kc_ [[buffer(8)]],
    device ushort* vc_ [[buffer(9)]],
    device ushort* out_ [[buffer(10)]],
    device ushort* kr_ [[buffer(11)]],
    device ushort* vr_ [[buffer(12)]],
    device const int* pos [[buffer(13)]],
    device float* scr_ [[buffer(14)]],
    constant InkP& p [[buffer(15)]],
    threadgroup float* tb [[threadgroup(0)]],
    threadgroup ushort* tkv [[threadgroup(1)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup InkTg<GP> S;
    threadgroup float part[INK_NSG][32];
    threadgroup float rsh[GP][32];
    const uint tid = tid3.x;
    device const ushort* pq = pq_ + p.sk[0];
    device const ushort* pk = pk_ + p.sk[1];
    device const ushort* pv = pv_ + p.sk[2];
    device const ushort* kt = kt_ + p.sk[3];
    device const ushort* vt = vt_ + p.sk[4];
    device const ushort* qs = qs_ + p.sk[5];
    device const ushort* rs = rs_ + p.sk[6];
    device const ushort* rp = rp_ + p.sk[7];
    device ushort* kc = kc_ + p.sk[8];
    device ushort* vc = vc_ + p.sk[9];
    device ushort* out = out_ + p.sk[10];
    device ushort* kr = kr_ + p.sk[11];
    device ushort* vr = vr_ + p.sk[12];
    device float* scr = scr_;

    const uint kvh = tg.x, split = tg.y;
    uint t = tg.z, PT = p.t;
    const uint D = p.d, H = p.h, Hk = p.hk, dr = p.dr, E = p.e, Cr = p.cr;
    const uint ckv = Hk * D;
    const bool ring = (p.flags & 2u) != 0;
    uint pos0 = ink_pos0(p, pos);
    if (RT) {
        // row r: the T = 1 call of sequence r
        const uint r = tg.z;
        pos0 = (uint)max(pos[r], 0);
        pq += (ulong)r * p.nq;
        pk += (ulong)r * p.pw;
        pv += (ulong)r * p.pw;
        out += (ulong)r * H * D;
        scr += (ulong)r * H * INK_MAXS * (D + 2);
        kc = ink_rt(p, r, 0);
        vc = ink_rt(p, r, 1);
        kr = ink_rt(p, r, 2);
        vr = ink_rt(p, r, 3);
        t = 0;
        PT = 1;
    }
    const InkSplit g = ink_split(p, pos0, t, split);
    if (split >= g.ns) return;
    const uint c = tid, ch = kvh * D + c;
    const ulong qrow = (ulong)t * p.nq;

    // q and r of the row (the prologue's arithmetic)
    float qv[GP], rv[GP];
    {
        float vals[2 * GP];
        for (uint h = 0; h < GP; ++h) {
            qv[h] = 0.0f;
            rv[h] = 0.0f;
            if (c < D) qv[h] = bf2f(pq[qrow + (kvh * GP + h) * D + c]) * bf2f(qs[(kvh * GP + h) * D + c]);
            if (c < dr) rv[h] = bf2f(pq[qrow + H * D + (kvh * GP + h) * dr + c]) * bf2f(rs[(kvh * GP + h) * dr + c]);
            vals[h] = qv[h] * qv[h];
            vals[GP + h] = rv[h] * rv[h];
        }
        for (uint x = 0; x < 2 * GP; ++x) {
            const float s = simd_sum(vals[x]);
            if (lane == 0) part[sg][x] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const float tau = ink_tau(p, g.pp);
        const bool use_tau = p.tau_floor > 0;
        for (uint h = 0; h < GP; ++h) {
            float tq = 0.0f, tr = 0.0f;
            for (uint j = 0; j < INK_NSG; ++j) tq += part[j][h];
            for (uint j = 0; j < INK_NSG; ++j) tr += part[j][GP + h];
            if (c < D) {
                float x = rA(qv[h] * ink_clipf(tq, p.qk_clip));
                if (use_tau) x = rA(x * tau);
                S.q[h][c] = x;
            }
            if (c < dr) {
                float x = rA(rv[h] * ink_clipf(tr, p.rel_clip));
                if (use_tau) x = rA(x * tau);
                rsh[h][c] = x;
            }
        }
    }
    // the call's own keys inside the split: rows t0 .. t1 (positions pos0 + t0 ..)
    const uint jn0 = max(g.ks, pos0), jn1 = g.ke;
    const uint t0 = jn0 - pos0, nnew = jn1 > jn0 ? jn1 - jn0 : 0u;
    float kcv[16], vcv[16];
    for (uint r = 0; r < nnew; ++r) {
        kcv[r] = 0.0f;
        vcv[r] = 0.0f;
        if (c < D) {
            kcv[r] = ink_conv(pk, kr, kt, t0 + r, pos0, p, ch, kvh);
            vcv[r] = ink_conv(pv, vr, vt, t0 + r, pos0, p, ch, kvh);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // part reused
    for (uint r = 0; r < nnew; ++r) {
        const float a = simd_sum(kcv[r] * kcv[r]);
        const float b = simd_sum(vcv[r] * vcv[r]);
        if (lane == 0) {
            part[sg][2 * r] = a;
            part[sg][2 * r + 1] = b;
        }
    }
    // the bias of the split's distances (the prologue's rel element: the r . relproj chain, rounded)
    for (uint x = tid; x < GP * g.nb; x += INK_NSG * 32) {
        const uint h = x / g.nb, e = g.dlo + x % g.nb;
        float a = 0.0f;
        for (uint j = 0; j < dr; ++j) a = fma(rsh[h][j], bf2f(rp[(ulong)j * E + e]), a);
        tb[h * g.nb + x % g.nb] = rA(a) * INK_LOG2E;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < nnew; ++r) {
        float tk = 0.0f, tv = 0.0f;
        for (uint j = 0; j < INK_NSG; ++j) tk += part[j][2 * r];
        for (uint j = 0; j < INK_NSG; ++j) tv += part[j][2 * r + 1];
        if (c < D) {
            tkv[r * D + c] = f2bf(kcv[r] * ink_clipf(tk, p.qk_clip));
            const float inv = precise::divide(1.0f, precise::sqrt(tv / (float)D + p.eps));
            tkv[(PT + r) * D + c] = f2bf(vcv[r] * inv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the threadgroup that holds the row's own key (the split with ke == pp + 1) writes its K / V row and its conv-ring row
    if (g.ks < g.ke && g.ke == g.pp + 1 && c < D) {
        const uint r = t - t0;
        if (!ring || t + Cr >= PT) {
            if ((p.flags & 4u) != 0) {
                const uint row = ring ? g.pp % Cr : g.pp;
                const ulong dst = ((ulong)kvh * Cr + row) * D + c;
                kc[dst] = tkv[r * D + c];
                vc[dst] = tkv[(PT + r) * D + c];
            } else {
                device ushort* stg = (device ushort*)(scr + (ulong)PT * H * INK_MAXS * (D + 2));
                const ulong o = ((ulong)t * Hk + kvh) * D + c;
                stg[o] = tkv[r * D + c];
                stg[(ulong)PT * ckv + o] = tkv[(PT + r) * D + c];
            }
        }
        if ((p.flags & 16u) != 0 && t + p.rr >= PT) {
            const ulong ro = (ulong)(g.pp % p.rr) * ckv + ch;
            kr[ro] = pk[(ulong)t * p.pw + ch];
            vr[ro] = pv[(ulong)t * p.pw + ch];
        }
    }
    // staged row r is the key at position pos0 + t0 + r
    ink_attn_body<GP, true>(S, tb, tkv, t0, g, kvh, split, t, PT, pos0, pk, pv, kc, vc, pq, out, scr, p, tid, lane, sg);
}

#define INK_DEC(NAME, GP, RT)                                                                                              \
    template [[host_name(NAME)]] [[kernel]] void ink_attn_dec<GP, RT>(                                                    \
        device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,       \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, device ushort*, device ushort*,   \
        device ushort*, device ushort*, device const int*, device float*, constant InkP&, threadgroup float*,               \
        threadgroup ushort*, uint3, uint3, uint, uint);
INK_DEC("ink_attn_dec_g1", 1, false)
INK_DEC("ink_attn_dec_g2", 2, false)
INK_DEC("ink_attn_dec_g3", 3, false)
INK_DEC("ink_attn_dec_rows_g1", 1, true)
INK_DEC("ink_attn_dec_rows_g2", 2, true)
INK_DEC("ink_attn_dec_rows_g3", 3, true)

// stage 2. bind: 0 pq, 1 out, 2 pos, 3 scratch, 4 InkP. Grid (H, T), threads >= D. RT (`ink_attn_comb_rows_bf16`): row r is
// the T = 1 call of sequence r at pos[r] on its own scratch block (as ink_attn_dec_rows writes it).
template <bool RT>
[[kernel]] void ink_attn_comb(device const ushort* pq_ [[buffer(0)]],
                              device ushort* out_ [[buffer(1)]],
                              device const int* pos [[buffer(2)]],
                              device const float* scr_ [[buffer(3)]],
                              constant InkP& p [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint tid = tid2.x;
    const uint h = tg.x, D = p.d, H = p.h;
    uint t = tg.y, PT = p.t;
    if (tid >= D) return;
    device const ushort* pq = pq_ + p.sk[6];
    device ushort* out = out_ + p.sk[7];
    device const float* scr = scr_;
    uint pos0 = ink_pos0(p, pos);
    if (RT) {
        const uint r = tg.y;
        pos0 = (uint)max(pos[r], 0);
        pq += (ulong)r * p.nq;
        out += (ulong)r * H * D;
        scr += (ulong)r * H * INK_MAXS * (D + 2);
        t = 0;
        PT = 1;
    }
    const uint pp = pos0 + t;
    const uint vis = pp - ink_lo(pp, p.window) + 1;
    const uint ns = ink_nsplit(vis, as_type<uint>(p.f3));
    if (ns == 1) return;
    const ulong base = ((ulong)t * H + h) * INK_MAXS;
    device const float* ml = scr + (ulong)PT * H * INK_MAXS * D;
    float mm = -INFINITY;
    for (uint s = 0; s < ns; ++s) mm = max(mm, ml[(base + s) * 2]);
    float ll = 0.0f, aa = 0.0f;
    for (uint s = 0; s < ns; ++s) {
        const float ms = ml[(base + s) * 2];
        const float c = ms == -INFINITY ? 0.0f : exp2(ms - mm);
        ll += ml[(base + s) * 2 + 1] * c;
        aa += scr[(base + s) * D + tid] * c;
    }
    const ulong pr = (ulong)t * p.nq;
    const float sink2 = bf2f(pq[pr + 2 * H * D + H * p.dr + h]) * INK_LOG2E;
    const float gate = bf2f(pq[pr + H * D + H * p.dr + h * D + tid]);
    out[((ulong)t * H + h) * D + tid] = ink_finish(aa, mm, ll, sink2, gate);
}

template [[host_name("ink_attn_comb_bf16")]] [[kernel]] void ink_attn_comb<false>(device const ushort*, device ushort*, device const int*,
                                                                                   device const float*, constant InkP&, uint2, uint2);
template [[host_name("ink_attn_comb_rows_bf16")]] [[kernel]] void ink_attn_comb<true>(device const ushort*, device ushort*, device const int*,
                                                                                       device const float*, constant InkP&, uint2, uint2);

// the cache write as its own pass. bind: 0 k, 1 v, 2 k cache, 3 v cache, 4 pos, 5 InkP. One thread per (row, kv head, dim).
kernel void ink_kv_write_bf16(device const ushort* kn_ [[buffer(0)]],
                              device const ushort* vn_ [[buffer(1)]],
                              device ushort* kc_ [[buffer(2)]],
                              device ushort* vc_ [[buffer(3)]],
                              device const int* pos [[buffer(4)]],
                              constant InkP& p [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    const uint D = p.d, Hk = p.hk, Cr = p.cr;
    if (gid >= p.t * Hk * D) return;
    const uint t = gid / (Hk * D), r = gid % (Hk * D), kvh = r / D, i = r % D;
    const bool ring = (p.flags & 2u) != 0;
    if (ring && t + Cr < p.t) return;
    const uint pp = ink_pos0(p, pos) + t;
    const uint row = ring ? pp % Cr : pp;
    const ulong dst = ((ulong)kvh * Cr + row) * D + i;
    (kc_ + p.sk[3])[dst] = (kn_ + p.sk[1])[gid];
    (vc_ + p.sk[4])[dst] = (vn_ + p.sk[2])[gid];
}
