"""Sources for the decode attention-core probe (lane ink-dfuse, measurement only; nothing here is shipped).

  ship.metal   kernels/metal/inkling2_attn.metal with common.h inlined, otherwise unchanged.
  probe.metal  the same file with switches (preprocessor macros set by the harness; all 0 = the shipped arithmetic):
    PR_NB n     key-loop lookahead: the K / V loads of the next n steps are in flight while a step computes (same bits)
    PR_EARLY 1  the first PR_NB steps' cache loads are issued in ink_attn_dec before the prologue (same bits; needs PR_NB)
    PR_NOMATH 1 the key loop keeps its loads and drops the score / online-softmax arithmetic and the slot merge (timing only)
    PR_NOPRO 1  ink_attn_dec keeps the prologue's loads and barriers and drops its arithmetic: no q / r clip reductions,
                no conv, no k clip / v norm, no rel bias (r . relproj) (timing only)
    plus pr_bytes (a byte stream over the split's K / V rows on the same grid) and pr_floor (an empty dispatch).
usage: python3 -I attnvar_gen.py KERNEL_DIR OUT_DIR"""
import sys, os

kd, od = sys.argv[1], sys.argv[2]
common = open(os.path.join(kd, "common.h")).read().replace("#pragma once\n", "")
src = open(os.path.join(kd, "inkling2_attn.metal")).read()


def sub(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)


ship = sub(src, '#include "common.h"', common)
os.makedirs(od, exist_ok=True)
open(os.path.join(od, "ship.metal"), "w").write(ship)

p = ship
p = sub(p, "constant constexpr uint INK_MAXS = 32;", """#ifndef PR_NB
#define PR_NB 0
#endif
#ifndef PR_EARLY
#define PR_EARLY 0
#endif
#ifndef PR_NOMATH
#define PR_NOMATH 0
#endif
#ifndef PR_NOPRO
#define PR_NOPRO 0
#endif
#define PR_NBX (PR_NB > 0 ? PR_NB : 1)
constant constexpr uint INK_MAXS = 32;""")

# ---- body: signature takes the preloaded steps (EARLY) ----
p = sub(p, """                          device const ushort* vc, device const ushort* pq, device ushort* out, device float* scr,
                          constant InkP& p, uint tid, uint lane, uint sg) {
    const uint D = p.d, H = p.h, Hk = p.hk, Cr = p.cr;""", """                          device const ushort* vc, device const ushort* pq, device ushort* out, device float* scr,
                          constant InkP& p, uint tid, uint lane, uint sg, thread uint4* kq, thread uint4* vq, bool pre) {
    const uint D = p.d, H = p.h, Hk = p.hk, Cr = p.cr;""")
p = sub(p, "ink_attn_body<GP, false>(S, tb, nullptr, 0, g, kvh, split, t, p.t, pos0, kn, vn, kc, vc, pq, out, scr, p, tid, lane, sg);",
        """uint4 kq0[PR_NBX], vq0[PR_NBX];
    ink_attn_body<GP, false>(S, tb, nullptr, 0, g, kvh, split, t, p.t, pos0, kn, vn, kc, vc, pq, out, scr, p, tid, lane, sg, kq0, vq0, false);""")

# ---- body: the key loop ----
old_loop_head = """    for (uint base = g.ks + sg * kps; base < g.ke; base += INK_NSG * kps) {
        const uint j = base + slot;
        const bool ok = j < g.ke;
        uint4 kw = uint4(0), vw = uint4(0);
        if (ok) {"""
new_loop_head = """    const uint step = INK_NSG * kps;
#define PR_LOAD(J, KW, VW)                                                                                       \\
    do {                                                                                                         \\
        const uint jj_ = (J);                                                                                    \\
        KW = uint4(0);                                                                                           \\
        VW = uint4(0);                                                                                           \\
        if (jj_ < g.ke) {                                                                                        \\
            if (jj_ >= pos0) {                                                                                   \\
                if (!KV_TG) {                                                                                    \\
                    const ulong o_ = ((ulong)(jj_ - pos0) * Hk + kvh) * D + le * 8;                              \\
                    KW = *(device const uint4*)(kn + o_);                                                        \\
                    VW = *(device const uint4*)(vn + o_);                                                        \\
                }                                                                                                \\
            } else {                                                                                             \\
                uint row_ = jj_;                                                                                 \\
                if (ring) {                                                                                      \\
                    row_ = (jj_ - g.lo) + rbase;                                                                 \\
                    if (row_ >= Cr) row_ -= Cr;                                                                  \\
                }                                                                                                \\
                const ulong o_ = (ulong)row_ * D + le * 8;                                                       \\
                KW = *(device const uint4*)(kcb + o_);                                                           \\
                VW = *(device const uint4*)(vcb + o_);                                                           \\
            }                                                                                                    \\
        }                                                                                                        \\
    } while (0)
#if PR_NB > 0
    if (!pre) {
#pragma unroll
        for (uint u = 0; u < PR_NB; ++u) PR_LOAD(g.ks + sg * kps + u * step + slot, kq[u], vq[u]);
    }
#endif
#if PR_NOMATH
    uint4 xacc = uint4(0);
#endif
    for (uint base = g.ks + sg * kps; base < g.ke; base += step) {
        const uint j = base + slot;
        const bool ok = j < g.ke;
        uint4 kw = uint4(0), vw = uint4(0);
#if PR_NB > 0
        kw = kq[0];
        vw = vq[0];
#pragma unroll
        for (uint u = 0; u + 1 < PR_NB; ++u) {
            kq[u] = kq[u + 1];
            vq[u] = vq[u + 1];
        }
        PR_LOAD(j + PR_NB * step, kq[PR_NB - 1], vq[PR_NB - 1]);
        if (KV_TG && ok && j >= pos0) {
            const uint ro = (j - pos0 - kvoff) * D + le * 8;
            kw = *(threadgroup const uint4*)(kvt + ro);
            vw = *(threadgroup const uint4*)(kvt + PT * D + ro);
        }
#else
        if (ok) {"""
p = sub(p, old_loop_head, new_loop_head)
p = sub(p, """                kw = *(device const uint4*)(kcb + o);
                vw = *(device const uint4*)(vcb + o);
            }
        }
        float kf[8], vf[8];""", """                kw = *(device const uint4*)(kcb + o);
                vw = *(device const uint4*)(vcb + o);
            }
        }
#endif
#if PR_NOMATH
        if (ok) xacc ^= kw ^ vw;
    }
    for (uint h = 0; h < GP; ++h) {
        m[h] = 0.0f;
        l[h] = 1.0f;
        for (uint i = 0; i < 8; ++i) acc[h][i] = as_type<float>(((xacc.x ^ xacc.y ^ xacc.z ^ xacc.w) + i) & 0x3effffffu);
    }
#else
        float kf[8], vf[8];""")
p = sub(p, """            m[h] = mn;
        }
    }
    // merge the slots of the simdgroup""", """            m[h] = mn;
        }
    }
#endif
    // merge the slots of the simdgroup""")
p = sub(p, """    for (uint o = lpk; o < 32; o <<= 1) {
        for (uint h = 0; h < GP; ++h) {
            const float m2""", """    for (uint o = lpk; o < 32 && !PR_NOMATH; o <<= 1) {
        for (uint h = 0; h < GP; ++h) {
            const float m2""")

# ---- ink_attn_dec: EARLY preload and NOPRO ----
p = sub(p, """    const InkSplit g = ink_split(p, pos0, t, split);
    if (split >= g.ns) return;
    const uint c = tid, ch = kvh * D + c;
    const ulong qrow = (ulong)t * p.nq;""", """    const InkSplit g = ink_split(p, pos0, t, split);
    if (split >= g.ns) return;
    const uint c = tid, ch = kvh * D + c;
    const ulong qrow = (ulong)t * p.nq;
    uint4 kq[PR_NBX], vq[PR_NBX];
#if PR_EARLY && PR_NB > 0
    {
        const uint lpk = D / 8, kps = 32 / lpk, slot = lane / lpk, le = lane % lpk, step = INK_NSG * kps;
        const uint rbase = ring ? g.lo % Cr : 0u;
        device const ushort* kcb = kc + (ulong)kvh * Cr * D;
        device const ushort* vcb = vc + (ulong)kvh * Cr * D;
#pragma unroll
        for (uint u = 0; u < PR_NB; ++u) {
            const uint jj = g.ks + sg * kps + u * step + slot;
            kq[u] = uint4(0);
            vq[u] = uint4(0);
            if (jj < g.ke && jj < pos0) {
                uint row = jj;
                if (ring) {
                    row = (jj - g.lo) + rbase;
                    if (row >= Cr) row -= Cr;
                }
                const ulong o = (ulong)row * D + le * 8;
                kq[u] = *(device const uint4*)(kcb + o);
                vq[u] = *(device const uint4*)(vcb + o);
            }
        }
    }
    const bool pre = true;
#else
    const bool pre = false;
#endif""")
p = sub(p, """        for (uint x = 0; x < 2 * GP; ++x) {
            const float s = simd_sum(vals[x]);
            if (lane == 0) part[sg][x] = s;
        }""", """        for (uint x = 0; x < 2 * GP && !PR_NOPRO; ++x) {
            const float s = simd_sum(vals[x]);
            if (lane == 0) part[sg][x] = s;
        }""")
p = sub(p, """            if (c < D) {
                float x = rA(qv[h] * ink_clipf(tq, p.qk_clip));
                if (use_tau) x = rA(x * tau);
                S.q[h][c] = x;
            }
            if (c < dr) {
                float x = rA(rv[h] * ink_clipf(tr, p.rel_clip));
                if (use_tau) x = rA(x * tau);
                rsh[h][c] = x;
            }""", """#if PR_NOPRO
            if (c < D) S.q[h][c] = qv[h];
            if (c < dr) rsh[h][c] = rv[h];
            (void)tq; (void)tr; (void)tau; (void)use_tau;
#else
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
#endif""")
p = sub(p, """        if (c < D) {
            kcv[r] = ink_conv(pk, kr, kt, t0 + r, pos0, p, ch, kvh);
            vcv[r] = ink_conv(pv, vr, vt, t0 + r, pos0, p, ch, kvh);
        }""", """        if (c < D) {
#if PR_NOPRO
            kcv[r] = bf2f(pk[(ulong)(t0 + r) * p.pw + ch]);
            vcv[r] = bf2f(pv[(ulong)(t0 + r) * p.pw + ch]);
#else
            kcv[r] = ink_conv(pk, kr, kt, t0 + r, pos0, p, ch, kvh);
            vcv[r] = ink_conv(pv, vr, vt, t0 + r, pos0, p, ch, kvh);
#endif
        }""")
p = sub(p, """    for (uint r = 0; r < nnew; ++r) {
        const float a = simd_sum(kcv[r] * kcv[r]);""", """    for (uint r = 0; r < nnew && !PR_NOPRO; ++r) {
        const float a = simd_sum(kcv[r] * kcv[r]);""")
p = sub(p, """        float a = 0.0f;
        for (uint j = 0; j < dr; ++j) a = fma(rsh[h][j], bf2f(rp[(ulong)j * E + e]), a);
        tb[h * g.nb + x % g.nb] = rA(a) * INK_LOG2E;""", """        float a = 0.0f;
        for (uint j = 0; j < dr && !PR_NOPRO; ++j) a = fma(rsh[h][j], bf2f(rp[(ulong)j * E + e]), a);
        tb[h * g.nb + x % g.nb] = rA(a) * INK_LOG2E;""")
p = sub(p, """        if (c < D) {
            tkv[r * D + c] = f2bf(kcv[r] * ink_clipf(tk, p.qk_clip));
            const float inv = precise::divide(1.0f, precise::sqrt(tv / (float)D + p.eps));
            tkv[(PT + r) * D + c] = f2bf(vcv[r] * inv);
        }""", """        if (c < D) {
#if PR_NOPRO
            tkv[r * D + c] = f2bf(kcv[r]);
            tkv[(PT + r) * D + c] = f2bf(vcv[r]);
            (void)tk; (void)tv;
#else
            tkv[r * D + c] = f2bf(kcv[r] * ink_clipf(tk, p.qk_clip));
            const float inv = precise::divide(1.0f, precise::sqrt(tv / (float)D + p.eps));
            tkv[(PT + r) * D + c] = f2bf(vcv[r] * inv);
#endif
        }""")
p = sub(p, "ink_attn_body<GP, true>(S, tb, tkv, t0, g, kvh, split, t, PT, pos0, pk, pv, kc, vc, pq, out, scr, p, tid, lane, sg);",
        "ink_attn_body<GP, true>(S, tb, tkv, t0, g, kvh, split, t, PT, pos0, pk, pv, kc, vc, pq, out, scr, p, tid, lane, sg, kq, vq, pre);")

p += """
// ---- probe-only kernels ----
// The split's K / V cache rows as a byte stream on the decode grid (Hk, splits, 1) x 128 threads: independent loads, xor-folded.
kernel void pr_bytes(device const ushort* kc_ [[buffer(8)]],
                     device const ushort* vc_ [[buffer(9)]],
                     device uint* sink [[buffer(14)]],
                     constant InkP& p [[buffer(15)]],
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint3 tid3 [[thread_position_in_threadgroup]]) {
    const uint kvh = tg.x, split = tg.y, tid = tid3.x;
    const InkSplit g = ink_split(p, p.pos0, 0, split);
    if (split >= g.ns) return;
    const uint D = p.d, Cr = p.cr;
    const bool ring = (p.flags & 2u) != 0;
    const uint rbase = ring ? g.lo % Cr : 0u;
    device const ushort* kcb = kc_ + (ulong)kvh * Cr * D;
    device const ushort* vcb = vc_ + (ulong)kvh * Cr * D;
    const uint lpk = D / 8, slot = tid / lpk, le = tid % lpk, kpr = (INK_NSG * 32) / lpk;
    uint4 x = uint4(0);
    for (uint j = g.ks + slot; j < g.ke; j += kpr) {
        uint row = j;
        if (ring) {
            row = (j - g.lo) + rbase;
            if (row >= Cr) row -= Cr;
        }
        const ulong o = (ulong)row * D + le * 8;
        x ^= *(device const uint4*)(kcb + o) ^ *(device const uint4*)(vcb + o);
    }
    sink[(kvh * INK_MAXS + split) * (INK_NSG * 32) + tid] = x.x ^ x.y ^ x.z ^ x.w;
}
kernel void pr_floor(device uint* sink [[buffer(14)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) sink[0] += 1;
}
// bf16 fill in [-1, 1): element i of a buffer from a hash of (i, seed)
kernel void pr_fill(device ushort* w [[buffer(0)]], constant uint2& pm [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= pm.x) return;
    uint v = (i + pm.y * 0x9e3779b1u) * 0x85ebca77u;
    v ^= v >> 15; v *= 0xc2b2ae3du; v ^= v >> 13; v *= 0x27d4eb2fu; v ^= v >> 16;
    w[i] = f2bf((float)(v >> 8) * (2.0f / 16777216.0f) - 1.0f);
}
"""
open(os.path.join(od, "probe.metal"), "w").write(p)
print("wrote", od)
