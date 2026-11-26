// Inkling-2-Mini attention ops on HIP (lane ink-hip-attn; contract: crates/intarsia-core/src/inkling2/ops.rs, f64 reference
// crates/intarsia-backend-cpu/src/ink_ref.rs; launcher crates/intarsia-backend-hip/src/inkattn.rs; ABI ik_hip_abi_inkattn.h).
// A port of the CUDA correctness form (kernels/cuda/inkling2_attn.cu: ink_prologue_k, ink_ring_write_k, ink_attn_s1_k,
// ink_attn_comb_k, ink_kv_write_k; CUDA's INTARSIA_INK_DECODE=splitkv path), the same structure and arithmetic:
//
//   ink_prologue_k     InkKvPrologue: one block per (kv head, row). K / V short conv (static or dynamic taps), k clip, v unit
//                      RMS, q / r scale + clip (+ tau), rel = r . relproj, the conv-ring write when it cannot race the conv reads
//                      of other rows (T <= Rr - K + 1), else ink_ring_write_k runs after it.
//   ink_attn_s1_k<G>   RelSinkAttn stage 1: one block per (kv head, split, row), 4 waves. The split count and the key -> (split,
//                      wave, slot, step) map depend only on the row's position (and the window), never on T, pos0 or the grid:
//                      an R-row call is bitwise R one-row calls (rows-exact), host or device position alike. With one split it
//                      writes the output, else a partial (max, sum, accumulator) into the scratch.
//   ink_attn_comb_k    stage 2: merges the partials of a (row, query head) in split order, then the sink and the gate.
//   ink_kv_write_k     the cache write after the attention when it could race the reads (ring with T > Cr - window + 1).
//
// HIP differences from the CUDA file: wave32 shuffles via __shfl_xor(v, o, 32) (every lane of the wave runs each shuffle);
// no __ldg / PDL; the tile and flash kernels are not ported (LDS, mma.sync). A long call runs as slices of at most SLICE rows
// in order (as CUDA's tile path does with 16): each slice reads the cache rows the slices before it wrote, so a row's bits do
// not change, and the scratch stays bounded (inkattn.rs `attn_scratch`). FP contraction is off in this file (the pragma below;
// hipcc's default is fast-honor-pragmas), so `a * b + c` is never fused where the CUDA source keeps it unfused.
//
// Scores live in the log2 domain: q is multiplied by scale * log2(e), the rel bias and the sink by log2(e) (the bias is not
// scaled by `scale`; zero for distances >= E). The sink enters once, in the final normalisation. Every reduction has a fixed
// order.
#include "ik_hip.h"
#include "ik_hip_abi_inkattn.h"
#include <math.h>

#pragma clang fp contract(off)

namespace {

typedef ik_bf16 bf16;

constexpr int MAXS = 32;        // most splits (scratch stride)
constexpr int ALIGN = 32;       // split chunks are a multiple of this
constexpr int NSG = 4;          // waves of an attention block
constexpr int SPLIT_KEYS = 64;  // default keys per split
constexpr int SLICE = 64;       // most rows of one attention launch (scratch bound; inkattn.rs ATTN_SLICE)
constexpr int KMAX = 4;         // most conv taps
constexpr float LOG2E = 1.4426950408889634f;
constexpr float SRC_EPS = 1e-6f;

// flags
constexpr int F_DYN = 1;          // dynamic conv taps
constexpr int F_RING = 2;         // K / V caches are rings
constexpr int F_KV_WRITE = 4;     // cache write in the attention kernel (else ink_kv_write_k after it)
constexpr int F_RING_WRITE = 16;  // conv-ring write in the kernel (else ink_ring_write_k after it)

struct InkP {
    const bf16 *pq, *pk, *pv, *kt, *vt, *qs, *rs, *rp;
    const bf16 *q, *kn, *vn, *rel;
    bf16 *qo, *relo, *ko, *vo, *kr, *vr, *kc, *vc, *out;
    float* scr;
    const int* pos;
    int t, h, hk, d, dr, e, k, rr, cr, pos0, flags, tau_floor, window, nq, pw, sk, toff;
    float qk_clip, rel_clip, eps, tau_alpha, scale, cscale;
};

// explicit, unfused arithmetic (the CUDA source's _rn intrinsics)
IK_DEVI float fadd(float a, float b) { return a + b; }
IK_DEVI float fmul(float a, float b) { return a * b; }
IK_DEVI float fdiv(float a, float b) { return a / b; }
IK_DEVI float ffma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

IK_DEVI float b2f(bf16 x) { return ik_b2f(x); }
IK_DEVI bf16 f2b(float x) { return ik_f2b(x); }
IK_DEVI float rA(float x) { return b2f(f2b(x)); }

IK_DEVI float shx(float v, int o) { return __shfl_xor(v, o, IK_WARP); }

// toff: row offset of a slice of a longer call (the launcher shifts the row pointers and the host pos0 by it)
IK_DEVI int ink_pos0(const InkP& p) { return p.pos ? max(p.pos[0], 0) + p.toff : p.pos0; }

__host__ __device__ inline unsigned ink_lo(unsigned pp, unsigned window) { return (window > 0 && pp + 1 > window) ? pp + 1 - window : 0u; }

// a split holds at least `sk` keys
__host__ __device__ inline unsigned ink_nsplit(unsigned vis, unsigned sk) {
    unsigned n = (vis + sk - 1) / sk;
    return n < 1 ? 1u : (n > (unsigned)MAXS ? (unsigned)MAXS : n);
}

__host__ __device__ inline unsigned ink_chunk(unsigned vis, unsigned ns) {
    const unsigned c = (vis + ns - 1) / ns;
    return (c + ALIGN - 1) / ALIGN * ALIGN;
}

// butterfly sum over the wave: every lane gets the same bits
IK_DEVI float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fadd(v, shx(v, o));
    return v;
}

IK_DEVI float ink_sigmoid(float z) { return fdiv(1.0f, fadd(1.0f, expf(-z))); }

// ------------------------------------------------------------------------------------------------------------------------
// InkKvPrologue arithmetic
// ------------------------------------------------------------------------------------------------------------------------

// conv input of channel `ch` at position q (row stride `pw` in the call's input, ring rows `rr` of width `ckv`)
IK_DEVI float ink_cx(const bf16* x, const bf16* ring, long q, unsigned pos0, unsigned pw, unsigned rr, unsigned ckv, unsigned ch) {
    if (q < 0) return 0.0f;
    if ((unsigned)q >= pos0) return b2f(x[(size_t)((unsigned)q - pos0) * pw + ch]);
    return b2f(ring[(size_t)((unsigned)q % rr) * ckv + ch]);
}

// one conv output (rounded): (sum_l (taps[l] + dyn[l]) * x(p - l) + x(p)) * s
IK_DEVI float ink_conv(const bf16* x, const bf16* ring, const bf16* taps, unsigned t, unsigned pos0, const InkP& p, unsigned ch, unsigned kvh) {
    const unsigned ckv = p.hk * p.d;
    const long pp = (long)(pos0 + t);
    const bool dyn = (p.flags & F_DYN) != 0;
    float w[KMAX], xv[KMAX];
#pragma unroll
    for (int l = 0; l < KMAX; ++l) {
        w[l] = 0.0f;
        xv[l] = 0.0f;
        if (l < p.k) {
            w[l] = b2f(taps[(size_t)l * ckv + ch]);
            if (dyn) w[l] = fadd(w[l], b2f(x[(size_t)t * p.pw + ckv + kvh * p.k + l]));
            xv[l] = ink_cx(x, ring, pp - (long)l, pos0, p.pw, p.rr, ckv, ch);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int l = 0; l < KMAX; ++l)
        if (l < p.k) acc = ffma(w[l], xv[l], acc);
    return rA(fmul(fadd(acc, b2f(x[(size_t)t * p.pw + ch])), p.cscale));
}

IK_DEVI float ink_clipf(float ss, float th) { return fminf(fdiv(th, fadd(sqrtf(ss), SRC_EPS)), 1.0f); }

IK_DEVI float ink_tau(const InkP& p, unsigned pp) {
    if (p.tau_floor == 0) return 1.0f;
    const float r = fmaxf(fdiv((float)(pp + 1), (float)p.tau_floor), 1.0f);
    return ffma(p.tau_alpha, logf(r), 1.0f);
}

IK_DEVI float ink_inv_rms(float ss, const InkP& p) { return fdiv(1.0f, sqrtf(fadd(fdiv(ss, (float)p.d), p.eps))); }

// q (or r) after the scale, clip and tau, rounded
IK_DEVI float ink_qr(float raw, float ss, float th, bool use_tau, float tau) {
    float x = rA(fmul(raw, ink_clipf(ss, th)));
    if (use_tau) x = rA(fmul(x, tau));
    return x;
}

// ink_rel of G heads (r rows of 32) at one e, the relproj loads shared: the r . relproj chain (j ascending), unrounded
IK_DEVI void ink_rel_g(const float (*r)[32], int G, const bf16* rp, int dr, int E, int e, float* out) {
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int j0 = 0; j0 < 32; j0 += 16) {
        if (j0 >= dr) break;
        float w[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) w[j] = j0 + j < dr ? b2f(rp[(size_t)(j0 + j) * E + e]) : 0.0f;
#pragma unroll
        for (int g = 0; g < 4; ++g)
            if (g < G)
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j0 + j < dr) a[g] = ffma(r[g][j0 + j], w[j], a[g]);
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) out[g] = a[g];
}

// One wave computes one vector of the prologue (lane-strided elements, a butterfly sum of squares).
constexpr int JMAX = 8;  // most elements per lane (head_dim <= 256)

template <int J>
IK_DEVI float job_conv(const InkP& p, const bf16* x, const bf16* ring, const bf16* taps, unsigned t, unsigned pos0, unsigned kvh, unsigned lane,
                       float* v) {
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < J; ++i) {
        const unsigned c = lane + 32 * i;
        v[i] = 0.0f;
        if (c < (unsigned)p.d) {
            v[i] = ink_conv(x, ring, taps, t, pos0, p, kvh * p.d + c, kvh);
            ss = ffma(v[i], v[i], ss);
        }
    }
    return warp_sum(ss);
}

// q (r = false: D elements) or r (r = true: dr elements) of query head h at call row t, after scale, clip and tau (rounded)
template <int J>
IK_DEVI void job_qr(const InkP& p, unsigned t, unsigned h, unsigned pp, bool r, unsigned lane, float* v) {
    const unsigned n = r ? p.dr : p.d;
    const size_t off = (size_t)t * p.nq + (r ? (size_t)p.h * p.d + h * p.dr : (size_t)h * p.d);
    const bf16* sc = r ? p.rs + h * p.dr : p.qs + h * p.d;
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < J; ++i) {
        const unsigned c = lane + 32 * i;
        v[i] = 0.0f;
        if (c < n) {
            v[i] = fmul(b2f(p.pq[off + c]), b2f(sc[c]));
            ss = ffma(v[i], v[i], ss);
        }
    }
    ss = warp_sum(ss);
    const float tau = ink_tau(p, pp);
    const bool use_tau = p.tau_floor > 0;
#pragma unroll
    for (int i = 0; i < J; ++i) v[i] = ink_qr(v[i], ss, r ? p.rel_clip : p.qk_clip, use_tau, tau);
}

IK_DEVI float k_out(float v, float ss, const InkP& p) { return rA(fmul(v, ink_clipf(ss, p.qk_clip))); }
IK_DEVI float v_out(float v, float ss, const InkP& p) { return rA(fmul(v, ink_inv_rms(ss, p))); }

// InkKvPrologue. Grid (Hk, T, rel slices), 256 threads (8 waves): waves take the jobs k, v, q_0 .., r_0 .. in turn, then all
// threads the rel tail.
template <int J>
__global__ void __launch_bounds__(256) ink_prologue_k(InkP p) {
    __shared__ float rsh[4][32];
    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const unsigned kvh = blockIdx.x, t = blockIdx.y;
    const unsigned D = p.d, H = p.h, G = p.h / p.hk, dr = p.dr, E = p.e;
    const unsigned ckv = p.hk * D;
    const unsigned pos0 = ink_pos0(p);
    const unsigned pp = pos0 + t;
    // blockIdx.z: slice of the rel tail; slice 0 also runs the k / v / q jobs and the ring write, every slice its r jobs
    const bool lead = blockIdx.z == 0;
    const unsigned njobs = 2 + 2 * G;
    for (unsigned job = lead ? warp : 2 + G + warp; job < njobs; job += nw) {
        float v[J];
        if (job < 2) {
            const float ss = job_conv<J>(p, job ? p.pv : p.pk, job ? p.vr : p.kr, job ? p.vt : p.kt, t, pos0, kvh, lane, v);
            bf16* dst = (job ? p.vo : p.ko) + ((size_t)t * p.hk + kvh) * D;
#pragma unroll
            for (int i = 0; i < J; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) dst[c] = f2b(job ? v_out(v[i], ss, p) : k_out(v[i], ss, p));
            }
        } else if (job < 2 + G) {
            const unsigned h = kvh * G + (job - 2);
            job_qr<J>(p, t, h, pp, false, lane, v);
#pragma unroll
            for (int i = 0; i < J; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) p.qo[((size_t)t * H + h) * D + c] = f2b(v[i]);
            }
        } else {
            const unsigned g = job - 2 - G;
            job_qr<J>(p, t, kvh * G + g, pp, true, lane, v);
            if (lane < dr) rsh[g][lane] = v[0];
        }
    }
    // the conv rings: only when no other row of the call still reads the row written (launcher: T <= Rr - K + 1)
    if (lead && (p.flags & F_RING_WRITE) != 0 && t + p.rr >= (unsigned)p.t) {
        for (unsigned c = tid; c < D; c += blockDim.x) {
            const unsigned ch = kvh * D + c;
            const size_t ro = (size_t)(pp % p.rr) * ckv + ch;
            p.kr[ro] = p.pk[(size_t)t * p.pw + ch];
            p.vr[ro] = p.pv[(size_t)t * p.pw + ch];
        }
    }
    __syncthreads();
    const unsigned per = (E + gridDim.z - 1) / gridDim.z, e1 = min(E, (blockIdx.z + 1) * per);
    for (unsigned e = blockIdx.z * per + tid; e < e1; e += blockDim.x) {
        float rv[4];
        ink_rel_g(rsh, G, p.rp, dr, E, e, rv);
        for (unsigned g = 0; g < G; ++g) p.relo[((size_t)t * H + kvh * G + g) * E + e] = f2b(rv[g]);
    }
}

// the conv-ring write as its own pass (rings whose rows a long chunk would overwrite while other rows still read them).
// One thread per (row, channel).
__global__ void ink_ring_write_k(InkP p) {
    const unsigned ckv = p.hk * p.d;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)p.t * ckv) return;
    const unsigned t = gid / ckv, ch = gid % ckv;
    if (t + p.rr < (unsigned)p.t) return;
    const unsigned pp = ink_pos0(p) + t;
    const size_t ro = (size_t)(pp % p.rr) * ckv + ch;
    p.kr[ro] = p.pk[(size_t)t * p.pw + ch];
    p.vr[ro] = p.pv[(size_t)t * p.pw + ch];
}

// ------------------------------------------------------------------------------------------------------------------------
// RelSinkAttn
// ------------------------------------------------------------------------------------------------------------------------

// final normalisation of one output element: accumulator `a` relative to max `m` (log2 domain), sum `l`, sink, gate
IK_DEVI bf16 ink_finish(float a, float m, float l, float sink2, float gate) {
    const float mt = fmaxf(m, sink2);
    const float sc = exp2f(m - mt);
    const float lt = ffma(l, sc, exp2f(sink2 - mt));
    return f2b(fmul(fdiv(fmul(a, sc), lt), ink_sigmoid(gate)));
}

// Block state of an attention block: q of the GP heads (rounded, before the softmax scale) and the merge buffers.
template <int GP>
struct InkTg {
    float q[GP][128];
    float acc[NSG][GP][128];
    float ml[NSG][GP][2];
};

// Split geometry of row t (function of its position and the window only).
struct InkSplit {
    unsigned pp, lo, ns, ks, ke, dlo, nb;
};

IK_DEVI InkSplit ink_split(const InkP& p, unsigned pos0, unsigned t, unsigned split) {
    InkSplit g;
    g.pp = pos0 + t;
    g.lo = ink_lo(g.pp, p.window);
    const unsigned vis = g.pp - g.lo + 1;
    g.ns = ink_nsplit(vis, p.sk);
    const unsigned chunk = ink_chunk(vis, g.ns);
    g.ks = min(g.lo + split * chunk, g.pp + 1);
    g.ke = min(g.ks + chunk, g.pp + 1);
    // biased distances of the split: pp - (ke - 1) .. pp - ks, those below E
    g.dlo = g.ke > g.ks ? g.pp - (g.ke - 1) : 0u;
    g.nb = (g.ke > g.ks && g.dlo < (unsigned)p.e) ? min(g.ke - g.ks, (unsigned)p.e - g.dlo) : 0u;
    return g;
}

// The attention of one (kv head, split, row) over keys ks .. ke given the staged q and bias; writes the output (one split) or
// the partial.
template <int GP>
IK_DEVI void ink_attn_body(InkTg<GP>& S, const float* bias, InkSplit g, unsigned kvh, unsigned split, unsigned t, unsigned pos0, const InkP& p,
                           unsigned tid, unsigned lane, unsigned sg) {
    const unsigned D = p.d, H = p.h, Hk = p.hk, Cr = p.cr;
    const bool ring = (p.flags & F_RING) != 0;
    const unsigned lpk = D / 8;     // lanes per key row
    const unsigned kps = 32 / lpk;  // keys per wave step
    const unsigned slot = lane / lpk, le = lane % lpk;
    const float qsc = fmul(p.scale, LOG2E);
    float qv[GP][8];
    for (int h = 0; h < GP; ++h)
        for (int i = 0; i < 8; ++i) qv[h][i] = fmul(S.q[h][le * 8 + i], qsc);
    float m[GP], l[GP], acc[GP][8];
    for (int h = 0; h < GP; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        for (int i = 0; i < 8; ++i) acc[h][i] = 0.0f;
    }
    const unsigned rbase = ring ? g.lo % Cr : 0u;
    const bf16* kcb = p.kc + (size_t)kvh * Cr * D;
    const bf16* vcb = p.vc + (size_t)kvh * Cr * D;
    for (unsigned base = g.ks + sg * kps; base < g.ke; base += NSG * kps) {
        const unsigned j = base + slot;
        const bool ok = j < g.ke;
        uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
        if (ok) {
            if (j >= pos0) {
                const size_t o = ((size_t)(j - pos0) * Hk + kvh) * D + le * 8;
                kw = *(const uint4*)(p.kn + o);
                vw = *(const uint4*)(p.vn + o);
            } else {
                unsigned row = j;
                if (ring) {
                    row = (j - g.lo) + rbase;
                    if (row >= Cr) row -= Cr;
                }
                const size_t o = (size_t)row * D + le * 8;
                kw = *(const uint4*)(kcb + o);
                vw = *(const uint4*)(vcb + o);
            }
        }
        float kf[8], vf[8];
        ik_unpack8(kw, kf);
        ik_unpack8(vw, vf);
        const unsigned dist = g.pp - j;
        const unsigned bi = dist - g.dlo;
        for (int h = 0; h < GP; ++h) {
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s = ffma(qv[h][i], kf[i], s);
            // every lane of the wave runs the shuffles (the slot's lanes are adjacent: xor below lpk stays in the slot)
            for (unsigned o = 1; o < lpk; o <<= 1) s = fadd(s, shx(s, o));
            if (!ok) continue;
            if (bi < g.nb) s = fadd(s, bias[h * g.nb + bi]);
            const float mn = fmaxf(m[h], s);
            const float corr = exp2f(m[h] - mn);
            const float pr = exp2f(s - mn);
            l[h] = ffma(l[h], corr, pr);
            for (int i = 0; i < 8; ++i) acc[h][i] = ffma(pr, vf[i], fmul(acc[h][i], corr));
            m[h] = mn;
        }
    }
    // merge the slots of the wave (xor tree over the slot bits; symmetric, so both partners compute the same value)
    for (unsigned o = lpk; o < 32; o <<= 1) {
        for (int h = 0; h < GP; ++h) {
            const float m2 = shx(m[h], o);
            const float l2 = shx(l[h], o);
            float a2[8];
            for (int i = 0; i < 8; ++i) a2[i] = shx(acc[h][i], o);
            const float mn = fmaxf(m[h], m2);
            if (mn == -INFINITY) continue;
            const float c1 = exp2f(m[h] - mn), c2 = exp2f(m2 - mn);
            l[h] = fadd(fmul(l[h], c1), fmul(l2, c2));
            for (int i = 0; i < 8; ++i) acc[h][i] = fadd(fmul(acc[h][i], c1), fmul(a2[i], c2));
            m[h] = mn;
        }
    }
    if (slot == 0) {
        for (int h = 0; h < GP; ++h) {
            for (int i = 0; i < 8; ++i) S.acc[sg][h][le * 8 + i] = acc[h][i];
            if (le == 0) {
                S.ml[sg][h][0] = m[h];
                S.ml[sg][h][1] = l[h];
            }
        }
    }
    __syncthreads();
    if (tid >= D) return;
    const unsigned i = tid;
    for (int hh = 0; hh < GP; ++hh) {
        const unsigned h = kvh * GP + hh;
        float mm = -INFINITY;
        for (int s = 0; s < NSG; ++s) mm = fmaxf(mm, S.ml[s][hh][0]);
        float ll = 0.0f, aa = 0.0f;
        for (int s = 0; s < NSG; ++s) {
            const float ms = S.ml[s][hh][0];
            const float c = ms == -INFINITY ? 0.0f : exp2f(ms - mm);
            ll = ffma(S.ml[s][hh][1], c, ll);
            aa = ffma(S.acc[s][hh][i], c, aa);
        }
        const size_t pr = (size_t)t * p.nq;
        if (g.ns == 1) {
            const float sink2 = fmul(b2f(p.pq[pr + 2 * H * D + H * p.dr + h]), LOG2E);
            const float gate = b2f(p.pq[pr + H * D + H * p.dr + h * D + i]);
            p.out[((size_t)t * H + h) * D + i] = ink_finish(aa, mm, ll, sink2, gate);
        } else {
            const size_t slotp = ((size_t)t * H + h) * MAXS + split;
            p.scr[slotp * D + i] = aa;
            if (i == 0) {
                float* ml = p.scr + (size_t)p.t * H * MAXS * D;
                ml[slotp * 2] = mm;
                ml[slotp * 2 + 1] = ll;
            }
        }
    }
}

// RelSinkAttn stage 1. Grid (Hk, max splits, T), NSG waves. q, k, v and the caches 16-byte aligned (launcher checks), D in
// {8..128}. Dynamic LDS: the split's bias, [GP][nb] f32 (at most 3 x E x 4 bytes).
template <int GP>
__global__ void __launch_bounds__(NSG * 32) ink_attn_s1_k(InkP p) {
    __shared__ InkTg<GP> S;
    extern __shared__ float tb[];
    const unsigned tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    const unsigned kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z;
    const unsigned D = p.d, H = p.h, Hk = p.hk, E = p.e, Cr = p.cr;
    const bool ring = (p.flags & F_RING) != 0;
    const unsigned pos0 = ink_pos0(p);
    const InkSplit g = ink_split(p, pos0, t, split);

    // the cache write of this row when it cannot race the reads (launcher: linear, or a ring with T <= Cr - window + 1)
    if ((p.flags & F_KV_WRITE) != 0 && split == 0 && (!ring || t + Cr >= (unsigned)p.t)) {
        const unsigned row = ring ? g.pp % Cr : g.pp;
        const size_t src = ((size_t)t * Hk + kvh) * D, dst = ((size_t)kvh * Cr + row) * D;
        for (unsigned i = tid; i < D; i += NSG * 32) {
            p.kc[dst + i] = p.kn[src + i];
            p.vc[dst + i] = p.vn[src + i];
        }
    }
    if (split >= g.ns) return;
    for (unsigned x = tid; x < GP * D; x += NSG * 32) S.q[x / D][x % D] = b2f(p.q[((size_t)t * H + kvh * GP) * D + x]);
    for (unsigned x = tid; x < GP * g.nb; x += NSG * 32) {
        const unsigned h = x / g.nb, b = x % g.nb;
        tb[h * g.nb + b] = fmul(b2f(p.rel[((size_t)t * H + kvh * GP + h) * E + g.dlo + b]), LOG2E);
    }
    __syncthreads();
    ink_attn_body<GP>(S, tb, g, kvh, split, t, pos0, p, tid, lane, sg);
}

// stage 2. Grid (H, T), threads >= D.
__global__ void ink_attn_comb_k(InkP p) {
    const unsigned tid = threadIdx.x;
    const unsigned h = blockIdx.x, t = blockIdx.y, D = p.d, H = p.h;
    if (tid >= D) return;
    const unsigned pp = ink_pos0(p) + t;
    const unsigned vis = pp - ink_lo(pp, p.window) + 1;
    const unsigned ns = ink_nsplit(vis, p.sk);
    if (ns == 1) return;
    const size_t base = ((size_t)t * H + h) * MAXS;
    const float* ml = p.scr + (size_t)p.t * H * MAXS * D;
    float mm = -INFINITY;
    for (unsigned s = 0; s < ns; ++s) mm = fmaxf(mm, ml[(base + s) * 2]);
    float ll = 0.0f, aa = 0.0f;
    for (unsigned s = 0; s < ns; ++s) {
        const float ms = ml[(base + s) * 2];
        const float c = ms == -INFINITY ? 0.0f : exp2f(ms - mm);
        ll = ffma(ml[(base + s) * 2 + 1], c, ll);
        aa = ffma(p.scr[(base + s) * D + tid], c, aa);
    }
    const size_t pr = (size_t)t * p.nq;
    const float sink2 = fmul(b2f(p.pq[pr + 2 * H * D + H * p.dr + h]), LOG2E);
    const float gate = b2f(p.pq[pr + H * D + H * p.dr + h * D + tid]);
    p.out[((size_t)t * H + h) * D + tid] = ink_finish(aa, mm, ll, sink2, gate);
}

// the cache write as its own pass. One thread per (row, kv head, dim).
__global__ void ink_kv_write_k(InkP p) {
    const unsigned D = p.d, Hk = p.hk, Cr = p.cr;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)p.t * Hk * D) return;
    const unsigned t = gid / (Hk * D), r = gid % (Hk * D), kvh = r / D, i = r % D;
    const bool ring = (p.flags & F_RING) != 0;
    if (ring && t + Cr < (unsigned)p.t) return;
    const unsigned pp = ink_pos0(p) + t;
    const unsigned row = ring ? pp % Cr : pp;
    const size_t dst = ((size_t)kvh * Cr + row) * D + i;
    p.kc[dst] = p.kn[gid];
    p.vc[dst] = p.vn[gid];
}

// ------------------------------------------------------------------------------------------------------------------------
// launchers
// ------------------------------------------------------------------------------------------------------------------------

int chk(const ik_tensor* t, int ndim) { return ik_check(t, IK_DTYPE_BF16, ndim); }
const bf16* cb(const ik_tensor* t) { return (const bf16*)t->data; }
bf16* mb(ik_tensor* t) { return (bf16*)t->data; }
int fits_u32(int64_t v) { return v >= 0 && v <= 0x7fffffffLL; }
int split_keys(const ik_ink_attn_attrs* a) { return a->split_keys > 0 ? (int)a->split_keys : SPLIT_KEYS; }

unsigned bias_rows(unsigned vis, unsigned e, unsigned sk) {
    const unsigned skr = (sk + 31) / 32 * 32;
    const unsigned ns = ink_nsplit(vis, sk);
    const unsigned chunk = ink_chunk(vis, ns);
    unsigned b = chunk > skr ? chunk : skr;
    return b < e ? b : e;
}

// grid splits and bias rows of an attention launch (device position: the capacity bound, never the host pos0)
void attn_geometry(const InkP& p, bool dev, unsigned* gy, unsigned* nbias) {
    unsigned vmax;
    if (dev) {
        vmax = p.window > 0 ? (unsigned)(p.window < p.cr ? p.window : p.cr) : (unsigned)p.cr;
    } else {
        const unsigned last = p.pos0 + p.t - 1;
        vmax = last - ink_lo(last, p.window) + 1;
    }
    *gy = ink_nsplit(vmax, p.sk);
    *nbias = bias_rows(vmax, p.e, p.sk);
}

}  // namespace

extern "C" int ik_ink_kv_prologue_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                                       const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_ink_attn_attrs* a, ik_tensor* q,
                                       ik_tensor* rel, ik_tensor* k, ik_tensor* v, ik_tensor* kring, ik_tensor* vring, const ik_tensor* pos,
                                       void* stream) {
    if (!a) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_pos(pos));
    IK_TRY(chk(pq, 2));
    IK_TRY(chk(pk, 2));
    IK_TRY(chk(pv, 2));
    IK_TRY(chk(kt, 2));
    IK_TRY(chk(vt, 2));
    IK_TRY(chk(qs, 2));
    IK_TRY(chk(rs, 2));
    IK_TRY(chk(rp, 2));
    IK_TRY(chk(kring, 2));
    IK_TRY(chk(vring, 2));
    IK_TRY(chk(q, 3));
    IK_TRY(chk(rel, 3));
    IK_TRY(chk(k, 3));
    IK_TRY(chk(v, 3));
    const int64_t h = a->n_heads, hk = a->n_kv_heads, d = a->head_dim, dr = a->d_rel;
    if (hk <= 0 || h % hk != 0 || h / hk > 4 || d <= 0 || d > 256 || dr <= 0 || dr > 32 || kt->shape[0] > KMAX || kt->shape[0] < 1)
        return IK_ERR_SHAPE;
    InkP p = {};
    p.pq = cb(pq);
    p.pk = cb(pk);
    p.pv = cb(pv);
    p.kt = cb(kt);
    p.vt = cb(vt);
    p.qs = cb(qs);
    p.rs = cb(rs);
    p.rp = cb(rp);
    p.kr = mb(kring);
    p.vr = mb(vring);
    p.t = (int)pq->shape[0];
    p.h = (int)h;
    p.hk = (int)hk;
    p.d = (int)d;
    p.dr = (int)dr;
    p.e = (int)rp->shape[1];
    p.k = (int)kt->shape[0];
    p.rr = (int)kring->shape[0];
    p.nq = (int)pq->shape[1];
    p.pw = (int)pk->shape[1];
    p.tau_floor = (int)a->tau_floor;
    p.qk_clip = a->qk_clip;
    p.rel_clip = a->rel_clip;
    p.eps = a->eps;
    p.tau_alpha = a->tau_alpha;
    p.cscale = a->dyn_conv ? powf((float)(2 * p.k + 1), -0.5f) : powf((float)(p.k + 1), -0.5f);
    if (a->dyn_conv) p.flags |= F_DYN;
    if (!fits_u32(a->pos0) || !fits_u32((int64_t)p.t * hk * d) || kring->shape[1] != hk * d || vring->shape[0] != kring->shape[0] ||
        vring->shape[1] != kring->shape[1] || p.rr < 1)
        return IK_ERR_SHAPE;
    if (p.t == 0) return IK_OK;
    p.qo = mb(q);
    p.relo = mb(rel);
    p.ko = mb(k);
    p.vo = mb(v);
    p.pos = pos ? (const int*)pos->data : nullptr;
    p.pos0 = (int)a->pos0;
    const bool fused = p.t + p.k <= p.rr + 1;
    if (fused) p.flags |= F_RING_WRITE;
    hipStream_t st = IK_STREAM(stream);
    // decode (T <= 4): the rel tail in slices of 256 elements over more blocks (latency, not work); otherwise one block per
    // (kv head, row). The elements' arithmetic does not depend on the slicing.
    unsigned nz = p.t <= 4 ? (unsigned)((p.e + 255) / 256) : 1u;
    if (nz > 8) nz = 8;
    if (nz < 1) nz = 1;
    if (p.d <= 128) ink_prologue_k<4><<<dim3(p.hk, p.t, nz), 256, 0, st>>>(p);
    else ink_prologue_k<JMAX><<<dim3(p.hk, p.t, nz), 256, 0, st>>>(p);
    if (!fused) {
        const size_t n = (size_t)p.t * p.hk * p.d;
        ink_ring_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
    }
    return ik_launch_status();
}

extern "C" int ik_rel_sink_attn_bf16(const ik_tensor* q, const ik_tensor* k, const ik_tensor* v, const ik_tensor* kc_in, const ik_tensor* vc_in,
                                     const ik_tensor* rel, const ik_tensor* pq, const ik_ink_attn_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc,
                                     const ik_scratch* scratch, const ik_tensor* pos, void* stream) {
    (void)kc_in;
    (void)vc_in;
    if (!a) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_pos(pos));
    IK_TRY(chk(q, 3));
    IK_TRY(chk(k, 3));
    IK_TRY(chk(v, 3));
    IK_TRY(chk(rel, 3));
    IK_TRY(chk(pq, 2));
    IK_TRY(chk(o, 2));
    IK_TRY(chk(kc, 3));
    IK_TRY(chk(vc, 3));
    InkP p = {};
    p.t = (int)q->shape[0];
    if (p.t == 0) return IK_OK;
    p.h = (int)a->n_heads;
    p.hk = (int)a->n_kv_heads;
    p.d = (int)a->head_dim;
    p.dr = (int)a->d_rel;
    const int G = p.hk > 0 ? p.h / p.hk : 0;
    if (G < 1 || G > 3 || p.h % p.hk != 0 || !(p.d == 8 || p.d == 16 || p.d == 32 || p.d == 64 || p.d == 128)) return IK_ERR_SHAPE;
    p.e = (int)rel->shape[2];
    p.cr = (int)kc->shape[1];
    p.window = (int)a->window;
    p.nq = (int)pq->shape[1];
    p.scale = a->scale;
    p.sk = split_keys(a);
    p.pos0 = (int)a->pos0;
    const bool ring = a->kv_ring != 0;
    if (ring) p.flags |= F_RING;
    if (ring && (p.window < 1 || p.window > p.cr)) return IK_ERR_SHAPE;
    const ik_tensor* al[5] = {q, k, v, kc, vc};
    for (const ik_tensor* x : al)
        if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    const int nslice = p.t < SLICE ? p.t : SLICE;
    if (!fits_u32(a->pos0) || !fits_u32(a->pos0 + p.t) || !fits_u32((int64_t)nslice * p.h * MAXS * (p.d + 2)) ||
        !fits_u32((int64_t)p.hk * p.cr * p.d))
        return IK_ERR_SHAPE;
    p.q = cb(q);
    p.kn = cb(k);
    p.vn = cb(v);
    p.rel = cb(rel);
    p.pq = cb(pq);
    p.out = mb(o);
    p.kc = mb(kc);
    p.vc = mb(vc);
    p.pos = pos ? (const int*)pos->data : nullptr;
    hipStream_t st = IK_STREAM(stream);
    // slices of at most SLICE rows in order: a slice reads the cache rows the slices before it wrote (its keys below its own
    // first position), so every row's bits are those of the one-row call
    for (int r0 = 0; r0 < p.t; r0 += SLICE) {
        InkP ps = p;
        const int n = p.t - r0 < SLICE ? p.t - r0 : SLICE;
        ps.t = n;
        ps.toff = r0;
        ps.pos0 = p.pos0 + r0;
        ps.q = p.q + (size_t)r0 * p.h * p.d;
        ps.kn = p.kn + (size_t)r0 * p.hk * p.d;
        ps.vn = p.vn + (size_t)r0 * p.hk * p.d;
        ps.rel = p.rel + (size_t)r0 * p.h * p.e;
        ps.pq = p.pq + (size_t)r0 * p.nq;
        ps.out = p.out + (size_t)r0 * p.h * p.d;
        const bool kv_fused = !ring || n + p.window <= p.cr + 1;
        if (kv_fused) ps.flags |= F_KV_WRITE;
        unsigned gy, nb;
        attn_geometry(ps, pos != nullptr, &gy, &nb);
        if (gy > 1) {
            if (!scratch || !scratch->data || scratch->bytes < (int64_t)n * p.h * MAXS * (p.d + 2) * 4) return IK_ERR_INVALID_ARG;
            ps.scr = (float*)scratch->data;
        }
        const size_t tbb = (size_t)G * nb * 4;
        const dim3 grid(p.hk, gy, n);
        switch (G) {
            case 1: ink_attn_s1_k<1><<<grid, NSG * 32, tbb, st>>>(ps); break;
            case 2: ink_attn_s1_k<2><<<grid, NSG * 32, tbb, st>>>(ps); break;
            default: ink_attn_s1_k<3><<<grid, NSG * 32, tbb, st>>>(ps); break;
        }
        if (gy > 1) ink_attn_comb_k<<<dim3(p.h, n), (p.d + 31) / 32 * 32, 0, st>>>(ps);
        if (!kv_fused) {
            const size_t nn = (size_t)n * p.hk * p.d;
            ink_kv_write_k<<<(unsigned)((nn + 255) / 256), 256, 0, st>>>(ps);
        }
    }
    return ik_launch_status();
}
