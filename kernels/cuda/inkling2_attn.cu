// Inkling-2-Mini attention ops on CUDA (lane ink-cuda-attn; contract: crates/intarsia-core/src/inkling2/ops.rs, f64 reference
// crates/intarsia-backend-cpu/src/ink_ref.rs; launcher crates/intarsia-backend-cuda/src/inkattn.rs; ABI ik_cuda_abi_ink.h).
// A port of the Metal kernels (kernels/metal/inkling2_attn.metal, lane ink-m-attn), the same structure and arithmetic:
//
//   ink_prologue_k     InkKvPrologue: one block per (kv head, row). K / V short conv (static or dynamic taps), k clip, v unit
//                      RMS, q / r scale + clip (+ tau), rel = r . relproj, the conv-ring write when it cannot race the conv reads
//                      of other rows (T <= Rr - K + 1), else ink_ring_write_k runs after it.
//   ink_attn_s1_k<G>   RelSinkAttn stage 1: one block per (kv head, split, row), 4 warps. The split count and the key -> (split,
//                      warp, slot, step) map depend only on the row's position (and the window), never on T, pos0 or the grid:
//                      an R-row call is bitwise R one-row calls (rows-exact), host or device position alike. With one split it
//                      writes the output, else a partial (max, sum, accumulator) into the scratch.
//   ink_attn_dec_k<G>  InkAttnDecode: the two above in one block per (kv head, split, row) for 1..16 rows; each block recomputes
//                      its row's q / r, its split's bias and the call's own K / V rows inside its split with the prologue's
//                      device functions, so the result is bitwise the pair.
//   ink_attn_comb_k    stage 2: merges the partials of a (row, query head) in split order, then the sink and the gate.
//   ink_kv_write_k     the cache write after the attention when it could race the reads (ring with T > Cr - window + 1).
//
// Scores live in the log2 domain: q is multiplied by scale * log2(e), the rel bias and the sink by log2(e) (the bias is not
// scaled by `scale`; zero for distances >= E). The sink enters once, in the final normalisation. Every reduction has a fixed
// order; the arithmetic the prologue and the fused kernel share is written with explicit _rn intrinsics so nvcc's FMA
// contraction cannot make the two differ.
#include "ik_launch.h"
#include "ik_cuda_abi_ink.h"
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

typedef __nv_bfloat16 bf16;

constexpr int MAXS = 32;        // most splits (scratch stride)
constexpr int ALIGN = 32;       // split chunks are a multiple of this
constexpr int NSG = 4;          // warps of an attention block
constexpr int SPLIT_KEYS = 64;  // default keys per split
constexpr int MAXROWS = 16;     // InkAttnDecode rows
constexpr float LOG2E = 1.4426950408889634f;
constexpr float SRC_EPS = 1e-6f;

// flags
constexpr int F_DYN = 1;        // dynamic conv taps
constexpr int F_RING = 2;       // K / V caches are rings
constexpr int F_KV_WRITE = 4;   // cache write in the attention kernel (else ink_kv_write_k after it)
constexpr int F_RING_WRITE = 16;  // conv-ring write in the kernel (else ink_ring_write_k after it)

struct InkP {
    const bf16 *pq, *pk, *pv, *kt, *vt, *qs, *rs, *rp;
    const bf16 *q, *kn, *vn, *rel;
    bf16 *qo, *relo, *ko, *vo, *kr, *vr, *kc, *vc, *out;
    float* scr;
    bf16* stg;  // fused: staged K / V rows for ink_kv_write_k ([2][T][Hk][D])
    const int* pos;
    int t, h, hk, d, dr, e, k, rr, cr, pos0, flags, tau_floor, window, nq, pw, sk;
    float qk_clip, rel_clip, eps, tau_alpha, scale, cscale;
};

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rA(float x) { return b2f(f2b(x)); }

__device__ __forceinline__ int ink_pos0(const InkP& p) { return p.pos ? max(p.pos[0], 0) : p.pos0; }

__host__ __device__ __forceinline__ unsigned ink_lo(unsigned pp, unsigned window) {
    return (window > 0 && pp + 1 > window) ? pp + 1 - window : 0u;
}

// a split holds at least `sk` keys
__host__ __device__ __forceinline__ unsigned ink_nsplit(unsigned vis, unsigned sk) {
    unsigned n = (vis + sk - 1) / sk;
    return n < 1 ? 1u : (n > (unsigned)MAXS ? (unsigned)MAXS : n);
}

__host__ __device__ __forceinline__ unsigned ink_chunk(unsigned vis, unsigned ns) {
    const unsigned c = (vis + ns - 1) / ns;
    return (c + ALIGN - 1) / ALIGN * ALIGN;
}

// butterfly sum over the warp: every lane gets the same bits
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float ink_sigmoid(float z) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-z))); }

// ------------------------------------------------------------------------------------------------------------------------
// InkKvPrologue arithmetic (shared by the prologue and the fused kernel)
// ------------------------------------------------------------------------------------------------------------------------

// conv input of channel `ch` at position q (row stride `pw` in the call's input, ring rows `rr` of width `ckv`)
__device__ __forceinline__ float ink_cx(const bf16* x, const bf16* ring, long q, unsigned pos0, unsigned pw, unsigned rr, unsigned ckv,
                                        unsigned ch) {
    if (q < 0) return 0.0f;
    if ((unsigned)q >= pos0) return b2f(x[(size_t)((unsigned)q - pos0) * pw + ch]);
    return b2f(ring[(size_t)((unsigned)q % rr) * ckv + ch]);
}

// one conv output (rounded): (sum_l (taps[l] + dyn[l]) * x(p - l) + x(p)) * s
__device__ __forceinline__ float ink_conv(const bf16* x, const bf16* ring, const bf16* taps, unsigned t, unsigned pos0, const InkP& p,
                                          unsigned ch, unsigned kvh) {
    const unsigned ckv = p.hk * p.d;
    const long pp = (long)(pos0 + t);
    const bool dyn = (p.flags & F_DYN) != 0;
    float acc = 0.0f;
    for (int l = 0; l < p.k; ++l) {
        float w = b2f(taps[(size_t)l * ckv + ch]);
        if (dyn) w = __fadd_rn(w, b2f(x[(size_t)t * p.pw + ckv + kvh * p.k + l]));
        acc = __fmaf_rn(w, ink_cx(x, ring, pp - (long)l, pos0, p.pw, p.rr, ckv, ch), acc);
    }
    return rA(__fmul_rn(__fadd_rn(acc, b2f(x[(size_t)t * p.pw + ch])), p.cscale));
}

__device__ __forceinline__ float ink_clipf(float ss, float th) { return fminf(__fdiv_rn(th, __fadd_rn(__fsqrt_rn(ss), SRC_EPS)), 1.0f); }

__device__ __forceinline__ float ink_tau(const InkP& p, unsigned pp) {
    if (p.tau_floor == 0) return 1.0f;
    const float r = fmaxf(__fdiv_rn((float)(pp + 1), (float)p.tau_floor), 1.0f);
    return __fmaf_rn(p.tau_alpha, logf(r), 1.0f);
}

__device__ __forceinline__ float ink_inv_rms(float ss, const InkP& p) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)p.d), p.eps)));
}

// q (or r) after the scale, clip and tau, rounded
__device__ __forceinline__ float ink_qr(float raw, float ss, float th, bool use_tau, float tau) {
    float x = rA(__fmul_rn(raw, ink_clipf(ss, th)));
    if (use_tau) x = rA(__fmul_rn(x, tau));
    return x;
}

// one rel element: the r . relproj chain (j ascending), unrounded
__device__ __forceinline__ float ink_rel(const float* r, const bf16* rp, int dr, int E, int e) {
    float a = 0.0f;
    for (int j = 0; j < dr; ++j) a = __fmaf_rn(r[j], b2f(rp[(size_t)j * E + e]), a);
    return a;
}

// Grid (Hk, T), threads a multiple of 32 >= max(D, dr), at most 256.
__global__ void ink_prologue_k(InkP p) {
    __shared__ float part[8][10];
    __shared__ float rsh[4][32];
    const unsigned tid = threadIdx.x, tpg = blockDim.x, lane = tid & 31, sg = tid >> 5, nsg = tpg >> 5;
    const unsigned kvh = blockIdx.x, t = blockIdx.y;
    const unsigned D = p.d, H = p.h, G = p.h / p.hk, dr = p.dr, E = p.e;
    const unsigned ckv = p.hk * D;
    const unsigned pos0 = ink_pos0(p);
    const unsigned pp = pos0 + t;
    const unsigned c = tid, ch = kvh * D + c;
    const size_t qrow = (size_t)t * p.nq;

    float vals[10];
    for (int i = 0; i < 10; ++i) vals[i] = 0.0f;
    float kc = 0.0f, vc = 0.0f;
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f}, rv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (c < D) {
        kc = ink_conv(p.pk, p.kr, p.kt, t, pos0, p, ch, kvh);
        vc = ink_conv(p.pv, p.vr, p.vt, t, pos0, p, ch, kvh);
        vals[0] = __fmul_rn(kc, kc);
        vals[1] = __fmul_rn(vc, vc);
        for (unsigned g = 0; g < G; ++g) {
            const unsigned h = kvh * G + g;
            qv[g] = __fmul_rn(b2f(p.pq[qrow + h * D + c]), b2f(p.qs[h * D + c]));
            vals[2 + g] = __fmul_rn(qv[g], qv[g]);
        }
    }
    if (c < dr) {
        for (unsigned g = 0; g < G; ++g) {
            const unsigned h = kvh * G + g;
            rv[g] = __fmul_rn(b2f(p.pq[qrow + H * D + h * dr + c]), b2f(p.rs[h * dr + c]));
            vals[2 + G + g] = __fmul_rn(rv[g], rv[g]);
        }
    }
    const unsigned nv = 2 + 2 * G;
    for (unsigned i = 0; i < nv; ++i) {
        const float s = warp_sum(vals[i]);
        if (lane == 0) part[sg][i] = s;
    }
    __syncthreads();
    float tot[10];
    for (unsigned i = 0; i < nv; ++i) {
        float s = 0.0f;
        for (unsigned j = 0; j < nsg; ++j) s = __fadd_rn(s, part[j][i]);
        tot[i] = s;
    }
    const float tau = ink_tau(p, pp);
    const bool use_tau = p.tau_floor > 0;
    if (c < D) {
        const size_t kvo = ((size_t)t * p.hk + kvh) * D + c;
        p.ko[kvo] = f2b(__fmul_rn(kc, ink_clipf(tot[0], p.qk_clip)));
        p.vo[kvo] = f2b(__fmul_rn(vc, ink_inv_rms(tot[1], p)));
        for (unsigned g = 0; g < G; ++g) {
            const unsigned h = kvh * G + g;
            p.qo[((size_t)t * H + h) * D + c] = f2b(ink_qr(qv[g], tot[2 + g], p.qk_clip, use_tau, tau));
        }
    }
    if (c < dr) {
        for (unsigned g = 0; g < G; ++g) rsh[g][c] = ink_qr(rv[g], tot[2 + G + g], p.rel_clip, use_tau, tau);
    }
    // the conv rings: only when no other row of the call still reads the row written (launcher: T <= Rr - K + 1)
    if ((p.flags & F_RING_WRITE) != 0 && c < D && t + p.rr >= (unsigned)p.t) {
        const size_t ro = (size_t)(pp % p.rr) * ckv + ch;
        p.kr[ro] = p.pk[(size_t)t * p.pw + ch];
        p.vr[ro] = p.pv[(size_t)t * p.pw + ch];
    }
    __syncthreads();
    for (unsigned e = tid; e < E; e += tpg) {
        for (unsigned g = 0; g < G; ++g) {
            p.relo[((size_t)t * H + kvh * G + g) * E + e] = f2b(ink_rel(rsh[g], p.rp, dr, E, e));
        }
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
__device__ __forceinline__ bf16 ink_finish(float a, float m, float l, float sink2, float gate) {
    const float mt = fmaxf(m, sink2);
    const float sc = exp2f(m - mt);
    const float lt = __fmaf_rn(l, sc, exp2f(sink2 - mt));
    return f2b(__fmul_rn(__fdiv_rn(__fmul_rn(a, sc), lt), ink_sigmoid(gate)));
}

__device__ __forceinline__ void unpack8(uint4 w, float* f) {
    f[0] = __uint_as_float(w.x << 16);
    f[1] = __uint_as_float(w.x & 0xffff0000u);
    f[2] = __uint_as_float(w.y << 16);
    f[3] = __uint_as_float(w.y & 0xffff0000u);
    f[4] = __uint_as_float(w.z << 16);
    f[5] = __uint_as_float(w.z & 0xffff0000u);
    f[6] = __uint_as_float(w.w << 16);
    f[7] = __uint_as_float(w.w & 0xffff0000u);
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

__device__ __forceinline__ InkSplit ink_split(const InkP& p, unsigned pos0, unsigned t, unsigned split) {
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
// the partial. KV_TG: the call's own keys (j >= pos0) come from block rows `kvt` instead of the k / v inputs.
template <int GP, bool KV_TG>
__device__ __forceinline__ void ink_attn_body(InkTg<GP>& S, const float* bias, const bf16* kvt, unsigned kvoff, InkSplit g, unsigned kvh,
                                              unsigned split, unsigned t, unsigned pos0, const bf16* kn, const bf16* vn, const bf16* kc,
                                              const bf16* vc, const bf16* pq, bf16* out, float* scr, const InkP& p, unsigned tid,
                                              unsigned lane, unsigned sg) {
    const unsigned D = p.d, H = p.h, Hk = p.hk, Cr = p.cr;
    const bool ring = (p.flags & F_RING) != 0;
    const unsigned lpk = D / 8;     // lanes per key row
    const unsigned kps = 32 / lpk;  // keys per warp step
    const unsigned slot = lane / lpk, le = lane % lpk;
    const float qsc = __fmul_rn(p.scale, LOG2E);
    float qv[GP][8];
    for (int h = 0; h < GP; ++h)
        for (int i = 0; i < 8; ++i) qv[h][i] = __fmul_rn(S.q[h][le * 8 + i], qsc);
    float m[GP], l[GP], acc[GP][8];
    for (int h = 0; h < GP; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        for (int i = 0; i < 8; ++i) acc[h][i] = 0.0f;
    }
    const unsigned rbase = ring ? g.lo % Cr : 0u;
    const bf16* kcb = kc + (size_t)kvh * Cr * D;
    const bf16* vcb = vc + (size_t)kvh * Cr * D;
    for (unsigned base = g.ks + sg * kps; base < g.ke; base += NSG * kps) {
        const unsigned j = base + slot;
        const bool ok = j < g.ke;
        uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
        if (ok) {
            if (j >= pos0) {
                if (KV_TG) {
                    const unsigned ro = (j - pos0 - kvoff) * D + le * 8;
                    kw = *(const uint4*)(kvt + ro);
                    vw = *(const uint4*)(kvt + (size_t)p.t * D + ro);
                } else {
                    const size_t o = ((size_t)(j - pos0) * Hk + kvh) * D + le * 8;
                    kw = __ldg((const uint4*)(kn + o));
                    vw = __ldg((const uint4*)(vn + o));
                }
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
        unpack8(kw, kf);
        unpack8(vw, vf);
        const unsigned dist = g.pp - j;
        const unsigned bi = dist - g.dlo;
        for (int h = 0; h < GP; ++h) {
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s = __fmaf_rn(qv[h][i], kf[i], s);
            for (unsigned o = 1; o < lpk; o <<= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
            if (!ok) continue;
            if (bi < g.nb) s = __fadd_rn(s, bias[h * g.nb + bi]);
            const float mn = fmaxf(m[h], s);
            const float corr = exp2f(m[h] - mn);
            const float pr = exp2f(s - mn);
            l[h] = __fmaf_rn(l[h], corr, pr);
            for (int i = 0; i < 8; ++i) acc[h][i] = __fmaf_rn(pr, vf[i], __fmul_rn(acc[h][i], corr));
            m[h] = mn;
        }
    }
    // merge the slots of the warp (xor tree over the slot bits; symmetric, so both partners compute the same value)
    for (unsigned o = lpk; o < 32; o <<= 1) {
        for (int h = 0; h < GP; ++h) {
            const float m2 = __shfl_xor_sync(0xffffffffu, m[h], o);
            const float l2 = __shfl_xor_sync(0xffffffffu, l[h], o);
            float a2[8];
            for (int i = 0; i < 8; ++i) a2[i] = __shfl_xor_sync(0xffffffffu, acc[h][i], o);
            const float mn = fmaxf(m[h], m2);
            if (mn == -INFINITY) continue;
            const float c1 = exp2f(m[h] - mn), c2 = exp2f(m2 - mn);
            l[h] = __fadd_rn(__fmul_rn(l[h], c1), __fmul_rn(l2, c2));
            for (int i = 0; i < 8; ++i) acc[h][i] = __fadd_rn(__fmul_rn(acc[h][i], c1), __fmul_rn(a2[i], c2));
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
            ll = __fmaf_rn(S.ml[s][hh][1], c, ll);
            aa = __fmaf_rn(S.acc[s][hh][i], c, aa);
        }
        const size_t pr = (size_t)t * p.nq;
        if (g.ns == 1) {
            const float sink2 = __fmul_rn(b2f(pq[pr + 2 * H * D + H * p.dr + h]), LOG2E);
            const float gate = b2f(pq[pr + H * D + H * p.dr + h * D + i]);
            out[((size_t)t * H + h) * D + i] = ink_finish(aa, mm, ll, sink2, gate);
        } else {
            const size_t slotp = ((size_t)t * H + h) * MAXS + split;
            scr[slotp * D + i] = aa;
            if (i == 0) {
                float* ml = scr + (size_t)p.t * H * MAXS * D;
                ml[slotp * 2] = mm;
                ml[slotp * 2 + 1] = ll;
            }
        }
    }
}

// RelSinkAttn stage 1. Grid (Hk, max splits, T), NSG warps. q, k, v and the caches 16-byte aligned (launcher checks), D in
// {8..128}. Dynamic shared memory: the split's bias, [GP][nb] f32.
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
        tb[h * g.nb + b] = __fmul_rn(b2f(p.rel[((size_t)t * H + kvh * GP + h) * E + g.dlo + b]), LOG2E);
    }
    __syncthreads();
    ink_attn_body<GP, false>(S, tb, nullptr, 0, g, kvh, split, t, pos0, p.kn, p.vn, p.kc, p.vc, p.pq, p.out, p.scr, p, tid, lane, sg);
}

// InkAttnDecode: InkKvPrologue + RelSinkAttn in one stage-1 block (1..=16 rows). Grid (Hk, max splits, T), NSG warps; D in
// {8..128}, dr <= 32. Dynamic shared memory: the bias [GP][nb] f32, then the K / V rows of the call's keys in the split
// ([2][T][D] bf16, 16-byte aligned).
template <int GP>
__global__ void __launch_bounds__(NSG * 32) ink_attn_dec_k(InkP p, unsigned tb_floats) {
    __shared__ InkTg<GP> S;
    __shared__ float part[NSG][32];
    __shared__ float rsh[GP][32];
    extern __shared__ float tb[];
    bf16* tkv = (bf16*)(tb + tb_floats);
    const unsigned tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    const unsigned kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z;
    const unsigned D = p.d, H = p.h, Hk = p.hk, dr = p.dr, E = p.e, Cr = p.cr;
    const unsigned ckv = Hk * D;
    const bool ring = (p.flags & F_RING) != 0;
    const unsigned pos0 = ink_pos0(p);
    const InkSplit g = ink_split(p, pos0, t, split);
    if (split >= g.ns) return;
    const unsigned c = tid, ch = kvh * D + c;
    const size_t qrow = (size_t)t * p.nq;

    // q and r of the row (the prologue's arithmetic)
    float qv[GP], rv[GP];
    {
        float vals[2 * GP];
        for (int h = 0; h < GP; ++h) {
            qv[h] = 0.0f;
            rv[h] = 0.0f;
            if (c < D) qv[h] = __fmul_rn(b2f(p.pq[qrow + (kvh * GP + h) * D + c]), b2f(p.qs[(kvh * GP + h) * D + c]));
            if (c < dr) rv[h] = __fmul_rn(b2f(p.pq[qrow + H * D + (kvh * GP + h) * dr + c]), b2f(p.rs[(kvh * GP + h) * dr + c]));
            vals[h] = __fmul_rn(qv[h], qv[h]);
            vals[GP + h] = __fmul_rn(rv[h], rv[h]);
        }
        for (int x = 0; x < 2 * GP; ++x) {
            const float s = warp_sum(vals[x]);
            if (lane == 0) part[sg][x] = s;
        }
    }
    __syncthreads();
    {
        const float tau = ink_tau(p, g.pp);
        const bool use_tau = p.tau_floor > 0;
        for (int h = 0; h < GP; ++h) {
            float tq = 0.0f, tr = 0.0f;
            for (int j = 0; j < NSG; ++j) tq = __fadd_rn(tq, part[j][h]);
            for (int j = 0; j < NSG; ++j) tr = __fadd_rn(tr, part[j][GP + h]);
            if (c < D) S.q[h][c] = ink_qr(qv[h], tq, p.qk_clip, use_tau, tau);
            if (c < dr) rsh[h][c] = ink_qr(rv[h], tr, p.rel_clip, use_tau, tau);
        }
    }
    // the call's own keys inside the split: rows t0 .. t0 + nnew (positions pos0 + t0 ..)
    const unsigned jn0 = max(g.ks, pos0), jn1 = g.ke;
    const unsigned t0 = jn0 - pos0, nnew = jn1 > jn0 ? jn1 - jn0 : 0u;
    float kcv[MAXROWS], vcv[MAXROWS];
    for (unsigned r = 0; r < nnew; ++r) {
        kcv[r] = 0.0f;
        vcv[r] = 0.0f;
        if (c < D) {
            kcv[r] = ink_conv(p.pk, p.kr, p.kt, t0 + r, pos0, p, ch, kvh);
            vcv[r] = ink_conv(p.pv, p.vr, p.vt, t0 + r, pos0, p, ch, kvh);
        }
    }
    __syncthreads();  // part reused
    for (unsigned r = 0; r < nnew; ++r) {
        const float a = warp_sum(__fmul_rn(kcv[r], kcv[r]));
        const float b = warp_sum(__fmul_rn(vcv[r], vcv[r]));
        if (lane == 0) {
            part[sg][2 * r] = a;
            part[sg][2 * r + 1] = b;
        }
    }
    // the bias of the split's distances (the prologue's rel element: the r . relproj chain, rounded)
    for (unsigned x = tid; x < GP * g.nb; x += NSG * 32) {
        const unsigned h = x / g.nb, e = g.dlo + x % g.nb;
        tb[h * g.nb + x % g.nb] = __fmul_rn(rA(ink_rel(rsh[h], p.rp, dr, E, e)), LOG2E);
    }
    __syncthreads();
    for (unsigned r = 0; r < nnew; ++r) {
        float tk = 0.0f, tv = 0.0f;
        for (int j = 0; j < NSG; ++j) tk = __fadd_rn(tk, part[j][2 * r]);
        for (int j = 0; j < NSG; ++j) tv = __fadd_rn(tv, part[j][2 * r + 1]);
        if (c < D) {
            tkv[r * D + c] = f2b(__fmul_rn(kcv[r], ink_clipf(tk, p.qk_clip)));
            tkv[(p.t + r) * D + c] = f2b(__fmul_rn(vcv[r], ink_inv_rms(tv, p)));
        }
    }
    __syncthreads();
    // the block that holds the row's own key (the split with ke == pp + 1) writes its K / V row and its conv-ring row
    if (g.ks < g.ke && g.ke == g.pp + 1 && c < D) {
        const unsigned r = t - t0;
        if (!ring || t + Cr >= (unsigned)p.t) {
            if ((p.flags & F_KV_WRITE) != 0) {
                const unsigned row = ring ? g.pp % Cr : g.pp;
                const size_t dst = ((size_t)kvh * Cr + row) * D + c;
                p.kc[dst] = tkv[r * D + c];
                p.vc[dst] = tkv[(p.t + r) * D + c];
            } else {
                const size_t o = ((size_t)t * Hk + kvh) * D + c;
                p.stg[o] = tkv[r * D + c];
                p.stg[(size_t)p.t * ckv + o] = tkv[(p.t + r) * D + c];
            }
        }
        if ((p.flags & F_RING_WRITE) != 0 && t + p.rr >= (unsigned)p.t) {
            const size_t ro = (size_t)(g.pp % p.rr) * ckv + ch;
            p.kr[ro] = p.pk[(size_t)t * p.pw + ch];
            p.vr[ro] = p.pv[(size_t)t * p.pw + ch];
        }
    }
    // staged row r is the key at position pos0 + t0 + r
    ink_attn_body<GP, true>(S, tb, tkv, t0, g, kvh, split, t, pos0, p.pk, p.pv, p.kc, p.vc, p.pq, p.out, p.scr, p, tid, lane, sg);
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
        ll = __fmaf_rn(ml[(base + s) * 2 + 1], c, ll);
        aa = __fmaf_rn(p.scr[(base + s) * D + tid], c, aa);
    }
    const size_t pr = (size_t)t * p.nq;
    const float sink2 = __fmul_rn(b2f(p.pq[pr + 2 * H * D + H * p.dr + h]), LOG2E);
    const float gate = b2f(p.pq[pr + H * D + H * p.dr + h * D + tid]);
    p.out[((size_t)t * H + h) * D + tid] = ink_finish(aa, mm, ll, sink2, gate);
}

// the cache write as its own pass. One thread per (row, kv head, dim); source k / v (or the fused kernel's staging).
__global__ void ink_kv_write_k(InkP p, const bf16* kn, const bf16* vn) {
    const unsigned D = p.d, Hk = p.hk, Cr = p.cr;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)p.t * Hk * D) return;
    const unsigned t = gid / (Hk * D), r = gid % (Hk * D), kvh = r / D, i = r % D;
    const bool ring = (p.flags & F_RING) != 0;
    if (ring && t + Cr < (unsigned)p.t) return;
    const unsigned pp = ink_pos0(p) + t;
    const unsigned row = ring ? pp % Cr : pp;
    const size_t dst = ((size_t)kvh * Cr + row) * D + i;
    p.kc[dst] = kn[gid];
    p.vc[dst] = vn[gid];
}

// ------------------------------------------------------------------------------------------------------------------------
// launchers
// ------------------------------------------------------------------------------------------------------------------------

int chk(const ik_tensor* t, int ndim) { return ik_check(t, IK_DTYPE_BF16, ndim); }

const bf16* cb(const ik_tensor* t) { return (const bf16*)t->data; }
bf16* mb(ik_tensor* t) { return (bf16*)t->data; }

int fits_u32(int64_t v) { return v >= 0 && v <= 0x7fffffffLL; }

int split_keys(const ik_ink_attrs* a) { return a->split_keys > 0 ? a->split_keys : SPLIT_KEYS; }

unsigned bias_rows(unsigned vis, unsigned e, unsigned sk) {
    const unsigned skr = (sk + 31) / 32 * 32;
    const unsigned ns = ink_nsplit(vis, sk);
    const unsigned chunk = ink_chunk(vis, ns);
    unsigned b = chunk > skr ? chunk : skr;
    return b < e ? b : e;
}

// common prologue checks; fills the prologue part of `p`
int prologue_params(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                    const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_ink_attrs* a, ik_tensor* kring, ik_tensor* vring,
                    InkP& p) {
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
    const int64_t h = a->n_heads, hk = a->n_kv_heads, d = a->head_dim, dr = a->d_rel;
    if (hk <= 0 || h % hk != 0 || h / hk > 4 || d <= 0 || d > 256 || dr <= 0 || dr > 32) return IK_ERR_SHAPE;
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
    if (!fits_u32(a->pos0) || !fits_u32(p.t * hk * d) || kring->shape[1] != hk * d || vring->shape[0] != kring->shape[0]) return IK_ERR_SHAPE;
    return IK_OK;
}

// grid splits and bias rows of an attention launch (device position: the capacity bound)
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

template <int GP>
void launch_s1(const InkP& p, unsigned gy, size_t tbb, cudaStream_t st) {
    ink_attn_s1_k<GP><<<dim3(p.hk, gy, p.t), NSG * 32, tbb, st>>>(p);
}

template <int GP>
void launch_dec(const InkP& p, unsigned gy, unsigned tbf, size_t smem, cudaStream_t st) {
    ink_attn_dec_k<GP><<<dim3(p.hk, gy, p.t), NSG * 32, smem, st>>>(p, tbf);
}

}  // namespace

extern "C" int ik_ink_kv_prologue_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                                       const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_ink_attrs* a, ik_tensor* q,
                                       ik_tensor* rel, ik_tensor* k, ik_tensor* v, ik_tensor* kring, ik_tensor* vring, const ik_tensor* pos,
                                       void* stream) {
    if (!a) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_pos(pos));
    InkP p = {};
    IK_TRY(prologue_params(pq, pk, pv, kt, vt, qs, rs, rp, a, kring, vring, p));
    IK_TRY(chk(q, 3));
    IK_TRY(chk(rel, 3));
    IK_TRY(chk(k, 3));
    IK_TRY(chk(v, 3));
    if (p.t == 0) return IK_OK;
    p.qo = mb(q);
    p.relo = mb(rel);
    p.ko = mb(k);
    p.vo = mb(v);
    p.pos = pos ? (const int*)pos->data : nullptr;
    p.pos0 = (int)a->pos0;
    const bool fused = p.t + p.k <= p.rr + 1;
    if (fused) p.flags |= F_RING_WRITE;
    int threads = p.d > p.dr ? p.d : p.dr;
    const int ew = p.e < 256 ? p.e : 256;
    if (ew > threads) threads = ew;
    threads = (threads + 31) / 32 * 32;
    if (threads < 32) threads = 32;
    cudaStream_t st = IK_STREAM(stream);
    ink_prologue_k<<<dim3(p.hk, p.t), threads, 0, st>>>(p);
    if (!fused) {
        const size_t n = (size_t)p.t * p.hk * p.d;
        ink_ring_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
    }
    return ik_launch_status();
}

extern "C" int ik_rel_sink_attn_bf16(const ik_tensor* q, const ik_tensor* k, const ik_tensor* v, const ik_tensor* kc_in, const ik_tensor* vc_in,
                                     const ik_tensor* rel, const ik_tensor* pq, const ik_ink_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc,
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
    const ik_tensor* al[5] = {q, k, v, kc, vc};
    for (const ik_tensor* x : al)
        if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    if (!fits_u32(a->pos0) || !fits_u32((int64_t)p.t * p.h * MAXS * (p.d + 2)) || !fits_u32((int64_t)p.hk * p.cr * p.d)) return IK_ERR_SHAPE;
    p.q = cb(q);
    p.kn = cb(k);
    p.vn = cb(v);
    p.rel = cb(rel);
    p.pq = cb(pq);
    p.out = mb(o);
    p.kc = mb(kc);
    p.vc = mb(vc);
    p.pos = pos ? (const int*)pos->data : nullptr;
    unsigned gy, nb;
    attn_geometry(p, pos != nullptr, &gy, &nb);
    if (gy > 1) {
        if (!scratch || !scratch->data || scratch->bytes < (int64_t)p.t * p.h * MAXS * (p.d + 2) * 4) return IK_ERR_INVALID_ARG;
        p.scr = (float*)scratch->data;
    }
    const bool kv_fused = !ring || p.t + p.window <= p.cr + 1;
    if (kv_fused) p.flags |= F_KV_WRITE;
    const size_t tbb = (size_t)G * nb * 4;
    cudaStream_t st = IK_STREAM(stream);
    switch (G) {
        case 1: launch_s1<1>(p, gy, tbb, st); break;
        case 2: launch_s1<2>(p, gy, tbb, st); break;
        default: launch_s1<3>(p, gy, tbb, st); break;
    }
    if (gy > 1) ink_attn_comb_k<<<dim3(p.h, p.t), (p.d + 31) / 32 * 32, 0, st>>>(p);
    if (!kv_fused) {
        const size_t n = (size_t)p.t * p.hk * p.d;
        ink_kv_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p, p.kn, p.vn);
    }
    return ik_launch_status();
}

extern "C" int ik_ink_attn_decode_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                                       const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_tensor* kc_in, const ik_tensor* vc_in,
                                       const ik_ink_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc, ik_tensor* kring, ik_tensor* vring,
                                       const ik_scratch* scratch, const ik_tensor* pos, void* stream) {
    (void)kc_in;
    (void)vc_in;
    if (!a) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_pos(pos));
    InkP p = {};
    IK_TRY(prologue_params(pq, pk, pv, kt, vt, qs, rs, rp, a, kring, vring, p));
    IK_TRY(chk(o, 2));
    IK_TRY(chk(kc, 3));
    IK_TRY(chk(vc, 3));
    if (p.t == 0) return IK_OK;
    const int G = p.h / p.hk;
    if (p.t > MAXROWS || G > 3 || !(p.d == 8 || p.d == 16 || p.d == 32 || p.d == 64 || p.d == 128)) return IK_ERR_SHAPE;
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data)) return IK_ERR_INVALID_ARG;
    p.cr = (int)kc->shape[1];
    p.window = (int)a->window;
    p.scale = a->scale;
    p.sk = split_keys(a);
    p.pos0 = (int)a->pos0;
    p.out = mb(o);
    p.kc = mb(kc);
    p.vc = mb(vc);
    p.pos = pos ? (const int*)pos->data : nullptr;
    const bool ring = a->kv_ring != 0;
    if (ring) p.flags |= F_RING;
    if (!fits_u32((int64_t)p.t * p.h * MAXS * (p.d + 2)) || !fits_u32((int64_t)p.hk * p.cr * p.d)) return IK_ERR_SHAPE;
    const size_t need = (size_t)p.t * p.h * MAXS * (p.d + 2) * 4 + (size_t)2 * p.t * p.hk * p.d * 2;
    if (!scratch || !scratch->data || scratch->bytes < (int64_t)need || !ik_aligned16(scratch->data)) return IK_ERR_INVALID_ARG;
    p.scr = (float*)scratch->data;
    p.stg = (bf16*)((char*)scratch->data + (size_t)p.t * p.h * MAXS * (p.d + 2) * 4);
    const bool kv_fused = !ring || p.t + p.window <= p.cr + 1;
    const bool ring_fused = p.t + p.k <= p.rr + 1;
    if (kv_fused) p.flags |= F_KV_WRITE;
    if (ring_fused) p.flags |= F_RING_WRITE;
    unsigned gy, nb;
    attn_geometry(p, pos != nullptr, &gy, &nb);
    const unsigned tbf = ((unsigned)G * nb + 3) / 4 * 4;  // keeps the K / V rows 16-byte aligned
    const size_t smem = (size_t)tbf * 4 + (size_t)2 * p.t * p.d * 2;
    cudaStream_t st = IK_STREAM(stream);
    switch (G) {
        case 1: launch_dec<1>(p, gy, tbf, smem, st); break;
        case 2: launch_dec<2>(p, gy, tbf, smem, st); break;
        default: launch_dec<3>(p, gy, tbf, smem, st); break;
    }
    if (gy > 1) ink_attn_comb_k<<<dim3(p.h, p.t), (p.d + 31) / 32 * 32, 0, st>>>(p);
    const size_t n = (size_t)p.t * p.hk * p.d;
    if (!kv_fused) ink_kv_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p, p.stg, p.stg + n);
    if (!ring_fused) ink_ring_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
    return ik_launch_status();
}
