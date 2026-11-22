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
#include <stdlib.h>
#include <string.h>

namespace {

typedef __nv_bfloat16 bf16;

constexpr int MAXS = 32;        // most splits (scratch stride)
constexpr int ALIGN = 32;       // split chunks are a multiple of this
constexpr int NSG = 4;          // warps of an attention block
constexpr int SPLIT_KEYS = 64;  // default keys per split
constexpr int MAXROWS = 16;     // InkAttnDecode rows
constexpr int KMAX = 8;         // most conv taps
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
    // loads first (independent), then the fma chain in tap order: the same bits as the plain loop, without K dependent
    // global-load latencies (K <= KMAX is checked at launch)
    float w[KMAX], xv[KMAX];
#pragma unroll
    for (int l = 0; l < KMAX; ++l) {
        w[l] = 0.0f;
        xv[l] = 0.0f;
        if (l < p.k) {
            w[l] = b2f(taps[(size_t)l * ckv + ch]);
            if (dyn) w[l] = __fadd_rn(w[l], b2f(x[(size_t)t * p.pw + ckv + kvh * p.k + l]));
            xv[l] = ink_cx(x, ring, pp - (long)l, pos0, p.pw, p.rr, ckv, ch);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int l = 0; l < KMAX; ++l)
        if (l < p.k) acc = __fmaf_rn(w[l], xv[l], acc);
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
    float w[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) w[j] = j < dr ? b2f(rp[(size_t)j * E + e]) : 0.0f;
    float a = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j)
        if (j < dr) a = __fmaf_rn(r[j], w[j], a);
    return a;
}

// ------------------------------------------------------------------------------------------------------------------------
// Warp jobs: one warp computes one vector of the prologue (lane-strided elements, a butterfly sum of squares), so the
// prologue and the fused decode kernel produce the same bits whatever their block shapes.
// ------------------------------------------------------------------------------------------------------------------------

constexpr int JMAX = 8;  // elements per lane (head_dim <= 256)

// conv'd K (or V) row of kv head kvh at call row t: v[i] = channel lane + 32 i; returns the sum of squares (every lane)
__device__ __forceinline__ float job_conv(const InkP& p, const bf16* x, const bf16* ring, const bf16* taps, unsigned t, unsigned pos0,
                                          unsigned kvh, unsigned lane, float* v) {
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < JMAX; ++i) {
        const unsigned c = lane + 32 * i;
        v[i] = 0.0f;
        if (c < (unsigned)p.d) {
            v[i] = ink_conv(x, ring, taps, t, pos0, p, kvh * p.d + c, kvh);
            ss = __fmaf_rn(v[i], v[i], ss);
        }
    }
    return warp_sum(ss);
}

// q (r = false: D elements) or r (r = true: dr elements) of query head h at call row t, after scale, clip and tau (rounded)
__device__ __forceinline__ void job_qr(const InkP& p, unsigned t, unsigned h, unsigned pp, bool r, unsigned lane, float* v) {
    const unsigned n = r ? p.dr : p.d;
    const size_t off = (size_t)t * p.nq + (r ? (size_t)p.h * p.d + h * p.dr : (size_t)h * p.d);
    const bf16* sc = r ? p.rs + h * p.dr : p.qs + h * p.d;
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < JMAX; ++i) {
        const unsigned c = lane + 32 * i;
        v[i] = 0.0f;
        if (c < n) {
            v[i] = __fmul_rn(b2f(p.pq[off + c]), b2f(sc[c]));
            ss = __fmaf_rn(v[i], v[i], ss);
        }
    }
    ss = warp_sum(ss);
    const float tau = ink_tau(p, pp);
    const bool use_tau = p.tau_floor > 0;
#pragma unroll
    for (int i = 0; i < JMAX; ++i) v[i] = ink_qr(v[i], ss, r ? p.rel_clip : p.qk_clip, use_tau, tau);
}

__device__ __forceinline__ float k_out(float v, float ss, const InkP& p) { return rA(__fmul_rn(v, ink_clipf(ss, p.qk_clip))); }
__device__ __forceinline__ float v_out(float v, float ss, const InkP& p) { return rA(__fmul_rn(v, ink_inv_rms(ss, p))); }

// InkKvPrologue. Grid (Hk, T), 256 threads (8 warps): warps take the jobs k, v, q_0 .., r_0 .. in turn, then all threads the rel
// tail (3 x E elements of a 16-long chain each).
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
        float v[JMAX];
        if (job < 2) {
            const float ss = job_conv(p, job ? p.pv : p.pk, job ? p.vr : p.kr, job ? p.vt : p.kt, t, pos0, kvh, lane, v);
            bf16* dst = (job ? p.vo : p.ko) + ((size_t)t * p.hk + kvh) * D;
#pragma unroll
            for (int i = 0; i < JMAX; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) dst[c] = f2b(job ? v_out(v[i], ss, p) : k_out(v[i], ss, p));
            }
        } else if (job < 2 + G) {
            const unsigned h = kvh * G + (job - 2);
            job_qr(p, t, h, pp, false, lane, v);
#pragma unroll
            for (int i = 0; i < JMAX; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) p.qo[((size_t)t * H + h) * D + c] = f2b(v[i]);
            }
        } else {
            const unsigned g = job - 2 - G;
            job_qr(p, t, kvh * G + g, pp, true, lane, v);
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
        for (unsigned g = 0; g < G; ++g) p.relo[((size_t)t * H + kvh * G + g) * E + e] = f2b(ink_rel(rsh[g], p.rp, dr, E, e));
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
// The decode / verify attention on tensor cores (1..16 rows): absolute key tiles shared by all rows of the call
// ------------------------------------------------------------------------------------------------------------------------
//
// Row t (position p, first key lo) cuts the keys into absolute tiles of S keys, S = tile_s(p - lo + 1): the smallest MIN_S * 2^k
// for which an interval of that many keys meets at most MAXT tiles. Its partial over tile tau is a function of (p, lo, S,
// tau) only: the tile's keys go in 64-key sub-chunks; inside one, warp group grp takes keys 16 grp .. 16 grp + 15 (QK^T and PV
// with mma.sync m16n8k16, bf16 in, f32 out, P rounded to bf16); the four groups merge in order 0 + 1 + 2 + 3. A row with one
// tile is finished in the block, else the partials merge in tile order (ink_tile_comb_k). So a row's bits do not depend on the
// other rows of the call, T or the grid: rows-exact. One block per (kv head, tile) serves every row whose class (S) it has:
// the K / V of a tile are read once for all rows. The M = T * G query rows (row t, head g at M index t G + g) are cut into
// m-tiles of 16; block = 4 warps per m-tile (warp = (m-tile, group)).

constexpr int CH = 64;    // keys per staged sub-chunk
constexpr int MAXT = 40;  // most tiles per row (the scratch stride)

__host__ __device__ __forceinline__ int tile_s(int vis, int min_s) {
    int s = min_s;
    while ((vis + s - 2) / s + 1 > MAXT) s <<= 1;
    return s;
}

__host__ __device__ __forceinline__ int row_s(int pp, int window, int min_s) { return tile_s(pp - (int)ink_lo(pp, window) + 1, min_s); }

// The tile classes of a call: rows 0 .. rb-1 have tile size sa (tiles ta .. ta + na - 1), rows rb .. T-1 size sb (tb ..). S is
// non-decreasing in the position, and changes at most once in 16 rows.
struct TileGeo {
    int sa, sb, rb, ta, na, tb, nb;
};

__host__ __device__ __forceinline__ TileGeo tile_geo(int pos0, int T, int window, int min_s) {
    TileGeo g;
    g.sa = row_s(pos0, window, min_s);
    g.rb = T;
    for (int t = 1; t < T; ++t)
        if (row_s(pos0 + t, window, min_s) != g.sa) {
            g.rb = t;
            break;
        }
    g.sb = g.rb < T ? row_s(pos0 + g.rb, window, min_s) : g.sa;
    g.ta = (int)ink_lo(pos0, window) / g.sa;
    g.na = (pos0 + g.rb - 1) / g.sa - g.ta + 1;
    g.tb = 0;
    g.nb = 0;
    if (g.rb < T) {
        g.tb = (int)ink_lo(pos0 + g.rb, window) / g.sb;
        g.nb = (pos0 + T - 1) / g.sb - g.tb + 1;
    }
    return g;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;  // src-size 0 zero-fills the 16 bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(bf16 lo, bf16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

template <int D>
struct TileSmem {
    static constexpr int LD = D + 8;                                   // padded row (elements): conflict-free ldmatrix
    static constexpr size_t KV = (size_t)2 * 2 * CH * LD * 2;          // [stage][k / v][CH][LD] bf16
    __host__ __device__ static size_t bias(int nmt) { return (size_t)nmt * 16 * CH * 2; }  // [nmt * 16][CH] bf16 (raw rel values)
    __host__ __device__ static size_t rsh(int m, int dr) { return ((size_t)m * dr * 4 + 15) / 16 * 16; }
    __host__ __device__ static size_t own(int t) { return (size_t)2 * t * D * 2; }
    // stages = 1 when every tile of the call is one sub-chunk (S = CH: no next sub-chunk to prefetch)
    __host__ __device__ static size_t kv(int stages) { return KV / 2 * stages; }
    __host__ __device__ static size_t bytes(int t, int g, int dr, bool fused, int stages) {
        const int m = t * g, nmt = (m + 15) / 16;
        return kv(stages) + bias(nmt) + (fused ? rsh(m, dr) + own(t) : 0);
    }
};

// One block per (kv head, tile slot y). FUSED: the prologue's jobs in the block (q / r of every row, the conv'd K / V of the
// call's keys in the tile) instead of reading q / k / v / rel. blockDim = 128 * nmt.
template <int D, bool FUSED>
__global__ void __launch_bounds__(384) ink_tile_k(InkP p, int min_s, int stages) {
    using SM = TileSmem<D>;
    constexpr int LD = SM::LD, NKS = D / 16, NDB = D / 8;
    extern __shared__ __align__(16) unsigned char sm[];
    const int G = p.h / p.hk, T = p.t, M = T * G, nmt = (M + 15) / 16;
    const int H = p.h, Hk = p.hk, Cr = p.cr, E = p.e, dr = p.dr;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nw = blockDim.x >> 5;
    const int mt = warp >> 2, grp = warp & 3;
    const int kvh = blockIdx.x;
    const bool ring = (p.flags & F_RING) != 0;
    const int pos0 = ink_pos0(p);
    const TileGeo geo = tile_geo(pos0, T, p.window, min_s);
    int S, tau, r0, r1;
    if ((int)blockIdx.y < geo.na) {
        S = geo.sa, tau = geo.ta + blockIdx.y, r0 = 0, r1 = geo.rb;
    } else if ((int)blockIdx.y < geo.na + geo.nb) {
        S = geo.sb, tau = geo.tb + blockIdx.y - geo.na, r0 = geo.rb, r1 = T;
    } else {
        return;
    }
    const int tk0 = tau * S, tk1 = tk0 + S - 1;
    // the keys any row of the block reads in this tile, and the first participating row's position (for the bias skip)
    int jlo = 0x7fffffff, jhi = -1, pmin = 0x7fffffff;
    for (int t = r0; t < r1; ++t) {
        const int pp = pos0 + t, lo = (int)ink_lo(pp, p.window);
        if (lo / S <= tau && tau <= pp / S) {
            jlo = min(jlo, max(lo, tk0));
            jhi = max(jhi, min(pp, tk1));
            pmin = min(pmin, pp);
        }
    }
    if (jhi < 0) return;

    bf16* KV = (bf16*)sm;
    const size_t kvb = SM::kv(stages);
    bf16* Bs = (bf16*)(sm + kvb);
    float* rsh = (float*)(sm + kvb + SM::bias(nmt));
    bf16* own = (bf16*)(sm + kvb + SM::bias(nmt) + SM::rsh(M, dr));
    const int on0 = max(pos0, tk0), on1 = min(pos0 + T - 1, tk1);  // the call's own keys in this tile
    const int nown = on1 >= on0 ? on1 - on0 + 1 : 0;

    if (FUSED) {
        // q of every row into the KV region (before the loop), r into rsh, the own K / V rows into `own`
        bf16* Qt = KV;
        const int njobs = 2 * M + 2 * nown;
        for (int job = warp; job < njobs; job += nw) {
            float v[JMAX];
            if (job < 2 * M) {
                const int mi = job % M, t = mi / G, h = kvh * G + mi % G;
                const bool r = job >= M;
                job_qr(p, t, h, pos0 + t, r, lane, v);
                if (!r) {
#pragma unroll
                    for (int i = 0; i < JMAX; ++i)
                        if (lane + 32 * i < D) Qt[mi * D + lane + 32 * i] = f2b(v[i]);
                } else if (lane < dr) {
                    rsh[mi * dr + lane] = v[0];
                }
            } else {
                const int jj = job - 2 * M, kind = jj & 1, t = on0 - pos0 + (jj >> 1);
                const float ss = job_conv(p, kind ? p.pv : p.pk, kind ? p.vr : p.kr, kind ? p.vt : p.kt, t, pos0, kvh, lane, v);
#pragma unroll
                for (int i = 0; i < JMAX; ++i)
                    if (lane + 32 * i < D) own[((size_t)kind * T + t) * D + lane + 32 * i] = f2b(kind ? v_out(v[i], ss, p) : k_out(v[i], ss, p));
            }
        }
        __syncthreads();
    }
    // the rows this block owns (its tile holds the row's own key and the row is of its class): the cache and conv-ring writes
    for (int x = tid; x < nown * D; x += blockDim.x) {
        const int t = on0 - pos0 + x / D, c = x % D, pp = pos0 + t;
        if (t < r0 || t >= r1 || pp / S != tau) continue;
        const size_t src = ((size_t)t * Hk + kvh) * D + c;
        const bf16 kx = FUSED ? own[(size_t)t * D + c] : p.kn[src];
        const bf16 vx = FUSED ? own[((size_t)T + t) * D + c] : p.vn[src];
        if (!ring || t + Cr >= T) {
            if ((p.flags & F_KV_WRITE) != 0) {
                const size_t dst = ((size_t)kvh * Cr + (ring ? pp % Cr : pp)) * D + c;
                p.kc[dst] = kx;
                p.vc[dst] = vx;
            } else if (FUSED) {
                p.stg[src] = kx;
                p.stg[(size_t)T * Hk * D + src] = vx;
            }
        }
        if (FUSED && (p.flags & F_RING_WRITE) != 0 && t + p.rr >= T) {
            const int ch = kvh * D + c, ckv = Hk * D;
            const size_t ro = (size_t)(pp % p.rr) * ckv + ch;
            p.kr[ro] = p.pk[(size_t)t * p.pw + ch];
            p.vr[ro] = p.pv[(size_t)t * p.pw + ch];
        }
    }

    // this thread's two query rows (mma C layout: rows g8 and g8 + 8 of the warp's m-tile)
    const int g8 = lane >> 2, c4 = lane & 3, lm = lane >> 3, lr = lane & 7;
    int rmi[2], rlo[2], rhi[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        rmi[k] = mt * 16 + g8 + 8 * k;
        rlo[k] = 1;
        rhi[k] = 0;  // no key
        if (rmi[k] < M) {
            const int t = rmi[k] / G, pp = pos0 + t, lo = (int)ink_lo(pp, p.window);
            if (t >= r0 && t < r1 && lo / S <= tau && tau <= pp / S) {
                rlo[k] = lo;
                rhi[k] = pp;
            }
        }
    }
    // Q fragments (A operand, 16 x D of the m-tile) into registers
    uint32_t qf[NKS][4];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int mi = rmi[k];
        const bf16* qrow = nullptr;
        if (mi < M) qrow = FUSED ? (const bf16*)KV + (size_t)mi * D : p.q + ((size_t)(mi / G) * H + kvh * G + mi % G) * D;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            qf[ks][k] = qrow ? *(const uint32_t*)(qrow + ks * 16 + 2 * c4) : 0u;
            qf[ks][2 + k] = qrow ? *(const uint32_t*)(qrow + ks * 16 + 8 + 2 * c4) : 0u;
        }
    }
    if (FUSED) __syncthreads();  // the KV region held q

    const bf16* kcb = p.kc + (size_t)kvh * Cr * D;
    const bf16* vcb = p.vc + (size_t)kvh * Cr * D;
    auto stage = [&](int ci, int buf) {
        const int jb = ci * CH;
        constexpr int CPR = D / 8;  // 16-byte chunks per row
        for (int x = tid; x < 2 * CH * CPR; x += blockDim.x) {
            const int kind = x / (CH * CPR), r = (x / CPR) % CH, cc = x % CPR;
            const int j = jb + r;
            bf16* dst = KV + ((size_t)(buf * 2 + kind) * CH + r) * LD + cc * 8;
            if (j < jlo || j > jhi) {
                *(uint4*)dst = make_uint4(0, 0, 0, 0);
            } else if (j >= pos0) {
                if (FUSED) {
                    *(uint4*)dst = *(const uint4*)(own + ((size_t)kind * T + (j - pos0)) * D + cc * 8);
                } else {
                    cp_async16(dst, (kind ? p.vn : p.kn) + ((size_t)(j - pos0) * Hk + kvh) * D + cc * 8, true);
                }
            } else {
                const int row = ring ? j % Cr : j;
                cp_async16(dst, (kind ? vcb : kcb) + (size_t)row * D + cc * 8, true);
            }
        }
    };

    float o[NDB][4];
#pragma unroll
    for (int i = 0; i < NDB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const float qsc = __fmul_rn(p.scale, LOG2E);

    const int c0 = jlo / CH, c1 = jhi / CH;
    stage(c0, 0);
    cp_commit();
    for (int ci = c0; ci <= c1; ++ci) {
        const int buf = (ci - c0) & 1, jb = ci * CH;
        __syncthreads();  // the previous sub-chunk's compute is done with the other buffer and Bs
        if (ci < c1) stage(ci + 1, buf ^ 1);
        cp_commit();
        // the raw rel bias of the sub-chunk's keys (zero at distances >= E and for masked keys)
        const bool need_bias = pmin - (jb + CH - 1) < E;
        if (need_bias) {
            for (int x = tid; x < nmt * 16 * CH; x += blockDim.x) {
                const int mi = x / CH, i = x % CH, j = jb + i;
                bf16 b = __float2bfloat16_rn(0.0f);
                if (mi < M) {
                    const int t = mi / G, pp = pos0 + t, d = pp - j;
                    if (t >= r0 && t < r1 && d >= 0 && d < E && j >= (int)ink_lo(pp, p.window)) {
                        if (FUSED) b = f2b(ink_rel(rsh + mi * dr, p.rp, dr, E, d));
                        else b = p.rel[((size_t)t * H + kvh * G + mi % G) * E + d];
                    }
                }
                Bs[x] = b;
            }
        }
        cp_wait<1>();
        __syncthreads();
        const int kb = grp * 16;
        if (jb + kb + 15 < jlo || jb + kb > jhi) continue;  // the group's keys hold none of the block's
        const bf16* Ks = KV + (size_t)(buf * 2) * CH * LD;
        const bf16* Vs = KV + (size_t)(buf * 2 + 1) * CH * LD;
        float s[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4(b0, b1, b2, b3, Ks + (size_t)(kb + (lm >> 1) * 8 + lr) * LD + ks * 16 + (lm & 1) * 8);
            mma_bf16(s[0], qf[ks], b0, b1);
            mma_bf16(s[1], qf[ks], b2, b3);
        }
        // scores (log2 domain), masks, online softmax per row
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int k = e >> 1, i = kb + nb * 8 + 2 * c4 + (e & 1), j = jb + i;
                float v = -INFINITY;
                if (j >= rlo[k] && j <= rhi[k]) {
                    const float b = need_bias ? __bfloat162float(Bs[rmi[k] * CH + i]) : 0.0f;
                    v = __fmaf_rn(s[nb][e], qsc, __fmul_rn(b, LOG2E));
                }
                s[nb][e] = v;
                mx[k] = fmaxf(mx[k], v);
            }
        }
        uint32_t pa[2][2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            mx[k] = fmaxf(mx[k], __shfl_xor_sync(0xffffffffu, mx[k], 1));
            mx[k] = fmaxf(mx[k], __shfl_xor_sync(0xffffffffu, mx[k], 2));
            const float mn = fmaxf(m[k], mx[k]);
            const float mu = mn == -INFINITY ? 0.0f : mn;
            const float corr = exp2f(m[k] - mu);
            m[k] = mn;
            const bf16 q0 = f2b(exp2f(s[0][2 * k] - mu)), q1 = f2b(exp2f(s[0][2 * k + 1] - mu));
            const bf16 q2 = f2b(exp2f(s[1][2 * k] - mu)), q3 = f2b(exp2f(s[1][2 * k + 1] - mu));
            const float ps = __fadd_rn(__fadd_rn(__fadd_rn(b2f(q0), b2f(q1)), b2f(q2)), b2f(q3));
            l[k] = __fmaf_rn(l[k], corr, ps);
            pa[0][k] = pack_bf16(q0, q1);
            pa[1][k] = pack_bf16(q2, q3);
#pragma unroll
            for (int i = 0; i < NDB; ++i) {
                o[i][2 * k] = __fmul_rn(o[i][2 * k], corr);
                o[i][2 * k + 1] = __fmul_rn(o[i][2 * k + 1], corr);
            }
        }
        const uint32_t a[4] = {pa[0][0], pa[0][1], pa[1][0], pa[1][1]};
#pragma unroll
        for (int dp = 0; dp < NDB / 2; ++dp) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4_t(b0, b1, b2, b3, Vs + (size_t)(kb + (lm & 1) * 8 + lr) * LD + (dp * 2 + (lm >> 1)) * 8);
            mma_bf16(o[dp * 2], a, b0, b1);
            mma_bf16(o[dp * 2 + 1], a, b2, b3);
        }
    }
    cp_wait<0>();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        l[k] = __fadd_rn(l[k], __shfl_xor_sync(0xffffffffu, l[k], 1));
        l[k] = __fadd_rn(l[k], __shfl_xor_sync(0xffffffffu, l[k], 2));
    }
    // merge the four key groups of each m-tile in order (0 + 1) + 2 + 3; the KV region is free now
    float* mb = (float*)sm;
    constexpr int SLOT = (2 * NDB + 2) * 32;  // floats per warp state: o (4 NDB) ... stored as [NDB * 4 + 4][32]
    for (int q = 1; q < 4; ++q) {
        __syncthreads();
        if (grp == q) {
            float* w = mb + (size_t)mt * (NDB * 4 + 4) * 32;
#pragma unroll
            for (int i = 0; i < NDB; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) w[(i * 4 + e) * 32 + lane] = o[i][e];
            w[(NDB * 4 + 0) * 32 + lane] = m[0];
            w[(NDB * 4 + 1) * 32 + lane] = m[1];
            w[(NDB * 4 + 2) * 32 + lane] = l[0];
            w[(NDB * 4 + 3) * 32 + lane] = l[1];
        }
        __syncthreads();
        if (grp == 0) {
            const float* w = mb + (size_t)mt * (NDB * 4 + 4) * 32;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const float m2 = w[(NDB * 4 + k) * 32 + lane], l2 = w[(NDB * 4 + 2 + k) * 32 + lane];
                const float mn = fmaxf(m[k], m2);
                if (mn == -INFINITY) continue;
                const float c1 = m[k] == -INFINITY ? 0.0f : exp2f(m[k] - mn), c2 = m2 == -INFINITY ? 0.0f : exp2f(m2 - mn);
                l[k] = __fadd_rn(__fmul_rn(l[k], c1), __fmul_rn(l2, c2));
#pragma unroll
                for (int i = 0; i < NDB; ++i) {
                    o[i][2 * k] = __fadd_rn(__fmul_rn(o[i][2 * k], c1), __fmul_rn(w[(i * 4 + 2 * k) * 32 + lane], c2));
                    o[i][2 * k + 1] = __fadd_rn(__fmul_rn(o[i][2 * k + 1], c1), __fmul_rn(w[(i * 4 + 2 * k + 1) * 32 + lane], c2));
                }
                m[k] = mn;
            }
        }
    }
    (void)SLOT;
    if (grp != 0) return;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        if (rhi[k] < rlo[k]) continue;
        const int mi = rmi[k], t = mi / G, h = kvh * G + mi % G, pp = pos0 + t;
        const int lt = rlo[k] / S, ntiles = pp / S - lt + 1;
        const size_t pr = (size_t)t * p.nq;
        if (ntiles == 1) {
            const float sink2 = __fmul_rn(b2f(p.pq[pr + 2 * H * D + H * dr + h]), LOG2E);
#pragma unroll
            for (int i = 0; i < NDB; ++i)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int dim = i * 8 + 2 * c4 + e;
                    const float gate = b2f(p.pq[pr + H * D + H * dr + h * D + dim]);
                    p.out[((size_t)t * H + h) * D + dim] = ink_finish(o[i][2 * k + e], m[k], l[k], sink2, gate);
                }
        } else {
            const size_t slotp = ((size_t)t * H + h) * MAXT + (tau - lt);
#pragma unroll
            for (int i = 0; i < NDB; ++i)
#pragma unroll
                for (int e = 0; e < 2; ++e) p.scr[slotp * D + i * 8 + 2 * c4 + e] = o[i][2 * k + e];
            if (c4 == 0) {
                float* ml = p.scr + (size_t)T * H * MAXT * D;
                ml[slotp * 2] = m[k];
                ml[slotp * 2 + 1] = l[k];
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------
// Prefill (T > 16): flash attention on tensor cores. Block = (kv head, 64 query rows); warp (g, q) takes the 16 rows q * 16 ..
// of query head g, so a staged K / V sub-chunk serves 64 rows x G heads. Same score arithmetic as the tile kernel (log2 domain,
// P rounded to bf16), one pass over the keys (no tiles, no scratch), so not bitwise the decode path (the prefill kernel only
// needs the tolerance; decode / verify rows are the tile kernel's). The cache write runs after it (ink_kv_write_k).
// ------------------------------------------------------------------------------------------------------------------------

constexpr int FBR = 64;  // query rows per block

template <int D>
__global__ void __launch_bounds__(384) ink_flash_k(InkP p) {
    using SM = TileSmem<D>;
    constexpr int LD = SM::LD, NKS = D / 16, NDB = D / 8;
    extern __shared__ __align__(16) unsigned char sm[];
    const int G = p.h / p.hk, T = p.t, H = p.h, Hk = p.hk, Cr = p.cr, E = p.e, dr = p.dr;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gh = warp >> 2, rq = warp & 3;
    const int kvh = blockIdx.x, rb = blockIdx.y * FBR;
    const int h = kvh * G + gh;
    const bool ring = (p.flags & F_RING) != 0;
    const int pos0 = ink_pos0(p);
    const int rlast = min(rb + FBR, T) - 1;
    const int jlo = (int)ink_lo(pos0 + rb, p.window), jhi = pos0 + rlast;
    const int g8 = lane >> 2, c4 = lane & 3, lm = lane >> 3, lr = lane & 7;
    // this thread's rows (t, positions) and the warp's key range
    int rt[2], rlo[2], rhi[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        rt[k] = rb + rq * 16 + g8 + 8 * k;
        rlo[k] = 1;
        rhi[k] = 0;
        if (rt[k] < T) {
            rhi[k] = pos0 + rt[k];
            rlo[k] = (int)ink_lo(rhi[k], p.window);
        }
    }
    const int w0 = rb + rq * 16, w1 = min(w0 + 15, T - 1);
    const bool wact = w0 < T;
    const int wlo = wact ? (int)ink_lo(pos0 + w0, p.window) : 1, whi = wact ? pos0 + w1 : 0;
    uint32_t qf[NKS][4];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const bf16* qrow = rt[k] < T ? p.q + ((size_t)rt[k] * H + h) * D : nullptr;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            qf[ks][k] = qrow ? *(const uint32_t*)(qrow + ks * 16 + 2 * c4) : 0u;
            qf[ks][2 + k] = qrow ? *(const uint32_t*)(qrow + ks * 16 + 8 + 2 * c4) : 0u;
        }
    }
    bf16* KV = (bf16*)sm;
    const bf16* kcb = p.kc + (size_t)kvh * Cr * D;
    const bf16* vcb = p.vc + (size_t)kvh * Cr * D;
    auto stage = [&](int ci, int buf) {
        const int jb = ci * CH;
        constexpr int CPR = D / 8;
        for (int x = tid; x < 2 * CH * CPR; x += blockDim.x) {
            const int kind = x / (CH * CPR), r = (x / CPR) % CH, cc = x % CPR;
            const int j = jb + r;
            bf16* dst = KV + ((size_t)(buf * 2 + kind) * CH + r) * LD + cc * 8;
            if (j < jlo || j > jhi) {
                *(uint4*)dst = make_uint4(0, 0, 0, 0);
            } else if (j >= pos0) {
                cp_async16(dst, (kind ? p.vn : p.kn) + ((size_t)(j - pos0) * Hk + kvh) * D + cc * 8, true);
            } else {
                cp_async16(dst, (kind ? vcb : kcb) + (size_t)(ring ? j % Cr : j) * D + cc * 8, true);
            }
        }
    };
    float o[NDB][4];
#pragma unroll
    for (int i = 0; i < NDB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const float qsc = __fmul_rn(p.scale, LOG2E);
    const bf16* relh[2] = {rt[0] < T ? p.rel + ((size_t)rt[0] * H + h) * E : p.rel, rt[1] < T ? p.rel + ((size_t)rt[1] * H + h) * E : p.rel};
    const int c0 = jlo / CH, c1 = jhi / CH;
    stage(c0, 0);
    cp_commit();
    for (int ci = c0; ci <= c1; ++ci) {
        const int buf = (ci - c0) & 1, jb = ci * CH;
        __syncthreads();
        if (ci < c1) stage(ci + 1, buf ^ 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (!wact || jb + CH - 1 < wlo || jb > whi) continue;
        const bf16* Ks = KV + (size_t)(buf * 2) * CH * LD;
        const bf16* Vs = KV + (size_t)(buf * 2 + 1) * CH * LD;
#pragma unroll 1
        for (int kb = 0; kb < CH; kb += 16) {
            if (jb + kb + 15 < wlo || jb + kb > whi) continue;
            float s[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < NKS; ++ks) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, Ks + (size_t)(kb + (lm >> 1) * 8 + lr) * LD + ks * 16 + (lm & 1) * 8);
                mma_bf16(s[0], qf[ks], b0, b1);
                mma_bf16(s[1], qf[ks], b2, b3);
            }
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int k = e >> 1, j = jb + kb + nb * 8 + 2 * c4 + (e & 1);
                    float v = -INFINITY;
                    if (j >= rlo[k] && j <= rhi[k]) {
                        const int d = rhi[k] - j;
                        const float b = d < E ? __bfloat162float(relh[k][d]) : 0.0f;
                        v = __fmaf_rn(s[nb][e], qsc, __fmul_rn(b, LOG2E));
                    }
                    s[nb][e] = v;
                    mx[k] = fmaxf(mx[k], v);
                }
            }
            uint32_t pa[2][2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                mx[k] = fmaxf(mx[k], __shfl_xor_sync(0xffffffffu, mx[k], 1));
                mx[k] = fmaxf(mx[k], __shfl_xor_sync(0xffffffffu, mx[k], 2));
                const float mn = fmaxf(m[k], mx[k]);
                const float mu = mn == -INFINITY ? 0.0f : mn;
                const float corr = exp2f(m[k] - mu);
                m[k] = mn;
                const bf16 q0 = f2b(exp2f(s[0][2 * k] - mu)), q1 = f2b(exp2f(s[0][2 * k + 1] - mu));
                const bf16 q2 = f2b(exp2f(s[1][2 * k] - mu)), q3 = f2b(exp2f(s[1][2 * k + 1] - mu));
                const float ps = __fadd_rn(__fadd_rn(__fadd_rn(b2f(q0), b2f(q1)), b2f(q2)), b2f(q3));
                l[k] = __fmaf_rn(l[k], corr, ps);
                pa[0][k] = pack_bf16(q0, q1);
                pa[1][k] = pack_bf16(q2, q3);
#pragma unroll
                for (int i = 0; i < NDB; ++i) {
                    o[i][2 * k] = __fmul_rn(o[i][2 * k], corr);
                    o[i][2 * k + 1] = __fmul_rn(o[i][2 * k + 1], corr);
                }
            }
            const uint32_t a[4] = {pa[0][0], pa[0][1], pa[1][0], pa[1][1]};
#pragma unroll
            for (int dp = 0; dp < NDB / 2; ++dp) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, Vs + (size_t)(kb + (lm & 1) * 8 + lr) * LD + (dp * 2 + (lm >> 1)) * 8);
                mma_bf16(o[dp * 2], a, b0, b1);
                mma_bf16(o[dp * 2 + 1], a, b2, b3);
            }
        }
    }
    cp_wait<0>();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        l[k] = __fadd_rn(l[k], __shfl_xor_sync(0xffffffffu, l[k], 1));
        l[k] = __fadd_rn(l[k], __shfl_xor_sync(0xffffffffu, l[k], 2));
        if (rt[k] >= T) continue;
        const int t = rt[k];
        const size_t pr = (size_t)t * p.nq;
        const float sink2 = __fmul_rn(b2f(p.pq[pr + 2 * H * D + H * dr + h]), LOG2E);
#pragma unroll
        for (int i = 0; i < NDB; ++i)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int dim = i * 8 + 2 * c4 + e;
                const float gate = b2f(p.pq[pr + H * D + H * dr + h * D + dim]);
                p.out[((size_t)t * H + h) * D + dim] = ink_finish(o[i][2 * k + e], m[k], l[k], sink2, gate);
            }
    }
}

// the tile partials of a (row, query head) merged in tile order, then the sink and the gate. Grid (H, T), threads >= D.
__global__ void ink_tile_comb_k(InkP p, int min_s) {
    const unsigned tid = threadIdx.x;
    const unsigned h = blockIdx.x, t = blockIdx.y, D = p.d, H = p.h;
    if (tid >= D) return;
    const int pp = ink_pos0(p) + t, lo = (int)ink_lo(pp, p.window), S = row_s(pp, p.window, min_s);
    const int ns = pp / S - lo / S + 1;
    if (ns == 1) return;
    const size_t base = ((size_t)t * H + h) * MAXT;
    const float* ml = p.scr + (size_t)p.t * H * MAXT * D;
    float mm = -INFINITY;
    for (int s = 0; s < ns; ++s) mm = fmaxf(mm, ml[(base + s) * 2]);
    float ll = 0.0f, aa = 0.0f;
    for (int s = 0; s < ns; ++s) {
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
    if (hk <= 0 || h % hk != 0 || h / hk > 4 || d <= 0 || d > 256 || dr <= 0 || dr > 32 || kt->shape[0] > KMAX) return IK_ERR_SHAPE;
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

// grid splits and bias rows of a split-KV attention launch (device position: the capacity bound)
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

// tile size floor: INTARSIA_INK_SPLIT_KEYS rounded up to a multiple of the sub-chunk, else one sub-chunk
int tile_min(const ik_ink_attrs* a) { return a->split_keys > 0 ? (a->split_keys + CH - 1) / CH * CH : CH; }

// grid y (tile slots) of a tile launch and whether any row can have more than one tile (device position: capacity bounds)
void tile_grid(const InkP& p, bool dev, int min_s, unsigned* ny, bool* multi) {
    if (!dev) {
        const TileGeo g = tile_geo(p.pos0, p.t, p.window, min_s);
        *ny = (unsigned)(g.na + g.nb);
        bool m = false;
        for (int t = 0; t < p.t; ++t) {
            const int pp = p.pos0 + t, S = row_s(pp, p.window, min_s);
            if (pp / S != (int)ink_lo(pp, p.window) / S) m = true;
        }
        *multi = m;
        return;
    }
    const int vmax = p.window > 0 ? (p.window < p.cr ? p.window : p.cr) : p.cr;
    const int smax = tile_s(vmax, min_s);
    *ny = smax == min_s ? (unsigned)((vmax + smax - 2) / smax + 2) : (unsigned)(2 * (MAXT + 1));
    *multi = true;
}

template <int D, bool FUSED>
int launch_tile_d(const InkP& p, int min_s, unsigned ny, cudaStream_t st) {
    const int G = p.h / p.hk, nmt = (p.t * G + 15) / 16;
    bool one_sub;
    if (p.pos) {
        const int vmax = p.window > 0 ? (p.window < p.cr ? p.window : p.cr) : p.cr;
        one_sub = tile_s(vmax, min_s) == CH;
    } else {
        const TileGeo g = tile_geo(p.pos0, p.t, p.window, min_s);
        one_sub = g.sa == CH && g.sb == CH;
    }
    // one stage when no block has a second sub-chunk (every tile S = CH)
    const int stages = (min_s == CH && one_sub) ? 1 : 2;
    const size_t smem = TileSmem<D>::bytes(p.t, G, p.dr, FUSED, stages);
    static bool attr = false;
    if (!attr) {
        const size_t most = TileSmem<D>::bytes(MAXROWS, 3, 32, FUSED, 2);
        if (cudaFuncSetAttribute(ink_tile_k<D, FUSED>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)most) != cudaSuccess) return IK_ERR_LAUNCH;
        attr = true;
    }
    ink_tile_k<D, FUSED><<<dim3(p.hk, ny), 128 * nmt, smem, st>>>(p, min_s, stages);
    return IK_OK;
}

template <bool FUSED>
int launch_tile(const InkP& p, int min_s, unsigned ny, cudaStream_t st) {
    switch (p.d) {
        case 16: return launch_tile_d<16, FUSED>(p, min_s, ny, st);
        case 32: return launch_tile_d<32, FUSED>(p, min_s, ny, st);
        case 64: return launch_tile_d<64, FUSED>(p, min_s, ny, st);
        case 128: return launch_tile_d<128, FUSED>(p, min_s, ny, st);
        default: return IK_ERR_SHAPE;
    }
}

bool tile_dims(int d) { return d == 16 || d == 32 || d == 64 || d == 128; }

template <int D>
int launch_flash_d(const InkP& p, cudaStream_t st) {
    const int G = p.h / p.hk;
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(ink_flash_k<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TileSmem<D>::KV) != cudaSuccess) return IK_ERR_LAUNCH;
        attr = true;
    }
    ink_flash_k<D><<<dim3(p.hk, (p.t + FBR - 1) / FBR), 128 * G, TileSmem<D>::KV, st>>>(p);
    return IK_OK;
}

int launch_flash(const InkP& p, cudaStream_t st) {
    switch (p.d) {
        case 16: return launch_flash_d<16>(p, st);
        case 32: return launch_flash_d<32>(p, st);
        case 64: return launch_flash_d<64>(p, st);
        case 128: return launch_flash_d<128>(p, st);
        default: return IK_ERR_SHAPE;
    }
}

// INTARSIA_INK_PREFILL=splitkv / INTARSIA_INK_DECODE=splitkv: the split-KV kernel for T > 16 / T <= 16 (diagnostics for
// timing comparisons; the split-KV decode path is rows-exact itself but not bitwise the tile kernel or InkAttnDecode)
bool env_is(const char* name, int* cache) {
    if (*cache < 0) {
        const char* e = getenv(name);
        *cache = e && strcmp(e, "splitkv") == 0;
    }
    return *cache == 1;
}
bool prefill_splitkv() {
    static int v = -1;
    return env_is("INTARSIA_INK_PREFILL", &v);
}
bool decode_splitkv() {
    static int v = -1;
    return env_is("INTARSIA_INK_DECODE", &v);
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
    cudaStream_t st = IK_STREAM(stream);
    // the rel tail in slices of 256 elements (4 blocks per (kv head, row) at E = 1024: decode is latency, not work)
    const unsigned nz = (unsigned)((p.e + 255) / 256);
    ink_prologue_k<<<dim3(p.hk, p.t, nz > 8 ? 8 : nz), 256, 0, st>>>(p);
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
    if (G < 1 || G > 3 || p.h % p.hk != 0 || !(p.d == 8 || tile_dims(p.d))) return IK_ERR_SHAPE;
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
    if (!fits_u32(a->pos0) || !fits_u32((int64_t)p.t * p.h * MAXT * (p.d + 2)) || !fits_u32((int64_t)p.hk * p.cr * p.d)) return IK_ERR_SHAPE;
    p.q = cb(q);
    p.kn = cb(k);
    p.vn = cb(v);
    p.rel = cb(rel);
    p.pq = cb(pq);
    p.out = mb(o);
    p.kc = mb(kc);
    p.vc = mb(vc);
    p.pos = pos ? (const int*)pos->data : nullptr;
    const bool kv_fused = !ring || p.t + p.window <= p.cr + 1;
    if (kv_fused) p.flags |= F_KV_WRITE;
    cudaStream_t st = IK_STREAM(stream);
    if (p.t <= MAXROWS && tile_dims(p.d) && !decode_splitkv()) {
        // decode / verify rows: the tile kernel (K / V read once for all rows; rows-exact)
        const int min_s = tile_min(a);
        unsigned ny;
        bool multi;
        tile_grid(p, pos != nullptr, min_s, &ny, &multi);
        if (multi) {
            if (!scratch || !scratch->data || scratch->bytes < (int64_t)p.t * p.h * MAXT * (p.d + 2) * 4) return IK_ERR_INVALID_ARG;
            p.scr = (float*)scratch->data;
        }
        IK_TRY(launch_tile<false>(p, min_s, ny, st));
        if (multi) ink_tile_comb_k<<<dim3(p.h, p.t), (p.d + 31) / 32 * 32, 0, st>>>(p, min_s);
    } else if (tile_dims(p.d) && !prefill_splitkv()) {
        // prefill chunks: flash attention, the cache write after it
        IK_TRY(launch_flash(p, st));
        const size_t n = (size_t)p.t * p.hk * p.d;
        ink_kv_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p, p.kn, p.vn);
        return ik_launch_status();
    } else {
        unsigned gy, nb;
        attn_geometry(p, pos != nullptr, &gy, &nb);
        if (gy > 1) {
            if (!scratch || !scratch->data || scratch->bytes < (int64_t)p.t * p.h * MAXS * (p.d + 2) * 4) return IK_ERR_INVALID_ARG;
            p.scr = (float*)scratch->data;
        }
        const size_t tbb = (size_t)G * nb * 4;
        switch (G) {
            case 1: launch_s1<1>(p, gy, tbb, st); break;
            case 2: launch_s1<2>(p, gy, tbb, st); break;
            default: launch_s1<3>(p, gy, tbb, st); break;
        }
        if (gy > 1) ink_attn_comb_k<<<dim3(p.h, p.t), (p.d + 31) / 32 * 32, 0, st>>>(p);
    }
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
    if (p.t > MAXROWS || G > 3 || !tile_dims(p.d)) return IK_ERR_SHAPE;
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
    if (!fits_u32((int64_t)p.t * p.h * MAXT * (p.d + 2)) || !fits_u32((int64_t)p.hk * p.cr * p.d)) return IK_ERR_SHAPE;
    const size_t part = (size_t)p.t * p.h * MAXT * (p.d + 2) * 4;
    const size_t need = part + (size_t)2 * p.t * p.hk * p.d * 2;
    if (!scratch || !scratch->data || scratch->bytes < (int64_t)need || !ik_aligned16(scratch->data)) return IK_ERR_INVALID_ARG;
    p.scr = (float*)scratch->data;
    p.stg = (bf16*)((char*)scratch->data + part);
    const bool kv_fused = !ring || p.t + p.window <= p.cr + 1;
    const bool ring_fused = p.t + p.k <= p.rr + 1;
    if (kv_fused) p.flags |= F_KV_WRITE;
    if (ring_fused) p.flags |= F_RING_WRITE;
    const int min_s = tile_min(a);
    unsigned ny;
    bool multi;
    tile_grid(p, pos != nullptr, min_s, &ny, &multi);
    cudaStream_t st = IK_STREAM(stream);
    IK_TRY(launch_tile<true>(p, min_s, ny, st));
    if (multi) ink_tile_comb_k<<<dim3(p.h, p.t), (p.d + 31) / 32 * 32, 0, st>>>(p, min_s);
    const size_t n = (size_t)p.t * p.hk * p.d;
    if (!kv_fused) ink_kv_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p, p.stg, p.stg + n);
    if (!ring_fused) ink_ring_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
    return ik_launch_status();
}
