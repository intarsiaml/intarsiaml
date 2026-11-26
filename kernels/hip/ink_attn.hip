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
//   ink_attn_dec_k<G>  InkAttnDecode (op 75, 1..16 rows): the pair in ONE dispatch. One block per (kv head, split, row), as
//                      stage 1; each block recomputes its row's q / r, its split's rel bias and the conv'd K / V of the call's
//                      own rows inside its split with the prologue's device functions (into LDS), runs the same attention
//                      body, and the last block of a (kv head, row) to finish (a self-resetting device ticket) merges the
//                      partials in split order with stage 2's code. Bitwise the pair at any T <= 16.
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
#include <type_traits>

#pragma clang fp contract(off)

namespace {

typedef ik_bf16 bf16;

constexpr int MAXS = 32;        // most splits (scratch stride)
constexpr int ALIGN = 32;       // split chunks are a multiple of this
constexpr int NSG = 4;          // waves of an attention block
constexpr int SPLIT_KEYS = 64;  // default keys per split
constexpr int SLICE = 64;       // most rows of one attention launch (scratch bound; inkattn.rs ATTN_SLICE)
constexpr int KMAX = 4;         // most conv taps
constexpr int MAXROWS = 16;     // InkAttnDecode rows
constexpr int MAXHK_TICK = 256; // most kv heads of an InkAttnDecode call (ticket array: MAXROWS x MAXHK_TICK)
constexpr float LOG2E = 1.4426950408889634f;
constexpr float SRC_EPS = 1e-6f;

// flags
constexpr int F_DYN = 1;          // dynamic conv taps
constexpr int F_RING = 2;         // K / V caches are rings
constexpr int F_KV_WRITE = 4;     // cache write in the attention kernel (else ink_kv_write_k after it)
constexpr int F_RING_WRITE = 16;  // conv-ring write in the kernel (else ink_ring_write_k after it)
constexpr int F_R_OUT = 32;      // prologue (op 75's split path): write r [T,H,dr] to relo instead of the rel table
constexpr int F_R_IN = 64;       // stage 1 (op 75's split path): p.rel is r [T,H,dr]; the block forms its split's bias from it
constexpr int F_DIAG_NOPRO = 256;    // diagnostic (INTARSIA_INK_DEC_DIAG bit 0): op 75 skips the prologue jobs (wrong output)
constexpr int F_DIAG_NOMERGE = 512;  // diagnostic (INTARSIA_INK_DEC_DIAG bit 1): op 75 skips the ticket merge (wrong output)
constexpr int F_DPP = 1024;          // INTARSIA_HIP_INK_DEC3 bit 1 (lane ink-hip-dec3): op 75's launcher takes the <.., DPP = true>
                                     // kernels, whose butterflies take the partner from DPP / v_permlanex16 (ik_xor_f) instead of
                                     // ds_bpermute: the same partners in the same order, same bits (the kernels do not read the flag)
constexpr int F_R_ROW = 2048;        // prologue (INTARSIA_HIP_INK_PF2 bit 1, lane ink-hip-pf2): write r (rounded) into the first dr entries
                                     // of each rel row and skip the rel tail; ink_rel_table_k forms the rows from it right after

struct InkP {
    const bf16 *pq, *pk, *pv, *kt, *vt, *qs, *rs, *rp;
    const bf16 *q, *kn, *vn, *rel;
    bf16 *qo, *relo, *ko, *vo, *kr, *vr, *kc, *vc, *out;
    float* scr;
    const int* pos;
    int t, h, hk, d, dr, e, k, rr, cr, pos0, flags, tau_floor, window, nq, pw, sk, toff;
    float qk_clip, rel_clip, eps, tau_alpha, scale, cscale;
    int qhi;  // flash in pieces (INTARSIA_HIP_INK_PF2 bit 2): one past the piece's last query tile; 0 = gridDim.x (one launch)
};

// The row-table form (lane ink-hip-batch, `OpCall::rows`): the table's device addresses ([rows][rts] entries), this call's first entry,
// and each state tensor's byte offset in its row's buffer (k cache, v cache, k ring, v ring: the carriers' offsets). Only the RT
// instantiation of ink_attn_dec_k takes `InkPR`, so every solo kernel's arguments stay `InkP`, byte for byte.
struct InkRt {
    const unsigned long long* rtab;
    int rts, rti;
    long long roff[4];
};
struct InkPR : InkP {
    InkRt rt;
};

// INTARSIA_HIP_INK_PF2 (lane ink-hip-pf2; prefill chunks only, every bit keeps the output bits): bit 1 = the prologue's rel tail
// as its own pass (ink_rel_table_k), bit 2 = flash attention in pieces of query tiles under the R9700 dispatch stall line
// (ik_hip.h), bit 128 = with bit 1, the prologue's OCC form (12 waves per SIMD); the experts' bits 4 / 8 / 16 are in
// ink_moe.hip. Unset = kPf2Default (1 | 2 | 4 | 16 | 128, the same constant there); 0 = main's forms. Read at every launch.
constexpr int kPf2Default = 151;
int pf2_mask() {
    const char* e = getenv("INTARSIA_HIP_INK_PF2");
    return e && e[0] >= '0' && e[0] <= '9' ? atoi(e) : kPf2Default;
}
// pieces of a launch estimated at est_us: one below the trigger, else ceil(est / budget), at most max_pieces. Budget
// INTARSIA_HIP_INK_PF2_US (default 900 us: one piece stays under the ~1 ms stall line with margin); the trigger is 950 us, or the
// budget if that is lower (tests set a tiny budget to force the most pieces).
int pf2_pieces(double est_us, int64_t max_pieces) {
    const char* e = getenv("INTARSIA_HIP_INK_PF2_US");
    const double budget = e && *e ? atof(e) : 900.0;
    const double trigger = budget < 950.0 ? budget : 950.0;
    if (!(budget > 0.0) || est_us <= trigger || max_pieces <= 1) return 1;
    const double n = est_us / budget;
    const int64_t p = n >= (double)max_pieces ? max_pieces : (int64_t)n + ((double)(int64_t)n < n ? 1 : 0);
    return (int)(p < 1 ? 1 : p);
}

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
IK_DEVI float warp_sum(float v, bool dpp) {
    if (dpp) {
        v = fadd(v, ik_xor_f<16>(v));
        v = fadd(v, ik_xor_f<8>(v));
        v = fadd(v, ik_xor_f<4>(v));
        v = fadd(v, ik_xor_f<2>(v));
        return fadd(v, ik_xor_f<1>(v));
    }
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

template <int J, bool DPP = false>
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
    return warp_sum(ss, DPP);
}

// q (r = false: D elements) or r (r = true: dr elements) of query head h at call row t, after scale, clip and tau (rounded)
template <int J, bool DPP = false>
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
    ss = warp_sum(ss, DPP);
    const float tau = ink_tau(p, pp);
    const bool use_tau = p.tau_floor > 0;
#pragma unroll
    for (int i = 0; i < J; ++i) v[i] = ink_qr(v[i], ss, r ? p.rel_clip : p.qk_clip, use_tau, tau);
}

IK_DEVI float k_out(float v, float ss, const InkP& p) { return rA(fmul(v, ink_clipf(ss, p.qk_clip))); }
IK_DEVI float v_out(float v, float ss, const InkP& p) { return rA(fmul(v, ink_inv_rms(ss, p))); }

// InkKvPrologue. Grid (Hk, T, rel slices), 256 threads (8 waves): waves take the jobs k, v, q_0 .., r_0 .. in turn, then all
// threads the rel tail. OCC (INTARSIA_HIP_INK_PF2 bit 128, prefill calls with bit 1): compiled for 12 waves per SIMD (a 128-VGPR
// cap; the plain form takes 160, 9 waves): the same instructions on the same values, the same bits. Window 4: +0.9 % prefill.
template <int J, bool DPP = false, bool OCC = false>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(OCC ? 12 : 1))) ink_prologue_k(InkP p) {
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
            const float ss = job_conv<J, DPP>(p, job ? p.pv : p.pk, job ? p.vr : p.kr, job ? p.vt : p.kt, t, pos0, kvh, lane, v);
            bf16* dst = (job ? p.vo : p.ko) + ((size_t)t * p.hk + kvh) * D;
#pragma unroll
            for (int i = 0; i < J; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) dst[c] = f2b(job ? v_out(v[i], ss, p) : k_out(v[i], ss, p));
            }
        } else if (job < 2 + G) {
            const unsigned h = kvh * G + (job - 2);
            job_qr<J, DPP>(p, t, h, pp, false, lane, v);
#pragma unroll
            for (int i = 0; i < J; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) p.qo[((size_t)t * H + h) * D + c] = f2b(v[i]);
            }
        } else {
            const unsigned g = job - 2 - G;
            job_qr<J, DPP>(p, t, kvh * G + g, pp, true, lane, v);
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
    if (p.flags & F_R_OUT) {
        // op 75's split path: stage 1 forms the bias itself from r (the same chain); r is rounded already
        for (unsigned x = tid; x < G * dr; x += blockDim.x) p.relo[((size_t)t * H + kvh * G + x / dr) * dr + x % dr] = f2b(rsh[x / dr][x % dr]);
        return;
    }
    if (p.flags & F_R_ROW) {
        // INTARSIA_HIP_INK_PF2 bit 1: r into its own rel row (entries 0 .. dr); ink_rel_table_k reads it before it writes the row
        for (unsigned x = tid; x < G * dr; x += blockDim.x) p.relo[((size_t)t * H + kvh * G + x / dr) * E + x % dr] = f2b(rsh[x / dr][x % dr]);
        return;
    }
    const unsigned per = (E + gridDim.z - 1) / gridDim.z, e1 = min(E, (blockIdx.z + 1) * per);
    for (unsigned e = blockIdx.z * per + tid; e < e1; e += blockDim.x) {
        float rv[4];
        ink_rel_g(rsh, G, p.rp, dr, E, e, rv);
        for (unsigned g = 0; g < G; ++g) p.relo[((size_t)t * H + kvh * G + g) * E + e] = f2b(rv[g]);
    }
}

// The prologue's rel tail as its own pass (INTARSIA_HIP_INK_PF2 bit 1, prefill chunks; lane ink-hip-pf2). In ink_prologue_k
// a block of one (kv head, row) forms G x E entries, each with 16 relproj loads, and the 100 MB table of a 2048-row chunk took
// 1.67 ms (94 GB/s). Here a block takes RT_ROWS call rows (all H heads): it stages their r (left by the prologue in the first dr
// entries of each rel row, F_R_ROW) in LDS before any write, and each thread holds the relproj columns of its entries
// e = 4 tid .. 4 tid + 3 in registers (one 8-byte store per row). Every entry is ink_rel_g's chain (from 0, j ascending, ffma) on
// the same r and relproj values, then the bf16 round: bitwise the prologue's tail. Launcher: dr <= RT_DR, E a multiple of 4 and
// <= 1024, the table 8-byte aligned.
constexpr int RT_ROWS = 8;  // call rows per block
constexpr int RT_EPT = 4;   // consecutive entries per thread (one 8-byte store per row)
constexpr int RT_DR = 16;   // most d_rel
__global__ void __launch_bounds__(256) ink_rel_table_k(InkP p) {
    extern __shared__ float rts[];  // r of the block's rows [RT_ROWS * H][dr]
    const int H = p.h, E = p.e, dr = p.dr, tid = threadIdx.x;
    const int t0 = (int)blockIdx.x * RT_ROWS;
    const int nrow = (p.t - t0 < RT_ROWS ? p.t - t0 : RT_ROWS) * H;
    bf16* rel = p.relo + (size_t)t0 * H * E;
    for (int x = tid; x < nrow * dr; x += blockDim.x) rts[x] = b2f(rel[(size_t)(x / dr) * E + x % dr]);
    const int e0 = RT_EPT * tid;
    float w[RT_EPT][RT_DR];
#pragma unroll
    for (int u = 0; u < RT_EPT; ++u)
#pragma unroll
        for (int j = 0; j < RT_DR; ++j) w[u][j] = (e0 + u < E && j < dr) ? b2f(p.rp[(size_t)j * E + e0 + u]) : 0.0f;
    __syncthreads();  // every r of the block is staged before the first entry of its rows is overwritten
    if (e0 >= E) return;
    for (int row = 0; row < nrow; ++row) {
        float r[RT_DR];
#pragma unroll
        for (int j = 0; j < RT_DR; ++j) r[j] = j < dr ? rts[row * dr + j] : 0.0f;
        float acc[RT_EPT];
#pragma unroll
        for (int u = 0; u < RT_EPT; ++u) {
            acc[u] = 0.0f;
#pragma unroll
            for (int j = 0; j < RT_DR; ++j)
                if (j < dr) acc[u] = ffma(r[j], w[u][j], acc[u]);
        }
        uint2 o;
        o.x = (uint32_t)f2b(acc[0]).bits | ((uint32_t)f2b(acc[1]).bits << 16);
        o.y = (uint32_t)f2b(acc[2]).bits | ((uint32_t)f2b(acc[3]).bits << 16);
        *reinterpret_cast<uint2*>(rel + (size_t)row * E + e0) = o;
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
// KV_TG: the call's own keys (j >= pos0) come from the block's LDS rows `kvt` (K plane, V plane MAXROWS * 128 later; row
// j - pos0 - kvoff) instead of the k / v inputs. Every thread returns from it (the caller may continue with a block-wide step).
template <int GP, bool KV_TG, bool DPP = false>
IK_DEVI void ink_attn_body(InkTg<GP>& S, const float* bias, const bf16* kvt, unsigned kvoff, InkSplit g, unsigned kvh, unsigned split,
                           unsigned t, unsigned pos0, const InkP& p, unsigned tid, unsigned lane, unsigned sg) {
    const unsigned D = p.d, H = p.h, Hk = p.hk, Cr = p.cr;
    const bool ring = (p.flags & F_RING) != 0;
    const unsigned lpk = D / 8;     // lanes per key row
    const unsigned kps = 32 / lpk;  // keys per wave step
    const bool dpp16 = DPP && lpk == 16;  // D = 128: the slot sum is xor 1, 2, 4, 8 and the merge xor 16
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
    // waves >= NSG (InkAttnDecode's extra prologue waves) take no keys: the key -> (wave, slot, step) map is the pair's
    for (unsigned base = g.ks + sg * kps; sg < NSG && base < g.ke; base += NSG * kps) {
        const unsigned j = base + slot;
        const bool ok = j < g.ke;
        uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
        if (ok) {
            if (j >= pos0) {
                if (KV_TG) {
                    const unsigned ro = (j - pos0 - kvoff) * D + le * 8;
                    kw = *(const uint4*)(kvt + ro);
                    vw = *(const uint4*)(kvt + MAXROWS * 128 + ro);
                } else {
                    const size_t o = ((size_t)(j - pos0) * Hk + kvh) * D + le * 8;
                    kw = *(const uint4*)(p.kn + o);
                    vw = *(const uint4*)(p.vn + o);
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
        ik_unpack8(kw, kf);
        ik_unpack8(vw, vf);
        const unsigned dist = g.pp - j;
        const unsigned bi = dist - g.dlo;
        for (int h = 0; h < GP; ++h) {
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s = ffma(qv[h][i], kf[i], s);
            // every lane of the wave runs the shuffles (the slot's lanes are adjacent: xor below lpk stays in the slot)
            if (dpp16) {
                s = fadd(s, ik_xor_f<1>(s));
                s = fadd(s, ik_xor_f<2>(s));
                s = fadd(s, ik_xor_f<4>(s));
                s = fadd(s, ik_xor_f<8>(s));
            } else {
                for (unsigned o = 1; o < lpk; o <<= 1) s = fadd(s, shx(s, o));
            }
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
            float m2, l2, a2[8];
            if (dpp16) {  // o = 16 only
                m2 = ik_xor_f<16>(m[h]);
                l2 = ik_xor_f<16>(l[h]);
                for (int i = 0; i < 8; ++i) a2[i] = ik_xor_f<16>(acc[h][i]);
            } else {
                m2 = shx(m[h], o);
                l2 = shx(l[h], o);
                for (int i = 0; i < 8; ++i) a2[i] = shx(acc[h][i], o);
            }
            const float mn = fmaxf(m[h], m2);
            if (mn == -INFINITY) continue;
            const float c1 = exp2f(m[h] - mn), c2 = exp2f(m2 - mn);
            l[h] = fadd(fmul(l[h], c1), fmul(l2, c2));
            for (int i = 0; i < 8; ++i) acc[h][i] = fadd(fmul(acc[h][i], c1), fmul(a2[i], c2));
            m[h] = mn;
        }
    }
    if (slot == 0 && sg < NSG) {
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
template <int GP, bool DPP = false>
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
    if (p.flags & F_R_IN) {
        // op 75's split path: the prologue's rel chain at this split's distances from r, rounded as the rel store (the bits of the
        // pair's rel table)
        __shared__ float rsh[4][32];
        for (unsigned x = tid; x < GP * (unsigned)p.dr; x += NSG * 32) rsh[x / p.dr][x % p.dr] = b2f(p.rel[((size_t)t * H + kvh * GP) * p.dr + x]);
        __syncthreads();
        for (unsigned b = tid; b < g.nb; b += NSG * 32) {
            float rv[4];
            ink_rel_g(rsh, GP, p.rp, p.dr, E, g.dlo + b, rv);
            for (int h = 0; h < GP; ++h) tb[h * g.nb + b] = fmul(rA(rv[h]), LOG2E);
        }
    } else {
        for (unsigned x = tid; x < GP * g.nb; x += NSG * 32) {
            const unsigned h = x / g.nb, b = x % g.nb;
            tb[h * g.nb + b] = fmul(b2f(p.rel[((size_t)t * H + kvh * GP + h) * E + g.dlo + b]), LOG2E);
        }
    }
    __syncthreads();
    ink_attn_body<GP, false, DPP>(S, tb, nullptr, 0, g, kvh, split, t, pos0, p, tid, lane, sg);
}

// merge of the ns partials of (row t, query head h), element `tid`, in split order; then the sink and the gate
IK_DEVI void ink_comb_elem(const InkP& p, unsigned t, unsigned h, unsigned tid, unsigned ns) {
    const unsigned D = p.d, H = p.h;
    const size_t base = ((size_t)t * H + h) * MAXS;
    const float* ml = p.scr + (size_t)p.t * H * MAXS * D;
    float mm = -INFINITY;
#pragma unroll 8
    for (unsigned s = 0; s < ns; ++s) mm = fmaxf(mm, ml[(base + s) * 2]);
    float ll = 0.0f, aa = 0.0f;
#pragma unroll 8
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

// stage 2. Grid (H, T), threads >= D.
__global__ void ink_attn_comb_k(InkP p) {
    const unsigned tid = threadIdx.x;
    const unsigned h = blockIdx.x, t = blockIdx.y;
    if (tid >= (unsigned)p.d) return;
    const unsigned pp = ink_pos0(p) + t;
    const unsigned vis = pp - ink_lo(pp, p.window) + 1;
    const unsigned ns = ink_nsplit(vis, p.sk);
    if (ns == 1) return;
    ink_comb_elem(p, t, h, tid, ns);
}

// InkAttnDecode tickets: per (row, kv head) the count of finished splits; the last block resets its entry to 0, so the array is
// zero before and after every launch (stream order serialises launches; graph replays alike).
__device__ unsigned ink_dec_tick[MAXROWS * MAXHK_TICK];

// Row-table form (lane ink-hip-batch): `p` becomes the T = 1 call of sequence r at device position pos[r]: its row-indexed
// operands shifted by r rows, its own scratch block of one row, its caches / rings from table entries rti .. rti + 3 (+ the carriers'
// offsets). The kernels then run that call's code at row 0.
IK_DEVI void ink_rt_row(InkPR& p, unsigned r) {
    const unsigned long long* e = p.rt.rtab + (size_t)r * p.rt.rts + p.rt.rti;
    p.pos0 = max(p.pos[r], 0);
    p.pos = nullptr;
    p.pq += (size_t)r * p.nq;
    p.pk += (size_t)r * p.pw;
    p.pv += (size_t)r * p.pw;
    p.out += (size_t)r * p.h * p.d;
    p.scr += (size_t)r * p.h * MAXS * (p.d + 2);
    p.kc = (bf16*)(e[0] + p.rt.roff[0]);
    p.vc = (bf16*)(e[1] + p.rt.roff[1]);
    p.kr = (bf16*)(e[2] + p.rt.roff[2]);
    p.vr = (bf16*)(e[3] + p.rt.roff[3]);
    p.t = 1;
}

// InkAttnDecode. Grid (Hk, max splits, T <= 16), DEC_WAVES waves: all of them take the prologue jobs (one round for the 2 G + 2
// jobs of a decode row), the bias and the merge; waves 0 .. NSG-1 run the attention body (the pair's key map). Dynamic LDS: the
// split's bias [GP][nb] f32. Diagnostic p.flags F_DIAG_*: timing only, wrong output.
// RT (`ink_attn_decode_bf16_rows`, lane ink-hip-batch): grid z = the rows of a row-table call, block row r runs the T = 1 call of
// sequence r (ink_rt_row); its ticket is row r's. RT = false is the solo kernel.
constexpr int DEC_WAVES = 8;
template <int GP, bool DPP = false, bool RT = false>
__global__ void __launch_bounds__(DEC_WAVES * 32) ink_attn_dec_k(std::conditional_t<RT, InkPR, InkP> p) {
    __shared__ InkTg<GP> S;
    __shared__ float rsh[4][32];
    __shared__ __align__(16) bf16 kvt[2 * MAXROWS * 128];
    __shared__ unsigned is_last;
    extern __shared__ float tb[];
    const unsigned tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    const unsigned kvh = blockIdx.x, split = blockIdx.y, t = RT ? 0u : blockIdx.z;
    if constexpr (RT) ink_rt_row(p, blockIdx.z);
    const unsigned D = p.d, Hk = p.hk, E = p.e, Cr = p.cr, ckv = Hk * D;
    const bool ring = (p.flags & F_RING) != 0;
    const unsigned pos0 = ink_pos0(p);
    const InkSplit g = ink_split(p, pos0, t, split);
    if (split >= g.ns) return;
    const unsigned vis = g.pp - g.lo + 1, chunk = ink_chunk(vis, g.ns);
    // the block whose split holds key pp (row t itself) writes row t's cache and conv-ring rows
    const bool writer = (g.pp - g.lo) / chunk == split;
    // the call's own rows among the split's keys: ua .. ub - 1
    unsigned ua = 0, ub = 0;
    if (g.ke > g.ks && g.ke > pos0) {
        ua = (g.ks > pos0 ? g.ks : pos0) - pos0;
        ub = g.ke - pos0;
    }
    const unsigned nown = ub - ua;
    // prologue jobs, one wave each: q of the GP heads, r of the GP heads, conv'd k / v of each own row
    const unsigned njobs = (p.flags & F_DIAG_NOPRO) ? 0u : 2 * GP + 2 * nown;
    for (unsigned job = sg; job < njobs; job += DEC_WAVES) {
        float v[4];
        if (job < (unsigned)GP) {
            job_qr<4, DPP>(p, t, kvh * GP + job, g.pp, false, lane, v);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) S.q[job][c] = v[i];  // rounded already: the bits the pair's q store / load gives
            }
        } else if (job < 2u * GP) {
            const unsigned gg = job - GP;
            job_qr<4, DPP>(p, t, kvh * GP + gg, g.pp, true, lane, v);
            if (lane < (unsigned)p.dr) rsh[gg][lane] = v[0];
        } else {
            const unsigned u = ua + (job - 2 * GP) / 2, kv = (job - 2 * GP) & 1;
            const float ss = job_conv<4, DPP>(p, kv ? p.pv : p.pk, kv ? p.vr : p.kr, kv ? p.vt : p.kt, u, pos0, kvh, lane, v);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned c = lane + 32 * i;
                if (c < D) kvt[kv * MAXROWS * 128 + (u - ua) * D + c] = f2b(kv ? v_out(v[i], ss, p) : k_out(v[i], ss, p));
            }
        }
    }
    __syncthreads();
    // the split's bias: the prologue's rel chain at the split's distances, rounded as the rel store, then the attention's scaling
    for (unsigned b = tid; b < g.nb; b += DEC_WAVES * 32) {
        float rv[4];
        ink_rel_g(rsh, GP, p.rp, p.dr, E, g.dlo + b, rv);
        for (int h = 0; h < GP; ++h) tb[h * g.nb + b] = fmul(rA(rv[h]), LOG2E);
    }
    if (writer) {
        const bf16* kr = kvt + (t - ua) * D;
        const bf16* vr = kvt + MAXROWS * 128 + (t - ua) * D;
        if ((p.flags & F_KV_WRITE) != 0) {
            if (!ring || t + Cr >= (unsigned)p.t) {
                const size_t dst = ((size_t)kvh * Cr + (ring ? g.pp % Cr : g.pp)) * D;
                for (unsigned i = tid; i < D; i += DEC_WAVES * 32) {
                    p.kc[dst + i] = kr[i];
                    p.vc[dst + i] = vr[i];
                }
            }
        } else {
            // staged for ink_kv_write_k (the pair's k / v layout [T][Hk][D])
            const size_t src = ((size_t)t * Hk + kvh) * D;
            for (unsigned i = tid; i < D; i += DEC_WAVES * 32) {
                p.ko[src + i] = kr[i];
                p.vo[src + i] = vr[i];
            }
        }
        if ((p.flags & F_RING_WRITE) != 0 && t + p.rr >= (unsigned)p.t) {
            for (unsigned c = tid; c < D; c += DEC_WAVES * 32) {
                const unsigned ch = kvh * D + c;
                const size_t ro = (size_t)(g.pp % p.rr) * ckv + ch;
                p.kr[ro] = p.pk[(size_t)t * p.pw + ch];
                p.vr[ro] = p.pv[(size_t)t * p.pw + ch];
            }
        }
    }
    __syncthreads();
    ink_attn_body<GP, true, DPP>(S, tb, kvt, ua, g, kvh, split, t, pos0, p, tid, lane, sg);
    if (g.ns == 1 || (p.flags & F_DIAG_NOMERGE)) return;
    // the partials of this split are written; the last split of (row, kv head) to arrive merges them all
    __threadfence();
    __syncthreads();
    unsigned* tick = &ink_dec_tick[(RT ? blockIdx.z : t) * MAXHK_TICK + kvh];
    if (tid == 0) {
        const unsigned old = atomicAdd(tick, 1u);
        is_last = old == g.ns - 1;
        if (is_last) atomicExch(tick, 0u);
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    for (unsigned x = tid; x < GP * D; x += DEC_WAVES * 32) ink_comb_elem(p, t, kvh * GP + x / D, x % D, g.ns);
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
// RelSinkAttn on WMMA flash attention for prefill chunks (rel_sink_attn_bf16_flash, T > 16; not bitwise the split-KV kernel)
// ------------------------------------------------------------------------------------------------------------------------
//
// A fork of flash_kernel (kernels/hip/attention_flash.hip; v_wmma_f32_16x16x16_bf16, wave32): the transposed problem
// S^T = K Q^T, O^T += V^T P^T, one lane per query row, online softmax in the log2 domain. Block = the G query heads of one kv
// head (4 waves = 64 query rows each), so a K / V tile is staged once for G heads. Changes: keys j >= pos0 come from the call's
// own k / v, older keys from the cache (ring row j % Cr, or linear row j); the SWA window's lower edge (keys below
// pp - window + 1 masked, tiles below the block's first row's window skipped); the rel bias rel[row, h, pp - j] * log2(e) for
// distances below E; the sink and the gate in the epilogue (ink_finish, as the split-KV kernel). The cache write runs after.
constexpr int FBQ = 64;  // query rows per wave group
constexpr int FBK = 64;  // keys per tile

typedef __bf16 ink_v8bf __attribute__((ext_vector_type(8)));
typedef float ink_v8f __attribute__((ext_vector_type(8)));

IK_DEVI ink_v8f ink_wmma(const uint4& a, const uint4& b, ink_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ink_v8bf, a), __builtin_bit_cast(ink_v8bf, b), c);
}

template <int D, int G>
__global__ void __launch_bounds__(128 * G) ink_flash_k(InkP p) {
    constexpr int BK = FBK;
    constexpr int LDK = D + 8;    // padded K tile row (elements)
    constexpr int LDV = BK + 8;   // padded row of the transposed V tile
    constexpr int NKS = D / 16;   // k steps of S^T
    constexpr int NKB = BK / 16;  // 16-key blocks per tile
    constexpr int NDB = D / 16;   // 16-dim blocks of O^T
    constexpr int CH = D / 8;     // 16 B chunks per key row
    constexpr int NT = 128 * G;

    __shared__ __align__(16) uint16_t Ks[BK * LDK];
    __shared__ __align__(16) uint16_t Vt[D * LDV];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, lk = (lane >> 4) * 8;
    const int qt = (p.qhi > 0 ? p.qhi : (int)gridDim.x) - 1 - (int)blockIdx.x;  // longest blocks first (a piece: its own tiles)
    const int kvh = blockIdx.y, h = kvh * G + (warp >> 2);
    const int T = p.t, H = p.h, Hk = p.hk, E = p.e;
    const int64_t Cr = p.cr;
    const bool ring = (p.flags & F_RING) != 0;
    const int q0 = qt * FBQ;
    const int qrow = q0 + (warp & 3) * 16 + lr;
    const int64_t pos0 = ink_pos0(p);
    const int q_last = (q0 + FBQ < T ? q0 + FBQ : T) - 1;
    const int64_t kv_end = pos0 + q_last + 1;                         // keys visible to the block's last row
    const int64_t lo_first = ink_lo((unsigned)(pos0 + q0), p.window);  // first key of the block's first row
    const int64_t lo_last = ink_lo((unsigned)(pos0 + q_last), p.window);
    const int kt0 = (int)(lo_first / BK), kt1 = (int)((kv_end + BK - 1) / BK);
    const int64_t mask_from = pos0 + q0 + 1;  // tiles ending beyond this need the causal mask
    const int64_t pp = pos0 + qrow;
    const int64_t mylo = ink_lo((unsigned)pp, p.window);
    const float sl2 = fmul(p.scale, LOG2E);

    const bf16* kcb = p.kc + (int64_t)kvh * Cr * D;
    const bf16* vcb = p.vc + (int64_t)kvh * Cr * D;
    const bf16* relr = p.rel + ((int64_t)(qrow < T ? qrow : 0) * H + h) * E;

    uint4 qb[NKS];
#pragma unroll
    for (int ks = 0; ks < NKS; ++ks) {
        qb[ks] = make_uint4(0, 0, 0, 0);
        if (qrow < T) {
            const uint32_t* q = reinterpret_cast<const uint32_t*>(p.q + ((int64_t)qrow * H + h) * D + ks * 16 + lk);
            qb[ks] = make_uint4(q[0], q[1], q[2], q[3]);
        }
    }
    ink_v8f o[NDB];
#pragma unroll
    for (int i = 0; i < NDB; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) o[i][e] = 0.f;
    float m = -INFINITY, l = 0.f;

    for (int kt = kt0; kt < kt1; ++kt) {
        __syncthreads();  // the previous tile's reads of Ks / Vt are done
        for (int idx = tid; idx < BK * CH; idx += NT) {
            const int r = idx / CH, ch = idx % CH;
            const int64_t key = (int64_t)kt * BK + r;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (key < kv_end && key >= lo_first) {
                if (key >= pos0) {
                    const int64_t o2 = ((key - pos0) * Hk + kvh) * D + ch * 8;
                    kv = *reinterpret_cast<const uint4*>(p.kn + o2);
                    vv = *reinterpret_cast<const uint4*>(p.vn + o2);
                } else {
                    const int64_t row = ring ? key % Cr : key;
                    kv = *reinterpret_cast<const uint4*>(kcb + row * D + ch * 8);
                    vv = *reinterpret_cast<const uint4*>(vcb + row * D + ch * 8);
                }
            }
            *reinterpret_cast<uint4*>(&Ks[r * LDK + ch * 8]) = kv;
            const uint32_t vw[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                Vt[(ch * 8 + 2 * e) * LDV + r] = (uint16_t)(vw[e] & 0xffffu);
                Vt[(ch * 8 + 2 * e + 1) * LDV + r] = (uint16_t)(vw[e] >> 16);
            }
        }
        __syncthreads();

        ink_v8f sc[NKB];
#pragma unroll
        for (int b = 0; b < NKB; ++b)
#pragma unroll
            for (int e = 0; e < 8; ++e) sc[b][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
#pragma unroll
            for (int b = 0; b < NKB; ++b) {
                const uint4 a = *reinterpret_cast<const uint4*>(&Ks[(b * 16 + lr) * LDK + ks * 16 + lk]);
                sc[b] = ink_wmma(a, qb[ks], sc[b]);
            }
        }

        // scores (log2 domain) + rel bias; causal and window masks
        const bool need_mask = (int64_t)(kt + 1) * BK > mask_from || (int64_t)kt * BK < lo_last;
        float mx = -INFINITY;
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int64_t key = (int64_t)kt * BK + b * 16 + lk + e;
                float v = fmul(sc[b][e], sl2);
                const bool ok = qrow < T && (!need_mask || (key <= pp && key >= mylo));
                if (ok) {
                    const int64_t d = pp - key;
                    if (d < E) v = fadd(v, fmul(b2f(relr[d]), LOG2E));
                } else {
                    v = -INFINITY;
                }
                sc[b][e] = v;
                mx = fmaxf(mx, v);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx);
        const float mu = mn == -INFINITY ? 0.f : mn;
        const float cr = exp2f(m - mu);
        m = mn;
        float ps = 0.f;
        uint4 pb[NKB];
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
            float pv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pv[e] = exp2f(sc[b][e] - mu);
                ps = fadd(ps, pv[e]);
            }
            pb[b] = ik_pack8(pv);
        }
        l = ffma(l, cr, ps);
#pragma unroll
        for (int i = 0; i < NDB; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) o[i][e] = fmul(o[i][e], cr);
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                const uint4 a = *reinterpret_cast<const uint4*>(&Vt[(db * 16 + lr) * LDV + b * 16 + lk]);
                o[db] = ink_wmma(a, pb[b], o[db]);
            }
        }
    }

    // epilogue: the two lane halves' row sums, then the sink and the gate per element
    l = fadd(l, __shfl_xor(l, 16, 32));
    if (qrow < T) {
        const size_t pr = (size_t)qrow * p.nq;
        const float sink2 = fmul(b2f(p.pq[pr + 2 * (size_t)H * D + (size_t)H * p.dr + h]), LOG2E);
        const bf16* gt = p.pq + pr + (size_t)H * D + (size_t)H * p.dr + (size_t)h * D;
#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            uint16_t ob[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) ob[e] = ink_finish(o[db][e], m, l, sink2, b2f(gt[db * 16 + lk + e])).bits;
            uint32_t* po = reinterpret_cast<uint32_t*>(p.out + ((int64_t)qrow * H + h) * D + db * 16 + lk);
#pragma unroll
            for (int e = 0; e < 4; ++e) po[e] = (uint32_t)ob[2 * e] | ((uint32_t)ob[2 * e + 1] << 16);
        }
    }
}

template <int D>
void launch_flash_grid(const InkP& p, dim3 grid, hipStream_t st) {
    switch (p.h / p.hk) {
        case 1: ink_flash_k<D, 1><<<grid, 128, 0, st>>>(p); break;
        case 2: ink_flash_k<D, 2><<<grid, 256, 0, st>>>(p); break;
        default: ink_flash_k<D, 3><<<grid, 384, 0, st>>>(p); break;
    }
}

// INTARSIA_HIP_INK_PF2 bit 2 (lane ink-hip-pf2): a call estimated over the R9700's ~1 ms dispatch stall line (ik_hip.h) runs as
// pieces of consecutive query tiles, highest first, balanced by their key tiles; a block of a piece is the block of the one
// launch with the same query tile, so the bits are the same. Pieces keep at least kFlashPieceBlocks blocks (a 2048-row call is
// 256 blocks, about 64 resident; INTARSIA_HIP_INK_PF2_BLOCKS sets it, for tests): a call that would need more pieces than that
// allows stays one launch. Host position only.
constexpr double kFlashUsPerTile = 0.30;  // us per (block, 64-key tile), from window 1's profile (sliding layers ~0.31, full ~0.2-0.27)
constexpr int kFlashPieceBlocks = 128;
template <int D>
void launch_flash_d(const InkP& p, hipStream_t st) {
    const int nqt = (p.t + FBQ - 1) / FBQ;
    int pieces = 1;
    int tiles[1024];
    if ((pf2_mask() & 2) && !p.pos && nqt <= 1024) {
        double sum = 0.0;
        for (int qt = 0; qt < nqt; ++qt) {
            const int q0 = qt * FBQ, q_last = (q0 + FBQ < p.t ? q0 + FBQ : p.t) - 1;
            const int64_t kv_end = (int64_t)p.pos0 + q_last + 1;
            const int64_t lo_first = ink_lo((unsigned)(p.pos0 + q0), p.window);
            tiles[qt] = (int)((kv_end + FBK - 1) / FBK - lo_first / FBK);
            sum += tiles[qt];
        }
        const char* eb = getenv("INTARSIA_HIP_INK_PF2_BLOCKS");  // tests: fewer blocks per piece (1 = a query tile per piece)
        const int minb = eb && *eb ? (atoi(eb) > 0 ? atoi(eb) : 1) : kFlashPieceBlocks;
        const int maxp = (int)((int64_t)nqt * p.hk / minb);
        const int want = pf2_pieces(sum * p.hk * kFlashUsPerTile, nqt);
        pieces = want <= (maxp < 1 ? 1 : maxp) ? want : 1;
        if (pieces > 1) {
            // cut by cumulative key tiles, from the top query tile down
            int hi = nqt;
            double left = sum;
            for (int i = 0; i < pieces; ++i) {
                int lo = hi;
                const double goal = left / (pieces - i);
                double acc = 0.0;
                while (lo > (pieces - 1 - i) && (acc < goal || lo == hi)) acc += tiles[--lo];
                if (i == pieces - 1) lo = 0;
                InkP pp = p;
                pp.qhi = hi;
                launch_flash_grid<D>(pp, dim3((unsigned)(hi - lo), p.hk), st);
                left -= acc;
                hi = lo;
            }
            return;
        }
    }
    launch_flash_grid<D>(p, dim3(nqt, p.hk), st);
}

// ------------------------------------------------------------------------------------------------------------------------
// launchers
// ------------------------------------------------------------------------------------------------------------------------

int chk(const ik_tensor* t, int ndim) { return ik_check(t, IK_DTYPE_BF16, ndim); }
const bf16* cb(const ik_tensor* t) { return (const bf16*)t->data; }
bf16* mb(ik_tensor* t) { return (bf16*)t->data; }
int fits_u32(int64_t v) { return v >= 0 && v <= 0x7fffffffLL; }
int split_keys(const ik_ink_attn_attrs* a) { return a->split_keys > 0 ? (int)a->split_keys : SPLIT_KEYS; }

// INTARSIA_INK_DEC_FUSED_KEYS: op 75 takes its fused kernel where a visit reads at most this many keys, else its split path
// (default 2048; 0 = always the split path, a large value = always fused). Both give the pair's bits.
int64_t dec_fused_keys() {
    static const int64_t v = [] {
        const char* e = getenv("INTARSIA_INK_DEC_FUSED_KEYS");
        return e && *e ? (int64_t)atoll(e) : (int64_t)2048;
    }();
    return v;
}

// INTARSIA_HIP_INK_DEC_FUSED_ROWS (lane ink-hip-vrow): op 75 takes its fused kernel only for calls of at most this many rows (verify
// passes of more rows take the split path, whose prologue forms q / r once per row instead of once per split block); unset: 1 (the
// decode step fused, verify passes split: 2..8-row passes 0.2-1.0 ms faster at a 1k context; main took the fused kernel for all, `=16`).
// Both paths give the pair's bits.
constexpr int kDecFusedRowsDefault = 1;
int g_dec_fused_rows = -1;  // tests (ik_ink_attn_dec_fused_rows): >= 0 overrides the env / the default
int dec_fused_rows() {
    if (g_dec_fused_rows >= 0) return g_dec_fused_rows;
    static const int v = [] {
        const char* e = getenv("INTARSIA_HIP_INK_DEC_FUSED_ROWS");
        return e && e[0] >= '0' && e[0] <= '9' ? atoi(e) : kDecFusedRowsDefault;
    }();
    return v;
}

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

namespace {
// common prologue checks; fills the prologue part of `p`
int prologue_params(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt, const ik_tensor* qs,
                    const ik_tensor* rs, const ik_tensor* rp, const ik_ink_attn_attrs* a, ik_tensor* kring, ik_tensor* vring, const ik_tensor* pos,
                    InkP& p) {
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
    const int64_t h = a->n_heads, hk = a->n_kv_heads, d = a->head_dim, dr = a->d_rel;
    if (hk <= 0 || h % hk != 0 || h / hk > 4 || d <= 0 || d > 256 || dr <= 0 || dr > 32 || kt->shape[0] > KMAX || kt->shape[0] < 1)
        return IK_ERR_SHAPE;
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
    return IK_OK;
}
}  // namespace

// The solo launchers. kernels/hip/ink_attn_rows.hip includes this file with IK_INK_ATTN_ROWS_TU defined and defines the row-table
// launcher (ink_attn_decode_bf16_rows) in its own translation unit: with the RT instantiations of ink_attn_dec_k in this module, its
// RT = false instantiations compiled to different code than main's (register allocation; lane ink-hip-batch), apart they are main's.
#ifndef IK_INK_ATTN_ROWS_TU
extern "C" int ik_ink_kv_prologue_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                                       const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_ink_attn_attrs* a, ik_tensor* q,
                                       ik_tensor* rel, ik_tensor* k, ik_tensor* v, ik_tensor* kring, ik_tensor* vring, const ik_tensor* pos,
                                       void* stream) {
    InkP p = {};
    IK_TRY(prologue_params(pq, pk, pv, kt, vt, qs, rs, rp, a, kring, vring, pos, p));
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
    hipStream_t st = IK_STREAM(stream);
    // decode (T <= 4): the rel tail in slices of 256 elements over more blocks (latency, not work); otherwise one block per
    // (kv head, row). The elements' arithmetic does not depend on the slicing.
    unsigned nz = p.t <= 4 ? (unsigned)((p.e + 255) / 256) : 1u;
    if (nz > 8) nz = 8;
    if (nz < 1) nz = 1;
    // prefill chunks (INTARSIA_HIP_INK_PF2 bit 1): the rel tail as its own pass (ink_rel_table_k), the same bits
    const size_t rt_lds = (size_t)RT_ROWS * p.h * p.dr * sizeof(float);
    const bool rel_pass = (pf2_mask() & 1) && p.t > MAXROWS && p.dr <= RT_DR && p.e % RT_EPT == 0 && p.e <= 256 * RT_EPT && rt_lds <= 32768 &&
                          (reinterpret_cast<uintptr_t>(p.relo) & 7u) == 0;
    if (rel_pass) p.flags |= F_R_ROW;
    if (p.d <= 128 && rel_pass && (pf2_mask() & 128)) ink_prologue_k<4, false, true><<<dim3(p.hk, p.t, nz), 256, 0, st>>>(p);
    else if (p.d <= 128) ink_prologue_k<4><<<dim3(p.hk, p.t, nz), 256, 0, st>>>(p);
    else ink_prologue_k<JMAX><<<dim3(p.hk, p.t, nz), 256, 0, st>>>(p);
    if (rel_pass) {
        ink_rel_table_k<<<(unsigned)((p.t + RT_ROWS - 1) / RT_ROWS), 256, rt_lds, st>>>(p);
        p.flags &= ~F_R_ROW;
    }
    if (!fused) {
        const size_t n = (size_t)p.t * p.hk * p.d;
        ink_ring_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
    }
    return ik_launch_status();
}

namespace {
// RelSinkAttn on the flash kernel (attrs->flash, T > 16, Dh >= 16): `p` filled by ik_rel_sink_attn_bf16; then the cache write
// (the kernel read the call's own keys from k / v)
int rel_sink_attn_flash(const InkP& p, hipStream_t st) {
    switch (p.d) {
        case 16: launch_flash_d<16>(p, st); break;
        case 32: launch_flash_d<32>(p, st); break;
        case 64: launch_flash_d<64>(p, st); break;
        default: launch_flash_d<128>(p, st); break;
    }
    const size_t n = (size_t)p.t * p.hk * p.d;
    ink_kv_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
    return ik_launch_status();
}
}  // namespace

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
    // prefill chunks on the flash kernel when the call asks for it (INTARSIA_HIP_INK_ATTN_FLASH, inkattn.rs; not bitwise the
    // split-KV form below)
    if (a->flash && p.t > MAXROWS && p.d >= 16) return rel_sink_attn_flash(p, st);
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

// tests (lane ink-hip-vrow): op 75's rows rule (dec_fused_rows); -1 = INTARSIA_HIP_INK_DEC_FUSED_ROWS / the default
extern "C" void ik_ink_attn_dec_fused_rows(int v) { g_dec_fused_rows = v; }

extern "C" int ik_ink_attn_decode_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                                       const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_tensor* kc_in, const ik_tensor* vc_in,
                                       const ik_ink_attn_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc, ik_tensor* kring, ik_tensor* vring,
                                       const ik_scratch* scratch, const ik_tensor* pos, void* stream) {
    (void)kc_in;
    (void)vc_in;
    InkP p = {};
    IK_TRY(prologue_params(pq, pk, pv, kt, vt, qs, rs, rp, a, kring, vring, pos, p));
    IK_TRY(chk(o, 2));
    IK_TRY(chk(kc, 3));
    IK_TRY(chk(vc, 3));
    if (p.t == 0) return IK_OK;
    const int G = p.h / p.hk;
    if (p.t > MAXROWS || G < 1 || G > 3 || p.d > 128 || !(p.d == 8 || p.d == 16 || p.d == 32 || p.d == 64 || p.d == 128) || p.hk > MAXHK_TICK)
        return IK_ERR_SHAPE;
    if (o->shape[0] != p.t || o->shape[1] != (int64_t)p.h * p.d || kc->shape[0] != p.hk || kc->shape[2] != p.d || vc->shape[1] != kc->shape[1])
        return IK_ERR_SHAPE;
    p.cr = (int)kc->shape[1];
    p.window = (int)a->window;
    p.scale = a->scale;
    p.sk = split_keys(a);
    const bool ring = a->kv_ring != 0;
    if (ring) p.flags |= F_RING;
    if (ring && (p.window < 1 || p.window > p.cr)) return IK_ERR_SHAPE;
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data) || !fits_u32((int64_t)p.hk * p.cr * p.d) || !fits_u32(a->pos0 + p.t)) return IK_ERR_INVALID_ARG;
    p.out = mb(o);
    p.kc = mb(kc);
    p.vc = mb(vc);
    p.pos = pos ? (const int*)pos->data : nullptr;
    p.pos0 = (int)a->pos0;
    const bool ring_fused = p.t + p.k <= p.rr + 1;
    if (ring_fused) p.flags |= F_RING_WRITE;
    const bool kv_fused = !ring || p.t + p.window <= p.cr + 1;
    if (kv_fused) p.flags |= F_KV_WRITE;
    if (a->diag & 1) p.flags |= F_DIAG_NOPRO;
    if (a->diag & 2) p.flags |= F_DIAG_NOMERGE;
    if (ik_hip_dec3_mask() & 1) p.flags |= F_DPP;
    unsigned gy, nb;
    attn_geometry(p, pos != nullptr, &gy, &nb);
    // scratch: the partials (as rel_sink_attn's), the staged K / V rows [2][T][Hk][D] bf16, then the split path's q [T,H,D] and
    // r [T,H,32] bf16 (inkattn.rs `decode_scratch`)
    const int64_t part = (int64_t)p.t * p.h * MAXS * (p.d + 2) * 4;
    const int64_t stage = (int64_t)2 * p.t * p.hk * p.d * 2;
    const int64_t qb = (int64_t)p.t * p.h * p.d * 2;
    if (!scratch || !scratch->data || scratch->bytes < part + stage + qb + (int64_t)p.t * p.h * 32 * 2) return IK_ERR_INVALID_ARG;
    p.scr = (float*)scratch->data;
    bf16* stg = (bf16*)((char*)scratch->data + part);
    p.ko = stg;
    p.vo = stg + (size_t)p.t * p.hk * p.d;
    hipStream_t st = IK_STREAM(stream);
    // the rule: the fused kernel where the visit reads at most dec_fused_keys() keys (window layers always; full layers at a
    // host position up to that context), else the split path (prologue + stage 1 + combine, the pair's kernels; bitwise the
    // same). Under a device position a full layer's context is unknown here: the capacity bound decides.
    const int64_t keys = p.window > 0 ? (p.window < p.cr ? p.window : p.cr) : (pos ? (int64_t)p.cr : (int64_t)p.pos0 + p.t);
    if (keys > dec_fused_keys() || (int64_t)p.t > dec_fused_rows()) {
        InkP q = p;
        q.flags = (q.flags & (F_DYN | F_RING_WRITE | F_DPP)) | F_R_OUT;
        q.qo = (bf16*)((char*)scratch->data + part + stage);
        q.relo = (bf16*)((char*)scratch->data + part + stage + qb);
        q.e = p.dr;  // unused under F_R_OUT
        if (q.flags & F_DPP) ink_prologue_k<4, true><<<dim3(p.hk, p.t, 1), 256, 0, st>>>(q);
        else ink_prologue_k<4><<<dim3(p.hk, p.t, 1), 256, 0, st>>>(q);
        if (!ring_fused) {
            const size_t n = (size_t)p.t * p.hk * p.d;
            ink_ring_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
        }
        InkP a1 = p;
        a1.flags = (p.flags & (F_RING | F_KV_WRITE | F_DPP)) | F_R_IN;
        a1.q = q.qo;
        a1.kn = p.ko;
        a1.vn = p.vo;
        a1.rel = q.relo;
        const size_t tbb = (size_t)G * nb * 4;
        const dim3 grid(p.hk, gy, p.t);
        switch (G) {
            case 1: if (a1.flags & F_DPP) ink_attn_s1_k<1, true><<<grid, NSG * 32, tbb, st>>>(a1); else ink_attn_s1_k<1><<<grid, NSG * 32, tbb, st>>>(a1); break;
            case 2: if (a1.flags & F_DPP) ink_attn_s1_k<2, true><<<grid, NSG * 32, tbb, st>>>(a1); else ink_attn_s1_k<2><<<grid, NSG * 32, tbb, st>>>(a1); break;
            default: if (a1.flags & F_DPP) ink_attn_s1_k<3, true><<<grid, NSG * 32, tbb, st>>>(a1); else ink_attn_s1_k<3><<<grid, NSG * 32, tbb, st>>>(a1); break;
        }
        if (gy > 1) ink_attn_comb_k<<<dim3(p.h, p.t), (p.d + 31) / 32 * 32, 0, st>>>(a1);
        if (!kv_fused) {
            const size_t n = (size_t)p.t * p.hk * p.d;
            ink_kv_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(a1);
        }
        return ik_launch_status();
    }
    const size_t tbb = (size_t)G * nb * 4;
    const dim3 grid(p.hk, gy, p.t);
    switch (G) {
        case 1: if (p.flags & F_DPP) ink_attn_dec_k<1, true><<<grid, DEC_WAVES * 32, tbb, st>>>(p); else ink_attn_dec_k<1><<<grid, DEC_WAVES * 32, tbb, st>>>(p); break;
        case 2: if (p.flags & F_DPP) ink_attn_dec_k<2, true><<<grid, DEC_WAVES * 32, tbb, st>>>(p); else ink_attn_dec_k<2><<<grid, DEC_WAVES * 32, tbb, st>>>(p); break;
        default: if (p.flags & F_DPP) ink_attn_dec_k<3, true><<<grid, DEC_WAVES * 32, tbb, st>>>(p); else ink_attn_dec_k<3><<<grid, DEC_WAVES * 32, tbb, st>>>(p); break;
    }
    if (!kv_fused) {
        InkP pw = p;
        pw.kn = p.ko;
        pw.vn = p.vo;
        const size_t n = (size_t)p.t * p.hk * p.d;
        ink_kv_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(pw);
    }
    if (!ring_fused) {
        const size_t n = (size_t)p.t * p.hk * p.d;
        ink_ring_write_k<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(p);
    }
    return ik_launch_status();
}
#endif  // IK_INK_ATTN_ROWS_TU
