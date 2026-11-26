// Inkling-2-Mini sublayer-boundary ops on HIP (lane ink-hip-mix): SconvRes (69), AttnResMix (67), ConvMix (74), LoopBlend (68).
// A port of kernels/cuda/ink_mix.cu (the correctness form: SIMT, wave32). Contract: crates/intarsia-core/src/inkling2/ops.rs;
// reference: intarsia-backend-cpu ink_ref.rs. bf16 activations, f32 math, the rounding points of the contract.
//
// The row kernel (one block per row, the same code for every row count, so rows-exact and chunk-invariant): the static conv
// with its residual epilogue, then the depth mix of the next sublayer (sources, queries, softmax over the sources), then the
// norm, the f32 mix, or the pair of the blend. Thread `tid` owns elements tid + j * blockDim (j < NE); every per-row sum is a
// per-thread partial in element order, a butterfly over the wave, then the wave partials in wave order, so the bits depend on
// D only. Block size and NE depend on D only.
//
// The ring: the conv reads ring rows of positions < pos0 and the call writes the rows of its last Rr positions. When
// T + K - 1 <= Rr (every decode and verify call: Rr = K - 1 + 16) the rows read and written are disjoint and the row kernel
// writes its own row; otherwise a second launch writes them after the reads.
//
// The blend (spec a12): the row kernel writes the pair rA([d w(d), m w(m)]) and the dest d (f32), then
//   h      = rA(pair . down^T)       a wave GEMV (one wave per (stream, controller row g), rows in groups of kHRows; each output a
//                                    lane partial in fixed order + a wave butterfly, so independent of the row count)
//   gate   y = (1 - tanh(fg)) d + tanh(gn) m with gn / fg = gain / forget . h (one thread per channel, 16-row chunks), and the
//          slice partials of sum(y^2)
//   norm   rA(y inv(y) norm) with the partials in slice order
// in chunks of kBlendRows rows (the scratch holds one chunk).
#include <stdlib.h>
#include <type_traits>
#include "ik_hip.h"
#include "ik_hip_abi_inkmix.h"

// Phase timestamps of block 0 / thread 0 (INK_PROBE builds only; tests/ink_mix_probe.rs reads them with ik_ink_probe): wall_clock64 ticks (100 MHz).
#ifdef INK_PROBE
__device__ unsigned long long g_probe[32];
#define INK_MARK(i) \
    do { __builtin_amdgcn_s_waitcnt(0); if (blockIdx.x == 0 && threadIdx.x == 0) g_probe[i] = wall_clock64(); } while (0)
#else
#define INK_MARK(i) \
    do { } while (0)
#endif

namespace {

IK_DEVI float ink_rbf(float v) { return ik_b2f(ik_f2b(v)); }

constexpr float kSrcEps = 1e-6f;
constexpr int kMaxThreads = 768;
constexpr int kNsMax = 8;    // stored sources + last
constexpr int kQMax = 4;     // queries
constexpr int kSMax = 3;     // blend streams
constexpr int kGMax = 128;   // controller rank
constexpr int kDMax = 6 * kMaxThreads;
constexpr int kNvMax = kNsMax + kQMax * kNsMax;  // values of the widest reduction
constexpr int kBlendRows = 256;                  // rows per blend chunk
constexpr int kGateCh = 128;                     // channels per gate block (one thread each)
constexpr int kGateRows = 16;                    // rows per gate block
constexpr int kFastRowsMax = 16;                 // launches of at most this many rows take the fast row kernel (mask bit 1)
constexpr int kFuseRowsMax = 2;                  // blend chunks of at most this many rows take the fused launches
constexpr int kFuseD = 2304;                     // widest row of the fused blend (the pair lives in shared memory)

// tickets of the gate kernel's fused norm: [stream][row group], zero between launches (the last block of a group resets its own)
__device__ unsigned int g_gate_ctr[3 * 16];

// DPP (INTARSIA_HIP_INK_DEC3 bit 1, lane ink-hip-dec3): the partners from DPP / v_permlanex16 instead of ds_bpermute; the same
// partners in the same order (ik_warp_sum_dpp), so the same bits
template <bool DPP>
IK_DEVI float bsum(float v) {
    if constexpr (DPP) return ik_warp_sum_dpp(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, IK_WARP);
    return v;
}

// red: [32][kNvMax] warp partials; fin: [kNvMax]. Every thread must call with the same nv. Adds value i of the warp into slot i.
template <bool DPP>
IK_DEVI void warp_put(float* red, int i, float v) {
    v = bsum<DPP>(v);
    if ((threadIdx.x & 31) == 0) red[(threadIdx.x >> 5) * kNvMax + i] = v;
}
template <bool FAST>
IK_DEVI void block_fin(const float* red, float* fin, int nv) {
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    if ((int)threadIdx.x < nv) {
        float s = 0.f;
        if constexpr (FAST) {
            // the same chain s = ((0 + w0) + w1) + ..., with the loads of several waves in flight
#pragma unroll 8
            for (int w = 0; w < nw; ++w) s += red[w * kNvMax + threadIdx.x];
        } else {
            for (int w = 0; w < nw; ++w) s += red[w * kNvMax + threadIdx.x];
        }
        fin[threadIdx.x] = s;
    }
    __syncthreads();
}

struct RowArgs {
    // conv (x null: no conv)
    const ik_bf16* x;
    const ik_bf16* taps;
    const ik_bf16* res;
    const ik_bf16* add;
    ik_bf16* ring;
    ik_bf16* y;
    ik_bf16* sum;
    int K, Rr;
    int ring_now;
    // AttnResMix: last [R,D] and its addend (used when x is null)
    const ik_bf16* last;
    const ik_bf16* last_add;
    // mix
    const ik_bf16* src;  // [N,T,D]
    int N;
    const float* q;  // [Q,D]
    int Q;
    const ik_bf16* norm;  // [Q,D] (no blend) or [S,D] (unused by the row kernel with a blend)
    ik_bf16* out_b;       // [Q,T,D] with norm
    float* out_f;         // [Q,T,D] without
    // blend: dest from the mix (dest null) or the LoopBlend input, msg f32 [S,T,D]
    const float* dest;  // [Sd,T,D]
    int dest_bcast;     // Sd == 1
    const float* msg;
    int S;
    ik_bf16* pair;  // scratch [S,Tc,2D]
    float* dbuf;    // scratch [S,Tc,D]: the dest of each stream (f32)
    // geometry
    int T, D, t0;  // rows of the whole call, width, first row of this launch (blend chunks)
    int64_t pos0;
    const int* pos;
    float eps;
    // fused blend (a.fuse): np = S * G / 8 blocks per row, the controller rank G, down [S,G,2D], the h scratch [S,rows,G]
    int fuse, np, G;
    const ik_bf16* down;
    ik_bf16* hbuf;
};

// The row-table form (lane ink-hip-batch, `OpCall::rows`): the table's device addresses ([rows][rts] entries), this call's entry, the
// ring carrier's byte offset in its row's buffer (`pos` is then i32 [T]). Only the RT instantiations take `RowArgsRT`, so the solo
// kernels' arguments stay `RowArgs`, byte for byte.
struct RowRt {
    const unsigned long long* rtab;
    int rts, rti;
    long long roff;
};
struct RowArgsRT : RowArgs {
    RowRt rt;
};

// The row kernel. FAST = false is the first form (loads where the arithmetic needs them); FAST = true issues every load of the row
// (taps and ring rows of the conv, res / addend, the sources, the queries, the norm weights) up front, unrolled and without
// branches between them, so one memory latency is paid instead of one per tap and per phase, and unrolls the wave-partial sum
// of block_fin. Arithmetic, operation order and rounding are the same in both: the two forms are bitwise equal (tests/ink_mix.rs
// runs every call in both).
// Blend with `a.fuse` (rows <= kFuseRowsMax, D <= kFuseD): np = S * G / 8 blocks per row; every block of a row computes the row
// redundantly, the block (s, gq) forms the pair of stream s in shared memory and the controller rows g = 8 gq .. 8 gq + 7 of that
// stream (one wave per g, the wave GEMV of ink_ctl_gemv_kernel, same lane order), block part 0 writes y / ring / sum.
// RT (`conv_mix_bf16_rows`, lane ink-hip-batch): row t is the T = 1 call of sequence t at position pos[t] on its own ring (table
// entry rti of row t + roff): its conv history is its own x row (qp == p) and its ring (qp < p). Every other operand keeps the
// call's row t (the planes, the mix and the blend are row-parallel). RT = false is the solo kernel.
template <int NE, bool FAST, bool DPP, bool RT = false>
__global__ void __launch_bounds__(kMaxThreads) ink_row_kernel(std::conditional_t<RT, RowArgsRT, RowArgs> a) {
    __shared__ float red[32 * kNvMax];
    __shared__ float fin[kNvMax];
    __shared__ __attribute__((aligned(16))) ik_bf16 pl[2 * kFuseD];
    const int np = a.fuse ? a.np : 1;
    const int bx = blockIdx.x / np;
    const int part = blockIdx.x - bx * np;
    const bool wr = part == 0;  // the block that writes the row's outputs
    const int t = a.t0 + bx;
    const int tl = bx;  // row in the blend chunk
    const int rows = gridDim.x / np;
    const int D = a.D;
    const int64_t T = a.T;
    const int nt = blockDim.x;
    int ix[NE];
    bool ok[NE];
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        ix[j] = threadIdx.x + j * nt;
        ok[j] = ix[j] < D;
        if (!ok[j]) ix[j] = 0;
    }
    INK_MARK(0);
    const int Q = a.Q;
    const bool do_mix = Q > 0 && !a.dest;
    float xs[kNsMax][NE];
    float qv[kQMax][NE];
    float nwv[kQMax][NE];
    float rv[NE], adv[NE], xcur[NE];
    float tv[8][NE], xr[8][NE];
    if constexpr (RT) a.ring = (ik_bf16*)(a.rt.rtab[(size_t)t * a.rt.rts + a.rt.rti] + a.rt.roff);
    const int64_t pos0 = RT ? (int64_t)a.pos[t] - t : (a.pos ? (int64_t)*a.pos : a.pos0);
    const int64_t p = pos0 + t;
    const int64_t own0 = RT ? p : pos0;  // the first position the conv reads from x (the call's own rows), older ones from the ring
    const bool fastconv = FAST && a.x && a.K <= 8;
    bool vr[8];
    if constexpr (FAST) {
        // Every load below is issued unconditionally (clamped, always valid addresses) and its value selected afterwards: a load under a
        // lane condition makes the compiler wait for it before the merge, so conditional loads cost one memory latency (~170 ns) each.
        if (fastconv) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int lc = l < a.K ? l : a.K - 1;
                const int64_t qp = p - lc;
                const bool inx = qp >= own0, inr = !inx && qp >= 0;
                const ik_bf16* row = inx ? a.x + (qp - pos0) * D : (inr ? a.ring + (int64_t)((int)qp % a.Rr) * D : a.taps);
                const ik_bf16* tap = a.taps + (int64_t)lc * D;
                vr[l] = inx || inr;
#pragma unroll
                for (int j = 0; j < NE; ++j) {
                    tv[l][j] = ik_b2f(tap[ix[j]]);
                    xr[l][j] = ik_b2f(row[ix[j]]);
                }
            }
        }
        if (a.x) {
            const ik_bf16* rs = a.res ? a.res : a.x;
            const ik_bf16* as = a.add ? a.add : a.x;
#pragma unroll
            for (int j = 0; j < NE; ++j) {
                const int64_t o = (int64_t)t * D + ix[j];
                const float r0 = ik_b2f(rs[o]), a0 = ik_b2f(as[o]);
                rv[j] = (ok[j] && a.res) ? r0 : 0.f;
                adv[j] = (ok[j] && a.add) ? a0 : 0.f;
            }
        }
        if (do_mix) {
            const ik_bf16* sfb = a.x ? a.x : a.last;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                const ik_bf16* sp = a.N > 0 ? a.src + ((int64_t)(n < a.N ? n : 0) * T + t) * D : sfb + (int64_t)t * D;
#pragma unroll
                for (int j = 0; j < NE; ++j) {
                    const float v = ik_b2f(sp[ix[j]]);
                    xs[n][j] = (n < a.N && ok[j]) ? v : 0.f;
                }
            }
#pragma unroll
            for (int qi = 0; qi < kQMax; ++qi) {
                if (qi < Q) {
#pragma unroll
                    for (int j = 0; j < NE; ++j) {
                        const float v = a.q[(int64_t)qi * D + ix[j]];
                        qv[qi][j] = ok[j] ? v : 0.f;
                    }
                }
            }
            if (!a.msg && a.norm) {
#pragma unroll
                for (int qi = 0; qi < kQMax; ++qi) {
                    if (qi < Q) {
#pragma unroll
                        for (int j = 0; j < NE; ++j) {
                            const float v = ik_b2f(a.norm[(int64_t)qi * D + ix[j]]);
                            nwv[qi][j] = ok[j] ? v : 0.f;
                        }
                    }
                }
            }
        }
    }
    float last[NE];
    if (a.x) {
        const float sc = rsqrtf((float)(a.K + 1));
        float yv[NE], acc[NE];
#pragma unroll
        for (int j = 0; j < NE; ++j) acc[j] = 0.f;
        if (fastconv) {
            // each acc[j] sums l = 0 .. K-1 in order, as the loop below
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                if (l < a.K) {
#pragma unroll
                    for (int j = 0; j < NE; ++j) acc[j] = fmaf(tv[l][j], vr[l] ? xr[l][j] : 0.f, acc[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < NE; ++j) xcur[j] = xr[0][j];  // tap 0 reads the current row: x[t]
        } else {
            // taps outer, elements inner: each acc[j] sums l = 0 .. K-1 in order, the source row of a tap (and its ring modulo) is
            // formed once per tap instead of once per tap and element
            for (int l = 0; l < a.K; ++l) {
                const int64_t qp = p - l;
                const ik_bf16* row = qp >= own0 ? a.x + (qp - pos0) * D : qp >= 0 ? a.ring + (int64_t)((int)qp % a.Rr) * D : nullptr;
                const ik_bf16* tap = a.taps + (int64_t)l * D;
#pragma unroll
                for (int j = 0; j < NE; ++j) acc[j] = fmaf(ik_b2f(tap[ix[j]]), row ? ik_b2f(row[ix[j]]) : 0.f, acc[j]);
            }
#pragma unroll
            for (int j = 0; j < NE; ++j) xcur[j] = ik_b2f(a.x[(int64_t)t * D + ix[j]]);
        }
        if constexpr (!FAST) {
#pragma unroll
            for (int j = 0; j < NE; ++j) {
                const int64_t o = (int64_t)t * D + ix[j];
                rv[j] = (ok[j] && a.res) ? ik_b2f(a.res[o]) : 0.f;
                adv[j] = (ok[j] && a.add) ? ik_b2f(a.add[o]) : 0.f;
            }
        }
#pragma unroll
        for (int j = 0; j < NE; ++j) yv[j] = ink_rbf((acc[j] + xcur[j]) * sc);
        // y, the ring row of this position (no row of the launch reads it), the sum
        const int64_t ring_off = a.Rr > 0 ? (p % a.Rr) * D : 0;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (!ok[j]) continue;
            const int c = ix[j];
            const int64_t o = (int64_t)t * D + c;
            if (wr) {
                a.y[o] = ik_f2b(yv[j]);
                if (a.ring_now && (RT || p >= pos0 + T - a.Rr)) a.ring[ring_off + c] = a.x[o];  // RT: every row is a T = 1 call
            }
            if (a.sum) {
                const float s = ink_rbf((rv[j] + adv[j]) + yv[j]);
                if (wr) a.sum[o] = ik_f2b(s);
                last[j] = s;
            } else if (a.res) {
                last[j] = rv[j] + yv[j];
            } else {
                last[j] = yv[j];
            }
        }
    } else if (a.last) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const int64_t o = (int64_t)t * D + ix[j];
            last[j] = ok[j] ? ik_b2f(a.last[o]) + (a.last_add ? ik_b2f(a.last_add[o]) : 0.f) : 0.f;
        }
    }
    INK_MARK(1);
    float mix[kQMax][NE];
    if (do_mix) {
        // ---- depth mix: sources src[0..N) and last ----
        const int NS = a.N + 1;
#pragma unroll
        for (int n = 0; n < kNsMax; ++n) {
#pragma unroll
            for (int j = 0; j < NE; ++j) {
                float v = 0.f;
                if constexpr (!FAST) {
                    if (n < a.N) v = ok[j] ? ik_b2f(a.src[((int64_t)n * T + t) * D + ix[j]]) : 0.f;
                } else {
                    if (n < a.N) v = xs[n][j];
                }
                if (n == a.N) v = ok[j] ? last[j] : 0.f;
                xs[n][j] = v;
            }
        }
        INK_MARK(8);
#pragma unroll
        for (int n = 0; n < kNsMax; ++n) {
            if (n >= NS) break;
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < NE; ++j) s = fmaf(xs[n][j], xs[n][j], s);
            warp_put<DPP>(red, n, s);
        }
        INK_MARK(9);
#pragma unroll
        for (int qi = 0; qi < kQMax; ++qi) {
            if (qi >= Q) break;
            float qq[NE];
#pragma unroll
            for (int j = 0; j < NE; ++j) {
                if constexpr (FAST) qq[j] = qv[qi][j];
                else qq[j] = ok[j] ? a.q[(int64_t)qi * D + ix[j]] : 0.f;
            }
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                if (n >= NS) break;
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < NE; ++j) s = fmaf(qq[j], xs[n][j], s);
                warp_put<DPP>(red, kNsMax + qi * kNsMax + n, s);
            }
        }
        INK_MARK(2);
        block_fin<FAST>(red, fin, kNvMax);
        INK_MARK(3);
        float wn[kNsMax];
#pragma unroll
        for (int n = 0; n < kNsMax; ++n) wn[n] = n < NS ? 1.f / sqrtf(fin[n] / (float)D + kSrcEps) : 0.f;
#pragma unroll
        for (int qi = 0; qi < kQMax; ++qi) {
            if (qi >= Q) break;
            float l[kNsMax], mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                l[n] = n < NS ? fin[kNsMax + qi * kNsMax + n] * wn[n] : -INFINITY;
                mx = fmaxf(mx, l[n]);
            }
            float e[kNsMax], se = 0.f;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                e[n] = n < NS ? expf(l[n] - mx) : 0.f;
                se += e[n];
            }
#pragma unroll
            for (int j = 0; j < NE; ++j) mix[qi][j] = 0.f;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                if (n >= NS) break;
                const float cf = e[n] / se * wn[n];
#pragma unroll
                for (int j = 0; j < NE; ++j) mix[qi][j] = fmaf(cf, xs[n][j], mix[qi][j]);
            }
        }
    }
    INK_MARK(4);
    if (Q > 0 && !a.msg) {
        if (a.norm) {
#pragma unroll
            for (int qi = 0; qi < kQMax; ++qi) {
                if (qi >= Q) break;
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < NE; ++j) s = fmaf(mix[qi][j], mix[qi][j], s);
                warp_put<DPP>(red, qi, s);
            }
            INK_MARK(5);
            block_fin<FAST>(red, fin, Q);
            INK_MARK(6);
#pragma unroll
            for (int qi = 0; qi < kQMax; ++qi) {
                if (qi >= Q) break;
                const float r = 1.f / sqrtf(fin[qi] / (float)D + a.eps);
#pragma unroll
                for (int j = 0; j < NE; ++j) {
                    if (!ok[j]) continue;
                    float nw;
                    if constexpr (FAST) nw = nwv[qi][j];
                    else nw = ik_b2f(a.norm[(int64_t)qi * D + ix[j]]);
                    a.out_b[((int64_t)qi * T + t) * D + ix[j]] = ik_f2b(mix[qi][j] * r * nw);
                }
            }
            INK_MARK(7);
        } else {
#pragma unroll
            for (int qi = 0; qi < kQMax; ++qi) {
                if (qi >= Q) break;
#pragma unroll
                for (int j = 0; j < NE; ++j)
                    if (ok[j]) a.out_f[((int64_t)qi * T + t) * D + ix[j]] = mix[qi][j];
            }
        }
        return;
    }
    if (!a.msg) return;
    if (a.fuse) {
        // ---- fused blend: stream s of this block, its pair in shared memory, its controller rows ----
        const int gqn = a.G / 8;
        const int s = part / gqn, gq = part - s * gqn;
        float dvs[NE], mvs[NE];
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            float d;
            if (a.dest) {
                d = ok[j] ? a.dest[((int64_t)(a.dest_bcast ? 0 : s) * T + t) * D + ix[j]] : 0.f;
            } else {
                d = 0.f;
#pragma unroll
                for (int qq = 0; qq < kQMax; ++qq)
                    if (qq == (Q == 1 ? 0 : (s < kQMax ? s : 0))) d = mix[qq][j];
            }
            dvs[j] = ok[j] ? d : 0.f;
            mvs[j] = ok[j] ? a.msg[((int64_t)s * T + t) * D + ix[j]] : 0.f;
        }
        float sd = 0.f, sm = 0.f;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            sd = fmaf(dvs[j], dvs[j], sd);
            sm = fmaf(mvs[j], mvs[j], sm);
        }
        warp_put<DPP>(red, 0, sd);
        warp_put<DPP>(red, 1, sm);
        block_fin<FAST>(red, fin, 2);
        const float wd = 1.f / sqrtf(fin[0] / (float)D + kSrcEps);
        const float wm = 1.f / sqrtf(fin[1] / (float)D + kSrcEps);
        float* db = a.dbuf + ((int64_t)s * rows + tl) * D;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (!ok[j]) continue;
            pl[ix[j]] = ik_f2b(dvs[j] * wd);
            pl[D + ix[j]] = ik_f2b(mvs[j] * wm);
            if (gq == 0) db[ix[j]] = dvs[j];
        }
        __syncthreads();
        const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, nwaves = nt >> 5;
        const int64_t nv = (2 * D) / 8;
        const uint4* xl = reinterpret_cast<const uint4*>(pl);
        for (int g = gq * 8 + wave; g < gq * 8 + 8; g += nwaves) {
            const uint4* w8 = reinterpret_cast<const uint4*>(a.down + ((int64_t)s * a.G + g) * 2 * D);
            float acc = 0.f;
#pragma unroll 6
            for (int64_t i = lane; i < nv; i += IK_WARP) {
                float wf[8], xf[8];
                ik_unpack8(ik_ldcs(w8 + i), wf);
                ik_unpack8(xl[i], xf);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc = fmaf(wf[e], xf[e], acc);
            }
            const float v = DPP ? ik_warp_sum_dpp(acc) : ik_warp_sum(acc);
            if (lane == 0) a.hbuf[((int64_t)s * rows + tl) * a.G + g] = ik_f2b(v);
        }
        return;
    }
    // ---- blend: the pair of every stream ----
    const int S = a.S;
    float dv[kSMax][NE], mv[kSMax][NE];
#pragma unroll
    for (int s = 0; s < kSMax; ++s) {
        if (s >= S) break;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            float d;
            if (a.dest) d = ok[j] ? a.dest[((int64_t)(a.dest_bcast ? 0 : s) * T + t) * D + ix[j]] : 0.f;
            else d = Q == 1 ? mix[0][j] : mix[s < kQMax ? s : 0][j];
            dv[s][j] = ok[j] ? d : 0.f;
            mv[s][j] = ok[j] ? a.msg[((int64_t)s * T + t) * D + ix[j]] : 0.f;
        }
        float sd = 0.f, sm = 0.f;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            sd = fmaf(dv[s][j], dv[s][j], sd);
            sm = fmaf(mv[s][j], mv[s][j], sm);
        }
        warp_put<DPP>(red, 2 * s, sd);
        warp_put<DPP>(red, 2 * s + 1, sm);
    }
    block_fin<FAST>(red, fin, 2 * S);
#pragma unroll
    for (int s = 0; s < kSMax; ++s) {
        if (s >= S) break;
        const float wd = 1.f / sqrtf(fin[2 * s] / (float)D + kSrcEps);
        const float wm = 1.f / sqrtf(fin[2 * s + 1] / (float)D + kSrcEps);
        ik_bf16* pr = a.pair + ((int64_t)s * rows + tl) * 2 * D;
        float* db = a.dbuf + ((int64_t)s * rows + tl) * D;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (!ok[j]) continue;
            pr[ix[j]] = ik_f2b(dv[s][j] * wd);
            pr[D + ix[j]] = ik_f2b(mv[s][j] * wm);
            db[ix[j]] = dv[s][j];
        }
    }
}

// the deferred ring write: positions max(pos0, pos0 + T - Rr) .. pos0 + T
__global__ void ink_ring_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ ring, int T, int D, int Rr, int64_t pos0h,
                                const int* pos) {
    const int64_t pos0 = pos ? (int64_t)*pos : pos0h;
    const int64_t q0 = (T > Rr) ? pos0 + T - Rr : pos0;
    const int64_t q = q0 + blockIdx.y;
    if (q >= pos0 + T) return;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < D; c += gridDim.x * blockDim.x)
        ring[(q % Rr) * D + c] = x[(q - pos0) * D + c];
}

// The fused norm of the gate kernel (blend chunks of at most kFuseRowsMax rows): the last block of a (stream, row group) to finish
// forms the group's norm, so the separate norm launch goes away; `on` = 0: the separate kernel below does it.
struct GateNorm {
    const ik_bf16* norm;
    ik_bf16* out;
    float eps;
    int on;
    int tail;  // INTARSIA_HIP_INK_DEC_FAST bit 2 (lane ink-hip-dec2): the slice partials staged by all threads at once, the norm loop's loads issued 4 at a time
};
constexpr int kTailPs = 256;  // slice partials of a group the fast tail stages (rows x slices; kFuseRowsMax x 18 at D 2304)

// gate: y = (1 - tanh(fg)) d + tanh(gn) m for one stream, 128 channels x 16 rows of a chunk; writes y and the block's slice
// partial of sum(y^2) per row. FAST unrolls the loads of the controller rows (the same fmaf chain per output).
template <bool FAST, bool DPP>
__global__ void __launch_bounds__(kGateCh) ink_blend_gate_kernel(const ik_bf16* __restrict__ h, const ik_bf16* __restrict__ gain,
                                                                 const ik_bf16* __restrict__ forget, const float* __restrict__ dbuf,
                                                                 const float* __restrict__ msg, float* __restrict__ ybuf,
                                                                 float* __restrict__ part, int Tc, int T, int t0, int D, int G,
                                                                 GateNorm gn_) {
    __shared__ float hs[kGateRows][kGMax];
    __shared__ float red[kGateRows][kGateCh / 32];
    __shared__ int last_blk;
    __shared__ float rsh[kGateRows];
    __shared__ float psh[kTailPs];
    const int s = blockIdx.y;
    const int r0 = blockIdx.z * kGateRows;
    const int rows = min(kGateRows, Tc - r0);
    const int c = blockIdx.x * kGateCh + threadIdx.x;
    const bool on = c < D;
    const int cc = on ? c : 0;
    const ik_bf16* gr = gain + ((int64_t)s * D + cc) * G;
    const ik_bf16* fr = forget + ((int64_t)s * D + cc) * G;
    // FAST: the first two controller chunks of this channel do not depend on h: their loads are issued before h is staged
    uint4 gp0 = make_uint4(0, 0, 0, 0), fp0 = gp0, gp1 = gp0, fp1 = gp0;
    if constexpr (FAST) {
        gp0 = ik_ldg(reinterpret_cast<const uint4*>(gr));
        fp0 = ik_ldg(reinterpret_cast<const uint4*>(fr));
        if (G > 8) {
            gp1 = ik_ldg(reinterpret_cast<const uint4*>(gr + 8));
            fp1 = ik_ldg(reinterpret_cast<const uint4*>(fr + 8));
        }
    }
    for (int i = threadIdx.x; i < rows * G; i += blockDim.x) hs[i / G][i % G] = ik_b2f(h[((int64_t)s * Tc + r0 + i / G) * G + i % G]);
    __syncthreads();
    float gn[kGateRows], fg[kGateRows];
#pragma unroll
    for (int r = 0; r < kGateRows; ++r) gn[r] = fg[r] = 0.f;
    auto step = [&](const uint4& gu, const uint4& fu, int g0) {
        float gw[8], fw[8];
        ik_unpack8(gu, gw);
        ik_unpack8(fu, fw);
#pragma unroll
        for (int r = 0; r < kGateRows; ++r) {
            if (r >= rows) break;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                gn[r] = fmaf(gw[e], hs[r][g0 + e], gn[r]);
                fg[r] = fmaf(fw[e], hs[r][g0 + e], fg[r]);
            }
        }
    };
    if constexpr (FAST) {
        step(gp0, fp0, 0);
        if (G > 8) step(gp1, fp1, 8);
        // the loads of 4 iterations are in flight together (the fmaf chain per output keeps its order)
#pragma unroll 4
        for (int g0 = 16; g0 < G; g0 += 8) step(ik_ldg(reinterpret_cast<const uint4*>(gr + g0)), ik_ldg(reinterpret_cast<const uint4*>(fr + g0)), g0);
    } else {
        for (int g0 = 0; g0 < G; g0 += 8) step(ik_ldg(reinterpret_cast<const uint4*>(gr + g0)), ik_ldg(reinterpret_cast<const uint4*>(fr + g0)), g0);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < kGateRows; ++r) {
        if (r >= rows) break;
        const int64_t o = ((int64_t)s * Tc + r0 + r) * D + cc;
        float yv = 0.f;
        if (on) {
            const float d = dbuf[o];
            const float m = msg[((int64_t)s * T + t0 + r0 + r) * D + cc];
            yv = (1.f - tanhf(fg[r])) * d + tanhf(gn[r]) * m;
            ybuf[o] = yv;
        }
        const float q = bsum<DPP>(yv * yv);
        if (lane == 0) red[r][warp] = q;
    }
    __syncthreads();
    if ((int)threadIdx.x < rows) {
        float q = 0.f;
        for (int w = 0; w < kGateCh / 32; ++w) q += red[threadIdx.x][w];
        part[((int64_t)s * Tc + r0 + threadIdx.x) * gridDim.x + blockIdx.x] = q;
    }
    if (!gn_.on) return;
    // ---- fused norm: the last block of this (stream, row group) ----
    __threadfence();  // ybuf (every thread) and part (threads < rows) before the ticket
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(&g_gate_ctr[s * 16 + blockIdx.z], 1u);
        const bool l = old == gridDim.x - 1;
        if (l) atomicExch(&g_gate_ctr[s * 16 + blockIdx.z], 0u);
        last_blk = l;
    }
    __syncthreads();
    if (!last_blk) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const int nsl = gridDim.x;
    if (gn_.tail && rows * nsl <= kTailPs) {
        // the same sums and products as below, in the same order; only the loads are issued together
        const float* pg = part + ((int64_t)s * Tc + r0) * nsl;  // the group's rows x slices, contiguous
        for (int i = threadIdx.x; i < rows * nsl; i += blockDim.x) psh[i] = pg[i];
        __syncthreads();
        if ((int)threadIdx.x < rows) {
            float q = 0.f;
            for (int i = 0; i < nsl; ++i) q += psh[threadIdx.x * nsl + i];
            rsh[threadIdx.x] = 1.f / sqrtf(q / (float)D + gn_.eps);
        }
        __syncthreads();
        const int bd = blockDim.x;
        const ik_bf16* nr = gn_.norm + (int64_t)s * D;
        for (int r = 0; r < rows; ++r) {
            const float* yr = ybuf + ((int64_t)s * Tc + r0 + r) * D;
            ik_bf16* orow = gn_.out + ((int64_t)s * T + t0 + r0 + r) * D;
            const float rr = rsh[r];
            int cj = threadIdx.x;
            for (; cj + 3 * bd < D; cj += 4 * bd) {
                float yv[4], nv[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    yv[u] = yr[cj + u * bd];
                    nv[u] = ik_b2f(nr[cj + u * bd]);
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) orow[cj + u * bd] = ik_f2b(yv[u] * rr * nv[u]);
            }
            for (; cj < D; cj += bd) orow[cj] = ik_f2b(yr[cj] * rr * ik_b2f(nr[cj]));
        }
        return;
    }
    if ((int)threadIdx.x < rows) {
        float q = 0.f;
        for (int i = 0; i < nsl; ++i) q += part[((int64_t)s * Tc + r0 + threadIdx.x) * nsl + i];
        rsh[threadIdx.x] = 1.f / sqrtf(q / (float)D + gn_.eps);
    }
    __syncthreads();
    for (int r = 0; r < rows; ++r) {
        const float* yr = ybuf + ((int64_t)s * Tc + r0 + r) * D;
        ik_bf16* orow = gn_.out + ((int64_t)s * T + t0 + r0 + r) * D;
        const float rr = rsh[r];
        for (int cj = threadIdx.x; cj < D; cj += blockDim.x) orow[cj] = ik_f2b(yr[cj] * rr * ik_b2f(gn_.norm[(int64_t)s * D + cj]));
    }
}

// norm: out[s, t] = rA(y inv(y) norm[s]); the slice partials summed in slice order
__global__ void ink_blend_norm_kernel(const float* __restrict__ ybuf, const float* __restrict__ part, const ik_bf16* __restrict__ norm,
                                      ik_bf16* __restrict__ out, int Tc, int T, int t0, int D, int nsl, float eps) {
    __shared__ float r;
    const int tl = blockIdx.x, s = blockIdx.y;
    if (threadIdx.x == 0) {
        float q = 0.f;
        for (int i = 0; i < nsl; ++i) q += part[((int64_t)s * Tc + tl) * nsl + i];
        r = 1.f / sqrtf(q / (float)D + eps);
    }
    __syncthreads();
    const float* yr = ybuf + ((int64_t)s * Tc + tl) * D;
    ik_bf16* orow = out + ((int64_t)s * T + t0 + tl) * D;
    for (int c = threadIdx.x; c < D; c += blockDim.x) orow[c] = ik_f2b(yr[c] * r * ik_b2f(norm[(int64_t)s * D + c]));
}

// ---- the controller's h projection: h[s, r, g] = rA(pair[s, r] . down[s, g]) -----------------------------------------------
// One wave per (stream, controller row g); the wave's lanes stride the 2D k range in 16-byte steps (lane partial in k order, then
// the butterfly), for kHRows activation rows at once so the weight row streams once per group. Each output is the same chain of
// operations whatever the row count and its place in a group: rows-exact.
constexpr int kHRows = 8;
constexpr int kHWaves = 8;

__global__ void __launch_bounds__(kHWaves* IK_WARP) ink_ctl_gemv_kernel(const ik_bf16* __restrict__ pair, const ik_bf16* __restrict__ down,
                                                                         ik_bf16* __restrict__ h, int rc, int D2, int G) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int s = blockIdx.y;
    const int g = blockIdx.x * kHWaves + warp;
    if (g >= G) return;
    const int r0 = blockIdx.z * kHRows;
    const int rows = min(kHRows, rc - r0);
    const int64_t nv = D2 / 8;
    const uint4* w8 = reinterpret_cast<const uint4*>(down + ((int64_t)s * G + g) * D2);
    const uint4* x8 = reinterpret_cast<const uint4*>(pair + ((int64_t)s * rc + r0) * D2);
    float acc[kHRows];
#pragma unroll
    for (int m = 0; m < kHRows; ++m) acc[m] = 0.f;
#pragma unroll 2
    for (int64_t i = lane; i < nv; i += IK_WARP) {
        float wf[8];
        ik_unpack8(ik_ldcs(w8 + i), wf);
#pragma unroll
        for (int m = 0; m < kHRows; ++m) {
            if (m >= rows) break;
            float xf[8];
            ik_unpack8(ik_ldg(x8 + m * nv + i), xf);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[m] = fmaf(wf[e], xf[e], acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < kHRows; ++m) {
        if (m >= rows) break;
        const float v = ik_warp_sum(acc[m]);
        if (lane == 0) h[((int64_t)s * rc + r0 + m) * G + g] = ik_f2b(v);
    }
}

int64_t rnd(int64_t v) { return (v + 255) & ~(int64_t)255; }

struct BlendScratch {
    ik_bf16* pair;
    float* dbuf;
    ik_bf16* h;
    float* ybuf;
    float* part;
};

int64_t blend_bytes(int64_t s, int64_t r, int64_t d, int64_t g) {
    const int64_t rc = r < kBlendRows ? r : kBlendRows;
    const int64_t nsl = (d + kGateCh - 1) / kGateCh;
    return rnd(s * rc * 2 * d * 2) + rnd(s * rc * d * 4) + rnd(s * rc * g * 2) + rnd(s * rc * d * 4) + rnd(s * rc * nsl * 4);
}

BlendScratch blend_carve(void* base, int64_t s, int64_t rc, int64_t d, int64_t g) {
    char* p = (char*)base;
    const int64_t nsl = (d + kGateCh - 1) / kGateCh;
    BlendScratch b;
    b.pair = (ik_bf16*)p;
    p += rnd(s * rc * 2 * d * 2);
    b.dbuf = (float*)p;
    p += rnd(s * rc * d * 4);
    b.h = (ik_bf16*)p;
    p += rnd(s * rc * g * 2);
    b.ybuf = (float*)p;
    p += rnd(s * rc * d * 4);
    b.part = (float*)p;
    return b;
}

int row_threads(int D, int* ne) {
    int e = (D + kMaxThreads - 1) / kMaxThreads;
    if (e < 1) e = 1;
    *ne = e;
    const int th = (D + e - 1) / e;
    return (th + 31) / 32 * 32;
}

// Bit mask of the fast forms: 1 = ink_row_kernel<NE, true> (every load of the row up front, unconditional) for launches of at most kFastRowsMax rows
// (decode / verify: -2..-3.6 us of 13..15; above that its 160+ VGPRs cut occupancy and it loses up to 60%), 2 = the fused blend launches (pair +
// controller rows in the row kernel's launch, the norm in the gate kernel's last block) for chunks of at most kFuseRowsMax rows (-7 us at 1..2 rows),
// 4 = the gate kernel's early loads (no gain measured, off), 8 = the fast row kernel at any row count (tests). 0 = the first forms of all. The bits
// are the same for every mask (tests/ink_mix.rs). Default 3. INTARSIA_INK_MIX_FAST=<mask> or ik_ink_mix_mode(mask) overrides.
int g_mix_mask = -1;
int g_mix_dec = -1;  // tests: 0 | 1 the gate kernel's fast norm tail; < 0: INTARSIA_HIP_INK_DEC_FAST bit 2 / the default (on)
bool mix_dec_tail() {
    if (g_mix_dec >= 0) return g_mix_dec != 0;
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("INTARSIA_HIP_INK_DEC_FAST");
        v = e && e[0] >= '0' && e[0] <= '9' ? atoi(e) : 7;
    }
    return (v & 2) != 0;
}

int mix_mask() {
    if (g_mix_mask < 0) {
        const char* e = getenv("INTARSIA_INK_MIX_FAST");
        g_mix_mask = e && e[0] >= '0' && e[0] <= '9' ? atoi(e) : 3;
    }
    return g_mix_mask;
}

// `blocks` = the grid (rows, or rows * np for the fused blend); `rt`: the row-table form (conv_mix_bf16_rows)
int launch_rows(const RowArgs& a, int blocks, hipStream_t st, const RowRt* rt = nullptr) {
    if (blocks <= 0) return IK_OK;
    int ne = 0;
    const int th = row_threads(a.D, &ne);
    const int rows = a.fuse ? blocks / a.np : blocks;
    const bool fast = ((mix_mask() & 1) != 0 && rows <= kFastRowsMax) || (mix_mask() & 8) != 0;
    const bool dpp = (ik_hip_dec3_mask() & 1) != 0;
#ifdef IK_INK_MIX_ROWS_TU
    if (rt) {
        // the row-table form (conv_mix_bf16_rows): the same forms as the solo call, RT instantiations
        RowArgsRT x;
        static_cast<RowArgs&>(x) = a;
        x.rt = *rt;
#define IK_ROW_RT_CASE(N)                                                                   \
    case N:                                                                                 \
        if (fast && dpp) ink_row_kernel<N, true, true, true><<<blocks, th, 0, st>>>(x);     \
        else if (fast) ink_row_kernel<N, true, false, true><<<blocks, th, 0, st>>>(x);      \
        else if (dpp) ink_row_kernel<N, false, true, true><<<blocks, th, 0, st>>>(x);       \
        else ink_row_kernel<N, false, false, true><<<blocks, th, 0, st>>>(x);               \
        break;
        switch (ne) {
            IK_ROW_RT_CASE(1) IK_ROW_RT_CASE(2) IK_ROW_RT_CASE(3) IK_ROW_RT_CASE(4) IK_ROW_RT_CASE(5) IK_ROW_RT_CASE(6)
            default: return IK_ERR_SHAPE;
        }
#undef IK_ROW_RT_CASE
        return ik_launch_status();
    }
#else
    if (rt) return IK_ERR_UNSUPPORTED;  // the RT instantiations live in ink_mix_rows.hip's translation unit only
#endif
#define IK_ROW_CASE(N)                                                                \
    case N:                                                                           \
        if (fast && dpp) ink_row_kernel<N, true, true><<<blocks, th, 0, st>>>(a);     \
        else if (fast) ink_row_kernel<N, true, false><<<blocks, th, 0, st>>>(a);      \
        else if (dpp) ink_row_kernel<N, false, true><<<blocks, th, 0, st>>>(a);       \
        else ink_row_kernel<N, false, false><<<blocks, th, 0, st>>>(a);               \
        break;
    switch (ne) {
        IK_ROW_CASE(1) IK_ROW_CASE(2) IK_ROW_CASE(3) IK_ROW_CASE(4) IK_ROW_CASE(5) IK_ROW_CASE(6)
        default: return IK_ERR_SHAPE;
    }
#undef IK_ROW_CASE
    return ik_launch_status();
}

bool absent(const ik_tensor* t) { return !t || ik_numel(t) == 0; }

const ik_bf16* bp(const ik_tensor* t) { return absent(t) ? nullptr : (const ik_bf16*)t->data; }

int chk_opt(const ik_tensor* t, int dtype, int ndim) { return absent(t) ? IK_OK : ik_check(t, dtype, ndim); }

// the blend over the chunks of `a` (row kernel with msg set), then gemv / gate / norm into out [S,T,D]
int run_blend(RowArgs a, const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, const ik_tensor* norm, ik_tensor* out,
              const ik_scratch* sc, hipStream_t st, const RowRt* rt = nullptr) {
    const int S = a.S, T = a.T, D = a.D;
    const int G = (int)down->shape[1];
    if (S < 1 || S > kSMax || G < 1 || G > kGMax || G % 8 != 0 || (2 * D) % 16 != 0) return IK_ERR_SHAPE;
    if (!ik_aligned16(down->data) || !ik_aligned16(gain->data) || !ik_aligned16(forget->data)) return IK_ERR_INVALID_ARG;
    if (!sc || !sc->data || sc->bytes < blend_bytes(S, T, D, G)) return IK_ERR_INVALID_ARG;
    const int rcmax = T < kBlendRows ? T : kBlendRows;
    const BlendScratch b = blend_carve(sc->data, S, rcmax, D, G);
    const int nsl = (D + kGateCh - 1) / kGateCh;
    for (int t0 = 0; t0 < T; t0 += kBlendRows) {
        const int rc = T - t0 < kBlendRows ? T - t0 : kBlendRows;
        RowArgs r = a;
        r.t0 = t0;
        r.pair = b.pair;
        r.dbuf = b.dbuf;
        // decode / verify chunks: the pair and the controller rows in the row kernel's own launch (np blocks per row), the norm in the
        // gate kernel's last block: 2 launches instead of 4
        const int np = S * (G / 8);
        const bool fuse = (mix_mask() & 2) != 0 && rc <= kFuseRowsMax && D <= kFuseD;
        if (fuse) {
            r.fuse = 1;
            r.np = np;
            r.G = G;
            r.down = (const ik_bf16*)down->data;
            r.hbuf = b.h;
            IK_TRY(launch_rows(r, rc * np, st, rt));
        } else {
            IK_TRY(launch_rows(r, rc, st, rt));
            const dim3 hg((unsigned)((G + kHWaves - 1) / kHWaves), (unsigned)S, (unsigned)((rc + kHRows - 1) / kHRows));
            ink_ctl_gemv_kernel<<<hg, kHWaves * IK_WARP, 0, st>>>(b.pair, (const ik_bf16*)down->data, b.h, rc, 2 * D, G);
            IK_TRY(ik_launch_status());
        }
        const bool fnorm = (mix_mask() & 2) != 0 && rc <= kFuseRowsMax;
        const GateNorm gnorm{(const ik_bf16*)norm->data, (ik_bf16*)out->data, a.eps, fnorm ? 1 : 0, mix_dec_tail() ? 1 : 0};
        const dim3 gg((unsigned)nsl, (unsigned)S, (unsigned)((rc + kGateRows - 1) / kGateRows));
        const bool gdpp = (ik_hip_dec3_mask() & 1) != 0;
#define IK_GATE(F, P)                                                                                                                   \
    ink_blend_gate_kernel<F, P><<<gg, kGateCh, 0, st>>>(b.h, (const ik_bf16*)gain->data, (const ik_bf16*)forget->data, b.dbuf, a.msg, \
                                                        b.ybuf, b.part, rc, T, t0, D, G, gnorm)
        if (mix_mask() & 4) {
            if (gdpp) IK_GATE(true, true);
            else IK_GATE(true, false);
        } else {
            if (gdpp) IK_GATE(false, true);
            else IK_GATE(false, false);
        }
#undef IK_GATE
        IK_TRY(ik_launch_status());
        if (!fnorm) {
            ink_blend_norm_kernel<<<dim3((unsigned)rc, (unsigned)S), 256, 0, st>>>(b.ybuf, b.part, (const ik_bf16*)norm->data,
                                                                                    (ik_bf16*)out->data, rc, T, t0, D, nsl, a.eps);
            IK_TRY(ik_launch_status());
        }
    }
    return IK_OK;
}

int check_blend_w(const ik_tensor* msg, const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, const ik_tensor* norm,
                  int64_t T, int64_t D) {
    IK_TRY(ik_check(msg, IK_DTYPE_F32, 3));
    IK_TRY(ik_check(down, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(gain, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(forget, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(norm, IK_DTYPE_BF16, 2));
    const int64_t S = msg->shape[0], G = down->shape[1];
    if (msg->shape[1] != T || msg->shape[2] != D || down->shape[0] != S || down->shape[2] != 2 * D || gain->shape[0] != S ||
        gain->shape[1] != D || gain->shape[2] != G || forget->shape[0] != S || forget->shape[1] != D || forget->shape[2] != G ||
        norm->shape[0] != S || norm->shape[1] != D)
        return IK_ERR_SHAPE;
    return IK_OK;
}

// the conv part's checks and args (SconvRes / ConvMix)
int conv_args(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend, int64_t pos0, const ik_tensor* pos,
              ik_tensor* y, ik_tensor* ring, ik_tensor* sum, RowArgs* a) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(taps, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(y, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(ring, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(res, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(addend, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(sum, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check_pos(pos));
    const int64_t T = x->shape[0], D = x->shape[1], K = taps->shape[0], Rr = ring->shape[0];
    if (taps->shape[1] != D || y->shape[0] != T || y->shape[1] != D || ring->shape[1] != D || Rr < K || K < 1) return IK_ERR_SHAPE;
    if (D < 1 || D > kDMax || T > 0x7fffffff) return IK_ERR_SHAPE;
    *a = RowArgs{};
    a->x = (const ik_bf16*)x->data;
    a->taps = (const ik_bf16*)taps->data;
    a->res = bp(res);
    a->add = bp(addend);
    a->ring = (ik_bf16*)ring->data;
    a->y = (ik_bf16*)y->data;
    a->sum = absent(sum) ? nullptr : (ik_bf16*)sum->data;
    a->K = (int)K;
    a->Rr = (int)Rr;
    a->ring_now = T + K - 1 <= Rr;
    a->T = (int)T;
    a->D = (int)D;
    a->pos0 = pos0;
    a->pos = pos ? (const int*)pos->data : nullptr;
    return IK_OK;
}

// the row-table form's arguments (conv_mix_bf16_rows): per-row positions i32 [T], the table, the ring carrier's offset; every row is
// a T = 1 call, so its ring row is written in the row kernel (ring_after does nothing)
int rt_args(RowArgs* a, RowRt* rt, const ik_tensor* pos, const ik_tensor* tab, int64_t tstride, int64_t tindex, int64_t roff) {
    if (!pos || !pos->data || pos->dtype != IK_DTYPE_I32 || ik_numel(pos) != a->T) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_tab(tab, a->T, tstride, tindex));
    if (roff < 0 || (roff & 1)) return IK_ERR_INVALID_ARG;
    a->pos = (const int*)pos->data;
    a->ring_now = 1;
    rt->rtab = (const unsigned long long*)tab->data;
    rt->rts = (int)tstride;
    rt->rti = (int)tindex;
    rt->roff = roff;
    return IK_OK;
}

int ring_after(const RowArgs& a, hipStream_t st) {
    if (a.ring_now || a.T == 0) return IK_OK;
    const int rows = a.T < a.Rr ? a.T : a.Rr;
    ink_ring_kernel<<<dim3((unsigned)((a.D + 255) / 256), (unsigned)rows), 256, 0, st>>>(a.x, a.ring, a.T, a.D, a.Rr, a.pos0, a.pos);
    return ik_launch_status();
}

}  // namespace

// The launchers. kernels/hip/ink_mix_rows.hip includes this file with IK_INK_MIX_ROWS_TU defined and defines the row-table launcher
// (conv_mix_bf16_rows) in its own translation unit, so the RT instantiations of ink_row_kernel never share a module with the solo
// kernels (lane ink-hip-batch: in one module the solo ink_attn_dec_k compiled differently from main; here both files keep main's).
extern "C" {
#ifndef IK_INK_MIX_ROWS_TU

/* the phase timestamps of the last row-kernel launch (INK_PROBE builds; IK_ERR_UNSUPPORTED otherwise): n values into out */
int ik_ink_probe(unsigned long long* out, int n) {
#ifdef INK_PROBE
    if (hipDeviceSynchronize() != hipSuccess) return IK_ERR_LAUNCH;
    return hipMemcpyFromSymbol(out, HIP_SYMBOL(g_probe), (size_t)(n < 32 ? n : 32) * 8) == hipSuccess ? IK_OK : IK_ERR_LAUNCH;
#else
    (void)out;
    (void)n;
    return IK_ERR_UNSUPPORTED;
#endif
}

/* tests / benchmarks: bit mask of the fast forms (1 row kernel loads, 2 fused blend launches, 4 gate kernel loads; 0 = the first forms of all; same bits) */
void ik_ink_mix_mode(int v) { g_mix_mask = v; }
void ik_ink_mix_dec(int v) { g_mix_dec = v; }

int64_t ik_ink_blend_scratch_bytes(int64_t s, int64_t r, int64_t d, int64_t g) { return blend_bytes(s, r, d, g); }

int ik_sconv_res_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend, int64_t pos0,
                      const ik_tensor* pos, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, void* stream) {
    RowArgs a;
    IK_TRY(conv_args(x, taps, res, addend, pos0, pos, y, ring, sum, &a));
    if (a.T == 0) return IK_OK;
    hipStream_t st = IK_STREAM(stream);
    IK_TRY(launch_rows(a, a.T, st));
    return ring_after(a, st);
}

int ik_attn_res_mix_bf16(const ik_tensor* src, const ik_tensor* last, const ik_tensor* addend, const ik_tensor* queries,
                         const ik_tensor* norm, float eps, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(last, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(queries, IK_DTYPE_F32, 2));
    IK_TRY(chk_opt(addend, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(norm, IK_DTYPE_BF16, 2));
    const int64_t R = last->shape[0], D = last->shape[1], N = src->ndim == 3 ? src->shape[0] : -1, Q = queries->shape[0];
    if (N < 0 || N + 1 > kNsMax || Q < 1 || Q > kQMax || D < 1 || D > kDMax || queries->shape[1] != D) return IK_ERR_SHAPE;
    if (N > 0) {
        IK_TRY(ik_check(src, IK_DTYPE_BF16, 3));
        if (src->shape[1] != R || src->shape[2] != D) return IK_ERR_SHAPE;
    }
    IK_TRY(ik_check(out, absent(norm) ? IK_DTYPE_F32 : IK_DTYPE_BF16, 3));
    if (R == 0) return IK_OK;
    RowArgs a{};
    a.last = (const ik_bf16*)last->data;
    a.last_add = bp(addend);
    a.src = N > 0 ? (const ik_bf16*)src->data : nullptr;
    a.N = (int)N;
    a.q = (const float*)queries->data;
    a.Q = (int)Q;
    a.norm = bp(norm);
    if (a.norm) a.out_b = (ik_bf16*)out->data;
    else a.out_f = (float*)out->data;
    a.T = (int)R;
    a.D = (int)D;
    a.eps = eps;
    return launch_rows(a, (int)R, IK_STREAM(stream));
}

#endif  // IK_INK_MIX_ROWS_TU

// ConvMix, solo (tab null) or the row-table form (tab: the batched step's row table, pos i32 [T], roff the ring carrier's offset)
static int conv_mix_run(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend,
                        const ik_tensor* src, const ik_tensor* queries, const ik_tensor* norm, const ik_tensor* msg,
                        const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, int64_t pos0, const ik_tensor* pos,
                        float eps, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, ik_tensor* mix, const ik_scratch* scratch,
                        const ik_tensor* tab, int64_t tstride, int64_t tindex, int64_t roff, void* stream) {
    RowArgs a;
    IK_TRY(conv_args(x, taps, res, addend, pos0, tab ? nullptr : pos, y, ring, sum, &a));
    RowRt rt = {};
    if (tab) IK_TRY(rt_args(&a, &rt, pos, tab, tstride, tindex, roff));
    const RowRt* rtp = tab ? &rt : nullptr;
    a.eps = eps;
    const int64_t T = a.T, D = a.D;
    const int64_t Q = queries->ndim == 2 ? queries->shape[0] : -1;
    if (Q < 0 || Q > kQMax) return IK_ERR_SHAPE;
    hipStream_t st = IK_STREAM(stream);
    if (Q > 0) {
        IK_TRY(ik_check(queries, IK_DTYPE_F32, 2));
        const int64_t N = src->ndim == 3 ? src->shape[0] : -1;
        if (N < 0 || N + 1 > kNsMax || queries->shape[1] != D) return IK_ERR_SHAPE;
        if (N > 0) {
            IK_TRY(ik_check(src, IK_DTYPE_BF16, 3));
            if (src->shape[1] != T || src->shape[2] != D) return IK_ERR_SHAPE;
        }
        a.src = N > 0 ? (const ik_bf16*)src->data : nullptr;
        a.N = (int)N;
        a.q = (const float*)queries->data;
        a.Q = (int)Q;
    }
    if (T == 0) return IK_OK;
    if (Q > 0 && !absent(msg)) {
        IK_TRY(check_blend_w(msg, down, gain, forget, norm, T, D));
        IK_TRY(ik_check(mix, IK_DTYPE_BF16, 3));
        const int64_t S = msg->shape[0];
        if (!(Q == 1 || Q == S) || mix->shape[0] != S) return IK_ERR_SHAPE;
        a.msg = (const float*)msg->data;
        a.S = (int)S;
        // every chunk's row kernel reads the ring before the deferred write (in-kernel writes only when T + K - 1 <= Rr: one chunk)
        IK_TRY(run_blend(a, down, gain, forget, norm, mix, scratch, st, rtp));
        return ring_after(a, st);
    }
    if (Q > 0) {
        IK_TRY(chk_opt(norm, IK_DTYPE_BF16, 2));
        IK_TRY(ik_check(mix, absent(norm) ? IK_DTYPE_F32 : IK_DTYPE_BF16, 3));
        a.norm = bp(norm);
        if (a.norm) a.out_b = (ik_bf16*)mix->data;
        else a.out_f = (float*)mix->data;
    }
    IK_TRY(launch_rows(a, (int)T, st, rtp));
    return ring_after(a, st);
}

#ifndef IK_INK_MIX_ROWS_TU
int ik_conv_mix_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend,
                     const ik_tensor* src, const ik_tensor* queries, const ik_tensor* norm, const ik_tensor* msg,
                     const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, int64_t pos0, const ik_tensor* pos,
                     float eps, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, ik_tensor* mix, const ik_scratch* scratch,
                     void* stream) {
    return conv_mix_run(x, taps, res, addend, src, queries, norm, msg, down, gain, forget, pos0, pos, eps, y, ring, sum, mix, scratch,
                        nullptr, 0, 0, 0, stream);
}

int ik_loop_blend_bf16(const ik_tensor* dest, const ik_tensor* msg, const ik_tensor* down, const ik_tensor* gain,
                       const ik_tensor* forget, const ik_tensor* norm, float eps, ik_tensor* out, const ik_scratch* scratch,
                       void* stream) {
    IK_TRY(ik_check(dest, IK_DTYPE_F32, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t S = out->shape[0], R = out->shape[1], D = out->shape[2];
    if (D < 1 || D > kDMax || R > 0x7fffffff) return IK_ERR_SHAPE;
    IK_TRY(check_blend_w(msg, down, gain, forget, norm, R, D));
    if (msg->shape[0] != S || !(dest->shape[0] == 1 || dest->shape[0] == S) || dest->shape[1] != R || dest->shape[2] != D) return IK_ERR_SHAPE;
    if (R == 0) return IK_OK;
    RowArgs a{};
    a.dest = (const float*)dest->data;
    a.dest_bcast = dest->shape[0] == 1;
    a.msg = (const float*)msg->data;
    a.S = (int)S;
    a.Q = 1;  // (the row kernel's blend path; no mix is computed with a dest)
    a.T = (int)R;
    a.D = (int)D;
    a.eps = eps;
    return run_blend(a, down, gain, forget, norm, out, scratch, IK_STREAM(stream));
}
#endif  // IK_INK_MIX_ROWS_TU

}  // extern "C"
