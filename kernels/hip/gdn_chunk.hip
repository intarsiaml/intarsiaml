// Chunked GatedDeltaNet delta rule for prefill (lane gdn-chunk, docs/results/2026-11-14-hip-gdn-chunk.md): Op::GdnDeltaRule
// on 64-token chunks with gfx12 WMMA instead of gdn.hip's token-by-token scan. Design ported from radiance (libr4d, tag
// v2.0.0, commit ddcb9c6; kernels/hip/NOTICE-radiance): r4d_gdn_kkt_solve_k128_c64_bf16.hip (the per-chunk triangular
// inverse), r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip (the scan, the recurrent state in WMMA accumulators across the chunk
// loop) and r4d_gdn_wmma.h (the fragment helpers), rewritten for the ik_* ABI: one sequence per call, the in-place f32
// state [Hv, dv, dk], q / k / v read from the conv output's rows, and the gates and the q / k normalisation (which radiance's
// caller prepares) computed here by two small kernels into the launch's scratch. The normalisation is applied as per-token
// f32 factors folded into the scales the kernels apply anyway, so q and k enter the WMMAs as the exact bf16 the conv stored.
//
// Semantics: crates/intarsia-core/src/gdn.rs. Per chunk of BT = 64 tokens (the last one may be short), value head h and
// its key head, with G_i the within-chunk running sum of the log decay g_t = -exp(A_log) * softplus(a_t + dt_bias):
//
//   A  = (I + strict_lower(diag(beta) . K K^T . e^{G_i - G_j}))^-1                    (gdn_chunk_kkt_kernel)
//   U  = K @ S0          W = beta * (V - e^{G} * U)          V' = A @ W               (the chunk's deltas)
//   O  = scale * e^{G} * (Q @ S0) + (scale * e^{G_i - G_j} * tril(Q @ K^T)) @ V'
//   S  = e^{G_last} * S0 + V'^T @ (e^{G_last - G_j} * K)
//
// which is the recurrence of gdn.hip unrolled over the chunk (S0 = the state before the chunk; q, k l2-normalised,
// scale = 1 / sqrt(dk)). Every decay factor formed is e^{G_i - G_j} with j <= i, so it is in (0, 1]: the gate span inside a
// chunk reaches hundreds of nats on real activations, and a split about a per-chunk reference would overflow f32 (radiance's
// guard, kept).
//
// Numerics: NOT bitwise gdn_delta_rule_bf16_v2 / _v3. The WMMA operands are bf16, so the state as the chunk's matmuls
// read it, W, A, V' and the decayed k of the state update are rounded to bf16 (the state itself stays f32 in the accumulators and in memory),
// and the sums run in WMMA order. Shapes: dk = dv = 128 (the 35B / 27B GDN), Hv / Hk in {1, 2, 3, 4, 6, 8}, any T >= 1.
// Token-range pieces (ik_split_pieces) cut at chunk boundaries continue the in-place state exactly, so they give the single
// launch's bits.
//
// Scratch (ik_scratch, kernels.rs gdn_chunk_scratch): bf16 A [T, Hv, 64], f32 G [T, Hv], f32 beta [T, Hv], f32 lq [T, Hk],
// f32 lk [T, Hk], each rounded up to 16 bytes.
//
// Compiled -mcumode (build.rs CU_MODE_SOURCES), as radiance compiles the scan. No fp contraction, as libr4d is built.
#include "ik_hip.h"
#include <math.h>
#include <stdlib.h>

#pragma STDC FP_CONTRACT OFF

namespace {

IK_DEVI float gc_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
IK_DEVI float gc_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// ---- gfx1201 wave32 bf16 WMMA fragments (r4d_gdn_wmma.h) -----------------------------------------------------------
// Layout:  16-bit A/B : idx = lane % 16, k = 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3)   ("two runs of four")
//          f32   C/D  : n   = lane % 16, m = 8 * (lane >> 4) + e
// So A wants [M][K] and B wants [N][K] (both K-contiguous), and D stored to an [N][M] buffer is one contiguous 8-element
// store.
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

// f32 -> bf16, round half away (one add; differs from round-to-nearest-even only on exact ties)
IK_DEVI unsigned short f2bf(float f) { return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16); }
IK_DEVI unsigned f2bf2(float a, float b) {
    const unsigned ua = __builtin_bit_cast(unsigned, a) + 0x8000u;
    const unsigned ub = __builtin_bit_cast(unsigned, b) + 0x8000u;
    return __builtin_amdgcn_perm(ub, ua, 0x07060302u);  // {b_hi16, a_hi16}
}
IK_DEVI uint2 f2bf4(float a, float b, float c, float d) { return make_uint2(f2bf2(a, b), f2bf2(c, d)); }
IK_DEVI float bf2f(unsigned short h) { return __builtin_bit_cast(float, (unsigned)h << 16); }

// K-contiguous fragment from a row pointer already advanced to this lane's row and half-K: two 8-byte groups 16 bytes
// apart
IK_DEVI v8bf fragP(const unsigned short* p) {
    const uint2* pw = (const uint2*)p;
    union {
        uint2 w[2];
        v8bf f;
    } u;
    u.w[0] = pw[0];
    u.w[1] = pw[2];
    return u.f;
}
IK_DEVI const unsigned short* rowP(const unsigned short* src, int pitch, int base, int lane) {
    return src + (base + (lane & 15)) * pitch + 4 * (lane >> 4);
}
// 32-bit element offset of a global fragment off a uniform base; `lim` clamps the row so a short last chunk cannot read
// past the tensor (those rows are discarded downstream)
IK_DEVI unsigned rowGo(unsigned rowstride, int row0, int lim, int lane) {
    int r = row0 + (lane & 15);
    r = r < lim ? r : lim;
    return (unsigned)r * rowstride + 4u * (unsigned)(lane >> 4);
}
IK_DEVI v8bf fragGo(const unsigned short* base, unsigned off) { return fragP(base + off); }
// fragment from a 32-bit LDS byte offset off the shared block's base (an opaque register keeps the k steps immediates)
IK_DEVI v8bf fragO(const char* sb, unsigned off) { return fragP((const unsigned short*)(sb + off)); }
#define GC_LDSOPAQUE(x) asm("" : "+v"(x))
IK_DEVI v8bf frag(const unsigned short* src, int pitch, int base, int kbase, int lane) {
    return fragP(rowP(src, pitch, base, lane) + kbase);
}
// the same fragment of a source stored transposed, src[k][idx]: 8 scalar reads
IK_DEVI v8bf fragT(const unsigned short* src, int pitch, int base, int kbase, int lane) {
    const int idx = base + (lane & 15);
    const int k0 = kbase + 4 * (lane >> 4);
    union {
        unsigned short h[8];
        v8bf f;
    } u;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        u.h[e] = src[(size_t)(k0 + e) * pitch + idx];
        u.h[4 + e] = src[(size_t)(k0 + 8 + e) * pitch + idx];
    }
    return u.f;
}
IK_DEVI v8f mma(v8bf a, v8bf b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c); }

// LDS-only barrier: __syncthreads() also fences global memory, and would wait on the in-flight global loads / stores
IK_DEVI void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

#define BT 64   // chunk
#define KD 128  // dk
#define VD 128  // dv
#define BV 64   // value rows per workgroup of the scan
#define KP 132  // row pitch of the [.][KD] LDS buffers, in shorts
#define AP 68   // row pitch of the [.][BT] LDS buffers, in shorts
#define WP AP

// ---- gates: G (the within-chunk running sum of the log decay) and beta ---------------------------------------------
// grid (chunks, Hv), 64 threads: thread i owns token chunk * 64 + i.
__global__ void __launch_bounds__(BT) gdn_chunk_gate_kernel(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
                                                            const ik_bf16* __restrict__ a_log,
                                                            const ik_bf16* __restrict__ dt_bias, float* __restrict__ g,
                                                            float* __restrict__ beta, int64_t T, int Hv) {
    __shared__ float sg[BT];
    const int i = threadIdx.x, h = blockIdx.y;
    const int64_t t = (int64_t)blockIdx.x * BT + i;
    const bool live = t < T;
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    float gi = 0.f, bi = 0.f;
    if (live) {
        gi = neg_a * gc_softplus(ik_b2f(a[t * Hv + h]) + dtb);
        bi = gc_sigmoid(ik_b2f(b[t * Hv + h]));
    }
    sg[i] = gi;
    __syncthreads();
    float acc = 0.f;  // fixed order: deterministic
    for (int j = 0; j < BT; ++j) acc += j <= i ? sg[j] : 0.f;
    if (live) {
        g[t * Hv + h] = acc;
        beta[t * Hv + h] = bi;
    }
}

// ---- q / k l2-normalisation factors (gdn.rs: x * rsqrt(sum(x^2) + eps), and q / sqrt(dk)) -----------------------------------
// The conv output's q and k stay as they are (bf16, exact WMMA operands); only the per-token factors are computed, as
// logs: lq = log(rsqrt(|q|^2 + eps) / sqrt(dk)), lk = log(rsqrt(|k|^2 + eps)) (bounded by log(1 / sqrt(eps))). The kernels
// below fold them into the per-token f32 scales they apply anyway, so normalising costs no bf16 rounding.
// grid (T, Hk), one wave: lane l reads elements 4 l .. 4 l + 3 of the key head's q and k.
__global__ void __launch_bounds__(32) gdn_chunk_norm_kernel(const ik_bf16* __restrict__ qkv, float eps, float scale,
                                                            float* __restrict__ lq, float* __restrict__ lk, int C, int Hk) {
    const int lane = threadIdx.x, kh = blockIdx.y;
    const int64_t t = blockIdx.x;
    const unsigned short* row = (const unsigned short*)(qkv + t * C);
    const uint2 qr = *(const uint2*)(row + kh * KD + 4 * lane);
    const uint2 kr = *(const uint2*)(row + (Hk + kh) * KD + 4 * lane);
    float q[4], k[4];
    q[0] = bf2f((unsigned short)qr.x); q[1] = bf2f((unsigned short)(qr.x >> 16));
    q[2] = bf2f((unsigned short)qr.y); q[3] = bf2f((unsigned short)(qr.y >> 16));
    k[0] = bf2f((unsigned short)kr.x); k[1] = bf2f((unsigned short)(kr.x >> 16));
    k[2] = bf2f((unsigned short)kr.y); k[3] = bf2f((unsigned short)(kr.y >> 16));
    float qq = (q[0] * q[0] + q[1] * q[1]) + (q[2] * q[2] + q[3] * q[3]);
    float kk = (k[0] * k[0] + k[1] * k[1]) + (k[2] * k[2] + k[3] * k[3]);
    qq = ik_warp_sum(qq);
    kk = ik_warp_sum(kk);
    if (lane == 0) {
        lq[t * Hk + kh] = logf(rsqrtf(qq + eps) * scale);
        lk[t * Hk + kh] = logf(rsqrtf(kk + eps));
    }
}

// ---- A = (I + strict_lower(diag(beta) . K K^T . e^{G_i - G_j}))^-1 per chunk (r4d_gdn_kkt_solve_k128_c64_bf16) --------
// One workgroup owns a (chunk, key head): the gram K K^T does not depend on beta or G, so it is computed once (its
// upper-triangle tiles; read transposed into A's strictly lower half an accumulator holds eight consecutive columns of
// one row) and each value head's scales are applied to it. The inverse is blocked 16 -> 32 -> 64: the four diagonal blocks
// by f32 forward substitution (one wave each, one lane per column), the merges WMMA with
//     [[P, 0], [R, Q]]^-1 = [[P^-1, 0], [-Q^-1 R P^-1, Q^-1]]
// in place in A (each stage reads the block it overwrites before writing it, barriers between stages).
#define KK_NBK 4      // 16x16 blocks per chunk edge
#define KK_NW 4
#define KK_NTHR (KK_NW * 32)
#define KK_NTILE 10   // upper-triangular tiles of a 4x4 tiling
#define KK_MAXT ((KK_NTILE + KK_NW - 1) / KK_NW)

struct KkSmem {
    unsigned short A[BT][AP];    // A on the way in, its inverse on the way out
    unsigned short TMP[32][36];  // one 32x32 intermediate, transposed
    float gi[BT], gj[BT], bt[BT];  // per token: G + lk (as a row), G - lk (as a column), beta
    float lk[BT];                  // the key head's log k scale, per token
};

// (mt, nt) of upper-triangular tile ti
IK_DEVI void kkt_tile(int ti, int* mt, int* nt) {
    int m = 0, n = ti;
    while (n >= KK_NBK - m) {
        n -= (KK_NBK - m);
        ++m;
    }
    *mt = m;
    *nt = n + m;
}

// grid (chunks, Hk). k: the conv output's k columns (qkv + Hk * KD; row pitch C, key head hg at hg * KD), not normalised:
// the gram of the normalised k is P_raw[i][j] * e^{lk_i + lk_j}, folded into the decay exponent (so the exponent of a kept
// element is at most lk_i + lk_j <= log(1 / eps), no overflow).
template <int HG_RATIO>
__global__ void __launch_bounds__(KK_NTHR) gdn_chunk_kkt_kernel(const unsigned short* __restrict__ k,
                                                                const float* __restrict__ lkn,
                                                                const float* __restrict__ beta,
                                                                const float* __restrict__ g,
                                                                unsigned short* __restrict__ Ao, int T, int C, int H,
                                                                int Hg) {
    __shared__ KkSmem s;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int lo = lane & 15, hi = lane >> 4;
    const int hg = blockIdx.y;
    const int t0 = blockIdx.x * BT;
    const int rows = min(BT, T - t0);

    // zero A once: the parts no store reaches (the upper tiles) are the same for every head
    for (int i = tid; i < BT * AP / 8; i += KK_NTHR) ((uint4*)&s.A[0][0])[i] = make_uint4(0, 0, 0, 0);

    // the gram, once for all HG_RATIO value heads of this key head
    const unsigned short* kp = k + (size_t)t0 * C + hg * KD;
    const unsigned krow = (unsigned)C;
    if (tid < BT) s.lk[tid] = lkn[(size_t)(t0 + min(tid, rows - 1)) * Hg + hg];  // published by the head loop's first barrier
    v8f P[KK_MAXT];
#pragma unroll
    for (int t = 0; t < KK_MAXT; ++t) P[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < KK_MAXT; ++t) {
        const int ti = warp + t * KK_NW;
        if (ti < KK_NTILE) {
            int mt, nt;
            kkt_tile(ti, &mt, &nt);
            const unsigned om = rowGo(krow, mt * 16, rows - 1, lane);
            const unsigned on = rowGo(krow, nt * 16, rows - 1, lane);
#pragma unroll
            for (int sk = 0; sk < KD / 16; ++sk) P[t] = mma(fragGo(kp, om + sk * 16), fragGo(kp, on + sk * 16), P[t]);
        }
    }

    for (int hh = 0; hh < HG_RATIO; ++hh) {
        const int hv = hg * HG_RATIO + hh;

        if (tid < BT) {
            const int t = min(tid, rows - 1);
            const float gt = g[(size_t)(t0 + t) * H + hv];
            s.gi[tid] = gt + s.lk[tid];  // the thread's own store (same tid), no barrier needed
            s.gj[tid] = gt - s.lk[tid];
            s.bt[tid] = beta[(size_t)(t0 + t) * H + hv];
        }
        lds_barrier();

        // scale, mask and place the gram as the strictly lower half of A: tile (mt, nt) holds P[m][n], the element written
        // is A[n][m] (n fixed per lane, m over the eight accumulator elements: one row of A, one 16-byte store)
#pragma unroll
        for (int t = 0; t < KK_MAXT; ++t) {
            const int ti = warp + t * KK_NW;
            if (ti < KK_NTILE) {
                int mt, nt;
                kkt_tile(ti, &mt, &nt);
                const int i = nt * 16 + lo;       // row of A
                const int j0 = mt * 16 + 8 * hi;  // first column of A
                const bool live = (i < rows);
                const float bi = live ? s.bt[i] : 0.0f;
                const float gi = live ? s.gi[i] : 0.0f;
                float vv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = j0 + e;
                    // a live row has i > j, so G_i <= G_j and the exponent is at most lk_i + lk_j. Padding rows and columns
                    // are masked to -inf (with gi = 0 the exponent -G_j would be large and positive, 0 * inf = NaN,
                    // and the merges would spread it into live rows).
                    const float d = (live && i > j && j < rows) ? (gi - s.gj[j]) : -INFINITY;
                    vv[e] = bi * expf(d) * P[t][e];
                }
                *(uint2*)(&s.A[i][j0]) = f2bf4(vv[0], vv[1], vv[2], vv[3]);
                *(uint2*)(&s.A[i][j0 + 4]) = f2bf4(vv[4], vv[5], vv[6], vv[7]);
            }
        }
        lds_barrier();

        // 1. the four diagonal blocks: D = (I + N)^-1 solves D = I - N D row by row; lane l owns column l of D
        for (int p = warp; p < KK_NBK; p += KK_NW) {
            const int b0 = p * 16;
            float d[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j < i) acc += bf2f(s.A[b0 + i][b0 + j]) * d[j];
                d[i] = ((lo == i) ? 1.0f : 0.0f) - acc;
            }
            if (lane < 16) {
#pragma unroll
                for (int i = 0; i < 16; ++i) s.A[b0 + i][b0 + lo] = f2bf(d[i]);
            }
        }
        lds_barrier();

        // 2. the two 32-merges: M10 = -D1 A10 D0
        for (int blk = warp; blk < 2; blk += KK_NW) {
            const int r0 = blk * 32;
            // X = A10 . D0 (left A10 row-major, right D0 transposed)
            v8f X = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            X = mma(frag(&s.A[0][0], AP, r0 + 16, r0, lane), fragT(&s.A[0][0], AP, r0, r0, lane), X);
            // staged transposed for the next product; each of the two merges owns 16 rows of TMP
            *(uint2*)(&s.TMP[blk * 16 + lo][8 * hi]) = f2bf4(X[0], X[1], X[2], X[3]);
            *(uint2*)(&s.TMP[blk * 16 + lo][8 * hi + 4]) = f2bf4(X[4], X[5], X[6], X[7]);
            __builtin_amdgcn_wave_barrier();
            // M10 = -D1 . X (left D1 row-major, in place in A; right X transposed, in TMP)
            v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            Y = mma(frag(&s.A[0][0], AP, r0 + 16, r0 + 16, lane), frag(&s.TMP[0][0], 36, blk * 16, 0, lane), Y);
            const int i = r0 + 16 + 8 * hi;  // rows of M10
            const int j = r0 + lo;           // column of M10
#pragma unroll
            for (int e = 0; e < 8; ++e) s.A[i + e][j] = f2bf(-Y[e]);
        }
        lds_barrier();

        // 3. the 64-merge: M[32:, :32] = -Q^-1 (R P^-1). Y = R . P^-1, R = A[32:, :32] row-major, P^-1 read transposed
        for (int t = warp; t < 4; t += KK_NW) {
            const int mt = t >> 1, nt = t & 1;
            v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int sk = 0; sk < 2; ++sk)
                Y = mma(frag(&s.A[0][0], AP, 32 + mt * 16, sk * 16, lane), fragT(&s.A[0][0], AP, nt * 16, sk * 16, lane), Y);
            *(uint2*)(&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi]) = f2bf4(Y[0], Y[1], Y[2], Y[3]);
            *(uint2*)(&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi + 4]) = f2bf4(Y[4], Y[5], Y[6], Y[7]);
        }
        lds_barrier();  // every wave has read R into TMP; now it may be overwritten
        // Z = -Q^-1 . Y, Q^-1 = A[32:, 32:] row-major, Y transposed = TMP
        for (int t = warp; t < 4; t += KK_NW) {
            const int mt = t >> 1, nt = t & 1;
            v8f Z = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int sk = 0; sk < 2; ++sk)
                Z = mma(frag(&s.A[0][0], AP, 32 + mt * 16, 32 + sk * 16, lane), frag(&s.TMP[0][0], 36, nt * 16, sk * 16, lane),
                        Z);
            const int i = 32 + mt * 16 + 8 * hi;
            const int j = nt * 16 + lo;
#pragma unroll
            for (int e = 0; e < 8; ++e) s.A[i + e][j] = f2bf(-Z[e]);
        }
        lds_barrier();

        // store: one row of the inverse is 64 bf16, contiguous
        {
            unsigned short* op = Ao + ((size_t)t0 * H + hv) * BT;
            const size_t orow = (size_t)H * BT;
            for (int r = tid / 8; r < rows; r += KK_NTHR / 8) {
                const int c8 = (tid & 7) * 8;
                *(uint4*)(op + (size_t)r * orow + c8) = *(const uint4*)(&s.A[r][c8]);
            }
        }
        if (hh + 1 < HG_RATIO) lds_barrier();
    }
}

// ---- the scan (r4d_gdn_chunk_scan_k128_v128_c64_bf16) --------------------------------------------------------------
// grid (dv / BV, Hv), 8 waves. The [4 token][4 v] tile grid is fixed by BT and BV, so 8 waves carry two tiles each; a
// wave's slice of the 8 k-tile x 4 v-tile state grid is 2 x 2. The choices below (LDS rows as opaque byte offsets in the
// main loop, k / v staged eight shorts per lane and prefetched a chunk ahead, the sched-group patterns, rounding every
// bf16 pack) are radiance's, measured on gfx1201; the kernel is near the 256-VGPR limit.
#define SC_NW 8
#define SC_NTHR (SC_NW * 32)
#define SC_NTV 2  // n tiles per wave in the main / V' / O loops
#define SC_SKT 2
#define SC_SVT 2
#define SC_NST (SC_SKT * SC_SVT)  // state tiles per wave

struct ScSmem {
    unsigned short k[BT][KP];  // A: K  /  B: K^T for P = Q @ K^T
    unsigned short A[BT][AP];  // B-source for sA (written as [i][j])
    unsigned short S[BV][KP];  // B: state, V-major (the state's own [dv, dk] layout)
    unsigned short W[BV][WP];  // B: W (transposed)
    unsigned short D[BV][AP];  // staged V as [token][v]; then V' as [v][token]
    // per token, with the q / k normalisation folded in (lq, lk: gdn_chunk_norm_kernel):
    float go[BT];  // scale_q * e^G = e^{G + lq}: the Q @ S0 term of the output
    float gu[BT];  // norm_k * e^G = e^{G + lk}: U in W
    float gv[BT];  // norm_k * e^{G_last - G}: K for the state update
    float gi[BT];  // G + lq: the score exponent's row part
    float gj[BT];  // G - lk: the score exponent's column part
    float bt[BT];
};

#define SC_SB 2  // k steps per sched group in the V' and O += loops
#define SC_SS 1  // k steps per sched group in the state update
// LDS reads and WMMAs per main-loop k step, for the sched groups (must match the loop exactly)
#define SC_MLDS (1 + 2 * SC_NTV)
#define SC_MWMA (3 * SC_NTV)
// byte offset of buffer[base + lo][4 * hi] from the base of the shared block
#define SC_ROWO(buf, pitch, base) \
    ((unsigned)((const char*)&(buf)[0][0] - (const char*)&s) + (unsigned)((((base) + lo) * (pitch) + 4 * hi) * 2))

// qkv: the conv output rows [q (Hg * KD) | k (Hg * KD) | v (H * VD)], row pitch C, q and k not normalised; lqn, lkn: their
// log scales [T, Hg]; Amat: [T, H, BT]; gcs, beta: [T, H]; o: [T, H, VD]; state: [H, VD, KD] f32, read at the start and
// written at the end by the workgroup that owns the rows.
__global__ void __launch_bounds__(SC_NTHR) gdn_chunk_scan_kernel(
    const unsigned short* __restrict__ qkv, const float* __restrict__ lqn, const float* __restrict__ lkn,
    const unsigned short* __restrict__ Amat, const float* __restrict__ gcs, const float* __restrict__ beta,
    unsigned short* __restrict__ o, float* state, int T, int C, int H, int Hg) {
    const unsigned short* const q = qkv;
    const unsigned short* const k = qkv + Hg * KD;
    const unsigned short* const v = qkv + 2 * Hg * KD;
    __shared__ ScSmem s;
    const int bv = blockIdx.x, hv = blockIdx.y;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int hq = hv / (H / Hg), v0 = bv * BV;
    const int te = T;
    const size_t st_base = (size_t)hv * VD * KD;
    const int nchunk = (te + BT - 1) / BT;
    const int lo = lane & 15, hi = lane >> 4;

    const int mt = w >> 1, ntb = (w & 1) * 2;  // [BT][BV]: 2 tiles / wave
    // The state accumulator is held transposed, C[m = k-dim][n = v], so k-dim is the per-lane contiguous axis: the
    // per-chunk writeback to s.S is four 8-short stores.
    const int kt0 = (w >> 1) * 2, vt0 = (w & 1) * 2;  // 2 x 2

#define SC_PFW 8  // shorts staged per lane per step
#define SC_PFZERO make_uint4(0u, 0u, 0u, 0u)
#define SC_NKB (BT * KD / (SC_NTHR * SC_PFW))
#define SC_NVB (BT * BV / (SC_NTHR * SC_PFW))
    uint4 kb[SC_NKB], vb[SC_NVB];
    // Staging index: a wave-uniform row (readfirstlane makes it provably uniform) and a loop-invariant lane column.
    //   k: r = (tid >> 5) + NW * u            c = (tid * 4) & 127
    //   v: r = 2 * (tid >> 5) + vh + 2 NW u   c = (tid * 4) & 63   vh = (tid >> 4) & 1
    const int wu = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int kc = (tid * SC_PFW) & (KD - 1), vc = (tid * SC_PFW) & (BV - 1);
    const int kr0 = 2 * wu + ((tid >> 4) & 1), vr0 = 4 * wu + ((tid >> 3) & 3);
#define SC_KRU (2 * SC_NW)
#define SC_VRU (4 * SC_NW)
    // Staged rows past the chunk end are zeroed, not clamped to the last live row (radiance: a clamp returned one repeated
    // token for a trailing chunk that is not a multiple of 16 tokens; zeroing k fixes it).
#define SC_PFSTORE(dst, val)                                \
    do {                                                    \
        unsigned short* dp_ = (dst);                        \
        *(uint2*)dp_ = make_uint2((val).x, (val).y);        \
        *(uint2*)(dp_ + 4) = make_uint2((val).z, (val).w);  \
    } while (0)
#define SC_PREFETCH(CH)                                                                              \
    {                                                                                                \
        const int pc0 = (CH) * BT, pnv = min(BT, te - pc0);                                          \
        const unsigned short* kb_ = k + (size_t)pc0 * C + hq * KD;                                   \
        const unsigned short* vb_ = v + (size_t)pc0 * C + hv * VD + v0;                              \
        _Pragma("unroll") for (int u = 0; u < SC_NKB; ++u) {                                         \
            const int r = kr0 + SC_KRU * u;                                                          \
            kb[u] = r < pnv ? *(const uint4*)(kb_ + (unsigned)(r * C) + kc) : SC_PFZERO;             \
        }                                                                                            \
        _Pragma("unroll") for (int u = 0; u < SC_NVB; ++u) {                                         \
            const int r = vr0 + SC_VRU * u;                                                          \
            vb[u] = r < pnv ? *(const uint4*)(vb_ + (unsigned)(r * C) + vc) : SC_PFZERO;             \
        }                                                                                            \
    }

    v8f St[SC_NST];  // state: SKT k-tiles x SVT v-tiles
#define SC_KT(t) (kt0 + (t) / SC_SVT)
#define SC_VT(t) (vt0 + (t) % SC_SVT)
#define SC_SROW(t) ((size_t)(v0 + SC_VT(t) * 16 + lo) * KD + SC_KT(t) * 16 + 8 * hi)
    {
        const float* p = state + st_base;
#pragma unroll
        for (int t = 0; t < SC_NST; ++t) {
            const float4 a = *(const float4*)(p + SC_SROW(t));
            const float4 b = *(const float4*)(p + SC_SROW(t) + 4);
            St[t][0] = a.x; St[t][1] = a.y; St[t][2] = a.z; St[t][3] = a.w;
            St[t][4] = b.x; St[t][5] = b.y; St[t][6] = b.z; St[t][7] = b.w;
        }
    }
#define SC_SWRITE()                                                                       \
    _Pragma("unroll") for (int t = 0; t < SC_NST; ++t) {                                  \
        unsigned short* sp = &s.S[SC_VT(t) * 16 + lo][SC_KT(t) * 16 + 8 * hi];            \
        *(uint2*)sp = f2bf4(St[t][0], St[t][1], St[t][2], St[t][3]);                      \
        *(uint2*)(sp + 4) = f2bf4(St[t][4], St[t][5], St[t][6], St[t][7]);                \
    }
    SC_SWRITE();
    lds_barrier();
    SC_PREFETCH(0);

    const char* const sbase = (const char*)&s;
    for (int ci = 0; ci < nchunk; ++ci) {
        const int c0 = ci * BT, nval = min(BT, te - c0);
        const int lim = nval - 1;
        const unsigned short* qg = q + (size_t)c0 * C + hq * KD;
        const unsigned short* ag = Amat + ((size_t)c0 * H + hv) * BT;
        const size_t qrs = (size_t)C, ars = (size_t)H * BT;
        const float gl = gcs[(size_t)(c0 + lim) * H + hv];

        // issued before the staging store so the LDS writes cover their latency; only waves 0-1 (tid < BT) run this
        float ggr = 0.f, btr = 0.f, lqr = 0.f, lkr = 0.f;
        if (tid < BT) {
            const bool ok = tid < nval;
            ggr = ok ? gcs[(size_t)(c0 + tid) * H + hv] : 0.f;
            btr = ok ? beta[(size_t)(c0 + tid) * H + hv] : 0.f;
            lqr = ok ? lqn[(size_t)(c0 + tid) * Hg + hq] : 0.f;
            lkr = ok ? lkn[(size_t)(c0 + tid) * Hg + hq] : 0.f;
        }
        {  // consume the previous iteration's prefetch, issued a whole chunk of compute ago
#pragma unroll
            for (int u = 0; u < SC_NKB; ++u) SC_PFSTORE(&s.k[kr0 + SC_KRU * u][kc], kb[u]);
#pragma unroll
            for (int u = 0; u < SC_NVB; ++u) SC_PFSTORE(&s.D[vr0 + SC_VRU * u][vc], vb[u]);
        }
        if (tid < BT) {
            const bool ok = tid < nval;
            s.bt[tid] = btr;
            s.go[tid] = expf(ggr + lqr);
            s.gu[tid] = expf(ggr + lkr);
            // G is a running sum of non-positive gates, so it is non-increasing and gl is its minimum: gl - ggr <= 0, and
            // the state factor underflowing to 0 is correct. A padding row gets gl, not 0: it is scored against live j
            // with very negative G_j, and G_i = 0 would give exp(+400) = inf in s.A. The row is discarded anyway (the O
            // store is guarded). lq and lk are bounded (<= log(1 / sqrt(eps))), so no exponent here can overflow.
            s.gi[tid] = (ok ? ggr : gl) + lqr;
            s.gj[tid] = (ok ? ggr : gl) - lkr;
            s.gv[tid] = ok ? expf(gl - ggr + lkr) : 0.f;
        }
        lds_barrier();                               // (1)
        SC_PREFETCH(min(ci + 1, nchunk - 1));        // issue only; consumed next iteration

        // ---- U = K @ S, O^T = S @ Q^T and P^T = K @ Q^T share the q and k fragments. Tiles above the diagonal are
        // computed anyway (skipping them is slower: the barrier arrival does not move).
        v8f O[SC_NTV], u[SC_NTV], p[SC_NTV];
#pragma unroll
        for (int i = 0; i < SC_NTV; ++i) {
            O[i] = (v8f){};
            u[i] = (v8f){};
            p[i] = (v8f){};
        }
        {
            const char* sb = sbase;
            unsigned oSn[SC_NTV], okn[SC_NTV];
            unsigned okm = SC_ROWO(s.k, KP, mt * 16);
            GC_LDSOPAQUE(okm);
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i) {
                oSn[i] = SC_ROWO(s.S, KP, (ntb + i) * 16);
                okn[i] = SC_ROWO(s.k, KP, (ntb + i) * 16);
                GC_LDSOPAQUE(oSn[i]);
                GC_LDSOPAQUE(okn[i]);
            }
#define pSnf(i, ko) fragO(sb, oSn[i] + (ko) * 2u)
#define pkmf(ko) fragO(sb, okm + (ko) * 2u)
#define pknf(i, ko) fragO(sb, okn[i] + (ko) * 2u)
            const unsigned qo = rowGo((unsigned)qrs, mt * 16, lim, lane);
#define pqf(ko) fragGo(qg, qo + (ko))
#pragma unroll
            for (int sk = 0; sk < KD / 16; ++sk) {
                const int ko = sk * 16;
                v8bf ak = pkmf(ko), aq = pqf(ko);
#pragma unroll
                for (int i = 0; i < SC_NTV; ++i) {
                    v8bf b = pSnf(i, ko);
                    u[i] = mma(ak, b, u[i]);
                    O[i] = mma(b, aq, O[i]);            // O^T: m = v-tile, n = token-tile
                    p[i] = mma(pknf(i, ko), aq, p[i]);  // P^T: m = j
                }
            }
            // Issue one k step's LDS reads against the previous step's WMMAs (per step: 5 LDS fragments, 1 global q
            // fragment = 2 loads, 6 WMMA); otherwise every WMMA waits on a full LDS drain.
#define SC_GRAN 2
#pragma unroll
            for (int i = 0; i < KD / (16 * SC_GRAN); ++i) {
                __builtin_amdgcn_sched_group_barrier(0x020, 2 * SC_GRAN, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, SC_MLDS * SC_GRAN, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, SC_MWMA * SC_GRAN, 0);
            }
        }
        {
            const float g = s.go[mt * 16 + lo];  // the token index is constant per lane
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e) O[i][e] *= g;
        }
        // W = beta * (V - ge * U)   (beta folded here: (A * beta_col) @ W == A @ (beta_row * W))
        {
            float wa[SC_NTV][8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {  // beta / ge read once for both tiles
                const int tok = mt * 16 + 8 * hi + e;
                const float f = s.bt[tok], g = s.gu[tok];
#pragma unroll
                for (int i = 0; i < SC_NTV; ++i) wa[i][e] = f * (bf2f(s.D[tok][(ntb + i) * 16 + lo]) - g * u[i][e]);
            }
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i) {
                unsigned short* wp = &s.W[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
                *(uint2*)wp = f2bf4(wa[i][0], wa[i][1], wa[i][2], wa[i][3]);
                *(uint2*)(wp + 4) = f2bf4(wa[i][4], wa[i][5], wa[i][6], wa[i][7]);
            }
        }
        // sA = scale_q_i * norm_k_j * e^{G_i - G_j} * tril(P). Where kept, G_i - G_j <= 0, so the exponent is at most
        // lq_i + lk_j and the exp cannot overflow.
        {
            const int ii = mt * 16 + lo;
            const float gis = s.gi[ii];
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i) {
                const int j0 = (ntb + i) * 16 + 8 * hi;
                float a[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) a[e] = (j0 + e <= ii) ? p[i][e] * expf(gis - s.gj[j0 + e]) : 0.f;
                *(uint2*)&s.A[ii][j0] = f2bf4(a[0], a[1], a[2], a[3]);
                *(uint2*)&s.A[ii][j0 + 4] = f2bf4(a[4], a[5], a[6], a[7]);
            }
        }
        lds_barrier();  // (2)

        // ---- V' = A @ W -> D (the transposed store is one contiguous pair of dwords)
        {
            v8f d[SC_NTV];
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i) d[i] = (v8f){};
            const unsigned ao = rowGo((unsigned)ars, mt * 16, lim, lane);
#define paf(ko) fragGo(ag, ao + (ko))
            const unsigned short* pwn[SC_NTV];
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i) pwn[i] = rowP(&s.W[0][0], WP, (ntb + i) * 16, lane);
#define pwnf(i, ko) fragP(pwn[i] + (ko))
#pragma unroll
            for (int sk = 0; sk < BT / 16; ++sk) {
                const int ko = sk * 16;
                v8bf a = paf(ko);
#pragma unroll
                for (int i = 0; i < SC_NTV; ++i) d[i] = mma(a, pwnf(i, ko), d[i]);
            }
#pragma unroll
            for (int i = 0; i < BT / (16 * SC_SB); ++i) {
                __builtin_amdgcn_sched_group_barrier(0x020, 2 * SC_SB, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, SC_NTV * SC_SB, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, SC_NTV * SC_SB, 0);
            }
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i) {
                unsigned short* dp = &s.D[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
                *(uint2*)dp = f2bf4(d[i][0], d[i][1], d[i][2], d[i][3]);
                *(uint2*)(dp + 4) = f2bf4(d[i][4], d[i][5], d[i][6], d[i][7]);
            }
        }
        // K <- (norm_k * gs) * K in place, for the state's V'^T @ K: (diag(gs) V')^T K == V'^T (diag(gs) K), and K is otherwise dead
        // (all raw-K reads retired before barrier (2); s.k is restaged next chunk; barrier (3) publishes the scaled K).
#pragma unroll
        for (int u = 0; u < BT * (KD / 4) / SC_NTHR; ++u) {
            const int idx = tid + u * SC_NTHR;
            const int t = idx / (KD / 4), c = (idx % (KD / 4)) * 4;
            const float f = s.gv[t];
            unsigned short* kp = &s.k[t][c];
            *(uint2*)kp = f2bf4(bf2f(kp[0]) * f, bf2f(kp[1]) * f, bf2f(kp[2]) * f, bf2f(kp[3]) * f);
        }
        lds_barrier();  // (3)

        // ---- O += sA @ V'; store. V' also feeds the state update, unchanged.
        {
            const unsigned short* pA = rowP(&s.A[0][0], AP, mt * 16, lane);
            const unsigned short* pdn[SC_NTV];
#pragma unroll
            for (int i = 0; i < SC_NTV; ++i) pdn[i] = rowP(&s.D[0][0], AP, (ntb + i) * 16, lane);
#define pAf(ko) fragP(pA + (ko))
#define pdnf(i, ko) fragP(pdn[i] + (ko))
#pragma unroll
            for (int sk = 0; sk < BT / 16; ++sk) {
                const int ko = sk * 16;
                v8bf b = pAf(ko);
#pragma unroll
                for (int i = 0; i < SC_NTV; ++i) O[i] = mma(pdnf(i, ko), b, O[i]);
            }
#pragma unroll
            for (int i = 0; i < BT / (16 * SC_SB); ++i) {
                __builtin_amdgcn_sched_group_barrier(0x100, (1 + SC_NTV) * SC_SB, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, SC_NTV * SC_SB, 0);
            }
        }
        {
            const int tok = mt * 16 + lo;
            if (tok < nval) {
                unsigned short* op = o + ((size_t)(c0 + tok) * H + hv) * VD + v0;
#pragma unroll
                for (int i = 0; i < SC_NTV; ++i) {
                    unsigned short* q2 = op + (ntb + i) * 16 + 8 * hi;
                    *(uint2*)q2 = f2bf4(O[i][0], O[i][1], O[i][2], O[i][3]);
                    *(uint2*)(q2 + 4) = f2bf4(O[i][4], O[i][5], O[i][6], O[i][7]);
                }
            }
        }

        // ---- S = e^{gl} * S + V'^T @ (gs * K)   (B needs K as [dim][token]: the strided fragT)
        {
            const float ec = expf(gl);  // gl <= 0; the per-token decay is already on K
#pragma unroll
            for (int t = 0; t < SC_NST; ++t)
#pragma unroll
                for (int e = 0; e < 8; ++e) St[t][e] *= ec;
            const unsigned short* pvn[SC_SVT];
#pragma unroll
            for (int j = 0; j < SC_SVT; ++j) pvn[j] = rowP(&s.D[0][0], AP, (vt0 + j) * 16, lane);
#define pvnf(j, ko) fragP(pvn[j] + (ko))
#pragma unroll
            for (int sk = 0; sk < BT / 16; ++sk) {
                v8bf bvf[SC_SVT];
#pragma unroll
                for (int j = 0; j < SC_SVT; ++j) bvf[j] = pvnf(j, sk * 16);
#pragma unroll
                for (int ki = 0; ki < SC_SKT; ++ki) {
                    v8bf ak = fragT(&s.k[0][0], KP, (kt0 + ki) * 16, sk * 16, lane);  // K^T: m = k-dim
#pragma unroll
                    for (int j = 0; j < SC_SVT; ++j) St[ki * SC_SVT + j] = mma(ak, bvf[j], St[ki * SC_SVT + j]);
                }
            }
#pragma unroll
            for (int i = 0; i < BT / (16 * SC_SS); ++i) {
                __builtin_amdgcn_sched_group_barrier(0x100, (8 * SC_SKT + SC_SVT) * SC_SS, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, SC_NST * SC_SS, 0);
            }
        }
        // no barrier needed before the writeback: barrier (2) proves every wave finished its U / O reads of s.S
        SC_SWRITE();
        lds_barrier();  // (4)
    }

    float* p = state + st_base;
#pragma unroll
    for (int t = 0; t < SC_NST; ++t) {
        *(float4*)(p + SC_SROW(t)) = make_float4(St[t][0], St[t][1], St[t][2], St[t][3]);
        *(float4*)(p + SC_SROW(t) + 4) = make_float4(St[t][4], St[t][5], St[t][6], St[t][7]);
    }
}

int64_t gc_r16(int64_t n) { return (n + 15) & ~(int64_t)15; }

}  // namespace

extern "C" {

// Op::GdnDeltaRule, chunked (see the head of this file). Arguments as ik_gdn_delta_rule_bf16_v2, plus the scratch.
int ik_gdn_delta_rule_bf16_chunk(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                                 const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state,
                                 const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check(qkv, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a_log, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(dt_bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(o, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(state, IK_DTYPE_F32, 3));
    const int64_t T = qkv->shape[0], C = qkv->shape[1];
    const int64_t Hv = o->shape[1], dv = o->shape[2], dk = state->shape[2];
    if (dk == 0 || dv == 0 || Hv == 0 || C <= Hv * dv || (C - Hv * dv) % (2 * dk) != 0) return IK_ERR_SHAPE;
    const int64_t Hk = (C - Hv * dv) / (2 * dk);
    if (Hv % Hk != 0 || o->shape[0] != T || state->shape[0] != Hv || state->shape[1] != dv ||
        a->shape[0] != T || a->shape[1] != Hv || b->shape[0] != T || b->shape[1] != Hv || a_log->shape[0] != Hv ||
        dt_bias->shape[0] != Hv)
        return IK_ERR_SHAPE;
    if (dk != KD || dv != VD) return IK_ERR_UNSUPPORTED;
    const int64_t ratio = Hv / Hk;
    if (!(ratio == 1 || ratio == 2 || ratio == 3 || ratio == 4 || ratio == 6 || ratio == 8)) return IK_ERR_UNSUPPORTED;
    // the kernels index tokens and row offsets in 32 bits
    if (T > (int64_t)1 << 20 || C > (int64_t)1 << 20 || Hv > 4096) return IK_ERR_UNSUPPORTED;
    if (T == 0) return IK_OK;

    const int64_t a_b = gc_r16(T * Hv * BT * 2), g_b = gc_r16(T * Hv * 4), l_b = gc_r16(T * Hk * 4);
    const int64_t need = a_b + 2 * g_b + 2 * l_b;
    if (!scratch || !scratch->data || scratch->bytes < need || !ik_aligned16(scratch->data)) return IK_ERR_INVALID_ARG;
    char* base = (char*)scratch->data;
    unsigned short* Am = (unsigned short*)base;
    float* g = (float*)(base + a_b);
    float* beta = (float*)(base + a_b + g_b);
    float* lq = (float*)(base + a_b + 2 * g_b);
    float* lk = (float*)(base + a_b + 2 * g_b + l_b);
    const unsigned short* qkvp = (const unsigned short*)qkv->data;
    const float scale = 1.0f / sqrtf((float)KD);

    hipStream_t s = IK_STREAM(stream);
    const int64_t nchunk = (T + BT - 1) / BT;
    ik_rt_stamp(s, "gdn_chunk", -1);  // INTARSIA_HIP_STAMP only; otherwise launches nothing
    gdn_chunk_gate_kernel<<<dim3((unsigned)nchunk, (unsigned)Hv), BT, 0, s>>>(
        (const ik_bf16*)a->data, (const ik_bf16*)b->data, (const ik_bf16*)a_log->data, (const ik_bf16*)dt_bias->data, g,
        beta, T, (int)Hv);
    IK_TRY(ik_launch_status());
    gdn_chunk_norm_kernel<<<dim3((unsigned)T, (unsigned)Hk), 32, 0, s>>>((const ik_bf16*)qkv->data, eps, scale, lq, lk,
                                                                         (int)C, (int)Hk);
    IK_TRY(ik_launch_status());
    const dim3 kgrid((unsigned)nchunk, (unsigned)Hk);
#define IK_GC_KKT(R) \
    gdn_chunk_kkt_kernel<R><<<kgrid, KK_NTHR, 0, s>>>(qkvp + Hk * KD, lk, beta, g, Am, (int)T, (int)C, (int)Hv, (int)Hk)
    switch (ratio) {
        case 1: IK_GC_KKT(1); break;
        case 2: IK_GC_KKT(2); break;
        case 3: IK_GC_KKT(3); break;
        case 4: IK_GC_KKT(4); break;
        case 6: IK_GC_KKT(6); break;
        default: IK_GC_KKT(8); break;
    }
#undef IK_GC_KKT
    IK_TRY(ik_launch_status());
    ik_rt_stamp(s, "gdn_chunk", 0);

    // The scan as token-range pieces cut at chunk boundaries (ik_split_pieces: a dispatch over ~1 ms stalls the queue): each
    // piece continues the in-place state the previous one stored, and a chunk sees the same tokens either way, so the
    // pieces give the single launch's bits. Estimate: kScanUsPerChunk per chunk at the 35B's 64 workgroups.
    constexpr double kScanUsPerChunk = 14.0;
    const dim3 sgrid((unsigned)(VD / BV), (unsigned)Hv);
    const double wg = (double)sgrid.x * (double)sgrid.y;
    const double est_us = (double)nchunk * kScanUsPerChunk * (wg > 64.0 ? wg / 64.0 : 1.0);
    const int pieces = ik_split_pieces(est_us, nchunk);
    for (int i = 0; i < pieces; ++i) {
        const int64_t t0 = (nchunk * i / pieces) * BT;
        const int64_t t1 = i + 1 == pieces ? T : (nchunk * (i + 1) / pieces) * BT;
        gdn_chunk_scan_kernel<<<sgrid, SC_NTHR, 0, s>>>(qkvp + t0 * C, lq + t0 * Hk, lk + t0 * Hk, Am + t0 * Hv * BT,
                                                        g + t0 * Hv, beta + t0 * Hv,
                                                        (unsigned short*)o->data + t0 * Hv * VD, (float*)state->data,
                                                        (int)(t1 - t0), (int)C, (int)Hv, (int)Hk);
        ik_rt_stamp(s, "gdn_chunk", i + 1);
    }
    return ik_launch_status();
}

}  // extern "C"
