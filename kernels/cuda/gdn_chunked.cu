// Chunked (WY form) Gated DeltaNet prefill: the same op as gdn_delta_rule (kernels/cuda/gdn.cu, semantics
// crates/intarsia-core/src/gdn.rs), restructured so that the token-sequential recurrence becomes a short chain of
// 64-token chunk steps made of small dense products. All arithmetic is fp32 (SIMT, no tensor cores): the result matches
// the sequential kernels to fp32 rounding (a different, fixed summation order), not to tensor-core precision.
//
// Per value head, chunk of C = 64 tokens r = 0..63 (q^, k^ the l2-normalised, scaled vectors; alpha_r the decay, beta_r the
// write strength, G_r = sum_{s<=r} log alpha_s, e_r = exp(G_r)), state S0 [dk][dv] at the start of the chunk:
//     delta_r  = beta_r (v_r - e_r k^_r.S0 - sum_{p<r} (e_r/e_p)(k^_r.k^_p) delta_p)        (the sequential update, unrolled)
//     =>  (I + L) delta = Y,   L[r][p] = beta_r (e_r/e_p)(k^_r.k^_p) (p < r),   Y = beta (v - e (K^ S0))
//     T = (I + L)^-1 (unit lower triangular; forward substitution),   delta = T Y
//     o_r      = e_r q^_r.S0 + sum_{p<=r} (e_r/e_p)(q^_r.k^_p) delta_p              (M[r][p] = (e_r/e_p)(q^_r.k^_p))
//     S_next   = e_63 S0 + sum_r (e_63/e_r) k^_r delta_r^T
// Kernel 1 (gdn_chunk_prep_kernel, grid chunks x Hv, fully parallel) computes everything that does not depend on the state:
// the two 64x64 Gram products, T and M (stored transposed) and the per-token scales, into the plan scratch.
// Kernel 2 (gdn_chunk_state_kernel, grid dv/32 x Hv) walks the chunks in order for 32 value columns of one head: the state
// slice lives in shared memory and each chunk is five small products (K^ S0 and Q^ S0 together, T Y, M delta, the state
// update). Tokens past the end of the call are padded with k = q = 0 and beta = 0, which leaves the recurrence unchanged.
// Deterministic: fixed summation order, no atomics.
#include "ik_launch.h"
#include <stdlib.h>

namespace {

constexpr int kC = 64;       // tokens per chunk
constexpr int kCols = 32;    // value columns per state block
constexpr int kKT = 32;      // key-dimension tile
constexpr int kSStride = 36; // smem row stride of the state slice [dk][kCols] (16-byte aligned rows)
constexpr int kMat = kC * kC;
// scratch floats per (head, chunk): Tt[64][64] | Mt[64][64] | beta[64] | e[64] | rk[64] | rqe[64] | kd[64]
constexpr int kScrFloats = 2 * kMat + 5 * kC;

IK_DEVI float gc_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
IK_DEVI float gc_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

IK_DEVI void unpack8(const uint4& u, float* f) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// ---- kernel 1: Gram products, T = (I + L)^-1, M, per-token scales -----------------------------------------------------
template <int DK>
__global__ void __launch_bounds__(256) gdn_chunk_prep_kernel(
    const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
    const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps, float* __restrict__ scr,
    int64_t T, int Hk, int Hv, int dv, int nchunks) {
    extern __shared__ __align__(16) float sm[];
    float* sA = sm;               // [64][64]  Gram of k^ -> L
    float* sB = sA + kMat;        // [64][64]  Gram of q^ k^ -> M
    float* sU = sB + kMat;        // [64][64]  tiles Kt[32][64] | Qt[32][64] during the Gram loop, then T
    float* sBG = sU + kMat;       // beta[64] | G[64] | rk[64] | rq[64]
    float* sTot = sBG + 4 * kC;   // one float
    float* sKt = sU;
    float* sQt = sU + kKT * kC;
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const int64_t t0 = (int64_t)c * kC;
    const int n = (int)min((int64_t)kC, T - t0);
    const float scale = rsqrtf((float)DK);

    // l2 norms of the tokens of this chunk (rq carries the 1/sqrt(dk) scale); padded tokens get 0
    for (int r = warp; r < kC; r += 8) {
        float qq = 0.f, kk = 0.f;
        if (r < n) {
            const ik_bf16* row = qkv + (t0 + r) * C;
            for (int i = lane; i < DK; i += 32) {
                const float qv = ik_b2f(row[kh * DK + i]), kv = ik_b2f(row[Hk * DK + kh * DK + i]);
                qq += qv * qv;
                kk += kv * kv;
            }
        }
        qq = ik_warp_sum(qq);
        kk = ik_warp_sum(kk);
        if (lane == 0) {
            sBG[2 * kC + r] = r < n ? rsqrtf(kk + eps) : 0.f;
            sBG[3 * kC + r] = r < n ? rsqrtf(qq + eps) * scale : 0.f;
        }
    }
    // beta and the cumulative log decay G (inclusive scan over the 64 tokens); padded tokens: beta = 0, g = 0
    float g = 0.f, bt = 0.f;
    if (tid < kC && tid < n) {
        const float neg_a = -expf(ik_b2f(a_log[h])), dtb = ik_b2f(dt_bias[h]);
        g = neg_a * gc_softplus(ik_b2f(a[(t0 + tid) * Hv + h]) + dtb);
        bt = gc_sigmoid(ik_b2f(b[(t0 + tid) * Hv + h]));
    }
    float x = g;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (tid == 31) *sTot = x;
    __syncthreads();
    if (tid < kC) {
        sBG[tid] = bt;
        sBG[kC + tid] = x + (tid >= 32 ? *sTot : 0.f);
    }
    __syncthreads();

    // Gram products: thread tile 4 x 4 of (k^ k^^T) and (q^ k^^T)
    const int ty = tid >> 4, tx = tid & 15;
    float gk[4][4], gq[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) gk[i][j] = gq[i][j] = 0.f;
    for (int kt = 0; kt < DK / kKT; ++kt) {
        {
            const int r = tid & 63, part = tid >> 6;
            float kf[8], qf[8];
            if (r < n) {
                const ik_bf16* row = qkv + (t0 + r) * C;
                unpack8(*reinterpret_cast<const uint4*>(row + Hk * DK + kh * DK + kt * kKT + part * 8), kf);
                unpack8(*reinterpret_cast<const uint4*>(row + kh * DK + kt * kKT + part * 8), qf);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) kf[i] = qf[i] = 0.f;
            }
            const float rk = sBG[2 * kC + r], rq = sBG[3 * kC + r];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sKt[(part * 8 + i) * kC + r] = kf[i] * rk;
                sQt[(part * 8 + i) * kC + r] = qf[i] * rq;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < kKT; ++k) {
            const float4 ka = *reinterpret_cast<const float4*>(&sKt[k * kC + ty * 4]);
            const float4 kb = *reinterpret_cast<const float4*>(&sKt[k * kC + tx * 4]);
            const float4 qa = *reinterpret_cast<const float4*>(&sQt[k * kC + ty * 4]);
            const float av[4] = {ka.x, ka.y, ka.z, ka.w}, bv[4] = {kb.x, kb.y, kb.z, kb.w}, qv[4] = {qa.x, qa.y, qa.z, qa.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    gk[i][j] += av[i] * bv[j];
                    gq[i][j] += qv[i] * bv[j];
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        *reinterpret_cast<float4*>(&sA[(ty * 4 + i) * kC + tx * 4]) = make_float4(gk[i][0], gk[i][1], gk[i][2], gk[i][3]);
        *reinterpret_cast<float4*>(&sB[(ty * 4 + i) * kC + tx * 4]) = make_float4(gq[i][0], gq[i][1], gq[i][2], gq[i][3]);
    }
    __syncthreads();
    // L (strictly lower) and M (lower with diagonal)
    for (int idx = tid; idx < kMat; idx += 256) {
        const int r = idx >> 6, p = idx & 63;
        const float ex = p <= r ? expf(sBG[kC + r] - sBG[kC + p]) : 0.f;
        const float gkv = sA[idx], gqv = sB[idx];
        sA[idx] = p < r ? sBG[r] * ex * gkv : 0.f;
        sB[idx] = p <= r ? ex * gqv : 0.f;
    }
    __syncthreads();
    // T = (I + L)^-1 by forward substitution, one column per thread: T[r][p] = -sum_{s=p}^{r-1} L[r][s] T[s][p]
    float* sT = sU;  // the tiles are dead
    if (tid < kC) {
        const int p = tid;
        for (int r = 0; r < p; ++r) sT[r * kC + p] = 0.f;
        sT[p * kC + p] = 1.f;
        for (int r = p + 1; r < kC; ++r) {
            float acc = 0.f;
            for (int s = p; s < r; ++s) acc += sA[r * kC + s] * sT[s * kC + p];
            sT[r * kC + p] = -acc;
        }
    }
    __syncthreads();
    // store: transposed T and M (Tt[s][r] = T[r][s]), per-token scales
    float* out = scr + ((size_t)h * nchunks + c) * kScrFloats;
    for (int idx = tid; idx < kMat; idx += 256) {
        const int s = idx >> 6, r = idx & 63;
        out[idx] = sT[r * kC + s];
        out[kMat + idx] = sB[r * kC + s];
    }
    if (tid < kC) {
        const float G = sBG[kC + tid], Glast = sBG[kC + kC - 1];
        const float e = expf(G);
        const float rk = sBG[2 * kC + tid], rq = sBG[3 * kC + tid];
        float* aux = out + 2 * kMat;
        aux[tid] = sBG[tid];                       // beta
        aux[kC + tid] = e;                         // e_r = exp(G_r)
        aux[2 * kC + tid] = rk;                    // k^ scale
        aux[3 * kC + tid] = rq * e;                // q^ e scale
        aux[4 * kC + tid] = rk * expf(Glast - G);  // k^ (e_last / e_r) scale
    }
}

// ---- kernel 2: the chunk recurrence on a 32-column slice of one head's state -------------------------------------------
// smem: S[dk][36] | Kt[32][64] | Qt[32][64] (= Y = delta) | Tt[64][64] | Mt[64][64] | aux[5*64]
template <int DK>
__global__ void __launch_bounds__(128) gdn_chunk_state_kernel(
    const ik_bf16* __restrict__ qkv, const float* __restrict__ scr, ik_bf16* __restrict__ o, float* __restrict__ state,
    int64_t T, int Hk, int Hv, int dv, int nchunks) {
    extern __shared__ __align__(16) float sm[];
    float* sS = sm;
    float* sKt = sS + DK * kSStride;
    float* sQt = sKt + kKT * kC;
    float* sTt = sQt + kKT * kC;
    float* sMt = sTt + kMat;
    float* sAux = sMt + kMat;  // beta | e | rk | rqe | kd
    float* sY = sQt;           // Y, then delta (the Q tile is dead by then)
    const int slice = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const int j00 = slice * kCols;
    const int ty = tid >> 3, tx = tid & 7;       // output tile 4 rows x 4 columns: r0 = ty*4, j0 = tx*4
    const int r0 = ty * 4, j0 = tx * 4;
    const int smax = (tid >> 5) * 16 + 15;       // triangular loops: rows of this warp are <= smax

    for (int idx = tid; idx < kCols * (DK / 4); idx += 128) {
        const int jl = idx / (DK / 4), k4 = (idx % (DK / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(state + ((size_t)h * dv + j00 + jl) * DK + k4);
        sS[(k4 + 0) * kSStride + jl] = v.x;
        sS[(k4 + 1) * kSStride + jl] = v.y;
        sS[(k4 + 2) * kSStride + jl] = v.z;
        sS[(k4 + 3) * kSStride + jl] = v.w;
    }
    __syncthreads();

    // tile loader: columns kt*32 .. +32 of the k (and q) vectors of the 64 tokens, scaled, transposed to [col][token]
    auto load_tile = [&](int64_t t0, int n, int kt, bool with_q, const float* kscale, const float* qscale) {
        const int r = tid & 63, part = tid >> 6;  // 16 columns each
        float kf[16], qf[16];
        if (r < n) {
            const ik_bf16* row = qkv + (t0 + r) * C;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                unpack8(*reinterpret_cast<const uint4*>(row + Hk * DK + kh * DK + kt * kKT + part * 16 + u * 8), kf + u * 8);
                if (with_q) unpack8(*reinterpret_cast<const uint4*>(row + kh * DK + kt * kKT + part * 16 + u * 8), qf + u * 8);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) kf[i] = qf[i] = 0.f;
        }
        const float ks = kscale[r], qs = with_q ? qscale[r] : 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sKt[(part * 16 + i) * kC + r] = kf[i] * ks;
            if (with_q) sQt[(part * 16 + i) * kC + r] = qf[i] * qs;
        }
    };

    for (int c = 0; c < nchunks; ++c) {
        const int64_t t0 = (int64_t)c * kC;
        const int n = (int)min((int64_t)kC, T - t0);
        const float* sc = scr + ((size_t)h * nchunks + c) * kScrFloats;
        for (int i = tid; i < (2 * kMat + 5 * kC) / 4; i += 128)
            reinterpret_cast<float4*>(sTt)[i] = reinterpret_cast<const float4*>(sc)[i];  // Tt, Mt, aux are contiguous in smem
        __syncthreads();

        // X = K^ S0 and O1 = (e Q^) S0 for this thread's 4 x 4 tile
        float xa[4][4], oa[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) xa[i][j] = oa[i][j] = 0.f;
        for (int kt = 0; kt < DK / kKT; ++kt) {
            load_tile(t0, n, kt, true, sAux + 2 * kC, sAux + 3 * kC);
            __syncthreads();
#pragma unroll 8
            for (int k = 0; k < kKT; ++k) {
                const float4 s4 = *reinterpret_cast<const float4*>(&sS[(kt * kKT + k) * kSStride + j0]);
                const float4 k4 = *reinterpret_cast<const float4*>(&sKt[k * kC + r0]);
                const float4 q4 = *reinterpret_cast<const float4*>(&sQt[k * kC + r0]);
                const float sv[4] = {s4.x, s4.y, s4.z, s4.w}, kv[4] = {k4.x, k4.y, k4.z, k4.w}, qv[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xa[i][j] += kv[i] * sv[j];
                        oa[i][j] += qv[i] * sv[j];
                    }
            }
            __syncthreads();
        }
        // Y = beta (v - e X)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = r0 + i;
            float vv[4] = {0.f, 0.f, 0.f, 0.f};
            if (r < n) {
                const uint2 raw = *reinterpret_cast<const uint2*>(qkv + (t0 + r) * C + 2 * Hk * DK + h * dv + j00 + j0);
                vv[0] = __uint_as_float(raw.x << 16);
                vv[1] = __uint_as_float(raw.x & 0xffff0000u);
                vv[2] = __uint_as_float(raw.y << 16);
                vv[3] = __uint_as_float(raw.y & 0xffff0000u);
            }
            const float beta = sAux[r], e = sAux[kC + r];
            *reinterpret_cast<float4*>(&sY[r * kCols + j0]) =
                make_float4(beta * (vv[0] - e * xa[i][0]), beta * (vv[1] - e * xa[i][1]),
                            beta * (vv[2] - e * xa[i][2]), beta * (vv[3] - e * xa[i][3]));
        }
        __syncthreads();
        // delta = T Y (triangular)
        float dl[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) dl[i][j] = 0.f;
        for (int s = 0; s <= smax; ++s) {
            const float4 y4 = *reinterpret_cast<const float4*>(&sY[s * kCols + j0]);
            const float4 t4 = *reinterpret_cast<const float4*>(&sTt[s * kC + r0]);
            const float yv[4] = {y4.x, y4.y, y4.z, y4.w}, tv[4] = {t4.x, t4.y, t4.z, t4.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) dl[i][j] += tv[i] * yv[j];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
            *reinterpret_cast<float4*>(&sY[(r0 + i) * kCols + j0]) = make_float4(dl[i][0], dl[i][1], dl[i][2], dl[i][3]);
        __syncthreads();
        // out = O1 + M delta (lower triangular), written as bf16
        for (int p = 0; p <= smax; ++p) {
            const float4 d4 = *reinterpret_cast<const float4*>(&sY[p * kCols + j0]);
            const float4 m4 = *reinterpret_cast<const float4*>(&sMt[p * kC + r0]);
            const float dv_[4] = {d4.x, d4.y, d4.z, d4.w}, mv[4] = {m4.x, m4.y, m4.z, m4.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) oa[i][j] += mv[i] * dv_[j];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = r0 + i;
            if (r < n) {
                const __nv_bfloat162 lo = __floats2bfloat162_rn(oa[i][0], oa[i][1]);
                const __nv_bfloat162 hi = __floats2bfloat162_rn(oa[i][2], oa[i][3]);
                uint2 pk;
                pk.x = *reinterpret_cast<const uint32_t*>(&lo);
                pk.y = *reinterpret_cast<const uint32_t*>(&hi);
                *reinterpret_cast<uint2*>(o + ((t0 + r) * Hv + h) * dv + j00 + j0) = pk;
            }
        }
        // S = e_last S + sum_r k^_r (e_last / e_r) delta_r^T, per key tile; thread tile 4 keys x 2 columns
        const float e_last = sAux[kC + kC - 1];
        const int k0 = (tid >> 4) * 4, jj = (tid & 15) * 2;
        for (int kt = 0; kt < DK / kKT; ++kt) {
            load_tile(t0, n, kt, false, sAux + 4 * kC, nullptr);
            __syncthreads();
            float acc[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = 0.f;
#pragma unroll 4
            for (int r = 0; r < kC; r += 4) {
                float4 ks[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) ks[i] = *reinterpret_cast<const float4*>(&sKt[(k0 + i) * kC + r]);
#pragma unroll
                for (int rr = 0; rr < 4; ++rr) {
                    const float2 d2 = *reinterpret_cast<const float2*>(&sY[(r + rr) * kCols + jj]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float kv = rr == 0 ? ks[i].x : rr == 1 ? ks[i].y : rr == 2 ? ks[i].z : ks[i].w;
                        acc[i][0] += kv * d2.x;
                        acc[i][1] += kv * d2.y;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float* sp = &sS[(kt * kKT + k0 + i) * kSStride + jj];
                sp[0] = e_last * sp[0] + acc[i][0];
                sp[1] = e_last * sp[1] + acc[i][1];
            }
            __syncthreads();
        }
    }
    for (int idx = tid; idx < kCols * (DK / 4); idx += 128) {
        const int jl = idx / (DK / 4), k4 = (idx % (DK / 4)) * 4;
        *reinterpret_cast<float4*>(state + ((size_t)h * dv + j00 + jl) * DK + k4) =
            make_float4(sS[(k4 + 0) * kSStride + jl], sS[(k4 + 1) * kSStride + jl], sS[(k4 + 2) * kSStride + jl],
                        sS[(k4 + 3) * kSStride + jl]);
    }
}

constexpr size_t prep_smem_bytes() { return (size_t)(3 * kMat + 4 * kC + 4) * sizeof(float); }
template <int DK>
constexpr size_t state_smem_bytes() { return (size_t)(DK * kSStride + 2 * kKT * kC + 2 * kMat + 5 * kC) * sizeof(float); }

template <int DK>
int launch_chunked(const ik_bf16* qkv, const ik_bf16* a, const ik_bf16* b, const ik_bf16* a_log, const ik_bf16* dt_bias,
                   float eps, ik_bf16* o, float* state, float* scr, int64_t T, int Hk, int Hv, int dv, cudaStream_t s) {
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gdn_chunk_prep_kernel<DK>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem_bytes());
        cudaFuncSetAttribute(reinterpret_cast<const void*>(gdn_chunk_state_kernel<DK>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)state_smem_bytes<DK>());
        return true;
    }();
    (void)attr_set;
    const int nchunks = (int)((T + kC - 1) / kC);
    gdn_chunk_prep_kernel<DK><<<dim3((unsigned)nchunks, (unsigned)Hv), 256, prep_smem_bytes(), s>>>(
        qkv, a, b, a_log, dt_bias, eps, scr, T, Hk, Hv, dv, nchunks);
    IK_TRY(ik_launch_status());
    gdn_chunk_state_kernel<DK><<<dim3((unsigned)(dv / kCols), (unsigned)Hv), 128, state_smem_bytes<DK>(), s>>>(
        qkv, scr, o, state, T, Hk, Hv, dv, nchunks);
    return ik_launch_status();
}

}  // namespace

extern "C" {

// Same contract as ik_gdn_delta_rule_bf16, plus plan scratch of ceil(T/64) * Hv * 8512 floats. dk in {64, 128}, dv % 32 == 0.
int ik_gdn_delta_rule_bf16_chunked(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
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
    if ((dk != 64 && dk != 128) || dv % kCols != 0) return IK_ERR_UNSUPPORTED;
    if (!ik_aligned16(state->data) || !ik_aligned16(qkv->data)) return IK_ERR_INVALID_ARG;
    if (T == 0) return IK_OK;
    const int64_t nchunks = (T + kC - 1) / kC;
    const int64_t need = nchunks * Hv * kScrFloats * 4;
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    cudaStream_t s = IK_STREAM(stream);
    if (dk == 128)
        return launch_chunked<128>((const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data,
                                   (const ik_bf16*)a_log->data, (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data,
                                   (float*)state->data, (float*)scratch->data, T, (int)Hk, (int)Hv, (int)dv, s);
    return launch_chunked<64>((const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data,
                              (const ik_bf16*)a_log->data, (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data,
                              (float*)state->data, (float*)scratch->data, T, (int)Hk, (int)Hv, (int)dv, s);
}

}  // extern "C"
