// Lane hp: chunked (WY form) Gated DeltaNet prefill, HIP port of kernels/cuda/gdn_chunked.cu (git 287c371, design notes in
// docs/results/2026-10-17-cuda-prefill-2.md sections 1, 4-6), timed against gdn_delta_rows_v2_kernel (kernels/hip/gdn.hip,
// included below) at the 35B's head layout (Hk 16, Hv 32, dv = dk = 128).
// Build (dev image): hipcc -O3 --offload-arch=gfx1201 -I kernels/hip -I kernels gdnwy.hip kernels/hip/ik_runtime.hip -o gdnwy
//
// The algorithm and the fp32 arithmetic (summation order included) are the CUDA kernels', see the header of
// gdn_chunked.cu: per value head and chunk of CH tokens
//     (I + L) delta = Y,  L[r][p] = beta_r (e_r/e_p)(k^_r.k^_p) (p < r),  Y = beta (v - e (K^ S0)),  T = (I + L)^-1
//     o = e Q^ S0 + M delta,  M[r][p] = (e_r/e_p)(q^_r.k^_p) (p <= r),  S' = e_last S0 + sum_r (e_last/e_r) k^_r delta_r^T
// kernel 1 (gdn_wy_prep_kernel, grid chunks x Hv, 256 threads) writes T^T, M^T and per-token scales to scratch, kernel 2
// (gdn_wy_state_kernel, grid dv/COLS x Hv, 128 threads) walks the chunks in order for COLS value columns of one head.
// Scratch layout per (head, chunk), unchanged: Tt[CH][CH] | Mt[CH][CH] | beta[CH] | e[CH] | rk[CH] | rqe[CH] | kd[CH].
//
// What differs from the CUDA source (none of it changes a float operation or its order):
//  * CUDA warp intrinsics -> ik_warp_sum / __shfl_up (width 32 == wave32); bf16 packing -> ik_pack2 (round to nearest even,
//    same as __floats2bfloat162_rn); no cudaFuncSetAttribute (not needed below 64 KB of LDS).
//  * LDS: the CUDA state kernel uses 68,864 B of shared memory (opt-in up to 100 KB on sm_121); a gfx1201 workgroup has
//    64 KB. M^T is therefore not kept resident next to T^T: it is read from scratch into registers at the top of the chunk
//    and stored into the T^T buffer once delta = T Y is done (T^T is dead from then on). 52,480 B, one extra barrier-free
//    store per chunk; same values, same order.
//  * Templated on the chunk length CH (64 = the CUDA kernel, 32 = extra) and on COLS, the value columns per state block
//    (32 = the CUDA kernel, 16 = extra: 256 blocks per call instead of 128, same 128 threads, so half the columns per
//    thread). Thread maps are the CUDA ones generalised: prep Gram tile (CH/16)^2 per thread, state output tile 4 rows x
//    (COLS * CH/4 / 128) columns, state update tile 4 keys x (COLS/16) columns.
// The deterministic result differs from v2 by fp32 reassociation (a different but fixed summation order), not bitwise.
#include "gdn.hip"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>

namespace {

constexpr int kWyKT = 32;  // key-dimension tile
typedef float wy_f4 __attribute__((ext_vector_type(4)));  // native vector: stays in registers (a float4 array went to scratch)

IK_DEVI float wy_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
IK_DEVI float wy_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// N consecutive floats from / to LDS (N = 4 -> float4, 2 -> float2, 1 -> float)
template <int N>
IK_DEVI void wy_ld(const float* p, float (&v)[N]) {
    if constexpr (N == 4) {
        const float4 t = *reinterpret_cast<const float4*>(p);
        v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
    } else if constexpr (N == 2) {
        const float2 t = *reinterpret_cast<const float2*>(p);
        v[0] = t.x; v[1] = t.y;
    } else {
        v[0] = p[0];
    }
}
template <int N>
IK_DEVI void wy_st(float* p, const float (&v)[N]) {
    if constexpr (N == 4) {
        *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    } else if constexpr (N == 2) {
        *reinterpret_cast<float2*>(p) = make_float2(v[0], v[1]);
    } else {
        p[0] = v[0];
    }
}
// N consecutive bf16 from global memory to floats / floats to bf16 (N multiple of 8, or 4, 2, 1)
template <int N>
IK_DEVI void wy_ldbf(const ik_bf16* p, float* f) {
    if constexpr (N >= 8) {
#pragma unroll
        for (int u = 0; u < N / 8; ++u) ik_unpack8(*reinterpret_cast<const uint4*>(p + u * 8), f + u * 8);
    } else if constexpr (N == 4) {
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        f[0] = __uint_as_float(raw.x << 16);
        f[1] = __uint_as_float(raw.x & 0xffff0000u);
        f[2] = __uint_as_float(raw.y << 16);
        f[3] = __uint_as_float(raw.y & 0xffff0000u);
    } else if constexpr (N == 2) {
        const uint32_t raw = *reinterpret_cast<const uint32_t*>(p);
        f[0] = __uint_as_float(raw << 16);
        f[1] = __uint_as_float(raw & 0xffff0000u);
    } else {
        f[0] = ik_b2f(p[0]);
    }
}
template <int N>
IK_DEVI void wy_stbf(ik_bf16* p, const float (&f)[N]) {
    if constexpr (N == 4) {
        uint2 pk;
        pk.x = ik_pack2(f[0], f[1]);
        pk.y = ik_pack2(f[2], f[3]);
        *reinterpret_cast<uint2*>(p) = pk;
    } else if constexpr (N == 2) {
        *reinterpret_cast<uint32_t*>(p) = ik_pack2(f[0], f[1]);
    } else {
        p[0] = ik_f2b(f[0]);
    }
}

template <int CH>
constexpr int wy_scr_floats() { return 2 * CH * CH + 5 * CH; }
template <int CH>
constexpr int wy_prep_u() { return (CH * CH > 2 * kWyKT * CH) ? CH * CH : 2 * kWyKT * CH; }  // floats of the tile / T buffer
template <int CH>
constexpr size_t wy_prep_smem() { return (size_t)(2 * CH * CH + wy_prep_u<CH>() + 4 * CH + 4) * sizeof(float); }
template <int DK, int CH, int COLS>
constexpr size_t wy_state_smem() {
    return (size_t)(DK * (COLS + 4) + 2 * kWyKT * CH + CH * CH + 5 * CH) * sizeof(float);
}

// ---- kernel 1: Gram products, T = (I + L)^-1, M, per-token scales ----------------------------------------------------
// smem: A[CH][CH] | B[CH][CH] | U (tiles Kt[32][CH] | Qt[32][CH] during the Gram loop, then T) | beta,G,rk,rq [4*CH] | tot
template <int DK, int CH>
__global__ void __launch_bounds__(256) gdn_wy_prep_kernel(
    const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
    const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps, float* __restrict__ scr,
    int64_t T, int Hk, int Hv, int dv, int nchunks) {
    static_assert(CH == 32 || CH == 64, "chunk length");
    static_assert(DK % kWyKT == 0, "dk tile");
    constexpr int MAT = CH * CH;
    constexpr int TR = CH / 16;           // Gram tile TR x TR per thread (16 x 16 threads)
    constexpr int NP = 256 / CH;          // tile-loader parts
    constexpr int CPP = kWyKT / NP;       // key columns per part
    extern __shared__ __align__(16) float wy_smem[];
    float* sA = wy_smem;                  // Gram of k^ -> L
    float* sB = sA + MAT;                 // Gram q^ k^ -> M
    float* sU = sB + MAT;                 // tiles, then T
    float* sBG = sU + wy_prep_u<CH>();    // beta[CH] | G[CH] | rk[CH] | rq[CH]
    float* sTot = sBG + 4 * CH;
    float* sKt = sU;
    float* sQt = sU + kWyKT * CH;
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const int64_t t0 = (int64_t)c * CH;
    const int n = (int)((T - t0) < (int64_t)CH ? (T - t0) : (int64_t)CH);
    const float scale = rsqrtf((float)DK);

    // l2 norms of the tokens of this chunk (rq carries the 1/sqrt(dk) scale); padded tokens get 0
    for (int r = warp; r < CH; r += 8) {
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
            sBG[2 * CH + r] = r < n ? rsqrtf(kk + eps) : 0.f;
            sBG[3 * CH + r] = r < n ? rsqrtf(qq + eps) * scale : 0.f;
        }
    }
    // beta and the cumulative log decay G (inclusive scan over the CH tokens); padded tokens: beta = 0, g = 0
    float g = 0.f, bt = 0.f;
    if (tid < CH && tid < n) {
        const float neg_a = -expf(ik_b2f(a_log[h])), dtb = ik_b2f(dt_bias[h]);
        g = neg_a * wy_softplus(ik_b2f(a[(t0 + tid) * Hv + h]) + dtb);
        bt = wy_sigmoid(ik_b2f(b[(t0 + tid) * Hv + h]));
    }
    float x = g;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float y = __shfl_up(x, o, 32);
        if (lane >= o) x += y;
    }
    if (tid == 31) *sTot = x;
    __syncthreads();
    if (tid < CH) {
        sBG[tid] = bt;
        sBG[CH + tid] = x + (tid >= 32 ? *sTot : 0.f);
    }
    __syncthreads();

    // Gram products: thread tile TR x TR of (k^ k^^T) and (q^ k^^T)
    const int ty = tid >> 4, tx = tid & 15;
    float gk[TR][TR], gq[TR][TR];
#pragma unroll
    for (int i = 0; i < TR; ++i)
#pragma unroll
        for (int j = 0; j < TR; ++j) gk[i][j] = gq[i][j] = 0.f;
    for (int kt = 0; kt < DK / kWyKT; ++kt) {
        {
            const int r = tid & (CH - 1), part = tid / CH;
            float kf[CPP], qf[CPP];
            if (r < n) {
                const ik_bf16* row = qkv + (t0 + r) * C;
                wy_ldbf<CPP>(row + Hk * DK + kh * DK + kt * kWyKT + part * CPP, kf);
                wy_ldbf<CPP>(row + kh * DK + kt * kWyKT + part * CPP, qf);
            } else {
#pragma unroll
                for (int i = 0; i < CPP; ++i) kf[i] = qf[i] = 0.f;
            }
            const float rk = sBG[2 * CH + r], rq = sBG[3 * CH + r];
#pragma unroll
            for (int i = 0; i < CPP; ++i) {
                sKt[(part * CPP + i) * CH + r] = kf[i] * rk;
                sQt[(part * CPP + i) * CH + r] = qf[i] * rq;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < kWyKT; ++k) {
            float av[TR], bv[TR], qv[TR];
            wy_ld<TR>(&sKt[k * CH + ty * TR], av);
            wy_ld<TR>(&sKt[k * CH + tx * TR], bv);
            wy_ld<TR>(&sQt[k * CH + ty * TR], qv);
#pragma unroll
            for (int i = 0; i < TR; ++i)
#pragma unroll
                for (int j = 0; j < TR; ++j) {
                    gk[i][j] += av[i] * bv[j];
                    gq[i][j] += qv[i] * bv[j];
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TR; ++i) {
        wy_st<TR>(&sA[(ty * TR + i) * CH + tx * TR], gk[i]);
        wy_st<TR>(&sB[(ty * TR + i) * CH + tx * TR], gq[i]);
    }
    __syncthreads();
    // L (strictly lower) and M (lower with diagonal)
    for (int idx = tid; idx < MAT; idx += 256) {
        const int r = idx / CH, p = idx % CH;
        const float ex = p <= r ? expf(sBG[CH + r] - sBG[CH + p]) : 0.f;
        const float gkv = sA[idx], gqv = sB[idx];
        sA[idx] = p < r ? sBG[r] * ex * gkv : 0.f;
        sB[idx] = p <= r ? ex * gqv : 0.f;
    }
    __syncthreads();
    // T = (I + L)^-1 by forward substitution, one column per thread: T[r][p] = -sum_{s=p}^{r-1} L[r][s] T[s][p]
    float* sT = sU;  // the tiles are dead
    if (tid < CH) {
        const int p = tid;
        for (int r = 0; r < p; ++r) sT[r * CH + p] = 0.f;
        sT[p * CH + p] = 1.f;
        for (int r = p + 1; r < CH; ++r) {
            float acc = 0.f;
            for (int s = p; s < r; ++s) acc += sA[r * CH + s] * sT[s * CH + p];
            sT[r * CH + p] = -acc;
        }
    }
    __syncthreads();
    // store: transposed T and M (Tt[s][r] = T[r][s]), per-token scales
    float* out = scr + ((size_t)h * nchunks + c) * wy_scr_floats<CH>();
    for (int idx = tid; idx < MAT; idx += 256) {
        const int s = idx / CH, r = idx % CH;
        out[idx] = sT[r * CH + s];
        out[MAT + idx] = sB[r * CH + s];
    }
    if (tid < CH) {
        const float G = sBG[CH + tid], Glast = sBG[CH + CH - 1];
        const float e = expf(G);
        const float rk = sBG[2 * CH + tid], rq = sBG[3 * CH + tid];
        float* aux = out + 2 * MAT;
        aux[tid] = sBG[tid];                       // beta
        aux[CH + tid] = e;                         // e_r = exp(G_r)
        aux[2 * CH + tid] = rk;                    // k^ scale
        aux[3 * CH + tid] = rq * e;                // q^ e scale
        aux[4 * CH + tid] = rk * expf(Glast - G);  // k^ (e_last / e_r) scale
    }
}

// ---- kernel 2: the chunk recurrence on a COLS-column slice of one head's state ----------------------------------------
// smem: S[dk][COLS+4] | Kt[32][CH] | Qt[32][CH] (= Y = delta, [CH][COLS]) | Tt[CH][CH] (= Mt after delta) | aux[5*CH]
template <int DK, int CH, int COLS>
__global__ void __launch_bounds__(128) gdn_wy_state_kernel(
    const ik_bf16* __restrict__ qkv, const float* __restrict__ scr, ik_bf16* __restrict__ o, float* __restrict__ state,
    int64_t T, int Hk, int Hv, int dv, int nchunks) {
    static_assert(CH == 32 || CH == 64, "chunk length");
    static_assert(COLS == 16 || COLS == 32, "columns per block");
    constexpr int MAT = CH * CH;
    constexpr int SS = COLS + 4;          // smem row stride of the state slice [dk][COLS] (16-byte aligned rows)
    constexpr int RG = CH / 4;            // row groups of 4 tokens
    constexpr int CG = 128 / RG;          // column groups
    constexpr int JT = COLS / CG;         // columns per thread, output tile 4 x JT
    constexpr int RPW = (32 / CG) * 4;    // token rows per warp (triangular loop bound is warp-uniform)
    constexpr int CGU = 16;               // state update: 16 column groups x 8 key groups of 4
    constexpr int JU = COLS / CGU;        // columns per thread in the update, tile 4 x JU
    constexpr int NPT = 128 / CH;         // tile-loader parts
    constexpr int CPT = kWyKT / NPT;      // key columns per part
    constexpr int NM = MAT / 4 / 128;     // float4 of M^T per thread
    static_assert(JT >= 1 && JU >= 1 && COLS % CG == 0, "tile maps");
    extern __shared__ __align__(16) float wy_smem[];
    float* sS = wy_smem;
    float* sKt = sS + DK * SS;
    float* sQt = sKt + kWyKT * CH;
    float* sTt = sQt + kWyKT * CH;        // T^T, then M^T
    float* sAux = sTt + MAT;              // beta | e | rk | rqe | kd
    float* sY = sQt;                      // Y, then delta (the Q tile is dead by then)
    const int slice = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const int j00 = slice * COLS;
    const int ty = tid / CG, tx = tid % CG;
    const int r0 = ty * 4, j0 = tx * JT;
    const int smax = (tid >> 5) * RPW + RPW - 1;  // triangular loops: rows of this warp are <= smax

    for (int idx = tid; idx < COLS * (DK / 4); idx += 128) {
        const int jl = idx / (DK / 4), k4 = (idx % (DK / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(state + ((size_t)h * dv + j00 + jl) * DK + k4);
        sS[(k4 + 0) * SS + jl] = v.x;
        sS[(k4 + 1) * SS + jl] = v.y;
        sS[(k4 + 2) * SS + jl] = v.z;
        sS[(k4 + 3) * SS + jl] = v.w;
    }
    __syncthreads();

    // tile loader: columns kt*32 .. +32 of the k (and q) vectors of the CH tokens, scaled, transposed to [col][token]
    auto load_tile = [&](int64_t t0, int n, int kt, bool with_q, const float* kscale, const float* qscale) {
        const int r = tid & (CH - 1), part = tid / CH;
        float kf[CPT], qf[CPT];
        if (r < n) {
            const ik_bf16* row = qkv + (t0 + r) * C;
            wy_ldbf<CPT>(row + Hk * DK + kh * DK + kt * kWyKT + part * CPT, kf);
            if (with_q) wy_ldbf<CPT>(row + kh * DK + kt * kWyKT + part * CPT, qf);
        } else {
#pragma unroll
            for (int i = 0; i < CPT; ++i) kf[i] = qf[i] = 0.f;
        }
        const float ks = kscale[r], qs = with_q ? qscale[r] : 0.f;
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            sKt[(part * CPT + i) * CH + r] = kf[i] * ks;
            if (with_q) sQt[(part * CPT + i) * CH + r] = qf[i] * qs;
        }
    };

    for (int c = 0; c < nchunks; ++c) {
        const int64_t t0 = (int64_t)c * CH;
        const int n = (int)((T - t0) < (int64_t)CH ? (T - t0) : (int64_t)CH);
        const float* sc = scr + ((size_t)h * nchunks + c) * wy_scr_floats<CH>();
        for (int i = tid; i < MAT / 4; i += 128)
            reinterpret_cast<float4*>(sTt)[i] = reinterpret_cast<const float4*>(sc)[i];  // T^T
        for (int i = tid; i < 5 * CH / 4; i += 128)
            reinterpret_cast<float4*>(sAux)[i] = reinterpret_cast<const float4*>(sc + 2 * MAT)[i];
        wy_f4 mreg[NM];  // M^T, parked in registers until T^T is dead
#pragma unroll
        for (int i = 0; i < NM; ++i) mreg[i] = reinterpret_cast<const wy_f4*>(sc + MAT)[i * 128 + tid];
        __syncthreads();

        // X = K^ S0 and O1 = (e Q^) S0 for this thread's 4 x JT tile
        float xa[4][JT], oa[4][JT];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < JT; ++j) xa[i][j] = oa[i][j] = 0.f;
        for (int kt = 0; kt < DK / kWyKT; ++kt) {
            load_tile(t0, n, kt, true, sAux + 2 * CH, sAux + 3 * CH);
            __syncthreads();
#pragma unroll 8
            for (int k = 0; k < kWyKT; ++k) {
                float sv[JT], kv[4], qv[4];
                wy_ld<JT>(&sS[(kt * kWyKT + k) * SS + j0], sv);
                wy_ld<4>(&sKt[k * CH + r0], kv);
                wy_ld<4>(&sQt[k * CH + r0], qv);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < JT; ++j) {
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
            float vv[JT];
#pragma unroll
            for (int j = 0; j < JT; ++j) vv[j] = 0.f;
            if (r < n) wy_ldbf<JT>(qkv + (t0 + r) * C + 2 * Hk * DK + h * dv + j00 + j0, vv);
            const float beta = sAux[r], e = sAux[CH + r];
            float y[JT];
#pragma unroll
            for (int j = 0; j < JT; ++j) y[j] = beta * (vv[j] - e * xa[i][j]);
            wy_st<JT>(&sY[r * COLS + j0], y);
        }
        __syncthreads();
        // delta = T Y (triangular)
        float dl[4][JT];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < JT; ++j) dl[i][j] = 0.f;
        for (int s = 0; s <= smax; ++s) {
            float yv[JT], tv[4];
            wy_ld<JT>(&sY[s * COLS + j0], yv);
            wy_ld<4>(&sTt[s * CH + r0], tv);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < JT; ++j) dl[i][j] += tv[i] * yv[j];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) wy_st<JT>(&sY[(r0 + i) * COLS + j0], dl[i]);
#pragma unroll
        for (int i = 0; i < NM; ++i) reinterpret_cast<wy_f4*>(sTt)[i * 128 + tid] = mreg[i];  // T^T is dead: M^T takes its place
        __syncthreads();
        // out = O1 + M delta (lower triangular), written as bf16
        for (int p = 0; p <= smax; ++p) {
            float dv_[JT], mv[4];
            wy_ld<JT>(&sY[p * COLS + j0], dv_);
            wy_ld<4>(&sTt[p * CH + r0], mv);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < JT; ++j) oa[i][j] += mv[i] * dv_[j];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = r0 + i;
            if (r < n) wy_stbf<JT>(o + ((t0 + r) * Hv + h) * dv + j00 + j0, oa[i]);
        }
        // S = e_last S + sum_r k^_r (e_last / e_r) delta_r^T, per key tile; thread tile 4 keys x JU columns
        const float e_last = sAux[CH + CH - 1];
        const int k0 = (tid / CGU) * 4, jj = (tid % CGU) * JU;
        for (int kt = 0; kt < DK / kWyKT; ++kt) {
            load_tile(t0, n, kt, false, sAux + 4 * CH, nullptr);
            __syncthreads();
            float acc[4][JU];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < JU; ++j) acc[i][j] = 0.f;
#pragma unroll 4
            for (int r = 0; r < CH; r += 4) {
                float4 ks[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) ks[i] = *reinterpret_cast<const float4*>(&sKt[(k0 + i) * CH + r]);
#pragma unroll
                for (int rr = 0; rr < 4; ++rr) {
                    float d2[JU];
                    wy_ld<JU>(&sY[(r + rr) * COLS + jj], d2);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float kv = rr == 0 ? ks[i].x : rr == 1 ? ks[i].y : rr == 2 ? ks[i].z : ks[i].w;
#pragma unroll
                        for (int j = 0; j < JU; ++j) acc[i][j] += kv * d2[j];
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float* sp = &sS[(kt * kWyKT + k0 + i) * SS + jj];
#pragma unroll
                for (int j = 0; j < JU; ++j) sp[j] = e_last * sp[j] + acc[i][j];
            }
            __syncthreads();
        }
    }
    for (int idx = tid; idx < COLS * (DK / 4); idx += 128) {
        const int jl = idx / (DK / 4), k4 = (idx % (DK / 4)) * 4;
        *reinterpret_cast<float4*>(state + ((size_t)h * dv + j00 + jl) * DK + k4) =
            make_float4(sS[(k4 + 0) * SS + jl], sS[(k4 + 1) * SS + jl], sS[(k4 + 2) * SS + jl], sS[(k4 + 3) * SS + jl]);
    }
}

struct WyArgs {
    const ik_bf16 *qkv, *a, *b, *a_log, *dt_bias;
    float eps;
    ik_bf16* o;
    float* state;
    float* scr;
    int64_t T;
    int Hk, Hv, dv;
};

// which: 0 both kernels, 1 prep only, 2 state only (scratch from an earlier prep)
template <int CH, int COLS>
void wy_run(const WyArgs& g, int which, hipStream_t s) {
    constexpr int DK = 128;
    static_assert(wy_state_smem<DK, CH, COLS>() <= 65536, "gfx1201 workgroup LDS is 64 KB");
    static_assert(wy_prep_smem<CH>() <= 65536, "gfx1201 workgroup LDS is 64 KB");
    const int nchunks = (int)((g.T + CH - 1) / CH);
    if (which != 2)
        gdn_wy_prep_kernel<DK, CH><<<dim3((unsigned)nchunks, (unsigned)g.Hv), 256, wy_prep_smem<CH>(), s>>>(
            g.qkv, g.a, g.b, g.a_log, g.dt_bias, g.eps, g.scr, g.T, g.Hk, g.Hv, g.dv, nchunks);
    if (which != 1)
        gdn_wy_state_kernel<DK, CH, COLS><<<dim3((unsigned)(g.dv / COLS), (unsigned)g.Hv), 128, wy_state_smem<DK, CH, COLS>(), s>>>(
            g.qkv, g.scr, g.o, g.state, g.T, g.Hk, g.Hv, g.dv, nchunks);
}

struct WyVariant {
    const char* name;
    int ch;
    void (*run)(const WyArgs&, int, hipStream_t);
};

}  // namespace

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

struct Case {
    const char* name;
    ik_bf16 *dq, *da, *db, *dal, *ddt;
    float* dinit;  // incoming f32 state
    size_t nstate;
};

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    const int Hk = 16, Hv = 32, dv = 128, dk = 128, C = 2 * Hk * dk + Hv * dv;
    const int Ts[] = {1, 5, 63, 64, 65, 200, 683, 4096};
    const int64_t Tmax = 4096;
    uint64_t r = 88172645463325252ull;
    auto rnd = [&]() {  // ~N(0,1)-ish: sum of uniforms
        float s = 0;
        for (int i = 0; i < 4; ++i) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            s += (float)(r >> 40) / (float)(1ull << 24) - 0.5f;
        }
        return s * 1.7f;
    };
    const size_t nstate = (size_t)Hv * dv * dk;
    // case 1: the gdnbench inputs. case 2: the CUDA test's correlated keys: every key is the same direction (per key head)
    // plus 0.05 noise, beta = sigmoid(6) ~ 1, dt_bias mean -6 (decay close to 1), incoming state scale 0.1.
    std::vector<uint16_t> hq((size_t)Tmax * C), ha((size_t)Tmax * Hv), hb((size_t)Tmax * Hv), hal(Hv), hdt(Hv);
    std::vector<float> hs(nstate);
    for (auto& x : hq) x = f2bf(rnd());
    for (auto& x : ha) x = f2bf(rnd());
    for (auto& x : hb) x = f2bf(1.5f * rnd());
    for (auto& x : hal) x = f2bf(0.5f * rnd());
    for (auto& x : hdt) x = f2bf(0.5f * rnd());
    for (auto& x : hs) x = 0.3f * rnd();
    std::vector<uint16_t> cq = hq, cb((size_t)Tmax * Hv), cdt(Hv);
    std::vector<float> cs(nstate);
    {
        std::vector<float> base((size_t)Hk * dk);
        for (auto& x : base) x = rnd();
        for (int64_t t = 0; t < Tmax; ++t)
            for (int i = 0; i < Hk * dk; ++i) cq[(size_t)t * C + Hk * dk + i] = f2bf(base[i] + 0.05f * rnd());
        for (auto& x : cb) x = f2bf(6.0f);
        for (auto& x : cdt) x = f2bf(-6.0f + 0.5f * rnd());
        for (auto& x : cs) x = 0.1f * rnd();
    }
    auto up = [&](const void* h, size_t bytes) {
        void* d;
        CK(hipMalloc(&d, bytes));
        CK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
        return d;
    };
    Case cases[2];
    cases[0] = {"random", (ik_bf16*)up(hq.data(), hq.size() * 2), (ik_bf16*)up(ha.data(), ha.size() * 2),
                (ik_bf16*)up(hb.data(), hb.size() * 2), (ik_bf16*)up(hal.data(), Hv * 2), (ik_bf16*)up(hdt.data(), Hv * 2),
                (float*)up(hs.data(), nstate * 4), nstate};
    cases[1] = {"correlated keys", (ik_bf16*)up(cq.data(), cq.size() * 2), (ik_bf16*)up(ha.data(), ha.size() * 2),
                (ik_bf16*)up(cb.data(), cb.size() * 2), (ik_bf16*)up(hal.data(), Hv * 2), (ik_bf16*)up(cdt.data(), Hv * 2),
                (float*)up(cs.data(), nstate * 4), nstate};

    const float eps = 1e-6f;
    const size_t ostride = (size_t)Tmax * Hv * dv;
    ik_bf16 *o_ref, *o_new;
    float *s_ref, *s_new, *scr;
    CK(hipMalloc(&o_ref, ostride * 2));
    CK(hipMalloc(&o_new, ostride * 2));
    CK(hipMalloc(&s_ref, nstate * 4));
    CK(hipMalloc(&s_new, nstate * 4));
    // plan scratch: ceil(T / CH) * Hv * (2 CH^2 + 5 CH) floats; CH = 32 needs 128 chunks * 2208, CH = 64 needs 64 * 8512
    const size_t scr_floats = std::max((size_t)(Tmax / 64) * Hv * (2 * 64 * 64 + 5 * 64), (size_t)(Tmax / 32) * Hv * (2 * 32 * 32 + 5 * 32));
    CK(hipMalloc(&scr, scr_floats * 4));

    const WyVariant variants[] = {
        {"wy64", 64, wy_run<64, 32>},        // the CUDA kernels
        {"wy64c16", 64, wy_run<64, 16>},     // chunk 64, 16-column state blocks (256 blocks)
        {"wy32", 32, wy_run<32, 32>},        // chunk 32
        {"wy32c16", 32, wy_run<32, 16>},     // chunk 32, 16-column state blocks
    };

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    std::vector<uint16_t> ro((size_t)Tmax * Hv * dv), go((size_t)Tmax * Hv * dv);
    std::vector<float> rs(nstate), gs(nstate);
    bool all_ok = true;

    for (const Case& cs_ : cases) {
        auto reset = [&](float* st) { CK(hipMemcpy(st, cs_.dinit, nstate * 4, hipMemcpyDeviceToDevice)); };
        auto run_v2 = [&](int64_t T) {
            gdn_delta_rows_v2_kernel<128, 2, 2><<<dim3(dv / 16, Hv), 32>>>(cs_.dq, cs_.da, cs_.db, cs_.dal, cs_.ddt, eps, o_ref, s_ref, T, Hk, Hv, dv);
        };
        auto best_of = [&](auto&& launch, auto&& prep) {
            float best = 1e30f;
            for (int rep = 0; rep < 5; ++rep) {
                prep();
                CK(hipDeviceSynchronize());
                CK(hipEventRecord(e0));
                launch();
                CK(hipEventRecord(e1));
                CK(hipEventSynchronize(e1));
                float ms;
                CK(hipEventElapsedTime(&ms, e0, e1));
                best = fminf(best, ms);
            }
            return best * 1e3f;  // us
        };
        printf("== %s inputs (Hk %d, Hv %d, dk %d, dv %d); diffs are chunked vs v2, us = best of 5 ==\n", cs_.name, Hk, Hv, dk, dv);
        for (int T : Ts) {
            reset(s_ref);
            CK(hipMemset(o_ref, 0, (size_t)T * Hv * dv * 2));
            run_v2(T);
            CK(hipGetLastError());
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(ro.data(), o_ref, (size_t)T * Hv * dv * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(rs.data(), s_ref, nstate * 4, hipMemcpyDeviceToHost));
            const float us_v2 = best_of([&] { run_v2(T); }, [&] { reset(s_ref); });
            printf("T=%-5d v2        %9.1f us/call %7.3f us/tok\n", T, us_v2, us_v2 / T);
            for (const WyVariant& v : variants) {
                const WyArgs g{cs_.dq, cs_.da, cs_.db, cs_.dal, cs_.ddt, eps, o_new, s_new, scr, T, Hk, Hv, dv};
                reset(s_new);
                CK(hipMemset(o_new, 0, (size_t)T * Hv * dv * 2));
                v.run(g, 0, 0);
                CK(hipGetLastError());
                CK(hipDeviceSynchronize());
                CK(hipMemcpy(go.data(), o_new, (size_t)T * Hv * dv * 2, hipMemcpyDeviceToHost));
                CK(hipMemcpy(gs.data(), s_new, nstate * 4, hipMemcpyDeviceToHost));
                double on = 0, od = 0, omax = 0, sn = 0, sd = 0, smax = 0;
                for (size_t i = 0; i < (size_t)T * Hv * dv; ++i) {
                    const double x = bf2f(ro[i]), y = bf2f(go[i]), d = fabs(x - y);
                    on += x * x;
                    od += d * d;
                    omax = fmax(omax, d);
                    if (!(d == d)) omax = INFINITY;  // NaN
                }
                for (size_t i = 0; i < nstate; ++i) {
                    const double x = rs[i], y = gs[i], d = fabs(x - y);
                    sn += x * x;
                    sd += d * d;
                    smax = fmax(smax, d);
                    if (!(d == d)) smax = INFINITY;
                }
                const double orel = sqrt(od / (on > 0 ? on : 1)), srel = sqrt(sd / (sn > 0 ? sn : 1));
                const bool ok = orel < 1e-3 && srel < 1e-3 && omax < INFINITY && smax < INFINITY;
                all_ok &= ok;
                const float us = best_of([&] { v.run(g, 0, 0); }, [&] { reset(s_new); });
                // per-kernel split (the state kernel reuses the scratch of the full run above)
                const float us_prep = best_of([&] { v.run(g, 1, 0); }, [&] {});
                const float us_state = best_of([&] { v.run(g, 2, 0); }, [&] { reset(s_new); });
                printf("T=%-5d %-9s %9.1f us/call %7.3f us/tok (prep %.1f + state %.1f us)  out rel_l2 %.2e max_abs %.2e  state rel_l2 %.2e max_abs %.2e  %s\n",
                       T, v.name, us, us / T, us_prep, us_state, orel, omax, srel, smax, ok ? "ok" : "MISMATCH");
            }
        }
    }
    puts(all_ok ? "ALL CLOSE (rel_l2 < 1e-3)" : "SOME MISMATCH");
    return all_ok ? 0 : 1;
}
