// matmul / logits: out[M,N] = x[M,K] @ w[N,K]^T, bf16 inputs, f32 accumulate (logits: f32 output).
//   *_bf16        native WMMA GEMM (gfx12 v_wmma_f32_16x16x16_bf16, wave32) for any M. The reduction order over K
//                 is fixed by the loop, so it is run-to-run identical (Deterministic). A SIMT fallback inside the
//                 same launcher takes shapes the tile loader does not (K % 32, alignment).
//   *_bf16_gemv   native warp-per-row-group GEMV for M <= 4 (decode). Deterministic: fixed reduction order. Memory
//                 bound: it streams each weight row exactly once with 16-byte non-temporal loads, R rows per warp so
//                 the activation loads are shared, U loads in flight per lane and row.
// A hipBLASLt route (Fast) is not built: the dev image carries no hipblaslt; see docs/backends/hip.md.
#include "ik_hip.h"
#include <stdio.h>
#include <stdlib.h>

namespace {

// ---- GEMV --------------------------------------------------------------------------------
constexpr int kGemvWarps = 8;  // warps per block

IK_DEVI void store_out(float* p, float v) { *p = v; }
IK_DEVI void store_out(ik_bf16* p, float v) { *p = ik_f2b(v); }

// R rows per warp, U uint4 loads per lane and row in flight.
// `blk` is the block's index along the rows of this matrix (blockIdx.x for the plain kernel; the pair kernel below
// passes the index inside the segment the block belongs to, so both run exactly this code).
template <int M, int R, int U, typename OutT>
__device__ __forceinline__ void gemv_body(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w,
                                          OutT* __restrict__ out, int64_t N, int64_t K, int64_t blk) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = (blk * kGemvWarps + warp) * R;
    if (n0 >= N) return;  // whole warp exits together; no block-level sync below
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);
    const uint4* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < N ? n0 + r : N - 1;  // tail rows re-read the last row; their result is dropped
        wr[r] = reinterpret_cast<const uint4*>(w + n * K);
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

    for (int64_t base = 0; base < nv; base += (int64_t)IK_WARP * U) {
        uint4 wv[R][U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
#pragma unroll
            for (int r = 0; r < R; ++r) wv[r][u] = ok ? ik_ldcs(wr[r] + idx) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[8];
                ik_unpack8(ok ? ik_ldg(x8 + m * nv + idx) : make_uint4(0, 0, 0, 0), xf);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float wf[8];
                    ik_unpack8(wv[r][u], wf);
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[r][m] += wf[e] * xf[e];
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float s = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) store_out(out + m * N + n0 + r, s);
        }
}

template <int M, int R, int U, typename OutT>
__global__ void __launch_bounds__(kGemvWarps * IK_WARP)
gemv_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w, OutT* __restrict__ out, int64_t N,
            int64_t K) {
    gemv_body<M, R, U, OutT>(x, w, out, N, K, (int64_t)blockIdx.x);
}

// Two projections of one activation row in one launch (the GDN a / b gates, the MoE router and shared-expert gate):
// blocks [0, nb0) run the first matrix, the rest the second, each through the same body as `gemv_kernel<1, R, U>`.
template <int R, int U, typename OutT>
__global__ void __launch_bounds__(kGemvWarps * IK_WARP)
gemv_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w0, OutT* __restrict__ o0, int64_t N0,
                 const ik_bf16* __restrict__ w1, OutT* __restrict__ o1, int64_t N1, int64_t K) {
    const int64_t per = (int64_t)kGemvWarps * R;
    const int64_t nb0 = (N0 + per - 1) / per;
    if ((int64_t)blockIdx.x < nb0) gemv_body<1, R, U, OutT>(x, w0, o0, N0, K, (int64_t)blockIdx.x);
    else gemv_body<1, R, U, OutT>(x, w1, o1, N1, K, (int64_t)blockIdx.x - nb0);
}

// ---- RMSNorm (+ residual add) folded into the pair GEMV --------------------------------------------------
// rmsnorm_add_matmul_pair: the decode step's `rmsnorm_add(x, res, w) -> (normed, sum)` followed by the pair GEMV on `normed`
// (the MoE router + shared-expert gate, the GDN b / a gates) as one launch. Every block recomputes the row's norm
// in its own prologue and keeps the normalised row in LDS, where the GEMV reads it instead of global memory; block 0
// alone stores `sum` (the residual stream the next layer carries) and `normed` (the other readers of the row), which
// are different buffers from the `x` / `res` every block reads, so no block sees a partly written input.
//
// The result is bitwise the two launches it replaces: the prologue is rmsnorm_kernel<true, true> statement for statement
// (the same 8-element chunks per thread, the same bf16 rounding of the sum before it is squared and normalised, the same
// ik_block_sum over the same 256 threads: kGemvWarps * IK_WARP is the block size launch_rows picks for any K >= 2048, and for
// a narrower row the extra threads contribute exact zeros), and the GEMV below is gemv_body<1, R, U> with its x read from LDS.
constexpr int kNormMaxK = 8192;  // the LDS row is K * 2 bytes: 16 KB at most

// rmsnorm_kernel<true, true> for one row: `xs` ends up holding the normalised row as bf16 chunks. Every thread of the
// block must call it (it holds block barriers).
__device__ __forceinline__ void norm_add_stage(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res,
                                               const ik_bf16* __restrict__ nw, ik_bf16* __restrict__ normed,
                                               ik_bf16* __restrict__ sum, uint4* xs, float* red, int64_t K, float eps,
                                               float woff, bool writer) {
    const int64_t nv = K / 8;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
        float f[8], g[8];
        ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(x) + i), f);
        ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(res) + i), g);
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] += g[k];
        const uint4 p = ik_pack8(f);
        if (writer) reinterpret_cast<uint4*>(sum)[i] = p;
        xs[i] = p;  // the rounded sum, until the second loop replaces it with the normalised row
        ik_unpack8(p, f);  // continue from the bf16-rounded sum
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
    }
    ss = ik_block_sum<true>(ss, red);
    const float r = rsqrtf(ss / (float)K + eps);
    for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
        float f[8], g[8];
        ik_unpack8(xs[i], f);
        ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(nw) + i), g);
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);
        const uint4 p = ik_pack8(f);
        xs[i] = p;
        if (writer) reinterpret_cast<uint4*>(normed)[i] = p;
    }
    IK_LDS_BARRIER();  // the whole normalised row is in LDS before any warp reads it
}

// gemv_body<1, R, U, ik_bf16> on the normalised row in LDS. The first batch of weight loads is issued before the
// prologue (the weights do not depend on it), so their DRAM latency hides behind the norm instead of following it. Every warp,
// live or not, goes through the prologue: its barriers must see the whole block.
template <int R, int U>
__device__ __forceinline__ void gemv_norm_body(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res,
                                               const ik_bf16* __restrict__ nw, ik_bf16* __restrict__ normed,
                                               ik_bf16* __restrict__ sum, float eps, float woff,
                                               const ik_bf16* __restrict__ w, ik_bf16* __restrict__ out, int64_t N,
                                               int64_t K, int64_t blk, bool writer, uint4* xs, float* red) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = (blk * kGemvWarps + warp) * R;
    const bool live = n0 < N;
    const int64_t nv = K / 8;
    const uint4* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < N ? n0 + r : (N > 0 ? N - 1 : 0);  // tail rows re-read the last row; their result is dropped
        wr[r] = reinterpret_cast<const uint4*>(w + n * K);
    }
    uint4 wv[R][U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int64_t idx = (int64_t)u * IK_WARP + lane;
        const bool ok = live && idx < nv;
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r][u] = ok ? ik_ldcs(wr[r] + idx) : make_uint4(0, 0, 0, 0);
    }
    norm_add_stage(x, res, nw, normed, sum, xs, red, K, eps, woff, writer);
    if (!live) return;  // after the prologue's barriers
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int64_t base = 0; base < nv; base += (int64_t)IK_WARP * U) {
        if (base != 0) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t idx = base + (int64_t)u * IK_WARP + lane;
                const bool ok = idx < nv;
#pragma unroll
                for (int r = 0; r < R; ++r) wv[r][u] = ok ? ik_ldcs(wr[r] + idx) : make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
            float xf[8];
            ik_unpack8(ok ? xs[idx] : make_uint4(0, 0, 0, 0), xf);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float wf[8];
                ik_unpack8(wv[r][u], wf);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[r] += wf[e] * xf[e];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float s = ik_warp_sum(acc[r]);
        if (lane == 0 && n0 + r < N) store_out(out + n0 + r, s);
    }
}

template <int R, int U>
__global__ void __launch_bounds__(kGemvWarps * IK_WARP)
rmsnorm_add_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res, const ik_bf16* __restrict__ nw,
                        ik_bf16* __restrict__ normed, ik_bf16* __restrict__ sum, float eps, float woff,
                        const ik_bf16* __restrict__ w0, ik_bf16* __restrict__ o0, int64_t N0,
                        const ik_bf16* __restrict__ w1, ik_bf16* __restrict__ o1, int64_t N1, int64_t K) {
    __shared__ float red[32];
    __shared__ uint4 xs[kNormMaxK / 8];
    const int64_t per = (int64_t)kGemvWarps * R;
    const int64_t nb0 = (N0 + per - 1) / per;
    const bool writer = blockIdx.x == 0;
    if ((int64_t)blockIdx.x < nb0)
        gemv_norm_body<R, U>(x, res, nw, normed, sum, eps, woff, w0, o0, N0, K, (int64_t)blockIdx.x, writer, xs, red);
    else
        gemv_norm_body<R, U>(x, res, nw, normed, sum, eps, woff, w1, o1, N1, K, (int64_t)blockIdx.x - nb0, writer, xs, red);
}

// scalar fallback (K % 8 != 0 or unaligned operands): one warp per row, same reduction shape
template <int M, typename OutT>
__global__ void __launch_bounds__(kGemvWarps * IK_WARP)
gemv_scalar_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w, OutT* __restrict__ out, int64_t N,
                   int64_t K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kGemvWarps + warp;
    if (n >= N) return;
    const ik_bf16* wr = w + n * K;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int64_t k = lane; k < K; k += IK_WARP) {
        const float wf = ik_b2f(wr[k]);
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] += wf * ik_b2f(x[m * K + k]);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) store_out(out + m * N + n, s);
    }
}

struct GemvCfg {
    int r, u;
};

// the (R,U) variants gemv_vec instantiates for M == 1 (M > 1 always uses (2,4))
constexpr bool gemv_instantiated(int r, int u) {
    return (r == 2 && (u == 2 || u == 4 || u == 8)) || (r == 1 && (u == 2 || u == 4 || u == 8 || u == 16)) ||
           (r == 4 && (u == 2 || u == 4 || u == 8));
}

// INTARSIA_HIP_GEMV="R,U" overrides the variant (tuning sweeps); default measured on the R9700 (see docs).
GemvCfg gemv_cfg() {
    static const GemvCfg cfg = [] {
        GemvCfg c{1, 8};
        if (const char* e = getenv("INTARSIA_HIP_GEMV")) {
            int r = 0, u = 0;
            // only the instantiated (R,U) set is accepted (see gemv_vec); anything else keeps the default
            if (sscanf(e, "%d,%d", &r, &u) == 2 && gemv_instantiated(r, u))
                c = {r, u};
            else
                fprintf(stderr, "intarsia-hip: INTARSIA_HIP_GEMV=%s is not an instantiated variant, using %d,%d\n", e, c.r, c.u);
        }
        return c;
    }();
    return cfg;
}

// returns false when no instantiated variant matches (nothing was launched)
template <int M, typename OutT>
bool gemv_vec(const ik_bf16* x, const ik_bf16* w, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    GemvCfg c = gemv_cfg();
    if (M > 1) c = {2, 4};  // multi-row decode is rare; keep the instantiation count down
    const unsigned block = kGemvWarps * IK_WARP;
#define IK_GEMV_CASE(RV, UV)                                                                                     \
    if (c.r == RV && c.u == UV) {                                                                                \
        const unsigned grid = (unsigned)((N + (int64_t)kGemvWarps * RV - 1) / ((int64_t)kGemvWarps * RV));         \
        gemv_kernel<M, RV, UV, OutT><<<grid, block, 0, s>>>(x, w, out, N, K);                                    \
        return true;                                                                                             \
    }
    IK_GEMV_CASE(2, 4)
    if constexpr (M == 1) {
        IK_GEMV_CASE(1, 2) IK_GEMV_CASE(1, 4) IK_GEMV_CASE(1, 8) IK_GEMV_CASE(1, 16) IK_GEMV_CASE(2, 2) IK_GEMV_CASE(2, 8)
        IK_GEMV_CASE(4, 2) IK_GEMV_CASE(4, 4) IK_GEMV_CASE(4, 8)
    }
#undef IK_GEMV_CASE
    return false;
}

template <int M, typename OutT>
bool gemv_dispatch(const ik_bf16* x, const ik_bf16* w, OutT* out, int64_t N, int64_t K, bool vec, hipStream_t s) {
    if (vec) return gemv_vec<M, OutT>(x, w, out, N, K, s);
    const unsigned grid = (unsigned)((N + kGemvWarps - 1) / kGemvWarps);
    gemv_scalar_kernel<M, OutT><<<grid, kGemvWarps * IK_WARP, 0, s>>>(x, w, out, N, K);
    return true;
}

template <typename OutT>
int gemv_launch(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M < 1 || M > 4) return IK_ERR_SHAPE;
    const bool vec = K % 8 == 0 && ik_aligned16(x->data) && ik_aligned16(w->data);
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const ik_bf16* wp = (const ik_bf16*)w->data;
    OutT* op = (OutT*)out->data;
    bool launched = false;
    switch (M) {
        case 1: launched = gemv_dispatch<1, OutT>(xp, wp, op, N, K, vec, s); break;
        case 2: launched = gemv_dispatch<2, OutT>(xp, wp, op, N, K, vec, s); break;
        case 3: launched = gemv_dispatch<3, OutT>(xp, wp, op, N, K, vec, s); break;
        default: launched = gemv_dispatch<4, OutT>(xp, wp, op, N, K, vec, s); break;
    }
    if (!launched) return IK_ERR_INVALID_ARG;  // never report OK for a launch that did not happen
    return ik_launch_status();
}

// ---- row-exact multi-row GEMV (matmul_bf16_gemv_rows, logits_bf16_gemv_rows; lane hb, batched decode) -----------------
// M = 1..kRowsMax rows in one launch, row m bitwise what the one-row GEMV above computes on row m alone. The kernels already
// take M as a template argument and every row's lane order (idx ascending per lane, one warp-sum) does not depend on it; the
// only thing the ordinary launcher refuses past M = 4 is the instantiation. The activations are read from global memory (no
// shared-memory stage), so there is no K limit either. Tested bitwise for M = 1..16 in tests/rows_exact.rs (G-B).
constexpr int kRowsMax = 16;

template <typename OutT>
int gemv_rows_launch(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M < 1 || M > kRowsMax) return IK_ERR_SHAPE;
    const bool vec = K % 8 == 0 && ik_aligned16(x->data) && ik_aligned16(w->data);
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const ik_bf16* wp = (const ik_bf16*)w->data;
    OutT* op = (OutT*)out->data;
    bool launched = false;
    switch (M) {
#define IK_ROWS_CASE(MV) \
    case MV: launched = gemv_dispatch<MV, OutT>(xp, wp, op, N, K, vec, s); break;
        IK_ROWS_CASE(1) IK_ROWS_CASE(2) IK_ROWS_CASE(3) IK_ROWS_CASE(4) IK_ROWS_CASE(5) IK_ROWS_CASE(6) IK_ROWS_CASE(7)
        IK_ROWS_CASE(8) IK_ROWS_CASE(9) IK_ROWS_CASE(10) IK_ROWS_CASE(11) IK_ROWS_CASE(12) IK_ROWS_CASE(13) IK_ROWS_CASE(14)
        IK_ROWS_CASE(15) IK_ROWS_CASE(16)
#undef IK_ROWS_CASE
        default: return IK_ERR_SHAPE;
    }
    if (!launched) return IK_ERR_INVALID_ARG;
    return ik_launch_status();
}

// ---- row-exact multi-row GEMV for the lm_head (logits_bf16_gemv_rows, M = kLogitsWideFrom..kRowsMax) -----------------------
// gemv_kernel<M, 2, 4> hoists the x loads and unpacks of all M rows next to U = 4 weight loads per row, and spills from M = 9
// (256 VGPR, 16..173 spilled; docs/results/2026-10-25-mr-hip-logits-rows.md: 4.5 ms at M = 8, 7.8 at 12, 16.7 at 16 on the
// 248320 x 5120 head, against 4.2 for one row). This kernel computes exactly the same numbers per (output row, activation row):
// the lane walks idx = lane, lane + 32, ... ascending, 8 fma per idx in element order, then the same warp-sum. Only the register
// shape differs: a warp computes kLogitsWideR = 4 weight rows (the weights of an idx are unpacked once for all M rows, and each
// loaded activation row feeds 4 of them) and each activation row is loaded, unpacked and consumed behind a scheduler barrier,
// so the registers do not grow with M (192 VGPR at M = 16, no spill, occupancy 8).
constexpr int kLogitsWideR = 4;

template <int M, typename OutT>
__global__ void __launch_bounds__(kGemvWarps * IK_WARP)
gemv_rows_wide_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w, OutT* __restrict__ out, int64_t N,
                      int64_t K) {
    constexpr int R = kLogitsWideR;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = ((int64_t)blockIdx.x * kGemvWarps + warp) * R;
    if (n0 >= N) return;  // whole warp exits together
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);
    const uint4* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < N ? n0 + r : N - 1;  // tail rows re-read the last row; their result is dropped
        wr[r] = reinterpret_cast<const uint4*>(w + n * K);
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    for (int64_t base = 0; base < nv; base += IK_WARP) {
        const int64_t idx = base + lane;
        const bool ok = idx < nv;
        float wf[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) ik_unpack8(ok ? ik_ldcs(wr[r] + idx) : make_uint4(0, 0, 0, 0), wf[r]);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            __builtin_amdgcn_sched_barrier(0);  // keeps the loads of row m + 1 from being hoisted above the fmas of row m
            float xf[8];
            ik_unpack8(ok ? ik_ldg(x8 + m * nv + idx) : make_uint4(0, 0, 0, 0), xf);
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[r][m] += wf[r][e] * xf[e];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float sum = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) store_out(out + m * N + n0 + r, sum);
        }
}

template <int M, typename OutT>
bool gemv_rows_wide_go(const ik_bf16* x, const ik_bf16* w, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + (int64_t)kGemvWarps * kLogitsWideR - 1) / ((int64_t)kGemvWarps * kLogitsWideR));
    gemv_rows_wide_kernel<M, OutT><<<grid, kGemvWarps * IK_WARP, 0, s>>>(x, w, out, N, K);
    return true;
}

// M rows from kLogitsWideFrom up (vector operands only; anything else, and fewer rows, take gemv_rows_launch)
constexpr int kLogitsWideFrom = 9;

template <typename OutT>
int logits_rows_launch(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    const bool vec = K % 8 == 0 && ik_aligned16(x->data) && ik_aligned16(w->data);
    if (!vec || M < kLogitsWideFrom || M > kRowsMax) return gemv_rows_launch<OutT>(x, w, out, stream);
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const ik_bf16* wp = (const ik_bf16*)w->data;
    OutT* op = (OutT*)out->data;
    bool launched = false;
    switch (M) {
#define IK_LW_CASE(MV) \
    case MV: launched = gemv_rows_wide_go<MV, OutT>(xp, wp, op, N, K, s); break;
        IK_LW_CASE(9) IK_LW_CASE(10) IK_LW_CASE(11) IK_LW_CASE(12) IK_LW_CASE(13) IK_LW_CASE(14) IK_LW_CASE(15) IK_LW_CASE(16)
#undef IK_LW_CASE
        default: return IK_ERR_SHAPE;
    }
    if (!launched) return IK_ERR_INVALID_ARG;
    return ik_launch_status();
}

// ---- WMMA GEMM ---------------------------------------------------------------------------
// D[n, m] = sum_k w[n, k] x[m, k]: the weights are the A operand (rows = output features) and the activations
// the B operand, so the accumulator layout (lane = column m, 8 consecutive rows n) lets each lane write 16
// contiguous bytes of out[m, n0..n0+7].
//
// gfx12 wave32 16x16x16 fragment layout (verified on hardware by the matmul and attention admission sweeps):
//   A (16 rows x 16 k): lane l holds row l % 16, k = (l / 16) * 8 .. +7   (8 bf16 = one uint4)
//   B (16 k x 16 cols): lane l holds col l % 16, k = (l / 16) * 8 .. +7
//   C/D (16 x 16 f32):  lane l holds col l % 16, rows (l / 16) * 8 + i, i = 0..7
typedef __bf16 ik_v8bf __attribute__((ext_vector_type(8)));
typedef float ik_v8f __attribute__((ext_vector_type(8)));

IK_DEVI ik_v8f wmma_bf16(const uint4& a, const uint4& b, ik_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ik_v8bf, a),
                                                             __builtin_bit_cast(ik_v8bf, b), c);
}

constexpr int kBK = 32;
constexpr int kLD = kBK + 8;  // padded LDS row (elements): 80 B

template <int WN, int WM, int TN, int TM, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32)
gemm_wmma_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w, OutT* __restrict__ out, int M, int N,
                 int K, bool out_vec) {
    constexpr int NT = WN * WM * IK_WARP;                      // threads per block
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;        // block tile: weight rows x tokens
    constexpr int CHW = BN * (kBK / 8), CHX = BM * (kBK / 8);  // 16 B chunks per tile
    constexpr int CW = (CHW + NT - 1) / NT, CX = (CHX + NT - 1) / NT;
    __shared__ __align__(16) uint16_t sw[2][BN * kLD];
    __shared__ __align__(16) uint16_t sx[2][BM * kLD];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wn = wave % WN, wm = wave / WN;
    // token blocks vary fastest: the blocks resident together share one weight panel, so the (large) weights are
    // streamed from DRAM once instead of once per token block
    const int m_blk = blockIdx.x * BM, n_blk = blockIdx.y * BN;
    const int nk = K / kBK;

    uint4 rw[CW], rx[CX];
    auto gload = [&](int k0) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int n = n_blk + r;
            rw[c] = (idx < CHW && n < N) ? *reinterpret_cast<const uint4*>(w + (size_t)n * K + k0 + ch * 8)
                                         : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int m = m_blk + r;
            rx[c] = (idx < CHX && m < M) ? *reinterpret_cast<const uint4*>(x + (size_t)m * K + k0 + ch * 8)
                                         : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) *reinterpret_cast<uint4*>(&sw[buf][(idx >> 2) * kLD + (idx & 3) * 8]) = rw[c];
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHX) *reinterpret_cast<uint4*>(&sx[buf][(idx >> 2) * kLD + (idx & 3) * 8]) = rx[c];
        }
    };

    ik_v8f acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    gload(0);
    sstore(0);
    IK_LDS_BARRIER();
    const int lr = lane & 15, lk = (lane >> 4) * 8;
    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) gload((kt + 1) * kBK);
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            uint4 af[TN], bf[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i)
                af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kLD + kk + lk]);
#pragma unroll
            for (int j = 0; j < TM; ++j)
                bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kLD + kk + lk]);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) acc[i][j] = wmma_bf16(af[i], bf[j], acc[i][j]);
        }
        if (kt + 1 < nk) sstore(buf ^ 1);
        IK_LDS_BARRIER();
    }

    const bool vec_ok = out_vec && (N % 8) == 0;  // out_vec: out is 16 B aligned
#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m_blk + wm * TM * 16 + j * 16 + lr;
            const int n = n_blk + wn * TN * 16 + i * 16 + lk;
            if (m >= M || n >= N) continue;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = acc[i][j][e];
            OutT* o = out + (size_t)m * N + n;
            if (vec_ok) {
                if (sizeof(OutT) == 2) {
                    *reinterpret_cast<uint4*>(o) = ik_pack8(v);
                } else {
                    reinterpret_cast<float4*>(o)[0] = make_float4(v[0], v[1], v[2], v[3]);
                    reinterpret_cast<float4*>(o)[1] = make_float4(v[4], v[5], v[6], v[7]);
                }
            } else {
                for (int e = 0; e < 8 && n + e < N; ++e) store_out(o + e, v[e]);
            }
        }
    }
}

// generic fallback: one thread per output element (shapes the tile loader does not take)
template <typename OutT>
__global__ void gemm_simple_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w,
                                   OutT* __restrict__ out, int64_t M, int64_t N, int64_t K) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    const int64_t m = i / N, n = i % N;
    float acc = 0.f;
    for (int64_t k = 0; k < K; ++k) acc += ik_b2f(x[m * K + k]) * ik_b2f(w[n * K + k]);
    store_out(out + i, acc);
}

struct GemmCfg {
    int tile;  // 0 = auto
};

GemmCfg gemm_cfg() {
    static const GemmCfg cfg = [] {
        GemmCfg c{0};
        if (const char* e = getenv("INTARSIA_HIP_GEMM_TILE")) c.tile = atoi(e);
        return c;
    }();
    return cfg;
}

template <typename OutT>
int gemm_launch(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M > 0x7fffffffLL || K > 0x7fffffffLL || N > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const ik_bf16* wp = (const ik_bf16*)w->data;
    OutT* op = (OutT*)out->data;
    if (K % kBK != 0 || !ik_aligned16(x->data) || !ik_aligned16(w->data)) {
        const int64_t total = M * N;
        gemm_simple_kernel<OutT><<<(unsigned)((total + 255) / 256), 256, 0, s>>>(xp, wp, op, M, N, K);
        return ik_launch_status();
    }
    const bool out_vec = ik_aligned16(out->data);
    // tile (weights x tokens) per M, measured on the R9700 (docs/backends/hip.md); INTARSIA_HIP_GEMM_TILE=<id> forces one
    int tile = gemm_cfg().tile;
    if (tile == 0) tile = M >= 128 ? 1 : 2;
#define IK_GEMM_TILE(WN, WM, TN, TM)                                                                          \
    {                                                                                                        \
        dim3 grid((unsigned)((M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((N + WN * TN * 16 - 1) / (WN * TN * 16))); \
        gemm_wmma_kernel<WN, WM, TN, TM, OutT><<<grid, WN * WM * 32, 0, s>>>(xp, wp, op, (int)M, (int)N, (int)K, out_vec); \
    }
    switch (tile) {
        case 1: IK_GEMM_TILE(4, 2, 2, 4) break;  // 128 x 128, 8 waves of 32 x 64
        case 2: IK_GEMM_TILE(4, 2, 2, 2) break;  // 128 x 64, 8 waves of 32 x 32
        case 3: IK_GEMM_TILE(4, 2, 2, 1) break;  // 128 x 32, 8 waves of 32 x 16
        case 4: IK_GEMM_TILE(2, 2, 2, 1) break;  // 64 x 32, 4 waves of 32 x 16
        case 5: IK_GEMM_TILE(2, 2, 4, 4) break;  // 128 x 128, 4 waves of 64 x 64
        case 6: IK_GEMM_TILE(2, 4, 4, 2) break;  // 128 x 128, 8 waves of 64 x 32
        case 7: IK_GEMM_TILE(2, 2, 4, 2) break;  // 128 x 64, 4 waves of 64 x 32
        case 8: IK_GEMM_TILE(4, 2, 4, 2) break;  // 256 x 64, 8 waves of 64 x 32
        default: return IK_ERR_INVALID_ARG;
    }
#undef IK_GEMM_TILE
    return ik_launch_status();
}

// ---- shared validation -------------------------------------------------------------------
int check_mm(const ik_tensor* x, const ik_tensor* w, const ik_tensor* out, int out_dtype) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, out_dtype, 2));
    if (x->shape[1] != w->shape[1] || out->shape[0] != x->shape[0] || out->shape[1] != w->shape[0]) return IK_ERR_SHAPE;
    return IK_OK;
}

int mm(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream, bool f32_out, bool gemv) {
    IK_TRY(check_mm(x, w, out, f32_out ? IK_DTYPE_F32 : IK_DTYPE_BF16));
    if (ik_numel(out) == 0) return IK_OK;
    if (x->shape[1] == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (f32_out ? 4 : 2), IK_STREAM(stream)) == hipSuccess
                   ? IK_OK
                   : IK_ERR_LAUNCH;
    if (gemv) return f32_out ? gemv_launch<float>(x, w, out, stream) : gemv_launch<ik_bf16>(x, w, out, stream);
    return f32_out ? gemm_launch<float>(x, w, out, stream) : gemm_launch<ik_bf16>(x, w, out, stream);
}

// the row-exact twin of mm(..., gemv = true): M = 1..kRowsMax
int mm_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream, bool f32_out) {
    IK_TRY(check_mm(x, w, out, f32_out ? IK_DTYPE_F32 : IK_DTYPE_BF16));
    if (ik_numel(out) == 0) return IK_OK;
    if (x->shape[1] == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (f32_out ? 4 : 2), IK_STREAM(stream)) == hipSuccess
                   ? IK_OK
                   : IK_ERR_LAUNCH;
    return f32_out ? gemv_rows_launch<float>(x, w, out, stream) : gemv_rows_launch<ik_bf16>(x, w, out, stream);
}

// the lm_head's row-exact twin (f32 logits)
int logits_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    IK_TRY(check_mm(x, w, out, IK_DTYPE_F32));
    if (ik_numel(out) == 0) return IK_OK;
    if (x->shape[1] == 0) return hipMemsetAsync(out->data, 0, ik_numel(out) * 4, IK_STREAM(stream)) == hipSuccess ? IK_OK : IK_ERR_LAUNCH;
    return logits_rows_launch<float>(x, w, out, stream);
}

// ---- pair launcher (M == 1, vector path only) -----------------------------------------------
int pair_check(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, const ik_tensor* o0, const ik_tensor* o1) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w0, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w1, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(o0, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(o1, IK_DTYPE_BF16, 2));
    const int64_t M = x->shape[0], K = x->shape[1];
    if (w0->shape[1] != K || w1->shape[1] != K || o0->shape[0] != M || o1->shape[0] != M ||
        o0->shape[1] != w0->shape[0] || o1->shape[1] != w1->shape[0])
        return IK_ERR_SHAPE;
    return IK_OK;
}

int pair(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* o0, ik_tensor* o1, void* stream) {
    IK_TRY(pair_check(x, w0, w1, o0, o1));
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    // decode only (one row) and the vector path; the registry predicate says the same, this keeps a direct call honest
    if (M != 1 || K % 8 != 0 || !ik_aligned16(x->data) || !ik_aligned16(w0->data) || !ik_aligned16(w1->data))
        return IK_ERR_SHAPE;
    if (N0 == 0 && N1 == 0) return IK_OK;
    if (K == 0) {
        hipStream_t z = IK_STREAM(stream);
        if (hipMemsetAsync(o0->data, 0, ik_numel(o0) * 2, z) != hipSuccess) return IK_ERR_LAUNCH;
        return hipMemsetAsync(o1->data, 0, ik_numel(o1) * 2, z) == hipSuccess ? IK_OK : IK_ERR_LAUNCH;
    }
    constexpr int R = 1, U = 8;  // the default variant of gemv_cfg()
    const int64_t per = (int64_t)kGemvWarps * R;
    const int64_t blocks = (N0 + per - 1) / per + (N1 + per - 1) / per;
    if (blocks > 0x7fffffffLL) return IK_ERR_SHAPE;
    gemv_pair_kernel<R, U, ik_bf16><<<(unsigned)blocks, kGemvWarps * IK_WARP, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)w0->data, (ik_bf16*)o0->data, N0, (const ik_bf16*)w1->data,
        (ik_bf16*)o1->data, N1, K);
    return ik_launch_status();
}

// ---- norm + pair launcher (M == 1, vector path only) ---------------------------------------------
bool overlaps(const void* a, const void* b, size_t bytes) {
    const char *p = (const char*)a, *q = (const char*)b;
    return p < q + bytes && q < p + bytes;
}

int norm_pair(const ik_tensor* x, const ik_tensor* res, const ik_tensor* nw, float eps, float woff, const ik_tensor* w0,
              const ik_tensor* w1, ik_tensor* normed, ik_tensor* sum, ik_tensor* o0, ik_tensor* o1, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(res, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(nw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(normed, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(sum, IK_DTYPE_BF16, 2));
    IK_TRY(pair_check(normed, w0, w1, o0, o1));
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (res->shape[0] != M || res->shape[1] != K || nw->shape[0] != K || normed->shape[0] != M || normed->shape[1] != K ||
        sum->shape[0] != M || sum->shape[1] != K)
        return IK_ERR_SHAPE;
    // decode only (one row), the vector path, and the row must fit the LDS staging; the registry predicate says the same
    if (M != 1 || K < 8 || K % 8 != 0 || K > kNormMaxK) return IK_ERR_SHAPE;
    if (!ik_aligned16(x->data) || !ik_aligned16(res->data) || !ik_aligned16(nw->data) || !ik_aligned16(normed->data) ||
        !ik_aligned16(sum->data) || !ik_aligned16(w0->data) || !ik_aligned16(w1->data))
        return IK_ERR_SHAPE;
    // block 0 stores `normed` and `sum` while the other blocks still read `x` and `res`
    const size_t rb = (size_t)K * 2;
    if (overlaps(normed->data, x->data, rb) || overlaps(normed->data, res->data, rb) || overlaps(sum->data, x->data, rb) ||
        overlaps(sum->data, res->data, rb) || overlaps(normed->data, sum->data, rb))
        return IK_ERR_INVALID_ARG;
    if (N0 == 0 && N1 == 0) return IK_ERR_SHAPE;  // (the norm would still have to be stored: not a case the model produces)
    constexpr int R = 1, U = 8;  // the default variant of gemv_cfg(), as pair() uses
    const int64_t per = (int64_t)kGemvWarps * R;
    const int64_t blocks = (N0 + per - 1) / per + (N1 + per - 1) / per;
    if (blocks > 0x7fffffffLL) return IK_ERR_SHAPE;
    rmsnorm_add_pair_kernel<R, U><<<(unsigned)blocks, kGemvWarps * IK_WARP, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)nw->data, (ik_bf16*)normed->data,
        (ik_bf16*)sum->data, eps, woff, (const ik_bf16*)w0->data, (ik_bf16*)o0->data, N0, (const ik_bf16*)w1->data,
        (ik_bf16*)o1->data, N1, K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_pair_bf16_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* out0,
                             ik_tensor* out1, void* stream) {
    return pair(x, w0, w1, out0, out1, stream);
}
int ik_rmsnorm_add_matmul_pair_bf16_gemv(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps,
                                         float w_offset, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* normed,
                                         ik_tensor* sum, ik_tensor* out0, ik_tensor* out1, void* stream) {
    return norm_pair(x, res, w, eps, w_offset, w0, w1, normed, sum, out0, out1, stream);
}
int ik_matmul_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, false, false);
}
int ik_matmul_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, false, true);
}
int ik_logits_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, true, false);
}
int ik_logits_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, true, true);
}
int ik_matmul_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm_rows(x, w, out, stream, false);
}
int ik_logits_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return logits_rows(x, w, out, stream);
}

}  // extern "C"
