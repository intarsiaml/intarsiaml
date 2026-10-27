// matmul_mxfp4 W4A8 decode (lane W4A8): out[M,N] = q8(x)[M,K] @ W^T, M = 1..16, with
//   W[n,k]    = e2m1(code) * 2^(scale[n,k/32] - 127)           (OCP MXFP4, layout as mxfp4.hip)
//   q8(x)[m]  = xq[m,k] * xs[m]                                 (dynamic per-token e4m3, ik_mx_a8.h)
// One kernel family serves a solo token and batched rows: per (output n, row m) the lane walks the groups u = lane, lane + 32,
// ... in order, t = v_dot4_f32_fp8_fp8 chain over the group's 32 products (a fixed order), acc = fma(t, 2^(sw - 127), acc),
// then the xor-butterfly warp sum and the multiplication by xs[m]. Nothing in it depends on M, on the rows a warp computes
// together, or on a row's position, so every output row is bitwise the M == 1 call on that row alone, and the fused pair is
// bitwise two single calls. The `act` argument is accepted (the registered op carries it) but not read: this IS the fp8 mode.
// Structure and register shaping as mxfp4_i8_rows_kernel (mxfp4.hip): R = 2 weight rows per warp share the activation loads,
// each activation row is consumed behind a scheduler barrier so the registers do not grow with M, the next XP (row, group)
// activation steps are in flight while the current dots run. No LDS in the pre-pass variant; the fused variant
// (mxfp4_a8_rows_fused_kernel, INTARSIA_HIP_MXFP4_A8_FUSE, default for M == 1) quantises the rows inside each block into LDS and
// runs the same body, so its outputs are bitwise the pre-pass variant's.
#include "ik_mx_a8.h"

#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr int kWarps = 8;
constexpr int kG = 32;      // elements per e8m0 scale
constexpr int kRowsR = 2;   // weight rows per warp
constexpr int kXP = 2;      // activation steps in flight

// M activation rows x the rows [0, N0) of (w0, s0, o0) then [0, N1) of (w1, s1, o1); N1 == 0: a single matmul.
// xq: u8 [M, K] as uint4 (32 bytes = 2 uint4 per group), xs: f32 [M]; outputs [M, N0] / [M, N1] row-major.
// the dot / accumulate / store loop shared by the pre-pass and the fused kernel; xq / xs are global (pre-pass) or LDS (fused)
// The per-warp row pointers of a block: weight rows n0, n0 + 1 of the concatenation of both matrices.
template <typename OutT>
struct A8Rows {
    int lane, G;
    int64_t n0, N;
    bool active;  // false: the warp has no rows (n0 >= N)
    const uint4* wr[kRowsR];
    const uint8_t* sr[kRowsR];
    OutT* op[kRowsR];
    int64_t ostride[kRowsR];
};

template <typename OutT>
__device__ __forceinline__ A8Rows<OutT> a8_rows_setup(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ s0,
                                                      OutT* __restrict__ o0, int64_t N0, const uint8_t* __restrict__ w1,
                                                      const uint8_t* __restrict__ s1, OutT* __restrict__ o1, int64_t N1, int K) {
    constexpr int R = kRowsR;
    A8Rows<OutT> p;
    // the warp index is wave-uniform: readfirstlane keeps the row pointers below in SGPRs
    const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / IK_WARP);
    p.lane = threadIdx.x % IK_WARP;
    p.N = N0 + N1;
    p.n0 = ((int64_t)blockIdx.x * kWarps + warp) * R;
    p.active = p.n0 < p.N;
    p.G = K / kG;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = p.n0 + r < p.N ? p.n0 + r : p.N - 1;  // a tail row past N re-reads the last row and is not stored
        const bool first = n < N0;
        const int64_t rr = first ? n : n - N0;
        p.wr[r] = reinterpret_cast<const uint4*>((first ? w0 : w1) + rr * (K / 2));
        p.sr[r] = (first ? s0 : s1) + rr * p.G;
        p.op[r] = (first ? o0 : o1) + rr;
        p.ostride[r] = first ? N0 : N1;  // output row stride of this weight row's matrix
    }
    return p;
}

// The first weight group (u = lane) of the warp's two rows, loaded ahead of the activation quantisation of the fused kernel so
// its DRAM latency overlaps the quantisation and the barrier. Loads exactly what the first iteration of the body would load.
struct A8First {
    uint4 wv[kRowsR];
    uint8_t sc[kRowsR];
};
template <typename OutT>
__device__ __forceinline__ A8First a8_first_load(const A8Rows<OutT>& p) {
    A8First f;
#pragma unroll
    for (int r = 0; r < kRowsR; ++r) {
        f.wv[r] = make_uint4(0, 0, 0, 0);
        f.sc[r] = 0;
    }
    if (p.active && p.lane < p.G) {
#pragma unroll
        for (int r = 0; r < kRowsR; ++r) {
            f.wv[r] = ik_ldcs(p.wr[r] + p.lane);
            f.sc[r] = p.sr[r][p.lane];
        }
    }
    return f;
}

// the dot / accumulate / store loop shared by the pre-pass and the fused kernel; xq / xs are global (pre-pass) or LDS (fused).
// PRE: the first group of this lane was already loaded into `first` (a8_first_load) by the fused kernel.
template <int M, typename OutT, bool PRE>
__device__ __forceinline__ void a8_rows_body(const A8Rows<OutT>& p, const A8First& first, const uint4* __restrict__ xq,
                                             const float* __restrict__ xs, int K) {
    constexpr int R = kRowsR;
    constexpr int D = kXP < M ? kXP : M;
    static_assert(D >= 1, "kXP >= 1");
    if (!p.active) return;  // whole warp exits together
    const int lane = p.lane, G = p.G;
    const int64_t n0 = p.n0, N = p.N;
    const uint4* const* wr = p.wr;
    const uint8_t* const* sr = p.sr;
    OutT* const* op = p.op;
    const int64_t* ostride = p.ostride;
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    // activation queue over the flattened (group, row) steps; a step past the last group re-reads a valid one (unused)
    uint4 qa[D], qc[D];
    if (lane < G) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int64_t gi = (int64_t)d * G + lane;
            qa[d] = xq[2 * gi];
            qc[d] = xq[2 * gi + 1];
        }
    }
    for (int u = lane; u < G; u += IK_WARP) {
        uint32_t wd[R][8];
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            uint4 wv;
            uint8_t sc;
            if (PRE && u == lane) {
                wv = first.wv[r];
                sc = first.sc[r];
            } else {
                wv = ik_ldcs(wr[r] + u);
                sc = sr[r][u];
            }
            s[r] = mx_a8_e8m0(sc);
            const uint2 a = mx_a8_e2m1x8_to_e4m3(wv.x), b = mx_a8_e2m1x8_to_e4m3(wv.y);
            const uint2 c = mx_a8_e2m1x8_to_e4m3(wv.z), d = mx_a8_e2m1x8_to_e4m3(wv.w);
            wd[r][0] = a.x; wd[r][1] = a.y; wd[r][2] = b.x; wd[r][3] = b.y;
            wd[r][4] = c.x; wd[r][5] = c.y; wd[r][6] = d.x; wd[r][7] = d.y;
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            __builtin_amdgcn_sched_barrier(0);  // one row at a time: the registers do not grow with M
            const uint4 xa = qa[0], xc = qc[0];
#pragma unroll
            for (int d = 0; d + 1 < D; ++d) { qa[d] = qa[d + 1]; qc[d] = qc[d + 1]; }
            {
                const int mm = m + D < M ? m + D : m + D - M;
                const int uw = m + D < M ? u : u + IK_WARP;
                const int uu = uw < G ? uw : u;
                const int64_t gi = (int64_t)mm * G + uu;
                qa[D - 1] = xq[2 * gi];
                qc[D - 1] = xq[2 * gi + 1];
            }
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r][m] = fmaf(mx_a8_dot32(wd[r], xa, xc), s[r], acc[r][m]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float xsm = xs[m];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float t = ik_warp_sum(acc[r][m]) * xsm;
            if (lane == 0 && n0 + r < N) ik_store(op[r] + (int64_t)m * ostride[r], t);
        }
    }
}

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_a8_rows_kernel(const uint4* __restrict__ xq, const float* __restrict__ xs, const uint8_t* __restrict__ w0,
                     const uint8_t* __restrict__ s0, OutT* __restrict__ o0, int64_t N0, const uint8_t* __restrict__ w1,
                     const uint8_t* __restrict__ s1, OutT* __restrict__ o1, int64_t N1, int K) {
    const A8Rows<OutT> p = a8_rows_setup<OutT>(w0, s0, o0, N0, w1, s1, o1, N1, K);
    const A8First none{};
    a8_rows_body<M, OutT, false>(p, none, xq, xs, K);
}

// LDS layout of the fused kernel (dynamic shared memory, the scratch layout of ik_mx_a8.h without the slack): xq u8 [M, K] at
// byte 0, xs f32 [M] at byte al16(M * K); size al16(M * K) + al16(4 * M). The static float red[kWarps] comes on top (32 bytes).
// Used only when the dynamic size is <= kFuseLdsMax (K = 5120: M <= 11, K = 6144: M <= 9, K = 17408: M <= 3, K = 2080: M <= 16).
constexpr int kFuseLdsMax = 60000;
__host__ __device__ inline size_t a8_fuse_xs_off(int M, int K) { return ((size_t)M * (size_t)K + 15) & ~(size_t)15; }
__host__ __device__ inline size_t a8_fuse_lds_bytes(int M, int K) { return a8_fuse_xs_off(M, K) + (((size_t)M * 4 + 15) & ~(size_t)15); }

// Same as mxfp4_a8_rows_kernel with the pre-pass folded in: each block first quantises the M activation rows into LDS with the
// pre-pass's own helpers (mx_a8_row_scale / mx_a8_row_quant, so xq and xs are bit-identical), then runs the same body reading
// them from LDS. The blocks are 256 threads, as the pre-pass kernel's. Requires K % 32 == 0, x 16-byte aligned.
template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_a8_rows_fused_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, const uint8_t* __restrict__ s0,
                           OutT* __restrict__ o0, int64_t N0, const uint8_t* __restrict__ w1, const uint8_t* __restrict__ s1,
                           OutT* __restrict__ o1, int64_t N1, int K) {
    static_assert(kWarps * IK_WARP == kMxA8QuantThreads, "the quantisation helpers assume the pre-pass block size");
    extern __shared__ uint4 a8_lds[];
    __shared__ float red[kWarps];
    uint8_t* lq = reinterpret_cast<uint8_t*>(a8_lds);
    float* lxs = reinterpret_cast<float*>(lq + a8_fuse_xs_off(M, K));
    const A8Rows<OutT> p = a8_rows_setup<OutT>(w0, s0, o0, N0, w1, s1, o1, N1, K);
    const A8First first = a8_first_load<OutT>(p);  // in flight while the block quantises
    const int C = K >> 3;
#pragma unroll 1
    for (int m = 0; m < M; ++m) {
        const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K);
        const float s = mx_a8_row_scale(xp, C, red);
        if (threadIdx.x == 0) lxs[m] = s;
        mx_a8_row_quant(xp, reinterpret_cast<uint2*>(lq + (size_t)m * K), C, s);
    }
    __syncthreads();  // every block thread takes part above; only now may a warp leave early (no rows in the body)
    a8_rows_body<M, OutT, true>(p, first, reinterpret_cast<const uint4*>(lq), lxs, K);
}

template <int M, typename OutT>
void a8_launch(const uint4* xq, const float* xs, const uint8_t* w0, const uint8_t* s0, OutT* o0, int64_t N0, const uint8_t* w1,
               const uint8_t* s1, OutT* o1, int64_t N1, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N0 + N1 + kWarps * kRowsR - 1) / (kWarps * kRowsR));
    mxfp4_a8_rows_kernel<M, OutT><<<grid, kWarps * IK_WARP, 0, s>>>(xq, xs, w0, s0, o0, N0, w1, s1, o1, N1, (int)K);
}

template <int M, typename OutT>
void a8_launch_fused(const ik_bf16* x, const uint8_t* w0, const uint8_t* s0, OutT* o0, int64_t N0, const uint8_t* w1,
                     const uint8_t* s1, OutT* o1, int64_t N1, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N0 + N1 + kWarps * kRowsR - 1) / (kWarps * kRowsR));
    mxfp4_a8_rows_fused_kernel<M, OutT><<<grid, kWarps * IK_WARP, a8_fuse_lds_bytes(M, (int)K), s>>>(x, w0, s0, o0, N0, w1, s1, o1,
                                                                                                   N1, (int)K);
}

// INTARSIA_HIP_MXFP4_A8_FUSE: quantise the activations inside the GEMV blocks (no pre-pass launch) whenever the LDS size fits.
// Unset (the default): single-row calls only (M == 1: plain decode and the MTP draft pass), where the fused kernel is never slower
// than pre-pass + kernel; the multi-row verify and prefill-tail calls keep the pre-pass (the fused blocks quantise all M rows each,
// which cuts occupancy and loses from about M = 3). "1": every M the LDS fits. "0" (or anything else): never. Read once per process.
static int a8_fuse_mode() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_HIP_MXFP4_A8_FUSE");
        if (!e) return 1;
        return (e[0] == '1' && e[1] == '\0') ? 2 : 0;
    }();
    return v;
}
static bool a8_fuse_takes(int M) { const int m = a8_fuse_mode(); return m == 2 || (m == 1 && M == 1); }

// one pre-pass (activation quantisation), then one launch over the rows of both matrices (N1 == 0: single). The scratch is
// required (IK_ERR_INVALID_ARG when null, misaligned or smaller than mx_a8_scratch_bytes), also when the fused path (which
// does not use it) is selected.
template <typename OutT>
int a8_rows(int M, const ik_bf16* x, const uint8_t* w0, const uint8_t* s0, OutT* o0, int64_t N0, const uint8_t* w1,
            const uint8_t* s1, OutT* o1, int64_t N1, int64_t K, const ik_scratch* scratch, hipStream_t s) {
    if (M < 1 || M > 16) return IK_ERR_SHAPE;
    if (!scratch || !scratch->data || (reinterpret_cast<uintptr_t>(scratch->data) & 15u) ||
        scratch->bytes < (int64_t)mx_a8_scratch_bytes(M, K))
        return IK_ERR_INVALID_ARG;
    // a null or misaligned x is the pre-pass's to reject or to serve: the fused kernel reads x as 16-byte words
    if (a8_fuse_takes(M) && x && !(reinterpret_cast<uintptr_t>(x) & 15u) && K > 0 && K <= 0x7fffffffLL && (K & 7) == 0 &&
        a8_fuse_lds_bytes(M, (int)K) <= (size_t)kFuseLdsMax) {
#define IK_MX_A8F(MM) case MM: a8_launch_fused<MM, OutT>(x, w0, s0, o0, N0, w1, s1, o1, N1, K, s); break
        switch (M) {
            IK_MX_A8F(1); IK_MX_A8F(2); IK_MX_A8F(3); IK_MX_A8F(4); IK_MX_A8F(5); IK_MX_A8F(6); IK_MX_A8F(7); IK_MX_A8F(8);
            IK_MX_A8F(9); IK_MX_A8F(10); IK_MX_A8F(11); IK_MX_A8F(12); IK_MX_A8F(13); IK_MX_A8F(14); IK_MX_A8F(15); IK_MX_A8F(16);
            default: return IK_ERR_SHAPE;
        }
#undef IK_MX_A8F
        return ik_launch_status();
    }
    IK_TRY(mx_a8_quant_launch(x, scratch->data, M, K, s));
    const uint4* xq = reinterpret_cast<const uint4*>(scratch->data);
    const float* xs = reinterpret_cast<const float*>(static_cast<unsigned char*>(scratch->data) + mx_a8_xs_offset(M, K));
#define IK_MX_A8(MM) case MM: a8_launch<MM, OutT>(xq, xs, w0, s0, o0, N0, w1, s1, o1, N1, K, s); break
    switch (M) {
        IK_MX_A8(1); IK_MX_A8(2); IK_MX_A8(3); IK_MX_A8(4); IK_MX_A8(5); IK_MX_A8(6); IK_MX_A8(7); IK_MX_A8(8);
        IK_MX_A8(9); IK_MX_A8(10); IK_MX_A8(11); IK_MX_A8(12); IK_MX_A8(13); IK_MX_A8(14); IK_MX_A8(15); IK_MX_A8(16);
        default: return IK_ERR_SHAPE;
    }
#undef IK_MX_A8
    return ik_launch_status();
}

static int zero_out(const ik_tensor* out, hipStream_t s) {
    return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), s) == hipSuccess ? IK_OK
                                                                                                                 : IK_ERR_LAUNCH;
}

}  // namespace

extern "C" {

// matmul_mxfp4 W4A8, M = 1..16 (the argument conventions of ik_matmul_mxfp4_gemv_rows). scratch >= mx_a8_scratch_bytes(M, K).
int ik_matmul_mxfp4_a8_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                                 ik_tensor* out, const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % kG != 0 || w->shape[1] != K / 2 || scales->shape[0] != N || scales->shape[1] != K / kG || act->shape[0] != 1 ||
        out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (M > 16) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    if (K == 0) return zero_out(out, s);
    if (K > 0x7fffffffLL) return IK_ERR_SHAPE;
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* sp = (const uint8_t*)scales->data;
    return out->dtype == IK_DTYPE_F32
               ? a8_rows((int)M, xp, wp, sp, (float*)out->data, N, (const uint8_t*)nullptr, (const uint8_t*)nullptr,
                         (float*)nullptr, 0, K, scratch, s)
               : a8_rows((int)M, xp, wp, sp, (ik_bf16*)out->data, N, (const uint8_t*)nullptr, (const uint8_t*)nullptr,
                         (ik_bf16*)nullptr, 0, K, scratch, s);
}

// MatmulMxfp4Pair W4A8: out0 = a8(x, w0, s0), out1 = a8(x, w1, s1) as ONE launch after ONE activation quantisation; every
// output row is bitwise the single call's.
int ik_matmul_mxfp4_pair_a8_gemv_rows(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1,
                                      const ik_tensor* s1, const ik_tensor* act, ik_tensor* out0, ik_tensor* out1,
                                      const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w0, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(w1, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(s0, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(s1, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out0 || !out1 || out0->dtype != out1->dtype || (out0->dtype != IK_DTYPE_BF16 && out0->dtype != IK_DTYPE_F32))
        return IK_ERR_DTYPE;
    IK_TRY(ik_check(out0, out0->dtype, 2));
    IK_TRY(ik_check(out1, out1->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (K % kG != 0 || w0->shape[1] != K / 2 || w1->shape[1] != K / 2 || s0->shape[0] != N0 || s1->shape[0] != N1 ||
        s0->shape[1] != K / kG || s1->shape[1] != K / kG || act->shape[0] != 1 || out0->shape[0] != M ||
        out1->shape[0] != M || out0->shape[1] != N0 || out1->shape[1] != N1)
        return IK_ERR_SHAPE;
    if (M == 0 || N0 + N1 == 0) return IK_OK;
    if (M > 16) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    if (K == 0) {
        IK_TRY(zero_out(out0, s));
        return zero_out(out1, s);
    }
    if (K > 0x7fffffffLL) return IK_ERR_SHAPE;
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t *a0 = (const uint8_t*)w0->data, *b0 = (const uint8_t*)s0->data;
    const uint8_t *a1 = (const uint8_t*)w1->data, *b1 = (const uint8_t*)s1->data;
    return out0->dtype == IK_DTYPE_F32
               ? a8_rows((int)M, xp, a0, b0, (float*)out0->data, N0, a1, b1, (float*)out1->data, N1, K, scratch, s)
               : a8_rows((int)M, xp, a0, b0, (ik_bf16*)out0->data, N0, a1, b1, (ik_bf16*)out1->data, N1, K, scratch, s);
}

// solo twins: the ordinary (non-rows_exact) registrations `matmul_mxfp4_a8_gemv` / `matmul_mxfp4_pair_a8_gemv` launch the very same kernel
// family, so a solo row and a batched row are the same instruction sequence
int ik_matmul_mxfp4_a8_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act, ik_tensor* out,
                            const ik_scratch* scratch, void* stream) {
    return ik_matmul_mxfp4_a8_gemv_rows(x, w, scales, act, out, scratch, stream);
}

int ik_matmul_mxfp4_pair_a8_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1, const ik_tensor* s1,
                                 const ik_tensor* act, ik_tensor* out0, ik_tensor* out1, const ik_scratch* scratch, void* stream) {
    return ik_matmul_mxfp4_pair_a8_gemv_rows(x, w0, s0, w1, s1, act, out0, out1, scratch, stream);
}

}  // extern "C"
