// matmul_nvfp4: out[M,N] = x[M,K] @ W^T with W[n,k] = e2m1(code) * e4m3(block_scale[n,k/16]) * global (W4A16).
//   w  u8   [N, K/2]   two codes per byte, element 2i in the low nibble
//   bs e4m3 [N, K/16]  plain row-major (checkpoint layout)
//   gs f32  [1]        weight_scale_2
//   *_gemv  M <= 8: warp-per-output-row, one 16-element block (8 weight bytes + 1 scale byte) per lane-iteration.
//   *_mma   any M: bf16 WMMA, e2m1*e4m3 dequantised to bf16 in shared memory (exact), f32 accumulate (ik_dq_gemm.h).
// Deterministic (fixed reduction order). Activations bf16; output bf16 or f32 (the lm_head writes f32 logits).
// HIP port of kernels/cuda/nvfp4.cu (follows TensorFold's NVFP4 kernels, MIT, Copyright (c) 2026 TensorFold
// contributors, see ik_quant.h and LICENSES/TensorFold-MIT.txt). RDNA4 has no fp4 hardware: the e2m1 codes are looked up in a 16-entry shared
// table (ik_quant.h) and the block scale (e4m3) is decoded with integer arithmetic.
#include "ik_dq_gemm.h"

namespace {

constexpr int kWarps = 8;

// ---- GEMV --------------------------------------------------------------------------------------------------------

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
nvfp4_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                  const float* __restrict__ gs, OutT* __restrict__ out, int64_t N, int K) {
    IK_FP4_LUT(lut);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;  // whole warp exits together
    const uint2* wr = reinterpret_cast<const uint2*>(w + n * (K / 2));
    const uint8_t* sr = bs + n * (K / 16);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int u = lane; u < K / 16; u += IK_WARP) {
        const uint2 wv = ik_ldcs(wr + u);
        const float sc = ik_e4m3_to_f(sr[u]);
        float wf[16];
        ik_fp4x8_to_f32(wv.x, lut, wf);
        ik_fp4x8_to_f32(wv.y, lut, wf + 8);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * 16);
            float xf[16];
            ik_unpack8(xp[0], xf);
            ik_unpack8(xp[1], xf + 8);
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
            acc[m] = fmaf(s, sc, acc[m]);
        }
    }
    const float g = gs[0];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) ik_store(out + (int64_t)m * N + n, s * g);
    }
}

// v2: bit-identical to the kernel above (same lane/iteration/accumulation order), restructured for latency. The weights,
// block scales and activations of the first kPre lane-iterations are requested before the shared table is built, so the
// table fill and its barrier overlap the global-memory latency instead of preceding it (a decode GEMV here is 0.3 to 2 MB,
// i.e. a few microseconds, of which that latency is most). Rows past N clamp to row 0 and are masked at the store, so every
// thread reaches the barrier.
constexpr int kPre = 4;

// XS: the activations are staged in shared memory once per block (one 16-byte copy per thread) instead of every warp
// reading the same few kilobytes from the cache hierarchy; the copy and the table fill share one barrier. M * K * 2 bytes of
// dynamic shared memory (the launcher only picks XS when that is at most 32 KiB).
template <int M, typename OutT, bool XS>
__global__ void __launch_bounds__(kWarps * IK_WARP)
nvfp4_gemv_v2_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                     const float* __restrict__ gs, OutT* __restrict__ out, int64_t N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const uint4* xsm = reinterpret_cast<const uint4*>(smem);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    const bool live = n < N;
    const int64_t nn = live ? n : 0;
    const uint2* wr = reinterpret_cast<const uint2*>(w + nn * (K / 2));
    const uint8_t* sr = bs + nn * (K / 16);
    const int nu = K / 16;
    uint2 wv[kPre];
    uint32_t sb[kPre];
    uint4 xv[XS ? 1 : kPre][M][2];
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        const int u = lane + IK_WARP * j;
        const bool ok = u < nu;
        wv[j] = ok ? ik_ldcs(wr + u) : make_uint2(0, 0);
        sb[j] = ok ? sr[u] : 0u;
        if (!XS) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)(ok ? u : 0) * 16);
                xv[XS ? 0 : j][m][0] = xp[0];
                xv[XS ? 0 : j][m][1] = xp[1];
            }
        }
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    if (XS) {
        uint4* xd = reinterpret_cast<uint4*>(smem);
        for (int i = threadIdx.x; i < M * K / 8; i += blockDim.x) xd[i] = reinterpret_cast<const uint4*>(x)[i];
    }
    __syncthreads();
    if (!live) return;  // whole warp exits together
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    auto step = [&](const uint2& wq, uint32_t sbyte, const uint4 (&xq)[M][2]) {
        const float sc = ik_e4m3_to_f(sbyte);
        float wf[16];
        ik_fp4x8_to_f32(wq.x, lut, wf);
        ik_fp4x8_to_f32(wq.y, lut, wf + 8);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float xf[16];
            ik_unpack8(xq[m][0], xf);
            ik_unpack8(xq[m][1], xf + 8);
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
            acc[m] = fmaf(s, sc, acc[m]);
        }
    };
    auto xload = [&](int u, uint4 (&xq)[M][2]) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            if (XS) {
                xq[m][0] = xsm[(int64_t)m * (K / 8) + u * 2];
                xq[m][1] = xsm[(int64_t)m * (K / 8) + u * 2 + 1];
            } else {
                const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * 16);
                xq[m][0] = xp[0];
                xq[m][1] = xp[1];
            }
        }
    };
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        if (lane + IK_WARP * j >= nu) continue;
        if (XS) {
            uint4 xq[M][2];
            xload(lane + IK_WARP * j, xq);
            step(wv[j], sb[j], xq);
        } else {
            step(wv[j], sb[j], xv[XS ? 0 : j]);
        }
    }
    for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP) {  // K > 2048: the remaining iterations, loaded as used
        uint4 xq[M][2];
        xload(u, xq);
        step(ik_ldcs(wr + u), sr[u], xq);
    }
    const float g = gs[0];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) ik_store(out + (int64_t)m * N + n, s * g);
    }
}

// Shared-expert gate + up + silu_glu in one launch (M == 1, bf16 out): a warp owns row n of BOTH the gate and the up matrix, so
// it reads the activations once and ends with silu(gate_n) * up_n. Each row runs exactly the arithmetic of
// nvfp4_gemv_v2_kernel<1, ik_bf16, false> (same lane/iteration order, same fma chain, scale applied after the warp sum), the
// two results are rounded to bf16 as that kernel's store would, and the epilogue is silu_glu_kernel's: the output bits equal
// those of the three launches this replaces.
__global__ void __launch_bounds__(kWarps * IK_WARP)
nvfp4_glu_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ wg, const uint8_t* __restrict__ bsg,
                 const float* __restrict__ gsg, const uint8_t* __restrict__ wu, const uint8_t* __restrict__ bsu,
                 const float* __restrict__ gsu, ik_bf16* __restrict__ out, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    const bool live = n < N;
    const int64_t nn = live ? n : 0;
    const uint2* wr[2] = {reinterpret_cast<const uint2*>(wg + nn * (K / 2)), reinterpret_cast<const uint2*>(wu + nn * (K / 2))};
    const uint8_t* sr[2] = {bsg + nn * (K / 16), bsu + nn * (K / 16)};
    const int nu = K / 16;
    uint2 wv[2][kPre];
    uint32_t sb[2][kPre];
    uint4 xv[kPre][2];
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        const int u = lane + IK_WARP * j;
        const bool ok = u < nu;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            wv[r][j] = ok ? ik_ldcs(wr[r] + u) : make_uint2(0, 0);
            sb[r][j] = ok ? sr[r][u] : 0u;
        }
        const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)(ok ? u : 0) * 16);
        xv[j][0] = xp[0];
        xv[j][1] = xp[1];
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    __syncthreads();
    if (!live) return;  // whole warp exits together
    float acc[2] = {0.f, 0.f};
    auto step = [&](int r, const uint2& wq, uint32_t sbyte, const uint4 (&xq)[2]) {
        const float sc = ik_e4m3_to_f(sbyte);
        float wf[16];
        ik_fp4x8_to_f32(wq.x, lut, wf);
        ik_fp4x8_to_f32(wq.y, lut, wf + 8);
        float xf[16];
        ik_unpack8(xq[0], xf);
        ik_unpack8(xq[1], xf + 8);
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
        acc[r] = fmaf(s, sc, acc[r]);
    };
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        if (lane + IK_WARP * j >= nu) continue;
        step(0, wv[0][j], sb[0][j], xv[j]);
        step(1, wv[1][j], sb[1][j], xv[j]);
    }
    for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP) {  // beyond the window (K > 2048)
        uint4 xq[2];
        const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)u * 16);
        xq[0] = xp[0];
        xq[1] = xp[1];
        step(0, ik_ldcs(wr[0] + u), sr[0][u], xq);
        step(1, ik_ldcs(wr[1] + u), sr[1][u], xq);
    }
    const float g = ik_b2f(ik_f2b(ik_warp_sum(acc[0]) * gsg[0]));  // what the gate GEMV stores
    const float u = ik_b2f(ik_f2b(ik_warp_sum(acc[1]) * gsu[0]));  // what the up GEMV stores
    if (lane == 0) out[n] = ik_f2b(ik_silu(g) * u);
}

// INTARSIA_HIP_XSTAGE=1 stages the activations in shared memory (when they fit); default 0
bool xstage_enabled() {
    static const bool on = [] {
        const char* e = getenv("INTARSIA_HIP_XSTAGE");
        return e && e[0] == '1';
    }();
    return on;
}

template <int M, typename OutT>
void gemv_run(const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t N, int64_t K,
              hipStream_t s, bool v2) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    const size_t xbytes = (size_t)M * K * 2;
    if (v2 && xstage_enabled() && xbytes <= 32768)
        nvfp4_gemv_v2_kernel<M, OutT, true><<<grid, kWarps * IK_WARP, xbytes, s>>>(x, w, bs, gs, out, N, (int)K);
    else if (v2) nvfp4_gemv_v2_kernel<M, OutT, false><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, bs, gs, out, N, (int)K);
    else nvfp4_gemv_kernel<M, OutT><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, bs, gs, out, N, (int)K);
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t N,
              int64_t K, hipStream_t s, bool v2) {
    switch (M) {
        case 1: gemv_run<1>(x, w, bs, gs, out, N, K, s, v2); break;
        case 2: gemv_run<2>(x, w, bs, gs, out, N, K, s, v2); break;
        case 3: gemv_run<3>(x, w, bs, gs, out, N, K, s, v2); break;
        case 4: gemv_run<4>(x, w, bs, gs, out, N, K, s, v2); break;
        case 5: gemv_run<5>(x, w, bs, gs, out, N, K, s, v2); break;
        case 6: gemv_run<6>(x, w, bs, gs, out, N, K, s, v2); break;
        case 7: gemv_run<7>(x, w, bs, gs, out, N, K, s, v2); break;
        case 8: gemv_run<8>(x, w, bs, gs, out, N, K, s, v2); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

// ---- MMA policy ---------------------------------------------------------------------------------------------------

struct Nvfp4Policy {
    IK_DEVI bool skip(const DqArgs&) { return false; }
    struct State {
        const float* lut;
        float g;
    };
    struct Raw {
        uint2 w;     // 16 fp4 codes
        uint32_t s;  // e4m3 block scale
    };
    IK_DEVI State init(const DqArgs& a) {
        IK_FP4_LUT(lut);
        State s;
        s.lut = lut;
        s.g = a.sc[0];
        return s;
    }
    IK_DEVI Raw load(const DqArgs& a, int64_t n, int k) {
        Raw r;
        r.w = *reinterpret_cast<const uint2*>(a.w + n * (a.K / 2) + k / 2);  // cached: a row is revisited every k step
        
        r.s = a.bs[n * (a.K / 16) + k / 16];
        return r;
    }
    IK_DEVI Raw zero() {
        Raw r;
        r.w = make_uint2(0, 0);
        r.s = 0;
        return r;
    }
    IK_DEVI void dequant(const State& st, const Raw& r, uint4& lo, uint4& hi) {
        const float sc = ik_e4m3_to_f(r.s);
        float f[16];
        ik_fp4x8_to_f32(r.w.x, st.lut, f);
        ik_fp4x8_to_f32(r.w.y, st.lut, f + 8);
#pragma unroll
        for (int e = 0; e < 16; ++e) f[e] *= sc;  // exact: <= 6 significant bits
        lo = ik_pack8(f);
        hi = ik_pack8(f + 8);
    }
    IK_DEVI uint4 xprep(const State&, uint4 v) { return v; }
    IK_DEVI float out_scale(const State& s) { return s.g; }
};

// ---- launchers ---------------------------------------------------------------------------------------------------

// variant: 0 = tensor-core GEMM, 1 = GEMV, 2 = GEMV v2 (same numerics, latency-restructured)
int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs, ik_tensor* out,
           void* stream, int variant) {
    const bool gemv = variant != 0;
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 8));
    IK_TRY(ik_check_a(bs, IK_DTYPE_FP8_E4M3, 2, 1));
    IK_TRY(ik_check_a(gs, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 16 != 0 || w->shape[1] != K / 2 || bs->shape[0] != N || bs->shape[1] != K / 16 || gs->shape[0] != 1 ||
        out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == hipSuccess
                   ? IK_OK
                   : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* bp = (const uint8_t*)bs->data;
    const float* gp = (const float*)gs->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > 8) return IK_ERR_SHAPE;
        return f32o ? gemv_by_m((int)M, xp, wp, bp, gp, (float*)out->data, N, K, s, variant == 2)
                    : gemv_by_m((int)M, xp, wp, bp, gp, (ik_bf16*)out->data, N, K, s, variant == 2);
    }
    DqArgs a;
    a.x = xp;
    a.w = wp;
    a.bs = bp;
    a.sc = gp;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    if (f32o) dq_gemm_run<Nvfp4Policy>(a, (float*)out->data, s);
    else dq_gemm_run<Nvfp4Policy>(a, (ik_bf16*)out->data, s);
    return ik_launch_status();
}

// matmul_nvfp4_gemv_rows (HIP only: declared in ik_hip_abi.h; lane hb, batched decode): M = 1..16 rows, row m bitwise the
// one-row ik_matmul_nvfp4_gemv_v2 on row m alone (tests/rows_exact.rs). The v2 GEMV takes 1..8 rows per launch, but its time
// is not linear in M: on the R9700 the 248320 x 2048 head takes 410 us at M = 1, 759 us at M = 4 and 3214 us at M = 8 (the
// register footprint of the M = 8 instantiation), so rows go in near-equal groups of at most kRowsGroup = 4: one weight
// pass per group, 16 rows = 4 launches of 4 (about 3.0 ms against 6.4 ms for two groups of 8). A row's arithmetic (lane
// order, one fma chain per 16-element block, one warp sum) does not depend on the launch's M, so the grouping is invisible.
constexpr int kRowsGroup = 4;
int launch_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs, ik_tensor* out,
                void* stream) {
    const int64_t M = x->shape[0];
    if (M < 1 || M > 16) return IK_ERR_SHAPE;
    if (M <= kRowsGroup) return launch(x, w, bs, gs, out, stream, 2);
    // shapes are validated by launch() on each group; the views below carry the same dtypes and a row-sliced shape
    const int64_t K = x->shape[1], N = w->shape[0];
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check(out, out->dtype, 2));
    if (out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    const int64_t esz = out->dtype == IK_DTYPE_F32 ? 4 : 2;
    const int64_t groups = (M + kRowsGroup - 1) / kRowsGroup;
    int64_t r0 = 0;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t m = (M - r0 + (groups - g) - 1) / (groups - g);
        ik_tensor xv = *x, ov = *out;
        xv.shape[0] = m;
        xv.data = (char*)x->data + r0 * K * 2;
        ov.shape[0] = m;
        ov.data = (char*)out->data + r0 * N * esz;
        IK_TRY(launch(&xv, w, bs, gs, &ov, stream, 2));
        r0 += m;
    }
    return IK_OK;
}

int glu(const ik_tensor* x, const ik_tensor* wg, const ik_tensor* bsg, const ik_tensor* gsg, const ik_tensor* wu,
        const ik_tensor* bsu, const ik_tensor* gsu, ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    for (const ik_tensor* w : {wg, wu}) IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 8));
    for (const ik_tensor* b : {bsg, bsu}) IK_TRY(ik_check_a(b, IK_DTYPE_FP8_E4M3, 2, 1));
    for (const ik_tensor* g : {gsg, gsu}) IK_TRY(ik_check_a(g, IK_DTYPE_F32, 1, 4));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = wg->shape[0];
    if (M != 1 || K % 16 != 0 || wu->shape[0] != N || wg->shape[1] != K / 2 || wu->shape[1] != K / 2 ||
        bsg->shape[0] != N || bsu->shape[0] != N || bsg->shape[1] != K / 16 || bsu->shape[1] != K / 16 ||
        gsg->shape[0] != 1 || gsu->shape[0] != 1 || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (N == 0) return IK_OK;
    if (K == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * 2, IK_STREAM(stream)) == hipSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (K > 0x7fffffffLL || N > 0x7fffffffLL * kWarps) return IK_ERR_SHAPE;
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    nvfp4_glu_kernel<<<grid, kWarps * IK_WARP, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const uint8_t*)wg->data, (const uint8_t*)bsg->data, (const float*)gsg->data,
        (const uint8_t*)wu->data, (const uint8_t*)bsu->data, (const float*)gsu->data, (ik_bf16*)out->data, N, (int)K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_nvfp4_glu_gemv(const ik_tensor* x, const ik_tensor* w_gate, const ik_tensor* block_scales_gate,
                             const ik_tensor* global_scale_gate, const ik_tensor* w_up, const ik_tensor* block_scales_up,
                             const ik_tensor* global_scale_up, ik_tensor* out, void* stream) {
    return glu(x, w_gate, block_scales_gate, global_scale_gate, w_up, block_scales_up, global_scale_up, out, stream);
}
int ik_matmul_nvfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                         const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, 1);
}
int ik_matmul_nvfp4_gemv_v2(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                            const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, 2);
}
int ik_matmul_nvfp4_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                              const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch_rows(x, w, block_scales, global_scale, out, stream);
}
int ik_matmul_nvfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                        const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, 0);
}

}  // extern "C"
