// matmul_mxfp4: out[M,N] = q(x)[M,K] @ W^T with W[n,k] = e2m1(code) * 2^(scale[n,k/32] - 127) (OCP MXFP4, AMD Quark).
//   w     u8  [N, K/2]   two codes per byte, element 2i in the low nibble
//   scale u8  [N, K/32]  e8m0, plain row-major (checkpoint layout)
//   act   f32 [1]        0: q(x) = x (W4A16); 1: q(x) = dynamic MXFP4 QDQ of x per 32-element group along K (W4A4, the
//                        scheme Quark declares; scale rule "even", see crates/intarsia-core/src/quant/mxfp4.rs)
//   *_gemv  M <= 8: warp-per-output-row, one 32-element group (16 weight bytes + 1 scale byte + 32 activations) per
//           lane-iteration. With a scratch (the registered kernel declares one) the activations are QDQ'd once by the
//           pre-pass below; without, every warp QDQs the groups it reads (each group once per output row: N times).
//   *_mma   any M: the activations are QDQ'd (or copied) once into scratch (bf16 [M, K]), then the dequantise-to-bf16 WMMA
//           GEMM of ik_dq_gemm.h runs on them (e2m1 * 2^e is exact in bf16; f32 accumulate).
// Deterministic (fixed reduction order). Activations bf16; output bf16 or f32. Same structure as nvfp4.hip.
#include "ik_dq_gemm.h"

namespace {

constexpr int kWarps = 8;
constexpr int kG = 32;  // elements per e8m0 scale

// e8m0 byte -> 2^(b - 127); 0 -> 2^-127 (f32 subnormal), 255 -> NaN
IK_DEVI float mx_e8m0(uint32_t b) {
    return b == 0u ? __uint_as_float(0x00400000u) : (b == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(b << 23));
}

// Quark "even" shared scale of a group with largest magnitude amax (>= 0): biased e8m0 byte
IK_DEVI uint32_t mx_scale_even(float amax) {
    if (amax != amax) return 255u;
    const uint32_t r = (__float_as_uint(amax) + (1u << 21)) & 0x7F800000u;
    const int biased = (int)(r >> 23);
    if (biased == 0) return 0u;
    if (biased == 255) return 254u;
    int e = biased - 129;
    e = e < -127 ? -127 : (e > 127 ? 127 : e);
    return (uint32_t)(e + 127);
}

// round to the nearest e2m1 value, ties to the even code, saturating at +-6 (crates/intarsia-core quant::nvfp4::f32_to_fp4)
IK_DEVI float mx_fp4_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.f;
    else if (a < 3.5f) r = 3.f;
    else if (a <= 5.f) r = 4.f;
    else r = 6.f;
    return copysignf(r, v);
}

// QDQ one group of 32 activations in place
IK_DEVI void mx_qdq32(float* f) {
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) amax = fmaxf(amax, fabsf(f[e]));
    const uint32_t sb = mx_scale_even(amax);
    const float s = mx_e8m0(sb), inv = mx_e8m0(254u - sb);  // 1 / 2^(sb-127) = 2^(127-sb), exact
#pragma unroll
    for (int e = 0; e < kG; ++e) f[e] = mx_fp4_round(f[e] * inv) * s;
}

// ---- GEMV --------------------------------------------------------------------------------------------------------

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                  const float* __restrict__ act, OutT* __restrict__ out, int64_t N, int K) {
    IK_FP4_LUT(lut);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;  // whole warp exits together
    const bool qdq = act != nullptr && act[0] != 0.f;  // null: x was QDQ'd (or copied) into scratch by the pre-pass
    const uint4* wr = reinterpret_cast<const uint4*>(w + n * (K / 2));
    const uint8_t* sr = sc + n * (K / kG);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int u = lane; u < K / kG; u += IK_WARP) {
        const uint4 wv = ik_ldcs(wr + u);
        const float s = mx_e8m0(sr[u]);
        float wf[kG];
        ik_fp4x8_to_f32(wv.x, lut, wf);
        ik_fp4x8_to_f32(wv.y, lut, wf + 8);
        ik_fp4x8_to_f32(wv.z, lut, wf + 16);
        ik_fp4x8_to_f32(wv.w, lut, wf + 24);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * kG);
            float xf[kG];
            ik_unpack8(xp[0], xf);
            ik_unpack8(xp[1], xf + 8);
            ik_unpack8(xp[2], xf + 16);
            ik_unpack8(xp[3], xf + 24);
            if (qdq) mx_qdq32(xf);
            float t = 0.f;
#pragma unroll
            for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
            acc[m] = fmaf(t, s, acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float t = ik_warp_sum(acc[m]);
        if (lane == 0) ik_store(out + (int64_t)m * N + n, t);
    }
}

// M == 1 (decode): the block stages x once in LDS (bf16 [K]), QDQ'd per 32-element group when act[0] != 0, so neither a
// pre-pass launch nor a per-row QDQ is needed. The grid is sized to the device (blocks per CU, INTARSIA_HIP_MXFP4_LDS)
// and each warp walks rows n, n + 8 * gridDim.x, ... so the staging is paid once per block, not once per 8 rows. Per
// output row the arithmetic is the one of mxfp4_gemv_kernel<1> on a pre-passed x (same lane-to-group map, same fma
// chain, same warp sum; the QDQ'd values are exact in bf16): bitwise equal.
template <typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_gemv1_lds_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, const uint8_t* __restrict__ sc0,
                       OutT* __restrict__ out0, int64_t N0, const uint8_t* __restrict__ w1, const uint8_t* __restrict__ sc1,
                       OutT* __restrict__ out1, int64_t N1, const float* __restrict__ act, int K) {
    // rows [0, N0) of (w0, sc0, out0), then rows [0, N1) of (w1, sc1, out1) (MatmulMxfp4Pair: both from one staged x)
    const int64_t N = N0 + N1;
    auto row = [&](int64_t r, const uint8_t*& w, const uint8_t*& sc, int64_t& rr) {
        if (r < N0) { w = w0; sc = sc0; rr = r; } else { w = w1; sc = sc1; rr = r - N0; }
    };
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);  // [4][G] x 8 bf16: chunk i of group g at xs[i * G + g] (lanes read
                                                 // consecutive 16-byte words: no LDS bank conflicts)
    IK_FP4_LUT(lut);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t stride = (int64_t)gridDim.x * kWarps;
    int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    const int G = K / kG;
    const bool qdq = act[0] != 0.f;
    constexpr int UN = 8;  // groups per lane with their loads in flight together
    // the first batch of the first row is in flight while x is staged
    uint4 wv[UN];
    uint32_t sb[UN];
    if (n < N) {
        const uint8_t *w, *sc;
        int64_t rr;
        row(n, w, sc, rr);
        const uint4* wr = reinterpret_cast<const uint4*>(w + rr * (K / 2));
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            const int u = lane + j * IK_WARP;
            if (u < G) {
                wv[j] = ik_ldcs(wr + u);
                sb[j] = sc[rr * G + u];
            }
        }
    }
    const uint4* xg = reinterpret_cast<const uint4*>(x);
    for (int g = threadIdx.x; g < G; g += kWarps * IK_WARP) {
        uint4 v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = xg[g * 4 + i];
        if (qdq) {
            float f[kG];
#pragma unroll
            for (int i = 0; i < 4; ++i) ik_unpack8(v[i], f + 8 * i);
            mx_qdq32(f);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = ik_pack8(f + 8 * i);  // exact: e2m1 * 2^e fits bf16
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) xs[i * G + g] = v[i];
    }
    __syncthreads();
    bool first = true;
    for (; n < N; n += stride) {  // whole warp walks the same rows
        const uint8_t *w, *sc;
        int64_t rr;
        row(n, w, sc, rr);
        const uint4* wr = reinterpret_cast<const uint4*>(w + rr * (K / 2));
        const uint8_t* sr = sc + rr * G;
        float acc = 0.f;
        for (int base = lane; base < G; base += UN * IK_WARP) {
#pragma unroll
            for (int j = 0; j < UN && !first; ++j) {
                const int u = base + j * IK_WARP;
                if (u < G) {
                    wv[j] = ik_ldcs(wr + u);
                    sb[j] = sr[u];
                }
            }
            first = false;
#pragma unroll
            for (int j = 0; j < UN; ++j) {
                const int u = base + j * IK_WARP;
                if (u < G) {  // same group order per lane as mxfp4_gemv_kernel
                    const float s = mx_e8m0(sb[j]);
                    float wf[kG];
                    ik_fp4x8_to_f32(wv[j].x, lut, wf);
                    ik_fp4x8_to_f32(wv[j].y, lut, wf + 8);
                    ik_fp4x8_to_f32(wv[j].z, lut, wf + 16);
                    ik_fp4x8_to_f32(wv[j].w, lut, wf + 24);
                    float xf[kG];
                    ik_unpack8(xs[u], xf);
                    ik_unpack8(xs[G + u], xf + 8);
                    ik_unpack8(xs[2 * G + u], xf + 16);
                    ik_unpack8(xs[3 * G + u], xf + 24);
                    float t = 0.f;
#pragma unroll
                    for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
                    acc = fmaf(t, s, acc);
                }
            }
        }
        const float t = ik_warp_sum(acc);
        if (lane == 0) ik_store((n < N0 ? out0 : out1) + rr, t);
    }
}

constexpr int64_t kLdsMax = 16 * 1024;  // x of K <= 8192 in LDS; a larger x (down_proj, K = 17408) costs occupancy:
                                        // the pre-pass + plain GEMV is faster there (measured on gfx1201)

// INTARSIA_HIP_MXFP4_LDS=<blocks per CU> for the M == 1 LDS kernel (default 4); 0: the pre-pass + plain GEMV path
static int mx_lds_blocks() {
    static const int v = [] {
        int per = 4;
        if (const char* e = getenv("INTARSIA_HIP_MXFP4_LDS")) per = atoi(e);
        if (per <= 0) return 0;
        int dev = 0, cus = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || cus <= 0)
            cus = 64;
        return per * cus;
    }();
    return v;
}

template <typename OutT>
int lds_launch(const ik_bf16* x, const uint8_t* w0, const uint8_t* s0, OutT* o0, int64_t N0, const uint8_t* w1,
               const uint8_t* s1, OutT* o1, int64_t N1, const float* act, int64_t K, hipStream_t s) {
    const int64_t full = (N0 + N1 + kWarps - 1) / kWarps;
    const unsigned grid = (unsigned)(full < mx_lds_blocks() ? full : mx_lds_blocks());
    mxfp4_gemv1_lds_kernel<OutT><<<grid, kWarps * IK_WARP, (size_t)K * 2, s>>>(x, w0, s0, o0, N0, w1, s1, o1, N1, act, (int)K);
    return ik_launch_status();
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, const float* act, OutT* out, int64_t N,
              int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
#define IK_MX_GEMV(MM) mxfp4_gemv_kernel<MM, OutT><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, sc, act, out, N, (int)K)
    switch (M) {
        case 1: IK_MX_GEMV(1); break;
        case 2: IK_MX_GEMV(2); break;
        case 3: IK_MX_GEMV(3); break;
        case 4: IK_MX_GEMV(4); break;
        case 5: IK_MX_GEMV(5); break;
        case 6: IK_MX_GEMV(6); break;
        case 7: IK_MX_GEMV(7); break;
        case 8: IK_MX_GEMV(8); break;
        default: return IK_ERR_SHAPE;
    }
#undef IK_MX_GEMV
    return ik_launch_status();
}

// ---- activation pre-pass (mma): xs[M, K] = act ? QDQ(x) : x, one thread per 32-element group --------------------

__global__ void mxfp4_xprep_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act, ik_bf16* __restrict__ xs,
                                   int64_t groups) {
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const uint4* xp = reinterpret_cast<const uint4*>(x) + g * 4;
    uint4* dp = reinterpret_cast<uint4*>(xs) + g * 4;
    if (act[0] == 0.f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) dp[i] = xp[i];
        return;
    }
    float f[kG];
#pragma unroll
    for (int i = 0; i < 4; ++i) ik_unpack8(xp[i], f + 8 * i);
    mx_qdq32(f);
#pragma unroll
    for (int i = 0; i < 4; ++i) dp[i] = ik_pack8(f + 8 * i);  // exact: e2m1 * 2^e fits bf16
}

// ---- MMA policy ---------------------------------------------------------------------------------------------------

struct Mxfp4Policy {
    IK_DEVI bool skip(const DqArgs&) { return false; }
    struct State {
        const float* lut;
    };
    struct Raw {
        uint2 w;     // 16 fp4 codes
        uint32_t s;  // e8m0 group scale
    };
    IK_DEVI State init(const DqArgs&) {
        IK_FP4_LUT(lut);
        State s;
        s.lut = lut;
        return s;
    }
    IK_DEVI Raw load(const DqArgs& a, int64_t n, int k) {
        Raw r;
        r.w = *reinterpret_cast<const uint2*>(a.w + n * (a.K / 2) + k / 2);
        r.s = a.bs[n * (a.K / kG) + k / kG];
        return r;
    }
    IK_DEVI Raw zero() {
        Raw r;
        r.w = make_uint2(0, 0);
        r.s = 127;  // 1.0, times zero codes
        return r;
    }
    IK_DEVI void dequant(const State& st, const Raw& r, uint4& lo, uint4& hi) {
        const float s = mx_e8m0(r.s);
        float f[16];
        ik_fp4x8_to_f32(r.w.x, st.lut, f);
        ik_fp4x8_to_f32(r.w.y, st.lut, f + 8);
#pragma unroll
        for (int e = 0; e < 16; ++e) f[e] *= s;  // exact: e2m1 * 2^e
        lo = ik_pack8(f);
        hi = ik_pack8(f + 8);
    }
    IK_DEVI uint4 xprep(const State&, uint4 v) { return v; }
    IK_DEVI float out_scale(const State&) { return 1.f; }
};

// ---- launchers ---------------------------------------------------------------------------------------------------

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, const ik_tensor* act, ik_tensor* out,
           const ik_scratch* scratch, void* stream, bool gemv) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(sc, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % kG != 0 || w->shape[1] != K / 2 || sc->shape[0] != N || sc->shape[1] != K / kG || act->shape[0] != 1 ||
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
    const uint8_t* sp = (const uint8_t*)sc->data;
    const float* ap = (const float*)act->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > 8) return IK_ERR_SHAPE;
        if (M == 1 && K * 2 <= kLdsMax && mx_lds_blocks() > 0) {
            return f32o ? lds_launch(xp, wp, sp, (float*)out->data, N, (const uint8_t*)nullptr, (const uint8_t*)nullptr,
                                     (float*)nullptr, 0, ap, K, s)
                        : lds_launch(xp, wp, sp, (ik_bf16*)out->data, N, (const uint8_t*)nullptr, (const uint8_t*)nullptr,
                                     (ik_bf16*)nullptr, 0, ap, K, s);
        }
        if (scratch && scratch->data && scratch->bytes >= M * K * 2 && !(reinterpret_cast<uintptr_t>(scratch->data) & 15u)) {
            // QDQ (or copy) the activations once, then the GEMV reads them as plain bf16
            ik_bf16* xs = (ik_bf16*)scratch->data;
            const int64_t groups = M * K / kG;
            mxfp4_xprep_kernel<<<(unsigned)((groups + 255) / 256), 256, 0, s>>>(xp, ap, xs, groups);
            IK_TRY(ik_launch_status());
            xp = xs;
            ap = nullptr;
        }
        return f32o ? gemv_by_m((int)M, xp, wp, sp, ap, (float*)out->data, N, K, s)
                    : gemv_by_m((int)M, xp, wp, sp, ap, (ik_bf16*)out->data, N, K, s);
    }
    if (!scratch || !scratch->data || scratch->bytes < M * K * 2) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    ik_bf16* xs = (ik_bf16*)scratch->data;
    const int64_t groups = M * K / kG;
    mxfp4_xprep_kernel<<<(unsigned)((groups + 255) / 256), 256, 0, s>>>(xp, ap, xs, groups);
    IK_TRY(ik_launch_status());
    DqArgs a;
    a.x = xs;
    a.w = wp;
    a.bs = sp;
    a.sc = nullptr;
    a.xq = nullptr;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    if (f32o) dq_gemm_run<Mxfp4Policy>(a, (float*)out->data, s);
    else dq_gemm_run<Mxfp4Policy>(a, (ik_bf16*)out->data, s);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_mxfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                         ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return launch(x, w, scales, act, out, scratch, stream, true);
}
// MatmulMxfp4Pair: out0 = matmul_mxfp4(x, w0, s0, act), out1 = matmul_mxfp4(x, w1, s1, act). M == 1 and K <= 8192: one
// launch of the LDS kernel over the rows of both (each row bitwise as in the single op); otherwise the two single ops.
int ik_matmul_mxfp4_pair_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1,
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
    if (M == 1 && K > 0 && K * 2 <= kLdsMax && mx_lds_blocks() > 0 && N0 > 0 && N1 > 0) {
        hipStream_t s = IK_STREAM(stream);
        const ik_bf16* xp = (const ik_bf16*)x->data;
        const float* ap = (const float*)act->data;
        const uint8_t *a0 = (const uint8_t*)w0->data, *b0 = (const uint8_t*)s0->data;
        const uint8_t *a1 = (const uint8_t*)w1->data, *b1 = (const uint8_t*)s1->data;
        return out0->dtype == IK_DTYPE_F32
                   ? lds_launch(xp, a0, b0, (float*)out0->data, N0, a1, b1, (float*)out1->data, N1, ap, K, s)
                   : lds_launch(xp, a0, b0, (ik_bf16*)out0->data, N0, a1, b1, (ik_bf16*)out1->data, N1, ap, K, s);
    }
    IK_TRY(launch(x, w0, s0, act, out0, scratch, stream, true));
    return launch(x, w1, s1, act, out1, scratch, stream, true);
}
int ik_matmul_mxfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                        ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return launch(x, w, scales, act, out, scratch, stream, false);
}

}  // extern "C"
