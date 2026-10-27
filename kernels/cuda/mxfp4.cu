// matmul_mxfp4: out[M,N] = q(x)[M,K] @ W^T with W[n,k] = e2m1(code) * 2^(scale[n,k/32] - 127) (OCP MXFP4, AMD Quark).
//   w     u8  [N, K/2]   two codes per byte, element 2i in the low nibble
//   scale u8  [N, K/32]  e8m0, plain row-major (checkpoint layout)
//   act   f32 [1]        0: q(x) = x (W4A16); 1: q(x) = dynamic MXFP4 QDQ of x per 32-element group along K (W4A4, Quark
//                        "even" scale; crates/intarsia-core/src/quant/mxfp4.rs)
// A port of kernels/hip/mxfp4.hip's GEMV (M1 drill): warp-per-output-row, one 32-element group (16 weight bytes, one scale
// byte, 32 activations) per lane-iteration, so a group's activation QDQ happens inside one lane. Any M: blockIdx.y walks
// M in chunks of 8 rows (prefill re-reads W once per chunk; correct, not fast; prefill-shaped W4A16 calls take the tensor-core
// GEMM in mxfp4_gemm.cu instead).
// Deterministic (fixed reduction order). Activations bf16; output bf16 or f32.
#include "ik_launch.h"
#include "ik_quant.cuh"

namespace {

constexpr int kWarps = 8;
constexpr int kG = 32;   // elements per e8m0 scale
constexpr int kMB = 8;   // activation rows per block

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

// nearest e2m1 value, ties to the even code, saturating at +-6 (quant::nvfp4::f32_to_fp4)
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

IK_DEVI void mx_qdq32(float* f) {
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) amax = fmaxf(amax, fabsf(f[e]));
    const uint32_t sb = mx_scale_even(amax);
    const float s = mx_e8m0(sb), inv = mx_e8m0(254u - sb);  // 2^(127 - sb): exact reciprocal
#pragma unroll
    for (int e = 0; e < kG; ++e) f[e] = mx_fp4_round(f[e] * inv) * s;
}

template <typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                  const float* __restrict__ act, OutT* __restrict__ out, int64_t M, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;  // whole warp exits together
    const int64_t m0 = (int64_t)blockIdx.y * kMB;
    const int mc = (int)((M - m0) < kMB ? (M - m0) : kMB);
    const bool qdq = act[0] != 0.f;
    const uint4* wr = reinterpret_cast<const uint4*>(w + n * (K / 2));
    const uint8_t* sr = sc + n * (K / kG);
    float acc[kMB];
#pragma unroll
    for (int m = 0; m < kMB; ++m) acc[m] = 0.f;
    for (int u = lane; u < K / kG; u += IK_WARP) {
        const uint4 wv = __ldcs(wr + u);
        const float s = mx_e8m0(sr[u]);
        float wf[kG];
        ik_fp4x8_to_f32(wv.x, wf);
        ik_fp4x8_to_f32(wv.y, wf + 8);
        ik_fp4x8_to_f32(wv.z, wf + 16);
        ik_fp4x8_to_f32(wv.w, wf + 24);
#pragma unroll
        for (int m = 0; m < kMB; ++m) {
            if (m >= mc) break;
            const uint4* xp = reinterpret_cast<const uint4*>(x + (m0 + m) * K + (int64_t)u * kG);
            float xf[kG];
            ik_unpack8(__ldg(xp), xf);
            ik_unpack8(__ldg(xp + 1), xf + 8);
            ik_unpack8(__ldg(xp + 2), xf + 16);
            ik_unpack8(__ldg(xp + 3), xf + 24);
            if (qdq) mx_qdq32(xf);
            float t = 0.f;
#pragma unroll
            for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
            acc[m] = fmaf(t, s, acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < kMB; ++m) {
        const float t = ik_warp_sum(acc[m]);
        if (lane == 0 && m < mc) ik_store(out + (m0 + m) * N + n, t);
    }
}

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, const ik_tensor* act, ik_tensor* out, void* stream) {
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
        return cudaMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == cudaSuccess
                   ? IK_OK
                   : IK_ERR_LAUNCH;
    if (K > 0x7fffffffLL || (M + kMB - 1) / kMB > 65535) return IK_ERR_SHAPE;
    const dim3 grid((unsigned)((N + kWarps - 1) / kWarps), (unsigned)((M + kMB - 1) / kMB));
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* sp = (const uint8_t*)sc->data;
    const float* ap = (const float*)act->data;
    if (out->dtype == IK_DTYPE_F32)
        mxfp4_gemv_kernel<float><<<grid, kWarps * IK_WARP, 0, s>>>(xp, wp, sp, ap, (float*)out->data, M, N, (int)K);
    else
        mxfp4_gemv_kernel<ik_bf16><<<grid, kWarps * IK_WARP, 0, s>>>(xp, wp, sp, ap, (ik_bf16*)out->data, M, N, (int)K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_mxfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                         ik_tensor* out, void* stream) {
    return launch(x, w, scales, act, out, stream);
}

// The row-exact twin (`rows_exact`, `OpCall::exact_rows`: verify passes and batched decode steps). The GEMV above is already
// bitwise per row at any M (every row has its own fixed reduction order; blockIdx.y only walks the rows), so the twin is the
// same launcher under a second name: the registry then keeps the tensor-core `ik_matmul_mxfp4_mma` (not bitwise the GEMV)
// away from those calls.
int ik_matmul_mxfp4_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                              ik_tensor* out, void* stream) {
    return launch(x, w, scales, act, out, stream);
}

}  // extern "C"
