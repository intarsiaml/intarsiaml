// matmul_nvfp4: out[M,N] = x[M,K] @ W^T with W[n,k] = e2m1(code) * e4m3(block_scale[n,k/16]) * global (W4A16).
//   w  u8   [N, K/2]   two codes per byte, element 2i in the low nibble
//   bs e4m3 [N, K/16]  plain row-major (checkpoint layout)
//   gs f32  [1]        weight_scale_2
//   *_gemv  M <= 8: warp-per-output-row, one 16-element block (8 weight bytes + 1 scale byte) per lane-iteration.
//   *_mma   any M: bf16 WMMA, e2m1*e4m3 dequantised to bf16 in shared memory (exact), f32 accumulate (ik_dq_gemm.h).
// Deterministic (fixed reduction order). Activations bf16; output bf16 or f32 (the lm_head writes f32 logits).
// HIP port of kernels/cuda/nvfp4.cu. RDNA4 has no fp4 hardware: the e2m1 codes are looked up in a 16-entry shared
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

template <int M, typename OutT>
void gemv_run(const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t N, int64_t K,
              hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    nvfp4_gemv_kernel<M, OutT><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, bs, gs, out, N, (int)K);
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int64_t N,
              int64_t K, hipStream_t s) {
    switch (M) {
        case 1: gemv_run<1>(x, w, bs, gs, out, N, K, s); break;
        case 2: gemv_run<2>(x, w, bs, gs, out, N, K, s); break;
        case 3: gemv_run<3>(x, w, bs, gs, out, N, K, s); break;
        case 4: gemv_run<4>(x, w, bs, gs, out, N, K, s); break;
        case 5: gemv_run<5>(x, w, bs, gs, out, N, K, s); break;
        case 6: gemv_run<6>(x, w, bs, gs, out, N, K, s); break;
        case 7: gemv_run<7>(x, w, bs, gs, out, N, K, s); break;
        case 8: gemv_run<8>(x, w, bs, gs, out, N, K, s); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

// ---- MMA policy ---------------------------------------------------------------------------------------------------

struct Nvfp4Policy {
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

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs, ik_tensor* out,
           void* stream, bool gemv) {
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
        return f32o ? gemv_by_m((int)M, xp, wp, bp, gp, (float*)out->data, N, K, s)
                    : gemv_by_m((int)M, xp, wp, bp, gp, (ik_bf16*)out->data, N, K, s);
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

}  // namespace

extern "C" {

int ik_matmul_nvfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                         const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, true);
}
int ik_matmul_nvfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                        const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    return launch(x, w, block_scales, global_scale, out, stream, false);
}

}  // extern "C"
