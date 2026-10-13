// matmul_fp8: out[M,N] = x[M,K] @ W^T, W = float(w_e4m3[N,K]) * weight_scale (per tensor).
//   scales = f32[2] = (weight_scale, input_scale); input_scale > 0 selects W8A8 (static activation quantisation:
//   xq = e4m3(x / input_scale), RNE, saturating; y = weight_scale * input_scale * sum xq*wq), otherwise W8A16.
//   The mode is read from the device tensor inside the kernels (a uniform branch), so launching never needs a
//   host read and the same kernels serve both recipes.
//   *_gemv  decode: M <= 4, warp-per-output-row, x staged in shared memory (quantised once per block).
//   *_mma   any M: bf16 WMMA with the weights dequantised to bf16 in shared memory (exact), f32 accumulate
//           (ik_dq_gemm.h). Used for prefill and as the correct path for every shape.
// Both are deterministic (fixed reduction order). Activations bf16; output bf16 or f32.
// HIP port of kernels/cuda/fp8.cu; the "dequantise to bf16, bf16 MMA with exact products" scheme follows the ideas of
// TensorFold's NVFP4 kernels (MIT, Copyright (c) 2026 TensorFold contributors, see ik_quant.h and
// LICENSES/TensorFold-MIT.txt). gfx12 also has an fp8 x fp8 WMMA (v_wmma_f32_16x16x16_fp8_fp8) that would
// serve the W8A8 case natively; not used yet (docs/results/2026-10-13-phase2a-hip.md).
#include "ik_dq_gemm.h"

namespace {

constexpr int kWarps = 8;

// ---- GEMV --------------------------------------------------------------------------------------------------------
// One warp owns R output rows. The weight loads of the first batch (UN 16-byte chunks per row and lane) are issued
// before the activations are staged in shared memory, so their latency overlaps the staging (the e4m3 activation
// quantisation is the longest part of the prologue); later batches are loaded as the previous one is consumed.

constexpr int kUn = 4;

// e4m3 as fp16: (sign << 15) | (magnitude << 7) is the e4m3 value times 2^-8, exactly, subnormals included (an e4m3
// subnormal m * 2^-9 becomes the fp16 subnormal m * 2^-17). With W8A8 the activations are e4m3 too, so they are exact
// fp16 and v_dot2_f32_f16 multiplies and accumulates the pairs in f32: ~3 ALU ops per element instead of ~18 for the
// generic decode. The 2^8 is folded into the output scale. (NaN codes are not preserved here; checkpoints have none.)
typedef _Float16 ik_h2 __attribute__((ext_vector_type(2)));

IK_DEVI uint32_t e4m3x2_to_f16x2(uint32_t w, uint32_t sel) {
    const uint32_t t = __builtin_amdgcn_perm(0u, w, sel);  // [0, b_lo, 0, b_hi]
    return (((t >> 1) & 0x3F803F80u) | (t & 0x80008000u));
}

template <int M, int R, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                OutT* __restrict__ out, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* xs = reinterpret_cast<ik_bf16*>(smem);  // [M][K] bf16: x, or (W8A8) xq as fp16 bits (exact)
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = ((int64_t)blockIdx.x * kWarps + warp) * R;
    const int nu = K / 16;
    const uint4* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = n0 + r < N ? reinterpret_cast<const uint4*>(w + (n0 + r) * K) : nullptr;
    uint4 wb[R][kUn];
    auto load = [&](int i0) {  // iterations i0 .. i0 + kUn - 1 of this lane
#pragma unroll
        for (int j = 0; j < kUn; ++j) {
            const int u = lane + IK_WARP * (i0 + j);
#pragma unroll
            for (int r = 0; r < R; ++r) wb[r][j] = (wr[r] && u < nu) ? ik_ldcs(wr[r] + u) : make_uint4(0, 0, 0, 0);
        }
    };
    load(0);

    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        const ik_bf16 v = x[i];
        if (actq) {
            const _Float16 q = (_Float16)ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(ik_b2f(v), s_in)));
            xs[i].bits = __builtin_bit_cast(uint16_t, q);
        } else {
            xs[i] = v;
        }
    }
    __syncthreads();
    if (!wr[0]) return;  // (all rows of the warp are past N; rows are consecutive)

    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    const int iters = (nu + IK_WARP - 1 - lane) / IK_WARP;  // iterations this lane runs
    if (actq) {
        for (int i0 = 0; i0 < iters; i0 += kUn) {
            if (i0 > 0) load(i0);
#pragma unroll
            for (int j = 0; j < kUn; ++j) {
                const int u = lane + IK_WARP * (i0 + j);
                if (u >= nu) break;
                uint4 xv[M][2];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const uint4* xp = reinterpret_cast<const uint4*>(xs + (int64_t)m * K + u * 16);
                    xv[m][0] = xp[0];
                    xv[m][1] = xp[1];
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t ww[4] = {wb[r][j].x, wb[r][j].y, wb[r][j].z, wb[r][j].w};
                    uint32_t wh[8];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        wh[2 * q] = e4m3x2_to_f16x2(ww[q], 0x010C000Cu);
                        wh[2 * q + 1] = e4m3x2_to_f16x2(ww[q], 0x030C020Cu);
                    }
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        const uint32_t xw[8] = {xv[m][0].x, xv[m][0].y, xv[m][0].z, xv[m][0].w,
                                                xv[m][1].x, xv[m][1].y, xv[m][1].z, xv[m][1].w};
                        float s = acc[r][m];
#pragma unroll
                        for (int q = 0; q < 8; ++q)
                            s = __builtin_amdgcn_fdot2(__builtin_bit_cast(ik_h2, wh[q]), __builtin_bit_cast(ik_h2, xw[q]), s, false);
                        acc[r][m] = s;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float s = ik_warp_sum(acc[r][m]);
                if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, s * 256.f * s_out);
            }
        return;
    }
    for (int i0 = 0; i0 < iters; i0 += kUn) {
        if (i0 > 0) load(i0);
#pragma unroll
        for (int j = 0; j < kUn; ++j) {
            const int u = lane + IK_WARP * (i0 + j);
            if (u >= nu) break;
            float xf[M][16];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4* xp = reinterpret_cast<const uint4*>(xs + (int64_t)m * K + u * 16);
                ik_unpack8(xp[0], xf[m]);
                ik_unpack8(xp[1], xf[m] + 8);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float wf[16];
                ik_fp8x16_to_f32(wb[r][j], wf);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    float s = 0.f;
#pragma unroll
                    for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[m][e], s);
                    acc[r][m] += s;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float s = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, s * s_out);
        }
}

// rows per warp: INTARSIA_HIP_FP8_ROWS=1|2|4 forces one, otherwise 1 for small N (keeps enough blocks in flight)
int gemv_rows(int64_t N) {
    static const int forced = [] {
        const char* e = getenv("INTARSIA_HIP_FP8_ROWS");
        return e ? atoi(e) : 0;
    }();
    if (forced == 1 || forced == 2 || forced == 4) return forced;
    return N >= 6144 ? 2 : 1;
}

template <int M, int R, typename OutT>
void gemv_run_r(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps * R - 1) / (kWarps * R));
    const size_t smem = (size_t)M * K * 2;
    fp8_gemv_kernel<M, R, OutT><<<grid, kWarps * IK_WARP, smem, s>>>(x, w, sc, out, (int)N, (int)K);
}

template <int M, typename OutT>
void gemv_run(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    switch (gemv_rows(N)) {
        case 1: gemv_run_r<M, 1>(x, w, sc, out, N, K, s); break;
        case 4: gemv_run_r<M, 4>(x, w, sc, out, N, K, s); break;
        default: gemv_run_r<M, 2>(x, w, sc, out, N, K, s); break;
    }
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    switch (M) {
        case 1: gemv_run<1>(x, w, sc, out, N, K, s); break;
        case 2: gemv_run<2>(x, w, sc, out, N, K, s); break;
        case 3: gemv_run<3>(x, w, sc, out, N, K, s); break;
        case 4: gemv_run<4>(x, w, sc, out, N, K, s); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

// ---- MMA policy (weights dequantised to bf16 tiles) ---------------------------------------------------------------

struct Fp8Policy {
    struct State {
        bool actq;
        float s_in, s_out;
    };
    typedef uint4 Raw;  // 16 e4m3 bytes
    IK_DEVI State init(const DqArgs& a) {
        State s;
        s.actq = a.sc[1] > 0.f;
        s.s_in = s.actq ? a.sc[1] : 1.f;
        s.s_out = a.sc[0] * s.s_in;
        return s;
    }
    IK_DEVI Raw load(const DqArgs& a, int64_t n, int k) {
        return *reinterpret_cast<const uint4*>(a.w + n * a.K + k);
    }
    IK_DEVI Raw zero() { return make_uint4(0, 0, 0, 0); }
    IK_DEVI void dequant(const State&, const Raw& r, uint4& lo, uint4& hi) {
        float f[16];
        ik_fp8x16_to_f32(r, f);  // exact in bf16
        lo = ik_pack8(f);
        hi = ik_pack8(f + 8);
    }
    IK_DEVI uint4 xprep(const State& s, uint4 v) {
        if (!s.actq) return v;
        float f[8];
        ik_unpack8(v, f);
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s.s_in)));
        return ik_pack8(f);  // fp8 values are exact in bf16
    }
    IK_DEVI float out_scale(const State& s) { return s.s_out; }
};

// ---- launchers ---------------------------------------------------------------------------------------------------

int check(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* out) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    if (K % 16 != 0) return IK_ERR_SHAPE;
    return IK_OK;
}

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream, bool gemv) {
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == hipSuccess
                   ? IK_OK
                   : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > 4 || M * K * 2 > 49152) return IK_ERR_SHAPE;
        return f32o ? gemv_by_m((int)M, xp, wp, sp, (float*)out->data, N, K, s)
                    : gemv_by_m((int)M, xp, wp, sp, (ik_bf16*)out->data, N, K, s);
    }
    DqArgs a;
    a.x = xp;
    a.w = wp;
    a.bs = nullptr;
    a.sc = sp;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    if (f32o) dq_gemm_run<Fp8Policy>(a, (float*)out->data, s);
    else dq_gemm_run<Fp8Policy>(a, (ik_bf16*)out->data, s);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, true);
}
int ik_matmul_fp8_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, false);
}

}  // extern "C"
