// matmul_mxfp4 for 5..8 activation rows and K <= 3072, W4A16, row-exact (lane ink-cuda-mixmoe): the Inkling-2 DSpark draft
// head ([200058 x 2304] MXFP4, the 7 draft rows of a block, once per draft). Bitwise `mxfp4_gemv_kernel` (mxfp4.cu) per output
// element: the same chain (32 fmaf over a 32-element group, then one fmaf with the group's e8m0 scale), the groups of an output
// row in the same lane order (lane, lane + 32, ...), the same warp_sum. What differs is the work split: one warp owns 4 output
// rows and every group it owns for them is loaded up front (K / 32 / 32 rounded up = NI per lane), so a warp has 4 x NI x 17
// bytes in flight and loads each activation group once for the 4 rows; the generic GEMV keeps one 16-byte group per lane in
// flight and reloads the activations per output row. The microbench (.scratch/mx4head_probe.cu, docs/results/
// 2026-11-21-inkling2-cuda-mixmoe.md): M 7 2.44 -> 1.38 ms (100 -> 177 GB/s), 0 of 1.4M outputs differ.
// `act` is accepted for the matmul_mxfp4 signature and not read: the registry routes W4A16 calls only.
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "ik_cuda_abi_inkmm.h"

namespace {

constexpr int kWarps = 8;
constexpr int kG = 32;  // elements per e8m0 scale
constexpr int kR = 4;   // output rows per warp
constexpr int kNIMax = 3;

// e8m0 byte -> 2^(b - 127); 0 -> 2^-127 (f32 subnormal), 255 -> NaN (mxfp4.cu mx_e8m0)
IK_DEVI float hx_e8m0(uint32_t b) {
    return b == 0u ? __uint_as_float(0x00400000u) : (b == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(b << 23));
}

template <typename OutT, int MT, int NI>
__global__ void __launch_bounds__(kWarps * IK_WARP)
ink_mx4head_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                   OutT* __restrict__ out, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = ((int64_t)blockIdx.x * kWarps + warp) * kR;
    if (n0 >= N) return;  // whole warp exits together
    const int G = K / kG;
    uint4 wv[NI][kR];
    uint32_t sb[NI][kR];
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int u = lane + IK_WARP * i;
#pragma unroll
        for (int r = 0; r < kR; ++r) {
            const int64_t nr = n0 + r < N ? n0 + r : N - 1;  // the tail warp re-reads the last row, never stores it
            if (u < G) {
                wv[i][r] = __ldcs(reinterpret_cast<const uint4*>(w + nr * (K / 2)) + u);
                sb[i][r] = sc[nr * G + u];
            }
        }
    }
    float acc[kR][MT];
#pragma unroll
    for (int r = 0; r < kR; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int u = lane + IK_WARP * i;
        if (u >= G) break;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * kG);
            float xf[kG];
            ik_unpack8(__ldg(xp), xf);
            ik_unpack8(__ldg(xp + 1), xf + 8);
            ik_unpack8(__ldg(xp + 2), xf + 16);
            ik_unpack8(__ldg(xp + 3), xf + 24);
#pragma unroll
            for (int r = 0; r < kR; ++r) {
                const float s = hx_e8m0(sb[i][r]);
                float wf[kG];
                ik_fp4x8_to_f32(wv[i][r].x, wf);
                ik_fp4x8_to_f32(wv[i][r].y, wf + 8);
                ik_fp4x8_to_f32(wv[i][r].z, wf + 16);
                ik_fp4x8_to_f32(wv[i][r].w, wf + 24);
                float t = 0.f;
#pragma unroll
                for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
                acc[r][m] = fmaf(t, s, acc[r][m]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kR; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float t = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, t);
        }
}

template <typename OutT, int MT>
void launch_ni(int ni, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, OutT* out, int64_t N, int K, cudaStream_t s) {
    const dim3 grid((unsigned)(((N + kR - 1) / kR + kWarps - 1) / kWarps));
    // NI 1 (K <= 1024) runs the NI 2 instance (its second group is skipped by the u < G guards): ptxas spills the NI 1 one
    switch (ni) {
        case 1:
        case 2: ink_mx4head_kernel<OutT, MT, 2><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, sc, out, N, K); break;
        default: ink_mx4head_kernel<OutT, MT, 3><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, sc, out, N, K); break;
    }
}

template <typename OutT>
void launch_m(int M, int ni, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, OutT* out, int64_t N, int K, cudaStream_t s) {
    switch (M) {
        case 5: launch_ni<OutT, 5>(ni, x, w, sc, out, N, K, s); break;
        case 6: launch_ni<OutT, 6>(ni, x, w, sc, out, N, K, s); break;
        case 7: launch_ni<OutT, 7>(ni, x, w, sc, out, N, K, s); break;
        default: launch_ni<OutT, 8>(ni, x, w, sc, out, N, K, s); break;
    }
}

}  // namespace

extern "C" int ik_matmul_mxfp4_gemv_ink_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, const ik_tensor* act,
                                             ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(sc, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % kG != 0 || w->shape[1] != K / 2 || sc->shape[0] != N || sc->shape[1] != K / kG || act->shape[0] != 1 ||
        out->shape[0] != M || out->shape[1] != N || M < 5 || M > 8 || K < kG || K > (int64_t)kG * IK_WARP * kNIMax)
        return IK_ERR_SHAPE;
    if (N == 0) return IK_OK;
    const int ni = (int)((K / kG + IK_WARP - 1) / IK_WARP);
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* sp = (const uint8_t*)sc->data;
    if (out->dtype == IK_DTYPE_F32)
        launch_m<float>((int)M, ni, xp, wp, sp, (float*)out->data, N, (int)K, s);
    else
        launch_m<ik_bf16>((int)M, ni, xp, wp, sp, (ik_bf16*)out->data, N, (int)K, s);
    return ik_launch_status();
}
