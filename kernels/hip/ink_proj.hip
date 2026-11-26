// Inkling-2-Mini attention in-projection (Op::InkInProj) on HIP (lane ink-hip-mix).
//
// Decode / verify form (T <= 16): out[m, n] = rA(x[m] . w[n]) for up to three segments (weight rows, activation plane, output) in ONE
// launch. One wave per weight row of the stacked weight [Nq + Nk + Nv, D]: its lanes stride the row in 16-byte steps (lane partial
// in k order, then the wave butterfly), for all M activation rows at once, so the weight streams from memory once whatever M and
// every output sum has the same lane / chunk / wave order whatever M: rows-exact. A port of in_proj_gemv_kernel<M, false>
// (kernels/cuda/inkling2_p12.cu), the bf16 arm.
//
// Prefill form (T > 16): three matmul_bf16 GEMMs (ik_matmul_bf16, the native WMMA GEMM of matmul.hip) on views of the segments.
#include "ik_hip.h"
#include "ik_hip_abi_inkmix.h"

namespace {

constexpr int kWarpsPerBlock = 8;

struct GemvSegs {
    const ik_bf16* x[3];
    ik_bf16* out[3];
    int n[3];
    int r0[4];
};

template <int M>
__global__ void __launch_bounds__(kWarpsPerBlock* IK_WARP) in_proj_gemv_kernel(GemvSegs sg, const ik_bf16* __restrict__ w, int64_t N,
                                                                                int64_t K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarpsPerBlock + warp;
    if (n >= N) return;
    const int j = n < sg.r0[1] ? 0 : (n < sg.r0[2] ? 1 : 2);
    const int64_t nn = n - sg.r0[j];
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(sg.x[j]);
    const uint4* w8 = reinterpret_cast<const uint4*>(w) + n * nv;
#pragma unroll 4
    for (int64_t i = lane; i < nv; i += IK_WARP) {
        float wf[8];
        ik_unpack8(ik_ldcs(w8 + i), wf);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float xf[8];
            ik_unpack8(ik_ldg(x8 + m * nv + i), xf);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[m] += wf[e] * xf[e];
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) sg.out[j][(int64_t)m * sg.n[j] + nn] = ik_f2b(s);
    }
}

// mode 1: the GEMV (T <= 16); 2: three matmul_bf16 GEMMs
int in_proj(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, int mode, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 2));
    ik_tensor* o[3] = {pq, pk, pv};
    const int64_t S = x->shape[0], T = x->shape[1], D = x->shape[2];
    if (!(S == 1 || S == 3) || w->shape[1] != D) return IK_ERR_SHAPE;
    int64_t nsum = 0;
    for (int j = 0; j < 3; ++j) {
        IK_TRY(ik_check(o[j], IK_DTYPE_BF16, 2));
        if (o[j]->shape[0] != T) return IK_ERR_SHAPE;
        nsum += o[j]->shape[1];
    }
    if (nsum != w->shape[0]) return IK_ERR_SHAPE;
    if (T == 0 || nsum == 0) return IK_OK;
    hipStream_t st = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const ik_bf16* wp = (const ik_bf16*)w->data;
    if (mode == 1) {
        if (T > 16 || D % 16 != 0 || D > 65536 || nsum > 0x7fffffff) return IK_ERR_SHAPE;
        if (!ik_aligned16(x->data) || !ik_aligned16(w->data)) return IK_ERR_INVALID_ARG;
        GemvSegs sg{};
        int64_t r0 = 0;
        for (int j = 0; j < 3; ++j) {
            sg.x[j] = xp + (j < S - 1 ? j : S - 1) * T * D;
            sg.out[j] = (ik_bf16*)o[j]->data;
            sg.n[j] = (int)o[j]->shape[1];
            sg.r0[j] = (int)r0;
            r0 += o[j]->shape[1];
        }
        sg.r0[3] = (int)r0;
        const unsigned grid = (unsigned)((nsum + kWarpsPerBlock - 1) / kWarpsPerBlock);
        const unsigned block = kWarpsPerBlock * IK_WARP;
#define IK_IPG_CASE(R) \
    case R: in_proj_gemv_kernel<R><<<grid, block, 0, st>>>(sg, wp, nsum, D); break;
        switch (T) {
            IK_IPG_CASE(1) IK_IPG_CASE(2) IK_IPG_CASE(3) IK_IPG_CASE(4) IK_IPG_CASE(5) IK_IPG_CASE(6) IK_IPG_CASE(7) IK_IPG_CASE(8)
            IK_IPG_CASE(9) IK_IPG_CASE(10) IK_IPG_CASE(11) IK_IPG_CASE(12) IK_IPG_CASE(13) IK_IPG_CASE(14) IK_IPG_CASE(15)
            IK_IPG_CASE(16)
            default: return IK_ERR_SHAPE;
        }
#undef IK_IPG_CASE
        return ik_launch_status();
    }
    // prefill: three GEMMs on views of the segments
    int64_t r0 = 0;
    for (int j = 0; j < 3; ++j) {
        const int64_t xi = j < S - 1 ? j : S - 1;
        ik_tensor xv = *x, wv = *w;
        xv.data = (void*)(xp + xi * T * D);
        xv.ndim = 2;
        xv.shape[0] = T, xv.shape[1] = D, xv.stride[0] = D, xv.stride[1] = 1;
        wv.data = (void*)(wp + r0 * D);
        wv.shape[0] = o[j]->shape[1];
        IK_TRY(ik_matmul_bf16(&xv, &wv, o[j], stream));
        r0 += o[j]->shape[1];
    }
    return IK_OK;
}

}  // namespace

extern "C" {

int ik_ink_in_proj_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream) {
    return in_proj(x, w, pq, pk, pv, 1, stream);
}
int ik_ink_in_proj_bf16_gemv_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream) {
    return in_proj(x, w, pq, pk, pv, 1, stream);
}
int ik_ink_in_proj_bf16_matmul(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream) {
    return in_proj(x, w, pq, pk, pv, 2, stream);
}

}  // extern "C"
