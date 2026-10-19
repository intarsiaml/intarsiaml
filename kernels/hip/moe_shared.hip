// shared_down_combine (Op::SharedDownCombine, core-op fusion P2a, docs/results/2026-10-19-core-op-fusion-p1.md): the MoE
// shared-expert down projection (NVFP4 GEMV, M <= 8 rows) with moe_combine as its epilogue, one launch instead of two.
//   out[m, n] = bf16(routed[m, n] + bf16(bf16(sigmoid(logit[m])) * so)),  so = bf16(x[m, :] @ W[n, :]^T)
// A warp owns output row n of every token (nvfp4_gemv_v2_kernel<M, bf16, false>'s mapping: same lane / iteration order, same
// fma chain, the global scale applied after the warp sum, the result rounded to bf16 as that kernel's store does), and lane 0
// then runs moe_combine_kernel's expression on it. A row is complete inside one warp (no split-K, no second pass), so the
// combine needs no wait and the bits equal ik_matmul_nvfp4_gemv_v2 followed by ik_moe_combine_bf16. Kept in its own file:
// nothing in nvfp4.hip / moe.hip changes.
#include "ik_dq_gemm.h"
#include "ik_hip_abi.h"

namespace {

constexpr int kWarps = 8;
constexpr int kPre = 4;  // nvfp4_gemv_v2_kernel's lane-iterations requested before the table barrier

template <int M>
__global__ void __launch_bounds__(kWarps * IK_WARP)
shared_down_combine_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                           const float* __restrict__ gs, const ik_bf16* __restrict__ routed, const ik_bf16* __restrict__ logit,
                           ik_bf16* __restrict__ out, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    const bool live = n < N;
    const int64_t nn = live ? n : 0;
    const uint2* wr = reinterpret_cast<const uint2*>(w + nn * (K / 2));
    const uint8_t* sr = bs + nn * (K / 16);
    const int nu = K / 16;
    uint2 wv[kPre];
    uint32_t sb[kPre];
    uint4 xv[kPre][M][2];
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        const int u = lane + IK_WARP * j;
        const bool ok = u < nu;
        wv[j] = ok ? ik_ldcs(wr + u) : make_uint2(0, 0);
        sb[j] = ok ? sr[u] : 0u;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)(ok ? u : 0) * 16);
            xv[j][m][0] = xp[0];
            xv[j][m][1] = xp[1];
        }
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
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
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        if (lane + IK_WARP * j >= nu) continue;
        step(wv[j], sb[j], xv[j]);
    }
    for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP) {  // K > 2048: the remaining iterations, loaded as used
        uint4 xq[M][2];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * 16);
            xq[m][0] = xp[0];
            xq[m][1] = xp[1];
        }
        step(ik_ldcs(wr + u), sr[u], xq);
    }
    const float g = gs[0];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) {
            const float so = ik_round_bf16(s * g);  // what the down GEMV stores (the shared-expert output)
            // moe_combine_kernel's epilogue on that value
            const float gl = ik_b2f(logit[m]);
            const float sg = ik_round_bf16(1.f / (1.f + expf(-gl)));
            const float sv = ik_round_bf16(sg * so);
            const int64_t i = (int64_t)m * N + n;
            out[i] = ik_f2b(ik_round_bf16(ik_b2f(routed[i]) + sv));
        }
    }
}

template <int M>
void run_m(const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, const ik_bf16* routed, const ik_bf16* logit,
           ik_bf16* out, int64_t N, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    shared_down_combine_kernel<M><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, bs, gs, routed, logit, out, N, (int)K);
}

}  // namespace

extern "C" {

// x [M, K] bf16 (16-byte aligned), w [N, K/2] u8, bs [N, K/16] e4m3, gs [1] f32, routed [M, N] bf16, logit [M, 1] bf16,
// out [M, N] bf16; 1 <= M <= 8 (the rows matmul_nvfp4_gemv_v2 takes), K % 16 == 0.
int ik_shared_down_combine_nvfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs,
                                      const ik_tensor* routed, const ik_tensor* logit, ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 8));
    IK_TRY(ik_check_a(bs, IK_DTYPE_FP8_E4M3, 2, 1));
    IK_TRY(ik_check_a(gs, IK_DTYPE_F32, 1, 4));
    IK_TRY(ik_check_a(routed, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(logit, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(out, IK_DTYPE_BF16, 2, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 16 != 0 || w->shape[1] != K / 2 || bs->shape[0] != N || bs->shape[1] != K / 16 || gs->shape[0] != 1 ||
        routed->shape[0] != M || routed->shape[1] != N || logit->shape[0] != M || logit->shape[1] != 1 || out->shape[0] != M ||
        out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M < 1 || M > 8 || N == 0 || K == 0) return IK_ERR_SHAPE;  // (the unfused ops handle every other shape)
    if (K > 0x7fffffffLL || N > 0x7fffffffLL * kWarps) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* bp = (const uint8_t*)bs->data;
    const float* gp = (const float*)gs->data;
    const ik_bf16* rp = (const ik_bf16*)routed->data;
    const ik_bf16* lp = (const ik_bf16*)logit->data;
    ik_bf16* op = (ik_bf16*)out->data;
    switch (M) {
        case 1: run_m<1>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
        case 2: run_m<2>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
        case 3: run_m<3>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
        case 4: run_m<4>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
        case 5: run_m<5>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
        case 6: run_m<6>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
        case 7: run_m<7>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
        default: run_m<8>(xp, wp, bp, gp, rp, lp, op, N, K, s); break;
    }
    return ik_launch_status();
}

}  // extern "C"
