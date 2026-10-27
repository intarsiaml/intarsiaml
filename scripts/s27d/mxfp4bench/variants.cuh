// s27d experimental MXFP4 GEMV variants; included after kernels/cuda/mxfp4.cu in the same translation unit (reuses its helpers)
namespace {
// ---- experimental variants (s27d): the lane's groups are processed in batches of B; the B weight words and scale bytes are loaded
// before the first of them is used (more bytes in flight per lane), the arithmetic per group is the baseline's, in the baseline's order.
template <typename OutT, int B>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_gemv_pipe_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                  const float* __restrict__ act, OutT* __restrict__ out, int64_t M, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;
    const int64_t m0 = (int64_t)blockIdx.y * kMB;
    const int mc = (int)((M - m0) < kMB ? (M - m0) : kMB);
    const bool qdq = act[0] != 0.f;
    const uint4* wr = reinterpret_cast<const uint4*>(w + n * (K / 2));
    const uint8_t* sr = sc + n * (K / kG);
    float acc[kMB];
#pragma unroll
    for (int m = 0; m < kMB; ++m) acc[m] = 0.f;
    const int G = K / kG;
    for (int u0 = lane; u0 < G; u0 += IK_WARP * B) {
        uint4 wv[B];
        uint32_t sb[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int u = u0 + b * IK_WARP;
            if (u < G) { wv[b] = __ldcs(wr + u); sb[b] = sr[u]; }
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int u = u0 + b * IK_WARP;
            if (u < G) {
                const float s = mx_e8m0(sb[b]);
                float wf[kG];
                ik_fp4x8_to_f32(wv[b].x, wf);
                ik_fp4x8_to_f32(wv[b].y, wf + 8);
                ik_fp4x8_to_f32(wv[b].z, wf + 16);
                ik_fp4x8_to_f32(wv[b].w, wf + 24);
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
        }
    }
#pragma unroll
    for (int m = 0; m < kMB; ++m) {
        const float t = ik_warp_sum(acc[m]);
        if (lane == 0 && m < mc) ik_store(out + (m0 + m) * N + n, t);
    }
}

}  // namespace
