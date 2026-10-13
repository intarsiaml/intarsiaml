// Causal GQA attention over a contiguous bf16 KV cache [Nkv, cap, D].
//
// One block per (query token, head), online softmax over key tiles of kThreads keys:
//   phase 1: thread j of the tile computes the score of key (tile + j)  (q lives in shared memory)
//   phase 2: thread d accumulates sum_j p_j * V[j][d] for its output dims
// Simple and correct first; split-KV / tensor-core versions are later kernel-pack work.
#include "ik_hip.h"

namespace {

constexpr int kThreads = 128;   // keys per tile == threads per block
constexpr int kMaxD = 256;      // head_dim limit (two output dims per thread)

template <bool VEC>
__global__ void __launch_bounds__(kThreads)
attn_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
            ik_bf16* __restrict__ out, int64_t N, int64_t NKV, int D, int64_t cap, int64_t pos0, float scale) {
    __shared__ float qs[kMaxD];
    __shared__ float sc[kThreads];
    __shared__ float red[kThreads / IK_WARP + 1];

    const int64_t t = blockIdx.x / N;
    const int64_t h = blockIdx.x % N;
    const int64_t kvh = h / (N / NKV);
    const int64_t visible = pos0 + t + 1;  // keys 0 .. pos0+t
    const ik_bf16* kbase = kc + kvh * cap * D;
    const ik_bf16* vbase = vc + kvh * cap * D;
    const ik_bf16* qrow = q + (t * N + h) * D;

    for (int d = threadIdx.x; d < D; d += kThreads) qs[d] = ik_b2f(qrow[d]);
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float acc[kMaxD / kThreads] = {0.f, 0.f};

    for (int64_t tile = 0; tile < visible; tile += kThreads) {
        const int64_t j = tile + threadIdx.x;
        float s = -INFINITY;
        if (j < visible) {
            const ik_bf16* kr = kbase + j * D;
            float dot = 0.f;
            if (VEC) {
                for (int d = 0; d < D; d += 8) {
                    float f[8];
                    ik_unpack8(*reinterpret_cast<const uint4*>(kr + d), f);
#pragma unroll
                    for (int e = 0; e < 8; ++e) dot += qs[d + e] * f[e];
                }
            } else {
                for (int d = 0; d < D; ++d) dot += qs[d] * ik_b2f(kr[d]);
            }
            s = dot * scale;
        }
        const float tmax = ik_block_max(s, red);
        const float m_new = fmaxf(m, tmax);
        const float p = (j < visible) ? expf(s - m_new) : 0.f;
        const float corr = expf(m - m_new);  // m == -inf on the first tile -> 0
        sc[threadIdx.x] = p;
        const float tsum = ik_block_sum(p, red);  // also orders the sc[] write before the reads below
        l = l * corr + tsum;
        m = m_new;

        const int64_t n_in_tile = (visible - tile < kThreads) ? (visible - tile) : kThreads;
#pragma unroll
        for (int u = 0; u < kMaxD / kThreads; ++u) {
            const int d = threadIdx.x + u * kThreads;
            if (d < D) {
                float a = acc[u] * corr;
                const ik_bf16* vcol = vbase + tile * D + d;
#pragma unroll 8
                for (int jj = 0; jj < n_in_tile; ++jj) a += sc[jj] * ik_b2f(vcol[(int64_t)jj * D]);
                acc[u] = a;
            }
        }
        __syncthreads();  // sc[] is rewritten by the next tile
    }

    const float inv = 1.f / l;
#pragma unroll
    for (int u = 0; u < kMaxD / kThreads; ++u) {
        const int d = threadIdx.x + u * kThreads;
        if (d < D) out[(t * N + h) * D + d] = ik_f2b(acc[u] * inv);
    }
}

int attn_launch(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                ik_tensor* out, void* stream, bool decode) {
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(kc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(vc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc->shape[0], cap = kc->shape[1];
    if (decode && T != 1) return IK_ERR_SHAPE;
    if (vc->shape[0] != NKV || vc->shape[1] != cap || vc->shape[2] != D || kc->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || D <= 0 || D > kMaxD || pos0 < 0 || pos0 + T > cap) return IK_ERR_SHAPE;
    if (T == 0 || N == 0) return IK_OK;
    const int64_t nblocks = T * N;
    if (nblocks > 0x7fffffffLL) return IK_ERR_SHAPE;
    const bool vec = D % 8 == 0 && ik_aligned16(kc->data) && ik_aligned16(vc->data) && cap * D % 8 == 0;
    hipStream_t s = IK_STREAM(stream);
    if (vec)
        attn_kernel<true><<<(unsigned)nblocks, kThreads, 0, s>>>((const ik_bf16*)q->data, (const ik_bf16*)kc->data,
                                                                  (const ik_bf16*)vc->data, (ik_bf16*)out->data, N,
                                                                  NKV, (int)D, cap, pos0, scale);
    else
        attn_kernel<false><<<(unsigned)nblocks, kThreads, 0, s>>>((const ik_bf16*)q->data, (const ik_bf16*)kc->data,
                                                                   (const ik_bf16*)vc->data, (ik_bf16*)out->data, N,
                                                                   NKV, (int)D, cap, pos0, scale);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_attn_prefill_bf16(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                         ik_tensor* out, void* stream) {
    return attn_launch(q, kc, vc, pos0, scale, out, stream, false);
}

int ik_attn_decode_bf16(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                        ik_tensor* out, void* stream) {
    return attn_launch(q, kc, vc, pos0, scale, out, stream, true);
}

}  // extern "C"
