// Decode-time MoE experts (T <= 4 tokens): the same op and the same arithmetic as moe_experts_bf16 (moe.cu), in two
// launches instead of four. With a handful of tokens the experts hardly overlap, so grouping the pairs by expert
// buys nothing and costs a launch; the per-token reduce folds into the down projection.
//
//   gate_up  grid (pair, I/8): the 8 warps of a block each stream one gate row and its up row of the pair's expert
//            (x staged in shared memory once per block), h[pair] = silu(gate) * up.
//   down     grid (token, H/8): each warp owns one output row, walks the token's K experts, and sums
//            round(round(down_e(h_k)) * w_k) over k in fixed order: no atomics, no y buffer, no reduce launch.
//
// Per-pair arithmetic is identical to moe.cu (same lane-strided f32 accumulation, same rounding points).
// Scratch (intarsia_core::quant::ops::moe_experts_decode_scratch): h bf16[T*K, I], rounded to 256 bytes.
#include "ik_launch.h"
#include "ik_quant.cuh"

namespace {

constexpr int kRowWarps = 8;
constexpr int kMaxK = 16;  // experts per token held in registers by the down kernel

IK_DEVI float round_bf16(float v) { return ik_b2f(ik_f2b(v)); }

__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_dec_gate_up_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ guw, const uint8_t* __restrict__ gus,
                       const float* __restrict__ gscales, const int* __restrict__ ids, ik_bf16* __restrict__ h, int E,
                       int H, int I, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* xs = reinterpret_cast<ik_bf16*>(smem);  // [H]
    const int p = blockIdx.x;
    const int e = ids[p];
    if (e < 0 || e >= E) return;  // block-uniform: an invalid id contributes nothing (the reduce skips it too)
    const uint4* xg = reinterpret_cast<const uint4*>(x + (int64_t)(p / K) * H);
    for (int i = threadIdx.x; i < H / 8; i += blockDim.x) reinterpret_cast<uint4*>(xs)[i] = xg[i];
    __syncthreads();
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    if (r >= I) return;
    const uint2* wg = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + r) * (H / 2));
    const uint2* wu = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + I + r) * (H / 2));
    const uint8_t* sg = gus + ((int64_t)e * 2 * I + r) * (H / 16);
    const uint8_t* su = gus + ((int64_t)e * 2 * I + I + r) * (H / 16);
    float ag = 0.f, au = 0.f;
#pragma unroll 2
    for (int u = lane; u < H / 16; u += IK_WARP) {
        const uint2 wgv = __ldcs(wg + u), wuv = __ldcs(wu + u);
        const float sgv = ik_e4m3_to_f(sg[u]), suv = ik_e4m3_to_f(su[u]);
        float fg[16], fu[16];
        ik_fp4x8_to_f32(wgv.x, fg);
        ik_fp4x8_to_f32(wgv.y, fg + 8);
        ik_fp4x8_to_f32(wuv.x, fu);
        ik_fp4x8_to_f32(wuv.y, fu + 8);
        const uint4* xp = reinterpret_cast<const uint4*>(xs + u * 16);
        float xf[16];
        ik_unpack8(xp[0], xf);
        ik_unpack8(xp[1], xf + 8);
        float pg = 0.f, pu = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            pg = fmaf(fg[i], xf[i], pg);
            pu = fmaf(fu[i], xf[i], pu);
        }
        ag = fmaf(pg, sgv, ag);
        au = fmaf(pu, suv, au);
    }
    const float g = round_bf16(ik_warp_sum(ag) * gscales[e * 3 + 0]);
    const float u = round_bf16(ik_warp_sum(au) * gscales[e * 3 + 1]);
    const float a = round_bf16(g / (1.f + expf(-g)));
    if (lane == 0) h[(int64_t)p * I + r] = ik_f2b(a * u);
}

__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_dec_down_kernel(const ik_bf16* __restrict__ h, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                    const float* __restrict__ gscales, const int* __restrict__ ids, const float* __restrict__ wts,
                    ik_bf16* __restrict__ out, int E, int H, int I, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* hs = reinterpret_cast<ik_bf16*>(smem);  // [K][I]: the token's K h rows are contiguous in `h`
    const int t = blockIdx.x;
    const uint4* hg = reinterpret_cast<const uint4*>(h + (int64_t)t * K * I);
    for (int i = threadIdx.x; i < K * I / 8; i += blockDim.x) reinterpret_cast<uint4*>(hs)[i] = hg[i];
    __syncthreads();
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    if (r >= H) return;
    int id[kMaxK];
    float acc[kMaxK];
#pragma unroll
    for (int k = 0; k < kMaxK; ++k) {
        id[k] = -1;
        acc[k] = 0.f;
        if (k < K) {
            const int v = ids[(int64_t)t * K + k];
            id[k] = (v >= 0 && v < E) ? v : -1;
        }
    }
    // u outermost: the K independent weight loads of one step are in flight together
#pragma unroll 1
    for (int u = lane; u < I / 16; u += IK_WARP) {
#pragma unroll
        for (int k = 0; k < kMaxK; ++k) {
            if (id[k] >= 0) {
                const int64_t row = (int64_t)id[k] * H + r;
                const uint2 wv = __ldcs(reinterpret_cast<const uint2*>(dnw + row * (I / 2)) + u);
                const float sv = ik_e4m3_to_f(dns[row * (I / 16) + u]);
                float fw[16], hf[16];
                ik_fp4x8_to_f32(wv.x, fw);
                ik_fp4x8_to_f32(wv.y, fw + 8);
                const uint4* hp = reinterpret_cast<const uint4*>(hs + (int64_t)k * I + u * 16);
                ik_unpack8(hp[0], hf);
                ik_unpack8(hp[1], hf + 8);
                float pdot = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) pdot = fmaf(fw[i], hf[i], pdot);
                acc[k] = fmaf(pdot, sv, acc[k]);
            }
        }
    }
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < kMaxK; ++k) {
        if (id[k] >= 0) {  // warp-uniform
            const float v = ik_warp_sum(acc[k]) * gscales[id[k] * 3 + 2];
            total += round_bf16(round_bf16(v) * wts[(int64_t)t * K + k]);
        }
    }
    if (lane == 0) out[(int64_t)t * H + r] = ik_f2b(total);
}

inline int64_t r256(int64_t v) { return (v + 255) & ~int64_t(255); }

}  // namespace

extern "C" {

int ik_moe_experts_decode_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                               const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                               const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                               const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(ids, IK_DTYPE_I32, 2, 4));
    IK_TRY(ik_check_a(weights, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(gate_up_w, IK_DTYPE_U8, 3, 8));
    IK_TRY(ik_check_a(gate_up_s, IK_DTYPE_FP8_E4M3, 3, 1));
    IK_TRY(ik_check_a(down_w, IK_DTYPE_U8, 3, 8));
    IK_TRY(ik_check_a(down_s, IK_DTYPE_FP8_E4M3, 3, 1));
    IK_TRY(ik_check_a(gscales, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(out, IK_DTYPE_BF16, 2, 2));
    const int64_t T = x->shape[0], H = x->shape[1], K = ids->shape[1];
    const int64_t E = gate_up_w->shape[0], I = gate_up_w->shape[1] / 2;
    if (H % 16 != 0 || I % 16 != 0 || gate_up_w->shape[1] != 2 * I || gate_up_w->shape[2] != H / 2 ||
        gate_up_s->shape[0] != E || gate_up_s->shape[1] != 2 * I || gate_up_s->shape[2] != H / 16 ||
        down_w->shape[0] != E || down_w->shape[1] != H || down_w->shape[2] != I / 2 || down_s->shape[0] != E ||
        down_s->shape[1] != H || down_s->shape[2] != I / 16 || gscales->shape[0] != E || gscales->shape[1] != 3 ||
        ids->shape[0] != T || weights->shape[0] != T || weights->shape[1] != K || out->shape[0] != T ||
        out->shape[1] != H)
        return IK_ERR_SHAPE;
    if (E < 1 || E > 0x7fffffffLL / 3 || K < 1 || K > E || K > kMaxK) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    const int64_t P = T * K;
    if (P > 0x7fffffffLL / 2 || (size_t)(K * I * 2) > 48 * 1024 || (size_t)(H * 2) > 48 * 1024) return IK_ERR_SHAPE;
    if (!scratch || !scratch->data || scratch->bytes < r256(P * I * 2)) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    ik_bf16* hbuf = (ik_bf16*)scratch->data;
    cudaStream_t s = IK_STREAM(stream);
    moe_dec_gate_up_kernel<<<dim3((unsigned)P, (unsigned)((I + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                             (size_t)H * 2, s>>>((const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data,
                                                 (const uint8_t*)gate_up_s->data, (const float*)gscales->data,
                                                 (const int*)ids->data, hbuf, (int)E, (int)H, (int)I, (int)K);
    IK_TRY(ik_launch_status());
    moe_dec_down_kernel<<<dim3((unsigned)T, (unsigned)((H + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                          (size_t)K * I * 2, s>>>(hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data,
                                                   (const float*)gscales->data, (const int*)ids->data,
                                                   (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H,
                                                   (int)I, (int)K);
    return ik_launch_status();
}

}  // extern "C"
