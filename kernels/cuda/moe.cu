// Qwen3.5/3.6 sparse-MoE block kernels (docs/quant-moe.md). Activations bf16.
//
//   moe_topk      per token: the K largest router logits (ties: lowest expert index first), softmax weights over
//                 the selected logits (== renormalised top-k of the full softmax), rounded to bf16 like HF.
//   moe_experts   NVFP4 (W4A16) routed experts for all (token, slot) pairs:
//                   group   counting-sort the pairs by expert, compact list of the active experts,
//                   gate_up one launch over active experts x output rows: each warp streams its gate row and up row
//                           once and applies them to up to 8 pairs of that expert (weights are read once per
//                           expert, not once per token), h = silu(gate) * up,
//                   down    same for the down projection, y[pair] = down(h[pair]),
//                   reduce  out[t] = sum_k round(y[t,k] * w[t,k]) in fixed k order (no atomics on `out`).
//                 Grids are sized by min(T*K, E) active experts, read from the grouping output, so decode (T=1)
//                 launches 8 expert slots, not 256. Every pair's arithmetic is independent of the grouping, so
//                 the result is bit-identical run to run even though the grouping itself uses atomics.
//   moe_combine   out = routed + sigmoid(gate_logit) * shared with HF's rounding points.
//
// Scratch layout (must match intarsia_core::quant::ops::moe_experts_scratch, regions rounded to 256 bytes):
//   pairs int[T*K] | offsets int[E+1] | active int[E+1] (last = count) | h bf16[T*K, I] | y bf16[T*K, H]
#include "ik_launch.h"
#include "ik_quant.cuh"
#include <math.h>

namespace {

constexpr int kMaxExperts = 1024;  // grouping kernel: one thread per expert
constexpr int kPairChunk = 8;      // pairs sharing one pass over an expert row

IK_DEVI float round_bf16(float v) { return ik_b2f(ik_f2b(v)); }

// ---- top-k ---------------------------------------------------------------------------------------------------------

constexpr int kTopkMaxPerLane = 16;  // E <= 512
constexpr int kTopkWarps = 4;

__global__ void __launch_bounds__(kTopkWarps * IK_WARP)
moe_topk_kernel(const ik_bf16* __restrict__ logits, int T, int E, int K, int norm, float* __restrict__ w_out,
                int* __restrict__ id_out) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int t = blockIdx.x * kTopkWarps + warp;
    if (t >= T) return;
    float v[kTopkMaxPerLane];
    bool sel[kTopkMaxPerLane];
#pragma unroll
    for (int j = 0; j < kTopkMaxPerLane; ++j) {
        const int e = lane + IK_WARP * j;
        v[j] = e < E ? ik_b2f(logits[(int64_t)t * E + e]) : -INFINITY;
        sel[j] = false;
    }
    __shared__ float s_val[kTopkWarps][IK_WARP];
    __shared__ int s_id[kTopkWarps][IK_WARP];
    float mx = 0.f, denom = 0.f;
    for (int kk = 0; kk < K; ++kk) {
        float best = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < kTopkMaxPerLane; ++j) {
            if (!sel[j] && v[j] > best) {  // strict: the first (lowest) expert wins ties within a lane
                best = v[j];
                bi = lane + IK_WARP * j;
            }
        }
#pragma unroll
        for (int o = IK_WARP / 2; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (kk == 0) mx = best;
        if ((bi & (IK_WARP - 1)) == lane) {
#pragma unroll
            for (int j = 0; j < kTopkMaxPerLane; ++j)
                if (j == (bi >> 5)) sel[j] = true;
        }
        if (lane == 0) {
            s_val[warp][kk & 31] = best;  // K <= 32 (checked by the launcher)
            s_id[warp][kk & 31] = bi;
        }
        denom += expf(best - mx);
    }
    if (!norm) {  // softmax over all experts
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < kTopkMaxPerLane; ++j) {
            const int e = lane + IK_WARP * j;
            if (e < E) part += expf(v[j] - mx);
        }
        denom = ik_warp_sum(part);
    }
    __syncwarp();
    if (lane < K) {
        const float w = expf(s_val[warp][lane] - mx) / denom;
        w_out[(int64_t)t * K + lane] = round_bf16(w);
        id_out[(int64_t)t * K + lane] = s_id[warp][lane];
    }
}

// ---- grouping ------------------------------------------------------------------------------------------------------

// exclusive prefix sum of one int per thread over a 1024-thread block; `wsum` = 32 ints of shared scratch
IK_DEVI int block_excl_scan(int v, int* wsum, int* total) {
    const int lane = threadIdx.x % IK_WARP, warp = threadIdx.x / IK_WARP, nw = blockDim.x / IK_WARP;
    int inc = v;
#pragma unroll
    for (int o = 1; o < IK_WARP; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += t;
    }
    __syncthreads();  // wsum may still be read by a previous scan
    if (lane == IK_WARP - 1) wsum[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int x = lane < nw ? wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < IK_WARP; o <<= 1) {
            const int t = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += t;
        }
        wsum[lane] = x;
    }
    __syncthreads();
    *total = wsum[nw - 1];
    return inc - v + (warp ? wsum[warp - 1] : 0);
}

__global__ void __launch_bounds__(1024)
moe_group_kernel(const int* __restrict__ ids, int P, int E, int* __restrict__ pairs, int* __restrict__ offsets,
                 int* __restrict__ active) {
    extern __shared__ int sh[];  // counts[E], cursor[E], wsum[32]
    int* counts = sh;
    int* cursor = sh + E;
    int* wsum = sh + 2 * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) counts[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < E) atomicAdd(&counts[e], 1);
    }
    __syncthreads();
    const int e = threadIdx.x;
    const int c = e < E ? counts[e] : 0;
    int total = 0, total_active = 0;
    const int off = block_excl_scan(c, wsum, &total);
    if (e < E) {
        offsets[e] = off;
        cursor[e] = off;
    }
    if (e == 0) offsets[E] = total;
    const int apos = block_excl_scan(c > 0 ? 1 : 0, wsum, &total_active);
    if (e < E && c > 0) active[apos] = e;
    if (e == 0) active[E] = total_active;
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const int ex = ids[p];
        if (ex >= 0 && ex < E) pairs[atomicAdd(&cursor[ex], 1)] = p;
    }
}

// ---- expert projections ----------------------------------------------------------------------------------------------

constexpr int kRowWarps = 8;

// Stage `pc` rows of 16-byte-aligned bf16 data (`rows[q]` = element offset of row q, length `len`) into shared memory.
IK_DEVI void stage_rows(ik_bf16* dst, const ik_bf16* src, const int* pairs_chunk, int pc, int len, int div) {
    const int per = len / 8;
    for (int i = threadIdx.x; i < pc * per; i += blockDim.x) {
        const int q = i / per, j = i % per;
        const int row = pairs_chunk[q] / div;
        reinterpret_cast<uint4*>(dst)[q * per + j] = reinterpret_cast<const uint4*>(src)[(int64_t)row * per + j];
    }
}

template <bool GELU>
__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_gate_up_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ guw, const uint8_t* __restrict__ gus,
                   const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                   const int* __restrict__ active, ik_bf16* __restrict__ h, int E, int H, int I, int K, int PC) {
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* xs = reinterpret_cast<ik_bf16*>(smem);  // [PC][H]
    const int slot = blockIdx.x;
    if (slot >= active[E]) return;  // block-uniform
    const int e = active[slot];
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    const bool valid = r < I;
    const int rr = valid ? r : 0;
    const uint2* wg = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + rr) * (H / 2));
    const uint2* wu = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + I + rr) * (H / 2));
    const uint8_t* sg = gus + ((int64_t)e * 2 * I + rr) * (H / 16);
    const uint8_t* su = gus + ((int64_t)e * 2 * I + I + rr) * (H / 16);
    const float gg = gscales[e * 3 + 0], gu = gscales[e * 3 + 1];
    for (int c0 = 0; c0 < cnt; c0 += PC) {
        const int pc = min(PC, cnt - c0);
        __syncthreads();
        stage_rows(xs, x, pairs + begin + c0, pc, H, K);
        __syncthreads();
        if (valid) {
            float ag[kPairChunk], au[kPairChunk];
#pragma unroll
            for (int q = 0; q < kPairChunk; ++q) ag[q] = au[q] = 0.f;
#pragma unroll 2
            for (int u = lane; u < H / 16; u += IK_WARP) {
                const uint2 wgv = __ldcs(wg + u), wuv = __ldcs(wu + u);
                const float sgv = ik_e4m3_to_f(sg[u]), suv = ik_e4m3_to_f(su[u]);
                float fg[16], fu[16];
                ik_fp4x8_to_f32(wgv.x, fg);
                ik_fp4x8_to_f32(wgv.y, fg + 8);
                ik_fp4x8_to_f32(wuv.x, fu);
                ik_fp4x8_to_f32(wuv.y, fu + 8);
#pragma unroll
                for (int q = 0; q < kPairChunk; ++q) {
                    if (q < pc) {
                        const uint4* xp = reinterpret_cast<const uint4*>(xs + (int64_t)q * H + u * 16);
                        float xf[16];
                        ik_unpack8(xp[0], xf);
                        ik_unpack8(xp[1], xf + 8);
                        float pg = 0.f, pu = 0.f;
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            pg = fmaf(fg[i], xf[i], pg);
                            pu = fmaf(fu[i], xf[i], pu);
                        }
                        ag[q] = fmaf(pg, sgv, ag[q]);
                        au[q] = fmaf(pu, suv, au[q]);
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < kPairChunk; ++q) {
                if (q < pc) {
                    const float g = round_bf16(ik_warp_sum(ag[q]) * gg);
                    const float u = round_bf16(ik_warp_sum(au[q]) * gu);
                    const float a = round_bf16(GELU ? ik_gelu_tanh(g) : g / (1.f + expf(-g)));
                    if (lane == 0) h[(int64_t)pairs[begin + c0 + q] * I + r] = ik_f2b(a * u);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_down_kernel(const ik_bf16* __restrict__ h, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                const int* __restrict__ active, ik_bf16* __restrict__ y, int E, int H, int I, int PC) {
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* hs = reinterpret_cast<ik_bf16*>(smem);  // [PC][I]
    const int slot = blockIdx.x;
    if (slot >= active[E]) return;
    const int e = active[slot];
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    const bool valid = r < H;
    const int rr = valid ? r : 0;
    const uint2* wd = reinterpret_cast<const uint2*>(dnw + ((int64_t)e * H + rr) * (I / 2));
    const uint8_t* sd = dns + ((int64_t)e * H + rr) * (I / 16);
    const float gd = gscales[e * 3 + 2];
    for (int c0 = 0; c0 < cnt; c0 += PC) {
        const int pc = min(PC, cnt - c0);
        __syncthreads();
        stage_rows(hs, h, pairs + begin + c0, pc, I, 1);
        __syncthreads();
        if (valid) {
            float acc[kPairChunk];
#pragma unroll
            for (int q = 0; q < kPairChunk; ++q) acc[q] = 0.f;
#pragma unroll 2
            for (int u = lane; u < I / 16; u += IK_WARP) {
                const uint2 wv = __ldcs(wd + u);
                const float sv = ik_e4m3_to_f(sd[u]);
                float fw[16];
                ik_fp4x8_to_f32(wv.x, fw);
                ik_fp4x8_to_f32(wv.y, fw + 8);
#pragma unroll
                for (int q = 0; q < kPairChunk; ++q) {
                    if (q < pc) {
                        const uint4* hp = reinterpret_cast<const uint4*>(hs + (int64_t)q * I + u * 16);
                        float hf[16];
                        ik_unpack8(hp[0], hf);
                        ik_unpack8(hp[1], hf + 8);
                        float p = 0.f;
#pragma unroll
                        for (int i = 0; i < 16; ++i) p = fmaf(fw[i], hf[i], p);
                        acc[q] = fmaf(p, sv, acc[q]);
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < kPairChunk; ++q) {
                if (q < pc) {
                    const float v = ik_warp_sum(acc[q]) * gd;
                    if (lane == 0) y[(int64_t)pairs[begin + c0 + q] * H + r] = ik_f2b(v);
                }
            }
        }
    }
}

template <typename OutT>
__global__ void __launch_bounds__(256)
moe_reduce_kernel(const ik_bf16* __restrict__ y, const int* __restrict__ ids, const float* __restrict__ w,
                  OutT* __restrict__ out, int E, int H, int K) {
    const int t = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    if (c >= H) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int id = ids[(int64_t)t * K + k];
        if (id < 0 || id >= E) continue;
        acc += round_bf16(ik_b2f(y[((int64_t)t * K + k) * H + c]) * w[(int64_t)t * K + k]);
    }
    ik_store(out + (int64_t)t * H + c, acc);
}

// ---- combine -------------------------------------------------------------------------------------------------------

template <typename OutT>
__global__ void __launch_bounds__(256)
moe_combine_kernel(const ik_bf16* __restrict__ routed, const ik_bf16* __restrict__ shared,
                   const ik_bf16* __restrict__ gate, OutT* __restrict__ out, int64_t T, int64_t H) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int64_t t = i / H;
    const float g = ik_b2f(gate[t]);
    const float sg = round_bf16(1.f / (1.f + expf(-g)));
    const float s = round_bf16(sg * ik_b2f(shared[i]));
    ik_store(out + i, round_bf16(ik_b2f(routed[i]) + s));
}

// ---- launchers ---------------------------------------------------------------------------------------------------

inline int64_t r256(int64_t v) { return (v + 255) & ~int64_t(255); }

struct ScratchLayout {
    int64_t pairs, offsets, active, h, y, total;
};

ScratchLayout scratch_layout(int64_t P, int64_t E, int64_t H, int64_t I) {
    ScratchLayout l;
    l.pairs = 0;
    l.offsets = l.pairs + r256(P * 4);
    l.active = l.offsets + r256((E + 1) * 4);
    l.h = l.active + r256((E + 1) * 4);
    l.y = l.h + r256(P * I * 2);
    l.total = l.y + r256(P * H * 2);
    return l;
}

}  // namespace

extern "C" {

int ik_moe_topk_bf16(const ik_tensor* logits, int top_k, int norm, ik_tensor* weights, ik_tensor* ids, void* stream) {
    IK_TRY(ik_check_a(logits, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(weights, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(ids, IK_DTYPE_I32, 2, 4));
    const int64_t T = logits->shape[0], E = logits->shape[1];
    if (top_k < 1 || top_k > 32 || top_k > E || E > IK_WARP * kTopkMaxPerLane) return IK_ERR_SHAPE;
    if (weights->shape[0] != T || weights->shape[1] != top_k || ids->shape[0] != T || ids->shape[1] != top_k)
        return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    if (T > 0x7fffffffLL) return IK_ERR_SHAPE;
    const unsigned grid = (unsigned)((T + kTopkWarps - 1) / kTopkWarps);
    moe_topk_kernel<<<grid, kTopkWarps * IK_WARP, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)logits->data, (int)T, (int)E, top_k, norm, (float*)weights->data, (int*)ids->data);
    return ik_launch_status();
}

// Shared by ik_moe_experts_bf16 and ik_moe_experts_gelu_bf16 (gelu_tanh instead of SiLU in the expert activation).
static int moe_experts_launch(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                              const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                              const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                              const ik_scratch* scratch, void* stream, bool gelu) {
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
    if (E < 1 || E > kMaxExperts || K < 1 || K > E) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    const int64_t P = T * K;
    if (P > 0x7fffffffLL || (int64_t)H * T > 0x7fffffffLL * 2) return IK_ERR_SHAPE;
    const ScratchLayout L = scratch_layout(P, E, H, I);
    if (!scratch || !scratch->data || scratch->bytes < L.total) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    // pairs per pass: as many as fit in 32 KiB of shared memory, at most kPairChunk
    const int64_t row_bytes = 2 * (H > I ? H : I);
    int PC = (int)(32768 / row_bytes);
    if (PC > kPairChunk) PC = kPairChunk;
    if (PC < 1) return IK_ERR_SHAPE;
    unsigned char* base = (unsigned char*)scratch->data;
    int* pairs = (int*)(base + L.pairs);
    int* offsets = (int*)(base + L.offsets);
    int* active = (int*)(base + L.active);
    ik_bf16* hbuf = (ik_bf16*)(base + L.h);
    ik_bf16* ybuf = (ik_bf16*)(base + L.y);
    cudaStream_t s = IK_STREAM(stream);
    const unsigned slots = (unsigned)(P < E ? P : E);

    moe_group_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), s>>>((const int*)ids->data, (int)P, (int)E, pairs, offsets, active);
    IK_TRY(ik_launch_status());
    if (gelu) {
        moe_gate_up_kernel<true><<<dim3(slots, (unsigned)((I + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                                   (size_t)PC * H * 2, s>>>((const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data,
                                                           (const uint8_t*)gate_up_s->data, (const float*)gscales->data,
                                                           pairs, offsets, active, hbuf, (int)E, (int)H, (int)I, (int)K, PC);
    } else {
        moe_gate_up_kernel<false><<<dim3(slots, (unsigned)((I + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                                    (size_t)PC * H * 2, s>>>((const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data,
                                                            (const uint8_t*)gate_up_s->data, (const float*)gscales->data,
                                                            pairs, offsets, active, hbuf, (int)E, (int)H, (int)I, (int)K, PC);
    }
    IK_TRY(ik_launch_status());
    moe_down_kernel<<<dim3(slots, (unsigned)((H + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                      (size_t)PC * I * 2, s>>>(hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data,
                                              (const float*)gscales->data, pairs, offsets, active, ybuf, (int)E, (int)H,
                                              (int)I, PC);
    IK_TRY(ik_launch_status());
    moe_reduce_kernel<ik_bf16><<<dim3((unsigned)T, (unsigned)((H + 255) / 256)), 256, 0, s>>>(
        ybuf, (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)K);
    return ik_launch_status();
}

int ik_moe_experts_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                        const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                        const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return moe_experts_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, false);
}

// gelu_tanh in the expert activation (OpAttrs::gelu); otherwise the same bits as ik_moe_experts_bf16.
int ik_moe_experts_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                             const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                             const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                             void* stream) {
    return moe_experts_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true);
}

int ik_moe_combine_bf16(const ik_tensor* routed, const ik_tensor* shared, const ik_tensor* gate_logit, ik_tensor* out,
                        void* stream) {
    IK_TRY(ik_check_a(routed, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(shared, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(gate_logit, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(out, IK_DTYPE_BF16, 2, 2));
    const int64_t T = routed->shape[0], H = routed->shape[1];
    if (shared->shape[0] != T || shared->shape[1] != H || gate_logit->shape[0] != T || gate_logit->shape[1] != 1 ||
        out->shape[0] != T || out->shape[1] != H)
        return IK_ERR_SHAPE;
    if (T * H == 0) return IK_OK;
    const int64_t n = T * H;
    moe_combine_kernel<ik_bf16><<<(unsigned)((n + 255) / 256), 256, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)routed->data, (const ik_bf16*)shared->data, (const ik_bf16*)gate_logit->data,
        (ik_bf16*)out->data, T, H);
    return ik_launch_status();
}

}  // extern "C"
