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

// ---- row-grouped decode experts (lane msd) ---------------------------------------------------------------------------
//
// The multi-token (rows) decode launchers, 2..16 tokens: tokens grouped by expert (moe_group_kernel above), each expert's weights
// read and dequantised once per up to kGrpPairs pairs, instead of once per pair as in moe_decode.cu's per-pair kernels (which spend
// most of their time re-dequantising the same fp4 weights when tokens share experts). Per-pair arithmetic is the per-pair kernels':
// lane-strided f32 accumulation (the same u order, the same 16-element fma chain, the block scale applied per chunk), the same
// roundings. The cross-lane sum is the same xor butterfly (16, 8, 4, 2, 1) on every value, computed with fewer shuffles
// (grp_warp_sum), so a pair's result does not depend on which other pairs share the expert or on how many there are.

constexpr int kGrpPairs = 16;  // most pairs of one expert per pass (= the most rows of a rows launch)
constexpr unsigned kGrpSlotBlocks = 12;  // slot-group blocks of the grouped launches (x row groups: a few waves of blocks)

// ik_warp_sum on each of the NV values (NV a power of two, <= 32) at once, with NV - 1 + log2(32 / NV) shuffles instead of
// NV * 5: at each xor offset o (16, 8, 4, ..) a lane keeps one half of its values and sends the other to its partner, who keeps
// that half: the sums are the butterfly's (v_l + v_{l ^ o} of the same value, commutative), the lane just does not compute the
// ones it would not use. Lane l returns the total of value (l >> (5 - log2 NV)).
template <int NV>
IK_DEVI float grp_warp_sum(float (&v)[NV], int lane) {
    int o = IK_WARP / 2;
#pragma unroll
    for (int n = NV; n > 1; n >>= 1, o >>= 1) {
        const bool hi = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < n / 2; ++i) {
            const float send = hi ? v[i] : v[i + n / 2];
            const float keep = hi ? v[i + n / 2] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
#pragma unroll
    for (; o > 0; o >>= 1) v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
    return v[0];
}

// 5 - log2(NQ): a lane of grp_warp_sum<NQ> owns value (lane >> kGrpShift<NQ>::v)
template <int NQ>
struct GrpShift {
    static constexpr int v = NQ == 16 ? 1 : NQ == 8 ? 2 : NQ == 4 ? 3 : NQ == 2 ? 4 : 5;
};

// word k (0..3) of a uint4, k a compile-time constant after unrolling
IK_DEVI uint32_t grp_u4(const uint4& v, int k) { return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w; }

// One pass of one gate/up row pair over NQ pairs. The activations are read straight from global memory (L1/L2 hits: a layer's
// activations are 16 tokens x H bf16), no staging and no block barriers: every warp is independent, so the latency of the loads
// is hidden by the other warps of the SM and by the loads of the next u iteration, not paid per block.
template <bool GELU, int NQ, int GW>
IK_DEVI void grp_gate_up_pass(const ik_bf16* __restrict__ x, int H, int I, const uint2* __restrict__ wg,
                              const uint2* __restrict__ wu, const uint8_t* __restrict__ sg, const uint8_t* __restrict__ su,
                              float gg, float gu, const int* __restrict__ pair_ids, int pc, unsigned kmagic, int lane, int r,
                              ik_bf16* __restrict__ h) {
    float ag[NQ], au[NQ];
    const uint4* xrow[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        ag[q] = au[q] = 0.f;
        xrow[q] = reinterpret_cast<const uint4*>(x + (int64_t)(q < pc ? __umulhi((unsigned)pair_ids[q], kmagic) : 0) * H);
    }
#pragma unroll 2
    for (int u = lane; u < H / 16; u += IK_WARP) {
        const uint2 wgv = __ldcs(wg + u), wuv = __ldcs(wu + u);
        const float sgv = ik_e4m3_to_f(sg[u]), suv = ik_e4m3_to_f(su[u]);
        float fg[16], fu[16];
        ik_fp4x8_to_f32(wgv.x, fg);
        ik_fp4x8_to_f32(wgv.y, fg + 8);
        ik_fp4x8_to_f32(wuv.x, fu);
        ik_fp4x8_to_f32(wuv.y, fu + 8);
        // groups of four pairs, no branch inside a group: the 16-long fma chains of the four pairs (and of the gate and the up
        // row) are independent, so they interleave; a pair >= pc reads token 0 and its sums are never used
#pragma unroll
        for (int g0 = 0; g0 < NQ; g0 += GW) {
            if (g0 < pc) {  // warp-uniform
                uint4 xr[GW][2];
#pragma unroll
                for (int j = 0; j < GW; ++j) {
                    xr[j][0] = __ldg(xrow[g0 + j] + 2 * u);
                    xr[j][1] = __ldg(xrow[g0 + j] + 2 * u + 1);
                }
                float pg[GW], pu[GW];
#pragma unroll
                for (int j = 0; j < GW; ++j) pg[j] = pu[j] = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
#pragma unroll
                    for (int j = 0; j < GW; ++j) {
                        const uint32_t w = grp_u4(xr[j][i >> 3], (i & 7) >> 1);
                        const float xv = (i & 1) ? __uint_as_float(w & 0xffff0000u) : __uint_as_float(w << 16);  // bf16 -> f32, exact
                        pg[j] = fmaf(fg[i], xv, pg[j]);
                        pu[j] = fmaf(fu[i], xv, pu[j]);
                    }
                }
#pragma unroll
                for (int j = 0; j < GW; ++j) {
                    ag[g0 + j] = fmaf(pg[j], sgv, ag[g0 + j]);
                    au[g0 + j] = fmaf(pu[j], suv, au[g0 + j]);
                }
            }
        }
    }
    float v[2 * NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        v[q] = ag[q];
        v[NQ + q] = au[q];
    }
    const float tot = grp_warp_sum<2 * NQ>(v, lane);  // lane l: value (l >> sh), the gate sums of pairs 0..NQ-1 then the up sums
    const float other = __shfl_xor_sync(0xffffffffu, tot, 16);  // lane l < 16 holds a gate sum, l + 16 the same pair's up sum
    constexpr int sh = GrpShift<NQ>::v - 1;
    const int q = lane >> sh;
    if (lane < 16 && (lane & ((1 << sh) - 1)) == 0 && q < pc) {
        const float g = round_bf16(tot * gg);
        const float u = round_bf16(other * gu);
        const float a = round_bf16(GELU ? ik_gelu_tanh(g) : g / (1.f + expf(-g)));
        h[(int64_t)pair_ids[q] * I + r] = ik_f2b(a * u);
    }
}

// The single-pair gate/up pass for two rows at once (see grp_down_lean): rows r0 and r0 + 8.
template <bool GELU>
IK_DEVI void grp_gate_up_lean(const ik_bf16* __restrict__ x, int H, int I, const uint8_t* __restrict__ guw,
                              const uint8_t* __restrict__ gus, int e, float gg, float gu, int pair, unsigned kmagic, int lane,
                              int r0, ik_bf16* __restrict__ h) {
    constexpr int R = 2;
    const uint4* xrow = reinterpret_cast<const uint4*>(x + (int64_t)__umulhi((unsigned)pair, kmagic) * H);
    float ag[R], au[R];
#pragma unroll
    for (int j = 0; j < R; ++j) ag[j] = au[j] = 0.f;
#pragma unroll 1
    for (int u = lane; u < H / 16; u += IK_WARP) {
        uint2 wgv[R], wuv[R];
        float sgv[R], suv[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const int rr = r0 + 8 * j < I ? r0 + 8 * j : r0;  // a row past I reads row r0 and is not stored
            const int64_t rg = (int64_t)e * 2 * I + rr, ru = (int64_t)e * 2 * I + I + rr;
            wgv[j] = __ldcs(reinterpret_cast<const uint2*>(guw + rg * (H / 2)) + u);
            wuv[j] = __ldcs(reinterpret_cast<const uint2*>(guw + ru * (H / 2)) + u);
            sgv[j] = ik_e4m3_to_f(gus[rg * (H / 16) + u]);
            suv[j] = ik_e4m3_to_f(gus[ru * (H / 16) + u]);
        }
        const uint4 x0 = __ldg(xrow + 2 * u), x1 = __ldg(xrow + 2 * u + 1);
        float xf[16];
        ik_unpack8(x0, xf);
        ik_unpack8(x1, xf + 8);
#pragma unroll
        for (int j = 0; j < R; ++j) {
            float fg[16], fu[16];
            ik_fp4x8_to_f32(wgv[j].x, fg);
            ik_fp4x8_to_f32(wgv[j].y, fg + 8);
            ik_fp4x8_to_f32(wuv[j].x, fu);
            ik_fp4x8_to_f32(wuv[j].y, fu + 8);
            float pg = 0.f, pu = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                pg = fmaf(fg[i], xf[i], pg);
                pu = fmaf(fu[i], xf[i], pu);
            }
            ag[j] = fmaf(pg, sgv[j], ag[j]);
            au[j] = fmaf(pu, suv[j], au[j]);
        }
    }
    float v[2 * R] = {ag[0], au[0], ag[1], au[1]};
    const float tot = grp_warp_sum<2 * R>(v, lane);  // lane l: value (l >> 3): gate row 0, up row 0, gate row 1, up row 1
    const float other = __shfl_xor_sync(0xffffffffu, tot, 8);
    const int j = lane >> 4;
    const int row = r0 + 8 * j;
    if ((lane & 15) == 0 && row < I) {
        const float g = round_bf16(tot * gg);
        const float u = round_bf16(other * gu);
        const float a = round_bf16(GELU ? ik_gelu_tanh(g) : g / (1.f + expf(-g)));
        h[(int64_t)pair * I + row] = ik_f2b(a * u);
    }
}

// grid (slot groups, I / 8 row groups): block (g, t) walks the active experts g, g + gridDim.x, .. against 8 gate/up rows (a warp per
// row). The grid is a few waves, not one block per possible expert: a block that finds no expert costs a latency chain of dependent
// loads (active, offsets), and a launch of mostly such blocks was most of the time of the sparse routings.
template <bool GELU>
__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_gate_up_grp_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ guw, const uint8_t* __restrict__ gus,
                       const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                       const int* __restrict__ active, ik_bf16* __restrict__ h, int E, int H, int I, unsigned kmagic) {
    const int nactive = active[E];
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r0 = blockIdx.y * (2 * kRowWarps) + warp;  // the warp's rows: r0 and r0 + 8
    for (int slot = blockIdx.x; slot < nactive; slot += gridDim.x) {
        const int e = active[slot];
        const int begin = offsets[e], cnt = offsets[e + 1] - begin;
        const float gg = gscales[e * 3 + 0], gu = gscales[e * 3 + 1];
        if (cnt == 1) {
            grp_gate_up_lean<GELU>(x, H, I, guw, gus, e, gg, gu, pairs[begin], kmagic, lane, r0, h);
            continue;
        }
#pragma unroll 1
        for (int j = 0; j < 2; ++j) {
            const int r = r0 + 8 * j;
            if (r >= I) break;
            const uint2* wg = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + r) * (H / 2));
            const uint2* wu = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + I + r) * (H / 2));
            const uint8_t* sg = gus + ((int64_t)e * 2 * I + r) * (H / 16);
            const uint8_t* su = gus + ((int64_t)e * 2 * I + I + r) * (H / 16);
            for (int c0 = 0; c0 < cnt; c0 += kGrpPairs) {
                const int pc = min(kGrpPairs, cnt - c0);
                // the pass width follows the pairs of this expert: 16 pairs on one expert (tokens on the same experts) are one pass
                // over the weights
                const int* pi = pairs + begin + c0;
                if (pc == 2) {
                    grp_gate_up_pass<GELU, 2, 2>(x, H, I, wg, wu, sg, su, gg, gu, pi, pc, kmagic, lane, r, h);
                } else if (pc <= 4) {
                    grp_gate_up_pass<GELU, 4, 4>(x, H, I, wg, wu, sg, su, gg, gu, pi, pc, kmagic, lane, r, h);
                } else if (pc <= 8) {
                    grp_gate_up_pass<GELU, 8, 4>(x, H, I, wg, wu, sg, su, gg, gu, pi, pc, kmagic, lane, r, h);
                } else {
                    grp_gate_up_pass<GELU, kGrpPairs, 4>(x, H, I, wg, wu, sg, su, gg, gu, pi, pc, kmagic, lane, r, h);
                }
            }
        }
    }
}

template <int NQ, int GW>
IK_DEVI void grp_down_pass(const ik_bf16* __restrict__ h, int I, const uint2* __restrict__ wd, const uint8_t* __restrict__ sd,
                           float gd, const int* __restrict__ pair_ids, int pc, int lane, int H, int r, ik_bf16* __restrict__ y) {
    float acc[NQ];
    const uint4* hrow[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        acc[q] = 0.f;
        hrow[q] = reinterpret_cast<const uint4*>(h + (int64_t)(q < pc ? pair_ids[q] : 0) * I);
    }
#pragma unroll 2
    for (int u = lane; u < I / 16; u += IK_WARP) {
        const uint2 wv = __ldcs(wd + u);
        const float sv = ik_e4m3_to_f(sd[u]);
        float fw[16];
        ik_fp4x8_to_f32(wv.x, fw);
        ik_fp4x8_to_f32(wv.y, fw + 8);
#pragma unroll
        for (int g0 = 0; g0 < NQ; g0 += GW) {
            if (g0 < pc) {  // warp-uniform; a pair >= pc reads the pair 0 row and its sum is never used
                uint4 hr[GW][2];
#pragma unroll
                for (int j = 0; j < GW; ++j) {
                    hr[j][0] = __ldg(hrow[g0 + j] + 2 * u);
                    hr[j][1] = __ldg(hrow[g0 + j] + 2 * u + 1);
                }
                float p[GW];
#pragma unroll
                for (int j = 0; j < GW; ++j) p[j] = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
#pragma unroll
                    for (int j = 0; j < GW; ++j) {
                        const uint32_t w = grp_u4(hr[j][i >> 3], (i & 7) >> 1);
                        const float hv = (i & 1) ? __uint_as_float(w & 0xffff0000u) : __uint_as_float(w << 16);
                        p[j] = fmaf(fw[i], hv, p[j]);
                    }
                }
#pragma unroll
                for (int j = 0; j < GW; ++j) acc[g0 + j] = fmaf(p[j], sv, acc[g0 + j]);
            }
        }
    }
    const float tot = grp_warp_sum<NQ>(acc, lane);  // lane l: pair (l >> sh)
    constexpr int sh = GrpShift<NQ>::v;
    const int q = lane >> sh;
    if ((lane & ((1 << sh) - 1)) == 0 && q < pc) y[(int64_t)pair_ids[q] * H + r] = ik_f2b(tot * gd);
}

// One pair against R rows at once (the per-pair kernels' shape: R independent loads and fma chains per warp, the activation row
// loaded once): used when an expert has a single pair, where the work is the weights' DRAM stream and the latency hiding of the
// loads matters, not the pairs. Row r of the warp's rows: r0 + 8 * j.
template <int R>
IK_DEVI void grp_down_lean(const ik_bf16* __restrict__ h, int I, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                           int e, int H, float gd, int pair, int lane, int r0, ik_bf16* __restrict__ y) {
    const uint4* hrow = reinterpret_cast<const uint4*>(h + (int64_t)pair * I);
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; ++j) acc[j] = 0.f;
#pragma unroll 1
    for (int u = lane; u < I / 16; u += IK_WARP) {
        uint2 wv[R];
        float sv[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const int rr = r0 + 8 * j < H ? r0 + 8 * j : r0;  // a row past H reads row r0 and is not stored
            wv[j] = __ldcs(reinterpret_cast<const uint2*>(dnw + ((int64_t)e * H + rr) * (I / 2)) + u);
            sv[j] = ik_e4m3_to_f(dns[((int64_t)e * H + rr) * (I / 16) + u]);
        }
        const uint4 h0 = __ldg(hrow + 2 * u), h1 = __ldg(hrow + 2 * u + 1);
        float hf[16];
        ik_unpack8(h0, hf);
        ik_unpack8(h1, hf + 8);
#pragma unroll
        for (int j = 0; j < R; ++j) {
            float fw[16];
            ik_fp4x8_to_f32(wv[j].x, fw);
            ik_fp4x8_to_f32(wv[j].y, fw + 8);
            float p = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) p = fmaf(fw[i], hf[i], p);
            acc[j] = fmaf(p, sv[j], acc[j]);
        }
    }
    const float tot = grp_warp_sum<R>(acc, lane);  // lane l: row j = l >> (5 - log2 R)
    constexpr int sh = GrpShift<R>::v;
    const int j = lane >> sh;
    const int row = r0 + 8 * j;
    if ((lane & ((1 << sh) - 1)) == 0 && row < H) y[(int64_t)pair * H + row] = ik_f2b(tot * gd);
}

__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_down_grp_kernel(const ik_bf16* __restrict__ h, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                    const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                    const int* __restrict__ active, ik_bf16* __restrict__ y, int E, int H, int I) {
    const int nactive = active[E];
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r0 = blockIdx.y * (4 * kRowWarps) + warp;  // the warp's rows: r0, r0 + 8, r0 + 16, r0 + 24
    for (int slot = blockIdx.x; slot < nactive; slot += gridDim.x) {
        const int e = active[slot];
        const int begin = offsets[e], cnt = offsets[e + 1] - begin;
        const float gd = gscales[e * 3 + 2];
        if (cnt == 1) {
            grp_down_lean<4>(h, I, dnw, dns, e, H, gd, pairs[begin], lane, r0, y);
            continue;
        }
#pragma unroll 1
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + 8 * j;
            if (r >= H) break;
            const uint2* wd = reinterpret_cast<const uint2*>(dnw + ((int64_t)e * H + r) * (I / 2));
            const uint8_t* sd = dns + ((int64_t)e * H + r) * (I / 16);
            for (int c0 = 0; c0 < cnt; c0 += kGrpPairs) {
                const int pc = min(kGrpPairs, cnt - c0);
                const int* pi = pairs + begin + c0;
                if (pc == 2) {
                    grp_down_pass<2, 2>(h, I, wd, sd, gd, pi, pc, lane, H, r, y);
                } else if (pc <= 4) {
                    grp_down_pass<4, 4>(h, I, wd, sd, gd, pi, pc, lane, H, r, y);
                } else if (pc <= 8) {
                    grp_down_pass<8, 4>(h, I, wd, sd, gd, pi, pc, lane, H, r, y);
                } else {
                    grp_down_pass<kGrpPairs, 4>(h, I, wd, sd, gd, pi, pc, lane, H, r, y);
                }
            }
        }
    }
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
                              const ik_scratch* scratch, void* stream, bool gelu, bool grp = false) {
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
    if (grp) {
        // row-grouped decode kernels (T <= kGrpPairs): one pass over an expert's weights per up to kGrpPairs pairs
        if (T > kGrpPairs || P > 0xffff || H % 16 != 0 || I % 16 != 0) return IK_ERR_SHAPE;
        const unsigned kmagic = (unsigned)(0x100000000ULL / (uint64_t)K + 1);  // pair / K == __umulhi(pair, kmagic) for pair < 2^16
        const unsigned gslots = slots < kGrpSlotBlocks ? slots : kGrpSlotBlocks;
        const dim3 gu_grid(gslots, (unsigned)((I + 2 * kRowWarps - 1) / (2 * kRowWarps)));
        const dim3 dn_grid(gslots, (unsigned)((H + 4 * kRowWarps - 1) / (4 * kRowWarps)));
        const ik_bf16* xp = (const ik_bf16*)x->data;
        const uint8_t* guw = (const uint8_t*)gate_up_w->data;
        const uint8_t* gus = (const uint8_t*)gate_up_s->data;
        const float* gsc = (const float*)gscales->data;
        const uint8_t* dnw = (const uint8_t*)down_w->data;
        const uint8_t* dns = (const uint8_t*)down_s->data;
        if (gelu) {
            moe_gate_up_grp_kernel<true><<<gu_grid, kRowWarps * IK_WARP, 0, s>>>(xp, guw, gus, gsc, pairs, offsets, active, hbuf, (int)E, (int)H, (int)I, kmagic);
        } else {
            moe_gate_up_grp_kernel<false><<<gu_grid, kRowWarps * IK_WARP, 0, s>>>(xp, guw, gus, gsc, pairs, offsets, active, hbuf, (int)E, (int)H, (int)I, kmagic);
        }
        IK_TRY(ik_launch_status());
        moe_down_grp_kernel<<<dn_grid, kRowWarps * IK_WARP, 0, s>>>(hbuf, dnw, dns, gsc, pairs, offsets, active, ybuf, (int)E, (int)H, (int)I);
        IK_TRY(ik_launch_status());
        moe_reduce_kernel<ik_bf16><<<dim3((unsigned)T, (unsigned)((H + 255) / 256)), 256, 0, s>>>(
            ybuf, (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)K);
        return ik_launch_status();
    }
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

// Row-grouped decode experts for 1..16 tokens (see "row-grouped decode experts" above): bitwise the per-pair rows kernels of
// moe_decode.cu. Scratch as ik_moe_experts_bf16's. Called by ik_moe_experts_decode_v2_rows_bf16 (INTARSIA_MOE_ROWS_GROUP_MIN).
int ik_moe_experts_rows_grouped_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                     const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                     const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                     const ik_scratch* scratch, void* stream) {
    return moe_experts_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, false, true);
}

int ik_moe_experts_rows_grouped_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                          const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                          const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                          const ik_scratch* scratch, void* stream) {
    return moe_experts_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true, true);
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
