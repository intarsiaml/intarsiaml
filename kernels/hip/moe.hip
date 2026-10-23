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
//                 Batch size: P = T*K <= 16 (decode, p1 kernels) and T < 16 (grouped GEMV kernels) share one lane and
//                 accumulation order, so a token's result is bit-identical to T = 1 (as on CUDA). From T = 16 the
//                 projections run on the tensor cores (moe_wmma.h) with another accumulation order: results then depend
//                 on the batch (chunk) size within the admission tolerance. INTARSIA_HIP_MOE_WMMA=0 disables that path
//                 (batch-invariant mode); see docs/backends/hip.md.
//   moe_combine   out = routed + sigmoid(gate_logit) * shared with HF's rounding points.
//
// Scratch layout (must match intarsia_core::quant::ops::moe_experts_scratch, regions rounded to 256 bytes):
//   pairs int[T*K] | offsets int[E+1] | active int[E+1] (last = count) | h bf16[T*K, I] | y bf16[T*K, H]
#include "ik_quant.h"
#include "moe_wmma_r7.h"  // includes moe_wmma.h
#include <algorithm>
#include <math.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxExperts = 1024;  // grouping kernel: one thread per expert
constexpr int kPairChunk = 8;      // pairs sharing one pass over an expert row
constexpr int kMwMinTokens = 16;   // from this many tokens on the projections run on the tensor cores (moe_wmma.h)

// INTARSIA_HIP_MOE_WMMA=0 turns the tensor-core projections off, so every token count takes the GEMV-style path and a
// token's result no longer depends on the batch size it was computed in (bit-identical to T=1, like the CUDA backend;
// about 9x slower MoE at T=256). Default on. Read once per process; part of the backend fingerprint (kernels.rs).
bool moe_wmma_enabled() {
    static const bool on = [] {
        const char* e = getenv("INTARSIA_HIP_MOE_WMMA");
        return !(e && e[0] == '0');
    }();
    return on;
}

// INTARSIA_HIP_MOE_Z=n (default 2): the grid z of the grouped projections is ceil(n * average pairs per expert / 32) pair tiles, not
// one block per tile an expert could possibly need (ceil(T / 32): at T = 4096 that is 128 per expert and feature tile, of which
// the average expert fills 4: ~250 000 launched blocks, 97 % of them exiting at once). A block steps through the further tiles
// of a busier expert by gridDim.z, and every pair's result is independent of the grouping, so the outputs are bit for bit the same
// for every n; 0 restores the full grid (A/B and debugging). Read once per process; part of the backend fingerprint (kernels.rs).
int moe_zmul() {
    static const int n = [] {
        const char* e = getenv("INTARSIA_HIP_MOE_Z");
        return e ? atoi(e) : 2;
    }();
    return n;
}

// INTARSIA_HIP_MOE_PAIRS = 32 | 64 | 0 (auto, the default): pairs per block of the grouped projections (moe_wmma.h, PAIRS). A
// 64-pair tile feeds each dequantised weight fragment to four WMMAs instead of two; auto takes it when the average expert has
// at least kMwPairsAuto pairs. Every pair's result is independent of the grouping, so all settings give the same bits.
// Read at every launch (the bitwise test of tests/moe_pairs.rs switches it in one process); part of the backend fingerprint.
constexpr int64_t kMwPairsAuto = 32;  // bench_prefill_shapes at the 35B's shape: 64 wins from T = 1024 (average 32), loses at T = 312 (10)
int moe_pairs(int64_t avg) {
    const char* e = getenv("INTARSIA_HIP_MOE_PAIRS");
    const int n = e ? atoi(e) : 0;
    if (n == 32 || n == 64) return n;
    return avg >= kMwPairsAuto ? 64 : 32;
}

// INTARSIA_HIP_MOE_MW = v | g:d: variant bits of the grouped projections, v for both of them, or g for gate_up and d for down.
// Each is one of 0 (the round-5 moe_wmma_kernel), 1 | 4 | 5 (moe_wmma.h moe_wmma_kernel_v: 1 skip empty fragments, 4 occupancy)
// or an IK_MW_R7 value (moe_wmma_r7.h moe_wmma_kernel_r7, round 7: 8 fp4 decode by the fp8 conversion instead of the LDS table,
// 16 two A fragments per wave in down, 32 six weight blocks in flight); every value gives the same bits (tests/moe_pairs.rs). Unset, empty or any other value: the tile's default for that projection, kMwDefault*
// (docs/results/2026-10-19-hip-round6.md, -round7.md). Read at every launch (the bitwise test switches it in one process); part of
// the backend fingerprint (kernels.rs).
#define IK_MW_R7(X) X(13) X(25) X(29) X(41) X(45)
constexpr int kMwDefault64 = kMwCvt | kMwSkip | kMwOcc | kMwDeep, kMwDefault64Dn = kMwCvt | kMwSkip | kMwTwo;
constexpr int kMwDefault32 = 0, kMwDefault32Dn = kMwCvt | kMwSkip | kMwDeep;
constexpr bool mw_accepted(int v) {
#define IK_MW_IS(n) || v == n
    return v == 0 || v == kMwSkip || v == kMwOcc || v == (kMwSkip | kMwOcc) IK_MW_R7(IK_MW_IS);
#undef IK_MW_IS
}
int moe_mw_variant(int pairs, bool down) {
    const int def = pairs == 64 ? (down ? kMwDefault64Dn : kMwDefault64) : (down ? kMwDefault32Dn : kMwDefault32);
    const char* e = getenv("INTARSIA_HIP_MOE_MW");
    if (!e || !e[0]) return def;
    const char* colon = strchr(e, ':');
    const char* f = down && colon ? colon + 1 : e;  // gate_up: the part before ':' (atoi stops there)
    if (!f[0] || f[0] == ':') return def;
    const int v = atoi(f);
    return mw_accepted(v) ? v : def;
}

// one projection: the round-5 kernel (V = 0), moe_wmma_kernel_v (round 6) or moe_wmma_kernel_r7 (round 7; kMwTwo doubles
// down's features per block)
// Round 8 split (ik_split_pieces): the estimated duration of one projection from its multiply-adds, P pairs x output rows
// (gate_up: gate + up, 2N) x Kd. Rates measured in the 4408-token prefill profile (docs/results/2026-10-19-hip-round8.md):
// 64-pair tile ~32 GMAC/ms (gate_up 2.14 ms, down 1.14 ms at 4096 tokens); the 32-pair tile is weight-bound at its sizes,
// ~7-10 GMAC/ms, taken as 9 so that the 312-token tail chunk (gate_up 0.75 ms) stays one launch.
constexpr double kMwMacPerUs64 = 32e6, kMwMacPerUs32 = 9e6;

// one projection: the round-5 kernel (V = 0), moe_wmma_kernel_v (round 6) or moe_wmma_kernel_r7 (round 7; kMwTwo doubles
// down's features per block), as one launch or (round 8) as pieces over consecutive feature tiles (grid y): each block of a
// piece is the block of the single launch with the same feature tile, so the pieces give the single launch's bits
template <bool GATEUP, int PAIRS, int V>
void moe_wmma_launch_p(int64_t N, int64_t P, unsigned slots, unsigned zt, const ik_bf16* src, const uint8_t* w,
                       const uint8_t* ws, const float* gsc, const int* pairs, const int* offsets, const int* active,
                       ik_bf16* dst, int64_t E, int64_t Kd, int64_t K, hipStream_t s) {
    constexpr int feat = (V & (kMwCvt | kMwTwo)) ? mw_r7_feat(GATEUP, V) : (GATEUP ? kMwFeatGateUp : kMwFeatDown);
    const int64_t Y = (N + feat - 1) / feat;  // feature tiles
    const double est_us = (double)P * (double)(GATEUP ? 2 * N : N) * (double)Kd / (PAIRS == 64 ? kMwMacPerUs64 : kMwMacPerUs32);
    // gate_up is never split: its pieces each paid the launch's load-imbalance tail (an expert with many more pairs than
    // the grid's z covers runs longest), 4 pieces took 114.1 ms against 85.7 for one launch in the 4408-token prefill (W3),
    // far more than the ~11 ms of gaps they removed. down's two pieces cost +3.0 ms against ~10.7 ms of gaps.
    const int pieces = GATEUP ? 1 : ik_split_pieces(est_us, Y);
    const char* stag = GATEUP ? (PAIRS == 64 ? "gate_up64" : "gate_up32") : (PAIRS == 64 ? "down64" : "down32");
    ik_rt_stamp(s, stag, -1);  // INTARSIA_HIP_STAMP only (round 9 diagnostics); otherwise launches nothing
    for (int i = 0; i < pieces; ++i) {
        const int y0 = (int)(Y * i / pieces), y1 = (int)(Y * (i + 1) / pieces);
        const dim3 g(slots, (unsigned)(y1 - y0), zt);
        if constexpr (V == 0)
            moe_wmma_kernel<GATEUP, PAIRS><<<g, kMwThreads, 0, s>>>(src, w, ws, gsc, pairs, offsets, active, dst, (int)E, (int)Kd, (int)N, (int)K, y0);
        else if constexpr ((V & (kMwCvt | kMwTwo)) == 0)
            moe_wmma_kernel_v<GATEUP, PAIRS, V><<<g, kMwThreads, 0, s>>>(src, w, ws, gsc, pairs, offsets, active, dst, (int)E, (int)Kd, (int)N, (int)K, y0);
        else
            moe_wmma_kernel_r7<GATEUP, PAIRS, V><<<g, kMwThreads, 0, s>>>(src, w, ws, gsc, pairs, offsets, active, dst, (int)E, (int)Kd, (int)N, (int)K, y0);
        ik_rt_stamp(s, stag, i);
    }
}

template <int PAIRS>
void moe_wmma_launch(int64_t T, int64_t P, int64_t E, int64_t H, int64_t I, int64_t K, unsigned slots, const ik_bf16* x,
                     const uint8_t* guw, const uint8_t* gus, const uint8_t* dw, const uint8_t* ds, const float* gsc,
                     const int* pairs, const int* offsets, const int* active, ik_bf16* hbuf, ik_bf16* ybuf, hipStream_t s) {
    unsigned zt = (unsigned)((T + PAIRS - 1) / PAIRS);
    if (moe_zmul() > 0) {
        const int64_t avg = (P + E - 1) / E;  // pairs per expert, on average
        const unsigned cap = (unsigned)std::max<int64_t>(1, ((int64_t)moe_zmul() * avg + PAIRS - 1) / PAIRS);
        if (cap < zt) zt = cap;
    }
#define IK_MW_SWITCH(SEL, CALL)                       \
    switch (SEL) {                                    \
        case kMwSkip: CALL(kMwSkip); break;           \
        case kMwOcc: CALL(kMwOcc); break;             \
        case kMwSkip | kMwOcc: CALL(kMwSkip | kMwOcc); break; \
        IK_MW_R7(IK_MW_CASE_##CALL)                   \
        default: CALL(0); break;                      \
    }
#define IK_MW_GU(V) moe_wmma_launch_p<true, PAIRS, V>(I, P, slots, zt, x, guw, gus, gsc, pairs, offsets, active, hbuf, E, H, K, s)
#define IK_MW_DN(V) moe_wmma_launch_p<false, PAIRS, V>(H, P, slots, zt, hbuf, dw, ds, gsc, pairs, offsets, active, ybuf, E, I, K, s)
#define IK_MW_CASE_IK_MW_GU(n) case n: IK_MW_GU(n); break;
#define IK_MW_CASE_IK_MW_DN(n) case n: IK_MW_DN(n); break;
    IK_MW_SWITCH(moe_mw_variant(PAIRS, false), IK_MW_GU)
    if (ik_launch_status() != IK_OK) return;
    IK_MW_SWITCH(moe_mw_variant(PAIRS, true), IK_MW_DN)
#undef IK_MW_CASE_IK_MW_DN
#undef IK_MW_CASE_IK_MW_GU
#undef IK_MW_DN
#undef IK_MW_GU
#undef IK_MW_SWITCH
}

IK_DEVI float round_bf16(float v) { return ik_round_bf16(v); }

// ---- top-k ---------------------------------------------------------------------------------------------------------

constexpr int kTopkMaxPerLane = 16;  // E <= 512
constexpr int kTopkWarps = 4;

__global__ void __launch_bounds__(kTopkWarps * IK_WARP)
moe_topk_kernel(const ik_bf16* __restrict__ logits, int T, int E, int K, int norm, float* __restrict__ w_out,
                int* __restrict__ id_out) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int t0 = blockIdx.x * kTopkWarps + warp;
    const bool live = t0 < T;  // no early return: the block barrier below needs every wave
    const int t = live ? t0 : 0;
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
            const float ov = __shfl_xor(best, o, IK_WARP);
            const int oi = __shfl_xor(bi, o, IK_WARP);
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
    __syncthreads();
    if (live && lane < K) {
        const float w = expf(s_val[warp][lane] - mx) / denom;
        w_out[(int64_t)t * K + lane] = round_bf16(w);
        id_out[(int64_t)t * K + lane] = s_id[warp][lane];
    }
}

// v2: the same selection and softmax, arithmetic included, with each candidate packed into one 32-bit key (monotone 16-bit
// image of the bf16 logit in the high half, 0xFFFF - expert in the low half), so the per-round warp argmax is one shuffle
// and one max per step instead of two shuffles, two compares and selects (the ties-to-lowest-index rule comes from the low
// half). K rounds of five dependent shuffle steps are this kernel's critical path (one block of 4 warps for a decode token).
// Differences from v1, none observable on finite non-zero-tied inputs: +0 sorts above -0, and a NaN logit is never picked.
IK_DEVI uint32_t topk_key16(uint32_t bits) {
    if ((bits & 0x7f80u) == 0x7f80u && (bits & 0x7fu)) return 0u;  // NaN: below every number
    return (bits & 0x8000u) ? (~bits & 0xffffu) : (bits | 0x8000u);
}
IK_DEVI float topk_val(uint32_t key16) {
    const uint32_t bits = (key16 & 0x8000u) ? (key16 ^ 0x8000u) : (~key16 & 0xffffu);
    return __uint_as_float(bits << 16);
}

__global__ void __launch_bounds__(kTopkWarps * IK_WARP)
moe_topk_v2_kernel(const ik_bf16* __restrict__ logits, int T, int E, int K, int norm, float* __restrict__ w_out,
                   int* __restrict__ id_out) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int t0 = blockIdx.x * kTopkWarps + warp;
    const bool live = t0 < T;
    const int t = live ? t0 : 0;
    float v[kTopkMaxPerLane];
    uint32_t key[kTopkMaxPerLane];
#pragma unroll
    for (int j = 0; j < kTopkMaxPerLane; ++j) {
        const int e = lane + IK_WARP * j;
        const ik_bf16 b = e < E ? logits[(int64_t)t * E + e] : ik_bf16{0xff80u};  // -inf
        v[j] = e < E ? ik_b2f(b) : -INFINITY;
        key[j] = e < E ? ((topk_key16(b.bits) << 16) | (0xffffu - (uint32_t)e)) : 0u;
    }
    __shared__ float s_val[kTopkWarps][IK_WARP];
    __shared__ int s_id[kTopkWarps][IK_WARP];
    float mx = 0.f, denom = 0.f;
    for (int kk = 0; kk < K; ++kk) {
        uint32_t best = key[0];
#pragma unroll
        for (int j = 1; j < kTopkMaxPerLane; ++j) best = max(best, key[j]);
#pragma unroll
        for (int o = IK_WARP / 2; o > 0; o >>= 1) best = max(best, (uint32_t)__shfl_xor((int)best, o, IK_WARP));
        const int bi = (int)(0xffffu - (best & 0xffffu));
        const float bv = topk_val(best >> 16);
        if (kk == 0) mx = bv;
        if ((bi & (IK_WARP - 1)) == lane) {
#pragma unroll
            for (int j = 0; j < kTopkMaxPerLane; ++j)
                if (j == (bi >> 5)) key[j] = 0u;
        }
        if (lane == 0) {
            s_val[warp][kk & 31] = bv;
            s_id[warp][kk & 31] = bi;
        }
        denom += expf(bv - mx);
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
    __syncthreads();
    if (live && lane < K) {
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
        const int t = __shfl_up(inc, o, IK_WARP);
        if (lane >= o) inc += t;
    }
    __syncthreads();  // wsum may still be read by a previous scan
    if (lane == IK_WARP - 1) wsum[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int x = lane < nw ? wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < IK_WARP; o <<= 1) {
            const int t = __shfl_up(x, o, IK_WARP);
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

__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_gate_up_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ guw, const uint8_t* __restrict__ gus,
                   const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                   const int* __restrict__ active, ik_bf16* __restrict__ h, int E, int H, int I, int K, int PC) {
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* xs = reinterpret_cast<ik_bf16*>(smem);  // [PC][H]
    IK_FP4_LUT(lut);
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
                const uint2 wgv = ik_ldcs(wg + u), wuv = ik_ldcs(wu + u);
                const float sgv = ik_e4m3_to_f(sg[u]), suv = ik_e4m3_to_f(su[u]);
                float fg[16], fu[16];
                ik_fp4x8_to_f32(wgv.x, lut, fg);
                ik_fp4x8_to_f32(wgv.y, lut, fg + 8);
                ik_fp4x8_to_f32(wuv.x, lut, fu);
                ik_fp4x8_to_f32(wuv.y, lut, fu + 8);
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
                    const float a = round_bf16(g / (1.f + expf(-g)));
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
    IK_FP4_LUT(lut);
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
                const uint2 wv = ik_ldcs(wd + u);
                const float sv = ik_e4m3_to_f(sd[u]);
                float fw[16];
                ik_fp4x8_to_f32(wv.x, lut, fw);
                ik_fp4x8_to_f32(wv.y, lut, fw + 8);
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

// Round 8: moe_reduce_kernel<ik_bf16> with 8 consecutive features per thread, the token's (up to 8) expert rows loaded as
// 16-byte vectors before any arithmetic (the scalar kernel read 2 bytes per thread per row: ~305 GB/s, 0.49 ms per
// 4096-token chunk for 134 MB of y). Every feature still sums round_bf16(y * w) over k = 0, 1, .. in order, skipping the same
// invalid ids, and is stored with the same bf16 conversion, so the bits are moe_reduce_kernel's (tests/moe_pairs.rs).
// Needs H % 8 == 0 and 16-byte aligned y / out; INTARSIA_HIP_MOE_REDUCE=0 runs the scalar kernel.
constexpr int kRdMaxK = 8;
__global__ void __launch_bounds__(256)
moe_reduce_v8_kernel(const ik_bf16* __restrict__ y, const int* __restrict__ ids, const float* __restrict__ w,
                     ik_bf16* __restrict__ out, int E, int H, int K) {
    const int t = blockIdx.x;
    const int c = (blockIdx.y * 256 + threadIdx.x) * 8;
    if (c >= H) return;
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
    for (int k0 = 0; k0 < K; k0 += kRdMaxK) {
        uint4 yv[kRdMaxK];
#pragma unroll
        for (int u = 0; u < kRdMaxK; ++u)
            if (k0 + u < K) yv[u] = *reinterpret_cast<const uint4*>(y + ((int64_t)t * K + k0 + u) * H + c);
#pragma unroll
        for (int u = 0; u < kRdMaxK; ++u) {
            const int k = k0 + u;
            if (k >= K) break;
            const int id = ids[(int64_t)t * K + k];
            if (id < 0 || id >= E) continue;
            const float wk = w[(int64_t)t * K + k];
            float v[8];
            ik_unpack8(yv[u], v);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += round_bf16(v[e] * wk);
        }
    }
    *reinterpret_cast<uint4*>(out + (int64_t)t * H + c) = ik_pack8(acc);
}

inline bool moe_reduce_v8_enabled() {
    const char* e = getenv("INTARSIA_HIP_MOE_REDUCE");
    return !(e && e[0] == '0' && e[1] == 0);
}

// ---- decode fast path (P = T*K <= 16, K <= 8): no grouping ------------------------------------------------------------
// Each (token, slot) pair is its own block row: gate_up reads the expert id straight from `ids`, and the down
// projection is fused with the weighted reduction (one warp per output feature loops over the token's K experts,
// all their weight loads in flight at once), so a decode step is topk + 2 launches instead of topk + 4. A pair's
// arithmetic follows the grouped GEMV-style kernels (same lane/accumulation order).

constexpr int kP1MaxK = 8;

__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_gate_up_p1_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ guw, const uint8_t* __restrict__ gus,
                      const float* __restrict__ gscales, const int* __restrict__ ids, ik_bf16* __restrict__ h, int E,
                      int H, int I, int K) {
    IK_FP4_LUT(lut);
    const int p = blockIdx.x;
    const int e = ids[p];
    if (e < 0 || e >= E) return;  // block-uniform
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    if (r >= I) return;  // warp-uniform, no barrier below
    const uint2* wg = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + r) * (H / 2));
    const uint2* wu = reinterpret_cast<const uint2*>(guw + ((int64_t)e * 2 * I + I + r) * (H / 2));
    const uint8_t* sg = gus + ((int64_t)e * 2 * I + r) * (H / 16);
    const uint8_t* su = gus + ((int64_t)e * 2 * I + I + r) * (H / 16);
    const ik_bf16* xr = x + (int64_t)(p / K) * H;
    float ag = 0.f, au = 0.f;
#pragma unroll 4
    for (int u = lane; u < H / 16; u += IK_WARP) {
        const uint2 wgv = ik_ldcs(wg + u), wuv = ik_ldcs(wu + u);
        const float sgv = ik_e4m3_to_f(sg[u]), suv = ik_e4m3_to_f(su[u]);
        float fg[16], fu[16];
        ik_fp4x8_to_f32(wgv.x, lut, fg);
        ik_fp4x8_to_f32(wgv.y, lut, fg + 8);
        ik_fp4x8_to_f32(wuv.x, lut, fu);
        ik_fp4x8_to_f32(wuv.y, lut, fu + 8);
        const uint4* xp = reinterpret_cast<const uint4*>(xr + u * 16);
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
    const float uu = round_bf16(ik_warp_sum(au) * gscales[e * 3 + 1]);
    const float a = round_bf16(g / (1.f + expf(-g)));
    if (lane == 0) h[(int64_t)p * I + r] = ik_f2b(a * uu);
}

template <typename OutT>
__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_down_reduce_p1_kernel(const ik_bf16* __restrict__ h, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                          const float* __restrict__ gscales, const int* __restrict__ ids, const float* __restrict__ wts,
                          OutT* __restrict__ out, int E, int H, int I, int K) {
    IK_FP4_LUT(lut);
    const int t = blockIdx.x;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    if (r >= H) return;
    const uint2* wd[kP1MaxK];
    const uint8_t* sd[kP1MaxK];
    float gd[kP1MaxK], acc[kP1MaxK];
#pragma unroll
    for (int k = 0; k < kP1MaxK; ++k) {
        const int id = k < K ? ids[(int64_t)t * K + k] : -1;
        const bool ok = id >= 0 && id < E;
        const int64_t row = ok ? (int64_t)id * H + r : 0;
        wd[k] = ok ? reinterpret_cast<const uint2*>(dnw + row * (I / 2)) : nullptr;
        sd[k] = ok ? dns + row * (I / 16) : nullptr;
        gd[k] = ok ? gscales[id * 3 + 2] : 0.f;
        acc[k] = 0.f;
    }
    for (int u = lane; u < I / 16; u += IK_WARP) {
        uint2 wv[kP1MaxK];
        float sv[kP1MaxK];
#pragma unroll
        for (int k = 0; k < kP1MaxK; ++k) {  // all expert rows' loads issued before any use
            wv[k] = wd[k] ? ik_ldcs(wd[k] + u) : make_uint2(0, 0);
            sv[k] = wd[k] ? ik_e4m3_to_f(sd[k][u]) : 0.f;
        }
#pragma unroll
        for (int k = 0; k < kP1MaxK; ++k) {
            if (!wd[k]) continue;
            const uint4* hp = reinterpret_cast<const uint4*>(h + ((int64_t)t * K + k) * I + u * 16);
            float fw[16], hf[16];
            ik_fp4x8_to_f32(wv[k].x, lut, fw);
            ik_fp4x8_to_f32(wv[k].y, lut, fw + 8);
            ik_unpack8(hp[0], hf);
            ik_unpack8(hp[1], hf + 8);
            float pp = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) pp = fmaf(fw[i], hf[i], pp);
            acc[k] = fmaf(pp, sv[k], acc[k]);
        }
    }
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < kP1MaxK; ++k) {
        if (!wd[k]) continue;
        const float y = round_bf16(ik_warp_sum(acc[k]) * gd[k]);  // y is stored as bf16 in the grouped path
        o += round_bf16(y * wts[(int64_t)t * K + k]);
    }
    if (lane == 0) ik_store(out + (int64_t)t * H + r, o);
}

// ---- decode fast path, v2 -----------------------------------------------------------------------------------------
// Bit-identical to the p1 kernels above (same lane / iteration / accumulation order); only the order of the memory
// requests changes: the ids are read first, then the first kPre1 iterations' weights, scales and activations are requested,
// and only then is the shared table built (its barrier used to precede every load, so a block spent the table fill and the
// ids' round trip before its first weight request left). Rows past the end clamp to row 0 and are masked, so every thread
// reaches the barrier.

constexpr int kPre1 = 4;

// XS: the activations (the token's x row; the token's K intermediate rows) are copied to shared memory once per block, behind
// the weight requests, instead of every warp reading the same kilobytes through the cache hierarchy. One barrier serves the
// copy and the table fill. Dynamic shared memory: H * 2 bytes (gate_up), K * I * 2 bytes (down).
template <bool XS, int PRE>
__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_gate_up_p1_v2_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ guw, const uint8_t* __restrict__ gus,
                         const float* __restrict__ gscales, const int* __restrict__ ids, ik_bf16* __restrict__ h, int E,
                         int H, int I, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int p = blockIdx.x;
    const int e = ids[p];
    const bool ok = e >= 0 && e < E;  // block-uniform
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    const bool live = ok && r < I;
    const int ee = ok ? e : 0, rr = r < I ? r : 0;
    const uint2* wg = reinterpret_cast<const uint2*>(guw + ((int64_t)ee * 2 * I + rr) * (H / 2));
    const uint2* wu = reinterpret_cast<const uint2*>(guw + ((int64_t)ee * 2 * I + I + rr) * (H / 2));
    const uint8_t* sg = gus + ((int64_t)ee * 2 * I + rr) * (H / 16);
    const uint8_t* su = gus + ((int64_t)ee * 2 * I + I + rr) * (H / 16);
    const ik_bf16* xr = x + (int64_t)(p / K) * H;
    const int nu = H / 16;
    uint2 wgv[PRE], wuv[PRE];
    uint32_t sgb[PRE], sub[PRE];
    uint4 xv[XS ? 1 : PRE][2];
#pragma unroll
    for (int j = 0; j < PRE; ++j) {
        const int u = lane + IK_WARP * j;
        const bool in = u < nu;
        wgv[j] = in ? ik_ldcs(wg + u) : make_uint2(0, 0);
        wuv[j] = in ? ik_ldcs(wu + u) : make_uint2(0, 0);
        sgb[j] = in ? sg[u] : 0u;
        sub[j] = in ? su[u] : 0u;
        if (!XS) {
            const uint4* xp = reinterpret_cast<const uint4*>(xr + (in ? u : 0) * 16);
            xv[XS ? 0 : j][0] = xp[0];
            xv[XS ? 0 : j][1] = xp[1];
        }
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    if (XS) {
        for (int i = threadIdx.x; i < H / 8; i += blockDim.x)
            reinterpret_cast<uint4*>(smem)[i] = reinterpret_cast<const uint4*>(xr)[i];
    }
    __syncthreads();
    if (!live) return;  // warp-uniform
    float ag = 0.f, au = 0.f;
    auto step = [&](const uint2& g2, const uint2& u2, uint32_t gsb, uint32_t usb, const uint4& x0, const uint4& x1) {
        const float sgv = ik_e4m3_to_f(gsb), suv = ik_e4m3_to_f(usb);
        float fg[16], fu[16];
        ik_fp4x8_to_f32(g2.x, lut, fg);
        ik_fp4x8_to_f32(g2.y, lut, fg + 8);
        ik_fp4x8_to_f32(u2.x, lut, fu);
        ik_fp4x8_to_f32(u2.y, lut, fu + 8);
        float xf[16];
        ik_unpack8(x0, xf);
        ik_unpack8(x1, xf + 8);
        float pg = 0.f, pu = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            pg = fmaf(fg[i], xf[i], pg);
            pu = fmaf(fu[i], xf[i], pu);
        }
        ag = fmaf(pg, sgv, ag);
        au = fmaf(pu, suv, au);
    };
    auto xs2 = [&](int u, uint4& x0, uint4& x1) {
        const uint4* xp = reinterpret_cast<const uint4*>(XS ? reinterpret_cast<const ik_bf16*>(smem) : xr) + u * 2;
        x0 = xp[0];
        x1 = xp[1];
    };
#pragma unroll
    for (int j = 0; j < PRE; ++j) {
        if (lane + IK_WARP * j >= nu) continue;
        if (XS) {
            uint4 x0, x1;
            xs2(lane + IK_WARP * j, x0, x1);
            step(wgv[j], wuv[j], sgb[j], sub[j], x0, x1);
        } else {
            step(wgv[j], wuv[j], sgb[j], sub[j], xv[XS ? 0 : j][0], xv[XS ? 0 : j][1]);
        }
    }
    for (int u = lane + IK_WARP * PRE; u < nu; u += IK_WARP) {
        uint4 x0, x1;
        xs2(u, x0, x1);
        step(ik_ldcs(wg + u), ik_ldcs(wu + u), sg[u], su[u], x0, x1);
    }
    const float g = round_bf16(ik_warp_sum(ag) * gscales[e * 3 + 0]);
    const float uu = round_bf16(ik_warp_sum(au) * gscales[e * 3 + 1]);
    const float a = round_bf16(g / (1.f + expf(-g)));
    if (lane == 0) h[(int64_t)p * I + r] = ik_f2b(a * uu);
}

template <typename OutT, bool XS>
__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_down_reduce_p1_v2_kernel(const ik_bf16* __restrict__ h, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                             const float* __restrict__ gscales, const int* __restrict__ ids, const float* __restrict__ wts,
                             OutT* __restrict__ out, int E, int H, int I, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int t = blockIdx.x;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int r = blockIdx.y * kRowWarps + warp;
    const bool rv = r < H;
    const int rr = rv ? r : 0;
    const int nu = I / 16;
    const ik_bf16* hrow = h + (int64_t)t * K * I;  // the token's K intermediate rows [K][I]
    const uint2* wd[kP1MaxK];
    const uint8_t* sd[kP1MaxK];
    float gd[kP1MaxK], acc[kP1MaxK];
    uint2 wv[kP1MaxK];
    uint32_t sb[kP1MaxK];
    uint4 hv[XS ? 1 : kP1MaxK][2];
#pragma unroll
    for (int k = 0; k < kP1MaxK; ++k) {
        const int id = k < K ? ids[(int64_t)t * K + k] : -1;
        const bool ok = id >= 0 && id < E;
        const int64_t row = ok ? (int64_t)id * H + rr : 0;
        wd[k] = ok ? reinterpret_cast<const uint2*>(dnw + row * (I / 2)) : nullptr;
        sd[k] = ok ? dns + row * (I / 16) : nullptr;
        gd[k] = ok ? gscales[id * 3 + 2] : 0.f;
        acc[k] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kP1MaxK; ++k) {  // iteration u = lane of every expert, requested before the table is built
        const bool in = wd[k] && lane < nu;
        wv[k] = in ? ik_ldcs(wd[k] + lane) : make_uint2(0, 0);
        sb[k] = in ? sd[k][lane] : 0u;
        if (!XS) {
            const uint4* hp = reinterpret_cast<const uint4*>(hrow + (int64_t)(k < K ? k : 0) * I + (in ? lane : 0) * 16);
            hv[XS ? 0 : k][0] = hp[0];
            hv[XS ? 0 : k][1] = hp[1];
        }
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    if (XS) {
        for (int i = threadIdx.x; i < K * I / 8; i += blockDim.x)
            reinterpret_cast<uint4*>(smem)[i] = reinterpret_cast<const uint4*>(hrow)[i];
    }
    __syncthreads();
    if (!rv) return;
    auto hload = [&](int k, int u, uint4& h0, uint4& h1) {
        const uint4* hp = reinterpret_cast<const uint4*>((XS ? reinterpret_cast<const ik_bf16*>(smem) : hrow) + (int64_t)k * I + u * 16);
        h0 = hp[0];
        h1 = hp[1];
    };
    auto step = [&](int k, const uint2& w2, uint32_t sbyte, const uint4& h0, const uint4& h1) {
        float fw[16], hf[16];
        ik_fp4x8_to_f32(w2.x, lut, fw);
        ik_fp4x8_to_f32(w2.y, lut, fw + 8);
        ik_unpack8(h0, hf);
        ik_unpack8(h1, hf + 8);
        float pp = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) pp = fmaf(fw[i], hf[i], pp);
        acc[k] = fmaf(pp, ik_e4m3_to_f(sbyte), acc[k]);
    };
    if (lane < nu) {
#pragma unroll
        for (int k = 0; k < kP1MaxK; ++k) {
            if (!wd[k]) continue;
            if (XS) {
                uint4 h0, h1;
                hload(k, lane, h0, h1);
                step(k, wv[k], sb[k], h0, h1);
            } else {
                step(k, wv[k], sb[k], hv[XS ? 0 : k][0], hv[XS ? 0 : k][1]);
            }
        }
    }
    for (int u = lane + IK_WARP; u < nu; u += IK_WARP) {
        uint2 wq[kP1MaxK];
        uint32_t sq[kP1MaxK];
#pragma unroll
        for (int k = 0; k < kP1MaxK; ++k) {
            wq[k] = wd[k] ? ik_ldcs(wd[k] + u) : make_uint2(0, 0);
            sq[k] = wd[k] ? sd[k][u] : 0u;
        }
#pragma unroll
        for (int k = 0; k < kP1MaxK; ++k) {
            if (!wd[k]) continue;
            uint4 h0, h1;
            hload(k, u, h0, h1);
            step(k, wq[k], sq[k], h0, h1);
        }
    }
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < kP1MaxK; ++k) {
        if (!wd[k]) continue;
        const float y = round_bf16(ik_warp_sum(acc[k]) * gd[k]);
        o += round_bf16(y * wts[(int64_t)t * K + k]);
    }
    if (lane == 0) ik_store(out + (int64_t)t * H + r, o);
}

// ---- decode fast path, v3: one launch -------------------------------------------------------------------------------
// The v2 pair as one grid of NG + ND blocks: roles [0, NG) are moe_gate_up_p1_v2_kernel<false, PRE>'s blocks (pair
// p = b / GY, row tile b % GY), roles [NG, NG + ND) are moe_down_reduce_p1_v2_kernel<bf16, false>'s (token t = b / GD,
// row tile b % GD), each with the same expressions in the same order, so every h value and every output is bit-identical
// to v2's. What changes:
//  * no kernel boundary between the two projections (about 3 us each inside a decode graph replay on gfx12,
//    docs/results/2026-10-19-hip-round4.md), and
//  * a down block requests its weights and block scales (all of them at I <= 512: one lane-iteration) before it waits for
//    h, so the down projection's DRAM traffic overlaps the gate_up tail instead of following it.
// Roles: a block's role is not its blockIdx but a ticket (thread 0: atomic add on sync[2] when the block starts): the first
// NG tickets run gate_up, the rest down. A down role therefore exists only once every gate_up role is held by a block that
// is already running, and gate_up blocks never wait, so the spin below cannot deadlock whatever order the hardware
// dispatches workgroups in (the rocPRIM / CUB tile-ticket rule: blockIdx order is not guaranteed by HIP).
// Synchronisation: every thread of a gate_up block fences (agent scope) after its h store, the block barrier, then thread 0
// adds 1 to sync[0] (release). A down block's thread 0 spins (acquire, agent scope, s_sleep) until sync[0] >= NG, the
// block barrier, then every thread fences (acquire, agent: drops stale L0 / L1 lines) before it reads h. The last down
// block to finish (sync[1]) puts the three counters back to 0, so the next launch on the stream (which starts after this
// one ends) finds them zeroed. sync is one 64-byte slot of g_moe_sync (module scope, zero when the code object loads), one
// slot per (device, stream), so launches on two streams never share counters.
constexpr int kMoeSyncSlots = 16;
constexpr int kMoeSyncStride = 16;  // unsigned ints: one 64-byte line per slot
__device__ unsigned int g_moe_sync[kMoeSyncSlots * kMoeSyncStride];

template <int PRE>
__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_experts_p1_v3_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ guw, const uint8_t* __restrict__ gus,
                         const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns, const float* __restrict__ gscales,
                         const int* __restrict__ ids, const float* __restrict__ wts, ik_bf16* h, ik_bf16* __restrict__ out,
                         int E, int H, int I, int K, int NG, int ND, int slot, int skew) {
    __shared__ float lut[16];
    __shared__ int ticket;
    unsigned int* sync = g_moe_sync + slot * kMoeSyncStride;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    if (threadIdx.x == 0) ticket = (int)__hip_atomic_fetch_add(sync + 2, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const int bid = ticket;  // the role
    if (bid < NG) {
        // test seam (INTARSIA_HIP_MOE_V3_SKEW, default 0): the last gate_up role starts late, so a down role that does not wait
        // for every gate_up role reads h before it is written (tests/moe_v3.rs v3_waits_for_every_gate_up_role). Bit 16
        // (INTARSIA_HIP_MOE_V3_SKEW_ALL=1): every gate_up role starts late, by 1 to 4 times the skew (tests/moe_v3.rs
        // v3_sentinel_stress: the h stores land while the down roles already hold the sentinel lines).
        // Bits 18..23 (INTARSIA_HIP_MOE_V3_SKEW_TOK=n, P2' M0): the late role is instead the last gate_up role of token n (its
        // down roles hold the earliest down tickets when n = 0, so they are resident and holding the sentinel lines at T >= 2);
        // bit 24 (SKEW_TOK=all): the last gate_up role of every token is late.
        const int sk = skew & 0xffff;
        const int tpr = K * ((I + kRowWarps - 1) / kRowWarps);  // gate_up roles per token
        const int tk = (skew >> 18) & 63;
        const bool late = ((skew >> 24) & 1) ? (bid + 1) % tpr == 0 : bid == (tk ? min(NG, tk * tpr) : NG) - 1;
        const int nsleep = ((skew >> 16) & 1) ? sk * (1 + bid % 4) : (late ? sk : 0);
        for (int i = 0; i < nsleep; ++i) __builtin_amdgcn_s_sleep(127);
        // ---- gate_up: moe_gate_up_p1_v2_kernel<false, PRE> with blockIdx = (role / GY, role % GY) ----
        const int GY = (I + kRowWarps - 1) / kRowWarps;
        const int p = bid / GY;
        const int e = ids[p];
        const bool ok = e >= 0 && e < E;  // block-uniform
        const int r = (bid % GY) * kRowWarps + warp;
        const bool live = ok && r < I;
        const int ee = ok ? e : 0, rr = r < I ? r : 0;
        const uint2* wg = reinterpret_cast<const uint2*>(guw + ((int64_t)ee * 2 * I + rr) * (H / 2));
        const uint2* wu = reinterpret_cast<const uint2*>(guw + ((int64_t)ee * 2 * I + I + rr) * (H / 2));
        const uint8_t* sg = gus + ((int64_t)ee * 2 * I + rr) * (H / 16);
        const uint8_t* su = gus + ((int64_t)ee * 2 * I + I + rr) * (H / 16);
        const ik_bf16* xr = x + (int64_t)(p / K) * H;
        const int nu = H / 16;
        uint2 wgv[PRE], wuv[PRE];
        uint32_t sgb[PRE], sub[PRE];
        uint4 xv[PRE][2];
#pragma unroll
        for (int j = 0; j < PRE; ++j) {
            const int u = lane + IK_WARP * j;
            const bool in = u < nu;
            wgv[j] = in ? ik_ldcs(wg + u) : make_uint2(0, 0);
            wuv[j] = in ? ik_ldcs(wu + u) : make_uint2(0, 0);
            sgb[j] = in ? sg[u] : 0u;
            sub[j] = in ? su[u] : 0u;
            const uint4* xp = reinterpret_cast<const uint4*>(xr + (in ? u : 0) * 16);
            xv[j][0] = xp[0];
            xv[j][1] = xp[1];
        }
        if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
        __syncthreads();
        if (live) {
            float ag = 0.f, au = 0.f;
            auto step = [&](const uint2& g2, const uint2& u2, uint32_t gsb, uint32_t usb, const uint4& x0, const uint4& x1) {
                const float sgv = ik_e4m3_to_f(gsb), suv = ik_e4m3_to_f(usb);
                float fg[16], fu[16];
                ik_fp4x8_to_f32(g2.x, lut, fg);
                ik_fp4x8_to_f32(g2.y, lut, fg + 8);
                ik_fp4x8_to_f32(u2.x, lut, fu);
                ik_fp4x8_to_f32(u2.y, lut, fu + 8);
                float xf[16];
                ik_unpack8(x0, xf);
                ik_unpack8(x1, xf + 8);
                float pg = 0.f, pu = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    pg = fmaf(fg[i], xf[i], pg);
                    pu = fmaf(fu[i], xf[i], pu);
                }
                ag = fmaf(pg, sgv, ag);
                au = fmaf(pu, suv, au);
            };
#pragma unroll
            for (int j = 0; j < PRE; ++j) {
                if (lane + IK_WARP * j >= nu) continue;
                step(wgv[j], wuv[j], sgb[j], sub[j], xv[j][0], xv[j][1]);
            }
            for (int u = lane + IK_WARP * PRE; u < nu; u += IK_WARP) {
                const uint4* xp = reinterpret_cast<const uint4*>(xr) + u * 2;
                const uint4 x0 = xp[0], x1 = xp[1];
                step(ik_ldcs(wg + u), ik_ldcs(wu + u), sg[u], su[u], x0, x1);
            }
            const float g = round_bf16(ik_warp_sum(ag) * gscales[e * 3 + 0]);
            const float uu = round_bf16(ik_warp_sum(au) * gscales[e * 3 + 1]);
            const float a = round_bf16(g / (1.f + expf(-g)));
            if (lane == 0) h[(int64_t)p * I + r] = ik_f2b(a * uu);
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) __hip_atomic_fetch_add(sync, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        return;
    }

    // ---- down + reduce: moe_down_reduce_p1_v2_kernel<bf16, false> with blockIdx = (role / GD, role % GD) ----
    const int bd = bid - NG;
    const int GD = (H + kRowWarps - 1) / kRowWarps;
    const int t = bd / GD;
    const int r = (bd % GD) * kRowWarps + warp;
    const bool rv = r < H;
    const int rr = rv ? r : 0;
    const int nu = I / 16;
    const ik_bf16* hrow = h + (int64_t)t * K * I;  // the token's K intermediate rows [K][I]
    const uint2* wd[kP1MaxK];
    const uint8_t* sd[kP1MaxK];
    float gd[kP1MaxK], acc[kP1MaxK];
    uint2 wv[kP1MaxK];
    uint32_t sb[kP1MaxK];
#pragma unroll
    for (int k = 0; k < kP1MaxK; ++k) {
        const int id = k < K ? ids[(int64_t)t * K + k] : -1;
        const bool okk = id >= 0 && id < E;
        const int64_t row = okk ? (int64_t)id * H + rr : 0;
        wd[k] = okk ? reinterpret_cast<const uint2*>(dnw + row * (I / 2)) : nullptr;
        sd[k] = okk ? dns + row * (I / 16) : nullptr;
        gd[k] = okk ? gscales[id * 3 + 2] : 0.f;
        acc[k] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kP1MaxK; ++k) {  // iteration u = lane of every expert, requested before the wait
        const bool in = wd[k] && lane < nu;
        wv[k] = in ? ik_ldcs(wd[k] + lane) : make_uint2(0, 0);
        sb[k] = in ? sd[k][lane] : 0u;
    }
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    // test seam (bit 17, INTARSIA_HIP_MOE_V3_PREREAD=1; default off): every wave reads the h lines it will use before the
    // wait, so they sit in this block's caches when the gate_up roles store h. Only the acquire below makes the later reads
    // see the stores (tests/moe_v3.rs v3_sentinel_stress, the test that the v3-nofence mutant must fail).
    if (((skew >> 17) & 1) && rv) {
        uint32_t t = 0;
        for (int k = 0; k < kP1MaxK; ++k) {
            if (!wd[k]) continue;
            for (int u = lane; u < nu; u += IK_WARP) {
                const uint4* hp = reinterpret_cast<const uint4*>(hrow + (int64_t)k * I + u * 16);
                const uint4 a = hp[0], b = hp[1];
                t ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
            }
        }
        __asm__ volatile("" ::"v"(t));
    }
    if (threadIdx.x == 0) {
        while (__hip_atomic_load(sync, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)NG) __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");  // every wave: h is read only after this
    if (rv) {
        auto hload = [&](int k, int u, uint4& h0, uint4& h1) {
            const uint4* hp = reinterpret_cast<const uint4*>(hrow + (int64_t)k * I + u * 16);
            h0 = hp[0];
            h1 = hp[1];
        };
        auto step = [&](int k, const uint2& w2, uint32_t sbyte, const uint4& h0, const uint4& h1) {
            float fw[16], hf[16];
            ik_fp4x8_to_f32(w2.x, lut, fw);
            ik_fp4x8_to_f32(w2.y, lut, fw + 8);
            ik_unpack8(h0, hf);
            ik_unpack8(h1, hf + 8);
            float pp = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) pp = fmaf(fw[i], hf[i], pp);
            acc[k] = fmaf(pp, ik_e4m3_to_f(sbyte), acc[k]);
        };
        if (lane < nu) {
#pragma unroll
            for (int k = 0; k < kP1MaxK; ++k) {
                if (!wd[k]) continue;
                uint4 h0, h1;
                hload(k < K ? k : 0, lane, h0, h1);
                step(k, wv[k], sb[k], h0, h1);
            }
        }
        for (int u = lane + IK_WARP; u < nu; u += IK_WARP) {
            uint2 wq[kP1MaxK];
            uint32_t sq[kP1MaxK];
#pragma unroll
            for (int k = 0; k < kP1MaxK; ++k) {
                wq[k] = wd[k] ? ik_ldcs(wd[k] + u) : make_uint2(0, 0);
                sq[k] = wd[k] ? sd[k][u] : 0u;
            }
#pragma unroll
            for (int k = 0; k < kP1MaxK; ++k) {
                if (!wd[k]) continue;
                uint4 h0, h1;
                hload(k, u, h0, h1);
                step(k, wq[k], sq[k], h0, h1);
            }
        }
        float o = 0.f;
#pragma unroll
        for (int k = 0; k < kP1MaxK; ++k) {
            if (!wd[k]) continue;
            const float y = round_bf16(ik_warp_sum(acc[k]) * gd[k]);
            o += round_bf16(y * wts[(int64_t)t * K + k]);
        }
        if (lane == 0) ik_store(out + (int64_t)t * H + r, o);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = __hip_atomic_fetch_add(sync + 1, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (done == (unsigned)ND - 1) {  // every down block is past its wait: reset for the next launch on this stream
            __hip_atomic_store(sync, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(sync + 1, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(sync + 2, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// The g_moe_sync slot of the calling thread's (device, stream); -1 when all kMoeSyncSlots are taken (the caller then runs
// the two v2 launches).
int moe_sync_slot(hipStream_t s) {
    static std::mutex mu;
    static std::vector<std::pair<int, hipStream_t>> owners;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) return -1;
    std::lock_guard<std::mutex> g(mu);
    for (size_t i = 0; i < owners.size(); ++i)
        if (owners[i].first == dev && owners[i].second == s) return (int)i;
    if ((int)owners.size() >= kMoeSyncSlots) return -1;
    owners.emplace_back(dev, s);
    return (int)owners.size() - 1;
}

// INTARSIA_HIP_XSTAGE=1 stages the activations in shared memory (when they fit); default 0
bool xstage_enabled() {
    static const bool on = [] {
        const char* e = getenv("INTARSIA_HIP_XSTAGE");
        return e && e[0] == '1';
    }();
    return on;
}

// INTARSIA_HIP_MOE_PRE = how many lane-iterations of the gate_up kernel are requested before its barrier (1, 2 or 4)
int moe_pre() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_HIP_MOE_PRE");
        const int x = e ? atoi(e) : 4;
        return (x == 1 || x == 2) ? x : 4;
    }();
    return v;
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


int moe_topk_impl(const ik_tensor* logits, int top_k, int norm, ik_tensor* weights, ik_tensor* ids, void* stream, bool v2) {
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
    if (v2)
        moe_topk_v2_kernel<<<grid, kTopkWarps * IK_WARP, 0, IK_STREAM(stream)>>>(
            (const ik_bf16*)logits->data, (int)T, (int)E, top_k, norm, (float*)weights->data, (int*)ids->data);
    else
        moe_topk_kernel<<<grid, kTopkWarps * IK_WARP, 0, IK_STREAM(stream)>>>(
            (const ik_bf16*)logits->data, (int)T, (int)E, top_k, norm, (float*)weights->data, (int*)ids->data);
    return ik_launch_status();
}

int moe_experts_impl(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                     const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                     const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream, bool v2,
                     bool v3 = false, bool allow_wmma = true) {
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
    if (!allow_wmma && T > 16) return IK_ERR_SHAPE;  // the row-exact entry (ik_moe_experts_bf16_rows): at most 16 tokens
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
    hipStream_t s = IK_STREAM(stream);
    const unsigned slots = (unsigned)(P < E ? P : E);

    if (P <= 16 && K <= kP1MaxK) {  // decode: no grouping, down projection fused with the reduction
        const dim3 g1((unsigned)P, (unsigned)((I + kRowWarps - 1) / kRowWarps));
        const dim3 g2((unsigned)T, (unsigned)((H + kRowWarps - 1) / kRowWarps));
        if (v2) {
            const size_t xb = (size_t)H * 2, hb = (size_t)K * I * 2;
            const bool xs = xstage_enabled() && xb <= 32768 && hb <= 32768;
            if (v3 && !xs) {  // one launch (moe_experts_p1_v3_kernel), bit-identical to the two below
                const int slot = moe_sync_slot(s);
                // test seams, read at every launch (unset = off): SKEW (bits 0..15), SKEW_ALL (bit 16), PREREAD (bit 17), SKEW_TOK (18..24)
                const char* sk = getenv("INTARSIA_HIP_MOE_V3_SKEW");
                int skew = sk ? (atoi(sk) & 0xffff) : 0;
                if (const char* a = getenv("INTARSIA_HIP_MOE_V3_SKEW_ALL"); a && a[0] == '1') skew |= 1 << 16;
                if (const char* q = getenv("INTARSIA_HIP_MOE_V3_PREREAD"); q && q[0] == '1') skew |= 1 << 17;
                // SKEW_TOK (bits 18..24, see the kernel): "all", or a token index n (stored as n + 1, n < 63)
                if (const char* tt = getenv("INTARSIA_HIP_MOE_V3_SKEW_TOK"); tt && tt[0]) {
                    if (tt[0] == 'a') skew |= 1 << 24;
                    else if (atoi(tt) >= 0 && atoi(tt) < 63) skew |= (atoi(tt) + 1) << 18;
                }
                const int64_t NG = P * ((I + kRowWarps - 1) / kRowWarps), ND = T * ((H + kRowWarps - 1) / kRowWarps);
                if (slot >= 0 && NG + ND <= 0x7fffffffLL) {
#define IK_EX_V3(PRE)                                                                                                 \
    moe_experts_p1_v3_kernel<PRE><<<(unsigned)(NG + ND), kRowWarps * IK_WARP, 0, s>>>(                                \
        (const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data, (const uint8_t*)gate_up_s->data,                    \
        (const uint8_t*)down_w->data, (const uint8_t*)down_s->data, (const float*)gscales->data, (const int*)ids->data, \
        (const float*)weights->data, hbuf, (ik_bf16*)out->data, (int)E, (int)H, (int)I, (int)K, (int)NG, (int)ND, slot, skew)
                    if (moe_pre() == 1) IK_EX_V3(1);
                    else if (moe_pre() == 2) IK_EX_V3(2);
                    else IK_EX_V3(4);
#undef IK_EX_V3
                    return ik_launch_status();
                }
            }
            if (xs)
                moe_gate_up_p1_v2_kernel<true, 4><<<g1, kRowWarps * IK_WARP, xb, s>>>(
                    (const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data, (const uint8_t*)gate_up_s->data,
                    (const float*)gscales->data, (const int*)ids->data, hbuf, (int)E, (int)H, (int)I, (int)K);
            else {
#define IK_GU_V2(PRE)                                                                                                 \
    moe_gate_up_p1_v2_kernel<false, PRE><<<g1, kRowWarps * IK_WARP, 0, s>>>(                                          \
        (const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data, (const uint8_t*)gate_up_s->data,                    \
        (const float*)gscales->data, (const int*)ids->data, hbuf, (int)E, (int)H, (int)I, (int)K)
                if (moe_pre() == 1) IK_GU_V2(1);
                else if (moe_pre() == 2) IK_GU_V2(2);
                else IK_GU_V2(4);
#undef IK_GU_V2
            }
            IK_TRY(ik_launch_status());
            if (xs)
                moe_down_reduce_p1_v2_kernel<ik_bf16, true><<<g2, kRowWarps * IK_WARP, hb, s>>>(
                    hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data, (const float*)gscales->data,
                    (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)I, (int)K);
            else
                moe_down_reduce_p1_v2_kernel<ik_bf16, false><<<g2, kRowWarps * IK_WARP, 0, s>>>(
                    hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data, (const float*)gscales->data,
                    (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)I, (int)K);
            return ik_launch_status();
        }
        moe_gate_up_p1_kernel<<<g1, kRowWarps * IK_WARP, 0, s>>>(
            (const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data, (const uint8_t*)gate_up_s->data,
            (const float*)gscales->data, (const int*)ids->data, hbuf, (int)E, (int)H, (int)I, (int)K);
        IK_TRY(ik_launch_status());
        moe_down_reduce_p1_kernel<ik_bf16><<<g2, kRowWarps * IK_WARP, 0, s>>>(
            hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data, (const float*)gscales->data,
            (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)I, (int)K);
        return ik_launch_status();
    }
    // the tensor-core kernels load expert weights as 16-byte vectors and block scales as 16-bit pairs; the checks above
    // only guarantee 8 and 1 bytes, so an unaligned view takes the GEMV-style (8-byte / byte access) kernels instead
    const bool wmma_aligned = ik_aligned16(gate_up_w->data) && ik_aligned16(down_w->data) &&
                              (reinterpret_cast<uintptr_t>(gate_up_s->data) & 3u) == 0 &&
                              (reinterpret_cast<uintptr_t>(down_s->data) & 3u) == 0;
    moe_group_kernel<<<1, 1024, (2 * E + 32) * sizeof(int), s>>>((const int*)ids->data, (int)P, (int)E, pairs, offsets, active);
    IK_TRY(ik_launch_status());
    if (allow_wmma && T >= kMwMinTokens && H % kMwKB == 0 && I % kMwKB == 0 && wmma_aligned && moe_wmma_enabled()) {  // prefill-sized: tensor-core grouped projections
        const int64_t avg = (P + E - 1) / E;  // pairs per expert, on average
        if (moe_pairs(avg) == 64)
            moe_wmma_launch<64>(T, P, E, H, I, K, slots, (const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data,
                                (const uint8_t*)gate_up_s->data, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data,
                                (const float*)gscales->data, pairs, offsets, active, hbuf, ybuf, s);
        else
            moe_wmma_launch<32>(T, P, E, H, I, K, slots, (const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data,
                                (const uint8_t*)gate_up_s->data, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data,
                                (const float*)gscales->data, pairs, offsets, active, hbuf, ybuf, s);
        IK_TRY(ik_launch_status());
    } else {
        moe_gate_up_kernel<<<dim3(slots, (unsigned)((I + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                             (size_t)PC * H * 2, s>>>((const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data,
                                                     (const uint8_t*)gate_up_s->data, (const float*)gscales->data, pairs,
                                                     offsets, active, hbuf, (int)E, (int)H, (int)I, (int)K, PC);
        IK_TRY(ik_launch_status());
        moe_down_kernel<<<dim3(slots, (unsigned)((H + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                          (size_t)PC * I * 2, s>>>(hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data,
                                                  (const float*)gscales->data, pairs, offsets, active, ybuf, (int)E, (int)H,
                                                  (int)I, PC);
        IK_TRY(ik_launch_status());
    }
    if (H % 8 == 0 && ik_aligned16(ybuf) && ik_aligned16(out->data) && moe_reduce_v8_enabled())
        moe_reduce_v8_kernel<<<dim3((unsigned)T, (unsigned)((H / 8 + 255) / 256)), 256, 0, s>>>(
            ybuf, (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)K);
    else
        moe_reduce_kernel<ik_bf16><<<dim3((unsigned)T, (unsigned)((H + 255) / 256)), 256, 0, s>>>(
            ybuf, (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_hip_moe_mw(int pairs) { return moe_mw_variant(pairs, false); }  // ik_hip_abi.h (registry pin): gate_up
int ik_hip_moe_mw_down(int pairs) { return moe_mw_variant(pairs, true); }
// ik_hip_abi.h (round 9 pin, tests/split_trigger.rs): the launch-piece count ik_split_pieces (ik_hip.h) gives
int ik_hip_split_pieces(double est_us, int64_t max_pieces) { return ik_split_pieces(est_us, max_pieces); }

int ik_moe_topk_bf16(const ik_tensor* logits, int top_k, int norm, ik_tensor* weights, ik_tensor* ids, void* stream) {
    return moe_topk_impl(logits, top_k, norm, weights, ids, stream, false);
}
int ik_moe_topk_bf16_v2(const ik_tensor* logits, int top_k, int norm, ik_tensor* weights, ik_tensor* ids, void* stream) {
    return moe_topk_impl(logits, top_k, norm, weights, ids, stream, true);
}

int ik_moe_experts_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                        const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                        const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return moe_experts_impl(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, false);
}
// v2: decode (P <= 16) takes the latency-restructured p1 kernels; every other shape runs the v1 path unchanged
int ik_moe_experts_bf16_v2(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                           const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                           const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return moe_experts_impl(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true);
}
// moe_experts_bf16_rows (HIP only, ik_hip_abi.h; lane hb, batched decode): T = 1..16 tokens, token t bitwise the T = 1
// ik_moe_experts_bf16_v2 call on token t alone. v2's decode path (P = T * K <= 16: the p1 kernels) and its grouped GEMV
// path (every other P: per-pair accumulation independent of the grouping and of the other tokens) already give a token the
// bits a lone token gets; what does not is the tensor-core path (T >= 16), which accumulates in another order. This entry
// never takes it, whatever INTARSIA_HIP_MOE_WMMA says.
int ik_moe_experts_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                             const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                             const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return moe_experts_impl(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true, false,
                            false);
}
// v3 (HIP only, ik_hip_abi.h): decode (P <= 16) in one launch, bit-identical to v2; every other shape is v2
int ik_moe_experts_bf16_v3(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                           const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s,
                           const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return moe_experts_impl(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true, true);
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
