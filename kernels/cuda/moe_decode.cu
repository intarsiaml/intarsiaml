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
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace {

constexpr int kRowWarps = 8;
constexpr int kMaxK = 16;  // experts per token held in registers by the down kernel

IK_DEVI float round_bf16(float v) { return ik_b2f(ik_f2b(v)); }

template <int U, bool GELU>
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
#pragma unroll U
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
    const float a = round_bf16(GELU ? ik_gelu_tanh(g) : g / (1.f + expf(-g)));
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

// v2 down projection: the same arithmetic as moe_dec_down_kernel, restructured because that kernel keeps all K experts of a
// row in registers (160 per thread: ONE 256-thread block per SM, 256 blocks = 5.3 waves, 94 GB/s on the 35B's down weights).
// Here a warp owns an expert SLOT (warp w: slots w, w + 8, ...) and R output rows of the block: R independent weight loads
// per lane, no activation staging (each lane reads its own 32 bytes of the slot's h row), few registers. The K per-slot terms
// round(round(down_k(h_k)) * w_k) go through shared memory and are summed in slot order 0..K-1 by one thread per output
// element: the same additions in the same order as the one-warp-per-row kernel, so the outputs are bitwise equal.
// An invalid id stores 0 (x + 0 == x here: the running total starts at +0 and so is never -0).
// TF: the grid is (token, row block) instead of (row block, token): blocks launch x-fastest, so the blocks of all T tokens for one
// row block run next to each other and tokens that route to the same expert read the same weight rows out of L2 instead of HBM.
// Only the block-to-(token, rows) map differs; the arithmetic of a (token, row) is the same code (the multi-token launchers use it).
template <int R, bool TF>
__global__ void __launch_bounds__(kRowWarps * IK_WARP)
moe_dec_down_v2_kernel(const ik_bf16* __restrict__ h, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                       const float* __restrict__ gscales, const int* __restrict__ ids, const float* __restrict__ wts,
                       ik_bf16* __restrict__ out, int E, int H, int I, int K) {
    __shared__ float term[kMaxK][R];
    const int t = TF ? blockIdx.x : blockIdx.y;
    const int r0 = (TF ? blockIdx.y : blockIdx.x) * R;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    for (int k = warp; k < K; k += kRowWarps) {
        const int v = ids[(int64_t)t * K + k];
        const bool ok = v >= 0 && v < E;
        float acc[R];
#pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = 0.f;
        if (ok) {
            const uint2* wrow[R];
            const uint8_t* srow[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int64_t row = (int64_t)v * H + (r0 + i < H ? r0 + i : H - 1);  // rows >= H are read, never stored
                wrow[i] = reinterpret_cast<const uint2*>(dnw + row * (I / 2));
                srow[i] = dns + row * (I / 16);
            }
            const uint4* hp = reinterpret_cast<const uint4*>(h + ((int64_t)t * K + k) * I);
#pragma unroll 1
            for (int u = lane; u < I / 16; u += IK_WARP) {
                uint2 wv[R];
                float sv[R];
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    wv[i] = __ldcs(wrow[i] + u);
                    sv[i] = ik_e4m3_to_f(srow[i][u]);
                }
                float hf[16];
                ik_unpack8(hp[2 * u], hf);
                ik_unpack8(hp[2 * u + 1], hf + 8);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    float fw[16];
                    ik_fp4x8_to_f32(wv[i].x, fw);
                    ik_fp4x8_to_f32(wv[i].y, fw + 8);
                    float pdot = 0.f;
#pragma unroll
                    for (int e = 0; e < 16; ++e) pdot = fmaf(fw[e], hf[e], pdot);
                    acc[i] = fmaf(pdot, sv[i], acc[i]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const float s = ik_warp_sum(acc[i]);
            if (lane == 0)
                term[k][i] = ok ? round_bf16(round_bf16(s * gscales[v * 3 + 2]) * wts[(int64_t)t * K + k]) : 0.f;
        }
    }
    __syncthreads();
    if (threadIdx.x < R && r0 + (int)threadIdx.x < H) {
        float total = 0.f;
        for (int k = 0; k < K; ++k) total += term[k][threadIdx.x];
        out[(int64_t)t * H + r0 + threadIdx.x] = ik_f2b(total);
    }
}

// Dedup ("dd") down projection for the multi-token (rows) launchers, lane moedn. v2 runs every (token, slot) pair as its own
// block group, so an expert that 2 tokens route to is read and decoded twice (128 pairs on ~61 distinct experts at T = 16, the
// second read from L2 at best), in 8192 one-iteration blocks. Here a block owns RT = R * G output rows of ALL T tokens: it lists
// the T * K pairs grouped by distinct valid expert in shared memory (a counting sort), and its warps take (expert, R-row group)
// items from a shared counter. A warp loads the item's R weight rows once (the next item's rows and first h chunk prefetched
// into registers while the current one computes, the next pair's h chunk while the current pair computes) and runs every pair
// routed to that expert on them. Per (pair, row) it is v2's code: the lane-strided u loop, the 16-term pdot
// fmaf chain, acc = fmaf(pdot, sv, acc) from 0, ik_warp_sum, round(round(s * g) * w); the term lands in shared memory under
// its slot, and one thread per (token, row) sums slots 0..K-1 from +0 as v2 does, so the outputs are bitwise v2's (and so the
// T = 1 kernel's). Which warp runs which item, and in what order, cannot change a bit. An invalid id's slot holds 0, as in v2.
// Shared memory: P * RT floats of terms and 6 * P + 1 ints (P = T * K <= 256): at most 38 KB.
template <int R>
IK_DEVI void dd_load_rows(const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns, int v, int rb, int H, int I,
                          int u, uint2* wv, float* sv) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int64_t row = (int64_t)v * H + (rb + i < H ? rb + i : H - 1);  // rows >= H are read, never stored
        wv[i] = __ldcs(reinterpret_cast<const uint2*>(dnw + row * (I / 2)) + u);
        sv[i] = ik_e4m3_to_f(dns[row * (I / 16) + u]);
    }
}

template <int R, int G>
__global__ void __launch_bounds__(kRowWarps * IK_WARP, 3)
moe_dec_down_dd_kernel(const ik_bf16* __restrict__ h, const uint8_t* __restrict__ dnw, const uint8_t* __restrict__ dns,
                       const float* __restrict__ gscales, const int* __restrict__ ids, const float* __restrict__ wts,
                       ik_bf16* __restrict__ out, int T, int E, int H, int I, int K) {
    constexpr int RT = R * G;
    extern __shared__ __align__(16) float dd_smem[];
    __shared__ int nlead, next_item;
    const int P = T * K;
    float* term = dd_smem;                                     // [P][RT]
    int* sid = reinterpret_cast<int*>(term + (size_t)P * RT);  // [P] the pair's expert, -1 if invalid
    int* fst = sid + P;                                        // [P] the first pair on the same expert
    int* rnk = fst + P;                                        // [P] the pair's rank among its expert's pairs
    int* dix = rnk + P;                                        // [P] at a first pair: its expert's index d
    int* ord = dix + P;                                        // [P] the valid pairs grouped by expert
    int* off = ord + P;                                        // [P + 1] expert d's pairs are ord[off[d] .. off[d + 1])
    const int r0 = blockIdx.x * RT;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    if (threadIdx.x == 0) nlead = 0, next_item = kRowWarps;
    for (int p = threadIdx.x; p <= P; p += blockDim.x) {
        off[p] = 0;
        if (p < P) {
            const int v = ids[p];
            sid[p] = v >= 0 && v < E ? v : -1;
        }
    }
    __syncthreads();
    // counting sort of the pairs by expert (experts in first-come order of the atomic: it only decides the work order)
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const int v = sid[p];
        if (v < 0) {
            for (int i = 0; i < RT; ++i) term[p * RT + i] = 0.f;
            continue;
        }
        int f = p, n = 0;
        for (int q = 0; q < p; ++q)
            if (sid[q] == v) f = n++ ? f : q;
        fst[p] = f;
        rnk[p] = n;
        if (f == p) dix[p] = atomicAdd(&nlead, 1);
    }
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += blockDim.x)
        if (sid[p] >= 0) atomicAdd(&off[dix[fst[p]] + 1], 1);
    __syncthreads();
    if (warp == 0) {
        int carry = 0;
        for (int b = 0; b < nlead; b += IK_WARP) {
            int x = b + lane < nlead ? off[b + lane + 1] : 0;
#pragma unroll
            for (int s = 1; s < IK_WARP; s <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, s);
                if (lane >= s) x += y;
            }
            if (b + lane < nlead) off[b + lane + 1] = x + carry;
            carry += __shfl_sync(0xffffffffu, x, IK_WARP - 1);
        }
    }
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += blockDim.x)
        if (sid[p] >= 0) ord[off[dix[fst[p]]] + rnk[p]] = p;
    __syncthreads();
    const int items = nlead * G;
    if (I / 16 <= IK_WARP) {
        // one u per lane (the 35B: I / 16 = 32): the item's weights stay in registers for all its pairs; items are handed out
        // by a shared counter, and the next item's weights and first h chunk load while the current one computes
        const int u = lane < I / 16 ? lane : 0;
        const bool on = lane < I / 16;
        uint2 nw[R];
        float ns[R];
        uint4 nh0 = make_uint4(0, 0, 0, 0), nh1 = nh0;
        int it = warp;
        if (it < items) {
            dd_load_rows<R>(dnw, dns, sid[ord[off[it / G]]], r0 + (it % G) * R, H, I, u, nw, ns);
            const uint4* hp = reinterpret_cast<const uint4*>(h + (int64_t)ord[off[it / G]] * I);
            nh0 = hp[2 * u], nh1 = hp[2 * u + 1];
        }
        while (it < items) {
            uint2 wv[R];
            float sv[R];
#pragma unroll
            for (int i = 0; i < R; ++i) wv[i] = nw[i], sv[i] = ns[i];
            uint4 ha0 = nh0, ha1 = nh1;
            const int d = it / G, g = it % G, o0 = off[d], o1 = off[d + 1];
            const int v = sid[ord[o0]];
            int ni = 0;
            if (lane == 0) ni = atomicAdd(&next_item, 1);
            it = __shfl_sync(0xffffffffu, ni, 0);
            if (it < items) {
                const int pn = ord[off[it / G]];
                dd_load_rows<R>(dnw, dns, sid[pn], r0 + (it % G) * R, H, I, u, nw, ns);
                const uint4* hp = reinterpret_cast<const uint4*>(h + (int64_t)pn * I);
                nh0 = hp[2 * u], nh1 = hp[2 * u + 1];
            }
            const float gs = gscales[v * 3 + 2];
            for (int j = o0; j < o1; ++j) {
                const int p = ord[j];
                uint4 hb0 = ha0, hb1 = ha1;
                if (j + 1 < o1) {
                    const uint4* hp = reinterpret_cast<const uint4*>(h + (int64_t)ord[j + 1] * I);
                    hb0 = hp[2 * u], hb1 = hp[2 * u + 1];
                }
                float acc[R];
#pragma unroll
                for (int i = 0; i < R; ++i) acc[i] = 0.f;
                if (on) {
                    float hf[16];
                    ik_unpack8(ha0, hf);
                    ik_unpack8(ha1, hf + 8);
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        float fw[16];
                        ik_fp4x8_to_f32(wv[i].x, fw);
                        ik_fp4x8_to_f32(wv[i].y, fw + 8);
                        float pdot = 0.f;
#pragma unroll
                        for (int e = 0; e < 16; ++e) pdot = fmaf(fw[e], hf[e], pdot);
                        acc[i] = fmaf(pdot, sv[i], acc[i]);
                    }
                }
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const float s = ik_warp_sum(acc[i]);
                    if (lane == 0) term[p * RT + g * R + i] = round_bf16(round_bf16(s * gs) * wts[p]);
                }
                ha0 = hb0, ha1 = hb1;
            }
        }
    } else {
        // the general u loop (I / 16 > 32): per pair, v2's loop; the item's rows come from L1 / L2 after the first pair
        for (int it = warp; it < items; it += kRowWarps) {
            const int d = it / G, g = it % G, rb = r0 + g * R;
            const int v = sid[ord[off[d]]];
            const float gs = gscales[v * 3 + 2];
            const uint2* wrow[R];
            const uint8_t* srow[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int64_t row = (int64_t)v * H + (rb + i < H ? rb + i : H - 1);  // rows >= H are read, never stored
                wrow[i] = reinterpret_cast<const uint2*>(dnw + row * (I / 2));
                srow[i] = dns + row * (I / 16);
            }
            for (int j = off[d]; j < off[d + 1]; ++j) {
                const int p = ord[j];
                float acc[R];
#pragma unroll
                for (int i = 0; i < R; ++i) acc[i] = 0.f;
                const uint4* hp = reinterpret_cast<const uint4*>(h + (int64_t)p * I);
#pragma unroll 1
                for (int u = lane; u < I / 16; u += IK_WARP) {
                    uint2 wv[R];
                    float sv[R];
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        wv[i] = wrow[i][u];
                        sv[i] = ik_e4m3_to_f(srow[i][u]);
                    }
                    float hf[16];
                    ik_unpack8(hp[2 * u], hf);
                    ik_unpack8(hp[2 * u + 1], hf + 8);
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        float fw[16];
                        ik_fp4x8_to_f32(wv[i].x, fw);
                        ik_fp4x8_to_f32(wv[i].y, fw + 8);
                        float pdot = 0.f;
#pragma unroll
                        for (int e = 0; e < 16; ++e) pdot = fmaf(fw[e], hf[e], pdot);
                        acc[i] = fmaf(pdot, sv[i], acc[i]);
                    }
                }
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const float s = ik_warp_sum(acc[i]);
                    if (lane == 0) term[p * RT + g * R + i] = round_bf16(round_bf16(s * gs) * wts[p]);
                }
            }
        }
    }
    __syncthreads();
    for (int o = threadIdx.x; o < T * RT; o += blockDim.x) {
        const int t = o / RT, i = o % RT;
        if (r0 + i >= H) continue;
        float total = 0.f;
        for (int k = 0; k < K; ++k) total += term[(t * K + k) * RT + i];
        out[(int64_t)t * H + r0 + i] = ik_f2b(total);
    }
}

inline int64_t r256(int64_t v) { return (v + 255) & ~int64_t(255); }

// Tuning knobs for A/B runs (host side, read when the launch is recorded): rows per block of the v2 down kernel
// (INTARSIA_MOE_DOWN_R = 1|2|4|8) and the unroll of the gate_up loop (INTARSIA_MOE_GU_UNROLL = 1|2|4).
int knob(const char* name, int dflt, int a, int b, int c, int d) {
    if (const char* v = getenv(name)) {
        const int x = atoi(v);
        if (x == a || x == b || x == c || x == d) return x;
    }
    return dflt;
}

template <int U, bool GELU>
void launch_gate_up(const ik_tensor* x, const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* gscales,
                    const ik_tensor* ids, ik_bf16* hbuf, int64_t P, int64_t E, int64_t H, int64_t I, int64_t K,
                    cudaStream_t s) {
    moe_dec_gate_up_kernel<U, GELU><<<dim3((unsigned)P, (unsigned)((I + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                                (size_t)H * 2, s>>>((const ik_bf16*)x->data, (const uint8_t*)gate_up_w->data,
                                                    (const uint8_t*)gate_up_s->data, (const float*)gscales->data,
                                                    (const int*)ids->data, hbuf, (int)E, (int)H, (int)I, (int)K);
}

template <int R, bool TF>
void launch_down_v2(const ik_bf16* hbuf, const ik_tensor* down_w, const ik_tensor* down_s, const ik_tensor* gscales,
                    const ik_tensor* ids, const ik_tensor* weights, ik_tensor* out, int64_t T, int64_t E, int64_t H,
                    int64_t I, int64_t K, cudaStream_t s) {
    const unsigned rb = (unsigned)((H + R - 1) / R);
    moe_dec_down_v2_kernel<R, TF><<<TF ? dim3((unsigned)T, rb) : dim3(rb, (unsigned)T), kRowWarps * IK_WARP, 0, s>>>(
        hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data, (const float*)gscales->data,
        (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H, (int)I, (int)K);
}

template <int R, int RT>
void launch_down_dd(const ik_bf16* hbuf, const ik_tensor* down_w, const ik_tensor* down_s, const ik_tensor* gscales,
                    const ik_tensor* ids, const ik_tensor* weights, ik_tensor* out, int64_t T, int64_t E, int64_t H,
                    int64_t I, int64_t K, cudaStream_t s) {
    const size_t smem = (size_t)T * K * (RT * sizeof(float) + 6 * sizeof(int)) + sizeof(int);
    moe_dec_down_dd_kernel<R, RT / R><<<(unsigned)((H + RT - 1) / RT), kRowWarps * IK_WARP, smem, s>>>(
        hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data, (const float*)gscales->data,
        (const int*)ids->data, (const float*)weights->data, (ik_bf16*)out->data, (int)T, (int)E, (int)H, (int)I, (int)K);
}

// The dedup down kernel's geometry (lane moedn; host side, read when the launch is recorded): INTARSIA_MOE_DOWN_DD = rows per
// block 8|16|32, 0 = the v2 kernel; INTARSIA_MOE_DOWN_DD_R = rows per warp item 2|4|8 (at most the block's rows).
bool launch_down_dd_knobs(const ik_bf16* hbuf, const ik_tensor* down_w, const ik_tensor* down_s, const ik_tensor* gscales,
                          const ik_tensor* ids, const ik_tensor* weights, ik_tensor* out, int64_t T, int64_t E, int64_t H,
                          int64_t I, int64_t K, cudaStream_t s) {
    const int rt = knob("INTARSIA_MOE_DOWN_DD", 16, 0, 8, 16, 32);
    const int r = knob("INTARSIA_MOE_DOWN_DD_R", 4, 2, 4, 8, 8);
    switch (rt * 10 + (r > rt ? rt : r)) {
#define IK_DD_CASE(rr, rt_) case rt_ * 10 + rr: launch_down_dd<rr, rt_>(hbuf, down_w, down_s, gscales, ids, weights, out, T, E, H, I, K, s); return true;
        IK_DD_CASE(2, 8) IK_DD_CASE(4, 8) IK_DD_CASE(8, 8)
        IK_DD_CASE(2, 16) IK_DD_CASE(4, 16) IK_DD_CASE(8, 16)
        IK_DD_CASE(2, 32) IK_DD_CASE(4, 32) IK_DD_CASE(8, 32)
#undef IK_DD_CASE
        default: return false;
    }
}

// Diagnostic, lane moe16 (INTARSIA_MOE_ROUTE_STATS=1; off by default, no effect on results): distinct experts per call of the
// multi-token (rows) launchers, by row count T. Reads the ids back with a stream sync, so it only runs on eager launches: a
// launch recorded into a CUDA graph is skipped (batched steps are graph-recorded unless INTARSIA_BATCH_GRAPHS=0). Prints one
// stderr line per T every 400 calls (10 steps of a 40-layer model): calls, mean / min / max distinct experts, pairs per call.
void route_stats(const ik_tensor* ids, int64_t T, int64_t K, int64_t E, cudaStream_t s) {
    static const bool on = [] { const char* v = getenv("INTARSIA_MOE_ROUTE_STATS"); return v && v[0] == '1'; }();
    if (!on || T < 1 || T > 16) return;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) return;
    std::vector<int> h((size_t)(T * K));
    if (cudaMemcpyAsync(h.data(), ids->data, h.size() * sizeof(int), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess)
        return;
    std::vector<char> seen((size_t)E, 0);
    int d = 0;
    for (int v : h)
        if (v >= 0 && v < E && !seen[(size_t)v]) seen[(size_t)v] = 1, ++d;
    static std::mutex mu;
    static long long calls[17], sum[17];
    static int mn[17], mx[17];
    std::lock_guard<std::mutex> g(mu);
    if (calls[T] == 0 || d < mn[T]) mn[T] = d;
    if (d > mx[T]) mx[T] = d;
    ++calls[T];
    sum[T] += d;
    if (calls[T] % 400 == 0)
        fprintf(stderr, "moe-route T=%lld K=%lld calls=%lld mean_distinct=%.2f min=%d max=%d pairs=%lld\n", (long long)T,
                (long long)K, calls[T], (double)sum[T] / (double)calls[T], mn[T], mx[T], (long long)(T * K));
}

// Shared by both entry points: validation, scratch, and the two launches. v2 selects the restructured down kernel.
int decode_launch(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights, const ik_tensor* gate_up_w,
                  const ik_tensor* gate_up_s, const ik_tensor* down_w, const ik_tensor* down_s, const ik_tensor* gscales,
                  ik_tensor* out, const ik_scratch* scratch, void* stream, bool v2, bool gelu, int64_t max_t = 0) {
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
    if (max_t > 0 && T > max_t) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    const int64_t P = T * K;
    if (P > 0x7fffffffLL / 2 || (size_t)(K * I * 2) > 48 * 1024 || (size_t)(H * 2) > 48 * 1024) return IK_ERR_SHAPE;
    if (!scratch || !scratch->data || scratch->bytes < r256(P * I * 2)) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    ik_bf16* hbuf = (ik_bf16*)scratch->data;
    cudaStream_t s = IK_STREAM(stream);
    if (max_t > 0) route_stats(ids, T, K, E, s);
    const int gu_unroll = v2 ? knob("INTARSIA_MOE_GU_UNROLL", 2, 1, 2, 4, 4) : 2;
    if (gelu) {
        switch (gu_unroll) {
            case 1: launch_gate_up<1, true>(x, gate_up_w, gate_up_s, gscales, ids, hbuf, P, E, H, I, K, s); break;
            case 4: launch_gate_up<4, true>(x, gate_up_w, gate_up_s, gscales, ids, hbuf, P, E, H, I, K, s); break;
            default: launch_gate_up<2, true>(x, gate_up_w, gate_up_s, gscales, ids, hbuf, P, E, H, I, K, s); break;
        }
    } else {
        switch (gu_unroll) {
            case 1: launch_gate_up<1, false>(x, gate_up_w, gate_up_s, gscales, ids, hbuf, P, E, H, I, K, s); break;
            case 4: launch_gate_up<4, false>(x, gate_up_w, gate_up_s, gscales, ids, hbuf, P, E, H, I, K, s); break;
            default: launch_gate_up<2, false>(x, gate_up_w, gate_up_s, gscales, ids, hbuf, P, E, H, I, K, s); break;
        }
    }
    IK_TRY(ik_launch_status());
    if (v2) {
        const bool tf = max_t > 0;  // the multi-token (rows) launchers
        if (tf && launch_down_dd_knobs(hbuf, down_w, down_s, gscales, ids, weights, out, T, E, H, I, K, s))
            return ik_launch_status();
        switch (knob("INTARSIA_MOE_DOWN_R", 4, 1, 2, 4, 8)) {
#define IK_DOWN_CASE(r) case r: if (tf) launch_down_v2<r, true>(hbuf, down_w, down_s, gscales, ids, weights, out, T, E, H, I, K, s); \
                                else launch_down_v2<r, false>(hbuf, down_w, down_s, gscales, ids, weights, out, T, E, H, I, K, s); break;
            IK_DOWN_CASE(1) IK_DOWN_CASE(2) IK_DOWN_CASE(8)
            default: if (tf) launch_down_v2<4, true>(hbuf, down_w, down_s, gscales, ids, weights, out, T, E, H, I, K, s);
                     else launch_down_v2<4, false>(hbuf, down_w, down_s, gscales, ids, weights, out, T, E, H, I, K, s); break;
#undef IK_DOWN_CASE
        }
        return ik_launch_status();
    }
    moe_dec_down_kernel<<<dim3((unsigned)T, (unsigned)((H + kRowWarps - 1) / kRowWarps)), kRowWarps * IK_WARP,
                          (size_t)K * I * 2, s>>>(hbuf, (const uint8_t*)down_w->data, (const uint8_t*)down_s->data,
                                                   (const float*)gscales->data, (const int*)ids->data,
                                                   (const float*)weights->data, (ik_bf16*)out->data, (int)E, (int)H,
                                                   (int)I, (int)K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_moe_experts_decode_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                               const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                               const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                               const ik_scratch* scratch, void* stream) {
    return decode_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, false, false);
}

// The same op and the same bits with the restructured down projection (moe_dec_down_v2_kernel).
int ik_moe_experts_decode_v2_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                  const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                  const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                  const ik_scratch* scratch, void* stream) {
    return decode_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true, false);
}

// gelu_tanh in the expert activation (OpAttrs::gelu): h = bf16(round_bf16(gelu_tanh(g)) * u). Otherwise the same bits.
int ik_moe_experts_decode_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                    const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                    const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                    const ik_scratch* scratch, void* stream) {
    return decode_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, false, true);
}

int ik_moe_experts_decode_v2_gelu_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                       const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                       const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                       const ik_scratch* scratch, void* stream) {
    return decode_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true, true);
}

// Row-exact multi-token twins of the v2 launchers (T <= 16): the kernels index only by pair (gate_up: pair = token * K + k)
// and by token (down), so token t of a T-token call is bitwise the T = 1 call on token t alone (tests/rows_exact.rs). Separate
// launchers so the registry can mark them `rows_exact` while the ordinary registrations keep their T <= 1 range.
int ik_moe_experts_decode_v2_rows_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                       const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                       const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                       const ik_scratch* scratch, void* stream) {
    return decode_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true, false, 16);
}

int ik_moe_experts_decode_v2_gelu_rows_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                            const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                            const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                            const ik_scratch* scratch, void* stream) {
    return decode_launch(x, ids, weights, gate_up_w, gate_up_s, down_w, down_s, gscales, out, scratch, stream, true, true, 16);
}

}  // extern "C"
