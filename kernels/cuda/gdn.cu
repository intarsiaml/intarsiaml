// Gated DeltaNet (linear attention) kernels: causal depthwise conv + SiLU, and the gated delta rule.
// Semantics: crates/intarsia-core/src/gdn.rs (HF Qwen3_5MoeGatedDeltaNet). Activations are bf16, state is f32.
//
// Design notes (sequential-over-tokens first; a chunked parallel form is future work):
//  * gdn_conv: one thread per (channel, tile of 32 tokens), history from x or the old f32 conv state [C,K-1];
//    a second tiny kernel then writes the new state in place.
//  * gdn_delta_rule: the recurrence is independent per (value head, value row j): one warp owns one row
//    S_j[dk] with dk/32 elements per lane in registers, and loops over the tokens. q/k are l2-normalised per
//    token by every warp redundantly (a warp reduction each, cheap against the state traffic). The state
//    [Hv,dv,dk] (dk contiguous) is read once and written once per launch, so decode (T=1) streams
//    2 * Hv*dv*dk*4 bytes per layer.
#include "ik_launch.h"
#include <stdlib.h>

namespace {

// ---- conv ---------------------------------------------------------------------------------
constexpr int kMaxTaps = 8;
constexpr int kConvTile = 32;  // tokens per thread: history comes from x (or the old state for the first tile)

IK_DEVI float silu(float g) { return g / (1.f + expf(-g)); }

// Grid (channels / 128, token tiles). The history of a tile is read from x (the previous K-1 inputs), or from the
// incoming state for positions before the sequence start, so tiles are independent. The new state is written by a
// second kernel (gdn_conv_state_kernel) once every tile is done reading the old one.
__global__ void gdn_conv_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w,
                                ik_bf16* __restrict__ y, const float* __restrict__ state, int64_t T, int C, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t)blockIdx.y * kConvTile;
    const int64_t t1 = min(t0 + kConvTile, T);
    float taps[kMaxTaps], h[kMaxTaps];  // h[0..K-2] = the K-1 previous inputs, oldest first
    for (int j = 0; j < K; ++j) taps[j] = ik_b2f(w[(int64_t)c * K + j]);
    for (int j = 0; j < K - 1; ++j) {
        const int64_t p = t0 - (K - 1) + j;  // position of this history entry
        h[j] = p >= 0 ? ik_b2f(x[p * C + c]) : state[(int64_t)c * (K - 1) + (K - 1) + p];
    }
    for (int64_t t = t0; t < t1; ++t) {
        const float cur = ik_b2f(x[t * C + c]);
        float acc = taps[K - 1] * cur;
        for (int j = 0; j < K - 1; ++j) acc += taps[j] * h[j];
        y[t * C + c] = ik_f2b(silu(acc));
        for (int j = 0; j < K - 2; ++j) h[j] = h[j + 1];
        h[K - 2] = cur;
    }
}

// state[c] = the last K-1 inputs of the extended sequence [old state | x]
__global__ void gdn_conv_state_kernel(const ik_bf16* __restrict__ x, float* __restrict__ state, int64_t T, int C,
                                      int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[kMaxTaps];
    for (int j = 0; j < K - 1; ++j) {
        const int64_t p = T - (K - 1) + j;
        h[j] = p >= 0 ? ik_b2f(x[p * C + c]) : state[(int64_t)c * (K - 1) + (K - 1) + p];
    }
    for (int j = 0; j < K - 1; ++j) state[(int64_t)c * (K - 1) + j] = h[j];
}

// Row-table launch (batched decode): row r (grid.y) is the T = 1 call of gdn_conv_kernel + gdn_conv_state_kernel on that
// sequence's own conv state (device address in entry [r][tindex] of the row table). One thread owns one channel of one row,
// reads its K-1 history values and writes the new ones after the output is computed (the two solo kernels do the same in
// two launches; no other thread touches this channel's history), with the same expressions: bitwise the solo pair.
__global__ void gdn_conv_rows_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w, ik_bf16* __restrict__ y,
                                     const uint8_t* __restrict__ tab, int64_t tstride, int64_t tindex, int C, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t r = blockIdx.y;
    float* state = reinterpret_cast<float*>(reinterpret_cast<const uint64_t*>(tab)[r * tstride + tindex]);
    x += r * C;
    y += r * C;
    float taps[kMaxTaps], h[kMaxTaps];
    for (int j = 0; j < K; ++j) taps[j] = ik_b2f(w[(int64_t)c * K + j]);
    for (int j = 0; j < K - 1; ++j) h[j] = state[(int64_t)c * (K - 1) + j];  // p = j - (K - 1) < 0: the old state
    const float cur = ik_b2f(x[c]);
    float acc = taps[K - 1] * cur;
    for (int j = 0; j < K - 1; ++j) acc += taps[j] * h[j];
    y[c] = ik_f2b(silu(acc));
    // the new state: the last K-1 inputs of [old state | x]
    for (int j = 0; j < K - 2; ++j) state[(int64_t)c * (K - 1) + j] = h[j + 1];
    state[(int64_t)c * (K - 1) + (K - 2)] = cur;
}

// ---- delta rule ---------------------------------------------------------------------------
constexpr int kWarpsPerBlock = 4;

IK_DEVI float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
IK_DEVI float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// Per-token inputs of one warp that do not depend on the state: normalised q (scaled) and k, v, beta, decay.
template <int PL>
struct Tok {
    float q[PL], k[PL], v, beta, decay;
};

// grid (dv / kWarpsPerBlock, Hv); warp w of block handles value row j = blockIdx.x * kWarpsPerBlock + w.
// The next token's inputs are loaded and normalised while the current token's state update runs (they do not
// depend on the state), so the dependent chain per token is just the two reductions (S^T k, S^T q).
template <int PL>  // state elements per lane: dk = 32 * PL
__global__ void gdn_delta_kernel(const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a,
                                 const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log,
                                 const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
                                 float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int DK = 32 * PL;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int j = blockIdx.x * kWarpsPerBlock + warp;
    const int h = blockIdx.y;
    if (j >= dv) return;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);

    float* srow = state + ((int64_t)h * dv + j) * DK;
    float s[PL];
#pragma unroll
    for (int i = 0; i < PL; ++i) s[i] = srow[lane + 32 * i];

    auto prepare = [&](int64_t t, Tok<PL>& tk) {
        const ik_bf16* row = qkv + t * C;
        const ik_bf16* qp = row + kh * DK;
        const ik_bf16* kp = row + Hk * DK + kh * DK;
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            tk.q[i] = ik_b2f(qp[lane + 32 * i]);
            tk.k[i] = ik_b2f(kp[lane + 32 * i]);
            qq += tk.q[i] * tk.q[i];
            kk += tk.k[i] * tk.k[i];
        }
        tk.v = ik_b2f(row[2 * Hk * DK + h * dv + j]);
        const float braw = ik_b2f(b[t * Hv + h]), araw = ik_b2f(a[t * Hv + h]);
        qq = ik_warp_sum(qq);
        kk = ik_warp_sum(kk);
        const float qr = rsqrtf(qq + eps) * scale, kr = rsqrtf(kk + eps);
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            tk.q[i] *= qr;
            tk.k[i] *= kr;
        }
        tk.beta = sigmoidf(braw);
        tk.decay = expf(neg_a * softplus(araw + dtb));
    };

    Tok<PL> cur, nxt;
    prepare(0, cur);
    for (int64_t t = 0; t < T; ++t) {
        if (t + 1 < T) prepare(t + 1, nxt);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            s[i] *= cur.decay;
            kv += s[i] * cur.k[i];
        }
        kv = ik_warp_sum(kv);
        const float delta = (cur.v - kv) * cur.beta;
        float out = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            s[i] += cur.k[i] * delta;
            out += s[i] * cur.q[i];
        }
        out = ik_warp_sum(out);
        if (lane == 0) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        cur = nxt;
    }
#pragma unroll
    for (int i = 0; i < PL; ++i) srow[lane + 32 * i] = s[i];
}

// ---- delta rule, one thread per state row (dk <= 128) -------------------------------------
// Thread j of a warp owns row S_j[dk] entirely in registers, so the recurrence needs no cross-lane reduction:
// q/k of the token are normalised once per warp, staged in shared memory and read back as broadcasts. Block = 1 warp
// = 32 value rows of one head; grid (dv / 32, Hv). The next token's raw inputs are loaded while the current
// token's update runs. Per token a thread does ~3*dk FMAs; ~10x faster than the warp-per-row kernel for prefill.
template <int DK>
__global__ void __launch_bounds__(32) gdn_delta_rows_kernel(
    const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
    const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
    float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int PL = DK / 32;
    __shared__ __align__(16) float sq[DK];
    __shared__ __align__(16) float sk[DK];
    const int lane = threadIdx.x;
    const int j = blockIdx.x * 32 + lane;
    const bool valid = j < dv;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);

    float s[DK];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) sv[i] = src[i];
    }

    float qr_[PL], kr_[PL], vraw = 0.f, araw = 0.f, braw = 0.f;
    auto load_raw = [&](int64_t t) {
        const ik_bf16* row = qkv + t * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr_[i] = ik_b2f(row[kh * DK + lane + 32 * i]);
            kr_[i] = ik_b2f(row[Hk * DK + kh * DK + lane + 32 * i]);
        }
        vraw = valid ? ik_b2f(row[2 * Hk * DK + h * dv + j]) : 0.f;
        araw = ik_b2f(a[t * Hv + h]);
        braw = ik_b2f(b[t * Hv + h]);
    };
    load_raw(0);
    for (int64_t t = 0; t < T; ++t) {
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr_[i] * qr_[i];
            kk += kr_[i] * kr_[i];
        }
        qq = ik_warp_sum(qq);
        kk = ik_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
        __syncwarp();  // the previous token's reads of sq/sk are done
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            sq[lane + 32 * i] = qr_[i] * qn;
            sk[lane + 32 * i] = kr_[i] * kn;
        }
        __syncwarp();
        const float v = vraw, beta = sigmoidf(braw), decay = expf(neg_a * softplus(araw + dtb));
        if (t + 1 < T) load_raw(t + 1);  // in flight during the update below
        float kv = 0.f;
        const float4* k4 = reinterpret_cast<const float4*>(sk);
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) {
            const float4 kk4 = k4[i];
            sv[i].x *= decay; sv[i].y *= decay; sv[i].z *= decay; sv[i].w *= decay;
            kv += sv[i].x * kk4.x + sv[i].y * kk4.y + sv[i].z * kk4.z + sv[i].w * kk4.w;
        }
        const float delta = (v - kv) * beta;
        float out = 0.f;
        const float4* q4 = reinterpret_cast<const float4*>(sq);
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) {
            const float4 kk4 = k4[i], qq4 = q4[i];
            sv[i].x += kk4.x * delta; sv[i].y += kk4.y * delta; sv[i].z += kk4.z * delta; sv[i].w += kk4.w * delta;
            out += sv[i].x * qq4.x + sv[i].y * qq4.y + sv[i].z * qq4.z + sv[i].w * qq4.w;
        }
        if (valid) o[(t * Hv + h) * dv + j] = ik_f2b(out);
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) dst[i] = sv[i];
    }
}

// ---- delta rule, LPR lanes per state row ---------------------------------------------------
// The one-thread-per-row kernel above occupies 4096 threads for the 35B (32 value heads x 128 rows), about 5 % of the
// GPU, and each token walks two dependent chains of dk FMAs: it is latency bound (~1.2 us per token and layer on GB10,
// docs/results/2026-10-13-prefill-cuda.md). Here LPR lanes share a row: lane l owns the float4 groups l, l + LPR, ...
// of S_j[dk] (so a warp's loads of one row are contiguous and the shared-memory reads of q/k are conflict-free
// broadcasts), keeps four independent partial sums per dot product, and the LPR partial sums are combined with
// shuffles. LPR = 4 gives 4x the threads and ~3x shorter chains. The sum order differs from gdn_delta_rows_kernel
// (fp32 rounding only); the order is fixed, so results are run-to-run identical.
template <int DK, int LPR>
__global__ void __launch_bounds__(32) gdn_delta_lanes_kernel(
    const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
    const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
    float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int PL = DK / 32;          // raw q/k elements per lane for the cooperative load + l2 norm
    constexpr int NG = DK / (4 * LPR);   // float4 groups of the state row per lane
    constexpr int RPW = 32 / LPR;        // state rows per warp
    __shared__ __align__(16) float sq[DK];
    __shared__ __align__(16) float sk[DK];
    const int lane = threadIdx.x;
    const int l = lane % LPR, r = lane / LPR;
    const int j = blockIdx.x * RPW + r;
    const bool valid = j < dv;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);

    float4 s[NG];
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) s[i] = src[i * LPR + l];
    } else {
#pragma unroll
        for (int i = 0; i < NG; ++i) s[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    float qr_[PL], kr_[PL], vraw = 0.f, araw = 0.f, braw = 0.f;
    auto load_raw = [&](int64_t t) {
        const ik_bf16* row = qkv + t * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr_[i] = ik_b2f(row[kh * DK + lane + 32 * i]);
            kr_[i] = ik_b2f(row[Hk * DK + kh * DK + lane + 32 * i]);
        }
        vraw = valid ? ik_b2f(row[2 * Hk * DK + h * dv + j]) : 0.f;
        araw = ik_b2f(a[t * Hv + h]);
        braw = ik_b2f(b[t * Hv + h]);
    };
    load_raw(0);
    const float4* sk4 = reinterpret_cast<const float4*>(sk);
    const float4* sq4 = reinterpret_cast<const float4*>(sq);
    for (int64_t t = 0; t < T; ++t) {
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr_[i] * qr_[i];
            kk += kr_[i] * kr_[i];
        }
        qq = ik_warp_sum(qq);
        kk = ik_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
        __syncwarp();  // the previous token's reads of sq/sk are done
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            sq[lane + 32 * i] = qr_[i] * qn;
            sk[lane + 32 * i] = kr_[i] * kn;
        }
        __syncwarp();
        const float v = vraw, beta = sigmoidf(braw), decay = expf(neg_a * softplus(araw + dtb));
        if (t + 1 < T) load_raw(t + 1);  // in flight during the update below
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
        for (int i = 0; i < NG; ++i) {
            const float4 k4 = sk4[i * LPR + l];
            s[i].x *= decay; s[i].y *= decay; s[i].z *= decay; s[i].w *= decay;
            p0 += s[i].x * k4.x; p1 += s[i].y * k4.y; p2 += s[i].z * k4.z; p3 += s[i].w * k4.w;
        }
        float kv = (p0 + p1) + (p2 + p3);
#pragma unroll
        for (int off = 1; off < LPR; off <<= 1) kv += __shfl_xor_sync(0xffffffffu, kv, off);
        const float delta = (v - kv) * beta;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
        for (int i = 0; i < NG; ++i) {
            const float4 k4 = sk4[i * LPR + l], q4 = sq4[i * LPR + l];
            s[i].x += k4.x * delta; s[i].y += k4.y * delta; s[i].z += k4.z * delta; s[i].w += k4.w * delta;
            o0 += s[i].x * q4.x; o1 += s[i].y * q4.y; o2 += s[i].z * q4.z; o3 += s[i].w * q4.w;
        }
        float out = (o0 + o1) + (o2 + o3);
#pragma unroll
        for (int off = 1; off < LPR; off <<= 1) out += __shfl_xor_sync(0xffffffffu, out, off);
        if (valid && l == 0) o[(t * Hv + h) * dv + j] = ik_f2b(out);
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) dst[i * LPR + l] = s[i];
    }
}

// ---- delta rule, the rows kernel's arithmetic split over LPR lanes per state row ------------
// BITWISE the same as gdn_delta_rows_kernel: the same fp32 operations in the same order, written with explicit
// round-to-nearest intrinsics so nvcc cannot contract them differently. The order is the one nvcc 13.0 -O3 emits for
// the rows kernel (cuobjdump -sass, docs/results/2026-10-19-cuda-prefill-4.md):
//   * decay: s = s * decay (FMUL), element-wise;
//   * a float4 group's dot product s.x*k.x + s.y*k.y + s.z*k.z + s.w*k.w is FMUL(y), then FFMA x, z, w;
//   * kv (and out) = ((0 + g0) + g1) + ... + g_{DK/4-1}, the group sums added left to right (FADD chain);
//   * delta = (v - kv) * beta; the update s = fma(k, delta, s); the output dot product as for kv, on the updated s.
// The element-wise work (decay, update, group sums: 3/4 of the rows kernel's instructions) is split over LPR lanes;
// lane l of a row owns the CONTIGUOUS groups [l*NG, (l+1)*NG), so the left-to-right sum is a carry handed lane to lane
// (LPR stages of NG adds and one shuffle). q/k are staged in shared memory with one pad float4 after every NG groups
// (the LPR lanes of a row then read distinct banks), double-buffered: the next token's l2 norm (two warp reductions)
// is computed during the current token's update instead of before it. Different summation trees (the lanes kernel
// above) change the 35B's greedy tokens and KL (docs/results/2026-10-19-cuda-prefill-3.md); this kernel does not.
//
// ROWS (batched decode, row-table launch): grid.z is the sequence. Row r is the T = 1 call on that sequence's own inputs
// (qkv, a, b rows r, output row r) and its own state, the buffer whose device address is entry [r][tindex] of the row table
// `tab` plus `toff` bytes. Everything after those pointers is the same code, so row r is bitwise the one-row launch.
template <int DK, int LPR, bool ROWS = false>
__global__ void __launch_bounds__(32) gdn_delta_split_kernel(
    const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
    const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
    float* __restrict__ state, int64_t T, int Hk, int Hv, int dv, const uint8_t* tab = nullptr, int64_t tstride = 0,
    int64_t tindex = 0, int64_t toff = 0) {
    constexpr int PL = DK / 32;            // raw q/k elements per lane for the cooperative load + l2 norm
    constexpr int NG = DK / (4 * LPR);     // float4 groups of the state row per lane (contiguous)
    constexpr int RPW = 32 / LPR;          // state rows per warp
    constexpr int SW = 4 * (DK / 4 + LPR); // floats per staged vector: one pad float4 after every NG groups
    __shared__ __align__(16) float sq[2][SW];
    __shared__ __align__(16) float sk[2][SW];
    const int lane = threadIdx.x;
    const int l = lane % LPR, r = lane / LPR;
    const int row0 = r * LPR;              // first lane of this row
    const int j = blockIdx.x * RPW + r;
    const bool valid = j < dv;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    if constexpr (ROWS) {
        const int64_t rr = blockIdx.z;
        qkv += rr * C;
        a += rr * Hv;
        b += rr * Hv;
        o += rr * Hv * dv;
        state = reinterpret_cast<float*>(reinterpret_cast<const uint64_t*>(tab)[rr * tstride + tindex] + (uint64_t)toff);
    }
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);

    // The warp's RPW state rows are contiguous in memory: they are loaded (and stored at the end) coalesced, float4 q of
    // the block by lane q % 32, through a padded shared-memory tile (one pad float4 after every NG groups and per row, so
    // the lanes that read a row's contiguous groups hit distinct banks).
    __shared__ __align__(16) float4 sst[RPW * (DK / 4 + LPR)];
    float4* const blk = reinterpret_cast<float4*>(state + ((int64_t)h * dv + (int64_t)blockIdx.x * RPW) * DK);
    const int nrows = min(RPW, dv - (int)blockIdx.x * RPW);
    auto tile_at = [&](int q) {  // the tile slot of float4 q of the block (row q / (DK/4), group q % (DK/4))
        const int rr = q / (DK / 4), gi = q % (DK / 4);
        return rr * (DK / 4 + LPR) + (gi / NG) * (NG + 1) + gi % NG;
    };
#pragma unroll
    for (int m = 0; m < NG; ++m) {
        const int q = lane + 32 * m;
        sst[tile_at(q)] = q / (DK / 4) < nrows ? blk[q] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncwarp();
    float4 s[NG];
    const int mine = r * (DK / 4 + LPR) + l * (NG + 1);
#pragma unroll
    for (int i = 0; i < NG; ++i) s[i] = sst[mine + i];

    float qr_[PL], kr_[PL], vraw = 0.f, araw = 0.f, braw = 0.f;
    auto load_raw = [&](int64_t t) {
        const ik_bf16* row = qkv + t * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr_[i] = ik_b2f(row[kh * DK + lane + 32 * i]);
            kr_[i] = ik_b2f(row[Hk * DK + kh * DK + lane + 32 * i]);
        }
        vraw = valid ? ik_b2f(row[2 * Hk * DK + h * dv + j]) : 0.f;
        araw = ik_b2f(a[t * Hv + h]);
        braw = ik_b2f(b[t * Hv + h]);
    };
    // normalise the loaded raw token into buffer `buf` and its scalars (the rows kernel's code, unchanged)
    float nv = 0.f, nbeta = 0.f, ndecay = 0.f;
    auto prep = [&](int buf) {
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr_[i] * qr_[i];
            kk += kr_[i] * kr_[i];
        }
        qq = ik_warp_sum(qq);
        kk = ik_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            const int e = lane + 32 * i, g = e >> 2;
            const int at = 4 * (g + g / NG) + (e & 3);
            sq[buf][at] = qr_[i] * qn;
            sk[buf][at] = kr_[i] * kn;
        }
        nv = vraw;
        nbeta = sigmoidf(braw);
        ndecay = expf(neg_a * softplus(araw + dtb));
    };
    load_raw(0);
    prep(0);
    load_raw(T > 1 ? 1 : 0);
    const int first = l * (NG + 1);        // this lane's first staged float4
    for (int64_t t = 0; t < T; ++t) {
        const int cur = (int)(t & 1);
        __syncwarp();  // buffer `cur` (written last token) is visible; the reads of buffer cur ^ 1 are done
        const float v = nv, beta = nbeta, decay = ndecay;
        // the next token's norm, off the critical path; on the last token it redoes the last token (never read)
        prep(cur ^ 1);
        load_raw(t + 2 < T ? t + 2 : T - 1);
        const float4* k4 = reinterpret_cast<const float4*>(sk[cur]) + first;
        const float4* q4 = reinterpret_cast<const float4*>(sq[cur]) + first;
        float g[NG];
#pragma unroll
        for (int i = 0; i < NG; ++i) {
            const float4 kk4 = k4[i];
            s[i].x = __fmul_rn(s[i].x, decay); s[i].y = __fmul_rn(s[i].y, decay);
            s[i].z = __fmul_rn(s[i].z, decay); s[i].w = __fmul_rn(s[i].w, decay);
            g[i] = __fmaf_rn(s[i].w, kk4.w, __fmaf_rn(s[i].z, kk4.z, __fmaf_rn(s[i].x, kk4.x, __fmul_rn(s[i].y, kk4.y))));
        }
        float kv = 0.f;
#pragma unroll
        for (int st = 0; st < LPR; ++st) {
            float c = kv;
#pragma unroll
            for (int i = 0; i < NG; ++i) c = __fadd_rn(c, g[i]);
            kv = __shfl_sync(0xffffffffu, c, row0 + st);
        }
        const float delta = __fmul_rn(__fsub_rn(v, kv), beta);
#pragma unroll
        for (int i = 0; i < NG; ++i) {
            const float4 kk4 = k4[i], qq4 = q4[i];
            s[i].x = __fmaf_rn(kk4.x, delta, s[i].x); s[i].y = __fmaf_rn(kk4.y, delta, s[i].y);
            s[i].z = __fmaf_rn(kk4.z, delta, s[i].z); s[i].w = __fmaf_rn(kk4.w, delta, s[i].w);
            g[i] = __fmaf_rn(s[i].w, qq4.w, __fmaf_rn(s[i].z, qq4.z, __fmaf_rn(s[i].x, qq4.x, __fmul_rn(s[i].y, qq4.y))));
        }
        float out = 0.f;
#pragma unroll
        for (int st = 0; st < LPR; ++st) {
            float c = out;
#pragma unroll
            for (int i = 0; i < NG; ++i) c = __fadd_rn(c, g[i]);
            out = st + 1 < LPR ? __shfl_sync(0xffffffffu, c, row0 + st) : c;  // the last stage: lane LPR - 1 holds it
        }
        if (valid && l == LPR - 1) o[(t * Hv + h) * dv + j] = ik_f2b(out);
    }
    __syncwarp();  // every lane's initial tile reads are long done; make the ordering explicit before reusing the tile
#pragma unroll
    for (int i = 0; i < NG; ++i) sst[mine + i] = s[i];
    __syncwarp();
#pragma unroll
    for (int m = 0; m < NG; ++m) {
        const int q = lane + 32 * m;
        if (q / (DK / 4) < nrows) blk[q] = sst[tile_at(q)];
    }
}

// ---- delta rule, decode (T = 1): the split kernel's arithmetic with the row sums gathered through shared memory --------
// BITWISE gdn_delta_split_kernel (and so gdn_delta_rows_kernel) for one token, dk = 128. The warp owns RPW state rows; lane L
// holds float4 group L of every one of them, so the state comes in (and goes out) as RPW fully coalesced 512-byte rows, straight
// into registers, with no staging tile. The q / k vectors are the same for all rows of a head: lane L normalises its own group
// (the same products raw * norm as the split kernel's staging), after the same lane-strided sums of squares and warp tree.
// Per row, the group dot products g_0 .. g_31 (the same FMUL / FFMA chain) go to shared memory and lane r adds row r's left to
// right from 0 (the same FADD chain the split kernel hands from lane to lane), then delta and, after the update, the output the
// same way. Only the memory path and which lane does which add change.
//
// WPB warps per block, each with its own RPW rows (the q / k norm is redone per warp, as in the split kernel).
// ROWS (batched decode, row-table launch): as in gdn_delta_split_kernel, grid.z is the sequence.
template <int DK, int RPW, int WPB, bool ROWS>
__global__ void __launch_bounds__(32 * WPB) gdn_delta_dec_kernel(
    const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
    const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
    float* __restrict__ state, int Hk, int Hv, int dv, const uint8_t* tab = nullptr, int64_t tstride = 0,
    int64_t tindex = 0, int64_t toff = 0) {
    static_assert(DK == 128, "one float4 group per lane");
    static_assert(RPW <= 32, "a lane sums a row");
    constexpr int PL = DK / 32;
    const int lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    const int j0 = (blockIdx.x * WPB + wp) * RPW;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    if constexpr (ROWS) {
        const int64_t rr = blockIdx.z;
        qkv += rr * C;
        a += rr * Hv;
        b += rr * Hv;
        o += rr * Hv * dv;
        state = reinterpret_cast<float*>(reinterpret_cast<const uint64_t*>(tab)[rr * tstride + tindex] + (uint64_t)toff);
    }
    const int nrows = min(RPW, dv - j0);
    if (nrows <= 0) return;  // a whole warp past dv (warp-level synchronisation only)
    float4* const blk = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j0) * DK);
    float4 s[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) s[r] = r < nrows ? blk[r * (DK / 4) + lane] : make_float4(0.f, 0.f, 0.f, 0.f);

    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    // the split kernel's load_raw + prep for t = 0
    float qr_[PL], kr_[PL];
#pragma unroll
    for (int i = 0; i < PL; ++i) {
        qr_[i] = ik_b2f(qkv[kh * DK + lane + 32 * i]);
        kr_[i] = ik_b2f(qkv[Hk * DK + kh * DK + lane + 32 * i]);
    }
    float qq = 0.f, kk = 0.f;
#pragma unroll
    for (int i = 0; i < PL; ++i) {
        qq += qr_[i] * qr_[i];
        kk += kr_[i] * kr_[i];
    }
    qq = ik_warp_sum(qq);
    kk = ik_warp_sum(kk);
    const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
    float kq[4], qv[4];  // this lane's group: elements 4 * lane .. 4 * lane + 3
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        qv[e] = ik_b2f(qkv[kh * DK + 4 * lane + e]) * qn;
        kq[e] = ik_b2f(qkv[Hk * DK + kh * DK + 4 * lane + e]) * kn;
    }
    const float4 k4 = make_float4(kq[0], kq[1], kq[2], kq[3]), q4 = make_float4(qv[0], qv[1], qv[2], qv[3]);
    const float v = lane < nrows ? ik_b2f(qkv[2 * Hk * DK + h * dv + j0 + lane]) : 0.f;  // lane r: row r's v
    const float beta = sigmoidf(ik_b2f(b[h]));
    const float decay = expf(neg_a * softplus(ik_b2f(a[h]) + dtb));

    __shared__ float gs_[WPB][RPW][33];  // [warp][row][group], padded: lane r reads row r without bank conflicts
    __shared__ float ds_[WPB][RPW];
    float (*gs)[33] = gs_[wp];
    float* ds = ds_[wp];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        s[r].x = __fmul_rn(s[r].x, decay); s[r].y = __fmul_rn(s[r].y, decay);
        s[r].z = __fmul_rn(s[r].z, decay); s[r].w = __fmul_rn(s[r].w, decay);
        gs[r][lane] = __fmaf_rn(s[r].w, k4.w, __fmaf_rn(s[r].z, k4.z, __fmaf_rn(s[r].x, k4.x, __fmul_rn(s[r].y, k4.y))));
    }
    __syncwarp();
    if (lane < RPW) {
        float c = 0.f;
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) c = __fadd_rn(c, gs[lane][i]);
        ds[lane] = __fmul_rn(__fsub_rn(v, c), beta);
    }
    __syncwarp();  // ds is visible; the reads of gs are done
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const float delta = ds[r];
        s[r].x = __fmaf_rn(k4.x, delta, s[r].x); s[r].y = __fmaf_rn(k4.y, delta, s[r].y);
        s[r].z = __fmaf_rn(k4.z, delta, s[r].z); s[r].w = __fmaf_rn(k4.w, delta, s[r].w);
        gs[r][lane] = __fmaf_rn(s[r].w, q4.w, __fmaf_rn(s[r].z, q4.z, __fmaf_rn(s[r].x, q4.x, __fmul_rn(s[r].y, q4.y))));
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
        if (r < nrows) blk[r * (DK / 4) + lane] = s[r];
    __syncwarp();
    if (lane < nrows) {
        float c = 0.f;
#pragma unroll
        for (int i = 0; i < DK / 4; ++i) c = __fadd_rn(c, gs[lane][i]);
        o[(int64_t)h * dv + j0 + lane] = ik_f2b(c);
    }
}

// INTARSIA_GDN_DEC = 0 keeps the split kernel for T = 1 (A/B); 1 | 2 | 4 | 8 | 16 picks the rows per warp (default 1), and
// INTARSIA_GDN_DEC_WPB = 1 | 2 | 4 the warps per block (default 2 from 12 rows, 1 below: evicted micro-timing, one row per
// warp, 35B: 16 rows 296 -> 283 us, 8 rows 152 -> 144, 1 row 23.3 -> 20.8). The bits depend on neither.
inline int gdn_env(const char* name, int dflt) {
    const char* e = getenv(name);
    return e ? atoi(e) : dflt;
}
inline int gdn_dec_rpw() {
    static const int v = [] {
        const int n = gdn_env("INTARSIA_GDN_DEC", 1);
        return (n == 0 || n == 1 || n == 2 || n == 4 || n == 8 || n == 16) ? n : 1;
    }();
    return v;
}
inline int gdn_dec_wpb() {
    static const int v = [] {
        const int n = gdn_env("INTARSIA_GDN_DEC_WPB", 0);
        return (n == 1 || n == 2 || n == 4) ? n : 0;  // 0: by the row count
    }();
    return v;
}

template <int RPW, int WPB, bool ROWS>
void gdn_dec_launch(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log, const ik_tensor* dt_bias,
                    float eps, ik_tensor* o, float* state, int Hk, int Hv, int dv, int R, const uint8_t* tab, int64_t stride,
                    int64_t index, int64_t byte_off, cudaStream_t s) {
    const dim3 grid((unsigned)((dv + RPW * WPB - 1) / (RPW * WPB)), (unsigned)Hv, (unsigned)R);
    gdn_delta_dec_kernel<128, RPW, WPB, ROWS><<<grid, 32 * WPB, 0, s>>>(
        (const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data, (const ik_bf16*)a_log->data,
        (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data, state, Hk, Hv, dv, tab, stride, index, byte_off);
}

template <int RPW, bool ROWS>
void gdn_dec_wpb_launch(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                        const ik_tensor* dt_bias, float eps, ik_tensor* o, float* state, int Hk, int Hv, int dv, int R,
                        const uint8_t* tab, int64_t stride, int64_t index, int64_t byte_off, cudaStream_t s) {
    switch (gdn_dec_wpb() ? gdn_dec_wpb() : R >= 12 ? 2 : 1) {
        case 2: gdn_dec_launch<RPW, 2, ROWS>(qkv, a, b, a_log, dt_bias, eps, o, state, Hk, Hv, dv, R, tab, stride, index, byte_off, s); break;
        case 4: gdn_dec_launch<RPW, 4, ROWS>(qkv, a, b, a_log, dt_bias, eps, o, state, Hk, Hv, dv, R, tab, stride, index, byte_off, s); break;
        default: gdn_dec_launch<RPW, 1, ROWS>(qkv, a, b, a_log, dt_bias, eps, o, state, Hk, Hv, dv, R, tab, stride, index, byte_off, s);
    }
}

// the decode kernel for T = 1, dk = 128 (ROWS: R rows through the row table)
template <bool ROWS>
void gdn_dec_run(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log, const ik_tensor* dt_bias,
                 float eps, ik_tensor* o, float* state, int Hk, int Hv, int dv, int R, const uint8_t* tab, int64_t stride,
                 int64_t index, int64_t byte_off, cudaStream_t s) {
#define IK_GDN_DEC(RPW) gdn_dec_wpb_launch<RPW, ROWS>(qkv, a, b, a_log, dt_bias, eps, o, state, Hk, Hv, dv, R, tab, stride, index, byte_off, s)
    switch (gdn_dec_rpw()) {
        case 2: IK_GDN_DEC(2); break;
        case 4: IK_GDN_DEC(4); break;
        case 8: IK_GDN_DEC(8); break;
        case 16: IK_GDN_DEC(16); break;
        default: IK_GDN_DEC(1);
    }
#undef IK_GDN_DEC
}

}  // namespace

extern "C" {

int ik_gdn_conv_bf16(const ik_tensor* x, const ik_tensor* taps, ik_tensor* y, ik_tensor* state, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(taps, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(y, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(state, IK_DTYPE_F32, 2));
    const int64_t T = x->shape[0], C = x->shape[1], K = taps->shape[1];
    if (K < 2 || K > kMaxTaps || taps->shape[0] != C || y->shape[0] != T || y->shape[1] != C ||
        state->shape[0] != C || state->shape[1] != K - 1)
        return IK_ERR_SHAPE;
    if (C == 0 || T == 0) return IK_OK;
    const int threads = 128;
    cudaStream_t s = IK_STREAM(stream);
    dim3 grid((unsigned)((C + threads - 1) / threads), (unsigned)((T + kConvTile - 1) / kConvTile));
    gdn_conv_kernel<<<grid, threads, 0, s>>>((const ik_bf16*)x->data, (const ik_bf16*)taps->data,
                                             (ik_bf16*)y->data, (const float*)state->data, T, (int)C, (int)K);
    IK_TRY(ik_launch_status());
    gdn_conv_state_kernel<<<(unsigned)((C + threads - 1) / threads), threads, 0, s>>>(
        (const ik_bf16*)x->data, (float*)state->data, T, (int)C, (int)K);
    return ik_launch_status();
}

int ik_gdn_delta_rule_bf16(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                           const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream) {
    IK_TRY(ik_check(qkv, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a_log, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(dt_bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(o, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(state, IK_DTYPE_F32, 3));
    const int64_t T = qkv->shape[0], C = qkv->shape[1];
    const int64_t Hv = o->shape[1], dv = o->shape[2], dk = state->shape[2];
    if (dk == 0 || dv == 0 || Hv == 0 || C <= Hv * dv || (C - Hv * dv) % (2 * dk) != 0) return IK_ERR_SHAPE;
    const int64_t Hk = (C - Hv * dv) / (2 * dk);
    if (Hv % Hk != 0 || o->shape[0] != T || state->shape[0] != Hv || state->shape[1] != dv ||
        a->shape[0] != T || a->shape[1] != Hv || b->shape[0] != T || b->shape[1] != Hv || a_log->shape[0] != Hv ||
        dt_bias->shape[0] != Hv)
        return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    if (dk <= 128) {
        const dim3 rgrid((unsigned)((dv + 31) / 32), (unsigned)Hv);
#define IK_GDN_ROWS(DK)                                                                                          \
    gdn_delta_rows_kernel<DK><<<rgrid, 32, 0, s>>>(                                                              \
        (const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data,                             \
        (const ik_bf16*)a_log->data, (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data, (float*)state->data, \
        T, (int)Hk, (int)Hv, (int)dv)
        switch (dk) {
            case 32: IK_GDN_ROWS(32); break;
            case 64: IK_GDN_ROWS(64); break;
            case 128: IK_GDN_ROWS(128); break;
            default: return IK_ERR_UNSUPPORTED;
        }
#undef IK_GDN_ROWS
        return ik_launch_status();
    }
    // dk = 256 does not fit one state row in registers: one warp per row
    if (dk != 256) return IK_ERR_UNSUPPORTED;
    dim3 grid((unsigned)((dv + kWarpsPerBlock - 1) / kWarpsPerBlock), (unsigned)Hv);
    const dim3 block(32 * kWarpsPerBlock);
    gdn_delta_kernel<8><<<grid, block, 0, s>>>(
        (const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data, (const ik_bf16*)a_log->data,
        (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data, (float*)state->data, T, (int)Hk, (int)Hv, (int)dv);
    return ik_launch_status();
}

// Same contract as ik_gdn_delta_rule_bf16; dk in {64, 128}. INTARSIA_GDN_LPR=2|4|8 picks the lanes per row (A/B).
int ik_gdn_delta_rule_bf16_lanes(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                                 const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream) {
    IK_TRY(ik_check(qkv, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a_log, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(dt_bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(o, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(state, IK_DTYPE_F32, 3));
    const int64_t T = qkv->shape[0], C = qkv->shape[1];
    const int64_t Hv = o->shape[1], dv = o->shape[2], dk = state->shape[2];
    if (dk == 0 || dv == 0 || Hv == 0 || C <= Hv * dv || (C - Hv * dv) % (2 * dk) != 0) return IK_ERR_SHAPE;
    const int64_t Hk = (C - Hv * dv) / (2 * dk);
    if (Hv % Hk != 0 || o->shape[0] != T || state->shape[0] != Hv || state->shape[1] != dv ||
        a->shape[0] != T || a->shape[1] != Hv || b->shape[0] != T || b->shape[1] != Hv || a_log->shape[0] != Hv ||
        dt_bias->shape[0] != Hv)
        return IK_ERR_SHAPE;
    if (dk != 64 && dk != 128) return IK_ERR_UNSUPPORTED;
    if (!ik_aligned16(state->data)) return IK_ERR_INVALID_ARG;
    if (T == 0) return IK_OK;
    static const int lpr_env = [] {
        const char* v = getenv("INTARSIA_GDN_LPR");
        const int n = v ? atoi(v) : 0;
        return (n == 2 || n == 4 || n == 8) ? n : 4;
    }();
    cudaStream_t s = IK_STREAM(stream);
#define IK_GDN_LANES(DK, LPR)                                                                                        \
    gdn_delta_lanes_kernel<DK, LPR><<<dim3((unsigned)((dv + (32 / LPR) - 1) / (32 / LPR)), (unsigned)Hv), 32, 0, s>>>( \
        (const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data, (const ik_bf16*)a_log->data,    \
        (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data, (float*)state->data, T, (int)Hk, (int)Hv, (int)dv)
    if (dk == 128) {
        if (lpr_env == 2) IK_GDN_LANES(128, 2);
        else if (lpr_env == 8) IK_GDN_LANES(128, 8);
        else IK_GDN_LANES(128, 4);
    } else {
        if (lpr_env == 2) IK_GDN_LANES(64, 2);
        else if (lpr_env == 8) IK_GDN_LANES(64, 8);
        else IK_GDN_LANES(64, 4);
    }
#undef IK_GDN_LANES
    return ik_launch_status();
}

// Same contract as ik_gdn_delta_rule_bf16 and bitwise the same results; dk in {64, 128}. INTARSIA_GDN_SPLIT_LPR=4|8
// picks other lanes per row than the default 2 (A/B only: the bits do not depend on it; 2 was fastest at T = 1..2048,
// docs/results/2026-10-19-cuda-prefill-4.md).
int ik_gdn_delta_rule_bf16_split(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                                 const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream) {
    IK_TRY(ik_check(qkv, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a_log, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(dt_bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(o, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(state, IK_DTYPE_F32, 3));
    const int64_t T = qkv->shape[0], C = qkv->shape[1];
    const int64_t Hv = o->shape[1], dv = o->shape[2], dk = state->shape[2];
    if (dk == 0 || dv == 0 || Hv == 0 || C <= Hv * dv || (C - Hv * dv) % (2 * dk) != 0) return IK_ERR_SHAPE;
    const int64_t Hk = (C - Hv * dv) / (2 * dk);
    if (Hv % Hk != 0 || o->shape[0] != T || state->shape[0] != Hv || state->shape[1] != dv ||
        a->shape[0] != T || a->shape[1] != Hv || b->shape[0] != T || b->shape[1] != Hv || a_log->shape[0] != Hv ||
        dt_bias->shape[0] != Hv)
        return IK_ERR_SHAPE;
    if (dk != 64 && dk != 128) return IK_ERR_UNSUPPORTED;
    if (!ik_aligned16(state->data)) return IK_ERR_INVALID_ARG;
    if (T == 0) return IK_OK;
    static const int lpr_env = [] {
        const char* v = getenv("INTARSIA_GDN_SPLIT_LPR");
        const int n = v ? atoi(v) : 0;
        return (n == 2 || n == 4 || n == 8) ? n : 2;
    }();
    cudaStream_t s = IK_STREAM(stream);
    if (T == 1 && dk == 128 && gdn_dec_rpw() > 0) {  // decode: the same bits from the decode kernel
        gdn_dec_run<false>(qkv, a, b, a_log, dt_bias, eps, o, (float*)state->data, (int)Hk, (int)Hv, (int)dv, 1, nullptr, 0, 0, 0, s);
        return ik_launch_status();
    }
#define IK_GDN_SPLIT(DK, LPR)                                                                                        \
    gdn_delta_split_kernel<DK, LPR><<<dim3((unsigned)((dv + (32 / LPR) - 1) / (32 / LPR)), (unsigned)Hv), 32, 0, s>>>( \
        (const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data, (const ik_bf16*)a_log->data,    \
        (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data, (float*)state->data, T, (int)Hk, (int)Hv, (int)dv)
    if (dk == 128) {
        if (lpr_env == 4) IK_GDN_SPLIT(128, 4);
        else if (lpr_env == 8) IK_GDN_SPLIT(128, 8);
        else IK_GDN_SPLIT(128, 2);
    } else {
        if (lpr_env == 4) IK_GDN_SPLIT(64, 4);
        else if (lpr_env == 8) IK_GDN_SPLIT(64, 8);
        else IK_GDN_SPLIT(64, 2);
    }
#undef IK_GDN_SPLIT
    return ik_launch_status();
}

// ---- row-table launches (batched decode, `OpCall::rows`) ------------------------------------------------------------------
// `tab`: u8 [rows, stride, 8], little-endian device addresses; the state tensor argument is row 0's buffer (shape carrier).
static int ik_check_tab(const ik_tensor* tab, int64_t rows, int64_t stride, int64_t index) {
    if (!tab || !tab->data || tab->dtype != IK_DTYPE_U8 || tab->ndim != 3 || tab->shape[0] != rows ||
        tab->shape[1] != stride || tab->shape[2] != 8 || index < 0 || index >= stride)
        return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(tab->data) & 7u) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

int ik_gdn_conv_bf16_rows(const ik_tensor* x, const ik_tensor* taps, ik_tensor* y, const ik_tensor* state0,
                          const ik_tensor* tab, int64_t stride, int64_t index, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(taps, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(y, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(state0, IK_DTYPE_F32, 2));
    const int64_t R = x->shape[0], C = x->shape[1], K = taps->shape[1];
    if (K < 2 || K > kMaxTaps || taps->shape[0] != C || y->shape[0] != R || y->shape[1] != C ||
        state0->shape[0] != C || state0->shape[1] != K - 1)
        return IK_ERR_SHAPE;
    IK_TRY(ik_check_tab(tab, R, stride, index));
    if (C == 0 || R == 0) return IK_OK;
    if (R > 65535) return IK_ERR_SHAPE;
    const int threads = 128;
    gdn_conv_rows_kernel<<<dim3((unsigned)((C + threads - 1) / threads), (unsigned)R), threads, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)taps->data, (ik_bf16*)y->data, (const uint8_t*)tab->data, stride, index,
        (int)C, (int)K);
    return ik_launch_status();
}

// The T = 1 launch of ik_gdn_delta_rule_bf16_split on every row at once; row r reads and writes the state at entry
// [r][index] of the table plus `byte_off` (the delta state sits after the conv state in a layer's recurrent buffer).
// `state0` (row 0's [Hv, dv, dk] view) carries the shape. Same lanes-per-row choice as the solo launcher.
int ik_gdn_delta_rule_bf16_split_rows(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                                      const ik_tensor* dt_bias, float eps, ik_tensor* o, const ik_tensor* state0,
                                      const ik_tensor* tab, int64_t stride, int64_t index, int64_t byte_off, void* stream) {
    IK_TRY(ik_check(qkv, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a_log, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(dt_bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(o, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(state0, IK_DTYPE_F32, 3));
    const int64_t R = qkv->shape[0], C = qkv->shape[1];
    const int64_t Hv = o->shape[1], dv = o->shape[2], dk = state0->shape[2];
    if (dk == 0 || dv == 0 || Hv == 0 || C <= Hv * dv || (C - Hv * dv) % (2 * dk) != 0) return IK_ERR_SHAPE;
    const int64_t Hk = (C - Hv * dv) / (2 * dk);
    if (Hv % Hk != 0 || o->shape[0] != R || state0->shape[0] != Hv || state0->shape[1] != dv || a->shape[0] != R ||
        a->shape[1] != Hv || b->shape[0] != R || b->shape[1] != Hv || a_log->shape[0] != Hv || dt_bias->shape[0] != Hv)
        return IK_ERR_SHAPE;
    if (dk != 64 && dk != 128) return IK_ERR_UNSUPPORTED;
    if (byte_off < 0 || (byte_off & 15) != 0) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_tab(tab, R, stride, index));
    if (R == 0) return IK_OK;
    if (R > 65535) return IK_ERR_SHAPE;
    static const int lpr_env = [] {
        const char* v = getenv("INTARSIA_GDN_SPLIT_LPR");
        const int n = v ? atoi(v) : 0;
        return (n == 2 || n == 4 || n == 8) ? n : 2;
    }();
    cudaStream_t s = IK_STREAM(stream);
    if (dk == 128 && gdn_dec_rpw() > 0) {
        gdn_dec_run<true>(qkv, a, b, a_log, dt_bias, eps, o, nullptr, (int)Hk, (int)Hv, (int)dv, (int)R, (const uint8_t*)tab->data,
                          stride, index, byte_off, s);
        return ik_launch_status();
    }
#define IK_GDN_SPLIT_ROWS(DK, LPR)                                                                                        \
    gdn_delta_split_kernel<DK, LPR, true>                                                                                 \
        <<<dim3((unsigned)((dv + (32 / LPR) - 1) / (32 / LPR)), (unsigned)Hv, (unsigned)R), 32, 0, s>>>(                  \
            (const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data, (const ik_bf16*)a_log->data,    \
            (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data, nullptr, 1, (int)Hk, (int)Hv, (int)dv,                \
            (const uint8_t*)tab->data, stride, index, byte_off)
    if (dk == 128) {
        if (lpr_env == 4) IK_GDN_SPLIT_ROWS(128, 4);
        else if (lpr_env == 8) IK_GDN_SPLIT_ROWS(128, 8);
        else IK_GDN_SPLIT_ROWS(128, 2);
    } else {
        if (lpr_env == 4) IK_GDN_SPLIT_ROWS(64, 4);
        else if (lpr_env == 8) IK_GDN_SPLIT_ROWS(64, 8);
        else IK_GDN_SPLIT_ROWS(64, 2);
    }
#undef IK_GDN_SPLIT_ROWS
    return ik_launch_status();
}

}  // extern "C"
