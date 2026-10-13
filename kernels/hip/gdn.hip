// HIP port of kernels/cuda/gdn.cu (wave32 == the CUDA warp, so the kernels are unchanged apart from the sync).
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
#include "ik_hip.h"
#include <math.h>

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

// ---- delta rule, SPLIT threads per state row (dk <= 128) ----------------------------------
// The CUDA kernel keeps a whole row S_j[dk] of one value row in the registers of one thread, which needs 128 f32 for
// dk = 128 and spills on gfx12 (256 VGPRs per lane at most). Here SPLIT adjacent lanes share a row (each holds
// dk / SPLIT contiguous elements) and the two per-token reductions are one xor-shuffle each, so the recurrence stays
// register-resident: dk <= 64 uses SPLIT 1 (identical to the CUDA kernel), dk = 128 uses SPLIT 2.
// q/k of the token are normalised once per wave, staged in shared memory and read back as broadcasts. Block = 1 wave
// = 32 / SPLIT value rows of one head; grid (dv / (32 / SPLIT), Hv). The next token's raw inputs are loaded while the
// current token's update runs.
template <int DK, int SPLIT>
__global__ void __launch_bounds__(32) gdn_delta_rows_kernel(
    const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
    const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
    float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int PL = DK / 32;      // q/k elements per lane (cooperative load)
    constexpr int EL = DK / SPLIT;   // state elements per thread
    constexpr int ROWS = 32 / SPLIT; // value rows per wave
    __shared__ __align__(16) float sq[DK];
    __shared__ __align__(16) float sk[DK];
    const int lane = threadIdx.x;
    const int part = lane % SPLIT;
    const int j = blockIdx.x * ROWS + lane / SPLIT;
    const bool valid = j < dv;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);

    float s[EL];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK + part * EL);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) sv[i] = src[i];
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
        __syncthreads();  // the previous token's reads of sq/sk are done (block = one wave)
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            sq[lane + 32 * i] = qr_[i] * qn;
            sk[lane + 32 * i] = kr_[i] * kn;
        }
        __syncthreads();
        const float v = vraw, beta = sigmoidf(braw), decay = expf(neg_a * softplus(araw + dtb));
        if (t + 1 < T) load_raw(t + 1);  // in flight during the update below
        float kv = 0.f;
        const float4* k4 = reinterpret_cast<const float4*>(sk) + part * (EL / 4);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) {
            if (i % 4 == 0) __builtin_amdgcn_sched_barrier(0);  // keep the sk reads from all going live at once
            const float4 kk4 = k4[i];
            sv[i].x *= decay; sv[i].y *= decay; sv[i].z *= decay; sv[i].w *= decay;
            kv += sv[i].x * kk4.x + sv[i].y * kk4.y + sv[i].z * kk4.z + sv[i].w * kk4.w;
        }
#pragma unroll
        for (int off = 1; off < SPLIT; off <<= 1) kv += __shfl_xor(kv, off, IK_WARP);
        const float delta = (v - kv) * beta;
        float out = 0.f;
        const float4* q4 = reinterpret_cast<const float4*>(sq) + part * (EL / 4);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) {
            if (i % 4 == 0) __builtin_amdgcn_sched_barrier(0);
            const float4 kk4 = k4[i], qq4 = q4[i];
            sv[i].x += kk4.x * delta; sv[i].y += kk4.y * delta; sv[i].z += kk4.z * delta; sv[i].w += kk4.w * delta;
            out += sv[i].x * qq4.x + sv[i].y * qq4.y + sv[i].z * qq4.z + sv[i].w * qq4.w;
        }
#pragma unroll
        for (int off = 1; off < SPLIT; off <<= 1) out += __shfl_xor(out, off, IK_WARP);
        if (valid && part == 0) o[(t * Hv + h) * dv + j] = ik_f2b(out);
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK + part * EL);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) dst[i] = sv[i];
    }
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
    hipStream_t s = IK_STREAM(stream);
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
    hipStream_t s = IK_STREAM(stream);
    if (dk <= 128) {
#define IK_GDN_ROWS(DK, SPLIT)                                                                                   \
    {                                                                                                            \
        const dim3 rgrid((unsigned)((dv + (32 / SPLIT) - 1) / (32 / SPLIT)), (unsigned)Hv);                      \
        gdn_delta_rows_kernel<DK, SPLIT><<<rgrid, 32, 0, s>>>(                                                   \
            (const ik_bf16*)qkv->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data,                         \
            (const ik_bf16*)a_log->data, (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data,                  \
            (float*)state->data, T, (int)Hk, (int)Hv, (int)dv);                                                  \
    }
        switch (dk) {
            case 32: IK_GDN_ROWS(32, 1); break;
            case 64: IK_GDN_ROWS(64, 1); break;
            case 128: IK_GDN_ROWS(128, 2); break;
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

}  // extern "C"
