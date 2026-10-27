// Gated DeltaNet (linear attention) and gated-attention helper ops of Qwen3.6-35B-A3B: GateSplit, SigmoidMul, GdnConv,
// GdnDeltaRule. Semantics: crates/intarsia-core/src/gdn.rs (HF Qwen3_5MoeGatedDeltaNet), graded against the cpu oracle.
// Port of kernels/hip/gdn.hip (design notes there): bf16 in / out, f32 math and state, one rounding at the end.
//
// gdn_conv: one thread per (channel, tile of 32 tokens); the K-1 inputs before the tile come from x (or from the f32 conv
// state for the first tile). The state update runs after every tile has read it: fused at the end of the kernel when
// T fits one tile (a thread only touches its own channel), a second tiny dispatch otherwise.
// gdn_delta_rule: the recurrence is independent per (value head, dv row j): one simdgroup owns the row S_j[dk] (dk / 32
// floats per lane, in registers) and walks all T tokens in ONE dispatch; q / k are l2-normalised per token by simd_sum
// (redundantly in every row of the head: no extra state traffic). Decode streams the state once in, once out.

#include "common.h"

inline float gdn_silu(float g) { return g / (1.0f + precise::exp(-g)); }
inline float gdn_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }
// PyTorch softplus: threshold 20, log1p(exp(x)) with log1p(y) = log(1 + y) * y / ((1 + y) - 1) (exact for small y)
inline float gdn_softplus(float x) {
    if (x > 20.0f) return x;
    const float e = precise::exp(x);
    const float u = 1.0f + e;
    return u == 1.0f ? e : precise::log(u) * (e / (u - 1.0f));
}

// ---- GateSplit: x [T,N,2D] -> q [T,N,D], gate [T,N,D] (per head: first D = query, next D = gate) -------------------------
// bind: 0 x, 1 q, 2 gate. a = T*N*D (outputs), b = D. s0..s2 skips. One thread per output element, a bit copy.
kernel void gate_split_bf16(device const ushort* x [[buffer(0)]],
                            device ushort* q [[buffer(1)]],
                            device ushort* gate [[buffer(2)]],
                            constant OpP& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a) return;
    const uint row = gid / p.b, i = gid - row * p.b;
    const ulong src = (ulong)row * 2 * p.b + i;
    q[p.s1 + gid] = x[p.s0 + src];
    gate[p.s2 + gid] = x[p.s0 + src + p.b];
}

// ---- SigmoidMul: out = a * sigmoid(g) (f32, one rounding) ----------------------------------------------------------------
// bind: 0 a, 1 g, 2 out. a = element count. s0..s2 skips.
kernel void sigmoid_mul_bf16(device const ushort* a [[buffer(0)]],
                             device const ushort* g [[buffer(1)]],
                             device ushort* out [[buffer(2)]],
                             constant OpP& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a) return;
    out[p.s2 + gid] = f2bf(bf2f(a[p.s0 + gid]) * gdn_sigmoid(bf2f(g[p.s1 + gid])));
}

// ---- GdnConv: depthwise causal conv over tokens (taps w [C,K], tap K-1 = current token) + SiLU ---------------------------
constant constexpr uint GDN_CONV_TILE = 32;
constant constexpr uint GDN_CONV_TG = 128;

// state [C,K-1] (index 0 oldest) <- the last K-1 of [old state | x]; ascending j reads only slots >= j, so it is in place
inline void gdn_conv_state_update(device const ushort* x, device float* state, uint c, uint T, uint C, uint K, uint skip) {
    for (uint j = 0; j + 1 < K; ++j) {
        const ulong e = (ulong)T + j;  // index into [old state (K-1) | x (T)]
        state[(ulong)c * (K - 1) + j] = e < K - 1 ? state[(ulong)c * (K - 1) + e] : bf2f(x[skip + (e - (K - 1)) * C + c]);
    }
}

// bind: 0 x [T,C], 1 w [C,K], 2 y [T,C], 3 state f32 [C,K-1]. a = T, b = C, c = K, d = update the state here.
// s0..s2 skips of x / w / y. Groups (ceil(C / 128), token tiles) of 128 threads.
template <uint K>
kernel void gdn_conv_t(device const ushort* x [[buffer(0)]],
                       device const ushort* w [[buffer(1)]],
                       device ushort* y [[buffer(2)]],
                       device float* state [[buffer(3)]],
                       constant OpP& p [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 lt2 [[thread_position_in_threadgroup]]) {
    const uint C = p.b;
    const uint c = tg.x * GDN_CONV_TG + lt2.x;
    if (c >= C) return;
    const long t0 = (long)tg.y * GDN_CONV_TILE;
    const long t1 = min(t0 + (long)GDN_CONV_TILE, (long)p.a);
    float taps[K], h[K - 1];
#pragma unroll
    for (uint j = 0; j < K; ++j) taps[j] = bf2f(w[p.s1 + (ulong)c * K + j]);
#pragma unroll
    for (uint j = 0; j < K - 1; ++j) {
        const long pos = t0 - (K - 1) + j;
        h[j] = pos >= 0 ? bf2f(x[p.s0 + (ulong)pos * C + c]) : state[(ulong)c * (K - 1) + (K - 1) + pos];
    }
    for (long t = t0; t < t1; ++t) {
        const float cur = bf2f(x[p.s0 + (ulong)t * C + c]);
        float acc = taps[K - 1] * cur;
#pragma unroll
        for (uint j = 0; j < K - 1; ++j) acc += taps[j] * h[j];
        y[p.s2 + (ulong)t * C + c] = f2bf(gdn_silu(acc));
#pragma unroll
        for (uint j = 0; j + 2 < K; ++j) h[j] = h[j + 1];
        h[K - 2] = cur;
    }
    if (p.d != 0) gdn_conv_state_update(x, state, c, p.a, C, K, p.s0);
}

// bind: 0 x, 1 state. a = T, b = C, c = K. s0 skip of x. Threads: C.
kernel void gdn_conv_state(device const ushort* x [[buffer(0)]],
                           device float* state [[buffer(1)]],
                           constant OpP& p [[buffer(2)]],
                           uint c [[thread_position_in_grid]]) {
    if (c >= p.b) return;
    gdn_conv_state_update(x, state, c, p.a, p.b, p.c, p.s0);
}

#define INST_CONV(K)                                                                                                    \
    template [[host_name("gdn_conv_bf16_k" #K)]] kernel void gdn_conv_t<K>(                                             \
        device const ushort*, device const ushort*, device ushort*, device float*, constant OpP&, uint2, uint2);
INST_CONV(2)
INST_CONV(3)
INST_CONV(4)
INST_CONV(5)
INST_CONV(6)
INST_CONV(7)
INST_CONV(8)

// ---- GdnDeltaRule ----------------------------------------------------------------------------------------------------
// bind: 0 qkv [T,C] = [q (Hk*dk) | k (Hk*dk) | v (Hv*dv)], 1 a [T,Hv], 2 b [T,Hv], 3 A_log [Hv], 4 dt_bias [Hv] (all bf16),
// 5 o [T,Hv,dv] bf16, 6 state f32 [Hv,dv,dk] (dk contiguous, transposed; in place), 7 params.
// Groups (ceil(dv / GDN_DELTA_SG), Hv) of GDN_DELTA_SG simdgroups, one per row j of head h.
constant constexpr uint GDN_DELTA_SG = 4;

struct GdnDeltaP {
    uint t, hk, hv, dv;
    uint s_qkv, s_a, s_b, s_alog, s_dtb, s_o;  // leading bf16 elements to skip per tensor
    float eps;
    uint pad;
};

template <uint DK>
kernel void gdn_delta_rule_t(device const ushort* qkv [[buffer(0)]],
                             device const ushort* ga [[buffer(1)]],
                             device const ushort* gb [[buffer(2)]],
                             device const ushort* alog [[buffer(3)]],
                             device const ushort* dtb [[buffer(4)]],
                             device ushort* o [[buffer(5)]],
                             device float* state [[buffer(6)]],
                             constant GdnDeltaP& p [[buffer(7)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = DK / 32;  // state elements per lane (lane, lane + 32, ...)
    const uint h = tg.y;
    const uint j = tg.x * GDN_DELTA_SG + sg;
    if (j >= p.dv) return;  // whole simdgroup
    const uint kh = h / (p.hv / p.hk);
    const ulong C = 2ul * p.hk * DK + (ulong)p.hv * p.dv;
    const float scale = 1.0f / precise::sqrt((float)DK);
    const float neg_a = -precise::exp(bf2f(alog[p.s_alog + h]));
    const float dtbias = bf2f(dtb[p.s_dtb + h]);

    device float* srow = state + ((ulong)h * p.dv + j) * DK;
    float s[PL];
    for (uint i = 0; i < PL; ++i) s[i] = srow[lane + 32 * i];

    device const ushort* qp = qkv + p.s_qkv + (ulong)kh * DK + lane;
    device const ushort* kp = qkv + p.s_qkv + ((ulong)p.hk + kh) * DK + lane;
    device const ushort* vp = qkv + p.s_qkv + 2ul * p.hk * DK + (ulong)h * p.dv + j;

    // raw bf16 of the current token; the next token's are loaded before the current one is computed
    ushort cq[PL], ck[PL], cv, ca, cb;
    for (uint i = 0; i < PL; ++i) { cq[i] = qp[32 * i]; ck[i] = kp[32 * i]; }
    cv = *vp;
    ca = ga[p.s_a + h];
    cb = gb[p.s_b + h];
    for (uint t = 0; t < p.t; ++t) {
        const uint tn = min(t + 1, p.t - 1);
        ushort nq[PL], nk[PL];
        for (uint i = 0; i < PL; ++i) { nq[i] = qp[tn * C + 32 * i]; nk[i] = kp[tn * C + 32 * i]; }
        const ushort nv = vp[tn * C];
        const ushort na = ga[p.s_a + (ulong)tn * p.hv + h];
        const ushort nb = gb[p.s_b + (ulong)tn * p.hv + h];

        float qn[PL], kn[PL], qq = 0.0f, kk = 0.0f;
        for (uint i = 0; i < PL; ++i) {
            qn[i] = bf2f(cq[i]);
            kn[i] = bf2f(ck[i]);
            qq += qn[i] * qn[i];
            kk += kn[i] * kn[i];
        }
        qq = simd_sum(qq);
        kk = simd_sum(kk);
        const float qr = (1.0f / precise::sqrt(qq + p.eps)) * scale;
        const float kr = 1.0f / precise::sqrt(kk + p.eps);
        const float beta = gdn_sigmoid(bf2f(cb));
        const float decay = precise::exp(neg_a * gdn_softplus(bf2f(ca) + dtbias));

        float kv = 0.0f;
        for (uint i = 0; i < PL; ++i) {
            kn[i] *= kr;
            qn[i] *= qr;
            s[i] *= decay;
            kv += s[i] * kn[i];
        }
        kv = simd_sum(kv);
        const float delta = (bf2f(cv) - kv) * beta;
        float out = 0.0f;
        for (uint i = 0; i < PL; ++i) {
            s[i] += kn[i] * delta;
            out += s[i] * qn[i];
        }
        out = simd_sum(out);
        if (lane == 0) o[p.s_o + ((ulong)t * p.hv + h) * p.dv + j] = f2bf(out);

        for (uint i = 0; i < PL; ++i) { cq[i] = nq[i]; ck[i] = nk[i]; }
        cv = nv;
        ca = na;
        cb = nb;
    }
    for (uint i = 0; i < PL; ++i) srow[lane + 32 * i] = s[i];
}

#define INST_DELTA(DK)                                                                                                  \
    template [[host_name("gdn_delta_rule_bf16_dk" #DK)]] kernel void gdn_delta_rule_t<DK>(                              \
        device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,   \
        device ushort*, device float*, constant GdnDeltaP&, uint2, uint, uint);
INST_DELTA(32)
INST_DELTA(64)
INST_DELTA(128)
INST_DELTA(256)

// ---- GdnDeltaRule, prefill form with a prep pass (lane MGD) ----------------------------------------------------------------
// Bitwise `gdn_delta_rule_t`. That kernel recomputes, in every one of the dv rows of a head, the per-token work that does not
// depend on the state: the l2 norms of q and k (two simd_sums), the reciprocal square roots, sigmoid(b), softplus / exp(a). The
// prep kernel does it once per (token, key head) and writes the normalised q, k (f32, the very values the recurrence forms:
// bf2f(q) * qr, bf2f(k) * kr) and the per-(token, value head) beta and decay; the recurrence then reads them and keeps only
// the state chain: s *= decay, one simd_sum for kv, delta, s += k delta, one simd_sum for the output. Every float operation of
// the recurrence is the original's, in the original's order, so the outputs and the state are bit-for-bit the original's.
// bind (prep): 0 qkv, 1 a, 2 b, 3 A_log, 4 dt_bias, 5 scratch f32 = qk [T,Hk,2,dk] (q * qr | k * kr), then (at float offset
// p.pad) gates [T,Hv,2] (beta | decay), 6 params. Groups (T, Hk) of 32 threads (one simdgroup).
template <uint DK>
kernel void gdn_delta_prep_t(device const ushort* qkv [[buffer(0)]],
                             device const ushort* ga [[buffer(1)]],
                             device const ushort* gb [[buffer(2)]],
                             device const ushort* alog [[buffer(3)]],
                             device const ushort* dtb [[buffer(4)]],
                             device float* scratch [[buffer(5)]],
                             constant GdnDeltaP& p [[buffer(6)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = DK / 32;
    const uint t = tg.x, kh = tg.y;
    device float* qk = scratch;
    device float* gates = scratch + p.pad;
    const ulong C = 2ul * p.hk * DK + (ulong)p.hv * p.dv;
    const float scale = 1.0f / precise::sqrt((float)DK);
    device const ushort* qp = qkv + p.s_qkv + t * C + (ulong)kh * DK + lane;
    device const ushort* kp = qkv + p.s_qkv + t * C + ((ulong)p.hk + kh) * DK + lane;
    float qn[PL], kn[PL], qq = 0.0f, kk = 0.0f;
    for (uint i = 0; i < PL; ++i) {
        qn[i] = bf2f(qp[32 * i]);
        kn[i] = bf2f(kp[32 * i]);
        qq += qn[i] * qn[i];
        kk += kn[i] * kn[i];
    }
    qq = simd_sum(qq);
    kk = simd_sum(kk);
    const float qr = (1.0f / precise::sqrt(qq + p.eps)) * scale;
    const float kr = 1.0f / precise::sqrt(kk + p.eps);
    device float* dst = qk + ((ulong)t * p.hk + kh) * 2 * DK + lane;
    for (uint i = 0; i < PL; ++i) {
        dst[32 * i] = qn[i] * qr;
        dst[DK + 32 * i] = kn[i] * kr;
    }
    // the value heads of this key head: lane r takes head kh * (Hv / Hk) + r
    const uint grp = p.hv / p.hk;
    if (lane < grp) {
        const uint h = kh * grp + lane;
        const float neg_a = -precise::exp(bf2f(alog[p.s_alog + h]));
        const float dtbias = bf2f(dtb[p.s_dtb + h]);
        const float beta = gdn_sigmoid(bf2f(gb[p.s_b + (ulong)t * p.hv + h]));
        const float decay = precise::exp(neg_a * gdn_softplus(bf2f(ga[p.s_a + (ulong)t * p.hv + h]) + dtbias));
        gates[((ulong)t * p.hv + h) * 2] = beta;
        gates[((ulong)t * p.hv + h) * 2 + 1] = decay;
    }
}

// bind (recurrence): 0 qkv (only v is read), 1 scratch (as written by the prep), 2 o [T,Hv,dv] bf16, 3 state f32 [Hv,dv,dk] (in
// place), 4 params. Groups (dv / GDN_PRE_ROWS, Hv) of GDN_DELTA_SG simdgroups (dv % GDN_PRE_ROWS == 0), GDN_PRE_RPS rows j of head h
// each, every row as in `gdn_delta_rule_t`. The four simdgroups of a group read the same q, k, beta, decay of every token: a tile of GDN_PRE_TILE tokens
// is staged in threadgroup memory by all 128 threads (the next tile is fetched into registers while the current one is consumed),
// so the serial loop reads no device memory.
// Value rows a simdgroup runs side by side (independent chains: their simd_sums overlap) and rows a threadgroup covers.
constant constexpr uint GDN_PRE_RPS = 4;
constant constexpr uint GDN_PRE_ROWS = GDN_DELTA_SG * GDN_PRE_RPS;

template <uint DK>
struct GdnPreTile {
    static constant constexpr uint TILE = DK >= 256 ? 8 : 16;          // tokens per tile
    static constant constexpr uint QPT = TILE * 2 * DK / (32 * GDN_DELTA_SG);  // q / k floats a thread stages per tile
};

#define PRE_FETCH(T0)                                                                                                   \
    {                                                                                                                   \
        const uint fn = min(TILE, p.t - (T0));                                                                          \
        for (uint r = 0; r < QPT; ++r) {                                                                                \
            const uint idx = tid + NT * r, tt = idx / (2 * DK), e = idx % (2 * DK);                                     \
            rq[r] = tt < fn ? qk[((T0) + tt) * qks + e] : 0.0f;                                                         \
        }                                                                                                               \
        rg = tid < 2 * TILE && tid / 2 < fn ? gates[((T0) + tid / 2) * gs + (tid & 1)] : 0.0f;                          \
        for (uint r = 0; r < VPT; ++r) {                                                                                \
            const uint idx = tid + NT * r, tt = idx / GDN_PRE_ROWS;                                                     \
            rv[r] = idx < TILE * GDN_PRE_ROWS && tt < fn ? vp[((T0) + tt) * C + (idx % GDN_PRE_ROWS)] : (ushort)0;      \
        }                                                                                                               \
    }
#define PRE_STAGE()                                                                                                     \
    {                                                                                                                   \
        for (uint r = 0; r < QPT; ++r) {                                                                                \
            const uint idx = tid + NT * r;                                                                              \
            sqk[idx / (2 * DK)][idx % (2 * DK)] = rq[r];                                                                \
        }                                                                                                               \
        if (tid < 2 * TILE) sgt[tid / 2][tid & 1] = rg;                                                                 \
        for (uint r = 0; r < VPT; ++r) {                                                                                \
            const uint idx = tid + NT * r;                                                                              \
            if (idx < TILE * GDN_PRE_ROWS) sv[idx / GDN_PRE_ROWS][idx % GDN_PRE_ROWS] = rv[r];                          \
        }                                                                                                               \
    }

template <uint DK>
kernel void gdn_delta_rule_pre_t(device const ushort* qkv [[buffer(0)]],
                                 device const float* scratch [[buffer(1)]],
                                 device ushort* o [[buffer(2)]],
                                 device float* state [[buffer(3)]],
                                 constant GdnDeltaP& p [[buffer(4)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = DK / 32;
    constexpr uint TILE = GdnPreTile<DK>::TILE;
    constexpr uint QPT = GdnPreTile<DK>::QPT;
    constexpr uint NT = 32 * GDN_DELTA_SG;
    constexpr uint RPS = GDN_PRE_RPS;
    constexpr uint VPT = (TILE * GDN_PRE_ROWS + NT - 1) / NT;  // v values a thread stages per tile
    threadgroup float sqk[TILE][2 * DK];
    threadgroup float sgt[TILE][2];
    threadgroup ushort sv[TILE][GDN_PRE_ROWS];

    const uint h = tg.y;
    const uint j0 = tg.x * GDN_PRE_ROWS + sg * RPS;  // rows j0 .. j0 + RPS - 1 < dv: the host requires dv % GDN_PRE_ROWS == 0
    const uint tid = sg * 32 + lane;
    const uint kh = h / (p.hv / p.hk);
    const ulong C = 2ul * p.hk * DK + (ulong)p.hv * p.dv;
    device const float* qk = scratch + (ulong)kh * 2 * DK;  // + t * Hk * 2 * dk
    device const float* gates = scratch + p.pad + (ulong)h * 2;  // + t * Hv * 2
    device const ushort* vp = qkv + p.s_qkv + 2ul * p.hk * DK + (ulong)h * p.dv + (ulong)tg.x * GDN_PRE_ROWS;  // + t * C
    const ulong qks = (ulong)p.hk * 2 * DK, gs = (ulong)p.hv * 2;

    float s[RPS][PL];
    for (uint r = 0; r < RPS; ++r) {
        device float* srow = state + ((ulong)h * p.dv + j0 + r) * DK;
        for (uint i = 0; i < PL; ++i) s[r][i] = srow[lane + 32 * i];
    }

    // the tile at token t0 -> registers (rq: q | k floats, rg: beta, decay, rv: v of the group's rows); slots past T are unused
    float rq[QPT], rg = 0.0f;
    ushort rv[VPT];
    PRE_FETCH(0)
    for (uint t0 = 0; t0 < p.t; t0 += TILE) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous tile is consumed
        PRE_STAGE()
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + TILE < p.t) PRE_FETCH(t0 + TILE)
        const uint n = min(TILE, p.t - t0);
        for (uint tt = 0; tt < n; ++tt) {
            const float cdecay = sgt[tt][1], cbeta = sgt[tt][0];
            float kv[RPS];
            for (uint r = 0; r < RPS; ++r) {
                kv[r] = 0.0f;
                for (uint i = 0; i < PL; ++i) {
                    s[r][i] *= cdecay;
                    kv[r] += s[r][i] * sqk[tt][DK + lane + 32 * i];
                }
            }
            for (uint r = 0; r < RPS; ++r) kv[r] = simd_sum(kv[r]);
            float out[RPS];
            for (uint r = 0; r < RPS; ++r) {
                const float delta = (bf2f(sv[tt][sg * RPS + r]) - kv[r]) * cbeta;
                out[r] = 0.0f;
                for (uint i = 0; i < PL; ++i) {
                    s[r][i] += sqk[tt][DK + lane + 32 * i] * delta;
                    out[r] += s[r][i] * sqk[tt][lane + 32 * i];
                }
            }
            for (uint r = 0; r < RPS; ++r) {
                const float ov = simd_sum(out[r]);
                if (lane == 0) o[p.s_o + ((ulong)(t0 + tt) * p.hv + h) * p.dv + j0 + r] = f2bf(ov);
            }
        }
    }
    for (uint r = 0; r < RPS; ++r) {
        device float* srow = state + ((ulong)h * p.dv + j0 + r) * DK;
        for (uint i = 0; i < PL; ++i) srow[lane + 32 * i] = s[r][i];
    }
}

#define INST_DELTA_PRE(DK)                                                                                              \
    template [[host_name("gdn_delta_prep_dk" #DK)]] kernel void gdn_delta_prep_t<DK>(                                   \
        device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,   \
        device float*, constant GdnDeltaP&, uint2, uint);                                                               \
    template [[host_name("gdn_delta_rule_pre_dk" #DK)]] kernel void gdn_delta_rule_pre_t<DK>(                           \
        device const ushort*, device const float*, device ushort*, device float*, constant GdnDeltaP&, uint2, uint, uint);
INST_DELTA_PRE(32)
INST_DELTA_PRE(64)
INST_DELTA_PRE(128)
INST_DELTA_PRE(256)

// ---- Row tables (batched decode, lane mr3) -------------------------------------------------------------------------------
// `gdn_conv_bf16_rows_k*` / `gdn_delta_rule_bf16_rows_dk*`: the T = 1 conv and delta-rule calls of the solo decode step on R
// rows (sequences) in ONE dispatch (threadgroup z = row). Row r of the activations ([R, ...]) is the single input row of
// sequence r; the recurrent state of row r is the f32 buffer whose device address is entry [r, index] of the row table (a
// u64 table, stride entries per row) plus byte_off. Every expression is the solo kernel's at T = 1 (same operations, same
// order), so row r is bitwise the solo call on sequence r (tests/row_tables.rs). The state update is in place.

inline device float* rt_state(device const ulong* tab, uint row, uint stride, uint index, uint byte_off) {
    return reinterpret_cast<device float*>(tab[(ulong)row * stride + index] + byte_off);
}

// bind: 0 x [R,C], 1 w [C,K], 2 y [R,C], 3 table, 4 OpP: a = R, b = C, c = K, d = stride, e = index, f = byte_off;
// s0..s2 skips of x / w / y. Groups (ceil(C / 128), 1, R) of 128 threads.
template <uint K>
kernel void gdn_conv_rows_t(device const ushort* x [[buffer(0)]],
                            device const ushort* w [[buffer(1)]],
                            device ushort* y [[buffer(2)]],
                            device const ulong* tab [[buffer(3)]],
                            constant OpP& p [[buffer(4)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint3 lt3 [[thread_position_in_threadgroup]]) {
    const uint C = p.b;
    const uint c = tg.x * GDN_CONV_TG + lt3.x;
    if (c >= C) return;
    const uint row = tg.z;
    device float* state = rt_state(tab, row, p.d, p.e, p.f);
    device const ushort* xr = x + (ulong)row * C;
    device ushort* yr = y + (ulong)row * C;
    float taps[K], h[K - 1];
#pragma unroll
    for (uint j = 0; j < K; ++j) taps[j] = bf2f(w[p.s1 + (ulong)c * K + j]);
#pragma unroll
    for (uint j = 0; j < K - 1; ++j) h[j] = state[(ulong)c * (K - 1) + j];
    const float cur = bf2f(xr[p.s0 + c]);
    float acc = taps[K - 1] * cur;
#pragma unroll
    for (uint j = 0; j < K - 1; ++j) acc += taps[j] * h[j];
    yr[p.s2 + c] = f2bf(gdn_silu(acc));
    gdn_conv_state_update(xr, state, c, 1, C, K, p.s0);
}

#define INST_CONV_ROWS(K)                                                                                               \
    template [[host_name("gdn_conv_bf16_rows_k" #K)]] kernel void gdn_conv_rows_t<K>(                                   \
        device const ushort*, device const ushort*, device ushort*, device const ulong*, constant OpP&, uint3, uint3);
INST_CONV_ROWS(2)
INST_CONV_ROWS(3)
INST_CONV_ROWS(4)
INST_CONV_ROWS(5)
INST_CONV_ROWS(6)
INST_CONV_ROWS(7)
INST_CONV_ROWS(8)

// bind: 0 qkv [R,C], 1 a [R,Hv], 2 b [R,Hv], 3 A_log, 4 dt_bias, 5 o [R,Hv,dv], 6 table, 7 params.
// Groups (ceil(dv / GDN_DELTA_SG), Hv, R) of GDN_DELTA_SG simdgroups.
struct GdnDeltaRowsP {
    uint hk, hv, dv, stride;
    uint s_qkv, s_a, s_b, s_alog, s_dtb, s_o;  // leading bf16 elements to skip per tensor
    uint index, byte_off;
    float eps;
    uint pad;
};

template <uint DK>
kernel void gdn_delta_rule_rows_t(device const ushort* qkv [[buffer(0)]],
                                  device const ushort* ga [[buffer(1)]],
                                  device const ushort* gb [[buffer(2)]],
                                  device const ushort* alog [[buffer(3)]],
                                  device const ushort* dtb [[buffer(4)]],
                                  device ushort* o [[buffer(5)]],
                                  device const ulong* tab [[buffer(6)]],
                                  constant GdnDeltaRowsP& p [[buffer(7)]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PL = DK / 32;
    const uint h = tg.y;
    const uint j = tg.x * GDN_DELTA_SG + sg;
    const uint row = tg.z;
    if (j >= p.dv) return;  // whole simdgroup
    const uint kh = h / (p.hv / p.hk);
    const ulong C = 2ul * p.hk * DK + (ulong)p.hv * p.dv;
    const float scale = 1.0f / precise::sqrt((float)DK);
    const float neg_a = -precise::exp(bf2f(alog[p.s_alog + h]));
    const float dtbias = bf2f(dtb[p.s_dtb + h]);

    device float* state = rt_state(tab, row, p.stride, p.index, p.byte_off);
    device float* srow = state + ((ulong)h * p.dv + j) * DK;
    float s[PL];
    for (uint i = 0; i < PL; ++i) s[i] = srow[lane + 32 * i];

    device const ushort* qp = qkv + (ulong)row * C + p.s_qkv + (ulong)kh * DK + lane;
    device const ushort* kp = qkv + (ulong)row * C + p.s_qkv + ((ulong)p.hk + kh) * DK + lane;
    device const ushort* vp = qkv + (ulong)row * C + p.s_qkv + 2ul * p.hk * DK + (ulong)h * p.dv + j;

    float qn[PL], kn[PL], qq = 0.0f, kk = 0.0f;
    for (uint i = 0; i < PL; ++i) {
        qn[i] = bf2f(qp[32 * i]);
        kn[i] = bf2f(kp[32 * i]);
        qq += qn[i] * qn[i];
        kk += kn[i] * kn[i];
    }
    const ushort cv = *vp;
    const ushort ca = ga[(ulong)row * p.hv + p.s_a + h];
    const ushort cb = gb[(ulong)row * p.hv + p.s_b + h];
    qq = simd_sum(qq);
    kk = simd_sum(kk);
    const float qr = (1.0f / precise::sqrt(qq + p.eps)) * scale;
    const float kr = 1.0f / precise::sqrt(kk + p.eps);
    const float beta = gdn_sigmoid(bf2f(cb));
    const float decay = precise::exp(neg_a * gdn_softplus(bf2f(ca) + dtbias));

    float kv = 0.0f;
    for (uint i = 0; i < PL; ++i) {
        kn[i] *= kr;
        qn[i] *= qr;
        s[i] *= decay;
        kv += s[i] * kn[i];
    }
    kv = simd_sum(kv);
    const float delta = (bf2f(cv) - kv) * beta;
    float out = 0.0f;
    for (uint i = 0; i < PL; ++i) {
        s[i] += kn[i] * delta;
        out += s[i] * qn[i];
    }
    out = simd_sum(out);
    if (lane == 0) o[(ulong)row * p.hv * p.dv + p.s_o + (ulong)h * p.dv + j] = f2bf(out);
    for (uint i = 0; i < PL; ++i) srow[lane + 32 * i] = s[i];
}

#define INST_DELTA_ROWS(DK)                                                                                             \
    template [[host_name("gdn_delta_rule_bf16_rows_dk" #DK)]] kernel void gdn_delta_rule_rows_t<DK>(                    \
        device const ushort*, device const ushort*, device const ushort*, device const ushort*, device const ushort*,   \
        device ushort*, device const ulong*, constant GdnDeltaRowsP&, uint3, uint, uint);
INST_DELTA_ROWS(32)
INST_DELTA_ROWS(64)
INST_DELTA_ROWS(128)
INST_DELTA_ROWS(256)
