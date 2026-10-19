// Core-op fusion phase 2, P3 (docs/results/2026-10-19-core-op-fusion-p2.md): Op::GdnConvDelta, the GatedDeltaNet conv
// folded into the delta rule. One launch instead of gdn_conv_bf16 + gdn_delta_rule_bf16_v2 (gdn.hip) for chunks of up to
// 32 tokens (every decode step; the conv launcher's single-tile form, gdn_conv_kernel<true>).
//
// What runs: gdn_delta_rows_v2_kernel<128, 2, PD> at PD = 1, statement for statement, except that the q / k / v values a
// lane reads for a token are computed by the lane itself from the raw projection x (the conv's input) instead of read from
// the conv output y: per channel it repeats gdn_conv_kernel's per-thread loop (taps in registers, the K-1 history entries
// in LDS for q / k and in registers for v, the history from the old conv state, the output rounded to bf16 as y stores it). Every lane computes the channels it reads,
// so the conv of a key head's q / k channels runs in each of the blocks that read them (16 at the 35B's shape), as v2's
// q / k normalisation already does.
//
// The conv state hazard (proposal section 6f): the old conv state of a channel must be read by every block that needs it
// before the new one is stored. v channels have one reader (the block of that value row), which stores its row's new state
// itself. A q / k channel of key head kh is read by every block of the (Hv / Hk) value heads of kh; their new state is
// stored by the block that arrives LAST: each block, once it holds its old q / k state in registers, adds 1 to the key
// head's counter (one atomic, acq_rel, agent scope; the release orders the block's state reads before it), and the block
// that draws ticket nblk - 1 knows that every other block of kh has read the old state, so it stores the new one after its
// token loop and puts the counter back to 0 for the next launch on the stream. No block waits for another: there is no spin
// and nothing can deadlock whatever order the hardware dispatches blocks in. The counters are one 64-entry slot of
// g_gdn_cd_sync per (device, stream) (module scope, zero when the code object loads), as moe.hip's v3 does; without a free
// slot the launcher runs the fused kernel without any state store and then gdn_cd_state_kernel (gdn_conv_state_kernel:
// the new state from x and the old state), two launches with the same bits.
//
// So every output, delta-state and conv-state bit is the unfused chain's (tests/fusion_p2.rs, against the two launches on
// the same device, T = 1 .. 32, the unfused side at every PD). Test seams (default off): INTARSIA_HIP_GDN_CD_SKEW = n
// delays the state reads of every other block (x + y odd) by n sleeps (a store before a sibling's read then shows),
// INTARSIA_HIP_GDN_CD_NOSLOT=1 forces the
// two-launch form.
#include "ik_hip.h"
#include <math.h>
#include <mutex>
#include <stdlib.h>
#include <utility>
#include <vector>

namespace {

// ---- gdn.hip's helpers, copied unchanged (the delta rule v2 and the conv) -------------------------------------------
IK_DEVI float silu(float g) { return g / (1.f + expf(-g)); }
IK_DEVI float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
IK_DEVI float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

IK_DEVI float gdn_xor16(float v) {  // the value of lane ^ 16 (swap the two 16-lane rows)
    return __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(v), __float_as_uint(v), 0x76543210u, 0xfedcba98u,
                                                        false, false));
}
template <int K>
IK_DEVI float gdn_xorrow(float v) {  // the value of lane ^ K, K < 16 (DPP row_xmask)
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + K, 0xf, 0xf, false));
}
// ik_warp_sum: v += partner(xor 16), then xor 8, 4, 2, 1
IK_DEVI float gdn_warp_sum(float v) {
    v += gdn_xor16(v);
    v += gdn_xorrow<8>(v);
    v += gdn_xorrow<4>(v);
    v += gdn_xorrow<2>(v);
    v += gdn_xorrow<1>(v);
    return v;
}
IK_DEVI float gdn_dot4(float4 s, float4 k) {
#pragma clang fp contract(off)
    float t = __builtin_fmaf(s.x, k.x, s.y * k.y);
    t = __builtin_fmaf(s.z, k.z, t);
    return __builtin_fmaf(s.w, k.w, t);
}
IK_DEVI void gdn_lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// One token of gdn_conv_kernel's per-thread loop for one channel: the conv output (rounded to bf16, as y stores it, and
// read back as the delta rule reads y), then the history shift. tp = the channel's 4 taps as packed bf16 (two per register:
// as f32 they spilled the kernel, 255 VGPRs + scratch), hh = the K-1 previous inputs (oldest first). cd_tap is ik_b2f.
IK_DEVI float cd_tap(const uint2& tp, int j) {
    const uint32_t w = j < 2 ? tp.x : tp.y;
    return __uint_as_float((j & 1) ? (w & 0xffff0000u) : (w << 16));
}
// hh[j * S]: S = 1 for a register array, the LDS row stride for a history kept in LDS
template <int K, int S = 1>
IK_DEVI float cd_conv(const uint2& tp, float* hh, float cur) {
    static_assert(K == 4, "4 taps: one uint2 per channel");
    // gdn_conv_kernel's arithmetic spelled out: its tap loop (K is a runtime value there) compiles to one rounded multiply
    // (the current tap) and one fma per history tap into that accumulator (v_mul_f32 + v_fmac_f32, gfx1201). Written as
    // `acc = a*b; acc += c*d; ...` with K known, the first add sees two products and the contraction may fuse either one
    // (fma(a, b, c*d) rounds differently from fma(c, d, a*b)): about 1 conv output in 3e5 then flipped its bf16 rounding
    // (tests/fusion_p2.rs, many draws). So no contraction here, and the fmas explicit.
    float acc;
    {
#pragma clang fp contract(off)
        acc = cd_tap(tp, K - 1) * cur;
        for (int j = 0; j < K - 1; ++j) acc = __builtin_fmaf(cd_tap(tp, j), hh[j * S], acc);
    }
    const float y = ik_b2f(ik_f2b(silu(acc)));
    for (int j = 0; j < K - 2; ++j) hh[j * S] = hh[(j + 1) * S];
    hh[(K - 2) * S] = cur;
    return y;
}

template <int PL>
struct CdRaw {  // raw bf16 projection values (the conv's input) of one token for one lane
    ik_bf16 q[PL], k[PL], v;
};

constexpr int kCdSyncSlots = 16;
constexpr int kCdSyncMaxHk = 64;  // key heads per slot (one counter each)
__device__ unsigned int g_gdn_cd_sync[kCdSyncSlots * kCdSyncMaxHk];

template <int DK, int SPLIT, int PD, int K>
__global__ void __launch_bounds__(32) gdn_conv_delta_kernel(
    const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ taps, const ik_bf16* __restrict__ a,
    const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias, float eps,
    ik_bf16* __restrict__ o, float* __restrict__ state, float* cstate, int64_t T, int Hk, int Hv, int dv, int slot,
    int skew) {
    unsigned int* sync = slot >= 0 ? g_gdn_cd_sync + slot * kCdSyncMaxHk : nullptr;
    static_assert(SPLIT == 2, "the xor-1 partner of the row split is DPP row_xmask:1");
    static_assert(32 % PD == 0, "a 32-token gate batch starts at the first slot of the ring");
    constexpr int PL = DK / 32;      // q/k elements per lane (cooperative load)
    constexpr int EL = DK / SPLIT;   // state elements per thread
    constexpr int ROWS = 32 / SPLIT; // value rows per wave
    constexpr int KH = K - 1;        // conv history entries per channel
    __shared__ __align__(16) float sq[DK];
    __shared__ __align__(16) float sk[DK];
    const int lane = threadIdx.x;
    const int part = lane % SPLIT;
    const int j = blockIdx.x * ROWS + lane / SPLIT;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;  // a row that exists, for the unconditional loads of idle lanes
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;

    float s[EL];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK + part * EL);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) sv[i] = src[i];
    }

    // the lane's conv channels: the q / k values it reads (key head kh) and its value row's v
    const int cq0 = kh * DK + lane, ck0 = Hk * DK + kh * DK + lane, cv = 2 * Hk * DK + h * dv + jl;
    if (skew && ((blockIdx.x + blockIdx.y) & 1))  // test seam (INTARSIA_HIP_GDN_CD_SKEW): every other block reads late
        for (int i = 0; i < skew; ++i) __builtin_amdgcn_s_sleep(127);
    // taps (packed) and the old conv state (gdn_conv_kernel's history of the first tile: every entry from the state)
    const uint2* taps2 = reinterpret_cast<const uint2*>(taps);  // K = 4: channel c's taps are 8 bytes at c * 8
    uint2 tq[PL], tk[PL];
    const uint2 tv = taps2[cv];
    // the q / k history in LDS (each lane only its own 2 * PL channels, so no barrier): in registers the kernel spilled
    // (255 VGPRs + 28 bytes of scratch per lane at PD = 1); the v history (3 values) stays in registers
    __shared__ float shq[KH][DK], shk[KH][DK];
    float hv[KH];
#pragma unroll
    for (int i = 0; i < PL; ++i) {
        const int64_t cq = cq0 + 32 * i, ck = ck0 + 32 * i;
        tq[i] = taps2[cq];
        tk[i] = taps2[ck];
#pragma unroll
        for (int t = 0; t < KH; ++t) {
            shq[t][lane + 32 * i] = cstate[cq * KH + t];
            shk[t][lane + 32 * i] = cstate[ck * KH + t];
        }
    }
#pragma unroll
    for (int t = 0; t < KH; ++t) hv[t] = cstate[(int64_t)cv * KH + t];
    // this block holds its old q / k conv state: take the key head's ticket (the release orders the reads above before it);
    // the last of the key head's nblk blocks stores the new q / k state
    bool qk_writer = false;
    __syncthreads();  // every lane's state loads before lane 0's ticket (the release below also waits for the wave's loads)
    if (sync) {
        const unsigned nblk = gridDim.x * (unsigned)(Hv / Hk);
        unsigned tk_ = 0;
        if (lane == 0) tk_ = __hip_atomic_fetch_add(sync + kh, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        tk_ = __builtin_amdgcn_readfirstlane(tk_);
        qk_writer = tk_ == nblk - 1;
        // every block of kh has drawn its ticket: reset for the next launch on this stream (it starts after this one ends)
        if (qk_writer && lane == 0) __hip_atomic_store(sync + kh, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    auto load = [&](CdRaw<PL>& r, int64_t t) {
        const ik_bf16* row = x + (t < last ? t : last) * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            r.q[i] = row[kh * DK + lane + 32 * i];
            r.k[i] = row[Hk * DK + kh * DK + lane + 32 * i];
        }
        r.v = row[2 * Hk * DK + h * dv + jl];
    };
    // gates of a batch of 32 tokens, lane l holding token g0 + l's; the raw a / b of the next batch are in flight
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float gbeta = 0.f, gdecay = 0.f;

    CdRaw<PL> ring[PD];
#pragma unroll
    for (int d = 0; d < PD; ++d) load(ring[d], d);
    load_gates(0);

    auto step = [&](CdRaw<PL>& r, int64_t t) {
        // the conv of this token for the lane's channels, in place of v2's read of y
        float qr[PL], kr[PL];
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr[i] = cd_conv<K, DK>(tq[i], &shq[0][lane + 32 * i], ik_b2f(r.q[i]));
            kr[i] = cd_conv<K, DK>(tk[i], &shk[0][lane + 32 * i], ik_b2f(r.k[i]));
        }
        const float vc = cd_conv<K>(tv, hv, ik_b2f(r.v));
        const float v = valid ? vc : 0.f;
        load(r, t + PD);  // this slot's next token, in flight for the next PD tokens
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr[i] * qr[i];
            kk += kr[i] * kr[i];
        }
        qq = gdn_warp_sum(qq);
        kk = gdn_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
        gdn_lds_sync();  // the previous token's reads of sq/sk are done
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            sq[lane + 32 * i] = qr[i] * qn;
            sk[lane + 32 * i] = kr[i] * kn;
        }
        gdn_lds_sync();
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(gbeta), gl));
        const float decay = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(gdecay), gl));
        float kv = 0.f;
        float4 kreg[EL / 4];
        const float4* k4 = reinterpret_cast<const float4*>(sk) + part * (EL / 4);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) {
            const float4 kk4 = k4[i];
            kreg[i] = kk4;
            sv[i].x *= decay; sv[i].y *= decay; sv[i].z *= decay; sv[i].w *= decay;
            kv += gdn_dot4(sv[i], kk4);
        }
        kv += gdn_xorrow<1>(kv);
        const float delta = (v - kv) * beta;
        float out = 0.f;
        const float4* q4 = reinterpret_cast<const float4*>(sq) + part * (EL / 4);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) {
            const float4 kk4 = kreg[i], qq4 = q4[i];
            sv[i].x = __builtin_fmaf(kk4.x, delta, sv[i].x);
            sv[i].y = __builtin_fmaf(kk4.y, delta, sv[i].y);
            sv[i].z = __builtin_fmaf(kk4.z, delta, sv[i].z);
            sv[i].w = __builtin_fmaf(kk4.w, delta, sv[i].w);
            out += gdn_dot4(sv[i], qq4);
        }
        out += gdn_xorrow<1>(out);
        if (valid && part == 0) o[(t * Hv + h) * dv + j] = ik_f2b(out);
    };

    auto gates = [&](int64_t t0) {
        if ((t0 & 31) == 0) {
            // gdn_delta_rows_kernel's per-token beta / decay, for tokens t0 .. t0 + 31 (one per lane)
            const float ar = ik_b2f(ga), br = ik_b2f(gb);
            gbeta = sigmoidf(br);
            gdecay = expf(neg_a * softplus(ar + dtb));
            load_gates(t0 + 32);
        }
    };
    const int64_t full = T - T % PD;
    for (int64_t t0 = 0; t0 < full; t0 += PD) {
        gates(t0);
#pragma unroll
        for (int d = 0; d < PD; ++d) step(ring[d], t0 + d);
    }
    if (full < T) {
        gates(full);
#pragma unroll
        for (int d = 0; d < PD; ++d)
            if (full + d < T) step(ring[d], full + d);
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK + part * EL);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) dst[i] = sv[i];
    }
    // the new conv state = each channel's final history (gdn_conv_kernel<true>'s FUSE_STATE store): the value row's v
    // channel by its one reader, the key head's q / k channels by the last block of kh (every lane its 2 * PL channels)
    if (sync) {
        if (valid && part == 0)
#pragma unroll
            for (int t = 0; t < KH; ++t) cstate[(int64_t)cv * KH + t] = hv[t];
        if (qk_writer) {
#pragma unroll
            for (int i = 0; i < PL; ++i) {
                const int64_t cq = cq0 + 32 * i, ck = ck0 + 32 * i;
#pragma unroll
                for (int t = 0; t < KH; ++t) {
                    cstate[cq * KH + t] = shq[t][lane + 32 * i];
                    cstate[ck * KH + t] = shk[t][lane + 32 * i];
                }
            }
        }
    }
}

// gdn.hip's gdn_conv_state_kernel: state[c] = the last K-1 inputs of the extended sequence [old state | x] (the two-launch
// form, after a fused launch that stored no conv state)
__global__ void gdn_cd_state_kernel(const ik_bf16* __restrict__ x, float* __restrict__ state, int64_t T, int C, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[8];
    for (int j = 0; j < K - 1; ++j) {
        const int64_t p = T - (K - 1) + j;
        h[j] = p >= 0 ? ik_b2f(x[p * C + c]) : state[(int64_t)c * (K - 1) + (K - 1) + p];
    }
    for (int j = 0; j < K - 1; ++j) state[(int64_t)c * (K - 1) + j] = h[j];
}

// The g_gdn_cd_sync slot of the calling thread's (device, stream); -1 when all kCdSyncSlots are taken.
int gdn_cd_sync_slot(hipStream_t s) {
    static std::mutex mu;
    static std::vector<std::pair<int, hipStream_t>> owners;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) return -1;
    std::lock_guard<std::mutex> g(mu);
    for (size_t i = 0; i < owners.size(); ++i)
        if (owners[i].first == dev && owners[i].second == s) return (int)i;
    if ((int)owners.size() >= kCdSyncSlots) return -1;
    owners.emplace_back(dev, s);
    return (int)owners.size() - 1;
}

int env_int(const char* name) {
    const char* e = getenv(name);
    return e ? atoi(e) : 0;
}

}  // namespace

extern "C" {

// ik_gdn_conv_bf16's and ik_gdn_delta_rule_bf16_v2's checks; dk = 128, K = 4 and T <= 32 only (the registration's)
int ik_gdn_conv_delta_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* a, const ik_tensor* b,
                           const ik_tensor* a_log, const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state,
                           ik_tensor* cstate, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(taps, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(a_log, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(dt_bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(o, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(state, IK_DTYPE_F32, 3));
    IK_TRY(ik_check(cstate, IK_DTYPE_F32, 2));
    const int64_t T = x->shape[0], C = x->shape[1], K = taps->shape[1];
    const int64_t Hv = o->shape[1], dv = o->shape[2], dk = state->shape[2];
    if (K < 2 || taps->shape[0] != C || cstate->shape[0] != C || cstate->shape[1] != K - 1) return IK_ERR_SHAPE;
    if (dk == 0 || dv == 0 || Hv == 0 || C <= Hv * dv || (C - Hv * dv) % (2 * dk) != 0) return IK_ERR_SHAPE;
    const int64_t Hk = (C - Hv * dv) / (2 * dk);
    if (Hv % Hk != 0 || o->shape[0] != T || state->shape[0] != Hv || state->shape[1] != dv ||
        a->shape[0] != T || a->shape[1] != Hv || b->shape[0] != T || b->shape[1] != Hv || a_log->shape[0] != Hv ||
        dt_bias->shape[0] != Hv)
        return IK_ERR_SHAPE;
    if (dk != 128 || K != 4 || T > 32) return IK_ERR_UNSUPPORTED;
    if (((uintptr_t)taps->data & 7) != 0) return IK_ERR_STRIDE;  // the packed tap loads
    if (T == 0) return IK_OK;
    constexpr int SPLIT = 2, ROWS = 32 / SPLIT;
    const dim3 grid((unsigned)((dv + ROWS - 1) / ROWS), (unsigned)Hv);
    // one token prefetched (PD = 1), whatever INTARSIA_HIP_GDN_PD says for gdn_delta_rule_bf16_v2: the prefetch depth does not
    // change a bit (v2's own claim, and tests/fusion_p2.rs runs the unfused side at PD 1, 2, 4), and at PD 2 / 4 this kernel
    // spills (255 / 256 VGPRs, 52 / 160 bytes of scratch per lane; PD 1: 255 VGPRs, no scratch; gfx1201, hipcc -O3)
    hipStream_t s = IK_STREAM(stream);
    const bool noslot = env_int("INTARSIA_HIP_GDN_CD_NOSLOT") == 1;  // read at every launch (test seam)
    const int slot = (Hk <= kCdSyncMaxHk && !noslot) ? gdn_cd_sync_slot(s) : -1;
    const int skew = env_int("INTARSIA_HIP_GDN_CD_SKEW");
    gdn_conv_delta_kernel<128, SPLIT, 1, 4><<<grid, 32, 0, s>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)taps->data, (const ik_bf16*)a->data, (const ik_bf16*)b->data,
        (const ik_bf16*)a_log->data, (const ik_bf16*)dt_bias->data, eps, (ik_bf16*)o->data, (float*)state->data,
        (float*)cstate->data, T, (int)Hk, (int)Hv, (int)dv, slot, skew);
    if (slot >= 0) return ik_launch_status();
    IK_TRY(ik_launch_status());
    const int threads = 128;
    gdn_cd_state_kernel<<<(unsigned)((C + threads - 1) / threads), threads, 0, s>>>((const ik_bf16*)x->data,
                                                                                   (float*)cstate->data, T, (int)C, (int)K);
    return ik_launch_status();
}

}  // extern "C"
