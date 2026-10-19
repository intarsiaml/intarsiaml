// moe_block (Op::MoeBlock, core-op fusion P2', docs/results/2026-10-19-p2prime-moeblock.md): the whole MoE block after the
// router logits in ONE launch, bitwise the default decode chain it replaces on HIP (T * K <= 16):
//   moe_topk_bf16_v2 -> moe_experts_bf16_v2 (gate_up + down/reduce) -> matmul_nvfp4_glu_gemv (or, at T > 1, the gate and up
//   nvfp4 GEMVs + silu_glu) -> shared_down_combine_nvfp4_gemv
// OPT-IN on HIP (INTARSIA_HIP_MOE_BLOCK=1, crates/intarsia-backend-hip/src/quantmoe.rs); off, the chain above runs.
//
// One grid, three role classes, the role taken from a ticket (thread 0: atomic add on sync[2] when the block starts), not
// from blockIdx, in dependency order:
//   tickets [0, NS)            shared glu roles: nvfp4_glu_kernel's blocks (token t = b / GS, row tile b % GS), writing the
//                              shared activation sa [T, IS] (scratch);
//   tickets [NS, NS + NG)      routed gate_up roles: moe_experts_p1_v3_kernel's gate_up roles (pair p = b / GY, row tile
//                              b % GY), writing h [T * K, I] (scratch); each wave recomputes its token's top-k
//                              (moe_topk_v2_kernel's selection, statement for statement) up to the pair's rank instead of
//                              reading ids;
//   tickets [NS + NG, + ND)    down roles: moe_experts_p1_v3_kernel's down + reduce roles (token t = b / GD, row tile
//                              b % GD), each wave with its token's full top-k (ids AND weights, moe_topk_v2_kernel's
//                              arithmetic), and as the epilogue of output row r: shared_down_combine_kernel<1>'s shared-down
//                              row r on sa and its combine on the routed row rounded to bf16 as the routed store rounds it.
// The only cross-block wait: a down role's thread 0 spins (acquire, agent) until sync[0] >= NS + NG (every shared and every
// routed producer has fenced and added 1, release), the block barrier, then every wave fences (acquire, agent) before it
// reads h or sa. Top-k has no edge (recomputed by every reader), and the shared-down + combine is row-local (P2a).
// Deadlock-freedom, for ANY dispatch order: a block takes its ticket only once it runs, so when a down role (ticket >=
// NS + NG) exists, every producer ticket is already held by a block that is running or done, and producers never wait;
// they all finish, sync[0] reaches NS + NG, and every down role proceeds. (The rocPRIM / CUB tile-ticket argument, v3's,
// unchanged: the chain is acyclic and tickets follow it.) The last down role puts the three counters back to 0, so the next
// launch on the stream (it starts after this one ends) finds them zeroed. Counters: one 64-byte slot of g_moe_block_sync
// per (device, stream); no free slot (more than 16 streams) -> IK_ERR_UNSUPPORTED and the caller runs the chain.
// Every expression is the one of the kernel it comes from, in the same order (comments name them), so the output bits are
// the chain's; tests/moe_block.rs checks that bitwise on the device, plus the sentinel stress of every edge.
#include "ik_hip_abi.h"
#include "ik_quant.h"
#include <math.h>
#include <mutex>
#include <stdlib.h>
#include <utility>
#include <vector>

namespace {

constexpr int kRowWarps = 8;            // moe.hip's (gate_up / down roles) and nvfp4.hip's kWarps (shared glu)
constexpr int kMaxK = 8;                // moe.hip kP1MaxK
constexpr int kTopkMaxPerLane = 16;     // moe.hip: E <= 512
constexpr int kGluPre = 4;              // nvfp4.hip kPre (nvfp4_glu_kernel's window)
// shared_down_combine_kernel requests kPre = 4 lane-iterations before its table barrier; the fused down role requests only the
// first (all of them at IS <= 512) before its wait and loads the rest as used: the same iterations in the same order, fewer
// registers held across the wait (occupancy)
constexpr int kSdcWin = 1;
constexpr int kSyncSlots = 16;
constexpr int kSyncStride = 16;         // unsigned ints: one 64-byte line per slot
__device__ unsigned int g_moe_block_sync[kSyncSlots * kSyncStride];

IK_DEVI float round_bf16(float v) { return ik_round_bf16(v); }

// moe.hip topk_key16 / topk_val (moe_topk_v2_kernel)
IK_DEVI uint32_t topk_key16(uint32_t bits) {
    if ((bits & 0x7f80u) == 0x7f80u && (bits & 0x7fu)) return 0u;  // NaN: below every number
    return (bits & 0x8000u) ? (~bits & 0xffffu) : (bits | 0x8000u);
}
IK_DEVI float topk_val(uint32_t key16) {
    const uint32_t bits = (key16 & 0x8000u) ? (key16 ^ 0x8000u) : (~key16 & 0xffffu);
    return __uint_as_float(bits << 16);
}

// moe_topk_v2_kernel for one token in one wave: rounds 0..R-1 of the selection (R = K for the weights, kk + 1 for one id).
// id[k] / val[k] are wave-uniform (the warp argmax leaves every lane with the round's key). WANT_W: also the softmax
// weights w[k] = round_bf16(expf(val[k] - mx) / denom), the expression moe_topk_v2_kernel's lane k stores.
// NPL = keys per lane: 8 when E <= 256, else 16 (moe_topk_v2_kernel's kTopkMaxPerLane): the slots j >= 8 hold e >= 256,
// which do not exist when E <= 256 (key 0, never the maximum: a real expert's key has a non-zero low half), so the selection
// and the softmax terms are the same; fewer registers. The softmax over all experts (norm = 0) re-reads the logits instead
// of keeping them in registers: the same values, added in the same order.
template <bool WANT_W, int NPL>
IK_DEVI void wave_topk(const ik_bf16* __restrict__ lrow, int E, int K, int R, int norm, int lane, int (&id)[kMaxK],
                       float (&w)[kMaxK]) {
    uint32_t key[NPL];
#pragma unroll
    for (int j = 0; j < NPL; ++j) {
        const int e = lane + IK_WARP * j;
        const ik_bf16 b = e < E ? lrow[e] : ik_bf16{0xff80u};  // -inf
        key[j] = e < E ? ((topk_key16(b.bits) << 16) | (0xffffu - (uint32_t)e)) : 0u;
    }
    float val[kMaxK];
    float mx = 0.f, denom = 0.f;
#pragma unroll
    for (int kk = 0; kk < kMaxK; ++kk) {
        id[kk] = -1;
        val[kk] = 0.f;
        if (kk >= R) continue;
        uint32_t best = key[0];
#pragma unroll
        for (int j = 1; j < NPL; ++j) best = max(best, key[j]);
#pragma unroll
        for (int o = IK_WARP / 2; o > 0; o >>= 1) best = max(best, (uint32_t)__shfl_xor((int)best, o, IK_WARP));
        // every lane holds the same key after the butterfly: keep it in a scalar register (the expert's weight addresses
        // and scales are then scalar, as when the chain's kernels load ids from memory)
        best = (uint32_t)__builtin_amdgcn_readfirstlane((int)best);
        const int bi = (int)(0xffffu - (best & 0xffffu));
        const float bv = topk_val(best >> 16);
        if (kk == 0) mx = bv;
        if ((bi & (IK_WARP - 1)) == lane) {
#pragma unroll
            for (int j = 0; j < NPL; ++j)
                if (j == (bi >> 5)) key[j] = 0u;
        }
        id[kk] = bi;
        val[kk] = bv;
        denom += expf(bv - mx);
    }
    if (WANT_W) {
        if (!norm) {  // softmax over all experts
            float part = 0.f;
#pragma unroll
            for (int j = 0; j < NPL; ++j) {
                const int e = lane + IK_WARP * j;
                if (e < E) part += expf(ik_b2f(lrow[e]) - mx);
            }
            denom = ik_warp_sum(part);
        }
#pragma unroll
        for (int k = 0; k < kMaxK; ++k) {
            const float ww = expf(val[k] - mx) / denom;
            w[k] = k < K ? __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(round_bf16(ww)))) : 0.f;
        }
    }
    (void)K;
}

struct BlockArgs {
    const ik_bf16* x;       // n [T, H]
    const ik_bf16* logits;  // [T, E]
    const ik_bf16* sl;      // shared gate logit [T, 1]
    const uint8_t* guw;     // routed experts (moe_experts_bf16 layout)
    const uint8_t* gus;
    const uint8_t* dnw;
    const uint8_t* dns;
    const float* gscales;
    const uint8_t* sgw;     // shared gate [IS, H/2], [IS, H/16], [1]
    const uint8_t* sgs;
    const float* sgg;
    const uint8_t* suw;     // shared up
    const uint8_t* sus;
    const float* sug;
    const uint8_t* sdw;     // shared down [H, IS/2], [H, IS/16], [1]
    const uint8_t* sds;
    const float* sdg;
    ik_bf16* h;             // scratch [T * K, I]
    ik_bf16* sa;            // scratch [T, IS]
    ik_bf16* out;           // [T, H]
    int E, H, I, IS, K, norm;
    int NS, NG, ND, slot, skew;
};

template <int PRE, int NPL>
__global__ void __launch_bounds__(kRowWarps * IK_WARP) __attribute__((amdgpu_waves_per_eu(12))) moe_block_kernel(const BlockArgs a) {
    __shared__ float lut[16];
    __shared__ int ticket;
    unsigned int* sync = g_moe_block_sync + a.slot * kSyncStride;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int E = a.E, H = a.H, I = a.I, IS = a.IS, K = a.K, NS = a.NS, NG = a.NG, ND = a.ND;
    // test seam (bit 21, INTARSIA_HIP_MOE_BLOCK_SCRAMBLE=1; default off): a block sleeps (blockIdx % 7) * SKEW before it
    // takes its ticket, so tickets are drawn in an order unrelated to blockIdx (the dispatch-order test of deadlock-freedom)
    if ((a.skew >> 21) & 1)
        for (int i = 0; i < (int)(blockIdx.x % 7) * (a.skew & 0xffff); ++i) __builtin_amdgcn_s_sleep(8);
    if (threadIdx.x == 0) ticket = (int)__hip_atomic_fetch_add(sync + 2, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const int bid = ticket;  // the role
    if (bid < NS + NG) {
        // test seam (INTARSIA_HIP_MOE_BLOCK_SKEW = sleeps, default 0, bits 0..15): one producer role starts late, chosen by
        // INTARSIA_HIP_MOE_BLOCK_SKEW_ROLE (bits 18..20): 0 the last routed role, 1 the last shared role, 2 the last routed
        // role of token 0, 3 the last shared role of token 0, 4 the last routed role of every token, 5 both of token 0's;
        // bit 16 (SKEW_ALL=1): every producer starts late, by 1 to 4 times the skew (tests/moe_block.rs sentinel stress).
        const int sk = a.skew & 0xffff, sel = (a.skew >> 18) & 7;
        const int GS = (IS + kRowWarps - 1) / kRowWarps, GY = (I + kRowWarps - 1) / kRowWarps;
        const int tpr = K * GY;  // routed roles per token
        const bool rl = bid >= NS;
        const int rb = bid - NS;  // routed role index
        bool late = false;
        if (sel == 0) late = bid == NS + NG - 1;
        else if (sel == 1) late = bid == NS - 1;
        else if (sel == 2) late = rl && rb == tpr - 1;
        else if (sel == 3) late = bid == GS - 1;
        else if (sel == 4) late = rl && (rb + 1) % tpr == 0;
        else late = (rl && rb == tpr - 1) || bid == GS - 1;
        const int nsleep = ((a.skew >> 16) & 1) ? sk * (1 + bid % 4) : (late ? sk : 0);
        for (int i = 0; i < nsleep; ++i) __builtin_amdgcn_s_sleep(127);
        if (!rl) {
            // ---- shared glu: nvfp4_glu_kernel with blockIdx = role % GS, x = n[t] ----
            const int t = bid / GS;
            const int n = (bid % GS) * kRowWarps + warp;
            const bool live = n < IS;
            const int nn = live ? n : 0;
            const ik_bf16* x = a.x + (int64_t)t * H;
            const uint2* wr[2] = {reinterpret_cast<const uint2*>(a.sgw + (int64_t)nn * (H / 2)),
                                  reinterpret_cast<const uint2*>(a.suw + (int64_t)nn * (H / 2))};
            const uint8_t* sr[2] = {a.sgs + (int64_t)nn * (H / 16), a.sus + (int64_t)nn * (H / 16)};
            const int nu = H / 16;
            uint2 wv[2][kGluPre];
            uint32_t sb[2][kGluPre];
            uint4 xv[kGluPre][2];
#pragma unroll
            for (int j = 0; j < kGluPre; ++j) {
                const int u = lane + IK_WARP * j;
                const bool ok = u < nu;
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    wv[r][j] = ok ? ik_ldcs(wr[r] + u) : make_uint2(0, 0);
                    sb[r][j] = ok ? sr[r][u] : 0u;
                }
                const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)(ok ? u : 0) * 16);
                xv[j][0] = xp[0];
                xv[j][1] = xp[1];
            }
            if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
            __syncthreads();
            if (live) {
                float acc[2] = {0.f, 0.f};
                auto step = [&](int r, const uint2& wq, uint32_t sbyte, const uint4 (&xq)[2]) {
                    const float sc = ik_e4m3_to_f(sbyte);
                    float wf[16];
                    ik_fp4x8_to_f32(wq.x, lut, wf);
                    ik_fp4x8_to_f32(wq.y, lut, wf + 8);
                    float xf[16];
                    ik_unpack8(xq[0], xf);
                    ik_unpack8(xq[1], xf + 8);
                    float s = 0.f;
#pragma unroll
                    for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
                    acc[r] = fmaf(s, sc, acc[r]);
                };
#pragma unroll
                for (int j = 0; j < kGluPre; ++j) {
                    if (lane + IK_WARP * j >= nu) continue;
                    step(0, wv[0][j], sb[0][j], xv[j]);
                    step(1, wv[1][j], sb[1][j], xv[j]);
                }
                for (int u = lane + IK_WARP * kGluPre; u < nu; u += IK_WARP) {  // beyond the window (H > 2048)
                    uint4 xq[2];
                    const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)u * 16);
                    xq[0] = xp[0];
                    xq[1] = xp[1];
                    step(0, ik_ldcs(wr[0] + u), sr[0][u], xq);
                    step(1, ik_ldcs(wr[1] + u), sr[1][u], xq);
                }
                const float g = ik_b2f(ik_f2b(ik_warp_sum(acc[0]) * a.sgg[0]));  // what the gate GEMV stores
                const float uu = ik_b2f(ik_f2b(ik_warp_sum(acc[1]) * a.sug[0]));  // what the up GEMV stores
                if (lane == 0) a.sa[(int64_t)t * IS + n] = ik_f2b(ik_silu(g) * uu);
            }
            __threadfence();  // shared producer: release half
            __syncthreads();
            if (threadIdx.x == 0) __hip_atomic_fetch_add(sync, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            return;
        }
        // ---- routed gate_up: moe_experts_p1_v3_kernel's gate_up role (moe_gate_up_p1_v2_kernel<false, PRE>) ----
        const int p = rb / GY;
        const int t = p / K;
        const ik_bf16* xr = a.x + (int64_t)t * H;
        int tid[kMaxK];
        float tw[kMaxK];
        wave_topk<false, NPL>(a.logits + (int64_t)t * E, E, K, p % K + 1, a.norm, lane, tid, tw);
        int e = -1;
#pragma unroll
        for (int k = 0; k < kMaxK; ++k)
            if (k == p % K) e = tid[k];  // the pair's expert: rank p % K of token t (ids[p] in the chain)
        const bool ok = e >= 0 && e < E;  // block-uniform
        const int r = (rb % GY) * kRowWarps + warp;
        const bool live = ok && r < I;
        const int ee = ok ? e : 0, rr = r < I ? r : 0;
        const uint2* wg = reinterpret_cast<const uint2*>(a.guw + ((int64_t)ee * 2 * I + rr) * (H / 2));
        const uint2* wu = reinterpret_cast<const uint2*>(a.guw + ((int64_t)ee * 2 * I + I + rr) * (H / 2));
        const uint8_t* sg = a.gus + ((int64_t)ee * 2 * I + rr) * (H / 16);
        const uint8_t* su = a.gus + ((int64_t)ee * 2 * I + I + rr) * (H / 16);
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
            const float g = round_bf16(ik_warp_sum(ag) * a.gscales[e * 3 + 0]);
            const float uu = round_bf16(ik_warp_sum(au) * a.gscales[e * 3 + 1]);
            const float ac = round_bf16(g / (1.f + expf(-g)));
            if (lane == 0) a.h[(int64_t)p * I + r] = ik_f2b(ac * uu);
        }
        __threadfence();  // routed producer: release half
        __syncthreads();
        if (threadIdx.x == 0) __hip_atomic_fetch_add(sync, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        return;
    }

    // ---- down + reduce (moe_experts_p1_v3_kernel's down role) + shared down + combine (shared_down_combine_kernel<1>) ----
    const int bd = bid - NS - NG;
    const int GD = (H + kRowWarps - 1) / kRowWarps;
    const int t = bd / GD;
    const int r = (bd % GD) * kRowWarps + warp;
    const bool rv = r < H;
    const int rr = rv ? r : 0;
    const int nu = I / 16;
    const ik_bf16* hrow = a.h + (int64_t)t * K * I;  // the token's K intermediate rows [K][I]
    const ik_bf16* sx = a.sa + (int64_t)t * IS;      // the token's shared activation [IS]
    int tid[kMaxK];
    float wts[kMaxK];
    wave_topk<true, NPL>(a.logits + (int64_t)t * E, E, K, K, a.norm, lane, tid, wts);
    const uint2* wd[kMaxK];
    const uint8_t* sd[kMaxK];
    float gd[kMaxK], acc[kMaxK];
    uint2 wv[kMaxK];
    uint32_t sb[kMaxK];
#pragma unroll
    for (int k = 0; k < kMaxK; ++k) {
        const int id = k < K ? tid[k] : -1;
        const bool okk = id >= 0 && id < E;
        const int64_t row = okk ? (int64_t)id * H + rr : 0;
        wd[k] = okk ? reinterpret_cast<const uint2*>(a.dnw + row * (I / 2)) : nullptr;
        sd[k] = okk ? a.dns + row * (I / 16) : nullptr;
        gd[k] = okk ? a.gscales[id * 3 + 2] : 0.f;
        acc[k] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kMaxK; ++k) {  // iteration u = lane of every expert, requested before the wait
        const bool in = wd[k] && lane < nu;
        wv[k] = in ? ik_ldcs(wd[k] + lane) : make_uint2(0, 0);
        sb[k] = in ? sd[k][lane] : 0u;
    }
    // the shared-down row's window (weights and block scales; its activations sa are read after the wait)
    const uint2* swr = reinterpret_cast<const uint2*>(a.sdw + (int64_t)rr * (IS / 2));
    const uint8_t* ssr = a.sds + (int64_t)rr * (IS / 16);
    const int snu = IS / 16;
    uint2 swv[kSdcWin];
    uint32_t ssb[kSdcWin];
#pragma unroll
    for (int j = 0; j < kSdcWin; ++j) {
        const int u = lane + IK_WARP * j;
        const bool okj = u < snu;
        swv[j] = okj ? ik_ldcs(swr + u) : make_uint2(0, 0);
        ssb[j] = okj ? ssr[u] : 0u;
    }
    const float gl = ik_b2f(a.sl[t]);
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    // test seam (bit 17, INTARSIA_HIP_MOE_BLOCK_PREREAD=1; default off): every wave reads the h AND sa lines it will use
    // before the wait, so they sit in this block's caches when the producers store them; only the acquire below makes the
    // later reads see the stores (tests/moe_block.rs sentinel stress)
    if (((a.skew >> 17) & 1) && rv) {
        uint32_t tt = 0;
        for (int k = 0; k < kMaxK; ++k) {
            if (!wd[k]) continue;
            for (int u = lane; u < nu; u += IK_WARP) {
                const uint4* hp = reinterpret_cast<const uint4*>(hrow + (int64_t)k * I + u * 16);
                const uint4 p0 = hp[0], p1 = hp[1];
                tt ^= p0.x ^ p0.y ^ p0.z ^ p0.w ^ p1.x ^ p1.y ^ p1.z ^ p1.w;
            }
        }
        for (int u = lane; u < snu; u += IK_WARP) {
            const uint4* sp = reinterpret_cast<const uint4*>(sx + (int64_t)u * 16);
            const uint4 p0 = sp[0], p1 = sp[1];
            tt ^= p0.x ^ p0.y ^ p0.z ^ p0.w ^ p1.x ^ p1.y ^ p1.z ^ p1.w;
        }
        __asm__ volatile("" ::"v"(tt));
    }
    if (threadIdx.x == 0) {
        while (__hip_atomic_load(sync, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)(NS + NG)) __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");  // every wave: h and sa are read only after this
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
            for (int k = 0; k < kMaxK; ++k) {
                if (!wd[k]) continue;
                uint4 h0, h1;
                hload(k < K ? k : 0, lane, h0, h1);
                step(k, wv[k], sb[k], h0, h1);
            }
        }
        for (int u = lane + IK_WARP; u < nu; u += IK_WARP) {
            uint2 wq[kMaxK];
            uint32_t sq[kMaxK];
#pragma unroll
            for (int k = 0; k < kMaxK; ++k) {
                wq[k] = wd[k] ? ik_ldcs(wd[k] + u) : make_uint2(0, 0);
                sq[k] = wd[k] ? sd[k][u] : 0u;
            }
#pragma unroll
            for (int k = 0; k < kMaxK; ++k) {
                if (!wd[k]) continue;
                uint4 h0, h1;
                hload(k, u, h0, h1);
                step(k, wq[k], sq[k], h0, h1);
            }
        }
        float o = 0.f;
#pragma unroll
        for (int k = 0; k < kMaxK; ++k) {
            if (!wd[k]) continue;
            const float y = round_bf16(ik_warp_sum(acc[k]) * gd[k]);
            o += round_bf16(y * wts[k]);
        }
        const float routed = ik_round_bf16(o);  // what the routed store (ik_store bf16) keeps
        // shared_down_combine_kernel<1>: the shared-down row r on sa[t], then moe_combine's epilogue
        float sacc = 0.f;
        auto sstep = [&](const uint2& wq, uint32_t sbyte, const uint4& x0, const uint4& x1) {
            const float sc = ik_e4m3_to_f(sbyte);
            float wf[16];
            ik_fp4x8_to_f32(wq.x, lut, wf);
            ik_fp4x8_to_f32(wq.y, lut, wf + 8);
            float xf[16];
            ik_unpack8(x0, xf);
            ik_unpack8(x1, xf + 8);
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
            sacc = fmaf(s, sc, sacc);
        };
#pragma unroll
        for (int j = 0; j < kSdcWin; ++j) {
            const int u = lane + IK_WARP * j;
            if (u >= snu) continue;
            const uint4* xp = reinterpret_cast<const uint4*>(sx + (int64_t)u * 16);
            sstep(swv[j], ssb[j], xp[0], xp[1]);
        }
        for (int u = lane + IK_WARP * kSdcWin; u < snu; u += IK_WARP) {  // IS > 2048: the remaining iterations
            const uint4* xp = reinterpret_cast<const uint4*>(sx + (int64_t)u * 16);
            sstep(ik_ldcs(swr + u), ssr[u], xp[0], xp[1]);
        }
        const float ssum = ik_warp_sum(sacc);
        if (lane == 0) {
            const float so = ik_round_bf16(ssum * a.sdg[0]);  // what the shared-down GEMV stores
            const float sg = ik_round_bf16(1.f / (1.f + expf(-gl)));
            const float sv = ik_round_bf16(sg * so);
            a.out[(int64_t)t * H + r] = ik_f2b(ik_round_bf16(routed + sv));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = __hip_atomic_fetch_add(sync + 1, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (done == (unsigned)ND - 1) {  // every down role is past its wait: reset for the next launch on this stream
            __hip_atomic_store(sync, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(sync + 1, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(sync + 2, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// The g_moe_block_sync slot of the calling thread's (device, stream); -1 when all kSyncSlots are taken.
int block_sync_slot(hipStream_t s) {
    static std::mutex mu;
    static std::vector<std::pair<int, hipStream_t>> owners;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) return -1;
    std::lock_guard<std::mutex> g(mu);
    for (size_t i = 0; i < owners.size(); ++i)
        if (owners[i].first == dev && owners[i].second == s) return (int)i;
    if ((int)owners.size() >= kSyncSlots) return -1;
    owners.emplace_back(dev, s);
    return (int)owners.size() - 1;
}

// INTARSIA_HIP_MOE_PRE (moe.hip moe_pre): gate_up lane-iterations requested before the barrier (1, 2 or 4); no bit changes
int block_pre() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_HIP_MOE_PRE");
        const int x = e ? atoi(e) : 4;
        return (x == 1 || x == 2) ? x : 4;
    }();
    return v;
}

inline int64_t r256(int64_t v) { return (v + 255) & ~int64_t(255); }

}  // namespace

extern "C" {

// Scratch bytes of ik_moe_block_bf16: h [T * K, I] bf16 then sa [T, IS] bf16, each 256-byte aligned.
int64_t ik_moe_block_scratch_bytes(int64_t T, int64_t K, int64_t I, int64_t IS) { return r256(T * K * I * 2) + r256(T * IS * 2); }

// in: the 17 inputs of Op::MoeBlock in order, an array of ik_tensor: x [T, H] bf16, logits [T, E] bf16, sl [T, 1] bf16, the five routed expert tensors (moe_experts_bf16's), shared gate / up
// (w [IS, H/2] u8, bs [IS, H/16] e4m3, gs [1] f32) and down (w [H, IS/2], bs [H, IS/16], gs [1]); out [T, H] bf16.
// 1 <= T * K <= 16, K <= 8, E <= 512, H / I / IS multiples of 16 (the shapes the default chain serves with its decode
// kernels); IK_ERR_SHAPE otherwise, IK_ERR_UNSUPPORTED without a free counter slot.
int ik_moe_block_bf16(const ik_tensor* in, int top_k, int norm, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    const ik_tensor *x = &in[0], *logits = &in[1], *sl = &in[2], *guw = &in[3], *gus = &in[4], *dnw = &in[5], *dns = &in[6],
                    *gsc = &in[7];
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(logits, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(sl, IK_DTYPE_BF16, 2, 2));
    IK_TRY(ik_check_a(guw, IK_DTYPE_U8, 3, 8));
    IK_TRY(ik_check_a(gus, IK_DTYPE_FP8_E4M3, 3, 1));
    IK_TRY(ik_check_a(dnw, IK_DTYPE_U8, 3, 8));
    IK_TRY(ik_check_a(dns, IK_DTYPE_FP8_E4M3, 3, 1));
    IK_TRY(ik_check_a(gsc, IK_DTYPE_F32, 2, 4));
    for (int j = 0; j < 3; ++j) {
        IK_TRY(ik_check_a(&in[8 + 3 * j], IK_DTYPE_U8, 2, 8));
        IK_TRY(ik_check_a(&in[9 + 3 * j], IK_DTYPE_FP8_E4M3, 2, 1));
        IK_TRY(ik_check_a(&in[10 + 3 * j], IK_DTYPE_F32, 1, 4));
    }
    IK_TRY(ik_check_a(out, IK_DTYPE_BF16, 2, 2));
    const int64_t T = x->shape[0], H = x->shape[1], E = logits->shape[1], K = top_k, I = guw->shape[1] / 2;
    const int64_t IS = in[8].shape[0];
    const ik_tensor *sgw = &in[8], *sgs = &in[9], *sgg = &in[10], *suw = &in[11], *sus = &in[12], *sug = &in[13],
                    *sdw = &in[14], *sds = &in[15], *sdg = &in[16];
    if (logits->shape[0] != T || sl->shape[0] != T || sl->shape[1] != 1 || out->shape[0] != T || out->shape[1] != H)
        return IK_ERR_SHAPE;
    if (H % 16 || I % 16 || IS % 16 || I == 0 || IS == 0 || H == 0) return IK_ERR_SHAPE;
    if (guw->shape[0] != E || guw->shape[1] != 2 * I || guw->shape[2] != H / 2 || gus->shape[0] != E ||
        gus->shape[1] != 2 * I || gus->shape[2] != H / 16 || dnw->shape[0] != E || dnw->shape[1] != H ||
        dnw->shape[2] != I / 2 || dns->shape[0] != E || dns->shape[1] != H || dns->shape[2] != I / 16 ||
        gsc->shape[0] != E || gsc->shape[1] != 3)
        return IK_ERR_SHAPE;
    for (const ik_tensor* w : {sgw, suw})
        if (w->shape[0] != IS || w->shape[1] != H / 2) return IK_ERR_SHAPE;
    for (const ik_tensor* b : {sgs, sus})
        if (b->shape[0] != IS || b->shape[1] != H / 16) return IK_ERR_SHAPE;
    if (sdw->shape[0] != H || sdw->shape[1] != IS / 2 || sds->shape[0] != H || sds->shape[1] != IS / 16) return IK_ERR_SHAPE;
    for (const ik_tensor* g : {sgg, sug, sdg})
        if (g->shape[0] != 1) return IK_ERR_SHAPE;
    if (T < 1 || K < 1 || K > kMaxK || K > E || E > IK_WARP * kTopkMaxPerLane || T * K > 16) return IK_ERR_SHAPE;
    const int64_t need = ik_moe_block_scratch_bytes(T, K, I, IS);
    if (!scratch || !scratch->data || scratch->bytes < need || (reinterpret_cast<uintptr_t>(scratch->data) & 15u))
        return IK_ERR_INVALID_ARG;
    hipStream_t s = IK_STREAM(stream);
    const int slot = block_sync_slot(s);
    if (slot < 0) return IK_ERR_UNSUPPORTED;
    // test seams, read at every launch (unset = off): SKEW (bits 0..15), SKEW_ALL (16), PREREAD (17), SKEW_ROLE (18..20),
    // SCRAMBLE (21)
    int skew = 0;
    if (const char* v = getenv("INTARSIA_HIP_MOE_BLOCK_SKEW")) skew = atoi(v) & 0xffff;
    if (const char* v = getenv("INTARSIA_HIP_MOE_BLOCK_SKEW_ALL"); v && v[0] == '1') skew |= 1 << 16;
    if (const char* v = getenv("INTARSIA_HIP_MOE_BLOCK_PREREAD"); v && v[0] == '1') skew |= 1 << 17;
    if (const char* v = getenv("INTARSIA_HIP_MOE_BLOCK_SKEW_ROLE")) skew |= (atoi(v) & 7) << 18;
    if (const char* v = getenv("INTARSIA_HIP_MOE_BLOCK_SCRAMBLE"); v && v[0] == '1') skew |= 1 << 21;
    BlockArgs a;
    a.x = (const ik_bf16*)x->data;
    a.logits = (const ik_bf16*)logits->data;
    a.sl = (const ik_bf16*)sl->data;
    a.guw = (const uint8_t*)guw->data;
    a.gus = (const uint8_t*)gus->data;
    a.dnw = (const uint8_t*)dnw->data;
    a.dns = (const uint8_t*)dns->data;
    a.gscales = (const float*)gsc->data;
    a.sgw = (const uint8_t*)sgw->data;
    a.sgs = (const uint8_t*)sgs->data;
    a.sgg = (const float*)sgg->data;
    a.suw = (const uint8_t*)suw->data;
    a.sus = (const uint8_t*)sus->data;
    a.sug = (const float*)sug->data;
    a.sdw = (const uint8_t*)sdw->data;
    a.sds = (const uint8_t*)sds->data;
    a.sdg = (const float*)sdg->data;
    unsigned char* base = (unsigned char*)scratch->data;
    a.h = (ik_bf16*)base;
    a.sa = (ik_bf16*)(base + r256(T * K * I * 2));
    a.out = (ik_bf16*)out->data;
    a.E = (int)E;
    a.H = (int)H;
    a.I = (int)I;
    a.IS = (int)IS;
    a.K = (int)K;
    a.norm = norm ? 1 : 0;
    a.NS = (int)(T * ((IS + kRowWarps - 1) / kRowWarps));
    a.NG = (int)(T * K * ((I + kRowWarps - 1) / kRowWarps));
    a.ND = (int)(T * ((H + kRowWarps - 1) / kRowWarps));
    a.slot = slot;
    a.skew = skew;
    const unsigned grid = (unsigned)(a.NS + a.NG + a.ND);
    const int pre = block_pre();
#define IK_MB(PRE)                                                                                  \
    if (E <= 8 * IK_WARP) moe_block_kernel<PRE, 8><<<grid, kRowWarps * IK_WARP, 0, s>>>(a);           \
    else moe_block_kernel<PRE, kTopkMaxPerLane><<<grid, kRowWarps * IK_WARP, 0, s>>>(a)
    if (pre == 1) IK_MB(1);
    else if (pre == 2) IK_MB(2);
    else IK_MB(4);
#undef IK_MB
    return ik_launch_status();
}

}  // extern "C"
