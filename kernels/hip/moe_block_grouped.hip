// moe_block_grouped (Op::MoeBlock for 5..16 rows, lane l7, docs/results/2026-10-25-l7-moe-block-rows.md): the whole MoE block of a batched
// decode step after the router and shared-gate logits, row-exact: token t of the output is BITWISE the T = 1 moe_block_bf16 launch on token t
// alone (crates/intarsia-backend-hip/tests/moe_block.rs).
//
// The fused moe_block_bf16_rows reads every token's experts separately and runs the shared expert once per token, so past 4 rows it loses to
// the unfused chain (hr3 s4); the unfused chain pays 21 launches per layer at 16 rows (the shared expert alone is 12: gate, up and down each
// run the one-row-exact nvfp4 GEMV in groups of at most 4 rows). This entry runs, in stream order:
//   1. ik_moe_topk_bf16_v2                the chain's top-k (ids, weights)
//   2. ik_moe_experts_bf16_rows           the chain's routed experts: tokens grouped by expert, each expert's weights read once per chunk of
//                                         8 pairs, per-pair arithmetic independent of the grouping, fixed k order in the reduction (moe.hip)
//   3. shared_glu_rows_kernel             shared gate + up + silu for ALL rows (blocks over output rows x groups of 4 tokens)
//   4. shared_down_combine_rows_kernel    shared down + moe_combine's epilogue for all rows in one pass
// 21 launches become 7 (the two logit GEMVs stay in the model). Steps 3 and 4 follow moe_block.hip's copies of nvfp4_glu_kernel and
// shared_down_combine_kernel<1> for every row: a warp owns output row n, lane u = lane, lane + 32, .. of K / 16, one fma chain per 16-element
// block, the block scale applied per block, ik_warp_sum, the global scale after the sum, the same bf16 roundings. A row's arithmetic does not
// depend on the other rows or on T (the accumulators are per row), so the one pass gives each row the bits a lone row gets.
#include "ik_dq_gemm.h"
#include "ik_hip_abi.h"

namespace {

constexpr int kWarps = 8;
constexpr int kPre = 4;       // lane-iterations whose weights are requested before the table barrier (nvfp4.hip kPre)
constexpr int kRowsMax = 16;  // tokens per launch
constexpr int kRowsGrp = 4;   // tokens per block of shared_glu_rows_kernel (blockIdx.y selects the group of tokens): 28.9 -> 13.9 us at 16 rows
// shared_down_combine_rows_kernel keeps all rows in one block: with groups of 4 it measured 25.4 us against 22.3 (rocprofv3, 16 rows, TP=1)
constexpr int kRowsGrpDown = kRowsMax;

// out[m, n] = bf16(silu(g) * u), g / u = what the gate / up GEMV stores for row m (nvfp4_glu_kernel for every row m < T at once)
__global__ void __launch_bounds__(kWarps * IK_WARP)
shared_glu_rows_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ wg, const uint8_t* __restrict__ bsg,
                       const float* __restrict__ gsg, const uint8_t* __restrict__ wu, const uint8_t* __restrict__ bsu,
                       const float* __restrict__ gsu, ik_bf16* __restrict__ out, int T, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    const bool live = n < N;
    const int64_t nn = live ? n : 0;
    const uint2* wr[2] = {reinterpret_cast<const uint2*>(wg + nn * (K / 2)), reinterpret_cast<const uint2*>(wu + nn * (K / 2))};
    const uint8_t* sr[2] = {bsg + nn * (K / 16), bsu + nn * (K / 16)};
    const int nu = K / 16;
    uint2 wv[2][kPre];
    uint32_t sb[2][kPre];
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        const int u = lane + IK_WARP * j;
        const bool ok = u < nu;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            wv[r][j] = ok ? ik_ldcs(wr[r] + u) : make_uint2(0, 0);
            sb[r][j] = ok ? sr[r][u] : 0u;
        }
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    __syncthreads();
    if (!live) return;  // whole warp exits together
    const int m0 = (int)blockIdx.y * kRowsGrp;  // this block's tokens m0 .. m0 + kRowsGrp - 1 (those below T)
    float ag[kRowsGrp], au[kRowsGrp];
#pragma unroll
    for (int m = 0; m < kRowsGrp; ++m) ag[m] = au[m] = 0.f;
    auto step = [&](int u, const uint2& gq, uint32_t gb, const uint2& uq, uint32_t ub) {
        const float scg = ik_e4m3_to_f(gb), scu = ik_e4m3_to_f(ub);
        float fg[16], fu[16];
        ik_fp4x8_to_f32(gq.x, lut, fg);
        ik_fp4x8_to_f32(gq.y, lut, fg + 8);
        ik_fp4x8_to_f32(uq.x, lut, fu);
        ik_fp4x8_to_f32(uq.y, lut, fu + 8);
#pragma unroll
        for (int m = 0; m < kRowsGrp; ++m) {
            if (m0 + m < T) {
                const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)(m0 + m) * K + (int64_t)u * 16);
                float xf[16];
                ik_unpack8(xp[0], xf);
                ik_unpack8(xp[1], xf + 8);
                float pg = 0.f, pu = 0.f;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    pg = fmaf(fg[e], xf[e], pg);
                    pu = fmaf(fu[e], xf[e], pu);
                }
                ag[m] = fmaf(pg, scg, ag[m]);
                au[m] = fmaf(pu, scu, au[m]);
            }
        }
    };
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        if (lane + IK_WARP * j >= nu) continue;
        step(lane + IK_WARP * j, wv[0][j], sb[0][j], wv[1][j], sb[1][j]);
    }
    for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP)  // K > 2048: the remaining iterations, loaded as used
        step(u, ik_ldcs(wr[0] + u), sr[0][u], ik_ldcs(wr[1] + u), sr[1][u]);
    const float g0 = gsg[0], u0 = gsu[0];
#pragma unroll
    for (int m = 0; m < kRowsGrp; ++m) {
        if (m0 + m < T) {
            const float g = ik_b2f(ik_f2b(ik_warp_sum(ag[m]) * g0));  // what the gate GEMV stores
            const float uu = ik_b2f(ik_f2b(ik_warp_sum(au[m]) * u0));  // what the up GEMV stores
            if (lane == 0) out[(int64_t)(m0 + m) * N + n] = ik_f2b(ik_silu(g) * uu);
        }
    }
}

// out[m, n] = bf16(routed[m, n] + bf16(bf16(sigmoid(logit[m])) * so)), so = bf16(x[m, :] @ W[n, :]^T): shared_down_combine_kernel<1> for every
// row m < T at once
__global__ void __launch_bounds__(kWarps * IK_WARP)
shared_down_combine_rows_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                                const float* __restrict__ gs, const ik_bf16* __restrict__ routed,
                                const ik_bf16* __restrict__ logit, ik_bf16* __restrict__ out, int T, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    const bool live = n < N;
    const int64_t nn = live ? n : 0;
    const uint2* wr = reinterpret_cast<const uint2*>(w + nn * (K / 2));
    const uint8_t* sr = bs + nn * (K / 16);
    const int nu = K / 16;
    uint2 wv[kPre];
    uint32_t sb[kPre];
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        const int u = lane + IK_WARP * j;
        const bool ok = u < nu;
        wv[j] = ok ? ik_ldcs(wr + u) : make_uint2(0, 0);
        sb[j] = ok ? sr[u] : 0u;
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    __syncthreads();
    if (!live) return;  // whole warp exits together
    const int m0 = (int)blockIdx.y * kRowsGrpDown;  // this block's tokens m0 .. m0 + kRowsGrpDown - 1 (those below T)
    float acc[kRowsGrpDown];
#pragma unroll
    for (int m = 0; m < kRowsGrpDown; ++m) acc[m] = 0.f;
    auto step = [&](int u, const uint2& wq, uint32_t sbyte) {
        const float sc = ik_e4m3_to_f(sbyte);
        float wf[16];
        ik_fp4x8_to_f32(wq.x, lut, wf);
        ik_fp4x8_to_f32(wq.y, lut, wf + 8);
#pragma unroll
        for (int m = 0; m < kRowsGrpDown; ++m) {
            if (m0 + m < T) {
                const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)(m0 + m) * K + (int64_t)u * 16);
                float xf[16];
                ik_unpack8(xp[0], xf);
                ik_unpack8(xp[1], xf + 8);
                float s = 0.f;
#pragma unroll
                for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
                acc[m] = fmaf(s, sc, acc[m]);
            }
        }
    };
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
        if (lane + IK_WARP * j >= nu) continue;
        step(lane + IK_WARP * j, wv[j], sb[j]);
    }
    for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP) step(u, ik_ldcs(wr + u), sr[u]);
    const float g = gs[0];
#pragma unroll
    for (int m = 0; m < kRowsGrpDown; ++m) {
        if (m0 + m < T) {
            const float s = ik_warp_sum(acc[m]);
            if (lane == 0) {
                const float so = ik_round_bf16(s * g);  // what the down GEMV stores (the shared-expert output)
                const float gl = ik_b2f(logit[m0 + m]);
                const float sg = ik_round_bf16(1.f / (1.f + expf(-gl)));
                const float sv = ik_round_bf16(sg * so);
                const int64_t i = (int64_t)(m0 + m) * N + n;
                out[i] = ik_f2b(ik_round_bf16(ik_b2f(routed[i]) + sv));
            }
        }
    }
}

inline int64_t r256(int64_t v) { return (v + 255) & ~int64_t(255); }

// experts' scratch (moe.hip scratch_layout / intarsia_core::quant::ops::moe_experts_scratch): pairs, offsets, active, h, y
int64_t experts_scratch_bytes(int64_t P, int64_t E, int64_t H, int64_t I) {
    return r256(P * 4) + r256((E + 1) * 4) + r256((E + 1) * 4) + r256(P * I * 2) + r256(P * H * 2);
}

struct Layout {
    int64_t w, ids, routed, sa, ex, total;
};

Layout layout(int64_t T, int64_t K, int64_t E, int64_t H, int64_t I, int64_t IS) {
    Layout l;
    l.w = 0;
    l.ids = l.w + r256(T * K * 4);
    l.routed = l.ids + r256(T * K * 4);
    l.sa = l.routed + r256(T * H * 2);
    l.ex = l.sa + r256(T * IS * 2);
    l.total = l.ex + experts_scratch_bytes(T * K, E, H, I);
    return l;
}

ik_tensor view2(void* p, int dtype, int64_t d0, int64_t d1) {
    ik_tensor t;
    t.data = p;
    t.dtype = dtype;
    t.ndim = 2;
    for (int i = 0; i < IK_MAX_DIMS; ++i) t.shape[i] = t.stride[i] = 1;
    t.shape[0] = d0;
    t.shape[1] = d1;
    t.stride[1] = 1;
    t.stride[0] = d1;
    return t;
}

}  // namespace

extern "C" {

// Scratch bytes of ik_moe_block_bf16_grouped: top-k weights f32 [T, K], ids i32 [T, K], routed bf16 [T, H], shared activation bf16 [T, IS],
// then the routed experts' scratch; every region rounded up to 256 bytes.
int64_t ik_moe_block_grouped_scratch_bytes(int64_t T, int64_t K, int64_t E, int64_t H, int64_t I, int64_t IS) {
    return layout(T, K, E, H, I, IS).total;
}

// in: the 17 inputs of Op::MoeBlock in order (see ik_moe_block_bf16); 1 <= T <= 16, 1 <= K <= 8, E <= 512, H / I / IS multiples of 16.
int ik_moe_block_bf16_grouped(const ik_tensor* in, int top_k, int norm, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    const ik_tensor *x = &in[0], *logits = &in[1], *sl = &in[2], *guw = &in[3], *gus = &in[4], *dnw = &in[5], *dns = &in[6], *gsc = &in[7];
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
    const ik_tensor *sgw = &in[8], *sgs = &in[9], *sgg = &in[10], *suw = &in[11], *sus = &in[12], *sug = &in[13], *sdw = &in[14],
                    *sds = &in[15], *sdg = &in[16];
    if (logits->shape[0] != T || sl->shape[0] != T || sl->shape[1] != 1 || out->shape[0] != T || out->shape[1] != H) return IK_ERR_SHAPE;
    if (H % 16 != 0 || I % 16 != 0 || IS % 16 != 0 || H == 0 || I == 0 || IS == 0) return IK_ERR_SHAPE;
    if (guw->shape[0] != E || guw->shape[1] != 2 * I || guw->shape[2] != H / 2 || gus->shape[0] != E || gus->shape[1] != 2 * I ||
        gus->shape[2] != H / 16 || dnw->shape[0] != E || dnw->shape[1] != H || dnw->shape[2] != I / 2 || dns->shape[0] != E ||
        dns->shape[1] != H || dns->shape[2] != I / 16 || gsc->shape[0] != E || gsc->shape[1] != 3)
        return IK_ERR_SHAPE;
    for (const ik_tensor* w : {sgw, suw})
        if (w->shape[0] != IS || w->shape[1] != H / 2) return IK_ERR_SHAPE;
    for (const ik_tensor* b : {sgs, sus})
        if (b->shape[0] != IS || b->shape[1] != H / 16) return IK_ERR_SHAPE;
    if (sdw->shape[0] != H || sdw->shape[1] != IS / 2 || sds->shape[0] != H || sds->shape[1] != IS / 16) return IK_ERR_SHAPE;
    for (const ik_tensor* g : {sgg, sug, sdg})
        if (g->shape[0] != 1) return IK_ERR_SHAPE;
    if (T < 1 || T > kRowsMax || K < 1 || K > 8 || K > E || E > 512) return IK_ERR_SHAPE;
    const Layout L = layout(T, K, E, H, I, IS);
    if (!scratch || !scratch->data || scratch->bytes < L.total || (reinterpret_cast<uintptr_t>(scratch->data) & 15u))
        return IK_ERR_INVALID_ARG;
    unsigned char* base = (unsigned char*)scratch->data;
    ik_tensor wt = view2(base + L.w, IK_DTYPE_F32, T, K), idt = view2(base + L.ids, IK_DTYPE_I32, T, K);
    ik_tensor routed = view2(base + L.routed, IK_DTYPE_BF16, T, H);
    const ik_scratch exs = {base + L.ex, L.total - L.ex};
    hipStream_t s = IK_STREAM(stream);
    IK_TRY(ik_moe_topk_bf16_v2(logits, top_k, norm, &wt, &idt, stream));
    IK_TRY(ik_moe_experts_bf16_rows(x, &idt, &wt, guw, gus, dnw, dns, gsc, &routed, &exs, stream));
    ik_bf16* sa = (ik_bf16*)(base + L.sa);
    const unsigned tgroups = (unsigned)((T + kRowsGrp - 1) / kRowsGrp);
    shared_glu_rows_kernel<<<dim3((unsigned)((IS + kWarps - 1) / kWarps), tgroups), kWarps * IK_WARP, 0, s>>>(
        (const ik_bf16*)x->data, (const uint8_t*)sgw->data, (const uint8_t*)sgs->data, (const float*)sgg->data, (const uint8_t*)suw->data,
        (const uint8_t*)sus->data, (const float*)sug->data, sa, (int)T, IS, (int)H);
    IK_TRY(ik_launch_status());
    shared_down_combine_rows_kernel<<<dim3((unsigned)((H + kWarps - 1) / kWarps), (unsigned)((T + kRowsGrpDown - 1) / kRowsGrpDown)),
                                      kWarps * IK_WARP, 0, s>>>(
        sa, (const uint8_t*)sdw->data, (const uint8_t*)sds->data, (const float*)sdg->data, (const ik_bf16*)routed.data,
        (const ik_bf16*)sl->data, (ik_bf16*)out->data, (int)T, H, (int)IS);
    return ik_launch_status();
}

}  // extern "C"
