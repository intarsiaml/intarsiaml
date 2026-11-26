// ink_route_bf16 (Op::InkRoute, 76) on HIP, lane ink-hip-mix: the router GEMV and the top-k of `moe_route` (kernels/hip/ink_route.hip) in ONE
// launch. Blocks of kWaves waves; wave w of block b computes logit e = b * kWaves + w of all T rows with the lane order of
// `logits_bf16_gemv` (idx = lane, lane + 32, .. ascending, `acc += w * x` per element in order, then ik_warp_sum), so the logits are
// bitwise `Logits`. The logits go to global memory (the `logits` output, or the call's scratch), the last block to finish (a ticket in a
// device-global counter that the last block resets) routes row r on wave r with the body of moe_route_kernel (same operation order, so
// bitwise `MoeRoute`). A row never depends on the other rows: rows-exact at any T <= 16 by construction (no twin).
// x bf16 [T,D], w bf16 [E,D], bias f32 [E] -> weights f32 [T,K], ids i32 [T,K], logits f32 [T,E] or [0]; E <= 256, K <= 32, T <= 16.
#include "ik_hip.h"
#include "ik_hip_abi_inkmix.h"

namespace {

constexpr int kWaves = 4;       // logits per block (E / 4 blocks: 32 for the real router, so the GEMV spreads over the CUs); the last block routes the rows, wave w rows w, w + 4, ..
constexpr int kRowsMax = 16;
constexpr int kEMax = 256;      // experts: the routing keeps 2 * E floats per wave in shared memory

__device__ unsigned int g_route_ctr = 0;

struct Pick {
    float v;
    int i;  // -1: none
};

// (va, ia) precedes (vb, ib): value descending, ties to the lower index
IK_DEVI bool before(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

IK_DEVI Pick better(Pick a, Pick b) {
    if (a.i < 0) return b;
    if (b.i < 0) return a;
    return before(a.v, a.i, b.v, b.i) ? a : b;
}

// shared-memory visibility inside one wave (the block-wide barrier of moe_route's one-wave block, at wave scope)
IK_DEVI void wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// moe_route_kernel's body for one row, on one wave; sc / bi / pick are this wave's shared arrays
IK_DEVI void route_row(const float* __restrict__ l, const float* __restrict__ bias, int E, int K, float scale, int norm, float temp, int sf,
                       float* sc, float* bi, int* pick, float* __restrict__ w, int* __restrict__ ids) {
    const int lane = threadIdx.x & 31;
    float mx = -INFINITY;
    for (int e = lane; e < E; e += IK_WARP) {
        const float v = l[e] / temp;
        float s;
        if (sf == 2) {
            s = sqrtf(fmaxf(v, 0.f) + log1pf(expf(-fabsf(v))));
        } else if (sf == 1) {
            s = 1.f / (1.f + expf(-v));
        } else {
            s = v;
            mx = fmaxf(mx, v);
        }
        sc[e] = s;
    }
    if (sf == 0) {
        mx = ik_warp_max(mx);
        float sum = 0.f;
        for (int e = lane; e < E; e += IK_WARP) {
            const float x = expf(sc[e] - mx);
            sc[e] = x;
            sum += x;
        }
        sum = ik_warp_sum(sum);
        for (int e = lane; e < E; e += IK_WARP) sc[e] = sc[e] / sum;
    }
    for (int e = lane; e < E; e += IK_WARP) bi[e] = sc[e] + bias[e];
    wave_sync();
    // K rounds, each the best element strictly after the previous pick in the total order (ties cannot repeat a pick)
    float pv = INFINITY;
    int pi = -1;
    for (int k = 0; k < K; ++k) {
        Pick b{0.f, -1};
        for (int e = lane; e < E; e += IK_WARP) {
            const float v = bi[e];
            if (v != v) continue;
            if (before(pv, pi, v, e)) b = better(b, Pick{v, e});
        }
#pragma unroll
        for (int o = IK_WARP / 2; o > 0; o >>= 1) {
            Pick x;
            x.v = __shfl_xor(b.v, o, IK_WARP);
            x.i = __shfl_xor(b.i, o, IK_WARP);
            b = better(b, x);
        }
        if (lane == 0) pick[k] = b.i;
        pv = b.v;
        pi = b.i;
    }
    wave_sync();
    if (lane == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += pick[k] >= 0 ? sc[pick[k]] : 0.f;
        const float den = (norm && K > 1) ? sum + 1e-20f : 1.f;
        for (int k = 0; k < K; ++k) {
            const int id = pick[k];
            const float v = id >= 0 ? sc[id] : 0.f;
            w[k] = (norm && K > 1 ? v / den : v) * scale;
            ids[k] = id;
        }
    }
}

__global__ void __launch_bounds__(kWaves* IK_WARP)
ink_route_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ wt, const float* __restrict__ bias, float* lg, int T, int E,
                 int64_t D, int K, float scale, int norm, float temp, int sf, float* __restrict__ ow, int* __restrict__ oid) {
    __shared__ float smem[kWaves * 2 * kEMax];
    __shared__ int picks[kWaves][32];
    __shared__ int last_blk;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * kWaves + wave;
    if (e < E) {
        const int64_t nv = D / 8;
        const uint4* w8 = reinterpret_cast<const uint4*>(wt) + (int64_t)e * nv;
        const uint4* x8 = reinterpret_cast<const uint4*>(x);
        float acc[kRowsMax];
#pragma unroll
        for (int m = 0; m < kRowsMax; ++m) acc[m] = 0.f;
#pragma unroll 6
        for (int64_t i = lane; i < nv; i += IK_WARP) {
            float wf[8];
            ik_unpack8(ik_ldcs(w8 + i), wf);
#pragma unroll
            for (int m = 0; m < kRowsMax; ++m) {
                if (m >= T) break;
                float xf[8];
                ik_unpack8(ik_ldg(x8 + m * nv + i), xf);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[m] += wf[k] * xf[k];
            }
        }
#pragma unroll
        for (int m = 0; m < kRowsMax; ++m) {
            if (m >= T) break;
            const float v = ik_warp_sum(acc[m]);
            if (lane == 0) lg[(int64_t)m * E + e] = v;
        }
    }
    __threadfence();  // the logits of this block before the ticket
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(&g_route_ctr, 1u);
        const bool l = old == gridDim.x - 1;
        if (l) atomicExch(&g_route_ctr, 0u);
        last_blk = l;
    }
    __syncthreads();
    if (!last_blk) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    float* sc = smem + wave * 2 * kEMax;
    for (int r = wave; r < T; r += kWaves)
        route_row(lg + (int64_t)r * E, bias, E, K, scale, norm, temp, sf, sc, sc + kEMax, picks[wave], ow + (int64_t)r * K, oid + (int64_t)r * K);
}

}  // namespace

extern "C" int ik_ink_route_bf16(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bias, int64_t top_k, float scale, int32_t norm_topk,
                                 float temperature, int32_t score_func, ik_tensor* weights, ik_tensor* ids, ik_tensor* logits,
                                 const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(bias, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(weights, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 2));
    const int64_t T = x->shape[0], D = x->shape[1], E = w->shape[0];
    if (w->shape[1] != D || bias->shape[0] != E || weights->shape[0] != T || weights->shape[1] != top_k || ids->shape[0] != T ||
        ids->shape[1] != top_k || top_k < 1 || top_k > 32 || top_k > E || E > kEMax || T > kRowsMax || D % 8 != 0 || score_func < 0 ||
        score_func > 2)
        return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    if (!ik_aligned16(x->data) || !ik_aligned16(w->data)) return IK_ERR_INVALID_ARG;
    float* lg;
    if (logits && ik_numel(logits) > 0) {
        IK_TRY(ik_check(logits, IK_DTYPE_F32, 2));
        if (logits->shape[0] != T || logits->shape[1] != E) return IK_ERR_SHAPE;
        lg = (float*)logits->data;
    } else {
        if (!scratch || !scratch->data || scratch->bytes < T * E * 4) return IK_ERR_INVALID_ARG;
        lg = (float*)scratch->data;
    }
    const unsigned grid = (unsigned)((E + kWaves - 1) / kWaves);
    ink_route_kernel<<<grid, kWaves * IK_WARP, 0, IK_STREAM(stream)>>>((const ik_bf16*)x->data, (const ik_bf16*)w->data,
                                                                       (const float*)bias->data, lg, (int)T, (int)E, D, (int)top_k, scale,
                                                                       norm_topk, temperature, score_func, (float*)weights->data,
                                                                       (int*)ids->data);
    return ik_launch_status();
}
