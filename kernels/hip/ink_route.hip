// moe_route (Op::MoeRoute) on HIP, for Inkling-2-Mini's router (lane ink-hip): a port of moe_route_kernel in
// kernels/cuda/dsv41_misc.cu (DSv4.1's Gate.forward form; score_func 0 softmax, 1 sigmoid, 2 sqrt-softplus), same operation
// order per row. logits f32 [T,E], bias f32 [E] -> weights f32 [T,K], ids i32 [T,K]. One wave per row, so a row's bits do not
// depend on T.
#include "ik_hip.h"
#include "ik_hip_abi_inkroute.h"

namespace {

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

__global__ void __launch_bounds__(IK_WARP) moe_route_kernel(const float* __restrict__ lg, const float* __restrict__ bias, int64_t E, int K,
                                                            float scale, int norm, float temp, int sf, float* __restrict__ w,
                                                            int* __restrict__ ids) {
    extern __shared__ float sm[];
    float* sc = sm;
    float* bi = sm + E;
    __shared__ int pick[32];
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const float* l = lg + row * E;
    float mx = -INFINITY;
    for (int64_t e = lane; e < E; e += IK_WARP) {
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
        for (int64_t e = lane; e < E; e += IK_WARP) {
            const float x = expf(sc[e] - mx);
            sc[e] = x;
            sum += x;
        }
        sum = ik_warp_sum(sum);
        for (int64_t e = lane; e < E; e += IK_WARP) sc[e] = sc[e] / sum;
    }
    for (int64_t e = lane; e < E; e += IK_WARP) bi[e] = sc[e] + bias[e];
    __syncthreads();
    // K rounds, each the best element strictly after the previous pick in the total order (ties cannot repeat a pick)
    float pv = INFINITY;
    int pi = -1;
    for (int k = 0; k < K; ++k) {
        Pick b{0.f, -1};
        for (int64_t e = lane; e < E; e += IK_WARP) {
            const float v = bi[e];
            if (v != v) continue;
            if (before(pv, pi, v, (int)e)) b = better(b, Pick{v, (int)e});
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
    __syncthreads();
    if (lane == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += pick[k] >= 0 ? sc[pick[k]] : 0.f;
        const float den = (norm && K > 1) ? sum + 1e-20f : 1.f;
        for (int k = 0; k < K; ++k) {
            const int id = pick[k];
            const float v = id >= 0 ? sc[id] : 0.f;
            w[row * K + k] = (norm && K > 1 ? v / den : v) * scale;
            ids[row * K + k] = id;
        }
    }
}

}  // namespace

extern "C" int ik_moe_route(const ik_tensor* logits, const ik_tensor* bias, int64_t top_k, float scale, int32_t norm_topk,
                            float temperature, int32_t score_func, ik_tensor* weights, ik_tensor* ids, void* stream) {
    IK_TRY(ik_check(logits, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(bias, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(weights, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 2));
    const int64_t T = logits->shape[0], E = logits->shape[1];
    if (top_k < 1 || top_k > 32 || top_k > E || E > 1024 || bias->shape[0] != E || weights->shape[0] != T ||
        weights->shape[1] != top_k || ids->shape[0] != T || ids->shape[1] != top_k || score_func < 0 || score_func > 2)
        return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    moe_route_kernel<<<(unsigned)T, IK_WARP, (size_t)E * 8, IK_STREAM(stream)>>>((const float*)logits->data, (const float*)bias->data, E,
                                                                               (int)top_k, scale, norm_topk, temperature, score_func,
                                                                               (float*)weights->data, (int*)ids->data);
    return ik_launch_status();
}
