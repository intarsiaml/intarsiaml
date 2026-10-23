// Top-k of f32 rows, for device-side sampling (R3b): the k largest values of each row in the order intarsia_core::sampler walks
// them (value descending, ties by ascending index; -0.0 and +0.0 tie), with their indices. The host then runs the unchanged sampler
// arithmetic on the k candidates, which is exactly what `sample_row` does with top_k = k after its own selection, so a token sampled
// from this output is the token host sampling picks for the same draw.
//
// Two stages. Stage 1 (grid S x R): a block takes a slice of the row and selects its k best by k rounds, each round the best element
// strictly after the previous pick in the total order (no flags to maintain, ties cannot repeat a pick). Stage 2 (grid R): a block
// merges the S*k candidates of its row with the same rounds. NaN never wins; a row that holds a NaN is flagged (the host falls back
// to its comparator path for such a row, whose NaN behaviour is not a total order).
#include "ik_launch.h"
#include <limits.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kSlices = 32;
constexpr int kMaxK = 64;

// a precedes b in the sampler's order
IK_DEVI bool before(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

struct Cand {
    float v;
    int i;  // -1: none
};

IK_DEVI Cand better(Cand a, Cand b) {
    if (a.i < 0) return b;
    if (b.i < 0) return a;
    return before(a.v, a.i, b.v, b.i) ? a : b;
}

// block-wide best of one candidate per thread; every thread gets the result
IK_DEVI Cand block_best(Cand c, Cand* sm) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        Cand x;
        x.v = __shfl_xor_sync(0xffffffffu, c.v, o);
        x.i = __shfl_xor_sync(0xffffffffu, c.i, o);
        c = better(c, x);
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sm[w] = c;
    __syncthreads();
    Cand r = sm[0];
    for (int j = 1; j < kWarps; ++j) r = better(r, sm[j]);
    return r;
}

__global__ void __launch_bounds__(kThreads) topk_stage1(const float* __restrict__ x, int64_t V, int k, float* __restrict__ cv,
                                                       int* __restrict__ ci, int* __restrict__ nan_flag) {
    __shared__ Cand sm[kWarps];
    const int64_t row = blockIdx.y;
    const int64_t per = (V + kSlices - 1) / kSlices;
    const int64_t lo = (int64_t)blockIdx.x * per, hi = min(lo + per, V);
    const float* r = x + row * V;
    bool has_nan = false;
    for (int64_t j = lo + threadIdx.x; j < hi; j += kThreads) has_nan |= (r[j] != r[j]);
    if (has_nan) atomicOr(nan_flag + row, 1);
    Cand prev = {0.f, -1};
    float* ov = cv + (row * kSlices + blockIdx.x) * k;
    int* oi = ci + (row * kSlices + blockIdx.x) * k;
    for (int t = 0; t < k; ++t) {
        Cand best = {0.f, -1};
        for (int64_t j = lo + threadIdx.x; j < hi; j += kThreads) {
            const float v = r[j];
            if (v != v) continue;
            if (prev.i >= 0 && !before(prev.v, prev.i, v, (int)j)) continue;  // not strictly after the previous pick
            Cand c = {v, (int)j};
            best = better(best, c);
        }
        best = block_best(best, sm);
        if (threadIdx.x == 0) {
            ov[t] = best.v;
            oi[t] = best.i;
        }
        prev = best;
        if (best.i < 0) {  // slice exhausted: the remaining entries are empty
            for (int u = t + 1 + threadIdx.x; u < k; u += kThreads) {
                ov[u] = 0.f;
                oi[u] = -1;
            }
            break;
        }
    }
}

__global__ void __launch_bounds__(kThreads) topk_stage2(const float* __restrict__ cv, const int* __restrict__ ci, int k,
                                                       int* __restrict__ ids, float* __restrict__ vals) {
    __shared__ Cand sm[kWarps];
    const int64_t row = blockIdx.x;
    const int n = kSlices * k;
    const float* v = cv + row * n;
    const int* ix = ci + row * n;
    Cand prev = {0.f, -1};
    for (int t = 0; t < k; ++t) {
        Cand best = {0.f, -1};
        for (int j = threadIdx.x; j < n; j += kThreads) {
            if (ix[j] < 0) continue;
            if (prev.i >= 0 && !before(prev.v, prev.i, v[j], ix[j])) continue;
            Cand c = {v[j], ix[j]};
            best = better(best, c);
        }
        best = block_best(best, sm);
        if (threadIdx.x == 0) {
            ids[row * k + t] = best.i;
            vals[row * k + t] = best.v;
        }
        prev = best;
    }
}

}  // namespace

extern "C" {

// logits f32 [R, V] -> ids i32 [R, k], vals f32 [R, k], nan i32 [R] (1: the row holds a NaN; its entries are not meaningful).
// scratch >= R * 32 * k * 8 bytes. 1 <= k <= 64, k <= V.
int ik_topk_f32(const ik_tensor* logits, int64_t k, ik_tensor* ids, ik_tensor* vals, ik_tensor* nan, const ik_scratch* scratch,
                void* stream) {
    IK_TRY(ik_check(logits, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 2));
    IK_TRY(ik_check(vals, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(nan, IK_DTYPE_I32, 1));
    const int64_t R = logits->shape[0], V = logits->shape[1];
    if (k < 1 || k > kMaxK || k > V) return IK_ERR_SHAPE;
    if (ids->shape[0] != R || ids->shape[1] != k || vals->shape[0] != R || vals->shape[1] != k || nan->shape[0] != R)
        return IK_ERR_SHAPE;
    if (R == 0) return IK_OK;
    if (R > 65535) return IK_ERR_SHAPE;
    const int64_t need = R * kSlices * k * 8;
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    cudaStream_t s = IK_STREAM(stream);
    float* cv = (float*)scratch->data;
    int* ci = (int*)((char*)scratch->data + R * kSlices * k * 4);
    if (cudaMemsetAsync(nan->data, 0, R * 4, s) != cudaSuccess) return IK_ERR_LAUNCH;
    topk_stage1<<<dim3(kSlices, (unsigned)R), kThreads, 0, s>>>((const float*)logits->data, V, (int)k, cv, ci, (int*)nan->data);
    IK_TRY(ik_launch_status());
    topk_stage2<<<(unsigned)R, kThreads, 0, s>>>(cv, ci, (int)k, (int*)ids->data, (float*)vals->data);
    return ik_launch_status();
}

}  // extern "C"
