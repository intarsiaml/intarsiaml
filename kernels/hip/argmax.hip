// argmax over the last dim: one block per row, lowest index wins ties, NaN never wins
// (same semantics as intarsia_core::sampler::argmax: first index of the strict maximum, 0 if none).
#include "ik_hip.h"
#include <limits.h>

namespace {

constexpr int kThreads = 1024;

IK_DEVI float to_f(float v) { return v; }
IK_DEVI float to_f(ik_bf16 v) { return ik_b2f(v); }

IK_DEVI bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

template <typename T>
__global__ void __launch_bounds__(kThreads) argmax_kernel(const T* __restrict__ x, int32_t* __restrict__ out, int64_t V) {
    __shared__ float sv[kThreads / IK_WARP];
    __shared__ int si[kThreads / IK_WARP];
    const T* row = x + (int64_t)blockIdx.x * V;
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int64_t i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = to_f(row[i]);
        if (v > bv) {  // ascending i per thread: strict > keeps the lowest index
            bv = v;
            bi = (int)i;
        }
    }
#pragma unroll
    for (int o = IK_WARP / 2; o > 0; o >>= 1) {
        const float ov = __shfl_xor(bv, o);
        const int oi = __shfl_xor(bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x % IK_WARP, w = threadIdx.x / IK_WARP;
    if (lane == 0) {
        sv[w] = bv;
        si[w] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float fv = sv[0];
        int fi = si[0];
        for (int i = 1; i < (int)(blockDim.x / IK_WARP); ++i)
            if (better(sv[i], si[i], fv, fi)) {
                fv = sv[i];
                fi = si[i];
            }
        out[blockIdx.x] = fi == INT_MAX ? 0 : fi;
    }
}

int argmax_launch(const ik_tensor* logits, ik_tensor* out, void* stream, int dtype) {
    IK_TRY(ik_check(logits, dtype, 2));
    IK_TRY(ik_check(out, IK_DTYPE_I32, 1));
    const int64_t R = logits->shape[0], V = logits->shape[1];
    if (out->shape[0] != R || V > INT_MAX) return IK_ERR_SHAPE;
    if (R == 0) return IK_OK;
    if (V == 0) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const unsigned bd = V >= kThreads ? kThreads : ik_round_up((unsigned)V, 32);
    if (dtype == IK_DTYPE_F32)
        argmax_kernel<float><<<(unsigned)R, bd, 0, s>>>((const float*)logits->data, (int32_t*)out->data, V);
    else
        argmax_kernel<ik_bf16><<<(unsigned)R, bd, 0, s>>>((const ik_bf16*)logits->data, (int32_t*)out->data, V);
    return ik_launch_status();
}

}  // namespace

extern "C" {
int ik_argmax_f32(const ik_tensor* logits, ik_tensor* out, void* stream) {
    return argmax_launch(logits, out, stream, IK_DTYPE_F32);
}
int ik_argmax_bf16(const ik_tensor* logits, ik_tensor* out, void* stream) {
    return argmax_launch(logits, out, stream, IK_DTYPE_BF16);
}
}  // extern "C"
