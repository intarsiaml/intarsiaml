// argmax over the last dim: one block per row, lowest index wins ties, NaN never wins
// (same semantics as intarsia_core::sampler::argmax: first index of the strict maximum, 0 if none).
#include "ik_hip.h"
#include <limits.h>

namespace {

constexpr int kThreads = 1024;

IK_DEVI float to_f(float v) { return v; }
IK_DEVI float to_f(ik_bf16 v) { return ik_b2f(v); }

IK_DEVI bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

IK_DEVI void argmax_xor_step(float& bv, int& bi, float ov, int oi) {
    if (better(ov, oi, bv, bi)) {
        bv = ov;
        bi = oi;
    }
}

// FAST (INTARSIA_HIP_INK_DEC3 bit 2, lane ink-hip-dec3): each thread issues the loads of 8 of its elements (clamped to the row, so
// always valid) before it compares them, in the same ascending order with the same strict >: the same (value, index) per thread.
// DPP (bit 1): the wave's butterfly takes the partner from DPP / v_permlanex16 instead of ds_bpermute; `better` is a total order
// (value, then lower index), so the wave's pick is the same.
template <typename T, bool FAST, bool DPP>
__global__ void __launch_bounds__(kThreads) argmax_kernel(const T* __restrict__ x, int32_t* __restrict__ out, int64_t V) {
    __shared__ float sv[kThreads / IK_WARP];
    __shared__ int si[kThreads / IK_WARP];
    const T* row = x + (int64_t)blockIdx.x * V;
    float bv = -INFINITY;
    int bi = INT_MAX;
    if constexpr (FAST) {
        constexpr int U = 8;
        const int64_t nt = blockDim.x;
        for (int64_t i0 = threadIdx.x; i0 < V; i0 += U * nt) {
            float v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u * nt;
                v[u] = to_f(row[i < V ? i : V - 1]);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int64_t i = i0 + u * nt;
                if (i < V && v[u] > bv) {
                    bv = v[u];
                    bi = (int)i;
                }
            }
        }
    } else {
        for (int64_t i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = to_f(row[i]);
            if (v > bv) {  // ascending i per thread: strict > keeps the lowest index
                bv = v;
                bi = (int)i;
            }
        }
    }
    if constexpr (DPP) {
        argmax_xor_step(bv, bi, ik_xor_f<16>(bv), ik_xor_i<16>(bi));
        argmax_xor_step(bv, bi, ik_xor_f<8>(bv), ik_xor_i<8>(bi));
        argmax_xor_step(bv, bi, ik_xor_f<4>(bv), ik_xor_i<4>(bi));
        argmax_xor_step(bv, bi, ik_xor_f<2>(bv), ik_xor_i<2>(bi));
        argmax_xor_step(bv, bi, ik_xor_f<1>(bv), ik_xor_i<1>(bi));
    } else {
#pragma unroll
        for (int o = IK_WARP / 2; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o);
            const int oi = __shfl_xor(bi, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
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

template <typename T>
void argmax_go(const T* x, int32_t* out, int64_t V, unsigned R, unsigned bd, hipStream_t s) {
    const int m = ik_hip_dec3_mask();
    const bool fast = (m & 2) != 0, dpp = (m & 1) != 0;
    if (fast && dpp) argmax_kernel<T, true, true><<<R, bd, 0, s>>>(x, out, V);
    else if (fast) argmax_kernel<T, true, false><<<R, bd, 0, s>>>(x, out, V);
    else if (dpp) argmax_kernel<T, false, true><<<R, bd, 0, s>>>(x, out, V);
    else argmax_kernel<T, false, false><<<R, bd, 0, s>>>(x, out, V);
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
        argmax_go<float>((const float*)logits->data, (int32_t*)out->data, V, (unsigned)R, bd, s);
    else
        argmax_go<ik_bf16>((const ik_bf16*)logits->data, (int32_t*)out->data, V, (unsigned)R, bd, s);
    return ik_launch_status();
}

int g_dec3 = -1;  // ik_hip_dec3_set override (tests); < 0: the env / default

}  // namespace

// INTARSIA_HIP_INK_DEC3 (ik_hip.h): bit mask, default 7, read once per process
extern "C" int ik_hip_dec3_mask(void) {
    if (g_dec3 >= 0) return g_dec3;
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("INTARSIA_HIP_INK_DEC3");
        v = e && e[0] >= '0' && e[0] <= '9' ? atoi(e) : 7;
    }
    return v;
}
extern "C" void ik_hip_dec3_set(int v) { g_dec3 = v; }

extern "C" {
int ik_argmax_f32(const ik_tensor* logits, ik_tensor* out, void* stream) {
    return argmax_launch(logits, out, stream, IK_DTYPE_F32);
}
int ik_argmax_bf16(const ik_tensor* logits, ik_tensor* out, void* stream) {
    return argmax_launch(logits, out, stream, IK_DTYPE_BF16);
}
}  // extern "C"
