// The DSpark draft's tap mix on HIP (lane ink-hip-spec): `HcPre` (DSv4.1's hc_pre, the op the Inkling-2 DSpark ingest runs with
// softmax(fuse_weight), crates/intarsia-core/src/model/dspark.rs). A port of hc_pre_kernel in kernels/cuda/dsv41_hc.cu, same
// operation order per element: acc = fmaf(mix[t, c], x[t, c, d], acc) for c = 0 .. S-1, from 0, rounded once at the store.
// x [T,S,D] f32 or bf16, mix f32 [T,S] -> y [T,D] f32 or bf16, S <= 8. Element-parallel, so a row's bits do not depend on T.
#include "ik_hip.h"
#include "ik_hip_abi_inkspec.h"

namespace {

constexpr int kThreads = 256;

IK_DEVI float ld(const float* p, int64_t i) { return p[i]; }
IK_DEVI float ld(const ik_bf16* p, int64_t i) { return ik_b2f(p[i]); }
IK_DEVI void st(float* p, int64_t i, float v) { p[i] = v; }
IK_DEVI void st(ik_bf16* p, int64_t i, float v) { p[i] = ik_f2b(v); }

template <typename Ti, typename To>
__global__ void __launch_bounds__(kThreads) hc_pre_kernel(const Ti* __restrict__ x, const float* __restrict__ mix, To* __restrict__ y,
                                                         int64_t T, int s, int64_t D) {
    const int64_t n = T * D;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = i / D, d = i % D;
        float acc = 0.f;
        for (int c = 0; c < s; ++c) acc = fmaf(mix[t * s + c], ld(x, (t * s + c) * D + d), acc);
        st(y, i, acc);
    }
}

template <typename Ti, typename To>
int launch(const ik_tensor* x, const ik_tensor* mix, ik_tensor* y, int64_t T, int64_t S, int64_t D, hipStream_t s) {
    const int64_t n = T * D;
    int64_t blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 65535 * 8) blocks = 65535 * 8;  // the loop strides over the rest
    hc_pre_kernel<Ti, To><<<(unsigned)blocks, kThreads, 0, s>>>((const Ti*)x->data, (const float*)mix->data, (To*)y->data, T, (int)S, D);
    return ik_launch_status();
}

inline bool is_float(const ik_tensor* t) { return t && (t->dtype == IK_DTYPE_F32 || t->dtype == IK_DTYPE_BF16); }

}  // namespace

extern "C" int ik_hc_pre_bf16(const ik_tensor* x, const ik_tensor* mix, ik_tensor* y, void* stream) {
    if (!is_float(x) || !is_float(y)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(x, x->dtype, 3));
    IK_TRY(ik_check(mix, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(y, y->dtype, 2));
    const int64_t T = x->shape[0], S = x->shape[1], D = x->shape[2];
    if (mix->shape[0] != T || mix->shape[1] != S || y->shape[0] != T || y->shape[1] != D || S < 1 || S > 8) return IK_ERR_SHAPE;
    if (T * D == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    const bool xb = x->dtype == IK_DTYPE_BF16, yb = y->dtype == IK_DTYPE_BF16;
    if (xb && yb) return launch<ik_bf16, ik_bf16>(x, mix, y, T, S, D, s);
    if (xb) return launch<ik_bf16, float>(x, mix, y, T, S, D, s);
    if (yb) return launch<float, ik_bf16>(x, mix, y, T, S, D, s);
    return launch<float, float>(x, mix, y, T, S, D, s);
}
