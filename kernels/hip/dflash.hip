// DFlash2 draft kernels (lane dfh).
//
// dyn_conv_bf16 (`Op::DynConv`): the draft's causal conv along the block rows with per-row dynamic taps.
//   x [T,H], k [T,K*G] (row t: K taps x G groups, tap-major), base [K,H] -> out [T,H]
//   out[t,c] = sum over taps o < K with t >= o of (base[o,c] + k[t, o*G + c / (H/G)]) * x[t-o, c]
// f32 accumulation (taps in ascending order), one rounding to bf16. One thread per (t, c): the op is memory bound and tiny
// (T <= 16 rows of H = 5120 in use), so there is no tiling.
#include "ik_hip.h"

namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads)
dyn_conv_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ k, const ik_bf16* __restrict__ base,
                ik_bf16* __restrict__ out, int64_t T, int64_t H, int taps, int64_t G) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int64_t t = i / H;
    const int64_t c = i % H;
    const int64_t grp = c / (H / G);
    const int n = (int)(t + 1 < taps ? t + 1 : taps);
    float acc = 0.f;
    for (int o = 0; o < n; ++o)
        acc += (ik_b2f(base[(int64_t)o * H + c]) + ik_b2f(k[t * taps * G + (int64_t)o * G + grp])) *
               ik_b2f(x[(t - o) * H + c]);
    out[i] = ik_f2b(acc);
}

}  // namespace

extern "C" {

int ik_dyn_conv_bf16(const ik_tensor* x, const ik_tensor* k, const ik_tensor* base, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(k, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(base, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t T = x->shape[0], H = x->shape[1];
    const int64_t taps = base->shape[0];
    if (base->shape[1] != H || taps <= 0 || taps > 64 || k->shape[0] != T) return IK_ERR_SHAPE;
    if (k->shape[1] % taps != 0 || k->shape[1] / taps <= 0) return IK_ERR_SHAPE;
    const int64_t G = k->shape[1] / taps;
    if (H % G != 0 || out->shape[0] != T || out->shape[1] != H) return IK_ERR_SHAPE;
    if (T == 0 || H == 0) return IK_OK;
    const int64_t n = T * H;
    if ((n + kThreads - 1) / kThreads > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    dyn_conv_kernel<<<(unsigned)((n + kThreads - 1) / kThreads), kThreads, 0, s>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)k->data, (const ik_bf16*)base->data, (ik_bf16*)out->data, T, H,
        (int)taps, G);
    return ik_launch_status();
}

}  // extern "C"
