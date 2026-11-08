// DSpine draft code-bridge row ops (hip twin of kernels/metal/bridge.metal; semantics crates/intarsia-core/src/ops.rs, the cpu oracle
// crates/intarsia-backend-cpu/src/bridge_ref.rs; docs/plans/2026-11-07-dspine-port.md section 5).
//
//   bridge_norm_bf16   (Op::BridgeNorm)   x [T,D] -> y [T,D] bf16, rms [T] f32:  rms = sqrt(max(mean x^2, FLT_MIN)), y = bf16(x / (rms + eps))
//   bridge_code_bf16   (Op::BridgeCode)   p [T,C] -> code [T,C]:                  code = bf16(scale * p / (sqrt(sum p^2) + eps))
//   bridge_gate_bf16   (Op::BridgeGate)   packed [T,3R], bias [R], s0 [2R] -> msg [T,R]: msg[0] = 0,
//                                          msg[t] = bf16(sigmoid((q[t] + s[t-1]) + bias) * v[t-1]) (row 1's sender is s0 = [s | v])
//   bridge_write_bf16  (Op::BridgeWrite)  raw [T,H], rms [T] f32, delta [T,H], w [H] -> out [T,H], normed [T,H]:
//                                          out = bf16(raw + (rms * delta) * (t != 0)), normed = bf16(bf16(out * rsqrt(mean out^2 + eps)) * w)
// One block per row (the gate: one thread per element); f32 math, the block sums in a fixed order. T <= 16 rows in use.
#include "ik_hip.h"

namespace {

constexpr int kThreads = 256;
constexpr float kFltMin = 1.1754943508222875e-38f;

__global__ void __launch_bounds__(kThreads)
bridge_norm_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ y, float* __restrict__ rms, int64_t D, float eps) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const ik_bf16* xr = x + row * D;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = ik_b2f(xr[i]);
        ss += v * v;
    }
    ss = ik_block_sum(ss, red);
    const float r = sqrtf(fmaxf(ss / (float)D, kFltMin));
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) y[row * D + i] = ik_f2b(ik_b2f(xr[i]) / (r + eps));
    if (threadIdx.x == 0) rms[row] = r;
}

__global__ void __launch_bounds__(kThreads)
bridge_code_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ y, int64_t D, float eps, float scale) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const ik_bf16* xr = x + row * D;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = ik_b2f(xr[i]);
        ss += v * v;
    }
    ss = ik_block_sum(ss, red);
    const float n = sqrtf(ss);
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) y[row * D + i] = ik_f2b((scale * ik_b2f(xr[i])) / (n + eps));
}

__global__ void __launch_bounds__(kThreads)
bridge_gate_kernel(const ik_bf16* __restrict__ packed, const ik_bf16* __restrict__ bias, const ik_bf16* __restrict__ s0,
                   ik_bf16* __restrict__ msg, int64_t T, int64_t R) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * R) return;
    const int64_t row = i / R, c = i % R;
    if (row == 0) {
        msg[i] = ik_f2b(0.f);
        return;
    }
    const float q = ik_b2f(packed[row * 3 * R + c]);
    float s, v;
    if (row == 1) {
        s = ik_b2f(s0[c]);
        v = ik_b2f(s0[R + c]);
    } else {
        s = ik_b2f(packed[(row - 1) * 3 * R + R + c]);
        v = ik_b2f(packed[(row - 1) * 3 * R + 2 * R + c]);
    }
    const float z = (q + s) + ik_b2f(bias[c]);
    msg[i] = ik_f2b((1.f / (1.f + expf(-z))) * v);
}

__global__ void __launch_bounds__(kThreads)
bridge_write_kernel(const ik_bf16* __restrict__ raw, const float* __restrict__ rms, const ik_bf16* __restrict__ delta,
                    const ik_bf16* __restrict__ w, ik_bf16* __restrict__ out, ik_bf16* __restrict__ normed, int64_t H, float eps) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const int64_t base = row * H;
    const float rr = rms[row];
    const float na = row == 0 ? 0.f : 1.f;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
        const ik_bf16 o = ik_f2b(ik_b2f(raw[base + i]) + (rr * ik_b2f(delta[base + i])) * na);
        out[base + i] = o;
        const float v = ik_b2f(o);
        ss += v * v;
    }
    ss = ik_block_sum(ss, red);
    const float inv = 1.f / sqrtf(ss / (float)H + eps);
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
        const float y = ik_b2f(ik_f2b(ik_b2f(out[base + i]) * inv));
        normed[base + i] = ik_f2b(y * ik_b2f(w[i]));
    }
}

}  // namespace

extern "C" {

int ik_bridge_norm_bf16(const ik_tensor* x, float eps, ik_tensor* y, ik_tensor* rms, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(y, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(rms, IK_DTYPE_F32, 1));
    const int64_t T = x->shape[0], D = x->shape[1];
    if (y->shape[0] != T || y->shape[1] != D || rms->shape[0] != T) return IK_ERR_SHAPE;
    if (T == 0 || D == 0) return IK_OK;
    bridge_norm_kernel<<<(unsigned)T, kThreads, 0, IK_STREAM(stream)>>>((const ik_bf16*)x->data, (ik_bf16*)y->data, (float*)rms->data, D, eps);
    return ik_launch_status();
}

int ik_bridge_code_bf16(const ik_tensor* x, float eps, float scale, ik_tensor* y, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(y, IK_DTYPE_BF16, 2));
    const int64_t T = x->shape[0], D = x->shape[1];
    if (y->shape[0] != T || y->shape[1] != D) return IK_ERR_SHAPE;
    if (T == 0 || D == 0) return IK_OK;
    bridge_code_kernel<<<(unsigned)T, kThreads, 0, IK_STREAM(stream)>>>((const ik_bf16*)x->data, (ik_bf16*)y->data, D, eps, scale);
    return ik_launch_status();
}

int ik_bridge_gate_bf16(const ik_tensor* packed, const ik_tensor* bias, const ik_tensor* s0, ik_tensor* msg, void* stream) {
    IK_TRY(ik_check(packed, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(s0, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(msg, IK_DTYPE_BF16, 2));
    const int64_t T = packed->shape[0], R = bias->shape[0];
    if (packed->shape[1] != 3 * R || s0->shape[0] != 2 * R || msg->shape[0] != T || msg->shape[1] != R) return IK_ERR_SHAPE;
    if (T == 0 || R == 0) return IK_OK;
    const int64_t n = T * R;
    if ((n + kThreads - 1) / kThreads > 0x7fffffffLL) return IK_ERR_SHAPE;
    bridge_gate_kernel<<<(unsigned)((n + kThreads - 1) / kThreads), kThreads, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)packed->data, (const ik_bf16*)bias->data, (const ik_bf16*)s0->data, (ik_bf16*)msg->data, T, R);
    return ik_launch_status();
}

int ik_bridge_write_bf16(const ik_tensor* raw, const ik_tensor* rms, const ik_tensor* delta, const ik_tensor* w, float eps, ik_tensor* out,
                         ik_tensor* normed, void* stream) {
    IK_TRY(ik_check(raw, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(rms, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(delta, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(normed, IK_DTYPE_BF16, 2));
    const int64_t T = raw->shape[0], H = raw->shape[1];
    if (rms->shape[0] != T || delta->shape[0] != T || delta->shape[1] != H || w->shape[0] != H) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != H || normed->shape[0] != T || normed->shape[1] != H) return IK_ERR_SHAPE;
    if (T == 0 || H == 0) return IK_OK;
    bridge_write_kernel<<<(unsigned)T, kThreads, 0, IK_STREAM(stream)>>>((const ik_bf16*)raw->data, (const float*)rms->data,
                                                                         (const ik_bf16*)delta->data, (const ik_bf16*)w->data,
                                                                         (ik_bf16*)out->data, (ik_bf16*)normed->data, H, eps);
    return ik_launch_status();
}

}  // extern "C"
