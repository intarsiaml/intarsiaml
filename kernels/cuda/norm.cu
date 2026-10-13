// RMSNorm family: rmsnorm, rmsnorm_add (residual add fused), qk_norm (per-head rmsnorm).
#include "ik_launch.h"

namespace {

// One block per row. out = bf16( x * rsqrt(mean(x^2) + eps) * w ). With ADD the row is first
// summed with `res`, rounded to bf16 (that rounded sum is what the next layer carries) and written
// to `sum`; the normalisation then reads the rounded sum, exactly like the reference.
template <bool ADD, bool VEC>
__global__ void rmsnorm_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res,
                               const ik_bf16* __restrict__ w, ik_bf16* __restrict__ out, ik_bf16* __restrict__ sum,
                               int64_t H, float eps) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const ik_bf16* xr = x + row * H;
    const ik_bf16* rr = ADD ? res + row * H : nullptr;
    ik_bf16* orow = out + row * H;
    ik_bf16* sr = ADD ? sum + row * H : nullptr;
    // the row the normalisation reads back (sum when ADD, else x)
    const ik_bf16* src = ADD ? sr : xr;

    float ss = 0.f;
    if (VEC) {
        const int64_t nv = H / 8;
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            float f[8];
            ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
            if (ADD) {
                float g[8];
                ik_unpack8(reinterpret_cast<const uint4*>(rr)[i], g);
#pragma unroll
                for (int k = 0; k < 8; ++k) f[k] += g[k];
                const uint4 p = ik_pack8(f);
                reinterpret_cast<uint4*>(sr)[i] = p;
                ik_unpack8(p, f);  // continue from the bf16-rounded sum
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
        }
    } else {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
            float v = ik_b2f(xr[i]);
            if (ADD) {
                const ik_bf16 b = ik_f2b(v + ik_b2f(rr[i]));
                sr[i] = b;
                v = ik_b2f(b);
            }
            ss += v * v;
        }
    }
    ss = ik_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)H + eps);

    if (VEC) {
        const int64_t nv = H / 8;
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            float f[8], g[8];
            ik_unpack8(reinterpret_cast<const uint4*>(src)[i], f);
            ik_unpack8(reinterpret_cast<const uint4*>(w)[i], g);
#pragma unroll
            for (int k = 0; k < 8; ++k) f[k] = f[k] * r * g[k];
            reinterpret_cast<uint4*>(orow)[i] = ik_pack8(f);
        }
    } else {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) orow[i] = ik_f2b(ik_b2f(src[i]) * r * ik_b2f(w[i]));
    }
}

int launch_rows(const ik_bf16* x, const ik_bf16* res, const ik_bf16* w, ik_bf16* out, ik_bf16* sum, int64_t rows,
                int64_t H, float eps, cudaStream_t s) {
    if (rows == 0 || H == 0) return IK_OK;
    const bool add = res != nullptr;
    bool vec = H % 8 == 0 && ik_aligned16(x) && ik_aligned16(w) && ik_aligned16(out);
    if (add) vec = vec && ik_aligned16(res) && ik_aligned16(sum);
    const int64_t items = vec ? H / 8 : H;
    const unsigned bd = ik_round_up((unsigned)(items < 256 ? items : 256), 32);
    if (rows > 0x7fffffffLL) return IK_ERR_SHAPE;
    const unsigned g = (unsigned)rows;
    if (add) {
        if (vec) rmsnorm_kernel<true, true><<<g, bd, 0, s>>>(x, res, w, out, sum, H, eps);
        else rmsnorm_kernel<true, false><<<g, bd, 0, s>>>(x, res, w, out, sum, H, eps);
    } else {
        if (vec) rmsnorm_kernel<false, true><<<g, bd, 0, s>>>(x, nullptr, w, out, nullptr, H, eps);
        else rmsnorm_kernel<false, false><<<g, bd, 0, s>>>(x, nullptr, w, out, nullptr, H, eps);
    }
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_rmsnorm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    if (w->shape[0] != x->shape[1] || out->shape[0] != x->shape[0] || out->shape[1] != x->shape[1])
        return IK_ERR_SHAPE;
    return launch_rows((const ik_bf16*)x->data, nullptr, (const ik_bf16*)w->data, (ik_bf16*)out->data, nullptr,
                       x->shape[0], x->shape[1], eps, IK_STREAM(stream));
}

int ik_rmsnorm_add_bf16(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, ik_tensor* out,
                        ik_tensor* sum_out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(res, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(sum_out, IK_DTYPE_BF16, 2));
    const int64_t R = x->shape[0], H = x->shape[1];
    if (res->shape[0] != R || res->shape[1] != H || w->shape[0] != H || out->shape[0] != R || out->shape[1] != H ||
        sum_out->shape[0] != R || sum_out->shape[1] != H)
        return IK_ERR_SHAPE;
    return launch_rows((const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)w->data,
                       (ik_bf16*)out->data, (ik_bf16*)sum_out->data, R, H, eps, IK_STREAM(stream));
}

int ik_qk_norm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t D = x->shape[2];
    if (w->shape[0] != D || out->shape[0] != x->shape[0] || out->shape[1] != x->shape[1] || out->shape[2] != D)
        return IK_ERR_SHAPE;
    return launch_rows((const ik_bf16*)x->data, nullptr, (const ik_bf16*)w->data, (ik_bf16*)out->data, nullptr,
                       x->shape[0] * x->shape[1], D, eps, IK_STREAM(stream));
}

}  // extern "C"
