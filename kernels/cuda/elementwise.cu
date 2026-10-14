// Data-movement and elementwise launchers: embedding, add, silu_glu, kv_append, rope.
#include "ik_launch.h"

namespace {

constexpr int kThreads = 256;

// ---- embedding ---------------------------------------------------------------------------
template <bool VEC>
__global__ void embedding_kernel(const int32_t* __restrict__ ids, const ik_bf16* __restrict__ table,
                                 ik_bf16* __restrict__ out, int64_t V, int64_t H) {
    const int64_t t = blockIdx.x;
    const int32_t id = ids[t];
    ik_bf16* o = out + t * H;
    if (id < 0 || id >= V) {  // the host validates ids; never read out of bounds on the device
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) o[i] = ik_f2b(0.f);
        return;
    }
    const ik_bf16* r = table + (int64_t)id * H;
    if (VEC) {
        const uint4* rv = reinterpret_cast<const uint4*>(r);
        uint4* ov = reinterpret_cast<uint4*>(o);
        for (int64_t i = threadIdx.x; i < H / 8; i += blockDim.x) ov[i] = rv[i];
    } else {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) o[i] = r[i];
    }
}

// ---- add ---------------------------------------------------------------------------------
template <bool VEC>
__global__ void add_kernel(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b, ik_bf16* __restrict__ out,
                           int64_t n) {
    if (VEC) {
        const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n / 8) return;
        float fa[8], fb[8];
        ik_unpack8(reinterpret_cast<const uint4*>(a)[i], fa);
        ik_unpack8(reinterpret_cast<const uint4*>(b)[i], fb);
#pragma unroll
        for (int k = 0; k < 8; ++k) fa[k] += fb[k];
        reinterpret_cast<uint4*>(out)[i] = ik_pack8(fa);
    } else {
        const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        out[i] = ik_f2b(ik_b2f(a[i]) + ik_b2f(b[i]));
    }
}

// ---- silu_glu ----------------------------------------------------------------------------
IK_DEVI float silu(float g) { return ik_silu(g); }

template <bool VEC>
__global__ void silu_glu_kernel(const ik_bf16* __restrict__ g, const ik_bf16* __restrict__ u,
                                ik_bf16* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (VEC) {
        if (i >= n / 8) return;
        float fg[8], fu[8];
        ik_unpack8(reinterpret_cast<const uint4*>(g)[i], fg);
        ik_unpack8(reinterpret_cast<const uint4*>(u)[i], fu);
#pragma unroll
        for (int k = 0; k < 8; ++k) fg[k] = silu(fg[k]) * fu[k];
        reinterpret_cast<uint4*>(out)[i] = ik_pack8(fg);
    } else {
        if (i >= n) return;
        out[i] = ik_f2b(silu(ik_b2f(g[i])) * ik_b2f(u[i]));
    }
}

// ---- kv_append ---------------------------------------------------------------------------
// new[T,Nkv,D] -> cache[Nkv,cap,D] rows pos0..pos0+T-1.
template <typename CacheT, int VEC>
__global__ void kv_append_kernel(const ik_bf16* __restrict__ src, CacheT* __restrict__ cache, int64_t T,
                                 int64_t NKV, int64_t D, int64_t cap, int64_t pos0_host, const int32_t* pos_dev) {
    // device-side position (CUDA graphs: the step is captured once, the position changes per replay)
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    // one thread per VEC consecutive elements of D
    const int64_t dv = D / VEC;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NKV * dv) return;
    const int64_t d0 = (i % dv) * VEC;
    const int64_t h = (i / dv) % NKV;
    const int64_t t = i / (dv * NKV);
    const ik_bf16* s = src + (t * NKV + h) * D + d0;
    CacheT* c = cache + (h * cap + pos0 + t) * D + d0;
    if (sizeof(CacheT) == 2 && VEC == 8) {
        *reinterpret_cast<uint4*>(c) = *reinterpret_cast<const uint4*>(s);
    } else if (sizeof(CacheT) == 2) {
#pragma unroll
        for (int k = 0; k < VEC; ++k) reinterpret_cast<ik_bf16*>(c)[k] = s[k];
    } else {
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            reinterpret_cast<__nv_fp8_storage_t*>(c)[k] =
                __nv_cvt_float_to_fp8(ik_b2f(s[k]), __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// ---- rope (neox / rotate-half over the first RD values of each head) ------------------------
// One thread per (token, frequency, group of kHeads heads); the angle is computed in double
// precision (positions reach tens of thousands, float sincos would drift) and narrowed like
// the reference's tables. x and out may alias (in-place), so no __restrict__ here. With a partial
// rotary dim RD < D the values RD..D-1 pass through unchanged (`rope_pass_kernel` copies them when
// x and out are different buffers).
constexpr int kRopeHeads = 4;
__global__ void rope_kernel(const ik_bf16* x, ik_bf16* out, int64_t T, int N, int D, int RD, int64_t pos0_host,
                            float theta, const int32_t* pos_dev) {
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int half = RD / 2;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * half) return;
    const int k = (int)(i % half);
    const int64_t t = i / half;
    const double inv = pow((double)theta, -2.0 * (double)k / (double)RD);
    double sd, cd;
    sincos((double)(pos0 + t) * inv, &sd, &cd);
    const float fs = (float)sd, fc = (float)cd;
    const int n0 = blockIdx.y * kRopeHeads;
    float a[kRopeHeads], b[kRopeHeads];
#pragma unroll
    for (int j = 0; j < kRopeHeads; ++j) {
        if (n0 + j < N) {
            const int64_t base = (t * N + n0 + j) * D;
            a[j] = ik_b2f(x[base + k]);
            b[j] = ik_b2f(x[base + k + half]);
        }
    }
#pragma unroll
    for (int j = 0; j < kRopeHeads; ++j) {
        if (n0 + j < N) {
            const int64_t base = (t * N + n0 + j) * D;
            out[base + k] = ik_f2b(a[j] * fc - b[j] * fs);
            out[base + k + half] = ik_f2b(b[j] * fc + a[j] * fs);
        }
    }
}

// the un-rotated tail of each head (values RD..D-1), for partial rotary
__global__ void rope_pass_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ out, int64_t rows, int D,
                                 int RD) {
    const int tail = D - RD;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * tail) return;
    const int64_t r = i / tail;
    const int j = RD + (int)(i % tail);
    out[r * D + j] = x[r * D + j];
}

// gate_split: x[T,N,2D] -> q[T,N,D], gate[T,N,D]  (per head: D query values then D gate values)
__global__ void gate_split_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ q, ik_bf16* __restrict__ gate,
                                  int64_t rows, int D) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * D) return;
    const int64_t r = i / D;
    const int j = (int)(i % D);
    q[i] = x[r * 2 * D + j];
    gate[i] = x[r * 2 * D + D + j];
}

// sigmoid_mul: out = a * sigmoid(g)
__global__ void sigmoid_mul_kernel(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ g,
                                   ik_bf16* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = ik_b2f(g[i]);
    out[i] = ik_f2b(ik_b2f(a[i]) * (1.f / (1.f + expf(-gv))));
}

inline unsigned blocks(int64_t n) { return (unsigned)((n + kThreads - 1) / kThreads); }

}  // namespace

extern "C" {

int ik_embedding_bf16(const ik_tensor* ids, const ik_tensor* table, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 1));
    IK_TRY(ik_check(table, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t T = ids->shape[0], V = table->shape[0], H = table->shape[1];
    if (out->shape[0] != T || out->shape[1] != H) return IK_ERR_SHAPE;
    if (T == 0 || H == 0) return IK_OK;
    const bool vec = H % 8 == 0 && ik_aligned16(table->data) && ik_aligned16(out->data);
    const unsigned bd = vec ? ik_round_up((unsigned)(H / 8 < 256 ? H / 8 : 256), 32) : 256;
    cudaStream_t s = IK_STREAM(stream);
    if (vec)
        embedding_kernel<true><<<(unsigned)T, bd, 0, s>>>((const int32_t*)ids->data, (const ik_bf16*)table->data,
                                                           (ik_bf16*)out->data, V, H);
    else
        embedding_kernel<false><<<(unsigned)T, bd, 0, s>>>((const int32_t*)ids->data, (const ik_bf16*)table->data,
                                                            (ik_bf16*)out->data, V, H);
    return ik_launch_status();
}

int ik_add_bf16(const ik_tensor* a, const ik_tensor* b, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t n = ik_numel(a);
    if (ik_numel(b) != n || ik_numel(out) != n) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    if (n % 8 == 0 && ik_aligned16(a->data) && ik_aligned16(b->data) && ik_aligned16(out->data))
        add_kernel<true><<<blocks(n / 8), kThreads, 0, s>>>((const ik_bf16*)a->data, (const ik_bf16*)b->data,
                                                              (ik_bf16*)out->data, n);
    else
        add_kernel<false><<<blocks(n), kThreads, 0, s>>>((const ik_bf16*)a->data, (const ik_bf16*)b->data,
                                                           (ik_bf16*)out->data, n);
    return ik_launch_status();
}

int ik_silu_glu_bf16(const ik_tensor* gate, const ik_tensor* up, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(gate, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(up, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t n = ik_numel(gate);
    if (ik_numel(up) != n || ik_numel(out) != n) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    if (n % 8 == 0 && ik_aligned16(gate->data) && ik_aligned16(up->data) && ik_aligned16(out->data))
        silu_glu_kernel<true><<<blocks(n / 8), kThreads, 0, s>>>((const ik_bf16*)gate->data, (const ik_bf16*)up->data,
                                                                   (ik_bf16*)out->data, n);
    else
        silu_glu_kernel<false><<<blocks(n), kThreads, 0, s>>>((const ik_bf16*)gate->data, (const ik_bf16*)up->data,
                                                                (ik_bf16*)out->data, n);
    return ik_launch_status();
}

int ik_kv_append_bf16(const ik_tensor* new_kv, int64_t pos0, ik_tensor* cache, const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check(new_kv, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    if (!cache) return IK_ERR_INVALID_ARG;
    if (cache->dtype != IK_DTYPE_BF16 && cache->dtype != IK_DTYPE_FP8_E4M3) return IK_ERR_DTYPE;
    IK_TRY(ik_check(cache, cache->dtype, 3));
    const int64_t T = new_kv->shape[0], NKV = new_kv->shape[1], D = new_kv->shape[2];
    const int64_t cap = cache->shape[1];
    if (cache->shape[0] != NKV || cache->shape[2] != D || pos0 < 0 || pos0 + T > cap) return IK_ERR_SHAPE;
    if (T == 0 || D == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* src = (const ik_bf16*)new_kv->data;
    if (cache->dtype == IK_DTYPE_BF16) {
        ik_bf16* c = (ik_bf16*)cache->data;
        if (D % 8 == 0 && ik_aligned16(src) && ik_aligned16(c))
            kv_append_kernel<ik_bf16, 8><<<blocks(T * NKV * (D / 8)), kThreads, 0, s>>>(src, c, T, NKV, D, cap, pos0, pos_dev);
        else
            kv_append_kernel<ik_bf16, 1><<<blocks(T * NKV * D), kThreads, 0, s>>>(src, c, T, NKV, D, cap, pos0, pos_dev);
    } else {
        __nv_fp8_storage_t* c = (__nv_fp8_storage_t*)cache->data;
        kv_append_kernel<__nv_fp8_storage_t, 1><<<blocks(T * NKV * D), kThreads, 0, s>>>(src, c, T, NKV, D, cap, pos0, pos_dev);
    }
    return ik_launch_status();
}

int ik_rope_bf16(const ik_tensor* x, int64_t pos0, float theta, int64_t rotary_dim, ik_tensor* out,
                 const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = x->shape[0], N = x->shape[1], D = x->shape[2];
    const int64_t RD = rotary_dim == 0 ? D : rotary_dim;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D || D % 2 != 0 || pos0 < 0) return IK_ERR_SHAPE;
    if (RD < 0 || RD > D || RD % 2 != 0) return IK_ERR_INVALID_ARG;
    if (T == 0 || N == 0 || D == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    if (RD < D && x->data != out->data)
        rope_pass_kernel<<<blocks(T * N * (D - RD)), kThreads, 0, s>>>((const ik_bf16*)x->data, (ik_bf16*)out->data,
                                                                     T * N, (int)D, (int)RD);
    if (RD == 0) return ik_launch_status();
    dim3 grid(blocks(T * (RD / 2)), (unsigned)((N + kRopeHeads - 1) / kRopeHeads));
    rope_kernel<<<grid, kThreads, 0, s>>>((const ik_bf16*)x->data, (ik_bf16*)out->data, T, (int)N, (int)D, (int)RD,
                                          pos0, theta, pos_dev);
    return ik_launch_status();
}

int ik_gate_split_bf16(const ik_tensor* x, ik_tensor* q, ik_tensor* gate, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(gate, IK_DTYPE_BF16, 3));
    const int64_t T = x->shape[0], N = x->shape[1], D = x->shape[2] / 2;
    if (x->shape[2] % 2 != 0 || q->shape[0] != T || q->shape[1] != N || q->shape[2] != D || gate->shape[0] != T ||
        gate->shape[1] != N || gate->shape[2] != D)
        return IK_ERR_SHAPE;
    if (T * N * D == 0) return IK_OK;
    gate_split_kernel<<<blocks(T * N * D), kThreads, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (ik_bf16*)q->data, (ik_bf16*)gate->data, T * N, (int)D);
    return ik_launch_status();
}

int ik_sigmoid_mul_bf16(const ik_tensor* a, const ik_tensor* g, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(g, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t n = ik_numel(a);
    if (ik_numel(g) != n || ik_numel(out) != n) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    sigmoid_mul_kernel<<<blocks(n), kThreads, 0, IK_STREAM(stream)>>>((const ik_bf16*)a->data, (const ik_bf16*)g->data,
                                                                      (ik_bf16*)out->data, n);
    return ik_launch_status();
}

}  // extern "C"
