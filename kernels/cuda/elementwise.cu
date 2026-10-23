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

// Row-table launch (batched decode, `OpCall::rows`): row t of new[T,Nkv,D] is the one token of sequence t, appended at
// position pos_dev[t] of the cache whose device address is entry [t][tindex] of the row table: kv_append_kernel's copy for
// each row's own T = 1 call.
template <typename CacheT, int VEC>
__global__ void kv_append_rows_kernel(const ik_bf16* __restrict__ src, const uint8_t* __restrict__ tab, int64_t tstride,
                                      int64_t tindex, int64_t T, int64_t NKV, int64_t D, int64_t cap,
                                      const int32_t* __restrict__ pos_dev) {
    const int64_t dv = D / VEC;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NKV * dv) return;
    const int64_t d0 = (i % dv) * VEC;
    const int64_t h = (i / dv) % NKV;
    const int64_t t = i / (dv * NKV);
    CacheT* cache = reinterpret_cast<CacheT*>(reinterpret_cast<const uint64_t*>(tab)[t * tstride + tindex]);
    const int64_t pos0 = (int64_t)pos_dev[t];
    const ik_bf16* s = src + (t * NKV + h) * D + d0;
    CacheT* c = cache + (h * cap + pos0) * D + d0;
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
// pos_per_token (batched decode rows): `pos_dev` holds one position per token, token t sits at pos_dev[t] (each row is the
// T = 1 call of its own sequence); otherwise token t sits at pos0 + t.
__global__ void rope_kernel(const ik_bf16* x, ik_bf16* out, int64_t T, int N, int D, int RD, int64_t pos0_host,
                            float theta, const int32_t* pos_dev, int pos_per_token = 0) {
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int half = RD / 2;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * half) return;
    const int k = (int)(i % half);
    const int64_t t = i / half;
    const int64_t pos = pos_per_token ? (int64_t)pos_dev[t] : pos0 + t;
    const double inv = pow((double)theta, -2.0 * (double)k / (double)RD);
    double sd, cd;
    sincos((double)pos * inv, &sd, &cd);
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


// ---- extended (Gemma 4) variants: scaled embedding / add, gelu_glu, proportional rope ------------------------------
// Separate kernels and launchers (`*_scaled`, `gelu_glu`, `rope_prop`) so the plain ones above keep their exact bits.

// out = round_bf16(f32(table row) * scale): one rounding (Gemma's embedding scale)
__global__ void embedding_scaled_kernel(const int32_t* __restrict__ ids, const ik_bf16* __restrict__ table,
                                        ik_bf16* __restrict__ out, int64_t V, int64_t H, float scale) {
    const int64_t t = blockIdx.x;
    const int32_t id = ids[t];
    ik_bf16* o = out + t * H;
    if (id < 0 || id >= V) {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) o[i] = ik_f2b(0.f);
        return;
    }
    const ik_bf16* r = table + (int64_t)id * H;
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) o[i] = ik_f2b(ik_b2f(r[i]) * scale);
}

// out = round_bf16(f32(a + b) * scale): a + b in f32, one rounding (Gemma's layer_scalar)
__global__ void add_scaled_kernel(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
                                  ik_bf16* __restrict__ out, int64_t n, float scale) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = ik_f2b((ik_b2f(a[i]) + ik_b2f(b[i])) * scale);
}

// gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
IK_DEVI float gelu_tanh_f(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

template <bool VEC>
__global__ void gelu_glu_kernel(const ik_bf16* __restrict__ g, const ik_bf16* __restrict__ u,
                                ik_bf16* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (VEC) {
        if (i >= n / 8) return;
        float fg[8], fu[8];
        ik_unpack8(reinterpret_cast<const uint4*>(g)[i], fg);
        ik_unpack8(reinterpret_cast<const uint4*>(u)[i], fu);
#pragma unroll
        for (int k = 0; k < 8; ++k) fg[k] = gelu_tanh_f(fg[k]) * fu[k];
        reinterpret_cast<uint4*>(out)[i] = ik_pack8(fg);
    } else {
        if (i >= n) return;
        out[i] = ik_f2b(gelu_tanh_f(ik_b2f(g[i])) * ik_b2f(u[i]));
    }
}

// proportional rope (HF rope_type "proportional", Gemma 4 global layers): pair (j, j + D/2) for j < RD/2 is rotated by
// pos * theta^(-2j/D) (the exponent's denominator is the full head dim D); every other value is copied. x and out may alias.
__global__ void rope_prop_kernel(const ik_bf16* x, ik_bf16* out, int64_t T, int N, int D, int RD, int64_t pos0_host,
                                 float theta, const int32_t* pos_dev) {
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int nrot = RD / 2;
    const int half = D / 2;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * nrot) return;
    const int k = (int)(i % nrot);
    const int64_t t = i / nrot;
    const double inv = pow((double)theta, -2.0 * (double)k / (double)D);
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

// the values a proportional rope leaves alone: [RD/2, D/2) and [D/2 + RD/2, D) of each head (x != out only)
__global__ void rope_prop_pass_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ out, int64_t rows, int D,
                                      int RD) {
    const int half = D / 2, nrot = RD / 2;
    const int per = half - nrot;  // untouched values in each half
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 2 * per) return;
    const int64_t r = i / (2 * per);
    const int q = (int)(i % (2 * per));
    const int j = (q < per ? nrot + q : half + nrot + (q - per));
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

// Row-table kv append: `cache0` is row 0's cache (shape / dtype carrier); sequence t's cache is the buffer at entry
// [t][index] of `tab`, its position pos[t] (the cache must have room: the caller's bound, as for the one-row launch).
int ik_kv_append_bf16_rows(const ik_tensor* new_kv, const ik_tensor* cache0, const ik_tensor* tab, int64_t stride,
                           int64_t index, const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check(new_kv, IK_DTYPE_BF16, 3));
    if (!cache0) return IK_ERR_INVALID_ARG;
    if (cache0->dtype != IK_DTYPE_BF16 && cache0->dtype != IK_DTYPE_FP8_E4M3) return IK_ERR_DTYPE;
    IK_TRY(ik_check(cache0, cache0->dtype, 3));
    const int64_t T = new_kv->shape[0], NKV = new_kv->shape[1], D = new_kv->shape[2];
    const int64_t cap = cache0->shape[1];
    if (cache0->shape[0] != NKV || cache0->shape[2] != D) return IK_ERR_SHAPE;
    if (!pos || !pos->data || pos->dtype != IK_DTYPE_I32 || ik_numel(pos) != T) return IK_ERR_INVALID_ARG;
    if (!tab || !tab->data || tab->dtype != IK_DTYPE_U8 || tab->ndim != 3 || tab->shape[0] != T || tab->shape[1] != stride ||
        tab->shape[2] != 8 || index < 0 || index >= stride || (reinterpret_cast<uintptr_t>(tab->data) & 7u))
        return IK_ERR_INVALID_ARG;
    if (T == 0 || D == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* src = (const ik_bf16*)new_kv->data;
    const int32_t* pos_dev = (const int32_t*)pos->data;
    const uint8_t* tp = (const uint8_t*)tab->data;
    if (cache0->dtype == IK_DTYPE_BF16) {
        // the one-row launch picks the 16-byte copy from the alignment of its own cache; every cache buffer here comes from the
        // allocator (256-byte aligned) and a position row is D * 2 bytes, so D % 8 == 0 decides it for all rows alike
        if (D % 8 == 0 && ik_aligned16(src) && ik_aligned16(cache0->data))
            kv_append_rows_kernel<ik_bf16, 8><<<blocks(T * NKV * (D / 8)), kThreads, 0, s>>>(src, tp, stride, index, T, NKV, D, cap, pos_dev);
        else
            kv_append_rows_kernel<ik_bf16, 1><<<blocks(T * NKV * D), kThreads, 0, s>>>(src, tp, stride, index, T, NKV, D, cap, pos_dev);
    } else {
        kv_append_rows_kernel<__nv_fp8_storage_t, 1><<<blocks(T * NKV * D), kThreads, 0, s>>>(src, tp, stride, index, T, NKV, D, cap, pos_dev);
    }
    return ik_launch_status();
}

int ik_rope_bf16(const ik_tensor* x, int64_t pos0, float theta, int64_t rotary_dim, ik_tensor* out,
                 const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    // one position (token t at pos + t) or, for a batch of decode rows, one per token
    const bool per_token = pos && x->shape[0] > 1 && ik_numel(pos) == x->shape[0];
    if (!per_token) IK_TRY(ik_check_pos(pos));
    else if (!pos->data || pos->dtype != IK_DTYPE_I32) return IK_ERR_INVALID_ARG;
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
                                          pos0, theta, pos_dev, per_token ? 1 : 0);
    return ik_launch_status();
}

int ik_embedding_scaled_bf16(const ik_tensor* ids, const ik_tensor* table, ik_tensor* out, float scale, void* stream) {
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 1));
    IK_TRY(ik_check(table, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t T = ids->shape[0], V = table->shape[0], H = table->shape[1];
    if (out->shape[0] != T || out->shape[1] != H) return IK_ERR_SHAPE;
    if (T == 0 || H == 0) return IK_OK;
    const unsigned bd = ik_round_up((unsigned)(H < 256 ? H : 256), 32);
    embedding_scaled_kernel<<<(unsigned)T, bd, 0, IK_STREAM(stream)>>>((const int32_t*)ids->data,
                                                                       (const ik_bf16*)table->data,
                                                                       (ik_bf16*)out->data, V, H, scale);
    return ik_launch_status();
}

int ik_add_scaled_bf16(const ik_tensor* a, const ik_tensor* b, ik_tensor* out, float scale, void* stream) {
    IK_TRY(ik_check(a, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(b, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t n = ik_numel(a);
    if (ik_numel(b) != n || ik_numel(out) != n) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    add_scaled_kernel<<<blocks(n), kThreads, 0, IK_STREAM(stream)>>>((const ik_bf16*)a->data, (const ik_bf16*)b->data,
                                                                     (ik_bf16*)out->data, n, scale);
    return ik_launch_status();
}

int ik_gelu_glu_bf16(const ik_tensor* gate, const ik_tensor* up, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(gate, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(up, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t n = ik_numel(gate);
    if (ik_numel(up) != n || ik_numel(out) != n) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    if (n % 8 == 0 && ik_aligned16(gate->data) && ik_aligned16(up->data) && ik_aligned16(out->data))
        gelu_glu_kernel<true><<<blocks(n / 8), kThreads, 0, s>>>((const ik_bf16*)gate->data, (const ik_bf16*)up->data,
                                                                   (ik_bf16*)out->data, n);
    else
        gelu_glu_kernel<false><<<blocks(n), kThreads, 0, s>>>((const ik_bf16*)gate->data, (const ik_bf16*)up->data,
                                                                (ik_bf16*)out->data, n);
    return ik_launch_status();
}

// proportional rope: pair (j, j + D/2), j < rotary_dim / 2, angle pos * theta^(-2j/D); the rest copied
int ik_rope_prop_bf16(const ik_tensor* x, int64_t pos0, float theta, int64_t rotary_dim, ik_tensor* out,
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
        rope_prop_pass_kernel<<<blocks(T * N * (D - RD)), kThreads, 0, s>>>((const ik_bf16*)x->data,
                                                                            (ik_bf16*)out->data, T * N, (int)D,
                                                                            (int)RD);
    if (RD == 0) return ik_launch_status();
    dim3 grid(blocks(T * (RD / 2)), (unsigned)((N + kRopeHeads - 1) / kRopeHeads));
    rope_prop_kernel<<<grid, kThreads, 0, s>>>((const ik_bf16*)x->data, (ik_bf16*)out->data, T, (int)N, (int)D,
                                               (int)RD, pos0, theta, pos_dev);
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
