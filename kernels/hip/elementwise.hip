// Data-movement and elementwise launchers: embedding, add, add_bias, silu_glu, kv_append, rope, rope_scaled.
#include "ik_hip.h"

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

// ---- add_bias (Op::AddBias): out[r, c] = x[r, c] + bias[c], one bf16 rounding of the f32 sum -------------------------
// x and out may be the same buffer (in place), so no __restrict__ on them. VEC: 8 columns per thread (N % 8 == 0, so a
// chunk never straddles a row), the same unpack / f32 add / pack as `add_kernel<true>`.
template <bool VEC>
__global__ void add_bias_kernel(const ik_bf16* x, const ik_bf16* __restrict__ bias, ik_bf16* out, int64_t n, int64_t N) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (VEC) {
        if (i >= n / 8) return;
        const int64_t c0 = (i * 8) % N;
        float fa[8], fb[8];
        ik_unpack8(reinterpret_cast<const uint4*>(x)[i], fa);
        ik_unpack8(*reinterpret_cast<const uint4*>(bias + c0), fb);
#pragma unroll
        for (int k = 0; k < 8; ++k) fa[k] += fb[k];
        reinterpret_cast<uint4*>(out)[i] = ik_pack8(fa);
    } else {
        if (i >= n) return;
        out[i] = ik_f2b(ik_b2f(x[i]) + ik_b2f(bias[i % N]));
    }
}

// ---- silu_glu ----------------------------------------------------------------------------
IK_DEVI float silu(float g) { return g / (1.f + expf(-g)); }

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
// float -> OCP e4m3 (fn: no infinities, 0x7f/0xff NaN), round-to-nearest-even, saturating to +-448; the same
// result as __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) so the fp8 KV cache is bit-identical to cuda.
IK_DEVI uint8_t f32_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint8_t sign = (u >> 24) & 0x80u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return sign | 0x7fu;  // NaN
    const float a = fabsf(f);
    if (a >= 448.f) return sign | 0x7eu;  // saturate (also covers inf and the round-up past 448)
    if (a < 0.015625f) {  // below the smallest normal 2^-6: subnormals are multiples of 2^-9
        const uint32_t q = (uint32_t)__float2int_rn(a * 512.f);  // 0..8; 8 is the smallest normal (0x08)
        return sign | (uint8_t)q;
    }
    const uint32_t ua = __float_as_uint(a);
    int e = (int)(ua >> 23) - 127;  // -6 .. 8
    uint32_t m = (ua & 0x7fffffu) >> 20;
    const uint32_t rem = ua & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    if (m == 8u) {
        m = 0;
        ++e;
    }
    if (e > 8 || (e == 8 && m == 7u)) return sign | 0x7eu;
    return sign | (uint8_t)(((e + 7) << 3) | (int)m);
}

// new[T,Nkv,D] -> cache[Nkv,cap,D] rows pos0..pos0+T-1.
template <typename CacheT, int VEC>
__global__ void kv_append_kernel(const ik_bf16* __restrict__ src, CacheT* __restrict__ cache, int64_t T,
                                 int64_t NKV, int64_t D, int64_t cap, int64_t pos0_host, const int32_t* pos_dev) {
    // device-side position (the position changes per launch of a captured step); null = the host `pos0`
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
            reinterpret_cast<uint8_t*>(c)[k] = f32_to_e4m3(ik_b2f(s[k]));
        }
    }
}

// ---- rope (neox / rotate-half over the first RD values of each head) ------------------------
// One thread per (token, frequency, group of kHeads heads); the angle is computed in double
// precision (positions reach tens of thousands, float sincos would drift) and narrowed like
// the reference's tables. x and out may alias (in-place), so no __restrict__ here. With a partial
// rotary dim RD < D the values RD..D-1 pass through unchanged: when x and out are different buffers the rope
// kernel copies them itself (`copy_tail`; thread k takes the tail values k, k + RD/2, ... of its heads, which covers
// each tail once). `rope_pass_kernel` is only used for RD == 0 (no rotation at all).
constexpr int kRopeHeads = 4;
__global__ void rope_kernel(const ik_bf16* x, ik_bf16* out, int64_t T, int N, int D, int RD, int64_t pos0_host,
                            float theta, const int32_t* pos_dev, int copy_tail) {
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
            if (copy_tail)
                for (int m = RD + k; m < D; m += half) out[base + m] = x[base + m];
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

// ---- rope_scaled (Op::RopeScaled): `rope_kernel` with a caller-supplied frequency table ------------------------------
// The angle of pair k at token t is (pos0 + t) * inv_freq[k] (f32 table widened to double, like the reference), and both
// sin and cos are multiplied by `mscale` in double before the narrowing to float (llama3 / yarn scaling). Everything else
// (grid, head grouping, the two products, in-place aliasing, the in-kernel tail copy `copy_tail`) is `rope_kernel`.
__global__ void rope_scaled_kernel(const ik_bf16* x, const float* __restrict__ inv_freq, ik_bf16* out, int64_t T,
                                   int N, int D, int RD, int64_t pos0_host, float mscale, const int32_t* pos_dev,
                                   int copy_tail) {
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int half = RD / 2;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * half) return;
    const int k = (int)(i % half);
    const int64_t t = i / half;
    double sd, cd;
    sincos((double)(pos0 + t) * (double)inv_freq[k], &sd, &cd);
    const float fs = (float)(sd * (double)mscale), fc = (float)(cd * (double)mscale);
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
            if (copy_tail)
                for (int m = RD + k; m < D; m += half) out[base + m] = x[base + m];
        }
    }
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

// ---- qk_norm + rope + kv_append (Op::QkNormRopeKv) ---------------------------------------
// The attention prologue in one launch instead of six (q norm, q rope, k norm, k rope, k append, v append). Grid (T, Nkv),
// block G + 1 waves (G = N / Nkv query heads per kv head): wave w < G takes query head kvh * G + w, wave G the k head kvh
// and the v copy. Each wave repeats, for its head, the arithmetic of the launch it replaces:
//  * qk_norm = rmsnorm_kernel<false, true> on one row of D: a 32-thread block (launch_rows' size for D / 8 <= 32 items),
//    lane i sums the squares of values 8i..8i+7, ik_block_sum of one wave is 0 + the warp sum; the normalised row is
//    rounded to bf16 (what qk_norm stores) and kept in LDS;
//  * rope_kernel on that bf16 row: the angle of each (token, frequency) in double, narrowed to float (computed once per
//    block here, as rope_kernel computes it once per group of heads), the same two products per pair, the un-rotated
//    tail RD..D-1 passed through;
//  * kv_append of the roped k and of v: a bf16 copy, or f32_to_e4m3 for an fp8 cache.
// So every output bit is the unfused chain's (tests/qk_rope_kv.rs). Needs D % 8 == 0, D <= 256 and 16-byte aligned rows
// (the unfused qk_norm's vector path; the launcher refuses anything else).
constexpr int kQkrMaxGroup = 8;
constexpr int kQkrMaxD = 256;

template <typename CacheT>
IK_DEVI void qkr_store(CacheT* c, ik_bf16 v) {
    if constexpr (sizeof(CacheT) == 2) *reinterpret_cast<ik_bf16*>(c) = v;
    else *reinterpret_cast<uint8_t*>(c) = f32_to_e4m3(ik_b2f(v));
}

template <typename CacheT>
__global__ void __launch_bounds__((kQkrMaxGroup + 1) * IK_WARP)
qk_norm_rope_kv_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ k, const ik_bf16* __restrict__ v,
                       const ik_bf16* __restrict__ qw, const ik_bf16* __restrict__ kw, ik_bf16* __restrict__ q_out,
                       CacheT* __restrict__ kcache, CacheT* __restrict__ vcache, int N, int NKV, int D, int RD,
                       int64_t cap, float eps, float woff, float theta, int64_t pos0_host, const int32_t* pos_dev) {
    __shared__ __align__(16) ik_bf16 nrm[kQkrMaxGroup + 1][kQkrMaxD];
    __shared__ float fsn[kQkrMaxD / 2], fcs[kQkrMaxD / 2];
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int G = N / NKV;
    const int lane = threadIdx.x % IK_WARP, w = threadIdx.x / IK_WARP;
    const int half = RD / 2;
    for (int f = threadIdx.x; f < half; f += blockDim.x) {
        const double inv = pow((double)theta, -2.0 * (double)f / (double)RD);
        double sd, cd;
        sincos((double)(pos0 + t) * inv, &sd, &cd);
        fsn[f] = (float)sd;
        fcs[f] = (float)cd;
    }
    const bool is_k = w == G;
    const ik_bf16* xr = is_k ? k + (t * NKV + kvh) * D : q + (t * N + (int64_t)kvh * G + w) * D;
    const ik_bf16* wt = is_k ? kw : qw;
    ik_bf16* nr = nrm[w];
    const int nv = D / 8;
    float ss = 0.f;
    for (int i = lane; i < nv; i += IK_WARP) {
        float f[8];
        ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
#pragma unroll
        for (int e = 0; e < 8; ++e) ss += f[e] * f[e];
    }
    ss = ik_warp_sum(ss);
    float tot = 0.f;  // ik_block_sum over one wave: 0 + the wave's sum
    tot += ss;
    const float r = rsqrtf(tot / (float)D + eps);
    for (int i = lane; i < nv; i += IK_WARP) {
        float f[8], g[8];
        ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
        ik_unpack8(reinterpret_cast<const uint4*>(wt)[i], g);
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = f[e] * r * (g[e] + woff);
        reinterpret_cast<uint4*>(nr)[i] = ik_pack8(f);
    }
    __syncthreads();
    ik_bf16 *dst_q = nullptr;
    CacheT* dst_k = nullptr;
    if (is_k) dst_k = kcache + ((int64_t)kvh * cap + pos0 + t) * D;
    else dst_q = q_out + (t * N + (int64_t)kvh * G + w) * D;
    for (int f = lane; f < half; f += IK_WARP) {
        const float a = ik_b2f(nr[f]), b = ik_b2f(nr[f + half]);
        const float fs = fsn[f], fc = fcs[f];
        // rope_kernel's contraction, spelled out (its ISA: lo = fma(a, fc, -(b * fs)), hi = fma(b, fc, a * fs)); left to
        // the compiler, this kernel fused the other product of hi and flipped a bf16 rounding now and then
        float lof, hif;
        {
#pragma clang fp contract(off)
            lof = __builtin_fmaf(a, fc, -(b * fs));
            hif = __builtin_fmaf(b, fc, a * fs);
        }
        const ik_bf16 lo = ik_f2b(lof);
        const ik_bf16 hi = ik_f2b(hif);
        if (is_k) {
            qkr_store(dst_k + f, lo);
            qkr_store(dst_k + f + half, hi);
        } else {
            dst_q[f] = lo;
            dst_q[f + half] = hi;
        }
    }
    for (int m = RD + lane; m < D; m += IK_WARP) {
        if (is_k) qkr_store(dst_k + m, nr[m]);
        else dst_q[m] = nr[m];
    }
    if (is_k) {
        const ik_bf16* vr = v + (t * NKV + kvh) * D;
        CacheT* dv = vcache + ((int64_t)kvh * cap + pos0 + t) * D;
        for (int m = lane; m < D; m += IK_WARP) qkr_store(dv + m, vr[m]);
    }
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
    hipStream_t s = IK_STREAM(stream);
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
    hipStream_t s = IK_STREAM(stream);
    if (n % 8 == 0 && ik_aligned16(a->data) && ik_aligned16(b->data) && ik_aligned16(out->data))
        add_kernel<true><<<blocks(n / 8), kThreads, 0, s>>>((const ik_bf16*)a->data, (const ik_bf16*)b->data,
                                                              (ik_bf16*)out->data, n);
    else
        add_kernel<false><<<blocks(n), kThreads, 0, s>>>((const ik_bf16*)a->data, (const ik_bf16*)b->data,
                                                           (ik_bf16*)out->data, n);
    return ik_launch_status();
}

int ik_add_bias_bf16(const ik_tensor* x, const ik_tensor* bias, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(bias, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t R = x->shape[0], N = x->shape[1];
    if (bias->shape[0] != N || out->shape[0] != R || out->shape[1] != N) return IK_ERR_SHAPE;
    const int64_t n = R * N;
    if (n == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    if (N % 8 == 0 && ik_aligned16(x->data) && ik_aligned16(bias->data) && ik_aligned16(out->data))
        add_bias_kernel<true><<<blocks(n / 8), kThreads, 0, s>>>((const ik_bf16*)x->data, (const ik_bf16*)bias->data,
                                                                   (ik_bf16*)out->data, n, N);
    else
        add_bias_kernel<false><<<blocks(n), kThreads, 0, s>>>((const ik_bf16*)x->data, (const ik_bf16*)bias->data,
                                                                (ik_bf16*)out->data, n, N);
    return ik_launch_status();
}

int ik_silu_glu_bf16(const ik_tensor* gate, const ik_tensor* up, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(gate, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(up, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t n = ik_numel(gate);
    if (ik_numel(up) != n || ik_numel(out) != n) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
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
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* src = (const ik_bf16*)new_kv->data;
    if (cache->dtype == IK_DTYPE_BF16) {
        ik_bf16* c = (ik_bf16*)cache->data;
        if (D % 8 == 0 && ik_aligned16(src) && ik_aligned16(c))
            kv_append_kernel<ik_bf16, 8><<<blocks(T * NKV * (D / 8)), kThreads, 0, s>>>(src, c, T, NKV, D, cap, pos0, pos_dev);
        else
            kv_append_kernel<ik_bf16, 1><<<blocks(T * NKV * D), kThreads, 0, s>>>(src, c, T, NKV, D, cap, pos0, pos_dev);
    } else {
        uint8_t* c = (uint8_t*)cache->data;
        kv_append_kernel<uint8_t, 1><<<blocks(T * NKV * D), kThreads, 0, s>>>(src, c, T, NKV, D, cap, pos0, pos_dev);
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
    hipStream_t s = IK_STREAM(stream);
    const bool copy_tail = RD < D && x->data != out->data;
    if (copy_tail && RD == 0)
        rope_pass_kernel<<<blocks(T * N * (D - RD)), kThreads, 0, s>>>((const ik_bf16*)x->data, (ik_bf16*)out->data,
                                                                     T * N, (int)D, (int)RD);
    if (RD == 0) return ik_launch_status();
    dim3 grid(blocks(T * (RD / 2)), (unsigned)((N + kRopeHeads - 1) / kRopeHeads));
    rope_kernel<<<grid, kThreads, 0, s>>>((const ik_bf16*)x->data, (ik_bf16*)out->data, T, (int)N, (int)D, (int)RD,
                                          pos0, theta, pos_dev, copy_tail ? 1 : 0);
    return ik_launch_status();
}

int ik_rope_scaled_bf16(const ik_tensor* x, const ik_tensor* inv_freq, int64_t pos0, float rope_mscale,
                        int64_t rotary_dim, ik_tensor* out, const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(inv_freq, IK_DTYPE_F32, 1));
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = x->shape[0], N = x->shape[1], D = x->shape[2];
    const int64_t RD = rotary_dim == 0 ? D : rotary_dim;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D || D % 2 != 0 || pos0 < 0) return IK_ERR_SHAPE;
    if (RD < 0 || RD > D || RD % 2 != 0) return IK_ERR_INVALID_ARG;
    if (inv_freq->shape[0] != RD / 2) return IK_ERR_SHAPE;
    if (!(rope_mscale > 0.f) || rope_mscale > 3.4028234e38f) return IK_ERR_INVALID_ARG;  // positive and finite
    if (T == 0 || N == 0 || D == 0) return IK_OK;
    if (((uintptr_t)inv_freq->data & 3u) != 0) return IK_ERR_INVALID_ARG;
    hipStream_t s = IK_STREAM(stream);
    const bool copy_tail = RD < D && x->data != out->data;
    if (copy_tail && RD == 0)
        rope_pass_kernel<<<blocks(T * N * (D - RD)), kThreads, 0, s>>>((const ik_bf16*)x->data, (ik_bf16*)out->data,
                                                                     T * N, (int)D, (int)RD);
    if (RD == 0) return ik_launch_status();
    dim3 grid(blocks(T * (RD / 2)), (unsigned)((N + kRopeHeads - 1) / kRopeHeads));
    rope_scaled_kernel<<<grid, kThreads, 0, s>>>((const ik_bf16*)x->data, (const float*)inv_freq->data,
                                                 (ik_bf16*)out->data, T, (int)N, (int)D, (int)RD, pos0, rope_mscale,
                                                 pos_dev, copy_tail ? 1 : 0);
    return ik_launch_status();
}

int ik_qk_norm_rope_kv_bf16(const ik_tensor* q, const ik_tensor* k, const ik_tensor* v, const ik_tensor* qw,
                            const ik_tensor* kw, float eps, float w_offset, int64_t pos0, float theta, int64_t rotary_dim,
                            ik_tensor* q_out, ik_tensor* kcache, ik_tensor* vcache, const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(k, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(v, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(qw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(kw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(q_out, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    if (!kcache || !vcache) return IK_ERR_INVALID_ARG;
    if ((kcache->dtype != IK_DTYPE_BF16 && kcache->dtype != IK_DTYPE_FP8_E4M3) || vcache->dtype != kcache->dtype)
        return IK_ERR_DTYPE;
    IK_TRY(ik_check(kcache, kcache->dtype, 3));
    IK_TRY(ik_check(vcache, vcache->dtype, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2], NKV = k->shape[1], cap = kcache->shape[1];
    const int64_t RD = rotary_dim == 0 ? D : rotary_dim;
    if (k->shape[0] != T || k->shape[2] != D || v->shape[0] != T || v->shape[1] != NKV || v->shape[2] != D ||
        q_out->shape[0] != T || q_out->shape[1] != N || q_out->shape[2] != D || qw->shape[0] != D || kw->shape[0] != D ||
        kcache->shape[0] != NKV || kcache->shape[2] != D || vcache->shape[0] != NKV || vcache->shape[1] != cap ||
        vcache->shape[2] != D || pos0 < 0 || pos0 + T > cap)
        return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || N / NKV > kQkrMaxGroup || D % 8 != 0 || D > kQkrMaxD || RD < 0 || RD > D || RD % 2 != 0)
        return IK_ERR_UNSUPPORTED;
    if (T == 0 || N == 0) return IK_OK;
    // the unfused qk_norm takes its vector path only for 16-byte aligned rows; this kernel is that path
    if (!ik_aligned16(q->data) || !ik_aligned16(k->data) || !ik_aligned16(qw->data) || !ik_aligned16(kw->data) ||
        !ik_aligned16(q_out->data))
        return IK_ERR_STRIDE;
    if (T > 0x7fffffffLL) return IK_ERR_SHAPE;
    const dim3 grid((unsigned)T, (unsigned)NKV);
    const unsigned threads = (unsigned)((N / NKV + 1) * IK_WARP);
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16 *qp = (const ik_bf16*)q->data, *kp = (const ik_bf16*)k->data, *vp = (const ik_bf16*)v->data;
    const ik_bf16 *qwp = (const ik_bf16*)qw->data, *kwp = (const ik_bf16*)kw->data;
    if (kcache->dtype == IK_DTYPE_BF16)
        qk_norm_rope_kv_kernel<ik_bf16><<<grid, threads, 0, s>>>(qp, kp, vp, qwp, kwp, (ik_bf16*)q_out->data,
                                                                  (ik_bf16*)kcache->data, (ik_bf16*)vcache->data, (int)N,
                                                                  (int)NKV, (int)D, (int)RD, cap, eps, w_offset, theta, pos0,
                                                                  pos_dev);
    else
        qk_norm_rope_kv_kernel<uint8_t><<<grid, threads, 0, s>>>(qp, kp, vp, qwp, kwp, (ik_bf16*)q_out->data,
                                                                  (uint8_t*)kcache->data, (uint8_t*)vcache->data, (int)N,
                                                                  (int)NKV, (int)D, (int)RD, cap, eps, w_offset, theta, pos0,
                                                                  pos_dev);
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
