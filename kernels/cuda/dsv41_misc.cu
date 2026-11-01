// DeepSeek-V4.1 element-wise and small ops: moe_route (Gate.forward, model.py:811), silu_glu_clamp (Expert.forward,
// model.py:832), act_qdq (act_quant / fp4_act_quant, kernel.py:98 / 184), rope_tail (apply_rotary_emb, model.py:392),
// window_kv (Attention._window_kv, model.py:701), compress_pool (Compressor.forward, model.py:429-486, ratio > 1).
// Contract: docs/design/dsv41-ops.md. Written from the official inference code (MIT, DeepSeek).
#include "dsv41_common.cuh"

using namespace dsv41;

namespace {

constexpr int kThreads = 256;

// ---------------------------------------------------------------------------------------------------------------- moe_route

struct Pick {
    float v;
    int i;  // -1: none
};

// (va, ia) precedes (vb, ib): value descending, ties to the lower index
IK_DEVI bool before(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

IK_DEVI Pick better(Pick a, Pick b) {
    if (a.i < 0) return b;
    if (b.i < 0) return a;
    return before(a.v, a.i, b.v, b.i) ? a : b;
}

// one warp per row
__global__ void moe_route_kernel(const float* __restrict__ lg, const float* __restrict__ bias, int64_t E, int K, float scale, int norm,
                                 float temp, int sf, float* __restrict__ w, int* __restrict__ ids) {
    extern __shared__ float sm[];
    float* sc = sm;
    float* bi = sm + E;
    __shared__ int pick[32];
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const float* l = lg + row * E;
    float mx = -INFINITY;
    for (int64_t e = lane; e < E; e += 32) {
        const float v = l[e] / temp;
        float s;
        if (sf == 2) {
            s = sqrtf(fmaxf(v, 0.f) + log1pf(expf(-fabsf(v))));
        } else if (sf == 1) {
            s = 1.f / (1.f + expf(-v));
        } else {
            s = v;
            mx = fmaxf(mx, v);
        }
        sc[e] = s;
    }
    if (sf == 0) {
        mx = ik_warp_max(mx);
        __syncwarp();
        float sum = 0.f;
        for (int64_t e = lane; e < E; e += 32) {
            const float x = expf(sc[e] - mx);
            sc[e] = x;
            sum += x;
        }
        sum = ik_warp_sum(sum);
        __syncwarp();
        for (int64_t e = lane; e < E; e += 32) sc[e] = sc[e] / sum;
    }
    for (int64_t e = lane; e < E; e += 32) bi[e] = sc[e] + bias[e];
    __syncwarp();
    // K rounds, each the best element strictly after the previous pick in the total order (ties cannot repeat a pick)
    float pv = INFINITY;
    int pi = -1;
    for (int k = 0; k < K; ++k) {
        Pick b{0.f, -1};
        for (int64_t e = lane; e < E; e += 32) {
            const float v = bi[e];
            if (v != v) continue;
            if (before(pv, pi, v, (int)e)) b = better(b, Pick{v, (int)e});
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            Pick x;
            x.v = __shfl_xor_sync(0xffffffffu, b.v, o);
            x.i = __shfl_xor_sync(0xffffffffu, b.i, o);
            b = better(b, x);
        }
        if (lane == 0) pick[k] = b.i;
        pv = b.v;
        pi = b.i;
    }
    __syncwarp();
    if (lane == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += pick[k] >= 0 ? sc[pick[k]] : 0.f;
        const float den = (norm && K > 1) ? sum + 1e-20f : 1.f;
        for (int k = 0; k < K; ++k) {
            const int id = pick[k];
            const float v = id >= 0 ? sc[id] : 0.f;
            w[row * K + k] = (norm && K > 1 ? v / den : v) * scale;
            ids[row * K + k] = id;
        }
    }
}

// ----------------------------------------------------------------------------------------------------------- silu_glu_clamp

template <typename Tg, typename Tu, typename To>
__global__ void silu_glu_clamp_kernel(const Tg* __restrict__ g, const Tu* __restrict__ u, To* __restrict__ y, int64_t n, float l) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float gv = ld(g, i), uv = ld(u, i);
        if (l > 0.f) {
            gv = fminf(gv, l);
            uv = fminf(fmaxf(uv, -l), l);
        }
        st(y, i, ik_silu(gv) * uv);
    }
}

// ----------------------------------------------------------------------------------------------------------------- act_qdq

// e4m3 round trip, round to nearest even, saturating at 448 (intarsia_core::dtype::f32_to_fp8_e4m3)
IK_DEVI float e4m3_rt(float v) {
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)));
}

// e2m1 round trip (intarsia_core::quant::nvfp4::f32_to_fp4 / fp4_to_f32): nearest, ties to the even code, saturating at 6
IK_DEVI float e2m1_rt(float x) {
    const float a = fabsf(x);
    float q;
    if (a < 0.25f || a == 0.25f) q = 0.f;
    else if (a < 0.75f) q = 0.5f;
    else if (a <= 1.25f) q = 1.f;
    else if (a < 1.75f) q = 1.5f;
    else if (a <= 2.5f) q = 2.f;
    else if (a < 3.5f) q = 3.f;
    else if (a <= 5.f) q = 4.f;
    else q = 6.f;
    return (x < 0.f && q != 0.f) ? -q : q;
}

// 2^ceil(log2(x)) from the f32 bits (`fast_round_scale`, the reference's pow2_ceil)
IK_DEVI float pow2_ceil(float x) {
    const unsigned b = __float_as_uint(x);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0 ? 1 : 0);
    e = min(max(e + 127, 1), 254);
    return __uint_as_float((unsigned)e << 23);
}

struct QdqP {
    float fmax, floor_;
    int fmt, sfmt;
};

IK_DEVI float qdq_scale(float amax, const QdqP& p) {
    amax = fmaxf(amax, p.floor_);
    if (p.sfmt == 0) return pow2_ceil(amax * (1.f / p.fmax));
    if (p.sfmt == 1) return e4m3_rt(amax / p.fmax);
    return amax * (1.f / p.fmax);
}

IK_DEVI float qdq_one(float v, float s, const QdqP& p) {
    const float c = fminf(fmaxf(v / s, -p.fmax), p.fmax);
    const float q = p.fmt == 0 ? e4m3_rt(c) : e2m1_rt(c);
    return q * s;
}

// blocks of b <= 32 elements (a power of two): b consecutive lanes share a block
template <typename Ti, typename To>
__global__ void qdq_small(const Ti* __restrict__ x, To* __restrict__ y, int64_t n, int b, QdqP p) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? ld(x, i) : 0.f;
    float a = fabsf(v);
    for (int o = b >> 1; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float s = qdq_scale(a, p);
    if (i < n) st(y, i, qdq_one(v, s, p));
}

// blocks of b = 32 k elements: one warp per block
template <typename Ti, typename To>
__global__ void qdq_wide(const Ti* __restrict__ x, To* __restrict__ y, int64_t nblk, int b, QdqP p) {
    const int64_t blk = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (blk >= nblk) return;
    const int64_t o = blk * b;
    float a = 0.f;
    for (int j = lane; j < b; j += 32) a = fmaxf(a, fabsf(ld(x, o + j)));
    a = ik_warp_max(a);
    const float s = qdq_scale(a, p);
    for (int j = lane; j < b; j += 32) st(y, o + j, qdq_one(ld(x, o + j), s, p));
}

// ---------------------------------------------------------------------------------------------------------------- rope_tail

template <typename Ti, typename To>
__global__ void rope_tail_kernel(const Ti* __restrict__ x, const float* __restrict__ f, To* __restrict__ y, int64_t T, int64_t N,
                                 int64_t Dh, int64_t half, int64_t pos0, int64_t stride, int inverse, const int* __restrict__ dpos) {
    const int64_t n = T * N * Dh;
    const int64_t base = Dh - 2 * half;
    const int64_t p0 = dpos ? (int64_t)*dpos : pos0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t e = i % Dh;
        if (e < base) {
            st(y, i, ld(x, i));
            continue;
        }
        const int64_t t = i / (N * Dh);
        const int64_t j = (e - base) >> 1;
        const int64_t pi = i - ((e - base) & 1);
        const float re = ld(x, pi), im = ld(x, pi + 1);
        const float ang = (float)(p0 + t * stride) * f[j];
        float sn, cs;
        sincosf(ang, &sn, &cs);
        if (inverse) sn = -sn;
        st(y, i, ((e - base) & 1) ? re * sn + im * cs : re * cs - im * sn);
    }
}

// ---------------------------------------------------------------------------------------------------------------- window_kv

template <typename T>
__global__ void window_build(const T* __restrict__ kv, const T* __restrict__ ring, T* __restrict__ win, int64_t Tn, int64_t D, int64_t W,
                             int64_t Wc, int64_t pos0, const int* __restrict__ dpos) {
    const int64_t n = (W + Tn) * D;
    const int64_t p0 = dpos ? (int64_t)*dpos : pos0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / D, d = i % D;
        if (r >= W) {
            win[i] = kv[i - W * D];
        } else {
            const int64_t p = p0 - W + r;
            win[i] = p >= 0 ? ring[(p % Wc) * D + d] : T(0.f);
        }
    }
}

template <typename T>
__global__ void ring_write(const T* __restrict__ src, T* __restrict__ ring, int64_t Tn, int64_t D, int64_t Rc, int64_t pos0,
                           const int* __restrict__ dpos) {
    const int64_t j0 = Tn > Rc ? Tn - Rc : 0;
    const int64_t n = (Tn - j0) * D;
    const int64_t p0 = dpos ? (int64_t)*dpos : pos0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t j = j0 + i / D, d = i % D;
        ring[((p0 + j) % Rc) * D + d] = src[j * D + d];
    }
}

// ------------------------------------------------------------------------------------------------------------ compress_pool

template <typename To>
__global__ void compress_pool_kernel(const float* __restrict__ kv, const float* __restrict__ sc, const float* __restrict__ rkv,
                                     const float* __restrict__ rsc, To* __restrict__ out, int64_t G, int64_t D, int64_t r, int64_t Rc,
                                     int64_t pos0) {
    const int64_t n = G * D;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t g = i / D, c = i % D;
        const int64_t grp = pos0 / r + g;
        float m = -INFINITY, l = 0.f, acc = 0.f;
        for (int64_t k = 0; k < r; ++k) {
            const int64_t p = grp * r + k;
            float kv_, s_;
            if (p >= pos0) {
                kv_ = kv[(p - pos0) * D + c];
                s_ = sc[(p - pos0) * D + c];
            } else {
                const int64_t slot = p % Rc;
                kv_ = rkv[slot * D + c];
                s_ = rsc[slot * D + c];
            }
            const float mn = fmaxf(m, s_);
            const float a = expf(m - mn), e = expf(s_ - mn);
            l = l * a + e;
            acc = acc * a + kv_ * e;
            m = mn;
        }
        st(out, i, acc / l);
    }
}

}  // namespace

extern "C" int ik_moe_route(const ik_tensor* logits, const ik_tensor* bias, int64_t top_k, float scale, int32_t norm_topk,
                            float temperature, int32_t score_func, ik_tensor* weights, ik_tensor* ids, void* stream) {
    IK_TRY(ik_check(logits, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(bias, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(weights, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 2));
    const int64_t T = logits->shape[0], E = logits->shape[1];
    if (top_k < 1 || top_k > 32 || top_k > E || E > 1024 || bias->shape[0] != E || weights->shape[0] != T ||
        weights->shape[1] != top_k || ids->shape[0] != T || ids->shape[1] != top_k || score_func < 0 || score_func > 2)
        return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    moe_route_kernel<<<(unsigned)T, 32, (size_t)E * 8, IK_STREAM(stream)>>>((const float*)logits->data, (const float*)bias->data, E,
                                                                            (int)top_k, scale, norm_topk, temperature, score_func,
                                                                            (float*)weights->data, (int*)ids->data);
    return ik_launch_status();
}

extern "C" int ik_silu_glu_clamp(const ik_tensor* gate, const ik_tensor* up, float limit, ik_tensor* out, void* stream) {
    IK_TRY(check_float(gate, gate ? gate->ndim : 0));
    IK_TRY(check_float(up, gate->ndim));
    IK_TRY(check_float(out, gate->ndim));
    const int64_t n = ik_numel(gate);
    if (ik_numel(up) != n || ik_numel(out) != n || !(limit >= 0.f)) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    DSV_FLOAT_DISPATCH(gate->dtype, Tg, DSV_FLOAT_DISPATCH(out->dtype, To, if (up->dtype != gate->dtype) return IK_ERR_DTYPE;
        silu_glu_clamp_kernel<Tg, Tg, To><<<grid1d(n, kThreads), kThreads, 0, s>>>((const Tg*)gate->data, (const Tg*)up->data,
                                                                                  (To*)out->data, n, limit)));
    return ik_launch_status();
}

extern "C" int ik_act_qdq(const ik_tensor* x, int32_t fmt, int64_t block, int32_t scale_fmt, ik_tensor* y, void* stream) {
    if (!x || !y || x->ndim < 1) return IK_ERR_INVALID_ARG;
    IK_TRY(check_float(x, x->ndim));
    IK_TRY(ik_check(y, x->dtype, x->ndim));
    const int64_t n = ik_numel(x);
    const int64_t last = x->shape[x->ndim - 1];
    if (ik_numel(y) != n || block < 1 || last % block != 0 || fmt < 0 || fmt > 1 || scale_fmt < 0 || scale_fmt > 2) return IK_ERR_SHAPE;
    const bool small = block <= 32 && (block & (block - 1)) == 0;
    if (!small && block % 32 != 0) return IK_ERR_SHAPE;
    if (n == 0) return IK_OK;
    QdqP p;
    p.fmt = fmt;
    p.sfmt = scale_fmt;
    if (fmt == 0) {
        p.fmax = 448.f;
        p.floor_ = 1e-4f;
    } else {
        p.fmax = 6.f;
        p.floor_ = scale_fmt == 1 ? 6.f * 0.001953125f /* 2^-9 */ : 6.f * 1.1754943508222875e-38f /* 2^-126 */;
    }
    cudaStream_t s = IK_STREAM(stream);
    if (small) {
        const unsigned blocks = (unsigned)cdiv(n, kThreads);
        DSV_FLOAT_DISPATCH(x->dtype, Ti,
            qdq_small<Ti, Ti><<<blocks, kThreads, 0, s>>>((const Ti*)x->data, (Ti*)y->data, n, (int)block, p));
    } else {
        const int64_t nblk = n / block;
        const unsigned blocks = (unsigned)cdiv(nblk * 32, kThreads);
        DSV_FLOAT_DISPATCH(x->dtype, Ti,
            qdq_wide<Ti, Ti><<<blocks, kThreads, 0, s>>>((const Ti*)x->data, (Ti*)y->data, nblk, (int)block, p));
    }
    return ik_launch_status();
}

extern "C" int ik_rope_tail(const ik_tensor* x, const ik_tensor* inv_freq, int64_t pos0, int64_t pos_stride, int32_t inverse,
                            ik_tensor* out, const ik_tensor* pos, void* stream) {
    IK_TRY(check_float(x, 3));
    IK_TRY(ik_check(inv_freq, IK_DTYPE_F32, 1));
    IK_TRY(check_float(out, 3));
    IK_TRY(ik_check_pos(pos));
    const int64_t T = x->shape[0], N = x->shape[1], Dh = x->shape[2], half = inv_freq->shape[0];
    if (half < 1 || 2 * half > Dh || pos_stride < 1 || out->shape[0] != T || out->shape[1] != N || out->shape[2] != Dh) return IK_ERR_SHAPE;
    const int64_t n = T * N * Dh;
    if (n == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    const int* dp = pos ? (const int*)pos->data : nullptr;
    DSV_FLOAT_DISPATCH(x->dtype, Ti, DSV_FLOAT_DISPATCH(out->dtype, To,
        rope_tail_kernel<Ti, To><<<grid1d(n, kThreads), kThreads, 0, s>>>((const Ti*)x->data, (const float*)inv_freq->data,
                                                                        (To*)out->data, T, N, Dh, half, pos0, pos_stride, inverse, dp)));
    return ik_launch_status();
}

extern "C" int ik_window_kv(const ik_tensor* kv, int64_t window, int64_t pos0, ik_tensor* win, ik_tensor* ring, const ik_tensor* pos,
                            void* stream) {
    IK_TRY(check_float(kv, 2));
    IK_TRY(ik_check(win, kv->dtype, 2));
    IK_TRY(ik_check(ring, kv->dtype, 2));
    IK_TRY(ik_check_pos(pos));
    const int64_t T = kv->shape[0], D = kv->shape[1], W = window, Wc = ring->shape[0];
    if (W < 1 || Wc < W || ring->shape[1] != D || win->shape[0] != W + T || win->shape[1] != D) return IK_ERR_SHAPE;
    if (D == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    const int* dp = pos ? (const int*)pos->data : nullptr;
    DSV_FLOAT_DISPATCH(kv->dtype, Tv,
        window_build<Tv><<<grid1d((W + T) * D, kThreads), kThreads, 0, s>>>((const Tv*)kv->data, (const Tv*)ring->data, (Tv*)win->data,
                                                                          T, D, W, Wc, pos0, dp);
        IK_TRY(ik_launch_status());
        if (T > 0) ring_write<Tv><<<grid1d((T < Wc ? T : Wc) * D, kThreads), kThreads, 0, s>>>((const Tv*)kv->data, (Tv*)ring->data,
                                                                                            T, D, Wc, pos0, dp));
    return ik_launch_status();
}

extern "C" int ik_compress_pool(const ik_tensor* kv, const ik_tensor* score, int64_t ratio, int64_t pos0, ik_tensor* out,
                                ik_tensor* kv_ring, ik_tensor* score_ring, void* stream) {
    IK_TRY(ik_check(kv, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(score, IK_DTYPE_F32, 2));
    IK_TRY(check_float(out, 2));
    IK_TRY(ik_check(kv_ring, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(score_ring, IK_DTYPE_F32, 2));
    const int64_t T = kv->shape[0], D = kv->shape[1], r = ratio, Rc = kv_ring->shape[0];
    if (r < 2 || pos0 < 0) return IK_ERR_SHAPE;
    const int64_t G = (pos0 + T) / r - pos0 / r;
    if (score->shape[0] != T || score->shape[1] != D || out->shape[0] != G || out->shape[1] != D || kv_ring->shape[1] != D ||
        score_ring->shape[0] != Rc || score_ring->shape[1] != D || Rc < r || Rc % r != 0)
        return IK_ERR_SHAPE;
    cudaStream_t s = IK_STREAM(stream);
    if (G * D > 0) {
        DSV_FLOAT_DISPATCH(out->dtype, To,
            compress_pool_kernel<To><<<grid1d(G * D, kThreads), kThreads, 0, s>>>((const float*)kv->data, (const float*)score->data,
                                                                                 (const float*)kv_ring->data, (const float*)score_ring->data,
                                                                                 (To*)out->data, G, D, r, Rc, pos0));
        IK_TRY(ik_launch_status());
    }
    if (T > 0 && D > 0) {
        const int64_t rows = T < Rc ? T : Rc;
        ring_write<float><<<grid1d(rows * D, kThreads), kThreads, 0, s>>>((const float*)kv->data, (float*)kv_ring->data, T, D, Rc, pos0, nullptr);
        ring_write<float><<<grid1d(rows * D, kThreads), kThreads, 0, s>>>((const float*)score->data, (float*)score_ring->data, T, D, Rc, pos0,
                                                                         nullptr);
    }
    return ik_launch_status();
}
