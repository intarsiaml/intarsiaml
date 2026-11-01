// DeepSeek-V4.1 layout and Engram ops: transpose01 (swap of the two leading axes, any element type), engram_gather
// (ParallelEngramEmbedding.forward: e4m3 rows times their E8M0 block scales), engram_merge (Engram.forward: per stream a
// normalised dot of the stream with its key gates the shared value). Contract: docs/design/dsv41-ops.md; the cpu reference is
// intarsia-backend-cpu's dsv41_ref. Written from the op contract and DeepSeek's official inference code (MIT).
#include "dsv41_common.cuh"

#include <cuda_fp8.h>

using namespace dsv41;

namespace {

constexpr int kThreads = 256;

// ------------------------------------------------------------------------------------------------------------- transpose01

// y[b, a, :] = x[a, b, :] as elements V of 1 to 16 bytes (the widest that divides a row of C elements and the pointers)
template <typename V>
__global__ void transpose01_kernel(const V* __restrict__ x, V* __restrict__ y, int64_t A, int64_t B, int64_t C, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t c = i % C;
        const int64_t r = i / C;  // output row b * A + a
        const int64_t b = r / A, a = r - b * A;
        y[i] = x[(a * B + b) * C + c];
    }
}

template <typename V>
int transpose01_run(const ik_tensor* x, ik_tensor* y, int64_t A, int64_t B, int64_t row_elems, cudaStream_t s) {
    const int64_t n = A * B * row_elems;
    transpose01_kernel<V><<<grid1d(n, kThreads), kThreads, 0, s>>>((const V*)x->data, (V*)y->data, A, B, row_elems, n);
    return ik_launch_status();
}

int elem_bytes(int dt) {
    switch (dt) {
        case IK_DTYPE_F32:
        case IK_DTYPE_I32:
            return 4;
        case IK_DTYPE_F16:
        case IK_DTYPE_BF16:
        case IK_DTYPE_I16:
            return 2;
        case IK_DTYPE_FP8_E4M3:
        case IK_DTYPE_FP8_E5M2:
        case IK_DTYPE_I8:
        case IK_DTYPE_U8:
            return 1;
        default:
            return 0;
    }
}

// ------------------------------------------------------------------------------------------------------------ engram_gather

// payload e4m3 * 2^(sc - 127) in f32, the reference's `fp8_e4m3_to_f32(b) * 2f32.powi(sc - 127)`: the power of two is exact
// (sc = 0 is the subnormal 2^-127, sc = 255 overflows to inf)
IK_DEVI float engram_scale(unsigned sc) {
    if (sc == 0) return __uint_as_float(0x00400000u);
    if (sc == 255) return __uint_as_float(0x7F800000u);
    return __uint_as_float(sc << 23);
}

template <typename To>
__global__ void engram_gather_kernel(const unsigned char* __restrict__ rec, To* __restrict__ y, int64_t n, int64_t R, int64_t Dh) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = i / Dh, j = i - row * Dh;
        const unsigned char* r = rec + row * R;
        const __nv_fp8_storage_t q = (__nv_fp8_storage_t)r[j];
        const float v = __half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)));
        st(y, i, v * engram_scale(r[Dh + (j >> 5)]));
    }
}

// ------------------------------------------------------------------------------------------------------------- engram_merge

constexpr int kMergeThreads = 512;

// float load / store of a tensor whose dtype is known at compile time (DT >= 0) or only at run time (DT < 0)
template <int DT>
IK_DEVI float ldx(const void* p, int dt, int64_t i) {
    const int d = DT >= 0 ? DT : dt;
    if (d == IK_DTYPE_F32) return ((const float*)p)[i];
    if (d == IK_DTYPE_BF16) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    return __half2float(((const __half*)p)[i]);
}

template <int DT>
IK_DEVI void stx(void* p, int dt, int64_t i, float v) {
    const int d = DT >= 0 ? DT : dt;
    if (d == IK_DTYPE_F32) ((float*)p)[i] = v;
    else if (d == IK_DTYPE_BF16) ((__nv_bfloat16*)p)[i] = __float2bfloat16_rn(v);
    else ((__half*)p)[i] = __float2half_rn(v);
}

// one block per (token, stream). dot = sum_d h q_w k_w key * rstd(h) * rstd(key) / sqrt(D); out = h + sigmoid(signed sqrt) * value
template <int DH, int DKV, int DW, int DO>
__global__ void engram_merge_kernel(const void* h, const void* kv, const void* qw, const void* kw, void* out, int64_t S, int64_t D, float eps,
                                    int dh, int dkv, int dw, int dout) {
    __shared__ float red[3][kMergeThreads / 32];
    __shared__ float gate_s;
    const int64_t row = blockIdx.x;
    const int64_t t = row / S, s = row - t * S;
    const int64_t hb = row * D, kb = t * (S + 1) * D + s * D, vb = t * (S + 1) * D + S * D, wb = s * D;
    float hh = 0.f, kk = 0.f, dd = 0.f;
    for (int64_t d = threadIdx.x; d < D; d += kMergeThreads) {
        const float hv = ldx<DH>(h, dh, hb + d), key = ldx<DKV>(kv, dkv, kb + d);
        hh += hv * hv;
        kk += key * key;
        dd += hv * ldx<DW>(qw, dw, wb + d) * ldx<DW>(kw, dw, wb + d) * key;
    }
    hh = ik_warp_sum(hh);
    kk = ik_warp_sum(kk);
    dd = ik_warp_sum(dd);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) {
        red[0][w] = hh;
        red[1][w] = kk;
        red[2][w] = dd;
    }
    __syncthreads();
    if (w == 0) {
        constexpr int NW = kMergeThreads / 32;
        hh = lane < NW ? red[0][lane] : 0.f;
        kk = lane < NW ? red[1][lane] : 0.f;
        dd = lane < NW ? red[2][lane] : 0.f;
        hh = ik_warp_sum(hh);
        kk = ik_warp_sum(kk);
        dd = ik_warp_sum(dd);
        if (lane == 0) {
            const float fd = (float)D;
            const float rstd = 1.f / sqrtf(hh / fd + eps) / sqrtf(kk / fd + eps);
            const float dot = dd * rstd / sqrtf(fd);
            const float g = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
            gate_s = 1.f / (1.f + expf(-g));
        }
    }
    __syncthreads();
    const float gate = gate_s;
    for (int64_t d = threadIdx.x; d < D; d += kMergeThreads)
        stx<DO>(out, dout, hb + d, ldx<DH>(h, dh, hb + d) + gate * ldx<DKV>(kv, dkv, vb + d));
}

}  // namespace

extern "C" int ik_transpose01(const ik_tensor* x, ik_tensor* y, void* stream) {
    if (!x || !y) return IK_ERR_INVALID_ARG;
    const int es = elem_bytes(x->dtype);
    if (es == 0) return IK_ERR_DTYPE;
    IK_TRY(ik_check(x, x->dtype, 3));
    IK_TRY(ik_check(y, x->dtype, 3));
    const int64_t A = x->shape[0], B = x->shape[1], C = x->shape[2];
    if (y->shape[0] != B || y->shape[1] != A || y->shape[2] != C) return IK_ERR_SHAPE;
    if (A * B * C == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    // widest vector (bytes) that divides a row of C elements and both base addresses
    const int64_t row_bytes = C * es;
    const uintptr_t al = (uintptr_t)x->data | (uintptr_t)y->data;
    int v = 16;
    while (v > 1 && (row_bytes % v != 0 || (al & (uintptr_t)(v - 1)) != 0)) v >>= 1;
    const int64_t re = row_bytes / v;
    switch (v) {
        case 16: return transpose01_run<uint4>(x, y, A, B, re, s);
        case 8: return transpose01_run<uint2>(x, y, A, B, re, s);
        case 4: return transpose01_run<unsigned>(x, y, A, B, re, s);
        case 2: return transpose01_run<unsigned short>(x, y, A, B, re, s);
        default: return transpose01_run<unsigned char>(x, y, A, B, re, s);
    }
}

extern "C" int ik_engram_gather(const ik_tensor* records, ik_tensor* y, void* stream) {
    IK_TRY(ik_check(records, IK_DTYPE_U8, 3));
    IK_TRY(check_float(y, 2));
    const int64_t T = records->shape[0], C = records->shape[1], R = records->shape[2];
    if (R <= 0 || R % 33 != 0) return IK_ERR_SHAPE;
    const int64_t Dh = R / 33 * 32;
    if (y->shape[0] != T || y->shape[1] != C * Dh) return IK_ERR_SHAPE;
    const int64_t n = T * C * Dh;
    if (n == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    DSV_FLOAT_DISPATCH(y->dtype, To,
        engram_gather_kernel<To><<<grid1d(n, kThreads), kThreads, 0, s>>>((const unsigned char*)records->data, (To*)y->data, n, R, Dh));
    return ik_launch_status();
}

extern "C" int ik_engram_merge(const ik_tensor* h, const ik_tensor* kv, const ik_tensor* q_w, const ik_tensor* k_w, float eps, ik_tensor* out,
                               void* stream) {
    IK_TRY(check_float(h, 3));
    IK_TRY(check_float(kv, 2));
    IK_TRY(check_float(q_w, 2));
    IK_TRY(check_float(k_w, 2));
    IK_TRY(check_float(out, 3));
    const int64_t T = h->shape[0], S = h->shape[1], D = h->shape[2];
    if (kv->shape[0] != T || kv->shape[1] != (S + 1) * D || q_w->shape[0] != S || q_w->shape[1] != D || k_w->shape[0] != S ||
        k_w->shape[1] != D || out->shape[0] != T || out->shape[1] != S || out->shape[2] != D || !(eps >= 0.f))
        return IK_ERR_SHAPE;
    if (q_w->dtype != k_w->dtype) return IK_ERR_DTYPE;
    if (T * S * D == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    const int dh = h->dtype, dkv = kv->dtype, dw = q_w->dtype, dout = out->dtype;
    const unsigned blocks = (unsigned)(T * S);
    if (dh == dkv && dh == dw && dh == dout) {
        switch (dh) {
            case IK_DTYPE_F32:
                engram_merge_kernel<IK_DTYPE_F32, IK_DTYPE_F32, IK_DTYPE_F32, IK_DTYPE_F32><<<blocks, kMergeThreads, 0, s>>>(
                    h->data, kv->data, q_w->data, k_w->data, out->data, S, D, eps, dh, dkv, dw, dout);
                break;
            case IK_DTYPE_BF16:
                engram_merge_kernel<IK_DTYPE_BF16, IK_DTYPE_BF16, IK_DTYPE_BF16, IK_DTYPE_BF16><<<blocks, kMergeThreads, 0, s>>>(
                    h->data, kv->data, q_w->data, k_w->data, out->data, S, D, eps, dh, dkv, dw, dout);
                break;
            default:
                engram_merge_kernel<IK_DTYPE_F16, IK_DTYPE_F16, IK_DTYPE_F16, IK_DTYPE_F16><<<blocks, kMergeThreads, 0, s>>>(
                    h->data, kv->data, q_w->data, k_w->data, out->data, S, D, eps, dh, dkv, dw, dout);
                break;
        }
    } else {
        engram_merge_kernel<-1, -1, -1, -1><<<blocks, kMergeThreads, 0, s>>>(h->data, kv->data, q_w->data, k_w->data, out->data, S, D, eps,
                                                                           dh, dkv, dw, dout);
    }
    return ik_launch_status();
}
