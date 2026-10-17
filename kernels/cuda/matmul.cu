// matmul / logits: out[M,N] = x[M,K] @ w[N,K]^T, bf16 inputs, f32 accumulate.
//   *_bf16        cuBLAS GemmEx (tensor cores). Fast: split-K / algorithm choice may vary run to run.
//   *_bf16_gemv   native warp-per-output-row GEMV for M <= 4 (decode). Deterministic: fixed
//                 reduction order, identical results for identical inputs. Memory bound: it streams
//                 each weight row exactly once with 16-byte loads.
// logits is the same math with an f32 output tensor.
#include "ik_launch.h"
#include "ik_internal.h"
#include <cublas_v2.h>
#include <mutex>

namespace {

// ---- cuBLAS state ------------------------------------------------------------------------
constexpr int kMaxDevices = 16;
constexpr size_t kWorkspaceBytes = 32u << 20;

struct DevState {
    cublasHandle_t h = nullptr;
    void* ws = nullptr;
    std::mutex mu;
};
DevState g_dev[kMaxDevices];

// ---- GEMV --------------------------------------------------------------------------------
constexpr int kWarpsPerBlock = 8;

IK_DEVI void store_out(float* p, float v) { *p = v; }
IK_DEVI void store_out(ik_bf16* p, float v) { *p = ik_f2b(v); }

// weights are read once: bypass L1 allocation (ld.global.nc with evict-first behaviour)
IK_DEVI uint4 ldg_stream(const uint4* p) { return __ldcs(p); }

// `blk` is the block's index along the rows of this matrix: blockIdx.x for the plain kernel, the index inside its
// segment for the pair kernel below, which therefore runs this very code. XS: `x` points into shared memory (the
// norm + pair kernel stages the normalised row there); the arithmetic is the same, only the load differs.
template <int M, typename OutT, bool VEC, bool XS = false>
__device__ __forceinline__ void gemv_body(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w,
                                          OutT* __restrict__ out, int64_t N, int64_t K, int64_t blk) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = blk * kWarpsPerBlock + warp;
    if (n >= N) return;  // whole warp exits together; no block-level sync below
    const ik_bf16* wr = w + n * K;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;

    if (VEC) {
        const int64_t nv = K / 8;
        const uint4* w8 = reinterpret_cast<const uint4*>(wr);
        const uint4* x8 = reinterpret_cast<const uint4*>(x);
#pragma unroll 4
        for (int64_t i = lane; i < nv; i += IK_WARP) {
            float wf[8];
            ik_unpack8(ldg_stream(w8 + i), wf);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[8];
                ik_unpack8(XS ? x8[m * nv + i] : __ldg(x8 + m * nv + i), xf);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[m] += wf[e] * xf[e];
            }
        }
    } else {
        for (int64_t k = lane; k < K; k += IK_WARP) {
            const float wf = ik_b2f(wr[k]);
#pragma unroll
            for (int m = 0; m < M; ++m) acc[m] += wf * ik_b2f(x[m * K + k]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) store_out(out + m * N + n, s);
    }
}

template <int M, typename OutT, bool VEC>
__global__ void __launch_bounds__(kWarpsPerBlock * IK_WARP)
gemv_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w, OutT* __restrict__ out, int64_t N,
            int64_t K) {
    gemv_body<M, OutT, VEC>(x, w, out, N, K, (int64_t)blockIdx.x);
}

// Two projections of one activation row in one launch (the GDN a / b gates, the MoE router and shared-expert gate):
// blocks [0, nb0) run the first matrix, the rest the second, each through the same body as `gemv_kernel<1, bf16, true>`.
__global__ void __launch_bounds__(kWarpsPerBlock * IK_WARP)
gemv_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ w0, ik_bf16* __restrict__ o0, int64_t N0,
                 const ik_bf16* __restrict__ w1, ik_bf16* __restrict__ o1, int64_t N1, int64_t K) {
    const int64_t nb0 = (N0 + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if ((int64_t)blockIdx.x < nb0) gemv_body<1, ik_bf16, true>(x, w0, o0, N0, K, (int64_t)blockIdx.x);
    else gemv_body<1, ik_bf16, true>(x, w1, o1, N1, K, (int64_t)blockIdx.x - nb0);
}

// ---- RMSNorm (+ residual add) folded into the pair GEMV ---------------------------------------------------------
// rmsnorm_add_matmul_pair: the decode step's `rmsnorm_add(x, res, w) -> (normed, sum)` followed by the pair GEMV on `normed`
// (the MoE router + shared-expert gate, the GDN b / a gates) as one launch. Every block recomputes the row's norm in its own
// prologue and keeps the normalised row in shared memory, where the GEMV reads it instead of global memory; block 0 alone
// stores `sum` (the residual stream the next layer carries) and `normed` (the other readers of the row). Those are different
// buffers from the `x` / `res` every block reads (the launcher refuses an overlap), so no block sees a partly written input.
//
// Bitwise the two launches it replaces: the prologue is rmsnorm_kernel<true, true> (norm.cu) statement for statement (the same
// 8-element chunks per thread, the bf16 rounding of the sum before it is squared and normalised, ik_block_sum over the same 256
// threads: rmsnorm's launch_rows picks 256 for any K >= 2048, and for a narrower row its smaller block holds the same chunks in
// the same threads while the extra warps here add exact zeros), and the GEMV is gemv_body<1, bf16, true> with x read from smem.
constexpr int kNormMaxK = 8192;  // the smem row is K * 2 bytes: 16 KB at most

// rmsnorm_kernel<true, true> for one row; `xs` ends up holding the normalised row (bf16 chunks). Every thread of the block
// must call it (block barriers inside).
__device__ __forceinline__ void norm_add_stage(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res,
                                               const ik_bf16* __restrict__ nw, ik_bf16* __restrict__ normed,
                                               ik_bf16* __restrict__ sum, uint4* xs, float* red, int64_t K, float eps,
                                               float woff, bool writer) {
    const int64_t nv = K / 8;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
        float f[8], g[8];
        ik_unpack8(__ldg(reinterpret_cast<const uint4*>(x) + i), f);
        ik_unpack8(__ldg(reinterpret_cast<const uint4*>(res) + i), g);
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] += g[k];
        const uint4 p = ik_pack8(f);
        if (writer) reinterpret_cast<uint4*>(sum)[i] = p;
        xs[i] = p;         // the rounded sum, until the second loop replaces it with the normalised row
        ik_unpack8(p, f);  // continue from the bf16-rounded sum
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
    }
    ss = ik_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)K + eps);
    for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
        float f[8], g[8];
        ik_unpack8(xs[i], f);
        ik_unpack8(__ldg(reinterpret_cast<const uint4*>(nw) + i), g);
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);
        const uint4 p = ik_pack8(f);
        xs[i] = p;
        if (writer) reinterpret_cast<uint4*>(normed)[i] = p;
    }
    __syncthreads();  // the whole normalised row is in smem before any warp reads it
}

__global__ void __launch_bounds__(kWarpsPerBlock * IK_WARP)
rmsnorm_add_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res, const ik_bf16* __restrict__ nw,
                        ik_bf16* __restrict__ normed, ik_bf16* __restrict__ sum, float eps, float woff,
                        const ik_bf16* __restrict__ w0, ik_bf16* __restrict__ o0, int64_t N0,
                        const ik_bf16* __restrict__ w1, ik_bf16* __restrict__ o1, int64_t N1, int64_t K) {
    __shared__ float red[32];
    __shared__ __align__(16) uint4 xs[kNormMaxK / 8];
    norm_add_stage(x, res, nw, normed, sum, xs, red, K, eps, woff, blockIdx.x == 0);
    const ik_bf16* xrow = reinterpret_cast<const ik_bf16*>(xs);
    const int64_t nb0 = (N0 + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if ((int64_t)blockIdx.x < nb0) gemv_body<1, ik_bf16, true, true>(xrow, w0, o0, N0, K, (int64_t)blockIdx.x);
    else gemv_body<1, ik_bf16, true, true>(xrow, w1, o1, N1, K, (int64_t)blockIdx.x - nb0);
}

template <int M, typename OutT>
void gemv_dispatch(const ik_bf16* x, const ik_bf16* w, OutT* out, int64_t N, int64_t K, bool vec, cudaStream_t s) {
    const unsigned grid = (unsigned)((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const unsigned block = kWarpsPerBlock * IK_WARP;
    if (vec) gemv_kernel<M, OutT, true><<<grid, block, 0, s>>>(x, w, out, N, K);
    else gemv_kernel<M, OutT, false><<<grid, block, 0, s>>>(x, w, out, N, K);
}

template <typename OutT>
int gemv_launch(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M < 1 || M > 4) return IK_ERR_SHAPE;
    const bool vec = K % 8 == 0 && ik_aligned16(x->data) && ik_aligned16(w->data);
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const ik_bf16* wp = (const ik_bf16*)w->data;
    OutT* op = (OutT*)out->data;
    switch (M) {
        case 1: gemv_dispatch<1, OutT>(xp, wp, op, N, K, vec, s); break;
        case 2: gemv_dispatch<2, OutT>(xp, wp, op, N, K, vec, s); break;
        case 3: gemv_dispatch<3, OutT>(xp, wp, op, N, K, vec, s); break;
        default: gemv_dispatch<4, OutT>(xp, wp, op, N, K, vec, s); break;
    }
    return ik_launch_status();
}

// ---- shared validation -------------------------------------------------------------------
int check_mm(const ik_tensor* x, const ik_tensor* w, const ik_tensor* out, int out_dtype) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, out_dtype, 2));
    if (x->shape[1] != w->shape[1] || out->shape[0] != x->shape[0] || out->shape[1] != w->shape[0]) return IK_ERR_SHAPE;
    return IK_OK;
}

int cublas_launch(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream, bool f32_out) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M > 0x7fffffffLL || K > 0x7fffffffLL || N > 0x7fffffffLL) return IK_ERR_SHAPE;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) return IK_ERR_LAUNCH;
    DevState& d = g_dev[dev];
    std::lock_guard<std::mutex> lock(d.mu);
    if (!d.h) return IK_ERR_UNSUPPORTED;  // ik_rt_init_device() was not called for this device
    if (cublasSetStream(d.h, IK_STREAM(stream)) != CUBLAS_STATUS_SUCCESS) return IK_ERR_LAUNCH;
    // cublasSetStream resets the workspace to the library default pool; re-attach ours so the call
    // never allocates.
    if (cublasSetWorkspace(d.h, d.ws, kWorkspaceBytes) != CUBLAS_STATUS_SUCCESS) return IK_ERR_LAUNCH;
    const float alpha = 1.f, beta = 0.f;
    // row-major out[M,N] = x[M,K] w[N,K]^T  ==  column-major out^T[N,M] = w^T-view[N,K] * x^T-view[K,M]
    const cublasStatus_t st = cublasGemmEx(
        d.h, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, w->data, CUDA_R_16BF, (int)K, x->data,
        CUDA_R_16BF, (int)K, &beta, out->data, f32_out ? CUDA_R_32F : CUDA_R_16BF, (int)N, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return st == CUBLAS_STATUS_SUCCESS ? IK_OK : IK_ERR_LAUNCH;
}

// ---- pair launcher (M == 1, vector path only) -------------------------------------------------
int pair(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* o0, ik_tensor* o1, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w0, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w1, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(o0, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(o1, IK_DTYPE_BF16, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (w0->shape[1] != K || w1->shape[1] != K || o0->shape[0] != M || o1->shape[0] != M || o0->shape[1] != N0 ||
        o1->shape[1] != N1)
        return IK_ERR_SHAPE;
    // decode only (one row) and the vector path; the registry predicate says the same, this keeps a direct call honest
    if (M != 1 || K % 8 != 0 || !ik_aligned16(x->data) || !ik_aligned16(w0->data) || !ik_aligned16(w1->data))
        return IK_ERR_SHAPE;
    if (N0 == 0 && N1 == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    if (K == 0) {
        if (cudaMemsetAsync(o0->data, 0, ik_numel(o0) * 2, s) != cudaSuccess) return IK_ERR_LAUNCH;
        return cudaMemsetAsync(o1->data, 0, ik_numel(o1) * 2, s) == cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    }
    const int64_t blocks = (N0 + kWarpsPerBlock - 1) / kWarpsPerBlock + (N1 + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks > 0x7fffffffLL) return IK_ERR_SHAPE;
    gemv_pair_kernel<<<(unsigned)blocks, kWarpsPerBlock * IK_WARP, 0, s>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)w0->data, (ik_bf16*)o0->data, N0, (const ik_bf16*)w1->data,
        (ik_bf16*)o1->data, N1, K);
    return ik_launch_status();
}

// ---- norm + pair launcher (M == 1, vector path only) ----------------------------------------------------------
bool overlaps(const void* a, const void* b, size_t bytes) {
    const char *p = (const char*)a, *q = (const char*)b;
    return p < q + bytes && q < p + bytes;
}

int norm_pair(const ik_tensor* x, const ik_tensor* res, const ik_tensor* nw, float eps, float woff, const ik_tensor* w0,
              const ik_tensor* w1, ik_tensor* normed, ik_tensor* sum, ik_tensor* o0, ik_tensor* o1, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(res, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(nw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(normed, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(sum, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w0, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w1, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(o0, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(o1, IK_DTYPE_BF16, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (res->shape[0] != M || res->shape[1] != K || nw->shape[0] != K || normed->shape[0] != M || normed->shape[1] != K ||
        sum->shape[0] != M || sum->shape[1] != K || w0->shape[1] != K || w1->shape[1] != K || o0->shape[0] != M ||
        o1->shape[0] != M || o0->shape[1] != N0 || o1->shape[1] != N1)
        return IK_ERR_SHAPE;
    // decode only (one row), the vector path, and the row must fit the smem staging; the registry predicate says the same
    if (M != 1 || K < 8 || K % 8 != 0 || K > kNormMaxK) return IK_ERR_SHAPE;
    if (!ik_aligned16(x->data) || !ik_aligned16(res->data) || !ik_aligned16(nw->data) || !ik_aligned16(normed->data) ||
        !ik_aligned16(sum->data) || !ik_aligned16(w0->data) || !ik_aligned16(w1->data))
        return IK_ERR_SHAPE;
    // block 0 stores `normed` and `sum` while the other blocks still read `x` and `res`
    const size_t rb = (size_t)K * 2;
    if (overlaps(normed->data, x->data, rb) || overlaps(normed->data, res->data, rb) || overlaps(sum->data, x->data, rb) ||
        overlaps(sum->data, res->data, rb) || overlaps(normed->data, sum->data, rb))
        return IK_ERR_INVALID_ARG;
    if (N0 == 0 && N1 == 0) return IK_ERR_SHAPE;  // (the norm would still have to be stored: not a case the model produces)
    const int64_t blocks = (N0 + kWarpsPerBlock - 1) / kWarpsPerBlock + (N1 + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks > 0x7fffffffLL) return IK_ERR_SHAPE;
    rmsnorm_add_pair_kernel<<<(unsigned)blocks, kWarpsPerBlock * IK_WARP, 0, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)nw->data, (ik_bf16*)normed->data,
        (ik_bf16*)sum->data, eps, woff, (const ik_bf16*)w0->data, (ik_bf16*)o0->data, N0, (const ik_bf16*)w1->data,
        (ik_bf16*)o1->data, N1, K);
    return ik_launch_status();
}

int mm(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream, bool f32_out, bool gemv) {
    IK_TRY(check_mm(x, w, out, f32_out ? IK_DTYPE_F32 : IK_DTYPE_BF16));
    if (ik_numel(out) == 0) return IK_OK;
    if (x->shape[1] == 0) return cudaMemsetAsync(out->data, 0, ik_numel(out) * (f32_out ? 4 : 2), IK_STREAM(stream)) ==
                                         cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (gemv) return f32_out ? gemv_launch<float>(x, w, out, stream) : gemv_launch<ik_bf16>(x, w, out, stream);
    return cublas_launch(x, w, out, stream, f32_out);
}

}  // namespace

extern "C" {

int ik_matmul_pair_bf16_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* out0,
                             ik_tensor* out1, void* stream) {
    return pair(x, w0, w1, out0, out1, stream);
}

int ik_rmsnorm_add_matmul_pair_bf16_gemv(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps,
                                         float w_offset, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* normed,
                                         ik_tensor* sum, ik_tensor* out0, ik_tensor* out1, void* stream) {
    return norm_pair(x, res, w, eps, w_offset, w0, w1, normed, sum, out0, out1, stream);
}

int ik_cublas_init(int device) {
    if (device < 0 || device >= kMaxDevices) return IK_ERR_INVALID_ARG;
    DevState& d = g_dev[device];
    std::lock_guard<std::mutex> lock(d.mu);
    if (d.h) return IK_OK;
    if (cublasCreate(&d.h) != CUBLAS_STATUS_SUCCESS) {
        d.h = nullptr;
        return IK_ERR_LAUNCH;
    }
    if (cudaMalloc(&d.ws, kWorkspaceBytes) != cudaSuccess) {
        cublasDestroy(d.h);
        d.h = nullptr;
        return IK_ERR_LAUNCH;
    }
    // keep reductions in f32 (no reduced-precision split-K accumulation) for parity with the oracle
    cublasSetMathMode(d.h, (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    return IK_OK;
}

void ik_cublas_shutdown(int device) {
    if (device < 0 || device >= kMaxDevices) return;
    DevState& d = g_dev[device];
    std::lock_guard<std::mutex> lock(d.mu);
    if (d.h) cublasDestroy(d.h);
    if (d.ws) cudaFree(d.ws);
    d.h = nullptr;
    d.ws = nullptr;
}

int ik_matmul_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, false, false);
}
int ik_matmul_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, false, true);
}
int ik_logits_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, true, false);
}
int ik_logits_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream) {
    return mm(x, w, out, stream, true, true);
}

}  // extern "C"
