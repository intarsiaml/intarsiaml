// matmul_fp8: out[M,N] = x[M,K] @ W^T, W = float(w_e4m3[N,K]) * weight_scale (per tensor).
//   scales = f32[2] = (weight_scale, input_scale); input_scale > 0 selects W8A8 (static activation quantisation:
//   xq = e4m3(x / input_scale), RNE, saturating; y = weight_scale * input_scale * sum xq*wq), otherwise W8A16.
//   The mode is read from the device tensor inside the kernels (a uniform branch), so launching never needs a
//   host read and the same kernels serve both recipes.
//   *_gemv  decode: M <= 4, warp-per-output-row, x staged in shared memory (quantised once per block).
//   *_mma   any M: 64-column tiles, weights dequantised to bf16 in shared memory (exact), bf16 tensor-core MMA,
//           f32 accumulate. Used for prefill and as the correct path for every shape.
// Both are deterministic (fixed reduction order). Activations bf16; output bf16 or f32.
#include "ik_launch.h"
#include "ik_quant.cuh"

namespace {

constexpr int kWarps = 8;

// ---- GEMV --------------------------------------------------------------------------------------------------------

// The GDN output norm folded into the staging (PRO): the value gated_rmsnorm_kernel<true> (norm.cu) stores for the 8-element
// chunk `i8` of the rows `x` / `z` of 128 values (sixteen chunks per row), as bf16 chunks. A row's sum of squares is that
// kernel's block reduction over its 32 threads, of which only the first sixteen hold a chunk: the xor butterfly of ik_warp_sum
// starting at offset 8 inside a 16-lane group sums the same values in the same order (its first step, offset 16, adds an exact
// zero, and a + b == b + a bitwise), so the sixteen lanes of a row may share a warp with the next row's. All 32 lanes of the
// warp must call it (full-mask shuffles); a lane past the last chunk passes a valid chunk index and drops the result.
IK_DEVI uint4 gated_norm_chunk(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z, const ik_bf16* __restrict__ nw,
                               int i8, float eps, float woff) {
    float f[8], g[8], zf[8];
    ik_unpack8(__ldg(reinterpret_cast<const uint4*>(x) + i8), f);
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = rsqrtf(ss / 128.f + eps);
    ik_unpack8(__ldg(reinterpret_cast<const uint4*>(nw) + (i8 & 15)), g);
    ik_unpack8(__ldg(reinterpret_cast<const uint4*>(z) + i8), zf);
#pragma unroll
    for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);
    float nrm[8];
    ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)
#pragma unroll
    for (int k = 0; k < 8; ++k) f[k] = ik_silu(zf[k]) * nrm[k];
    return ik_pack8(f);
}

// `blk` is the block's index along the rows of this matrix: blockIdx.x for fp8_gemv_kernel, the index inside its segment
// for the pair kernel below, which therefore runs this very code (its staging uses the segment's own scales).
// PRO stages the output of the GDN gated rmsnorm of `x` (the core, rows of 128) and `pz` (the gate) with weight `pnw` instead
// of reading `x` as it is (M == 1); the rest of the body is shared.
template <int M, typename OutT, bool PRO = false>
__device__ __forceinline__ void fp8_gemv_body(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w,
                                              const float* __restrict__ sc, OutT* __restrict__ out, int N, int K,
                                              int64_t blk, const ik_bf16* __restrict__ pz = nullptr,
                                              const ik_bf16* __restrict__ pnw = nullptr, float peps = 0.f,
                                              float pwoff = 0.f) {
    static_assert(!PRO || M == 1, "the folded norm is the single-row staging");
    extern __shared__ __align__(16) unsigned char smem[];
    ik_bf16* xs = reinterpret_cast<ik_bf16*>(smem);  // [M][K] bf16: x, or xq (fp8 values are exact in bf16)
    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    if constexpr (PRO) {
        const int nch = K / 8;  // chunks of 8; a multiple of 16 (K = rows * 128), so a 16-lane group is all in or all out
        for (int base = 0; base < nch; base += blockDim.x) {
            const int i8 = base + threadIdx.x;
            uint4 v = gated_norm_chunk(x, pz, pnw, i8 < nch ? i8 : 0, peps, pwoff);
            if (actq) {
                float f[8];
                ik_unpack8(v, f);
#pragma unroll
                for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
                v = ik_pack8(f);  // fp8 values are exact in bf16
            }
            if (i8 < nch) reinterpret_cast<uint4*>(xs)[i8] = v;
        }
    } else {
        for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
            const ik_bf16 v = x[i];
            xs[i] = actq ? ik_f2b(ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(ik_b2f(v), s_in)))) : v;
        }
    }
    __syncthreads();
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = blk * kWarps + warp;
    if (n >= N) return;
    const uint4* wr = reinterpret_cast<const uint4*>(w + n * K);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int u = lane; u < K / 16; u += IK_WARP) {
        float wf[16];
        ik_fp8x16_to_f32(__ldcs(wr + u), wf);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(xs + (int64_t)m * K + u * 16);
            float xf[16];
            ik_unpack8(xp[0], xf);
            ik_unpack8(xp[1], xf + 8);
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) s = fmaf(wf[e], xf[e], s);
            acc[m] += s;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) ik_store(out + (int64_t)m * N + n, s * s_out);
    }
}

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                OutT* __restrict__ out, int N, int K) {
    fp8_gemv_body<M, OutT>(x, w, sc, out, N, K, (int64_t)blockIdx.x);
}

// Two projections of one activation row in one launch (GDN qkv + z, attention k + v): blocks [0, nb0) run the first
// matrix with its own scales, the rest the second. Each block stages the activations itself, so the two segments need
// not share an input scale (the checkpoint's do).
template <typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_pair_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, const float* __restrict__ sc0,
                     OutT* __restrict__ o0, int N0, const uint8_t* __restrict__ w1, const float* __restrict__ sc1,
                     OutT* __restrict__ o1, int N1, int K) {
    const int nb0 = (N0 + kWarps - 1) / kWarps;
    if ((int)blockIdx.x < nb0) fp8_gemv_body<1, OutT>(x, w0, sc0, o0, N0, K, (int64_t)blockIdx.x);
    else fp8_gemv_body<1, OutT>(x, w1, sc1, o1, N1, K, (int64_t)blockIdx.x - nb0);
}

// The GDN output norm folded into the fp8 out_proj: y = fp8_gemv(gated_rmsnorm(core, z, w), W). Every block normalises the
// rows itself while it stages the activations, so the norm launch and its output round trip disappear. The staged row, and
// therefore the result, is bitwise the two launches it replaces (the body is the one fp8_gemv_kernel<1, bf16> runs).
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_gated_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z, const ik_bf16* __restrict__ nw, float eps,
                      float woff, const uint8_t* __restrict__ w, const float* __restrict__ sc, ik_bf16* __restrict__ out,
                      int N, int K) {
    fp8_gemv_body<1, ik_bf16, true>(x, w, sc, out, N, K, (int64_t)blockIdx.x, z, nw, eps, woff);
}

template <int M, typename OutT>
void gemv_run(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    const size_t smem = (size_t)M * K * 2;
    fp8_gemv_kernel<M, OutT><<<grid, kWarps * IK_WARP, smem, s>>>(x, w, sc, out, (int)N, (int)K);
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    switch (M) {
        case 1: gemv_run<1>(x, w, sc, out, N, K, s); break;
        case 2: gemv_run<2>(x, w, sc, out, N, K, s); break;
        case 3: gemv_run<3>(x, w, sc, out, N, K, s); break;
        case 4: gemv_run<4>(x, w, sc, out, N, K, s); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

// ---- MMA (dequantise-to-bf16 tiles) -----------------------------------------------------------------------------

constexpr int BN = 64, BK = 64, LDS = BK + 8;  // LDS: padded row stride in bf16 elements (conflict-free fragment loads)

template <int BM, typename OutT>
__global__ void __launch_bounds__(128)
fp8_mma_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
               OutT* __restrict__ out, int M, int N, int K) {
    __shared__ __align__(16) ik_bf16 xs[BM][LDS];
    __shared__ __align__(16) ik_bf16 ws[BN][LDS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    float acc[BM / 16][2][4];
#pragma unroll
    for (int i = 0; i < BM / 16; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // activations: BM x 64, 8 elements per thread-iteration
        for (int idx = tid; idx < BM * 8; idx += 128) {
            const int r = idx >> 3, c = (idx & 7) * 8;
            const int gm = m0 + r, gk = k0 + c;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gm < M && gk < K) {
                v = *reinterpret_cast<const uint4*>(x + (int64_t)gm * K + gk);
                if (actq) {
                    float f[8];
                    ik_unpack8(v, f);
#pragma unroll
                    for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
                    v = ik_pack8(f);  // fp8 values are exact in bf16
                }
            }
            *reinterpret_cast<uint4*>(&xs[r][c]) = v;
        }
        // weights: 64 rows x 64 bytes -> bf16 (exact), 16 bytes per thread-iteration
        for (int idx = tid; idx < BN * 4; idx += 128) {
            const int r = idx >> 2, c = (idx & 3) * 16;
            const int gn = n0 + r, gk = k0 + c;
            uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
            if (gn < N && gk < K) {
                float f[16];
                ik_fp8x16_to_f32(__ldcs(reinterpret_cast<const uint4*>(w + (int64_t)gn * K + gk)), f);
                lo = ik_pack8(f);
                hi = ik_pack8(f + 8);
            }
            *reinterpret_cast<uint4*>(&ws[r][c]) = lo;
            *reinterpret_cast<uint4*>(&ws[r][c + 8]) = hi;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[BM / 16][4];
#pragma unroll
            for (int i = 0; i < BM / 16; ++i) {
                const int r = i * 16 + (lane >> 2), c = kk + (lane & 3) * 2;
                a[i][0] = *reinterpret_cast<const uint32_t*>(&xs[r][c]);
                a[i][1] = *reinterpret_cast<const uint32_t*>(&xs[r + 8][c]);
                a[i][2] = *reinterpret_cast<const uint32_t*>(&xs[r][c + 8]);
                a[i][3] = *reinterpret_cast<const uint32_t*>(&xs[r + 8][c + 8]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int nr = warp * 16 + j * 8 + (lane >> 2), c = kk + (lane & 3) * 2;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&ws[nr][c]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&ws[nr][c + 8]);
#pragma unroll
                for (int i = 0; i < BM / 16; ++i) ik_mma_bf16_16816(acc[i][j], a[i][0], a[i][1], a[i][2], a[i][3], b0, b1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < BM / 16; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + (lane >> 2) + h * 8;
                const int col = n0 + warp * 16 + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                if (col < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * h] * s_out);
                if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * h + 1] * s_out);
            }
}

template <int BM, typename OutT>
void mma_run(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t M, int64_t N, int64_t K, cudaStream_t s) {
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    fp8_mma_kernel<BM, OutT><<<grid, 128, 0, s>>>(x, w, sc, out, (int)M, (int)N, (int)K);
}

template <typename OutT>
int mma_by_m(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t M, int64_t N, int64_t K, cudaStream_t s) {
    if (M <= 16) mma_run<16>(x, w, sc, out, M, N, K, s);
    else if (M <= 32) mma_run<32>(x, w, sc, out, M, N, K, s);
    else mma_run<64>(x, w, sc, out, M, N, K, s);
    return ik_launch_status();
}

// ---- launchers ---------------------------------------------------------------------------------------------------

int check(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* out) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    if (K % 16 != 0) return IK_ERR_SHAPE;
    return IK_OK;
}

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream, bool gemv) {
    IK_TRY(check(x, w, scales, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0) return cudaMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const float* sp = (const float*)scales->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > 4 || M * K * 2 > 49152) return IK_ERR_SHAPE;
        return f32o ? gemv_by_m((int)M, xp, wp, sp, (float*)out->data, N, K, s)
                    : gemv_by_m((int)M, xp, wp, sp, (ik_bf16*)out->data, N, K, s);
    }
    return f32o ? mma_by_m(xp, wp, sp, (float*)out->data, M, N, K, s) : mma_by_m(xp, wp, sp, (ik_bf16*)out->data, M, N, K, s);
}

// M == 1, bf16 activations and outputs.
int pair(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* sc0, const ik_tensor* w1, const ik_tensor* sc1,
         ik_tensor* o0, ik_tensor* o1, void* stream) {
    IK_TRY(check(x, w0, sc0, o0));
    IK_TRY(check(x, w1, sc1, o1));
    if (o0->dtype != IK_DTYPE_BF16 || o1->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (M != 1 || M * K * 2 > 49152) return IK_ERR_SHAPE;
    if (N0 == 0 && N1 == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    if (K == 0) {
        if (cudaMemsetAsync(o0->data, 0, ik_numel(o0) * 2, s) != cudaSuccess) return IK_ERR_LAUNCH;
        return cudaMemsetAsync(o1->data, 0, ik_numel(o1) * 2, s) == cudaSuccess ? IK_OK : IK_ERR_LAUNCH;
    }
    if (N0 > 0x7fffffffLL || N1 > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    const unsigned grid = (unsigned)((N0 + kWarps - 1) / kWarps + (N1 + kWarps - 1) / kWarps);
    fp8_gemv_pair_kernel<ik_bf16><<<grid, kWarps * IK_WARP, (size_t)K * 2, s>>>(
        (const ik_bf16*)x->data, (const uint8_t*)w0->data, (const float*)sc0->data, (ik_bf16*)o0->data, (int)N0,
        (const uint8_t*)w1->data, (const float*)sc1->data, (ik_bf16*)o1->data, (int)N1, (int)K);
    return ik_launch_status();
}

// x / z [R, 128] (R * 128 == K), nw [128], w [N, K] fp8, out [1, N] bf16.
int gated_matmul(const ik_tensor* x, const ik_tensor* z, const ik_tensor* nw, float eps, float woff, const ik_tensor* w,
                 const ik_tensor* scales, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(z, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(nw, IK_DTYPE_BF16, 1));
    const int64_t rows = x->shape[0], D = x->shape[1], K = rows * D, N = w->shape[0];
    if (z->shape[0] != rows || z->shape[1] != D || nw->shape[0] != D || D != 128) return IK_ERR_SHAPE;
    // the matmul sees the normalised rows as one activation row [1, K]
    ik_tensor xv = *x;
    xv.ndim = 2;
    xv.shape[0] = 1;
    xv.shape[1] = K;
    xv.stride[0] = K;
    xv.stride[1] = 1;
    IK_TRY(check(&xv, w, scales, out));
    if (out->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    if (K * 2 > 49152 || !ik_aligned16(x->data) || !ik_aligned16(z->data) || !ik_aligned16(nw->data)) return IK_ERR_SHAPE;
    if (N == 0 || K == 0) return IK_ERR_SHAPE;  // (nothing the model produces; the unfused ops handle empty shapes)
    if (N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    fp8_gemv_gated_kernel<<<grid, kWarps * IK_WARP, (size_t)K * 2, IK_STREAM(stream)>>>(
        (const ik_bf16*)x->data, (const ik_bf16*)z->data, (const ik_bf16*)nw->data, eps, woff, (const uint8_t*)w->data,
        (const float*)scales->data, (ik_bf16*)out->data, (int)N, (int)K);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_gated_rmsnorm_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* z, const ik_tensor* nw, float eps, float w_offset,
                                     const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return gated_matmul(x, z, nw, eps, w_offset, w, scales, out, stream);
}

int ik_matmul_fp8_pair_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* scales0, const ik_tensor* w1,
                            const ik_tensor* scales1, ik_tensor* out0, ik_tensor* out1, void* stream) {
    return pair(x, w0, scales0, w1, scales1, out0, out1, stream);
}

int ik_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, true);
}
int ik_matmul_fp8_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, false);
}

}  // extern "C"
