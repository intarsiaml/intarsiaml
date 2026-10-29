// matmul_fp8: out[M,N] = x[M,K] @ W^T, W = float(w_e4m3[N,K]) * weight_scale (per tensor).
//   scales = f32[2] = (weight_scale, input_scale); input_scale > 0 selects W8A8 (static activation quantisation:
//   xq = e4m3(x / input_scale), RNE, saturating; y = weight_scale * input_scale * sum xq*wq), otherwise W8A16.
//   The mode is read from the device tensor inside the kernels (a uniform branch), so launching never needs a
//   host read and the same kernels serve both recipes.
//   *_gemv  decode: M <= 4, warp-per-output-row, x staged in shared memory (quantised once per block).
//   *_mma   any M: 64-column tiles, weights dequantised to bf16 in shared memory (exact), bf16 tensor-core MMA,
//           f32 accumulate. Used for prefill and as the correct path for every shape.
// Both are deterministic (fixed reduction order). Activations bf16; output bf16 or f32.
#include <cstdlib>
#include <cstring>

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

// ---- Row-exact multi-row GEMV (G-B) ---------------------------------------------------------------------------------
//
// out[m, n] of an M-row call is bitwise out[0, n] of the M = 1 call on row m alone (fp8_gemv_kernel<1>), for every M up to
// kRowsMax. The arithmetic IS fp8_gemv_body's: a warp owns an output row and walks the K axis lane-strided (lane l takes the
// 16-element blocks u = l, l + 32, ...), the same fmaf chain per block, the same `acc += s`, the same warp tree; M only adds
// independent accumulators. The staged activations are the same bf16 values too (xq in W8A8, x itself in W8A16). What differs
// is how much of x is staged at a time: fp8_gemv_body keeps all of M * K * 2 bytes in shared memory (128 KB at M = 16, K = 4096),
// which no block can hold, so this kernel stages K in chunks of KC elements (a multiple of 512 = 32 lanes * 16, so a lane's
// block sequence u = l, l + 32, ... is the same sequence, chunk after chunk, in the same order). Any K % 16 == 0 therefore
// works at any M <= 16, W8A8 and W8A16 alike, with at most 32 KB of shared memory per block.
// Layout of a staged chunk: [m][half][u] 16-byte vectors, so a warp reads 32 consecutive vectors (no bank conflicts).
constexpr int kRowsMax = 16;

template <int M>
constexpr int rows_chunk() {
    return (16384 / M / 512 > 0 ? 16384 / M / 512 : 1) * 512;  // M * chunk * 2 bytes <= 32 KB
}

// NT: outputs per warp. A block stages the activations of M rows (and, for W8A8, requantises every element) for the kWarps * NT
// outputs it owns, and each staged 16-element x vector is unpacked once per lane-block for all NT outputs; NT = 1 is the original
// one-output-per-warp kernel. The arithmetic of every (m, n) is untouched (same lane-strided blocks, same 16-long fmaf chain,
// same `acc += s`, same warp tree), so any NT gives the bits of NT = 1.
template <int M, int NT, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
fp8_gemv_rows_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                     OutT* __restrict__ out, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int KC = rows_chunk<M>();
    uint4* xs = reinterpret_cast<uint4*>(smem);  // [M][2][KC / 16]
    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t nb = ((int64_t)blockIdx.x * kWarps + warp) * NT;  // this warp's outputs: nb .. nb + NT - 1
    const uint4* wr[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) wr[j] = reinterpret_cast<const uint4*>(w + (nb + j < N ? nb + j : 0) * K);  // a dead output reads row 0
    float acc[NT][M];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[j][m] = 0.f;
    for (int c0 = 0; c0 < K; c0 += KC) {
        const int kc = K - c0 < KC ? K - c0 : KC;  // a multiple of 16
        const int ub = c0 / 16;                      // a multiple of 32
        __syncthreads();                             // the previous chunk is consumed
        for (int i = threadIdx.x; i < M * (kc / 8); i += blockDim.x) {
            const int m = i / (kc / 8), j = i % (kc / 8);  // j: 8-element piece of the chunk
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * K + c0) + j);
            if (actq) {
                float f[8];
                ik_unpack8(v, f);
#pragma unroll
                for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
                v = ik_pack8(f);  // fp8 values are exact in bf16
            }
            xs[(m * 2 + (j & 1)) * (KC / 16) + (j >> 1)] = v;
        }
        __syncthreads();
        if (nb < N) {  // a dead warp (all its outputs past N) still took part in the staging and the barriers
#pragma unroll(NT == 1 ? 2 : 1)
            for (int u = lane; u < kc / 16; u += IK_WARP) {
                float wf[NT][16];
#pragma unroll
                for (int j = 0; j < NT; ++j) ik_fp8x16_to_f32(__ldcs(wr[j] + ub + u), wf[j]);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    float xf[16];
                    ik_unpack8(xs[(m * 2) * (KC / 16) + u], xf);
                    ik_unpack8(xs[(m * 2 + 1) * (KC / 16) + u], xf + 8);
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        float sdot = 0.f;
#pragma unroll
                        for (int e = 0; e < 16; ++e) sdot = fmaf(wf[j][e], xf[e], sdot);
                        acc[j][m] += sdot;
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float t = ik_warp_sum(acc[j][m]);  // all 32 lanes call it (full-mask shuffles)
            if (lane == 0 && nb + j < N) ik_store(out + (int64_t)m * N + nb + j, t * s_out);
        }
}

// ---- Row-exact multi-row GEMV, persistent variant: the whole K staged once per block -------------------------------------
//
// The chunked kernel above re-stages (and, in W8A8, re-quantises with a correctly rounded division per element) the M activation
// rows for every kWarps * NT outputs, about as many instructions as the dot products themselves, with a barrier per chunk. When
// M * K * 2 bytes fit one block's shared memory (M = 16: K <= 3072, i.e. every 2048-wide projection of the 35B), this kernel stages
// all of x once per block and the block's warps then walk the output tiles with no barrier at all. The arithmetic of every
// (m, n) is again the one-row kernel's: the same lane-strided 16-element blocks, fmaf chain, `acc += s` and warp tree.
constexpr size_t kFullSmemMax = 96 * 1024;

// Raise a kernel's dynamic shared-memory cap to `bytes` once per device (a per-call size would race between host threads
// launching the kernel with different K: one shrinking the cap between another's set and launch fails that launch).
template <auto Kern>  // one flag per kernel instantiation
bool smem_cap_once(size_t bytes) {
    static bool done[64];
    int d = 0;
    cudaGetDevice(&d);
    if (d < 0 || d >= 64) d = 0;
    if (!done[d]) done[d] = cudaFuncSetAttribute((const void*)Kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes) == cudaSuccess;
    return done[d];
}

template <int M, int NT, int W, typename OutT>
__global__ void __launch_bounds__(W * IK_WARP, 1)
fp8_gemv_rows_full_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                          OutT* __restrict__ out, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);  // [M][2][K / 16]
    const int KV = K / 16;
    const bool actq = sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = sc[0] * s_in;
    for (int i = threadIdx.x; i < M * (K / 8); i += blockDim.x) {
        const int m = i / (K / 8), j = i % (K / 8);  // j: 8-element piece of the row
        uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * K) + j);
        if (actq) {
            float f[8];
            ik_unpack8(v, f);
#pragma unroll
            for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
            v = ik_pack8(f);  // fp8 values are exact in bf16
        }
        xs[(m * 2 + (j & 1)) * KV + (j >> 1)] = v;
    }
    __syncthreads();
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t tiles = ((int64_t)N + NT - 1) / NT;
    for (int64_t t = (int64_t)blockIdx.x * W + warp; t < tiles; t += (int64_t)gridDim.x * W) {
        const int64_t nb = t * NT;  // this tile's outputs: nb .. nb + NT - 1
        const uint4* wr[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) wr[j] = reinterpret_cast<const uint4*>(w + (nb + j < N ? nb + j : 0) * K);  // a dead output reads row 0
        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < M; ++m) acc[j][m] = 0.f;
#pragma unroll 2
        for (int u = lane; u < KV; u += IK_WARP) {
            float wf[NT][16];
#pragma unroll
            for (int j = 0; j < NT; ++j) ik_fp8x16_to_f32(__ldcs(wr[j] + u), wf[j]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[16];
                ik_unpack8(xs[(m * 2) * KV + u], xf);
                ik_unpack8(xs[(m * 2 + 1) * KV + u], xf + 8);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    float sdot = 0.f;
#pragma unroll
                    for (int e = 0; e < 16; ++e) sdot = fmaf(wf[j][e], xf[e], sdot);
                    acc[j][m] += sdot;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float r = ik_warp_sum(acc[j][m]);  // all 32 lanes call it (full-mask shuffles)
                if (lane == 0 && nb + j < N) ik_store(out + (int64_t)m * N + nb + j, r * s_out);
            }
    }
}

inline int sm_count() {
    static int cached[64];
    int d = 0;
    cudaGetDevice(&d);
    if (d < 0 || d >= 64) d = 0;
    if (!cached[d]) {
        int v = 0;
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d);
        cached[d] = v > 0 ? v : 48;
    }
    return cached[d];
}

template <int M, int NT, int W, typename OutT>
int rows_run_full(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    const size_t smem = (size_t)M * K * 2;
    auto kern = fp8_gemv_rows_full_kernel<M, NT, W, OutT>;
    const bool cap_set = smem_cap_once<fp8_gemv_rows_full_kernel<M, NT, W, OutT>>(kFullSmemMax);
    if (!cap_set) return IK_ERR_LAUNCH;
    const int64_t tiles = (N + NT - 1) / NT;
    const int64_t blocks = (tiles + W - 1) / W;
    const unsigned grid = (unsigned)(blocks < sm_count() ? blocks : sm_count());
    kern<<<grid, W * IK_WARP, smem, s>>>(x, w, sc, out, (int)N, (int)K);
    return ik_launch_status();
}

// ---- Row-exact multi-row GEMV, e4m3-staged persistent variant ------------------------------------------------------------
//
// W8A8 (every fp8 projection of the 35B): the staged activations are e4m3 values, so a block can hold their codes, one byte
// each, instead of their bf16 images: M * K bytes, half the persistent kernel's shared memory. That fits M = 16 at K = 4096
// (the GDN out / o_proj calls, which otherwise take the chunked kernel and re-quantise all of x in each of its blocks) and leaves
// room for more than one block per SM at K = 2048. A code unpacks to exactly the float its bf16 image unpacks to (e4m3 -> f16 ->
// f32 is exact), so every (m, n) runs the one-row kernel's arithmetic: the same lane-strided 16-element blocks, the same 16-long
// fmaf chain, the same `acc += s` and warp tree.
// W8A16 (no input scale; decided on the device, the launcher never reads the scales): x is not on the e4m3 grid, so the block
// stages bf16 rows in groups that fit the same M * K bytes (M / 2 rows twice, then the odd row) and walks the tiles once per
// group: the same arithmetic per (m, n), the weights read once per group (a correct fallback; no model here takes it).
constexpr size_t kQSmemMax = 96 * 1024;

// rows m0 .. m0 + RM - 1 of the call, staged in xs: XQ as codes [RM][KV] (16 codes per uint4, element order), else as bf16
// halves [RM][2][KV] (the persistent kernel's layout)
template <int RM, int NT, int W, bool XQ, typename OutT>
__device__ __forceinline__ void rows_q_tiles(const uint4* __restrict__ xs, const uint8_t* __restrict__ w, float s_out,
                                             OutT* __restrict__ out, int m0, int N, int K) {
    const int KV = K / 16;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t tiles = ((int64_t)N + NT - 1) / NT;
    for (int64_t t = (int64_t)blockIdx.x * W + warp; t < tiles; t += (int64_t)gridDim.x * W) {
        const int64_t nb = t * NT;  // this tile's outputs: nb .. nb + NT - 1
        const uint4* wr[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) wr[j] = reinterpret_cast<const uint4*>(w + (nb + j < N ? nb + j : 0) * (int64_t)K);  // a dead output reads row 0
        float acc[NT][RM];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < RM; ++m) acc[j][m] = 0.f;
#pragma unroll 2
        for (int u = lane; u < KV; u += IK_WARP) {
            if constexpr (XQ) {
                // lane moe16c: codes and weights to f16 pairs only; the f16 x f16 + f32 fma takes them (ik_fma2_f16: the same
                // 16-long chain, element order, f32 products and roundings as fmaf on the f32 images below)
                uint32_t wh[NT][8];
#pragma unroll
                for (int j = 0; j < NT; ++j) ik_fp8x16_to_h2(__ldcs(wr[j] + u), wh[j]);
#pragma unroll
                for (int m = 0; m < RM; ++m) {
                    uint32_t xh[8];
                    ik_fp8x16_to_h2(xs[m * KV + u], xh);
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        float sdot = 0.f;
#pragma unroll
                        for (int p = 0; p < 8; ++p) sdot = ik_fma2_f16(wh[j][p], xh[p], sdot);
                        acc[j][m] += sdot;
                    }
                }
            } else {
                float wf[NT][16];
#pragma unroll
                for (int j = 0; j < NT; ++j) ik_fp8x16_to_f32(__ldcs(wr[j] + u), wf[j]);
#pragma unroll
                for (int m = 0; m < RM; ++m) {
                    float xf[16];
                    ik_unpack8(xs[(m * 2) * KV + u], xf);
                    ik_unpack8(xs[(m * 2 + 1) * KV + u], xf + 8);
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        float sdot = 0.f;
#pragma unroll
                        for (int e = 0; e < 16; ++e) sdot = fmaf(wf[j][e], xf[e], sdot);
                        acc[j][m] += sdot;
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < RM; ++m) {
                const float r = ik_warp_sum(acc[j][m]);  // all 32 lanes call it (full-mask shuffles)
                if (lane == 0 && nb + j < N) ik_store(out + (int64_t)(m0 + m) * N + nb + j, r * s_out);
            }
    }
}

// bf16 rows m0 .. m0 + rm - 1 of x into xs as [rm][2][KV] (W8A16: x itself)
IK_DEVI void rows_q_stage_bf16(const ik_bf16* __restrict__ x, uint4* __restrict__ xs, int m0, int rm, int K) {
    const int KV = K / 16;
    for (int i = threadIdx.x; i < rm * (K / 8); i += blockDim.x) {
        const int m = i / (K / 8), j = i % (K / 8);  // j: 8-element piece of the row
        xs[(m * 2 + (j & 1)) * KV + (j >> 1)] = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)(m0 + m) * K) + j);
    }
}

template <int M, int NT, int W, int MINB, typename OutT>
__global__ void __launch_bounds__(W * IK_WARP, MINB)
fp8_gemv_rows_q_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc,
                       OutT* __restrict__ out, int N, int K) {
    static_assert(M >= 2, "one bf16 row is 2 K bytes, more than the M * K the block holds at M = 1");
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);
    const int KV = K / 16;
    if (sc[1] > 0.f) {  // W8A8: codes [M][KV]
        const float s_in = sc[1];
        for (int i = threadIdx.x; i < M * KV; i += blockDim.x) {
            const int m = i / KV, u = i % KV;
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K) + 2 * u;
            float f[16];
            ik_unpack8(__ldg(xp), f);
            ik_unpack8(__ldg(xp + 1), f + 8);
            uint32_t c[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                c[q] = (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 * q], s_in)) | ((uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 * q + 1], s_in)) << 8) |
                       ((uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 * q + 2], s_in)) << 16) | ((uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 * q + 3], s_in)) << 24);
            xs[i] = make_uint4(c[0], c[1], c[2], c[3]);
        }
        __syncthreads();
        rows_q_tiles<M, NT, W, true>(xs, w, sc[0] * s_in, out, 0, N, K);
        return;
    }
    constexpr int G = M / 2;  // W8A16: bf16 rows, G at a time (G * K * 2 <= M * K bytes)
#pragma unroll 1
    for (int g = 0; g < 2; ++g) {
        __syncthreads();  // the previous group is consumed
        rows_q_stage_bf16(x, xs, g * G, G, K);
        __syncthreads();
        rows_q_tiles<G, NT, W, false>(xs, w, sc[0], out, g * G, N, K);
    }
    if constexpr (M % 2 == 1) {
        __syncthreads();
        rows_q_stage_bf16(x, xs, M - 1, 1, K);
        __syncthreads();
        rows_q_tiles<1, NT, W, false>(xs, w, sc[0], out, M - 1, N, K);
    }
}

template <int M, int NT, int W, int MINB, typename OutT>
int rows_run_q(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    const size_t smem = (size_t)M * K;
    auto kern = fp8_gemv_rows_q_kernel<M, NT, W, MINB, OutT>;
    const bool cap_set = smem_cap_once<fp8_gemv_rows_q_kernel<M, NT, W, MINB, OutT>>(kQSmemMax);
    if (!cap_set) return IK_ERR_LAUNCH;
    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, W * IK_WARP, smem) != cudaSuccess || occ < 1) occ = 1;
    const int64_t tiles = (N + NT - 1) / NT;
    const int64_t blocks = (tiles + W - 1) / W;
    const int64_t cap = (int64_t)sm_count() * occ;
    const unsigned grid = (unsigned)(blocks < cap ? blocks : cap);
    kern<<<grid, W * IK_WARP, smem, s>>>(x, w, sc, out, (int)N, (int)K);
    return ik_launch_status();
}

constexpr int64_t kQSmallN = 1024;

// INTARSIA_CUDA_FP8_ROWS_QSHAPE = NTxW or NTxWxB (B: blocks per SM the register budget is sized for): the e4m3-staged kernel's shape
template <int M, typename OutT>
int rows_run_q_shape(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    const char* f = getenv("INTARSIA_CUDA_FP8_ROWS_QSHAPE");
    if (f) {
        if (!strcmp(f, "1x16")) return rows_run_q<M, 1, 16, 1>(x, w, sc, out, N, K, s);
        if (!strcmp(f, "2x16")) return rows_run_q<M, 2, 16, 1>(x, w, sc, out, N, K, s);
        if (!strcmp(f, "1x8")) return rows_run_q<M, 1, 8, 2>(x, w, sc, out, N, K, s);
        if (!strcmp(f, "1x8x3")) return rows_run_q<M, 1, 8, 3>(x, w, sc, out, N, K, s);
        if (!strcmp(f, "4x8")) return rows_run_q<M, 4, 8, 1>(x, w, sc, out, N, K, s);
    }
    if (N <= kQSmallN) return rows_run_q<M, 1, 16, 1>(x, w, sc, out, N, K, s);
    return rows_run_q<M, 2, 8, 2>(x, w, sc, out, N, K, s);
}

// INTARSIA_CUDA_FP8_ROWS_Q (default on, decided 2026-10-28 over the lane's speed gates): unset / `1` = the e4m3-staged kernel where the bf16 activations do not fit the persistent kernel
// (M * K * 2 > 96 KB, i.e. K = 4096 at M >= 13), for every call at M >= 12, and for N <= 1024 at M >= 4 (shape 1x16), `0` = never (the kernels of before), `all` = wherever
// M >= 2 and M * K bytes fit
inline int rows_q_mode() {
    const char* e = getenv("INTARSIA_CUDA_FP8_ROWS_Q");
    if (e && !strcmp(e, "all")) return 2;
    if (e && e[0] == '0' && e[1] == 0) return 0;
    return 1;  // default on (docs/results/2026-10-28-msd2-dense-gemv.md, Decision)
}

// Which kernel runs an M-row call over N outputs and K inputs. INTARSIA_CUDA_FP8_ROWS = 0 is the original kernel (the switch that
// turns this change off), `tile` the chunked kernel with INTARSIA_CUDA_FP8_ROWS_NT outputs per warp (1 | 2 | 4, default: the largest
// that leaves a grid of kRowsMinBlocks blocks), anything else (default) the persistent kernel when x fits shared memory and the
// tiled one when it does not. INTARSIA_CUDA_FP8_ROWS_FULL = NTxW (1x16 | 2x16 | 2x8 | 4x8) picks the persistent kernel's shape.
constexpr int kRowsMinBlocks = 96;
inline int rows_tile_nt(int M, int64_t N) {
    const char* e = getenv("INTARSIA_CUDA_FP8_ROWS_NT");
    if (e && (e[0] == '1' || e[0] == '2' || e[0] == '4') && e[1] == 0) return e[0] - '0';
    if (M < 4) return 1;
    for (int nt = 4; nt > 1; nt /= 2)
        if ((N + kWarps * nt - 1) / (kWarps * nt) >= kRowsMinBlocks) return nt;
    return 1;
}

template <int M, int NT, typename OutT>
int rows_run_nt(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps * NT - 1) / (kWarps * NT));
    fp8_gemv_rows_kernel<M, NT, OutT><<<grid, kWarps * IK_WARP, (size_t)M * rows_chunk<M>() * 2, s>>>(x, w, sc, out, (int)N, (int)K);
    return ik_launch_status();
}

template <int M, typename OutT>
int rows_run(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    const char* mode = getenv("INTARSIA_CUDA_FP8_ROWS");
    const bool off = mode && mode[0] == '0' && mode[1] == 0;
    const bool tile = mode && !strcmp(mode, "tile");
    if constexpr (M >= 2) {
        const int q = off || tile ? 0 : rows_q_mode();
        const bool fits_q = (size_t)M * K <= kQSmemMax, fits_full = (size_t)M * K * 2 <= kFullSmemMax;
        const bool small_n = N <= kQSmallN && M >= 4;  // k / v: the persistent kernel's 2x16 grid is a third of the SMs
        // from 12 rows the codes' half-size staging (2 blocks per SM) beats the bf16 one at K = 2048 too (evicted 16-row
        // micro-timing 8192x2048: 88.7 -> 82.5 us, 4096x2048: 50.7 -> 49.0)
        if (fits_q && (q == 2 || (q == 1 && (!fits_full || small_n || M >= 12)))) return rows_run_q_shape<M>(x, w, sc, out, N, K, s);
    }
    if (!off && !tile && (size_t)M * K * 2 <= kFullSmemMax) {
        const char* f = getenv("INTARSIA_CUDA_FP8_ROWS_FULL");
        const bool known = f && (!strcmp(f, "1x16") || !strcmp(f, "2x16") || !strcmp(f, "2x8") || !strcmp(f, "4x8"));
        if (known && !strcmp(f, "1x16")) return rows_run_full<M, 1, 16>(x, w, sc, out, N, K, s);
        if (known && !strcmp(f, "2x8")) return rows_run_full<M, 2, 8>(x, w, sc, out, N, K, s);
        if (known && !strcmp(f, "4x8")) return rows_run_full<M, 4, 8>(x, w, sc, out, N, K, s);
        return rows_run_full<M, 2, 16>(x, w, sc, out, N, K, s);
    }
    switch (off ? 1 : rows_tile_nt(M, N)) {
        case 2: return rows_run_nt<M, 2>(x, w, sc, out, N, K, s);
        case 4: return rows_run_nt<M, 4>(x, w, sc, out, N, K, s);
        default: return rows_run_nt<M, 1>(x, w, sc, out, N, K, s);
    }
}

template <typename OutT>
int rows_by_m(int M, const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int64_t N, int64_t K, cudaStream_t s) {
    switch (M) {
#define IK_FP8_ROWS_CASE(m) case m: return rows_run<m>(x, w, sc, out, N, K, s);
        IK_FP8_ROWS_CASE(1) IK_FP8_ROWS_CASE(2) IK_FP8_ROWS_CASE(3) IK_FP8_ROWS_CASE(4)
        IK_FP8_ROWS_CASE(5) IK_FP8_ROWS_CASE(6) IK_FP8_ROWS_CASE(7) IK_FP8_ROWS_CASE(8)
        IK_FP8_ROWS_CASE(9) IK_FP8_ROWS_CASE(10) IK_FP8_ROWS_CASE(11) IK_FP8_ROWS_CASE(12)
        IK_FP8_ROWS_CASE(13) IK_FP8_ROWS_CASE(14) IK_FP8_ROWS_CASE(15) IK_FP8_ROWS_CASE(16)
#undef IK_FP8_ROWS_CASE
        default: return IK_ERR_SHAPE;
    }
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

// rows: the row-exact multi-row GEMV (M <= kRowsMax, activations staged as e4m3 bytes), `gemv`: the decode GEMV (M <= 4).
int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream, bool gemv,
           bool rows = false) {
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
    if (rows) {
        if (M > kRowsMax) return IK_ERR_SHAPE;
        if (!ik_aligned16(x->data) || !ik_aligned16(w->data)) return IK_ERR_SHAPE;
        return f32o ? rows_by_m((int)M, xp, wp, sp, (float*)out->data, N, K, s)
                    : rows_by_m((int)M, xp, wp, sp, (ik_bf16*)out->data, N, K, s);
    }
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
// Row-exact multi-row GEMV: row m of an M-row call (M <= 16) is bitwise the M = 1 `ik_matmul_fp8_gemv` call on row m alone.
int ik_matmul_fp8_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, false, true);
}
int ik_matmul_fp8_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return launch(x, w, scales, out, stream, false);
}

}  // extern "C"
