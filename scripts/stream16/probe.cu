// Lane stream16: standalone streaming probe for the 16-row decode GEMVs on GB10 (spark1, sm_121).
// The baseline kernels are copies of kernels/cuda/fp8.cu fp8_gemv_rows_q_kernel (W8A8, e4m3-staged x) and
// kernels/cuda/nvfp4.cu nvfp4_gemv_full_kernel (the lm_head); the variants only change which loads are in flight when
// (register double-buffered chunks, an L2 bulk prefetch some tiles ahead, block shape). Every (m, n) keeps the baseline's
// lane-strided u sequence, chains and warp tree, and the probe memcmp's each variant's output against the baseline.
// Weights cycle through a >= 256 MB pool so every call streams from DRAM (as in the decode step).
//   nvcc -O3 --std=c++17 -arch=sm_121 -I <repo>/kernels/cuda probe.cu -o probe && ./probe [fp8|nv|all]
#include "ik_quant.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

#ifndef PM
#define PM 16
#endif
constexpr int M = PM;

IK_DEVI void bulk_prefetch_l2(const void* p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// ---------------------------------------------------------------- fp8 rows_q, W8A8 -------------------------------------
IK_DEVI void stage_codes(const ik_bf16* __restrict__ x, uint4* __restrict__ xs, float s_in, int K) {
    const int KV = K / 16;
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
}

// one u block of one tile: the baseline's XQ body
template <int NT>
IK_DEVI void fp8_u(const uint4* __restrict__ xs, const uint4 (&wv)[NT], int u, int KV, float (&acc)[NT][M]) {
    uint32_t wh[NT][8];
#pragma unroll
    for (int j = 0; j < NT; ++j) ik_fp8x16_to_h2(wv[j], wh[j]);
#pragma unroll
    for (int m = 0; m < M; ++m) {
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
}

template <int NT, typename OutT>
IK_DEVI void fp8_store(float (&acc)[NT][M], OutT* out, int64_t nb, int N, float s_out, int lane) {
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float r = ik_warp_sum(acc[j][m]);
            if (lane == 0 && nb + j < N) ik_store(out + (int64_t)m * N + nb + j, r * s_out);
        }
}

// baseline: rows_q_tiles<16, NT, W, true> as on main
template <int NT, int W, int MINB>
__global__ void __launch_bounds__(W * IK_WARP, MINB)
fp8_base(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc, ik_bf16* __restrict__ out, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);
    stage_codes(x, xs, sc[1], K);
    const float s_out = sc[0] * sc[1];
    const int KV = K / 16;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t tiles = ((int64_t)N + NT - 1) / NT;
    const int64_t stride = (int64_t)gridDim.x * W;
    for (int64_t t = (int64_t)blockIdx.x * W + warp; t < tiles; t += stride) {
        const int64_t nb = t * NT;
        const uint4* wr[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) wr[j] = reinterpret_cast<const uint4*>(w + (nb + j < N ? nb + j : 0) * (int64_t)K);
        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < M; ++m) acc[j][m] = 0.f;
#pragma unroll 2
        for (int u = lane; u < KV; u += IK_WARP) {
            uint4 wv[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j) wv[j] = __ldcs(wr[j] + u);
            fp8_u<NT>(xs, wv, u, KV, acc);
        }
        fp8_store<NT>(acc, out, nb, N, s_out, lane);
    }
}

// pipelined: the warp's (tile, chunk) sequence with chunk c+1's loads (CH u blocks per row) issued before chunk c's math;
// KPL = KV / 32 u blocks per lane per row (compile time); PF > 0: lane j < NT bulk-prefetches row j of the tile PF strides ahead
// into L2 at each tile start. BLK = 1: a block owns a contiguous run of tiles (warp-interleaved inside it) instead of the grid stride.
template <int NT, int W, int MINB, int KPL, int CH, int PF, int BLK, int EP = 0, int ER = 0, int NOQ = 0>
__global__ void __launch_bounds__(W * IK_WARP, MINB)
fp8_pipe(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ sc, ik_bf16* __restrict__ out, int N, int K) {
    static_assert(KPL % CH == 0, "chunks tile the lane's u blocks");
    constexpr int NC = KPL / CH;
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);
    const int KV = K / 16;  // == 32 * KPL
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t tiles = ((int64_t)N + NT - 1) / NT;
    int64_t t, tend, stride;
    if (BLK) {
        const int64_t per = (tiles + gridDim.x - 1) / gridDim.x;
        t = (int64_t)blockIdx.x * per + warp; const int64_t e = (int64_t)(blockIdx.x + 1) * per; tend = e < tiles ? e : tiles; stride = W;
    } else {
        t = (int64_t)blockIdx.x * W + warp; tend = tiles; stride = (int64_t)gridDim.x * W;
    }
    auto row = [&](int64_t tt, int j) {
        const int64_t n = tt * NT + j;
        return reinterpret_cast<const uint4*>(w + (n < N ? n : 0) * (int64_t)K);
    };
    // early: the warp's first EP tiles into L2 and (ER) its first chunk into registers before the block stages x
    if (EP > 0 && lane < NT) {
#pragma unroll
        for (int e = 0; e < EP; ++e)
            if (t + e * stride < tend) bulk_prefetch_l2(row(t + e * stride, lane), (uint32_t)K);
    }
    uint4 cur[CH][NT], nxt[CH][NT];
    if (ER && t < tend) {
#pragma unroll
        for (int i = 0; i < CH; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) cur[i][j] = __ldcs(row(t, j) + lane + 32 * i);
    }
    if (NOQ) {  // diagnostic: x already holds the codes (no quantising prologue)
        for (int i = threadIdx.x; i < M * KV; i += blockDim.x) xs[i] = __ldg(reinterpret_cast<const uint4*>(x) + i);
        __syncthreads();
    } else {
        stage_codes(x, xs, sc[1], K);
    }
    const float s_out = sc[0] * sc[1];
    if (t >= tend) return;
    if (!ER) {
#pragma unroll
        for (int i = 0; i < CH; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) cur[i][j] = __ldcs(row(t, j) + lane + 32 * i);
    }
    for (;;) {
        if (PF > 0 && lane < NT) {
            const int64_t tp = t + PF * stride;
            if (tp < tend) bulk_prefetch_l2(row(tp, lane), (uint32_t)K);
        }
        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < M; ++m) acc[j][m] = 0.f;
        const int64_t tn = t + stride;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const bool last = c + 1 == NC;
            const int64_t tl = last ? tn : t;
            const int cb = last ? 0 : (c + 1) * CH;
            if (tl < tend) {
#pragma unroll
                for (int i = 0; i < CH; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j) nxt[i][j] = __ldcs(row(tl, j) + lane + 32 * (cb + i));
            }
#pragma unroll
            for (int i = 0; i < CH; ++i) fp8_u<NT>(xs, cur[i], lane + 32 * (c * CH + i), KV, acc);
#pragma unroll
            for (int i = 0; i < CH; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) cur[i][j] = nxt[i][j];
        }
        fp8_store<NT>(acc, out, t * NT, N, s_out, lane);
        t = tn;
        if (t >= tend) break;
    }
}

__global__ void codes_kernel(const ik_bf16* x, uint4* xq, const float* sc, int K) { stage_codes(x, xq, sc[1], K); }

// read-only ceiling at the baseline's footprint: the same tiles and loads, a xor instead of the math
template <int NT, int W>
__global__ void __launch_bounds__(W * IK_WARP) fp8_read(const uint8_t* __restrict__ w, ik_bf16* __restrict__ out, int N, int K) {
    const int KV = K / 16;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t tiles = ((int64_t)N + NT - 1) / NT;
    uint32_t a = 0;
    for (int64_t t = (int64_t)blockIdx.x * W + warp; t < tiles; t += (int64_t)gridDim.x * W) {
#pragma unroll 4
        for (int u = lane; u < KV; u += IK_WARP)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4 v = __ldcs(reinterpret_cast<const uint4*>(w + (t * NT + j) * (int64_t)K) + u);
                a ^= v.x ^ v.y ^ v.z ^ v.w;
            }
    }
    if (a == 0x12345678u) out[0] = ik_f2b(1.f);
}

// ---------------------------------------------------------------- nvfp4 lm_head -------------------------------------------
IK_DEVI void stage_bf16(const ik_bf16* __restrict__ x, uint4* __restrict__ xs, int K) {
    const int KV = K / 16;
    for (int i = threadIdx.x; i < M * (K / 8); i += blockDim.x) {
        const int m = i / (K / 8), j = i % (K / 8);
        xs[(m * 2 + (j & 1)) * KV + (j >> 1)] = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)m * K) + j);
    }
    __syncthreads();
}

template <int NT>
IK_DEVI void nv_u(const uint4* __restrict__ xs, const uint2 (&wv)[NT], const uint8_t (&sb)[NT], int u, int KV, float (&acc)[NT][M]) {
    uint32_t wb[NT][8];
    float sc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        sc[j] = ik_e4m3_to_f(sb[j]);
        ik_fp4x8_to_bf2(wv[j].x, wb[j]);
        ik_fp4x8_to_bf2(wv[j].y, wb[j] + 4);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const uint4 x0 = xs[(m * 2) * KV + u], x1 = xs[(m * 2 + 1) * KV + u];
        const uint32_t xb[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            float sd = 0.f;
#pragma unroll
            for (int p = 0; p < 8; ++p) sd = ik_fma2_bf16(wb[j][p], xb[p], sd);
            acc[j][m] = fmaf(sd, sc[j], acc[j][m]);
        }
    }
}

template <int NT>
IK_DEVI void nv_store(float (&acc)[NT][M], float* out, int64_t nb, int64_t N, float g, int lane) {
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float r = ik_warp_sum(acc[j][m]);
            if (lane == 0 && nb + j < N) ik_store(out + (int64_t)m * N + nb + j, r * g);
        }
}

// baseline: nvfp4_gemv_full_kernel<16, NT, W, float> as on main
template <int NT, int W>
__global__ void __launch_bounds__(W * IK_WARP, 1)
nv_base(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs, const float* __restrict__ gs,
        float* __restrict__ out, int64_t N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);
    stage_bf16(x, xs, K);
    const int KV = K / 16;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t tiles = (N + NT - 1) / NT;
    const float g = gs[0];
    for (int64_t t = (int64_t)blockIdx.x * W + warp; t < tiles; t += (int64_t)gridDim.x * W) {
        const int64_t nb = t * NT;
        const uint2* wr[NT];
        const uint8_t* sr[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int64_t n = nb + j < N ? nb + j : 0;
            wr[j] = reinterpret_cast<const uint2*>(w + n * (K / 2));
            sr[j] = bs + n * (K / 16);
        }
        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < M; ++m) acc[j][m] = 0.f;
#pragma unroll(NT == 1 ? 4 : 2)
        for (int u = lane; u < KV; u += IK_WARP) {
            uint2 wv[NT];
            uint8_t sb[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j) { wv[j] = __ldcs(wr[j] + u); sb[j] = sr[j][u]; }
            nv_u<NT>(xs, wv, sb, u, KV, acc);
        }
        nv_store<NT>(acc, out, nb, N, g, lane);
    }
}

template <int NT, int W, int KPL, int CH, int PF>
__global__ void __launch_bounds__(W * IK_WARP, 1)
nv_pipe(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs, const float* __restrict__ gs,
        float* __restrict__ out, int64_t N, int K) {
    static_assert(KPL % CH == 0, "chunks tile the lane's u blocks");
    constexpr int NC = KPL / CH;
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);
    stage_bf16(x, xs, K);
    const int KV = K / 16;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t tiles = (N + NT - 1) / NT;
    const int64_t stride = (int64_t)gridDim.x * W;
    const float g = gs[0];
    int64_t t = (int64_t)blockIdx.x * W + warp;
    if (t >= tiles) return;
    auto wrow = [&](int64_t tt, int j) {
        const int64_t n = tt * NT + j;
        return reinterpret_cast<const uint2*>(w + (n < N ? n : 0) * (K / 2));
    };
    auto srow = [&](int64_t tt, int j) {
        const int64_t n = tt * NT + j;
        return bs + (n < N ? n : 0) * (K / 16);
    };
    uint2 cur[CH][NT], nxt[CH][NT];
    uint8_t cs[CH][NT], ns[CH][NT];
#pragma unroll
    for (int i = 0; i < CH; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) { cur[i][j] = __ldcs(wrow(t, j) + lane + 32 * i); cs[i][j] = srow(t, j)[lane + 32 * i]; }
    for (;;) {
        if (PF > 0 && lane < NT) {
            const int64_t tp = t + PF * stride;
            if (tp < tiles) bulk_prefetch_l2(wrow(tp, lane), (uint32_t)(K / 2));
        }
        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < M; ++m) acc[j][m] = 0.f;
        const int64_t tn = t + stride;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const bool last = c + 1 == NC;
            const int64_t tl = last ? tn : t;
            const int cb = last ? 0 : (c + 1) * CH;
            if (tl < tiles) {
#pragma unroll
                for (int i = 0; i < CH; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        nxt[i][j] = __ldcs(wrow(tl, j) + lane + 32 * (cb + i));
                        ns[i][j] = srow(tl, j)[lane + 32 * (cb + i)];
                    }
            }
#pragma unroll
            for (int i = 0; i < CH; ++i) nv_u<NT>(xs, cur[i], cs[i], lane + 32 * (c * CH + i), KV, acc);
#pragma unroll
            for (int i = 0; i < CH; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) { cur[i][j] = nxt[i][j]; cs[i][j] = ns[i][j]; }
        }
        nv_store<NT>(acc, out, t * NT, N, g, lane);
        t = tn;
        if (t >= tiles) break;
    }
}

// ---------------------------------------------------------------- host ----------------------------------------------------
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 11); }
static int sms() { int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }

struct Pool {
    std::vector<uint8_t*> w;
    size_t bytes;
};
static Pool make_pool(size_t bytes, const std::vector<uint8_t>& host) {
    Pool p; p.bytes = bytes;
    const size_t n = (size_t)((256ull << 20) + bytes - 1) / bytes;
    for (size_t i = 0; i < (n < 1 ? 1 : n); ++i) {
        uint8_t* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice)); p.w.push_back(d);
    }
    return p;
}
static void free_pool(Pool& p) { for (auto d : p.w) cudaFree(d); p.w.clear(); }

template <typename F>
static float time_us(int n_pool, F&& launch) {
    for (int i = 0; i < 6; ++i) launch(i % n_pool);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int iters = n_pool * ((240 + n_pool - 1) / n_pool);
    float best = 1e30f;
    for (int rep = 0; rep < 3; ++rep) {
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launch(i % n_pool);
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        const float us = ms * 1e3f / iters;
        if (us < best) best = us;
    }
    CK(cudaGetLastError());
    return best;
}

template <auto Kern>
static int occ_of(int threads, size_t smem) {
    CK(cudaFuncSetAttribute((const void*)Kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, Kern, threads, smem));
    return occ < 1 ? 1 : occ;
}

static void fp8_shape(int N, int K) {
    const size_t wb = (size_t)N * K;
    std::vector<uint8_t> hw(wb);
    for (auto& b : hw) { const uint32_t r = rnd(); b = (uint8_t)((r & 0x80) | (r % 0x58)); }
    Pool p = make_pool(wb, hw);
    std::vector<uint16_t> hx((size_t)M * K);
    for (auto& v : hx) { const float f = ((int)(rnd() % 4001) - 2000) / 1000.f; uint32_t u; memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
    ik_bf16* x; CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    float hs[2] = {0.0123f, 0.0371f}; float* sc; CK(cudaMalloc(&sc, 8)); CK(cudaMemcpy(sc, hs, 8, cudaMemcpyHostToDevice));
    uint4* xq; CK(cudaMalloc(&xq, (size_t)M * K)); codes_kernel<<<1, 256>>>(x, xq, sc, K); CK(cudaDeviceSynchronize());
    const size_t ob = (size_t)M * N * 2;
    ik_bf16 *o_ref, *o; CK(cudaMalloc(&o_ref, ob)); CK(cudaMalloc(&o, ob));
    std::vector<uint8_t> h_ref(ob), h(ob);
    const size_t smem = (size_t)M * K;
    const int np = (int)p.w.size(), S = sms();
    auto report = [&](const char* name, float us, bool check, unsigned grid) {
        bool same = true;
        if (check) { CK(cudaMemcpy(h.data(), o, ob, cudaMemcpyDeviceToHost)); same = !memcmp(h.data(), h_ref.data(), ob); }
        printf("FP8 %5dx%-5d %-34s grid %4u %8.2f us %6.1f GB/s %s\n", N, K, name, grid, us, wb / us / 1e3, check ? (same ? "bitwise" : "DIFFERS") : "");
    };
    // baseline (main's default at M = 16: 1x16x1 for N <= 1024, else 2x8x2)
    {
        auto run = [&](auto kern, int nt, int W, const char* name, bool ref, const ik_bf16* xin) {
            const int occ = [&] { CK(cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
                                  int v = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, W * 32, smem)); return v < 1 ? 1 : v; }();
            const int64_t blocks = ((N + nt - 1) / nt + W - 1) / W, cap = (int64_t)S * occ;
            const unsigned grid = (unsigned)(blocks < cap ? blocks : cap);
            kern<<<grid, W * 32, smem>>>(xin, p.w[0], sc, ref ? o_ref : o, N, K);
            CK(cudaDeviceSynchronize());
            if (ref) CK(cudaMemcpy(h_ref.data(), o_ref, ob, cudaMemcpyDeviceToHost));
            const float us = time_us(np, [&](int i) { kern<<<grid, W * 32, smem>>>(xin, p.w[i], sc, o, N, K); });
            if (!ref) { CK(cudaMemset(o, 0, ob)); kern<<<grid, W * 32, smem>>>(xin, p.w[0], sc, o, N, K); CK(cudaDeviceSynchronize()); }
            report(name, us, !ref, grid);
            return grid;
        };
        if (N <= 1024) run(fp8_base<1, 16, 1>, 1, 16, "base 1x16x1 (main)", true, x);
        else run(fp8_base<2, 8, 2>, 2, 8, "base 2x8x2 (main)", true, x);
        run(fp8_base<1, 16, 1>, 1, 16, "base 1x16x1", false, x);
        run(fp8_base<2, 16, 1>, 2, 16, "base 2x16x1", false, x);
#define PIPE(NT, W, MB, KPL, CH, PF, EP, ER, NOQ)                                                                                         \
        run(fp8_pipe<NT, W, MB, KPL, CH, PF, 0, EP, ER, NOQ>, NT, W, "pipe " #NT "x" #W "x" #MB " ch" #CH " pf" #PF " ep" #EP " er" #ER " noq" #NOQ, false, NOQ ? (const ik_bf16*)xq : x)
        if (K == 2048) {
            PIPE(2, 8, 2, 4, 1, 0, 0, 0, 1); PIPE(2, 16, 1, 4, 1, 0, 0, 0, 1); PIPE(1, 16, 1, 4, 1, 0, 0, 0, 1);
            PIPE(2, 8, 2, 4, 1, 0, 0, 0, 0); PIPE(2, 8, 2, 4, 1, 0, 1, 0, 0); PIPE(2, 8, 2, 4, 1, 0, 2, 0, 0); PIPE(2, 8, 2, 4, 1, 0, 0, 1, 0);
            PIPE(2, 8, 2, 4, 1, 0, 1, 1, 0); PIPE(2, 8, 2, 4, 1, 0, 3, 1, 0);
            PIPE(2, 16, 1, 4, 1, 0, 0, 0, 0); PIPE(2, 16, 1, 4, 1, 0, 1, 0, 0); PIPE(2, 16, 1, 4, 1, 0, 2, 0, 0); PIPE(2, 16, 1, 4, 1, 0, 1, 1, 0);
            PIPE(1, 16, 1, 4, 1, 0, 0, 0, 0); PIPE(1, 16, 1, 4, 1, 0, 2, 0, 0); PIPE(1, 16, 1, 4, 1, 0, 2, 1, 0); PIPE(1, 16, 1, 4, 4, 0, 2, 1, 0);
            PIPE(2, 8, 2, 4, 2, 0, 2, 1, 0); PIPE(1, 8, 3, 4, 1, 0, 2, 1, 0);
        } else if (K == 4096) {
            PIPE(2, 8, 1, 8, 1, 0, 0, 0, 1); PIPE(1, 16, 1, 8, 1, 0, 0, 0, 1); PIPE(1, 16, 1, 8, 2, 4, 0, 0, 1);
            PIPE(1, 16, 1, 8, 2, 4, 0, 0, 0); PIPE(1, 16, 1, 8, 2, 4, 1, 0, 0); PIPE(1, 16, 1, 8, 2, 4, 2, 0, 0); PIPE(1, 16, 1, 8, 2, 4, 2, 1, 0);
            PIPE(1, 16, 1, 8, 2, 4, 4, 1, 0); PIPE(1, 16, 1, 8, 2, 2, 2, 1, 0); PIPE(1, 16, 1, 8, 2, 8, 2, 1, 0);
            PIPE(1, 16, 1, 8, 1, 0, 2, 1, 0); PIPE(1, 16, 1, 8, 1, 4, 2, 1, 0); PIPE(1, 16, 1, 8, 4, 4, 2, 1, 0);
            PIPE(2, 16, 1, 8, 1, 0, 2, 1, 0); PIPE(2, 16, 1, 8, 2, 4, 2, 1, 0); PIPE(2, 8, 1, 8, 1, 0, 2, 1, 0);
            PIPE(1, 32, 1, 8, 1, 0, 2, 1, 0); PIPE(1, 32, 1, 8, 2, 4, 2, 1, 0);
        }
#undef PIPE
        {   // the prologue alone: the block stages x and exits
            for (int W : {8, 16}) {
                const unsigned grid = (unsigned)(W == 8 ? S * (K == 2048 ? 2 : 1) : S);
                uint4* scratch; CK(cudaMalloc(&scratch, (size_t)M * K));
                const float us = time_us(np, [&](int) { codes_kernel<<<grid, W * 32>>>(x, scratch, sc, K); });
                cudaFree(scratch);
                printf("FP8 %5dx%-5d %-34s grid %4u %8.2f us (codes staged into global, %d threads)\n", N, K, "prologue only", grid, us, W * 32);
            }
        }
    }
    free_pool(p); cudaFree(xq); cudaFree(x); cudaFree(sc); cudaFree(o); cudaFree(o_ref);
}

static void nv_shape(int64_t N, int K) {
    const size_t wb = (size_t)N * K / 2, sb = (size_t)N * K / 16;
    std::vector<uint8_t> hw(wb), hb(sb);
    for (auto& b : hw) b = (uint8_t)rnd();
    for (auto& b : hb) b = (uint8_t)(0x28 + rnd() % 0x20);
    uint8_t *w, *bs; CK(cudaMalloc(&w, wb)); CK(cudaMalloc(&bs, sb));
    CK(cudaMemcpy(w, hw.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(bs, hb.data(), sb, cudaMemcpyHostToDevice));
    std::vector<uint16_t> hx((size_t)M * K);
    for (auto& v : hx) { const float f = ((int)(rnd() % 4001) - 2000) / 1000.f; uint32_t u; memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
    ik_bf16* x; CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    float hg = 0.00071f; float* gs; CK(cudaMalloc(&gs, 4)); CK(cudaMemcpy(gs, &hg, 4, cudaMemcpyHostToDevice));
    const size_t ob = (size_t)M * N * 4;
    float *o_ref, *o; CK(cudaMalloc(&o_ref, ob)); CK(cudaMalloc(&o, ob));
    std::vector<uint8_t> h_ref(ob), h(ob);
    const size_t smem = (size_t)M * K * 2;
    const int S = sms();
    const double bytes = (double)wb + sb;
    auto run = [&](auto kern, int nt, int W, const char* name, bool ref) {
        CK(cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, W * 32, smem)); if (occ < 1) occ = 1;
        const int64_t blocks = ((N + nt - 1) / nt + W - 1) / W, cap = (int64_t)S * occ;
        const unsigned grid = (unsigned)(blocks < cap ? blocks : cap);
        CK(cudaMemset(o, 0, ob));
        kern<<<grid, W * 32, smem>>>(x, w, bs, gs, ref ? o_ref : o, N, K);
        CK(cudaDeviceSynchronize());
        bool same = true;
        if (ref) CK(cudaMemcpy(h_ref.data(), o_ref, ob, cudaMemcpyDeviceToHost));
        else { CK(cudaMemcpy(h.data(), o, ob, cudaMemcpyDeviceToHost)); same = !memcmp(h.data(), h_ref.data(), ob); }
        const float us = time_us(1, [&](int) { kern<<<grid, W * 32, smem>>>(x, w, bs, gs, o, N, K); });
        printf("NV  %6ldx%-5d %-34s grid %4u occ %d %8.2f us %6.1f GB/s %s\n", (long)N, K, name, grid, occ, us, bytes / us / 1e3,
               ref ? "" : (same ? "bitwise" : "DIFFERS"));
    };
    if (M >= 12) run(nv_base<4, 8>, 4, 8, "base 4x8 (main)", true);
    else run(nv_base<2, 16>, 2, 16, "base 2x16 (main)", true);
    run(nv_base<2, 16>, 2, 16, "base 2x16", false);
#define NVP(NT, W, KPL, CH, PF) run(nv_pipe<NT, W, KPL, CH, PF>, NT, W, "pipe " #NT "x" #W " ch" #CH " pf" #PF, false)
    NVP(2, 8, 4, 4, 0); NVP(2, 8, 4, 4, 2); NVP(2, 8, 4, 2, 0); NVP(1, 8, 4, 4, 0); NVP(1, 16, 4, 4, 0); NVP(1, 16, 4, 2, 0);
    NVP(3, 8, 4, 4, 0); NVP(2, 12, 4, 4, 0); NVP(1, 12, 4, 4, 0); NVP(4, 8, 4, 1, 0);
#undef NVP
    cudaFree(w); cudaFree(bs); cudaFree(x); cudaFree(gs); cudaFree(o); cudaFree(o_ref);
}

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "all";
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s, %d SMs, L2 %d MB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20);
    if (M == 16 && (what == "all" || what == "fp8")) {
        fp8_shape(8192, 2048); fp8_shape(4096, 2048); fp8_shape(12288, 2048); fp8_shape(2048, 4096);
    }
    if (what == "all" || what == "nv") nv_shape(248320, 2048);
    return 0;
}
