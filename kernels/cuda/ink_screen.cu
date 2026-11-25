// Screened greedy head on CUDA (lane ink-cuda-attn, brief 54): HeadScreen and ScreenArgmax. Contract and proof:
// crates/intarsia-core/src/screen.rs; cpu reference crates/intarsia-backend-cpu/src/screen_ref.rs (the core's screen_row /
// argmax_row); the Metal twins kernels/metal/screen.metal. LogitsGather is the P12 head kernel's index mode
// (inkling2_p12.cu, ik_logits_gather_p12): it must share the head's arithmetic.
//
// HeadScreen uses exactly the contract's f32 operations (explicit fmaf, the 32-lane strided sum of squares and its xor tree,
// correctly rounded sqrtf: nvcc's default -prec-sqrt=true, no fast math), so its candidates and counts are bitwise the cpu
// reference's. One 1024-thread block per activation row.
//
// Why the selection bound covers CUDA's screen values `a` (the fp8 GEMV the model runs before HeadScreen): the bound needs
// a_n to be a tree of correctly rounded f32 adds / fmas of the D exact products h_k * q_nk, of depth <= D + 1, times one
// scale multiply (screen.rs, "Accumulation order"). CUDA's `matmul_fp8_gemv[_rows]` (fp8.cu fp8_gemv_body /
// fp8_gemv_rows_kernel; the model checks the screen GEMV is one of these two, `screened_argmax`) is exactly that:
// - e4m3 decodes exactly (cvt e4m3 -> half -> float, `ik_fp8x16_to_f32`), bf16 activations are staged unchanged
//   (W8A16: the screen copy's input scale is 0, so no requantisation), and an e4m3 x bf16 product (4 + 8 significant bits)
//   is exact in f32;
// - each lane runs 16-term fmaf chains, adds each chain into its accumulator (`acc += s`, K / 512 adds), then the 5-level
//   warp tree: correctly rounded f32 operations (IEEE fma / add; no fast math, no tensor cores), depth 16 + K / 512 + 5,
//   far below D + 1;
// - one final multiply by the row scale s (covered by the +2 of gamma_{D+2}).
// The tensor-core forms (`matmul_fp8_mma`, cuBLAS) do not accumulate with correctly rounded adds and are not accepted.
#include "ik_launch.h"
#include "ik_internal.h"
#include "ik_cuda_abi_inkp12.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr float kWiden = 9.5367431640625e-7f;            // 2^-20
constexpr float kHnMargin = 1.00048828125f;              // 1 + 2^-11
constexpr int kNonFinite = 0x7fffffff;
constexpr float kHnFloor = 8.881784197001252e-16f;       // 2^-50
constexpr float kTAbs = 1.262177448353619e-29f;          // 2^-96
constexpr float kMMax = 1.2379400392853803e27f;          // 2^90

// INTARSIA_INK_HEAD_SCREEN_STATS=1 (diagnostic): per screened row, [rows, fallbacks (cnt > C or non-finite), sum of cnt
// on the candidate path, largest cnt], counted on the device (graph replays included) and printed at process exit
__device__ unsigned long long g_stats[4];

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

__device__ __forceinline__ float2 lohi(float a, float hn, float b) {
    const float m = fmaf(hn, b, fabsf(a));
    const float lo = __fsub_rn(fmaf(-kWiden, m, fmaf(-hn, b, a)), kTAbs);
    const float hi = __fadd_rn(fmaf(kWiden, m, fmaf(hn, b, a)), kTAbs);
    return make_float2(lo, hi);
}

__global__ void __launch_bounds__(kThreads) head_screen_kernel(const float* __restrict__ A, const uint16_t* __restrict__ H,
                                                               const float* __restrict__ beta, int* __restrict__ cand,
                                                               int* __restrict__ cnt, int V, int D, int C, int stats) {
    __shared__ float s_hn;
    __shared__ float s_part[kWarps];
    __shared__ unsigned s_bad[kWarps];
    __shared__ unsigned s_cnt[kWarps];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const float* a = A + (int64_t)t * V;
    const uint16_t* h = H + (int64_t)t * D;
    if (wp == 0) {
        float q = 0.0f;
        for (int k = lane; k < D; k += 32) {
            const float x = bf2f(h[k]);
            q = fmaf(x, x, q);
        }
        // the xor tree p_l += p_{l^16}, ^8, ^4, ^2, ^1 (addition commutes: every lane holds the cpu's lane-0 value)
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) q = __fadd_rn(q, __shfl_xor_sync(0xffffffffu, q, off));
        if (lane == 0) s_hn = fmaf(__fsqrt_rn(q), kHnMargin, kHnFloor);
    }
    __syncthreads();
    const float hn = s_hn;
    // pass 1: L = max lo, and any non-finite screen value (or m past 2^90; NaN fails the <=)
    float lmax = -INFINITY;
    unsigned bad = isfinite(hn) ? 0u : 1u;
    for (int n = tid; n < V; n += kThreads) {
        const float an = a[n], bn = beta[n];
        bad |= (isfinite(an) && fmaf(hn, bn, fabsf(an)) <= kMMax) ? 0u : 1u;
        lmax = fmaxf(lmax, lohi(an, hn, bn).x);
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, off));
        bad |= __shfl_xor_sync(0xffffffffu, bad, off);
    }
    if (lane == 0) {
        s_part[wp] = lmax;
        s_bad[wp] = bad;
    }
    __syncthreads();
    float L = -INFINITY;
    unsigned anybad = 0;
    for (int j = 0; j < kWarps; ++j) {
        L = fmaxf(L, s_part[j]);
        anybad |= s_bad[j];
    }
    int* cr = cand + (int64_t)t * C;
    if (anybad != 0 || !isfinite(L)) {
        for (int j = tid; j < C; j += kThreads) cr[j] = -1;
        if (tid == 0) {
            cnt[t] = kNonFinite;
            if (stats) {
                atomicAdd(&g_stats[0], 1ull);
                atomicAdd(&g_stats[1], 1ull);
            }
        }
        return;
    }
    // pass 2: count this thread's contiguous chunk, prefix over the threads, write in ascending n
    const int chunk = (V + kThreads - 1) / kThreads;
    const int n0 = min(tid * chunk, V), n1 = min(n0 + chunk, V);
    unsigned mine = 0;
    for (int n = n0; n < n1; ++n) mine += lohi(a[n], hn, beta[n]).y >= L ? 1u : 0u;
    unsigned inc = mine;  // inclusive warp scan
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const unsigned v = __shfl_up_sync(0xffffffffu, inc, off);
        if (lane >= off) inc += v;
    }
    const unsigned pre = inc - mine, tot = __shfl_sync(0xffffffffu, inc, 31);
    if (lane == 0) s_cnt[wp] = tot;
    __syncthreads();
    unsigned base = 0, all = 0;
    for (int j = 0; j < kWarps; ++j) {
        if (j < wp) base += s_cnt[j];
        all += s_cnt[j];
    }
    for (int64_t j = (int64_t)all + tid; j < C; j += kThreads) cr[j] = -1;  // slots past the candidates (contract: -1)
    unsigned pos = base + pre;
    if (mine != 0 && pos < (unsigned)C) {
        for (int n = n0; n < n1 && pos < (unsigned)C; ++n) {
            if (lohi(a[n], hn, beta[n]).y >= L) cr[pos++] = n;
        }
    }
    if (tid == 0) {
        cnt[t] = (int)min(all, (unsigned)(kNonFinite - 1));
        if (stats) {
            atomicAdd(&g_stats[0], 1ull);
            if (all > (unsigned)C) atomicAdd(&g_stats[1], 1ull);
            else atomicAdd(&g_stats[2], (unsigned long long)all);
            atomicMax(&g_stats[3], (unsigned long long)all);
        }
    }
}

// ScreenArgmax: the lowest-index maximum over the candidates (cand ascending, so the lowest j) or over the full row, with
// argmax's comparisons (strict >, +-0 equal, NaN never wins). One 1024-thread block per row.
__global__ void __launch_bounds__(kThreads) screen_argmax_kernel(const int* __restrict__ cand, const int* __restrict__ cnt,
                                                                 const float* __restrict__ Y, int* __restrict__ tok, int V,
                                                                 int C) {
    __shared__ float pv[kWarps];
    __shared__ unsigned pi[kWarps];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const float* y = Y + (int64_t)t * V;
    const int* cr = cand + (int64_t)t * C;
    const int ct = cnt[t];
    const bool path = ct >= 1 && ct <= C;
    const int neff = path ? ct : V;
    float bv = -INFINITY;
    unsigned bi = 0xffffffffu;
    for (int j = tid; j < neff; j += kThreads) {
        const float x = y[j];
        if (x > bv) {
            bv = x;
            bi = path ? (unsigned)cr[j] : (unsigned)j;
        }
    }
    float m = bv;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    unsigned mi = bv == m ? bi : 0xffffffffu;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) mi = min(mi, __shfl_xor_sync(0xffffffffu, mi, off));
    if (lane == 0) {
        pv[wp] = m;
        pi[wp] = mi;
    }
    __syncthreads();
    if (tid == 0) {
        float gm = -INFINITY;
        unsigned gi = 0xffffffffu;
        for (int j = 0; j < kWarps; ++j) {
            if (pv[j] > gm || (pv[j] == gm && pi[j] < gi)) {
                gm = pv[j];
                gi = pi[j];
            }
        }
        // all -inf / NaN: the cpu's cand[0] (path) or 0 (full row)
        tok[t] = gi == 0xffffffffu ? (path ? cr[0] : 0) : (int)gi;
    }
}

bool stats_on() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("INTARSIA_INK_HEAD_SCREEN_STATS");
        on = e && *e && strcmp(e, "0") != 0;
        if (on) atexit([] {
            unsigned long long v[4] = {0, 0, 0, 0};
            if (cudaMemcpyFromSymbol(v, g_stats, sizeof v) == cudaSuccess)
                fprintf(stderr, "INTARSIA_INK_HEAD_SCREEN_STATS cuda: %llu screened rows, %llu fallbacks, mean candidates %.2f (path), max %llu\n",
                        v[0], v[1], v[0] > v[1] ? (double)v[2] / (double)(v[0] - v[1]) : 0.0, v[3]);
        });
    }
    return on == 1;
}

}  // namespace

extern "C" {

// a [T, V] f32, h [T, D] bf16, beta [V] f32 -> cand [T, C] i32, cnt [T] i32
int ik_head_screen_f32(const ik_tensor* a, const ik_tensor* h, const ik_tensor* beta, ik_tensor* cand, ik_tensor* cnt, void* stream) {
    IK_TRY(ik_check(a, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(h, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(beta, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(cand, IK_DTYPE_I32, 2));
    IK_TRY(ik_check(cnt, IK_DTYPE_I32, 1));
    const int64_t T = a->shape[0], V = a->shape[1], D = h->shape[1], C = cand->shape[1];
    if (h->shape[0] != T || beta->shape[0] != V || cand->shape[0] != T || cnt->shape[0] != T || C < 1) return IK_ERR_SHAPE;
    if (V > 0x7fffffff || D > 0x7fffffff || C > 0x7fffffff || T > 0x7fffffff) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    head_screen_kernel<<<(unsigned)T, kThreads, 0, IK_STREAM(stream)>>>((const float*)a->data, (const uint16_t*)h->data,
                                                                         (const float*)beta->data, (int*)cand->data,
                                                                         (int*)cnt->data, (int)V, (int)D, (int)C, stats_on() ? 1 : 0);
    return ik_launch_status();
}

// cand [T, C] i32, cnt [T] i32, y [T, V] f32 -> tok [T] i32
int ik_screen_argmax_f32(const ik_tensor* cand, const ik_tensor* cnt, const ik_tensor* y, ik_tensor* tok, void* stream) {
    IK_TRY(ik_check(cand, IK_DTYPE_I32, 2));
    IK_TRY(ik_check(cnt, IK_DTYPE_I32, 1));
    IK_TRY(ik_check(y, IK_DTYPE_F32, 2));
    IK_TRY(ik_check(tok, IK_DTYPE_I32, 1));
    const int64_t T = cand->shape[0], C = cand->shape[1], V = y->shape[1];
    if (cnt->shape[0] != T || y->shape[0] != T || tok->shape[0] != T || C < 1) return IK_ERR_SHAPE;
    if (V > 0x7fffffff || C > 0x7fffffff || T > 0x7fffffff) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    screen_argmax_kernel<<<(unsigned)T, kThreads, 0, IK_STREAM(stream)>>>((const int*)cand->data, (const int*)cnt->data,
                                                                           (const float*)y->data, (int*)tok->data, (int)V, (int)C);
    return ik_launch_status();
}

}  // extern "C"
