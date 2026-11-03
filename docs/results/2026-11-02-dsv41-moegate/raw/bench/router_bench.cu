// standalone router (f16 weights, K=5120) microbench: old dense_f16w_vec_kernel<R=16> vs tile candidates, bitwise check
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)
static inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
namespace {
constexpr int kWarps = 8;
template <typename Y> __device__ __forceinline__ void st(Y* y, size_t i, float v);
template <> __device__ __forceinline__ void st<float>(float* y, size_t i, float v) { y[i] = v; }
template <> __device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16* y, size_t i, float v) { y[i] = __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ void st<__half>(__half* y, size_t i, float v) { y[i] = __float2half_rn(v); }

template <typename X> __device__ __forceinline__ void ld8(const X* p, float (&v)[8]);
template <> __device__ __forceinline__ void ld8<float>(const float* p, float (&v)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
template <> __device__ __forceinline__ void ld8<__nv_bfloat16>(const __nv_bfloat16* p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(h[j]); v[2 * j] = f.x; v[2 * j + 1] = f.y; }
}
template <> __device__ __forceinline__ void ld8<__half>(const __half* p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) { const float2 f = __half22float2(h[j]); v[2 * j] = f.x; v[2 * j + 1] = f.y; }
}

// ---- OLD (verbatim from kernels/cuda/dsv41_dense.cu) ----
template <typename X, typename Y, int R, int WPO, int KO = 0, int UNR = 2>
__global__ void __launch_bounds__(kWarps * 32) old_kernel(const X* __restrict__ x, const __half* __restrict__ w, Y* __restrict__ y, int M, int N, int K) {
    __shared__ float part_sum[kWarps][R];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int part = warp % WPO;
    const int n = blockIdx.x * (kWarps / WPO) + warp / WPO;
    const int m0 = blockIdx.y * R;
    const int rn = min(R, M - m0);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    if (n < N) {
        const __half* wr = w + (size_t)n * K;
        const X* xr = x + (size_t)m0 * K;
        const int k8n = K >> 3;
#pragma unroll UNR
        for (int k8 = part * 32 + lane; k8 < k8n; k8 += 32 * WPO) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(wr) + k8);
            const __half2* h = reinterpret_cast<const __half2*>(&u);
            float wf[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float2 f = __half22float2(h[j]); wf[2 * j] = f.x; wf[2 * j + 1] = f.y; }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < rn) {
                    float xv[8];
                    if (KO & 1) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) xv[j] = 0.5f * (j + r);
                    } else {
                        ld8<X>(xr + (size_t)r * K + (size_t)k8 * 8, xv);
                    }
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] = (KO & 2) ? acc[r] + xv[j] : fmaf(xv[j], wf[j], acc[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    }
    if constexpr (WPO == 1) {
        if (n < N && lane < R && lane < rn) {
            float v = 0.f;
#pragma unroll
            for (int r = 0; r < R; ++r) if (r == lane) v = acc[r];
            st<Y>(y, (size_t)(m0 + lane) * N + n, v);
        }
    } else {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) part_sum[warp][r] = acc[r];
        }
        __syncthreads();
        if (part == 0 && n < N && lane < rn) {
            float v = part_sum[warp][lane];
#pragma unroll
            for (int p = 1; p < WPO; ++p) v += part_sum[warp + p][lane];
            st<Y>(y, (size_t)(m0 + lane) * N + n, v);
        }
    }
}
int wpo_for(int64_t N) { int wpo = 1; while (wpo < kWarps && cdiv(N * wpo, kWarps) < 96) wpo *= 2; return wpo; }

// ---- NEW: register micro-tile RM rows x RN outputs per warp, a lane is a stripe ----
// block = 8 warps = WPO parts x (8/WPO) output groups; a group covers RN outputs; block covers (8/WPO)*RN outputs x RM rows
template <typename X, typename Y, int RM, int RN, int WPO>
__global__ void __launch_bounds__(kWarps * 32) tile_kernel(const X* __restrict__ x, const __half* __restrict__ w, Y* __restrict__ y, int M, int N, int K) {
    constexpr int NG = kWarps / WPO;
    __shared__ float part_sum[kWarps][RM * RN];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int part = warp % WPO, grp = warp / WPO;
    const int n0 = (blockIdx.x * NG + grp) * RN;
    const int m0 = blockIdx.y * RM;
    const int rm = min(RM, M - m0);
    const int rnn = min(RN, N - n0);
    float acc[RM][RN];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) acc[r][c] = 0.f;
    if (rnn > 0) {
        const int k8n = K >> 3;
        for (int k8 = part * 32 + lane; k8 < k8n; k8 += 32 * WPO) {
            float wf[RN][8];
#pragma unroll
            for (int c = 0; c < RN; ++c) {
                if (c < rnn) {
                    const uint4 u = __ldg(reinterpret_cast<const uint4*>(w + (size_t)(n0 + c) * K) + k8);
                    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                    for (int j = 0; j < 4; ++j) { const float2 f = __half22float2(h[j]); wf[c][2 * j] = f.x; wf[c][2 * j + 1] = f.y; }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) wf[c][j] = 0.f;
                }
            }
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                if (r < rm) {
                    float xv[8];
                    ld8<X>(x + (size_t)(m0 + r) * K + (size_t)k8 * 8, xv);
#pragma unroll
                    for (int c = 0; c < RN; ++c)
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[r][c] = fmaf(xv[j], wf[c][j], acc[r][c]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[r][c] += __shfl_xor_sync(0xffffffffu, acc[r][c], o);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RM; ++r)
#pragma unroll
            for (int c = 0; c < RN; ++c) part_sum[warp][r * RN + c] = acc[r][c];
    }
    __syncthreads();
    if (part == 0 && lane < RM * RN) {
        const int r = lane / RN, c = lane % RN;
        if (r < rm && c < rnn) {
            float v = part_sum[warp][lane];
#pragma unroll
            for (int p = 1; p < WPO; ++p) v += part_sum[warp + p][lane];
            st<Y>(y, (size_t)(m0 + r) * N + n0 + c, v);
        }
    }
}
}  // namespace

template <typename X, typename Y, int R, int KO = 0, int UNR = 2>
void run_old(const X* x, const __half* w, Y* y, int M, int N, int K) {
    const int wpo = wpo_for(N);
    dim3 grid((unsigned)cdiv((int64_t)N * wpo, kWarps), (unsigned)cdiv(M, R));
    switch (wpo) {
        case 1: old_kernel<X, Y, R, 1, KO, UNR><<<grid, 256>>>(x, w, y, M, N, K); break;
        case 2: old_kernel<X, Y, R, 2, KO, UNR><<<grid, 256>>>(x, w, y, M, N, K); break;
        case 4: old_kernel<X, Y, R, 4, KO, UNR><<<grid, 256>>>(x, w, y, M, N, K); break;
        default: old_kernel<X, Y, R, 8, KO, UNR><<<grid, 256>>>(x, w, y, M, N, K); break;
    }
}
template <typename X, typename Y, int RM, int RN>
void run_tile(const X* x, const __half* w, Y* y, int M, int N, int K) {
    const int wpo = wpo_for(N);
    auto go = [&](auto wc) {
        constexpr int WPO = decltype(wc)::value;
        constexpr int NG = kWarps / WPO;
        dim3 grid((unsigned)cdiv(N, NG * RN), (unsigned)cdiv(M, RM));
        tile_kernel<X, Y, RM, RN, WPO><<<grid, 256>>>(x, w, y, M, N, K);
    };
    switch (wpo) {
        case 1: go(std::integral_constant<int, 1>{}); break;
        case 2: go(std::integral_constant<int, 2>{}); break;
        case 4: go(std::integral_constant<int, 4>{}); break;
        default: go(std::integral_constant<int, 8>{}); break;
    }
}

struct Cand { const char* name; void (*fn)(const __nv_bfloat16*, const __half*, float*, int, int, int); };
static int bench_one(int M, int N, int reps);
int main(int argc, char** argv) {
    int N = argc > 2 ? atoi(argv[2]) : 384;
    int reps = argc > 3 ? atoi(argv[3]) : 20;
    int first = argc > 1 ? atoi(argv[1]) : 2048;
    if (first == 0) { for (int M : {512, 1034, 2048}) { bench_one(M, N, reps); } } else bench_one(first, N, reps);
    if (argc > 4) { for (int M : {512, 2048}) bench_one(M, atoi(argv[4]), reps); }
    return 0;
}
static int bench_one(int M, int N, int reps) {
    const int K = 5120;
    printf("\n=== M=%d N=%d K=%d wpo=%d ===\n", M, N, K, wpo_for(N));
    std::vector<__nv_bfloat16> hx((size_t)M * K); std::vector<__half> hw((size_t)N * K);
    uint64_t s = 88172645463325252ull;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return ((s >> 11) * (1.0 / 9007199254740992.0)) - 0.5; };
    for (auto& v : hx) v = __float2bfloat16((float)rnd() * 4);
    for (auto& v : hw) v = __float2half((float)rnd() * 0.05f);
    __nv_bfloat16* dx; __half* dw; float *dy0, *dy1;
    CK(cudaMalloc(&dx, hx.size() * 2)); CK(cudaMalloc(&dw, hw.size() * 2));
    CK(cudaMalloc(&dy0, (size_t)M * N * 4)); CK(cudaMalloc(&dy1, (size_t)M * N * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    // reference = old R=1 (the decode kernel)
    std::vector<float> ref((size_t)M * N), got((size_t)M * N);
    run_old<__nv_bfloat16, float, 1>(dx, dw, dy0, M, N, K); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), dy0, ref.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<Cand> cands = {
        {"old R=16", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 16>(x, w, y, M, N, K); }},
        {"old R=2", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 2>(x, w, y, M, N, K); }},
        {"old R=4", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 4>(x, w, y, M, N, K); }},
        {"old R=6", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 6>(x, w, y, M, N, K); }},
        {"old R=8", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 8>(x, w, y, M, N, K); }},
        {"old R=10", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 10>(x, w, y, M, N, K); }},
        {"old R=12", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 12>(x, w, y, M, N, K); }},
        {"old R=8 unroll1", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 8, 0, 1>(x, w, y, M, N, K); }},
        {"old R=8 unroll4", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 8, 0, 4>(x, w, y, M, N, K); }},
        {"old R=4 unroll4", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 4, 0, 4>(x, w, y, M, N, K); }},
        {"old R=16 unroll1", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 16, 0, 1>(x, w, y, M, N, K); }},
        {"KO old R=8 no x loads", [](auto x, auto w, auto y, int M, int N, int K) { run_old<__nv_bfloat16, float, 8, 1>(x, w, y, M, N, K); }},
        {"tile 4x4", [](auto x, auto w, auto y, int M, int N, int K) { run_tile<__nv_bfloat16, float, 4, 4>(x, w, y, M, N, K); }},
        {"tile 4x8", [](auto x, auto w, auto y, int M, int N, int K) { run_tile<__nv_bfloat16, float, 4, 8>(x, w, y, M, N, K); }},
        {"tile 2x8", [](auto x, auto w, auto y, int M, int N, int K) { run_tile<__nv_bfloat16, float, 2, 8>(x, w, y, M, N, K); }},
    };

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<std::vector<float>> times(cands.size());
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        CK(cudaMemset(dy1, 0xff, (size_t)M * N * 4));
        cands[ci].fn(dx, dw, dy1, M, N, K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dy1, got.size() * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0; for (size_t i = 0; i < got.size(); ++i) if (memcmp(&got[i], &ref[i], 4) != 0) ++bad;
        if (!strstr(cands[ci].name, "KO")) printf("%-24s bitwise vs R=1: %s (%zu/%zu differ)\n", cands[ci].name, bad ? "NO" : "yes", bad, got.size());
    }
    for (int rep = 0; rep < reps; ++rep)
        for (size_t ci = 0; ci < cands.size(); ++ci) {
            cudaEventRecord(e0);
            cands[ci].fn(dx, dw, dy1, M, N, K);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); times[ci].push_back(ms);
        }
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        auto t = times[ci]; std::sort(t.begin(), t.end());
        printf("%-24s min %.3f ms  median %.3f ms  (%.2f TFLOP/s at median)\n", cands[ci].name, t[0], t[t.size() / 2], 2.0 * M * N * K / (t[t.size() / 2] * 1e-3) / 1e12);
    }
    cudaFree(dx); cudaFree(dw); cudaFree(dy0); cudaFree(dy1);
    return 0;
}
