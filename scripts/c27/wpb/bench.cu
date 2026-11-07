// c27: MXFP4 GEMV block-granularity microbench. Same per-warp code as kernels/cuda/mxfp4.cu (`mxfp4_gemv_kernel`, `mxfp4_gemv_mr_kernel`), warps per block WPB
// a template parameter (the engine uses 8): a block's duration is the unit of the launch tail and ramp, so a smaller block should shorten both.
//   nvcc -O3 -arch=sm_121 -I../../../kernels/cuda -o bench bench.cu      ./bench [reps]      NB=64 weight copies rotate (the engine streams 15 GB a token)
// Per 27B shape and M in {1, 3}: the production launcher (reference), then WPB in {1,2,4,8,16}: us, GB/s, bitwise equal to the reference.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../../kernels/cuda/mxfp4.cu"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
static ik_tensor T(void* p, int dt, int64_t a, int64_t b) { ik_tensor t; memset(&t, 0, sizeof t); t.data = p; t.dtype = dt; t.ndim = 2; t.shape[0] = a; t.shape[1] = b; t.stride[0] = b; t.stride[1] = 1; return t; }

// one-row baseline per warp (M = 1 production path: mxfp4_gemv_kernel with mc = 1), WPB warps per block
template <int WPB>
__global__ void __launch_bounds__(WPB * IK_WARP)
base_wpb(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, ik_bf16* __restrict__ out, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * WPB + warp;
    if (n >= N) return;
    const uint4* wr = reinterpret_cast<const uint4*>(w + n * (K / 2));
    const uint8_t* sr = sc + n * (K / kG);
    float acc = 0.f;
    for (int u = lane; u < K / kG; u += IK_WARP) {
        const uint4 wv = __ldcs(wr + u);
        const float s = mx_e8m0(sr[u]);
        float wf[kG];
        ik_fp4x8_to_f32(wv.x, wf); ik_fp4x8_to_f32(wv.y, wf + 8); ik_fp4x8_to_f32(wv.z, wf + 16); ik_fp4x8_to_f32(wv.w, wf + 24);
        const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)u * kG);
        float xf[kG];
        ik_unpack8(__ldg(xp), xf); ik_unpack8(__ldg(xp + 1), xf + 8); ik_unpack8(__ldg(xp + 2), xf + 16); ik_unpack8(__ldg(xp + 3), xf + 24);
        float t = 0.f;
#pragma unroll
        for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
        acc = fmaf(t, s, acc);
    }
    const float t = ik_warp_sum(acc);
    if (lane == 0) ik_store(out + n, t);
}

template <int MT, int R, int WPB>
__global__ void __launch_bounds__(WPB * IK_WARP)
mr_wpb(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, ik_bf16* __restrict__ out, int64_t N, int K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = ((int64_t)blockIdx.x * WPB + warp) * R;
    if (n0 >= N) return;
    const uint4* wr[R]; const uint8_t* sr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { const int64_t nr = n0 + r < N ? n0 + r : N - 1; wr[r] = reinterpret_cast<const uint4*>(w + nr * (K / 2)); sr[r] = sc + nr * (K / kG); }
    float acc[R][MT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;
    for (int u = lane; u < K / kG; u += IK_WARP) {
        uint4 wv[R]; uint32_t sb[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { wv[r] = __ldcs(wr[r] + u); sb[r] = sr[r][u]; }
        uint4 xv[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) { const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * kG);
#pragma unroll
            for (int j = 0; j < 4; ++j) xv[m][j] = __ldg(xp + j); }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s = mx_e8m0(sb[r]);
            float wf[kG];
            ik_fp4x8_to_f32(wv[r].x, wf); ik_fp4x8_to_f32(wv[r].y, wf + 8); ik_fp4x8_to_f32(wv[r].z, wf + 16); ik_fp4x8_to_f32(wv[r].w, wf + 24);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                float xf[kG];
                ik_unpack8(xv[m][0], xf); ik_unpack8(xv[m][1], xf + 8); ik_unpack8(xv[m][2], xf + 16); ik_unpack8(xv[m][3], xf + 24);
                float t = 0.f;
#pragma unroll
                for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
                acc[r][m] = fmaf(t, s, acc[r][m]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) { const float t = ik_warp_sum(acc[r][m]); if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, t); }
}

template <int WPB> static void run(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, ik_bf16* out, int64_t N, int K, cudaStream_t s) {
    if (M == 1) { base_wpb<WPB><<<(unsigned)((N + WPB - 1) / WPB), WPB * IK_WARP, 0, s>>>(x, w, sc, out, N, K); return; }
    const unsigned g = (unsigned)(((N + 1) / 2 + WPB - 1) / WPB);
    if (M == 2) mr_wpb<2, 2, WPB><<<g, WPB * IK_WARP, 0, s>>>(x, w, sc, out, N, K);
    else if (M == 3) mr_wpb<3, 2, WPB><<<g, WPB * IK_WARP, 0, s>>>(x, w, sc, out, N, K);
    else { const unsigned g4 = (unsigned)(((N + 3) / 4 + WPB - 1) / WPB); mr_wpb<4, 4, WPB><<<g4, WPB * IK_WARP, 0, s>>>(x, w, sc, out, N, K); }
}
static void run_wpb(int wpb, int M, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, ik_bf16* out, int64_t N, int K, cudaStream_t s) {
    switch (wpb) { case 1: run<1>(M, x, w, sc, out, N, K, s); break; case 2: run<2>(M, x, w, sc, out, N, K, s); break; case 4: run<4>(M, x, w, sc, out, N, K, s); break;
                   case 8: run<8>(M, x, w, sc, out, N, K, s); break; default: run<16>(M, x, w, sc, out, N, K, s); }
}
int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 60;
    const int NB = getenv("NB") ? atoi(getenv("NB")) : 64;
    struct Sh { const char* name; int64_t N; int K; } shapes[] = {{"gate/up", 17408, 5120}, {"down", 5120, 17408}, {"out/o", 5120, 6144}, {"qkv", 10240, 5120}, {"z", 6144, 5120}, {"q", 12288, 5120}, {"k/v", 1024, 5120}};
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float* act; CK(cudaMalloc(&act, 4)); CK(cudaMemset(act, 0, 4));
    int allok = 1;
    for (auto sh : shapes) {
        std::vector<uint8_t*> w(NB), sc(NB);
        size_t wb = (size_t)sh.N * sh.K / 2, sb = (size_t)sh.N * sh.K / 32;
        std::vector<uint8_t> hw(wb), hs(sb);
        for (size_t i = 0; i < wb; ++i) hw[i] = (uint8_t)(rand() & 0xff);
        for (size_t i = 0; i < sb; ++i) hs[i] = (uint8_t)(120 + rand() % 8);
        for (int b = 0; b < NB; ++b) { CK(cudaMalloc(&w[b], wb)); CK(cudaMalloc(&sc[b], sb)); CK(cudaMemcpy(w[b], hw.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sc[b], hs.data(), sb, cudaMemcpyHostToDevice)); }
        for (int M : {1, 3}) {
            std::vector<ik_bf16> hx((size_t)M * sh.K); for (auto& v : hx) v = __float2bfloat16((rand() % 2001 - 1000) / 500.f);
            ik_bf16 *x, *o0, *o1; CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&o0, (size_t)M * sh.N * 2)); CK(cudaMalloc(&o1, (size_t)M * sh.N * 2));
            auto xt = T(x, IK_DTYPE_BF16, M, sh.K); auto at = T(act, IK_DTYPE_F32, 1, 1); at.ndim = 1; at.shape[0] = 1;
            auto timeit = [&](auto fn) {
                for (int i = 0; i < 5; ++i) fn(i % NB);
                float best = 1e30f;
                for (int blk = 0; blk < 5; ++blk) { CK(cudaEventRecord(e0, s)); for (int i = 0; i < reps; ++i) fn(i % NB); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = ms * 1000.f / reps < best ? ms * 1000.f / reps : best; }
                return best;
            };
            auto prod = [&](int b) { auto wt = T(w[b], IK_DTYPE_U8, sh.N, sh.K / 2); auto st = T(sc[b], IK_DTYPE_U8, sh.N, sh.K / 32); auto ot = T(o0, IK_DTYPE_BF16, M, sh.N);
                if (M == 1) ik_matmul_mxfp4_gemv(&xt, &wt, &st, &at, &ot, s); else ik_matmul_mxfp4_gemv_mr(&xt, &wt, &st, &at, &ot, s); };
            double bytes = (double)wb + sb;
            float tp = timeit(prod);
            CK(cudaMemset(o0, 0, (size_t)M * sh.N * 2)); prod(0); CK(cudaDeviceSynchronize());
            std::vector<ik_bf16> ref((size_t)M * sh.N); CK(cudaMemcpy(ref.data(), o0, ref.size() * 2, cudaMemcpyDeviceToHost));
            printf("%-8s M=%d prod %7.1f us %6.1f GB/s |", sh.name, M, tp, bytes / tp / 1e3);
            for (int wpb : {1, 2, 4, 8, 16}) {
                float t = timeit([&](int b) { run_wpb(wpb, M, x, w[b], sc[b], o1, sh.N, sh.K, s); });
                CK(cudaMemset(o1, 0, (size_t)M * sh.N * 2)); run_wpb(wpb, M, x, w[0], sc[0], o1, sh.N, sh.K, s); CK(cudaDeviceSynchronize());
                std::vector<ik_bf16> got((size_t)M * sh.N); CK(cudaMemcpy(got.data(), o1, got.size() * 2, cudaMemcpyDeviceToHost));
                int eq = memcmp(got.data(), ref.data(), got.size() * 2) == 0; allok &= eq;
                printf(" w%-2d %6.1f (%+5.1f%%)%s", wpb, t, 100.f * (tp - t) / tp, eq ? "" : " DIFF");
            }
            printf("\n"); fflush(stdout);
            cudaFree(x); cudaFree(o0); cudaFree(o1);
        }
        for (int b = 0; b < NB; ++b) { cudaFree(w[b]); cudaFree(sc[b]); }
    }
    printf("bitwise equal everywhere: %s\n", allok ? "yes" : "NO");
    return 0;
}
