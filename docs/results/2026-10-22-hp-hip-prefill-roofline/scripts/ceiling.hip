// Lane hp (docs/results/2026-10-22-hp-hip-prefill-roofline.md): the measured matrix ceilings of one R9700 (gfx1201).
//   1. WMMA issue peak: every wave runs independent v_wmma_f32_16x16x16_{bf16,fp8_fp8} on register operands (no memory
//      traffic), enough waves to fill every SIMD. This is the instruction-throughput ceiling at the clock the card holds
//      under that load, not the spec sheet's.
//   2. hipBLASLt (the vendor GEMM, loaded with dlopen from the ROCm 7.2.4 image that has it): bf16 x bf16 and
//      e4m3 x e4m3 -> bf16 GEMMs at the 35B's prefill shapes; every heuristic algo is timed, the best is reported.
//   3. Our matmul_fp8_mma_f8 kernel (kernels/hip/ik_f8_gemm.h) at the same shapes, isolated.
// Build (dev image): hipcc -O3 --offload-arch=gfx1201 -I kernels/hip ceiling.hip kernels/hip/ik_runtime.hip -ldl -o ceiling
// Run: ceiling [libhipblaslt.so path]   (HIP_VISIBLE_DEVICES pins the card)
#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "ik_f8_gemm.h"

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));

constexpr int kAcc = 8;  // independent accumulators per wave (WMMA latency hiding)

template <bool FP8>
__global__ void __launch_bounds__(256) wmma_peak(const uint4* in, float* out, int iters) {
    const int lane = threadIdx.x & 31;
    uint4 a = in[lane], b = in[32 + lane];
    v8f acc[kAcc];
#pragma unroll
    for (int i = 0; i < kAcc; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < kAcc; ++i) {
            if constexpr (FP8)
                acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    __builtin_bit_cast(v2i, make_uint2(a.x, a.y)), __builtin_bit_cast(v2i, make_uint2(b.x, b.y)), acc[i]);
            else
                acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(v8bf, a),
                                                                          __builtin_bit_cast(v8bf, b), acc[i]);
        }
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < kAcc; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) s += acc[i][e];
    if (s == 12345.f) out[threadIdx.x] = s;  // never true for the inputs used; keeps the loop alive
}

template <bool FP8>
double peak(int cus) {
    uint4* in;
    float* out;
    CK(hipMalloc(&in, 64 * sizeof(uint4)));
    CK(hipMalloc(&out, 256 * sizeof(float)));
    std::vector<uint32_t> h(256, FP8 ? 0x30303030u : 0x3c003c00u);  // small finite values
    CK(hipMemcpy(in, h.data(), 64 * sizeof(uint4), hipMemcpyHostToDevice));
    const int iters = 4096, blocks = cus * 16;  // 8 waves per block, 16 blocks per CU
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    wmma_peak<FP8><<<blocks, 256>>>(in, out, 64);
    CK(hipDeviceSynchronize());
    double best = 0;
    for (int r = 0; r < 5; ++r) {
        CK(hipEventRecord(e0));
        wmma_peak<FP8><<<blocks, 256>>>(in, out, iters);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms;
        CK(hipEventElapsedTime(&ms, e0, e1));
        const double flops = (double)blocks * 8 * iters * kAcc * 16 * 16 * 16 * 2;
        best = fmax(best, flops / (ms * 1e-3) / 1e12);
    }
    CK(hipFree(in));
    CK(hipFree(out));
    return best;
}

// ---- hipBLASLt through dlopen (the image that runs this has the library but no headers) ----
typedef void* LtHandle;
typedef void* LtDesc;
typedef void* LtLayout;
typedef void* LtPref;
struct LtAlgo {
    uint8_t data[16];
    size_t max_workspace_bytes;
};
struct LtHeur {
    LtAlgo algo;
    size_t workspaceSize;
    int state;
    float wavesCount;
    int reserved[4];
};
constexpr int OP_N = 111, OP_T = 112;  // hipblasOperation_t
constexpr int COMPUTE_32F = 2;         // hipblasComputeType_t HIPBLAS_COMPUTE_32F
constexpr int DESC_TRANSA = 0, DESC_TRANSB = 1, PREF_MAX_WS = 1;

struct Lt {
    int (*create)(LtHandle*);
    int (*descCreate)(LtDesc*, int, hipDataType);
    int (*descSet)(LtDesc, int, const void*, size_t);
    int (*layoutCreate)(LtLayout*, hipDataType, uint64_t, uint64_t, int64_t);
    int (*prefCreate)(LtPref*);
    int (*prefSet)(LtPref, int, const void*, size_t);
    int (*heur)(LtHandle, LtDesc, LtLayout, LtLayout, LtLayout, LtLayout, LtPref, int, LtHeur*, int*);
    int (*matmul)(LtHandle, LtDesc, const void*, const void*, LtLayout, const void*, LtLayout, const void*, const void*,
                  LtLayout, void*, LtLayout, const LtAlgo*, void*, size_t, hipStream_t);
    LtHandle h = nullptr;
    bool ok = false;
    bool load(const char* path) {
        void* so = dlopen(path, RTLD_NOW);
        if (!so) {
            fprintf(stderr, "hipBLASLt: %s\n", dlerror());
            return false;
        }
#define SYM(f, n)                                                 \
    f = reinterpret_cast<decltype(f)>(dlsym(so, n));              \
    if (!f) {                                                     \
        fprintf(stderr, "hipBLASLt: no symbol %s\n", n);          \
        return false;                                             \
    }
        SYM(create, "hipblasLtCreate");
        SYM(descCreate, "hipblasLtMatmulDescCreate");
        SYM(descSet, "hipblasLtMatmulDescSetAttribute");
        SYM(layoutCreate, "hipblasLtMatrixLayoutCreate");
        SYM(prefCreate, "hipblasLtMatmulPreferenceCreate");
        SYM(prefSet, "hipblasLtMatmulPreferenceSetAttribute");
        SYM(heur, "hipblasLtMatmulAlgoGetHeuristic");
        SYM(matmul, "hipblasLtMatmul");
#undef SYM
        int st = create(&h);
        if (st) fprintf(stderr, "hipblasLtCreate status %d\n", st);
        ok = st == 0;
        return ok;
    }
};

// Y[M, N] = X[M, K] W[N, K]^T as the column-major GEMM C(N x M) = op_T(W: K x N) * X(K x M): weights A (T), tokens B (N).
double lt_gemm(Lt& lt, bool fp8, int M, int N, int K, void* w, void* x, void* y, void* ws, size_t wsb, int* nalgo) {
    *nalgo = 0;
    const hipDataType ab = fp8 ? HIP_R_8F_E4M3 : HIP_R_16BF;
    LtDesc d;
    LtLayout la, lb, lc;
    LtPref p;
    int st;
    if ((st = lt.descCreate(&d, COMPUTE_32F, HIP_R_32F))) { fprintf(stderr, "descCreate %d\n", st); return -1; }
    const int ta = OP_T, tb = OP_N;
    lt.descSet(d, DESC_TRANSA, &ta, sizeof ta);
    lt.descSet(d, DESC_TRANSB, &tb, sizeof tb);
    lt.layoutCreate(&la, ab, K, N, K);
    lt.layoutCreate(&lb, ab, K, M, K);
    lt.layoutCreate(&lc, HIP_R_16BF, N, M, N);
    lt.prefCreate(&p);
    const uint64_t wsz = wsb;
    lt.prefSet(p, PREF_MAX_WS, &wsz, sizeof wsz);
    LtHeur res[16];
    memset(res, 0, sizeof res);
    int cnt = 0;
    st = lt.heur(lt.h, d, la, lb, lc, lc, p, 16, res, &cnt);
    if (st || cnt == 0) {
        fprintf(stderr, "heuristic status %d, %d algos (%s M=%d N=%d K=%d)\n", st, cnt, fp8 ? "fp8" : "bf16", M, N, K);
        return -1;
    }
    *nalgo = cnt;
    const float alpha = 1.f, beta = 0.f;
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    double best = 0;
    for (int a = 0; a < cnt; ++a) {
        if (res[a].workspaceSize > wsb) continue;
        if (lt.matmul(lt.h, d, &alpha, w, la, x, lb, &beta, y, lc, y, lc, &res[a].algo, ws, wsb, 0)) continue;
        CK(hipDeviceSynchronize());
        const int reps = 20;
        CK(hipEventRecord(e0));
        for (int r = 0; r < reps; ++r) lt.matmul(lt.h, d, &alpha, w, la, x, lb, &beta, y, lc, y, lc, &res[a].algo, ws, wsb, 0);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms;
        CK(hipEventElapsedTime(&ms, e0, e1));
        best = fmax(best, 2.0 * M * N * K * reps / (ms * 1e-3) / 1e12);
    }
    return best;
}

double ours_f8(int M, int N, int K, uint8_t* w, uint8_t* x, float* sc, ik_bf16* y) {
    f8_gemm_run<ik_bf16>(x, w, sc, y, M, N, K, 1, 0);
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    const int reps = 20;
    CK(hipEventRecord(e0));
    for (int r = 0; r < reps; ++r) f8_gemm_run<ik_bf16>(x, w, sc, y, M, N, K, 1, 0);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    return 2.0 * M * N * K * reps / (ms * 1e-3) / 1e12;
}

int main(int argc, char** argv) {
    hipDeviceProp_t pr;
    CK(hipGetDeviceProperties(&pr, 0));
    printf("device %s, %d CUs, clock %d MHz (prop)\n", pr.name, pr.multiProcessorCount, pr.clockRate / 1000);
    printf("WMMA issue peak bf16 %.1f TFLOPS\n", peak<false>(pr.multiProcessorCount));
    printf("WMMA issue peak fp8  %.1f TFLOPS\n", peak<true>(pr.multiProcessorCount));

    // M tokens x N out x K in: the 35B's dense fp8 projections at a 4096-token chunk, and the routed experts as if every
    // (token, expert) pair of a 4096-token chunk were one dense GEMM (the grouped-GEMM ideal: 4096 * 8 pairs)
    struct S { const char* what; int M, N, K; } shapes[] = {
        {"qkv / q+gate", 4096, 8192, 2048}, {"z", 4096, 4096, 2048},       {"o / out_proj", 4096, 2048, 4096},
        {"k / v", 4096, 512, 2048},         {"experts gate_up (dense ideal)", 32768, 1024, 2048},
        {"experts down (dense ideal)", 32768, 2048, 512}, {"shared expert gate_up", 4096, 1024, 2048},
        {"shared expert down", 4096, 2048, 512},
    };
    size_t maxw = 0, maxx = 0, maxy = 0;
    for (auto& s : shapes) {
        maxw = std::max(maxw, (size_t)s.N * s.K * 2);
        maxx = std::max(maxx, (size_t)s.M * s.K * 2);
        maxy = std::max(maxy, (size_t)s.M * s.N * 2);
    }
    void *w, *x, *y, *ws;
    float* sc;
    const size_t wsb = 64 << 20;
    CK(hipMalloc(&w, maxw));
    CK(hipMalloc(&x, maxx));
    CK(hipMalloc(&y, maxy));
    CK(hipMalloc(&ws, wsb));
    CK(hipMalloc(&sc, 2 * sizeof(float)));
    {  // inputs: small finite values in either format (bytes 0x30 are e4m3 0.125; bf16 words 0x3c30 ~ 0.0107)
        std::vector<uint8_t> h(std::max(maxw, maxx), 0x30);
        CK(hipMemcpy(w, h.data(), maxw, hipMemcpyHostToDevice));
        CK(hipMemcpy(x, h.data(), maxx, hipMemcpyHostToDevice));
        float s2[2] = {1.f, 1.f};
        CK(hipMemcpy(sc, s2, sizeof s2, hipMemcpyHostToDevice));
    }
    Lt lt;
    const char* path = argc > 1 ? argv[1] : "libhipblaslt.so.1";
    lt.load(path);
    printf("\n%-32s %6s %6s %6s %12s %12s %12s\n", "shape", "M", "N", "K", "hipBLASLt bf16", "hipBLASLt fp8", "ours f8");
    for (auto& s : shapes) {
        int n1 = 0, n2 = 0;
        const double b = lt.ok ? lt_gemm(lt, false, s.M, s.N, s.K, w, x, y, ws, wsb, &n1) : -1;
        const double f = lt.ok ? lt_gemm(lt, true, s.M, s.N, s.K, w, x, y, ws, wsb, &n2) : -1;
        const double o = ours_f8(s.M, s.N, s.K, (uint8_t*)w, (uint8_t*)x, sc, (ik_bf16*)y);
        printf("%-32s %6d %6d %6d %8.1f (%2d) %8.1f (%2d) %12.1f\n", s.what, s.M, s.N, s.K, b, n1, f, n2, o);
    }
    printf("(TFLOPS = 2 M N K / time; hipBLASLt: best of the heuristic's algos (count in parentheses), 20 reps each)\n");
    return 0;
}
