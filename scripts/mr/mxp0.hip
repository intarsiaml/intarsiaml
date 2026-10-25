// Lane mr, P0 ablation for docs/plans/2026-10-25-mr-hip-mxfp4-rows2-scope.md. Standalone harness, no production code touched.
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 -I kernels/hip scripts/mr/mxp0.hip -o mxp0
//   ./mxp0 N K [iters] [rounds]
// Variants of mxfp4_gemv_rows_kernel (R = 2 weight rows per warp, x already pre-passed, W4A16 so x is plain bf16):
//   prod  the production kernel, as shipped (reference for the template below)
//   0     the same code as a template (check: equals prod in time)
//   1     (A)  x pre-converted to f32: no bf16 unpack, twice the x bytes
//   2     (b)  bf16 x, but every loop trip reads the same group (L0 hits; keeps the load instructions and the unpack)
//   3     (c)  f32 x and the same group every trip
//   4     (d)  ALU floor: x row 0 held in registers for all rows, no x loads, no unpack (values differ; time only)
//   5     (B)  bf16 x, two activation rows per barrier group (four independent fma chains, may pair into VOPD)
//   6     (A+B) f32 x and two activation rows per barrier group
// Variants 0, 1, 5, 6 keep every row's arithmetic, so they must be bitwise equal to the M = 1 launch on that row.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include "../../kernels/hip/mxfp4.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

template <int V, int M>
__global__ void __launch_bounds__(kWarps * IK_WARP)
p0_kernel(const void* __restrict__ xv, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, float* __restrict__ out,
          int64_t N, int K) {
    constexpr int R = kRowsR;
    constexpr bool XF = (V == 1 || V == 3 || V == 6);
    constexpr bool FIX = (V == 2 || V == 3);
    constexpr bool ALU = (V == 4);
    constexpr int MB = (V == 5 || V == 6) ? 2 : 1;
    static_assert(M % MB == 0, "M must be a multiple of the barrier group");
    IK_FP4_LUT(lut);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = ((int64_t)blockIdx.x * kWarps + warp) * R;
    if (n0 >= N) return;
    const int G = K / kG;
    const uint4* wr[R];
    const uint8_t* sr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < N ? n0 + r : N - 1;
        wr[r] = reinterpret_cast<const uint4*>(w + n * (K / 2));
        sr[r] = sc + n * G;
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    float xr[ALU ? kG : 1];
    if constexpr (ALU) {
        const uint4* xp = reinterpret_cast<const uint4*>(reinterpret_cast<const ik_bf16*>(xv) + (int64_t)lane * kG);
#pragma unroll
        for (int c = 0; c < 4; ++c) ik_unpack8(xp[c], xr + 8 * c);
    }
    for (int u = lane; u < G; u += IK_WARP) {
        int ux = u;
        if constexpr (FIX) {
            int z;
            asm volatile("v_mov_b32 %0, 0" : "=v"(z));  // opaque zero: the loads cannot be hoisted out of the loop
            ux = lane + z;
        }
        float wf[R][kG], s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 wv = ik_ldcs(wr[r] + u);
            s[r] = mx_e8m0(sr[r][u]);
            ik_fp4x8_to_f32(wv.x, lut, wf[r]);
            ik_fp4x8_to_f32(wv.y, lut, wf[r] + 8);
            ik_fp4x8_to_f32(wv.z, lut, wf[r] + 16);
            ik_fp4x8_to_f32(wv.w, lut, wf[r] + 24);
        }
#pragma unroll
        for (int m0 = 0; m0 < M; m0 += MB) {
            __builtin_amdgcn_sched_barrier(0);
            float t[MB][R];
#pragma unroll
            for (int mb = 0; mb < MB; ++mb)
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    t[mb][r] = 0.f;
                    // ALU floor: every row sees the same x, so identical chains would be merged by the compiler; an opaque zero keeps them apart
                    if constexpr (ALU) asm volatile("v_mov_b32 %0, 0" : "=v"(t[mb][r]));
                }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float xf[MB][8];
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
                    if constexpr (ALU) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) xf[mb][e] = xr[8 * c + e];
                    } else if constexpr (XF) {
                        const float4* xp = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(xv) +
                                                                          (int64_t)(m0 + mb) * K + (int64_t)ux * kG);
                        const float4 a = xp[2 * c], b = xp[2 * c + 1];
                        xf[mb][0] = a.x; xf[mb][1] = a.y; xf[mb][2] = a.z; xf[mb][3] = a.w;
                        xf[mb][4] = b.x; xf[mb][5] = b.y; xf[mb][6] = b.z; xf[mb][7] = b.w;
                    } else {
                        const uint4* xp = reinterpret_cast<const uint4*>(reinterpret_cast<const ik_bf16*>(xv) +
                                                                        (int64_t)(m0 + mb) * K + (int64_t)ux * kG);
                        ik_unpack8(xp[c], xf[mb]);
                    }
                }
#pragma unroll
                for (int mb = 0; mb < MB; ++mb)
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int e = 0; e < 8; ++e) t[mb][r] = fmaf(wf[r][8 * c + e], xf[mb][e], t[mb][r]);
            }
#pragma unroll
            for (int mb = 0; mb < MB; ++mb)
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r][m0 + mb] = fmaf(t[mb][r], s[r], acc[r][m0 + mb]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float t = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) out[(int64_t)m * N + n0 + r] = t;
        }
}

// variant id: -1 = production kernel, 0..6 = template
template <int V, int M>
static void launch_vm(const void* xb, const void* xf, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
    const unsigned grid = (unsigned)((N + kWarps * kRowsR - 1) / (kWarps * kRowsR));
    constexpr bool XF = (V == 1 || V == 3 || V == 6);
    if constexpr (V < 0)
        mxfp4_gemv_rows_kernel<M, float><<<grid, kWarps * IK_WARP>>>(reinterpret_cast<const ik_bf16*>(xb), w, sc, out, N, K);
    else if constexpr (M % ((V == 5 || V == 6) ? 2 : 1) == 0)
        p0_kernel<V, M><<<grid, kWarps * IK_WARP>>>(XF ? xf : xb, w, sc, out, N, K);
}

template <int V>
static void launch_v(int M, const void* xb, const void* xf, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
    switch (M) {
        case 2: launch_vm<V, 2>(xb, xf, w, sc, out, N, K); break;
        case 4: launch_vm<V, 4>(xb, xf, w, sc, out, N, K); break;
        case 8: launch_vm<V, 8>(xb, xf, w, sc, out, N, K); break;
        case 12: launch_vm<V, 12>(xb, xf, w, sc, out, N, K); break;
        case 16: launch_vm<V, 16>(xb, xf, w, sc, out, N, K); break;
        default: printf("unsupported M %d\n", M); exit(3);
    }
}

static void launch(int v, int M, const void* xb, const void* xf, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
    switch (v) {
        case -1: launch_v<-1>(M, xb, xf, w, sc, out, N, K); break;
        case 0: launch_v<0>(M, xb, xf, w, sc, out, N, K); break;
        case 1: launch_v<1>(M, xb, xf, w, sc, out, N, K); break;
        case 2: launch_v<2>(M, xb, xf, w, sc, out, N, K); break;
        case 3: launch_v<3>(M, xb, xf, w, sc, out, N, K); break;
        case 4: launch_v<4>(M, xb, xf, w, sc, out, N, K); break;
        case 5: launch_v<5>(M, xb, xf, w, sc, out, N, K); break;
        case 6: launch_v<6>(M, xb, xf, w, sc, out, N, K); break;
    }
}

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float nrm() { float s = 0; for (int i = 0; i < 12; ++i) s += (rnd() & 0xffff) / 65536.f; return s - 6.f; }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float unbf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

static ik_tensor T(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t; memset(&t, 0, sizeof t); t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1; for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    for (int i = 1; i < argc; ++i) {
        char* end; strtoll(argv[i], &end, 10);
        if (*end) { printf("argument %d is not a number: '%s'\n", i, argv[i]); return 1; }
    }
    const int64_t N = argc > 1 ? atoll(argv[1]) : 17408, K = argc > 2 ? atoll(argv[2]) : 5120;
    const int iters = argc > 3 ? atoi(argv[3]) : 200, rounds = argc > 4 ? atoi(argv[4]) : 3;
    if (N <= 0 || K <= 0 || K % 32 || iters < 1 || iters > 2000 || rounds < 1 || rounds > 20) { printf("bad N/K/iters/rounds\n"); return 1; }
    const int MMAX = 16, COPIES = 3;
    const int64_t wb = N * K / 2, sb = N * K / 32;
    std::vector<uint8_t> hw(wb), hs(sb);
    for (auto& b : hw) b = (uint8_t)rnd();
    for (auto& b : hs) b = (uint8_t)(120 + rnd() % 15);
    std::vector<uint16_t> hx((size_t)MMAX * K);
    for (auto& v : hx) v = bf16(nrm());
    std::vector<float> hxf(hx.size());
    for (size_t i = 0; i < hx.size(); ++i) hxf[i] = unbf(hx[i]);
    uint8_t *dw[COPIES], *ds[COPIES]; uint16_t* dx; float* dxf; float* dact; float* dout; float* dref; void* dscr;
    for (int c = 0; c < COPIES; ++c) {
        HC(hipMalloc(&dw[c], wb)); HC(hipMalloc(&ds[c], sb));
        HC(hipMemcpy(dw[c], hw.data(), wb, hipMemcpyHostToDevice)); HC(hipMemcpy(ds[c], hs.data(), sb, hipMemcpyHostToDevice));
    }
    HC(hipMalloc(&dx, hx.size() * 2)); HC(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dxf, hxf.size() * 4)); HC(hipMemcpy(dxf, hxf.data(), hxf.size() * 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dact, 4)); float zero = 0.f; HC(hipMemcpy(dact, &zero, 4, hipMemcpyHostToDevice));  // W4A16: x is used as is
    HC(hipMalloc(&dout, (size_t)MMAX * N * 4)); HC(hipMalloc(&dref, (size_t)MMAX * N * 4));
    HC(hipMalloc(&dscr, (size_t)MMAX * K * 2 + 4096));
    ik_scratch scr{dscr, (int64_t)MMAX * K * 2 + 4096};
    // reference: each row alone through the production M = 1 launch (the decode path)
    for (int m = 0; m < MMAX; ++m) {
        ik_tensor tx = T(dx + (size_t)m * K, IK_DTYPE_BF16, {1, K}), tw = T(dw[0], IK_DTYPE_U8, {N, K / 2}), ts = T(ds[0], IK_DTYPE_U8, {N, K / 32});
        ik_tensor ta = T(dact, IK_DTYPE_F32, {1}), to = T(dref + (size_t)m * N, IK_DTYPE_F32, {1, N});
        const int r = ik_matmul_mxfp4_gemv(&tx, &tw, &ts, &ta, &to, &scr, nullptr);
        if (r != 0) { printf("reference launch status %d\n", r); exit(3); }
    }
    HC(hipDeviceSynchronize());
    std::vector<float> ref((size_t)MMAX * N), got((size_t)MMAX * N);
    HC(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost));
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const int vars[] = {-1, 0, 1, 2, 3, 4, 5, 6};
    const char* names[] = {"prod", "0 template", "1 f32 x (A)", "2 fixed group", "3 f32+fixed", "4 ALU floor", "5 2 rows/barrier (B)", "6 f32 + 2 rows (A+B)"};
    const bool exact[] = {true, true, true, false, false, false, true, true};
    const int Ms[] = {4, 8, 12, 16};
    printf("## mxp0 %ldx%ld iters=%d rounds=%d (us per call, min over rounds; [max]); bitwise vs the M=1 launch of each row\n", (long)N, (long)K, iters, rounds);
    double base[4] = {0, 0, 0, 0};
    for (int vi = 0; vi < 8; ++vi) {
        printf("%-22s", names[vi]);
        for (int mi = 0; mi < 4; ++mi) {
            const int M = Ms[mi], v = vars[vi];
            double best = 1e30, worst = 0;
            for (int rd = 0; rd < rounds; ++rd) {
                for (int i = 0; i < 20; ++i) launch(v, M, dx, dxf, dw[i % COPIES], ds[i % COPIES], dout, N, (int)K);
                HC(hipEventRecord(e0, nullptr));
                for (int i = 0; i < iters; ++i) launch(v, M, dx, dxf, dw[i % COPIES], ds[i % COPIES], dout, N, (int)K);
                HC(hipEventRecord(e1, nullptr)); HC(hipEventSynchronize(e1));
                float ms; HC(hipEventElapsedTime(&ms, e0, e1));
                const double us = ms * 1e3 / iters;
                best = std::min(best, us); worst = std::max(worst, us);
            }
            if (vi == 0) base[mi] = best;
            launch(v, M, dx, dxf, dw[0], ds[0], dout, N, (int)K); HC(hipDeviceSynchronize());
            HC(hipMemcpy(got.data(), dout, (size_t)M * N * 4, hipMemcpyDeviceToHost));
            long bad = 0;
            for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) if (memcmp(&got[(size_t)m * N + n], &ref[(size_t)m * N + n], 4)) ++bad;
            printf(" | M=%2d %6.1f [%6.1f] x%.2f %s", M, best, worst, best / base[mi], bad ? (exact[vi] ? "MISMATCH!" : "differs") : "bitwise");
        }
        printf("\n");
    }
    return 0;
}
