// Lane mxr, Phase 0 harness. Standalone, no production code touched.
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 -I kernels/hip scripts/mxr/mxr0.hip -o mxr0
//   ./mxr0 N K [iters] [rounds] [edge 0|1] [Ms...]
// Question: on the W4A4 path (x QDQ'd to MXFP4, the 27B's scheme) every 32-element group sum t = sum_e wf[e] * xq[e] of
// the production GEMV is exact in fp32 (products are e2m1 * e2m1 * 2^ex, |sum| <= 4608 * 2^(ex-2), so every partial sum
// fits 24 bits). Then t == (float)T * 2^(sbx - 129) with T the integer dot of the doubled codes, which v_dot4_i32_iu8
// computes in any order. Same t, same acc = fma(t, s, acc) per lane in the same group order, same warp sum: the row is
// bitwise the production M = 1 launch, and the per-row ALU drops from 32 fma + 32 unpack to 8 dot4 + 3.
// Groups whose x scale byte is outside [3, 244] (tiny / huge / inf / NaN) take the production fp chain on the bf16 QDQ x.
// Arms (each launch includes its pre-pass): prod1 = production M = 1 launcher per row (reference), rows = production
// matmul_mxfp4_gemv_rows, i8rR[b] = the int8 dot4 kernel with R weight rows per warp ([b] = scheduler barrier per row).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include "../../kernels/hip/mxfp4.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

// 8 e2m1 codes (element j = nibble j) -> 8 signed int8 = 2 * e2m1 value: .x = elements 0,2,4,6, .y = 1,3,5,7.
// The order inside a dot4 does not matter (integer sums); x and w go through this same function so the pairs line up.
IK_DEVI uint2 mxr_i8x8(uint32_t w) {
    constexpr uint32_t P0 = 0x03020100u, P1 = 0x0C080604u;  // 0 1 2 3 | 4 6 8 12
    constexpr uint32_t N0 = 0xFDFEFF00u, N1 = 0xF4F8FAFCu;  // -0 -1 -2 -3 | -4 -6 -8 -12
    const uint32_t ie = w & 0x07070707u, io = (w >> 4) & 0x07070707u;
    // per-byte mask 0xFF where the sign bit is set: v_perm selector 12 -> 0x00, 13 -> 0xFF
    const uint32_t me = __builtin_amdgcn_perm(0u, 0u, ((w >> 3) & 0x01010101u) | 0x0C0C0C0Cu);
    const uint32_t mo = __builtin_amdgcn_perm(0u, 0u, ((w >> 7) & 0x01010101u) | 0x0C0C0C0Cu);
    const uint32_t ev = (__builtin_amdgcn_perm(N1, N0, ie) & me) | (__builtin_amdgcn_perm(P1, P0, ie) & ~me);
    const uint32_t od = (__builtin_amdgcn_perm(N1, N0, io) & mo) | (__builtin_amdgcn_perm(P1, P0, io) & ~mo);
    return make_uint2(ev, od);
}

IK_DEVI int mxr_dot4(uint32_t a, uint32_t b, int c) { return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, c, false); }

// pre-pass, one thread per (row, group): xb = the production QDQ (bf16, as mxfp4_xprep_kernel), xi = int8 codes in
// mxr_i8x8 order (32 B per group), xs = 2^(sb - 129) when sb in [3, 244], else 0 (fallback marker). act == 0: xb = x, xs = 0.
__global__ void mxr_xprep_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act, ik_bf16* __restrict__ xb,
                                 uint4* __restrict__ xi, float* __restrict__ xs, int64_t groups) {
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const uint4* xp = reinterpret_cast<const uint4*>(x) + g * 4;
    uint4* dp = reinterpret_cast<uint4*>(xb) + g * 4;
    if (act[0] == 0.f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) dp[i] = xp[i];
        xs[g] = 0.f;
        xi[2 * g] = make_uint4(0, 0, 0, 0);
        xi[2 * g + 1] = make_uint4(0, 0, 0, 0);
        return;
    }
    float f[kG];
#pragma unroll
    for (int i = 0; i < 4; ++i) ik_unpack8(xp[i], f + 8 * i);
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) amax = fmaxf(amax, fabsf(f[e]));
    const uint32_t sb = mx_scale_even(amax);
    const float inv = mx_e8m0(254u - sb);
    uint32_t d[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint32_t w = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) w |= mx_fp4_code(f[q * 8 + e] * inv) << (4 * e);
        const uint2 b = mxr_i8x8(w);
        d[2 * q] = b.x;
        d[2 * q + 1] = b.y;
    }
    xi[2 * g] = make_uint4(d[0], d[1], d[2], d[3]);
    xi[2 * g + 1] = make_uint4(d[4], d[5], d[6], d[7]);
    xs[g] = (sb >= 3u && sb <= 244u) ? __uint_as_float((sb - 2u) << 23) : 0.f;
    mx_qdq32(f);
#pragma unroll
    for (int i = 0; i < 4; ++i) dp[i] = ik_pack8(f + 8 * i);
}

// the production per-group chain (mxfp4_gemv_kernel<1>) on the bf16 QDQ x: the fallback for out-of-range x scales
__device__ __attribute__((noinline)) float mxr_chain(uint4 wv, const ik_bf16* __restrict__ xg, const float* lut) {
    float wf[kG];
    ik_fp4x8_to_f32(wv.x, lut, wf);
    ik_fp4x8_to_f32(wv.y, lut, wf + 8);
    ik_fp4x8_to_f32(wv.z, lut, wf + 16);
    ik_fp4x8_to_f32(wv.w, lut, wf + 24);
    const uint4* xp = reinterpret_cast<const uint4*>(xg);
    float xf[kG];
    ik_unpack8(xp[0], xf);
    ik_unpack8(xp[1], xf + 8);
    ik_unpack8(xp[2], xf + 16);
    ik_unpack8(xp[3], xf + 24);
    float t = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
    return t;
}

IK_DEVI float mxr_chain_inl(uint4 wv, const ik_bf16* __restrict__ xg, const float* lut) {
    float wf[kG];
    ik_fp4x8_to_f32(wv.x, lut, wf);
    ik_fp4x8_to_f32(wv.y, lut, wf + 8);
    ik_fp4x8_to_f32(wv.z, lut, wf + 16);
    ik_fp4x8_to_f32(wv.w, lut, wf + 24);
    const uint4* xp = reinterpret_cast<const uint4*>(xg);
    float xf[kG];
    ik_unpack8(xp[0], xf);
    ik_unpack8(xp[1], xf + 8);
    ik_unpack8(xp[2], xf + 16);
    ik_unpack8(xp[3], xf + 24);
    float t = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
    return t;
}

// FB: 1 = fallback through a noinline call, 2 = inlined fallback, 0 = no fallback (timing only: wrong on edge groups)
template <int M, int R, bool BAR, typename OutT, int FB = 1>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxr_i8_rows_kernel(const ik_bf16* __restrict__ xb, const uint4* __restrict__ xi, const float* __restrict__ xs,
                   const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, OutT* __restrict__ out, int64_t N, int K) {
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
    for (int u = lane; u < G; u += IK_WARP) {
        uint4 wv[R];
        uint32_t wi[R][8];
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wv[r] = ik_ldcs(wr[r] + u);
            s[r] = mx_e8m0(sr[r][u]);
            const uint2 a = mxr_i8x8(wv[r].x), b = mxr_i8x8(wv[r].y), c = mxr_i8x8(wv[r].z), d = mxr_i8x8(wv[r].w);
            wi[r][0] = a.x; wi[r][1] = a.y; wi[r][2] = b.x; wi[r][3] = b.y;
            wi[r][4] = c.x; wi[r][5] = c.y; wi[r][6] = d.x; wi[r][7] = d.y;
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            if constexpr (BAR) __builtin_amdgcn_sched_barrier(0);
            const int64_t gi = (int64_t)m * G + u;
            const uint4 xa = xi[2 * gi], xc = xi[2 * gi + 1];
            const float f = xs[gi];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int T = 0;
                T = mxr_dot4(wi[r][0], xa.x, T);
                T = mxr_dot4(wi[r][1], xa.y, T);
                T = mxr_dot4(wi[r][2], xa.z, T);
                T = mxr_dot4(wi[r][3], xa.w, T);
                T = mxr_dot4(wi[r][4], xc.x, T);
                T = mxr_dot4(wi[r][5], xc.y, T);
                T = mxr_dot4(wi[r][6], xc.z, T);
                T = mxr_dot4(wi[r][7], xc.w, T);
                float t = (float)T * f;
                if constexpr (FB == 1) { if (__builtin_expect(f == 0.f, 0)) t = mxr_chain(wv[r], xb + gi * kG, lut); }
                if constexpr (FB == 2) { if (__builtin_expect(f == 0.f, 0)) t = mxr_chain_inl(wv[r], xb + gi * kG, lut); }
                acc[r][m] = fmaf(t, s[r], acc[r][m]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float t = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, t);
        }
}

struct Bufs {
    const ik_bf16* x; const float* act; ik_bf16* xb; uint4* xi; float* xs;
};

template <int M, int R, bool BAR, int FB = 1>
static void i8_launch(const Bufs& b, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
    const int64_t groups = (int64_t)M * (K / kG);
    mxr_xprep_kernel<<<(unsigned)((groups + 255) / 256), 256>>>(b.x, b.act, b.xb, b.xi, b.xs, groups);
    const unsigned grid = (unsigned)((N + kWarps * R - 1) / (kWarps * R));
    mxr_i8_rows_kernel<M, R, BAR, float, FB><<<grid, kWarps * IK_WARP>>>(b.xb, b.xi, b.xs, w, sc, out, N, K);
}

template <int R, bool BAR, int FB = 1>
static void i8_by_m(int M, const Bufs& b, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
#define C(MM) case MM: i8_launch<MM, R, BAR, FB>(b, w, sc, out, N, K); break
    switch (M) {
        C(1); C(2); C(3); C(4); C(5); C(6); C(7); C(8); C(9); C(10); C(11); C(12); C(13); C(14); C(15); C(16);
        default: printf("bad M\n"); exit(3);
    }
#undef C
}

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float nrm() { float s = 0; for (int i = 0; i < 12; ++i) s += (rnd() & 0xffff) / 65536.f; return s - 6.f; }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x40); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

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
    const int edge = argc > 5 ? atoi(argv[5]) : 0;
    std::vector<int> Ms;
    for (int i = 6; i < argc; ++i) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {1, 2, 3, 4, 6, 8, 12, 16};
    if (N <= 0 || K <= 0 || K % 32 || iters < 1 || iters > 2000 || rounds < 1 || rounds > 20) { printf("bad N/K/iters/rounds\n"); return 1; }
    for (int M : Ms) if (M < 1 || M > 16) { printf("bad M %d\n", M); return 1; }
    const int MMAX = 16, COPIES = 3;
    const int64_t G = K / 32, wb = N * K / 2, sb = N * G;
    std::vector<uint8_t> hw(wb), hs(sb);
    for (auto& b : hw) b = (uint8_t)rnd();
    // weight scales: the checkpoint's range, or (edge) every byte incl. 0 (2^-127), 1, 2, 250..254 and 255 (NaN) sometimes
    for (auto& b : hs) b = edge ? (uint8_t)((rnd() % 8 == 0) ? (rnd() % 2 ? rnd() % 4 : 250 + rnd() % 6) : 100 + rnd() % 50) : (uint8_t)(120 + rnd() % 15);
    std::vector<uint16_t> hx((size_t)MMAX * K);
    for (auto& v : hx) v = bf16(nrm());
    int nedge = 0;
    if (edge) {  // per-group extremes of x: tiny, huge, overflow, inf, NaN, all zero, -0, 2^-126 boundary
        for (int64_t g = 0; g < (int64_t)MMAX * G; ++g) {
            const uint32_t kind = rnd() % 16;
            if (kind > 8) continue;
            ++nedge;
            for (int e = 0; e < 32; ++e) {
                float v; uint16_t& h = hx[(size_t)g * 32 + e];
                uint32_t u = (uint32_t)h << 16; memcpy(&v, &u, 4);
                switch (kind) {
                    case 0: v = ldexpf(v, -118); break;  // sb ~ 7..10
                    case 1: v = ldexpf(v, -124); break;  // sb ~ 1..4: crosses the [3, 244] edge
                    case 2: v = ldexpf(v, -130); break;  // subnormal x
                    case 3: v = ldexpf(v, 112); break;   // sb ~ 240..243
                    case 4: v = ldexpf(v, 118); break;   // sb ~ 246..250
                    case 5: v = ldexpf(v, 126); break;   // sb 254, overflow in QDQ
                    case 6: if (e == 5) v = (rnd() % 2) ? INFINITY : -INFINITY; break;
                    case 7: if (e == 9) v = NAN; break;
                    case 8: v = (rnd() % 2) ? 0.f : -0.f; break;
                }
                h = bf16(v);
            }
        }
    }
    uint8_t *dw[COPIES], *ds[COPIES]; uint16_t* dx; float* dact; float* dout; float* dref; void* dscr;
    ik_bf16* dxb; uint4* dxi; float* dxs;
    for (int c = 0; c < COPIES; ++c) {
        HC(hipMalloc(&dw[c], wb)); HC(hipMalloc(&ds[c], sb));
        HC(hipMemcpy(dw[c], hw.data(), wb, hipMemcpyHostToDevice)); HC(hipMemcpy(ds[c], hs.data(), sb, hipMemcpyHostToDevice));
    }
    HC(hipMalloc(&dx, hx.size() * 2)); HC(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dact, 4)); float one = 1.f; HC(hipMemcpy(dact, &one, 4, hipMemcpyHostToDevice));  // W4A4
    HC(hipMalloc(&dout, (size_t)MMAX * N * 4)); HC(hipMalloc(&dref, (size_t)MMAX * N * 4));
    HC(hipMalloc(&dscr, (size_t)MMAX * K * 2 + 4096));
    HC(hipMalloc(&dxb, (size_t)MMAX * K * 2)); HC(hipMalloc(&dxi, (size_t)MMAX * K)); HC(hipMalloc(&dxs, (size_t)MMAX * G * 4));
    ik_scratch scr{dscr, (int64_t)MMAX * K * 2 + 4096};
    auto prod1 = [&](int m, int c, float* o) {
        ik_tensor tx = T(dx + (size_t)m * K, IK_DTYPE_BF16, {1, K}), tw = T(dw[c], IK_DTYPE_U8, {N, K / 2}), ts = T(ds[c], IK_DTYPE_U8, {N, G});
        ik_tensor ta = T(dact, IK_DTYPE_F32, {1}), to = T(o, IK_DTYPE_F32, {1, N});
        const int r = ik_matmul_mxfp4_gemv(&tx, &tw, &ts, &ta, &to, &scr, nullptr);
        if (r != 0) { printf("prod1 status %d\n", r); exit(3); }
    };
    auto rows = [&](int M, int c, float* o) {
        ik_tensor tx = T(dx, IK_DTYPE_BF16, {M, K}), tw = T(dw[c], IK_DTYPE_U8, {N, K / 2}), ts = T(ds[c], IK_DTYPE_U8, {N, G});
        ik_tensor ta = T(dact, IK_DTYPE_F32, {1}), to = T(o, IK_DTYPE_F32, {M, N});
        const int r = M == 1 ? ik_matmul_mxfp4_gemv(&tx, &tw, &ts, &ta, &to, &scr, nullptr)
                             : ik_matmul_mxfp4_gemv_rows(&tx, &tw, &ts, &ta, &to, &scr, nullptr);
        if (r != 0) { printf("rows status %d\n", r); exit(3); }
    };
    const Bufs bufs{reinterpret_cast<const ik_bf16*>(dx), dact, dxb, dxi, dxs};
    // reference: each row alone through the production M = 1 launch (the decode path)
    for (int m = 0; m < MMAX; ++m) prod1(m, 0, dref + (size_t)m * N);
    HC(hipDeviceSynchronize());
    std::vector<float> ref((size_t)MMAX * N), got((size_t)MMAX * N);
    HC(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost));
    long nanref = 0; for (float v : ref) nanref += (v != v);
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    struct Arm { const char* name; int id; };
    const Arm arms[] = {{"prod1 (M=1 xM)", 0}, {"rows (prod)", 1}, {"i8r2b", 2}, {"i8r2", 3}, {"i8r4b", 4}, {"i8r4", 5}, {"i8r1b", 6}, {"i8r2b-inl", 7}, {"i8r2b-nofb*", 8}, {"i8r4b-nofb*", 9}};
    auto run = [&](int id, int M, int c, float* o) {
        switch (id) {
            case 0: for (int m = 0; m < M; ++m) prod1(m, c, o + (size_t)m * N); break;
            case 1: rows(M, c, o); break;
            case 2: i8_by_m<2, true>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 3: i8_by_m<2, false>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 4: i8_by_m<4, true>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 5: i8_by_m<4, false>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 6: i8_by_m<1, true>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 7: i8_by_m<2, true, 2>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 8: i8_by_m<2, true, 0>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 9: i8_by_m<4, true, 0>(M, bufs, dw[c], ds[c], o, N, (int)K); break;
        }
    };
    printf("## mxr0 %ldx%ld iters=%d rounds=%d edge=%d (edge groups %d of %ld, NaN ref outputs %ld); us per call incl. pre-pass, min over rounds [max]; x = ratio to prod1 M=1; bitwise vs prod1 per row\n",
           (long)N, (long)K, iters, rounds, edge, nedge, (long)(MMAX * G), nanref);
    double t1 = 0;
    long total_bad = 0;
    for (const Arm& a : arms) {
        printf("%-16s", a.name);
        for (int M : Ms) {
            if (a.id == 0 && M > 1) { printf(" |"); continue; }
            double best = 1e30, worst = 0;
            for (int rd = 0; rd < rounds; ++rd) {
                for (int i = 0; i < 10; ++i) run(a.id, M, i % COPIES, dout);
                HC(hipEventRecord(e0, nullptr));
                for (int i = 0; i < iters; ++i) run(a.id, M, i % COPIES, dout);
                HC(hipEventRecord(e1, nullptr)); HC(hipEventSynchronize(e1));
                float ms; HC(hipEventElapsedTime(&ms, e0, e1));
                const double us = ms * 1e3 / iters;
                best = std::min(best, us); worst = std::max(worst, us);
            }
            if (a.id == 0) t1 = best;
            HC(hipMemset(dout, 0xCD, (size_t)MMAX * N * 4));
            run(a.id, M, 0, dout); HC(hipDeviceSynchronize());
            HC(hipMemcpy(got.data(), dout, (size_t)M * N * 4, hipMemcpyDeviceToHost));
            long bad = 0;
            for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) if (memcmp(&got[(size_t)m * N + n], &ref[(size_t)m * N + n], 4)) ++bad;
            if (!(edge && a.id >= 8)) total_bad += bad;
            printf(" | M=%2d %7.1f [%7.1f] x%.2f %s", M, best, worst, t1 > 0 ? best / t1 : 0.0, bad ? ("MISMATCH " + std::to_string(bad)).c_str() : "ok");
        }
        printf("\n");
    }
    printf("## total mismatches %ld\n", total_bad);
    return total_bad ? 4 : 0;
}
