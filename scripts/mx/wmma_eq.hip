// Lane mx: is a W4A4 MXFP4 product chain on the gfx12 fp8 WMMA bitwise equal to the bf16 WMMA chain of today's kernel?
//
// Today (kernels/hip/mxfp4.hip *_mma, ik_dq_gemm.h): weights W[n,k] = e2m1 * 2^(sw[n,k/32]-127) and activations
// X[m,k] = e2m1 * 2^(sa[m,k/32]-127) are bf16 (exact), and per 16-k step acc = wmma_bf16(W_step, X_step, acc).
// Candidate: the same e2m1 codes as e4m3 bytes (exact, unscaled); acc is carried in units of the current group's scale
// 2^(sw+sa-254) (per element), acc = wmma_fp8(codes_step, codes_step, acc), and at each group boundary acc *= 2^(e_g - e_g+1)
// (a power of two, exact); at the end acc *= 2^e_last. If the matrix core's sum of 16 products + C is the same function of
// (C, exact products) on both paths and scale invariant (any floating-point rounding is, away from under/overflow), the
// outputs are equal bit for bit, including the rows whose f32 sums round. This program counts mismatches.
//
//   hipcc --offload-arch=gfx1201 -O3 -o wmma_eq wmma_eq.hip && ./wmma_eq [tiles] [seed]
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));

constexpr int K = 256;  // 8 groups of 32, 16 steps of 16
constexpr int NG = K / 32;

// e2m1 code (sign bit 3, magnitude 0..7) -> value
__host__ __device__ inline float e2m1v(int c) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (c & 8) ? -mag[c & 7] : mag[c & 7];
}
// e2m1 code -> e4m3 byte (exact): magnitudes 0,.5,1,1.5,2,3,4,6 = 0x00,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c
__host__ __device__ inline uint8_t e2m1_to_e4m3(int c) {
    const uint8_t t[8] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
    return t[c & 7] | ((c & 8) ? 0x80 : 0);
}
__host__ __device__ inline uint16_t f2bf(float f) {  // exact inputs only (e2m1 * 2^e), so truncation is exact
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

// One 16 x 16 tile per wave. wc/xc: codes [16][K] (rows n / cols m), sw/sa: e8m0 [16][NG].
__global__ void eq_kernel(const uint8_t* __restrict__ wc, const uint8_t* __restrict__ xc, const uint8_t* __restrict__ sw,
                          const uint8_t* __restrict__ sa, float* __restrict__ o_bf, float* __restrict__ o_f8, float* __restrict__ o_nc, int tiles) {
    const int t = blockIdx.x;
    if (t >= tiles) return;
    const int lane = threadIdx.x, lr = lane & 15, lk = (lane >> 4) * 8;
    const uint8_t* W = wc + (size_t)t * 16 * K;
    const uint8_t* X = xc + (size_t)t * 16 * K;
    const uint8_t* SW = sw + (size_t)t * 16 * NG;
    const uint8_t* SA = sa + (size_t)t * 16 * NG;
    // path 1: bf16 chain
    v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = 0; s < K / 16; ++s) {
        const int g = s / 2;
        uint16_t a[8], b[8];
        for (int e = 0; e < 8; ++e) {
            const int k = s * 16 + lk + e;
            a[e] = f2bf(ldexpf(e2m1v(W[lr * K + k]), (int)SW[lr * NG + g] - 127));
            b[e] = f2bf(ldexpf(e2m1v(X[lr * K + k]), (int)SA[lr * NG + g] - 127));
        }
        v8bf av, bv;
        memcpy(&av, a, 16);
        memcpy(&bv, b, 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, acc);
    }
    // path 2: fp8 chain with the carried scale. Lane holds col m = lr, rows n = lk + i.
    v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
    int ecur[8];
    for (int i = 0; i < 8; ++i) ecur[i] = (int)SW[(lk + i) * NG + 0] + (int)SA[lr * NG + 0] - 254;
    for (int s = 0; s < K / 16; ++s) {
        const int g = s / 2;
        if (s > 0 && (s & 1) == 0) {  // group boundary: rescale into the new group's units
            for (int i = 0; i < 8; ++i) {
                const int en = (int)SW[(lk + i) * NG + g] + (int)SA[lr * NG + g] - 254;
                c[i] = ldexpf(c[i], ecur[i] - en);
                ecur[i] = en;
            }
        }
        uint8_t a[8], b[8];
        for (int e = 0; e < 8; ++e) {
            const int k = s * 16 + lk + e;
            a[e] = e2m1_to_e4m3(W[lr * K + k]);
            b[e] = e2m1_to_e4m3(X[lr * K + k]);
        }
        v2i av, bv;
        memcpy(&av, a, 8);
        memcpy(&bv, b, 8);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, c);
    }
    for (int i = 0; i < 8; ++i) c[i] = ldexpf(c[i], ecur[i]);
    // negative control: per-group partials from zero, then acc += partial * 2^e (a different association)
    v8f nc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < NG; ++g) {
        v8f p = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int s = 2 * g; s < 2 * g + 2; ++s) {
            uint8_t a[8], b[8];
            for (int e = 0; e < 8; ++e) {
                const int k = s * 16 + lk + e;
                a[e] = e2m1_to_e4m3(W[lr * K + k]);
                b[e] = e2m1_to_e4m3(X[lr * K + k]);
            }
            v2i av, bv;
            memcpy(&av, a, 8);
            memcpy(&bv, b, 8);
            p = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, p);
        }
        for (int i = 0; i < 8; ++i) nc[i] += ldexpf(p[i], (int)SW[(lk + i) * NG + g] + (int)SA[lr * NG + g] - 254);
    }
    for (int i = 0; i < 8; ++i) {
        const int n = lk + i, m = lr;
        o_bf[(size_t)t * 256 + n * 16 + m] = acc[i];
        o_f8[(size_t)t * 256 + n * 16 + m] = c[i];
        o_nc[(size_t)t * 256 + n * 16 + m] = nc[i];
    }
}

static uint32_t rng_state;
static uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char** argv) {
    const int tiles = argc > 1 ? atoi(argv[1]) : 20000;
    rng_state = argc > 2 ? (uint32_t)atoi(argv[2]) : 12345u;
    std::vector<uint8_t> wc((size_t)tiles * 16 * K), xc((size_t)tiles * 16 * K), sw((size_t)tiles * 16 * NG),
        sa((size_t)tiles * 16 * NG);
    // regimes per tile: 0 = model-like (weights span 0..3 octaves, activations span 0..12), 1 = wide (activations 0..24,
    // forces rounding), 2 = cancellation (pairs of opposite large terms), 3 = sparse codes
    for (int t = 0; t < tiles; ++t) {
        const int reg = t % 4;
        const int wb = 115 + (int)(rnd() % 8), ab = 120 + (int)(rnd() % 10);
        for (int r = 0; r < 16; ++r)
            for (int g = 0; g < NG; ++g) {
                sw[((size_t)t * 16 + r) * NG + g] = (uint8_t)(wb + (int)(rnd() % 4));
                const int span = reg == 1 ? 25 : 13;
                sa[((size_t)t * 16 + r) * NG + g] = (uint8_t)(ab + (int)(rnd() % span));
            }
        for (int r = 0; r < 16; ++r)
            for (int k = 0; k < K; ++k) {
                uint32_t cw = rnd() & 15, cx = rnd() & 15;
                if (reg == 3) {
                    if (rnd() % 4) cw = 0;
                    if (rnd() % 3) cx = 0;
                }
                if (reg == 2 && (k & 1)) {  // mirror the previous k with the sign flipped and a nudged magnitude
                    cw = (wc[((size_t)t * 16 + r) * K + k - 1] ^ 8);
                    cx = (xc[((size_t)t * 16 + r) * K + k - 1] & 7) == 7 ? xc[((size_t)t * 16 + r) * K + k - 1] - 1
                                                                       : xc[((size_t)t * 16 + r) * K + k - 1] + 1;
                }
                wc[((size_t)t * 16 + r) * K + k] = (uint8_t)cw;
                xc[((size_t)t * 16 + r) * K + k] = (uint8_t)cx;
            }
    }
    uint8_t *dwc, *dxc, *dsw, *dsa;
    float *dob, *dof, *don;
    hipMalloc(&dwc, wc.size());
    hipMalloc(&dxc, xc.size());
    hipMalloc(&dsw, sw.size());
    hipMalloc(&dsa, sa.size());
    hipMalloc(&dob, (size_t)tiles * 256 * 4);
    hipMalloc(&dof, (size_t)tiles * 256 * 4);
    hipMalloc(&don, (size_t)tiles * 256 * 4);
    hipMemcpy(dwc, wc.data(), wc.size(), hipMemcpyHostToDevice);
    hipMemcpy(dxc, xc.data(), xc.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsw, sw.data(), sw.size(), hipMemcpyHostToDevice);
    hipMemcpy(dsa, sa.data(), sa.size(), hipMemcpyHostToDevice);
    eq_kernel<<<tiles, 32>>>(dwc, dxc, dsw, dsa, dob, dof, don, tiles);
    if (hipDeviceSynchronize() != hipSuccess) {
        printf("kernel failed\n");
        return 2;
    }
    std::vector<float> ob((size_t)tiles * 256), of((size_t)tiles * 256), on((size_t)tiles * 256);
    hipMemcpy(on.data(), don, on.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(ob.data(), dob, ob.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(of.data(), dof, of.size() * 4, hipMemcpyDeviceToHost);
    // exact reference (f64 sum of exact products; exact while the span fits 53 bits, which these regimes do)
    long long mism[4] = {0}, inexact[4] = {0}, total[4] = {0}, ncm[4] = {0};
    for (int t = 0; t < tiles; ++t) {
        const int reg = t % 4;
        for (int n = 0; n < 16; ++n)
            for (int m = 0; m < 16; ++m) {
                double ex = 0;
                for (int k = 0; k < K; ++k) {
                    const int g = k / 32;
                    ex += ldexp((double)e2m1v(wc[((size_t)t * 16 + n) * K + k]), (int)sw[((size_t)t * 16 + n) * NG + g] - 127) *
                          ldexp((double)e2m1v(xc[((size_t)t * 16 + m) * K + k]), (int)sa[((size_t)t * 16 + m) * NG + g] - 127);
                }
                const float b = ob[(size_t)t * 256 + n * 16 + m], f = of[(size_t)t * 256 + n * 16 + m];
                uint32_t ub, uf;
                memcpy(&ub, &b, 4);
                memcpy(&uf, &f, 4);
                ++total[reg];
                if ((double)b != ex) ++inexact[reg];
                if (ub != uf) ++mism[reg];
                const float c2 = on[(size_t)t * 256 + n * 16 + m];
                uint32_t un;
                memcpy(&un, &c2, 4);
                if (un != ub) ++ncm[reg];
            }
    }
    const char* names[4] = {"model-like", "wide", "cancel", "sparse"};
    long long tm = 0;
    for (int r = 0; r < 4; ++r) {
        printf("regime %-10s outputs %9lld  bf16-chain inexact (rounded) %9lld  fp8-carried != bf16-chain %9lld  control (per-group fma) != bf16-chain %9lld\n", names[r],
               total[r], inexact[r], mism[r], ncm[r]);
        tm += mism[r];
    }
    printf("RESULT %s (%lld mismatches)\n", tm == 0 ? "BITWISE-EQUAL" : "DIFFERENT", tm);
    return tm == 0 ? 0 : 1;
}
