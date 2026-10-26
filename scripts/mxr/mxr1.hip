// Lane mxr, Phase 0 harness, round 2 (window w2). Standalone, no production code touched.
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 -I kernels/hip scripts/mxr/mxr1.hip -o mxr1
//   ./mxr1 N K [iters] [rounds] [edge 0|1] [Ms...]
// Round 1 (mxr0.hip, docs/results/2026-10-26-mxr-hip-mxfp4-rows.md) showed the int8 dot4 rows kernel is bitwise the
// production M = 1 launch on every shape and edge case, and that the cost is in the per-group fallback (a noinline call or
// an inlined LUT chain raises VGPRs) and in the pre-pass (one thread per group, ~6 us). This round:
//   FB 4  the fallback is per call: the pre-pass writes one flag per (row, block) when a group's x scale byte is outside
//         [3, 244] (or act == 0); the kernel reads the flags with scalar loads and, if any is set, runs the production
//         fp chain for the whole call (slow, only for pathological inputs), else the int8 path with no per-group check.
//         FB 0 = no fallback at all (timing only, wrong on edge inputs)
//   PF    the next group's weights are loaded before the current group is computed
//   prep4 a pre-pass with 4 threads per group (8 elements each, amax over 2 xor shuffles)
//   m1    M = 1 with the pre-pass fused: each block of a device-sized grid stages x (int8 codes, scales, bf16) in LDS
// Every arm is checked bitwise against the production M = 1 launch of each row.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include "../../kernels/hip/mxfp4.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

// 8 e2m1 codes (element j = nibble j) -> 8 signed int8 = 2 * e2m1 value: .x = elements 0,2,4,6, .y = 1,3,5,7
IK_DEVI uint2 mxr_i8x8(uint32_t w) {
    constexpr uint32_t P0 = 0x03020100u, P1 = 0x0C080604u;  // 0 1 2 3 | 4 6 8 12
    constexpr uint32_t N0 = 0xFDFEFF00u, N1 = 0xF4F8FAFCu;  // -0 -1 -2 -3 | -4 -6 -8 -12
    const uint32_t ie = w & 0x07070707u, io = (w >> 4) & 0x07070707u;
    const uint32_t me = __builtin_amdgcn_perm(0u, 0u, ((w >> 3) & 0x01010101u) | 0x0C0C0C0Cu);  // sel 12 -> 0x00, 13 -> 0xFF
    const uint32_t mo = __builtin_amdgcn_perm(0u, 0u, ((w >> 7) & 0x01010101u) | 0x0C0C0C0Cu);
    const uint32_t ev = (__builtin_amdgcn_perm(N1, N0, ie) & me) | (__builtin_amdgcn_perm(P1, P0, ie) & ~me);
    const uint32_t od = (__builtin_amdgcn_perm(N1, N0, io) & mo) | (__builtin_amdgcn_perm(P1, P0, io) & ~mo);
    return make_uint2(ev, od);
}
IK_DEVI int mxr_dot4(uint32_t a, uint32_t b, int c) { return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, c, false); }
IK_DEVI float mxr_scale(uint32_t sb) { return (sb >= 3u && sb <= 244u) ? __uint_as_float((sb - 2u) << 23) : 0.f; }

// the production per-group chain (mxfp4_gemv_kernel<1>) with the weights rebuilt from the int8 codes (2 * e2m1, so * 0.5 is
// the LUT value; a -0 code becomes +0, which no finite sum sees) and x from the bf16 QDQ copy
IK_DEVI float mxr_chain_i8(const uint32_t* wi, const uint4* xp) {
    float xf[kG];
    ik_unpack8(xp[0], xf);
    ik_unpack8(xp[1], xf + 8);
    ik_unpack8(xp[2], xf + 16);
    ik_unpack8(xp[3], xf + 24);
    float t = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) {
        const uint32_t d = wi[2 * (e >> 3) + (e & 1)];
        const float wf = (float)(int)(int8_t)(d >> (8 * ((e & 7) >> 1))) * 0.5f;
        t = fmaf(wf, xf[e], t);
    }
    return t;
}

// the production per-group chain (mxfp4_gemv_kernel<1>, LUT values from ik_fp4_value, which fills IK_FP4_LUT) on the
// bf16 QDQ x: the slow path, bitwise the production arithmetic by construction
IK_DEVI float mxr_chain_prod(uint4 wv, const uint4* xp) {
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
    float t = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float xf[8];
        ik_unpack8(xp[c], xf);
#pragma unroll
        for (int e = 0; e < 8; ++e) t = fmaf(ik_fp4_value((ww[c] >> (4 * e)) & 15u), xf[e], t);
    }
    return t;
}

// one 32-element group of activations -> int8 codes (mxr_i8x8 order, 8 dwords), scale, bf16 QDQ (as mxfp4_xprep_kernel)
IK_DEVI void mxr_prep_group(const uint4* xp, bool qdq, uint32_t* d, float& xsv, uint4* bq) {
    float f[kG];
#pragma unroll
    for (int i = 0; i < 4; ++i) ik_unpack8(xp[i], f + 8 * i);
    if (!qdq) {
#pragma unroll
        for (int i = 0; i < 8; ++i) d[i] = 0;
        xsv = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) bq[i] = xp[i];
        return;
    }
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) amax = fmaxf(amax, fabsf(f[e]));
    const uint32_t sb = mx_scale_even(amax);
    const float inv = mx_e8m0(254u - sb);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint32_t w = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) w |= mx_fp4_code(f[q * 8 + e] * inv) << (4 * e);
        const uint2 b = mxr_i8x8(w);
        d[2 * q] = b.x;
        d[2 * q + 1] = b.y;
    }
    xsv = mxr_scale(sb);
    mx_qdq32(f);
#pragma unroll
    for (int i = 0; i < 4; ++i) bq[i] = ik_pack8(f + 8 * i);
}

// pre-pass, one thread per group (round 1)
__global__ void mxr_xprep_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act, ik_bf16* __restrict__ xb,
                                 uint4* __restrict__ xi, float* __restrict__ xs, int64_t groups) {
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    uint32_t d[8];
    mxr_prep_group(reinterpret_cast<const uint4*>(x) + g * 4, act[0] != 0.f, d, xs[g], reinterpret_cast<uint4*>(xb) + g * 4);
    xi[2 * g] = make_uint4(d[0], d[1], d[2], d[3]);
    xi[2 * g + 1] = make_uint4(d[4], d[5], d[6], d[7]);
}

// pre-pass, 4 threads per group (8 elements each); the 4 threads of a group are consecutive lanes of one warp
__global__ void __launch_bounds__(256) mxr_xprep4_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act,
                                                          ik_bf16* __restrict__ xb, uint4* __restrict__ xi,
                                                          float* __restrict__ xs, int64_t groups) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // 8-element chunk
    const bool ok = i < groups * 4;
    const uint4 v = ok ? reinterpret_cast<const uint4*>(x)[i] : make_uint4(0, 0, 0, 0);
    const int64_t g = i >> 2;
    const int q = (int)(i & 3);
    uint2* xi2 = reinterpret_cast<uint2*>(xi);
    if (act[0] == 0.f) {
        if (ok) {
            reinterpret_cast<uint4*>(xb)[i] = v;
            xi2[g * 4 + q] = make_uint2(0, 0);
            if (q == 0) xs[g] = 0.f;
        }
        return;
    }
    float f[8];
    ik_unpack8(v, f);
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
    amax = fmaxf(amax, __shfl_xor(amax, 1, IK_WARP));
    amax = fmaxf(amax, __shfl_xor(amax, 2, IK_WARP));
    const uint32_t sb = mx_scale_even(amax);
    const float s = mx_e8m0(sb), inv = mx_e8m0(254u - sb);
    uint32_t w = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) w |= mx_fp4_code(f[e] * inv) << (4 * e);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = mx_fp4_round(f[e] * inv) * s;  // mx_qdq32's per-element step
    if (!ok) return;
    xi2[g * 4 + q] = mxr_i8x8(w);
    reinterpret_cast<uint4*>(xb)[i] = ik_pack8(f);
    if (q == 0) xs[g] = mxr_scale(sb);
}

// pre-pass with flags: grid (S, M), 4 threads per group; flags[m * S + s] = 1 when a group of that block takes the slow path
__global__ void __launch_bounds__(256) mxr_xprep4f_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act,
                                                           ik_bf16* __restrict__ xb, uint4* __restrict__ xi,
                                                           float* __restrict__ xs, uint8_t* __restrict__ flags, int G) {
    const int m = blockIdx.y, S = gridDim.x;
    const int C = G * 4;  // 8-element chunks per row
    const bool qdq = act[0] != 0.f;
    bool bad = !qdq;
    uint2* xi2 = reinterpret_cast<uint2*>(xi);
    for (int c = blockIdx.x * 256 + threadIdx.x; c < C; c += S * 256) {  // the 4 chunks of a group share an iteration
        const int64_t i = (int64_t)m * C + c, g = i >> 2;
        const int q = c & 3;
        const uint4 v = reinterpret_cast<const uint4*>(x)[i];
        if (!qdq) {
            reinterpret_cast<uint4*>(xb)[i] = v;
            xi2[g * 4 + q] = make_uint2(0, 0);
            if (q == 0) xs[g] = 0.f;
            continue;
        }
        float f[8];
        ik_unpack8(v, f);
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
        amax = fmaxf(amax, __shfl_xor(amax, 1, IK_WARP));
        amax = fmaxf(amax, __shfl_xor(amax, 2, IK_WARP));
        const uint32_t sb = mx_scale_even(amax);
        const float s = mx_e8m0(sb), inv = mx_e8m0(254u - sb);
        uint32_t w = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) w |= mx_fp4_code(f[e] * inv) << (4 * e);
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = mx_fp4_round(f[e] * inv) * s;  // mx_qdq32's per-element step
        xi2[g * 4 + q] = mxr_i8x8(w);
        reinterpret_cast<uint4*>(xb)[i] = ik_pack8(f);
        const float xsv = mxr_scale(sb);
        bad |= xsv == 0.f;
        if (q == 0) xs[g] = xsv;
    }
    const int any = __syncthreads_or(bad ? 1 : 0);
    if (threadIdx.x == 0) flags[m * S + blockIdx.x] = any ? 1 : 0;
}

// the int8 rows kernel. FB 0: no fallback (wrong on edge groups, timing only); FB 4: per-call flags, slow path = production chain.
template <int M, int R, int FB, bool PF, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxr_i8_rows_kernel(const ik_bf16* __restrict__ xb, const uint4* __restrict__ xi, const float* __restrict__ xs,
                   const uint8_t* __restrict__ flags, int nflags,
                   const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, OutT* __restrict__ out, int64_t N, int K) {
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
    if constexpr (FB == 4) {
        uint32_t bad = 0;
        for (int i = 0; i < nflags; ++i) bad |= flags[i];  // uniform: scalar loads
        if (__builtin_expect(bad != 0u, 0)) {
#pragma unroll 1
            for (int m = 0; m < M; ++m) {
#pragma unroll 1
                for (int r = 0; r < R; ++r) {
                    float a = 0.f;
                    for (int u = lane; u < G; u += IK_WARP) {
                        const float t = mxr_chain_prod(ik_ldcs(wr[r] + u), reinterpret_cast<const uint4*>(xb + ((int64_t)m * G + u) * kG));
                        a = fmaf(t, mx_e8m0(sr[r][u]), a);
                    }
                    const float t = ik_warp_sum(a);
                    if (lane == 0 && n0 + r < N) ik_store(out + (int64_t)m * N + n0 + r, t);
                }
            }
            return;
        }
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    uint4 wn[R];
    uint32_t bn[R];
    if constexpr (PF) {
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (lane < G) { wn[r] = ik_ldcs(wr[r] + lane); bn[r] = sr[r][lane]; }
    }
    for (int u = lane; u < G; u += IK_WARP) {
        uint4 wv[R];
        uint32_t sbw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if constexpr (PF) {
                wv[r] = wn[r];
                sbw[r] = bn[r];
                if (u + IK_WARP < G) { wn[r] = ik_ldcs(wr[r] + u + IK_WARP); bn[r] = sr[r][u + IK_WARP]; }
            } else {
                wv[r] = ik_ldcs(wr[r] + u);
                sbw[r] = sr[r][u];
            }
        }
        uint32_t wi[R][8];
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s[r] = mx_e8m0(sbw[r]);
            const uint2 a = mxr_i8x8(wv[r].x), b = mxr_i8x8(wv[r].y), c = mxr_i8x8(wv[r].z), d = mxr_i8x8(wv[r].w);
            wi[r][0] = a.x; wi[r][1] = a.y; wi[r][2] = b.x; wi[r][3] = b.y;
            wi[r][4] = c.x; wi[r][5] = c.y; wi[r][6] = d.x; wi[r][7] = d.y;
        }
        const uint4* xg = xi + 2 * (int64_t)u;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            __builtin_amdgcn_sched_barrier(0);
            const uint4 xa = xg[0], xc = xg[1];
            const float f = xs[(int64_t)m * G + u];
            float t[R];
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
                t[r] = (float)T * f;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r][m] = fmaf(t[r], s[r], acc[r][m]);
            xg += 2 * (int64_t)G;
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

// M = 1 with the pre-pass fused (K <= 8192): LDS = codes plane 0 [G] uint4, plane 1 [G] uint4, scale [G] f32, bf16 [G][4] uint4
template <int R, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxr_i8_m1_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act, const uint8_t* __restrict__ w,
                 const uint8_t* __restrict__ sc, OutT* __restrict__ out, int64_t N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int G = K / kG;
    uint4* p0 = reinterpret_cast<uint4*>(smem);
    uint4* p1 = p0 + G;
    uint4* pb = p1 + G;                              // bf16 QDQ, 4 uint4 per group
    float* pxs = reinterpret_cast<float*>(pb + 4 * G);
    const bool qdq = act[0] != 0.f;
    bool badl = false;
    for (int g = threadIdx.x; g < G; g += kWarps * IK_WARP) {
        uint32_t d[8];
        float xsv;
        uint4 bq[4];
        mxr_prep_group(reinterpret_cast<const uint4*>(x) + (int64_t)g * 4, qdq, d, xsv, bq);
        p0[g] = make_uint4(d[0], d[1], d[2], d[3]);
        p1[g] = make_uint4(d[4], d[5], d[6], d[7]);
        pxs[g] = xsv;
        badl |= xsv == 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) pb[4 * g + i] = bq[i];
    }
    const bool bad = __syncthreads_or(badl ? 1 : 0) != 0;  // also the barrier after the staging
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t stride = (int64_t)gridDim.x * kWarps * R;
    if (__builtin_expect(bad, 0)) {
        for (int64_t n = (int64_t)blockIdx.x * kWarps + warp; n < N; n += (int64_t)gridDim.x * kWarps) {
            const uint4* wr = reinterpret_cast<const uint4*>(w + n * (K / 2));
            float a = 0.f;
            for (int u = lane; u < G; u += IK_WARP) a = fmaf(mxr_chain_prod(ik_ldcs(wr + u), pb + 4 * u), mx_e8m0(sc[n * G + u]), a);
            const float t = ik_warp_sum(a);
            if (lane == 0) ik_store(out + n, t);
        }
        return;
    }
    for (int64_t n0 = ((int64_t)blockIdx.x * kWarps + warp) * R; n0 < N; n0 += stride) {
        const uint4* wr[R];
        const uint8_t* sr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int64_t n = n0 + r < N ? n0 + r : N - 1;
            wr[r] = reinterpret_cast<const uint4*>(w + n * (K / 2));
            sr[r] = sc + n * G;
        }
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.f;
        uint4 wn[R];
        uint32_t bn[R];
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (lane < G) { wn[r] = ik_ldcs(wr[r] + lane); bn[r] = sr[r][lane]; }
        for (int u = lane; u < G; u += IK_WARP) {
            uint4 wv[R];
            uint32_t sbw[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                wv[r] = wn[r];
                sbw[r] = bn[r];
                if (u + IK_WARP < G) { wn[r] = ik_ldcs(wr[r] + u + IK_WARP); bn[r] = sr[r][u + IK_WARP]; }
            }
            const uint4 xa = p0[u], xc = p1[u];
            const float f = pxs[u];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                uint32_t wi[8];
                const uint2 a = mxr_i8x8(wv[r].x), b = mxr_i8x8(wv[r].y), c = mxr_i8x8(wv[r].z), d = mxr_i8x8(wv[r].w);
                wi[0] = a.x; wi[1] = a.y; wi[2] = b.x; wi[3] = b.y; wi[4] = c.x; wi[5] = c.y; wi[6] = d.x; wi[7] = d.y;
                int T = 0;
                T = mxr_dot4(wi[0], xa.x, T);
                T = mxr_dot4(wi[1], xa.y, T);
                T = mxr_dot4(wi[2], xa.z, T);
                T = mxr_dot4(wi[3], xa.w, T);
                T = mxr_dot4(wi[4], xc.x, T);
                T = mxr_dot4(wi[5], xc.y, T);
                T = mxr_dot4(wi[6], xc.z, T);
                T = mxr_dot4(wi[7], xc.w, T);
                acc[r] = fmaf((float)T * f, mx_e8m0(sbw[r]), acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float t = ik_warp_sum(acc[r]);
            if (lane == 0 && n0 + r < N) ik_store(out + n0 + r, t);
        }
    }
}

struct Bufs {
    const ik_bf16* x; const float* act; ik_bf16* xb; uint4* xi; float* xs; uint8_t* flags;
};

template <int M, int R, int FB, bool PF>
static void i8_launch(int prep, const Bufs& b, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
    const int G = K / kG;
    const int64_t groups = (int64_t)M * G;
    int S = (G * 4 + 255) / 256;
    S = S > 8 ? 8 : S;
    if (prep == 5)
        mxr_xprep4f_kernel<<<dim3((unsigned)S, (unsigned)M), 256>>>(b.x, b.act, b.xb, b.xi, b.xs, b.flags, G);
    else if (prep == 4)
        mxr_xprep4_kernel<<<(unsigned)((groups * 4 + 255) / 256), 256>>>(b.x, b.act, b.xb, b.xi, b.xs, groups);
    else
        mxr_xprep_kernel<<<(unsigned)((groups + 255) / 256), 256>>>(b.x, b.act, b.xb, b.xi, b.xs, groups);
    const unsigned grid = (unsigned)((N + kWarps * R - 1) / (kWarps * R));
    mxr_i8_rows_kernel<M, R, FB, PF, float><<<grid, kWarps * IK_WARP>>>(b.xb, b.xi, b.xs, b.flags, M * S, w, sc, out, N, K);
}

template <int R, int FB, bool PF>
static void i8_by_m(int M, int prep, const Bufs& b, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
#define C(MM) case MM: i8_launch<MM, R, FB, PF>(prep, b, w, sc, out, N, K); break
    switch (M) {
        C(1); C(2); C(3); C(4); C(5); C(6); C(7); C(8); C(9); C(10); C(11); C(12); C(13); C(14); C(15); C(16);
        default: printf("bad M\n"); exit(3);
    }
#undef C
}

template <int R>
static void m1_launch(const Bufs& b, const uint8_t* w, const uint8_t* sc, float* out, int64_t N, int K) {
    const int G = K / kG;
    const size_t lds = (size_t)G * (16 + 16 + 64 + 4);
    const int64_t full = (N + kWarps * R - 1) / (kWarps * R);
    const unsigned grid = (unsigned)(full < mx_lds_blocks() ? full : mx_lds_blocks());
    mxr_i8_m1_kernel<R, float><<<grid, kWarps * IK_WARP, lds>>>(b.x, b.act, w, sc, out, N, K);
}

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float nrm() { float s = 0; for (int i = 0; i < 12; ++i) s += (rnd() & 0xffff) / 65536.f; return s - 6.f; }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x40); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

// host copy of mx_scale_even (kernels/hip/mxfp4.hip)
static uint32_t h_scale_even(float amax) {
    if (amax != amax) return 255u;
    uint32_t u; memcpy(&u, &amax, 4);
    const uint32_t r = (u + (1u << 21)) & 0x7F800000u;
    const int biased = (int)(r >> 23);
    if (biased == 0) return 0u;
    if (biased == 255) return 254u;
    int e = biased - 129;
    e = e < -127 ? -127 : (e > 127 ? 127 : e);
    return (uint32_t)(e + 127);
}

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
    // weight scales: the checkpoint's range; edge: 100..149, 1/32 of groups at 0..3 or 250..254, 255 (NaN) only on rows n % 97 == 5
    for (int64_t i = 0; i < sb; ++i) {
        const int64_t n = i / G;
        uint8_t v = (uint8_t)(120 + rnd() % 15);
        if (edge) {
            v = (uint8_t)(100 + rnd() % 50);
            if (rnd() % 32 == 0) v = (uint8_t)(rnd() % 2 ? rnd() % 4 : 250 + rnd() % 5);
            if (n % 97 == 5 && rnd() % 8 == 0) v = 255;
        }
        hs[i] = v;
    }
    std::vector<uint16_t> hx((size_t)MMAX * K);
    for (auto& v : hx) v = bf16(nrm());
    int nedge = 0;
    if (edge == 2) {  // the fast path's extremes: x scale bytes 3..6 and 238..244 (inside [3, 244]), zero and -0 groups
        for (int64_t g = 0; g < (int64_t)MMAX * G; ++g) {
            const uint32_t r = rnd();
            if (r % 3) continue;
            ++nedge;
            uint16_t* h = &hx[(size_t)g * 32];
            if (r / 3 % 8 == 0) { for (int e = 0; e < 32; ++e) h[e] = (r / 24 % 2) ? 0x8000 : 0; continue; }
            const int target = (r / 3 % 2) ? 3 + (int)(r / 6 % 4) : 238 + (int)(r / 6 % 7);
            float f[32];
            for (int e = 0; e < 32; ++e) { uint32_t u = (uint32_t)h[e] << 16; memcpy(&f[e], &u, 4); }
            for (int k = -140; k <= 140; ++k) {
                float a = 0.f, v[32];
                for (int e = 0; e < 32; ++e) { v[e] = ldexpf(f[e], k); uint16_t b = bf16(v[e]); uint32_t u = (uint32_t)b << 16; memcpy(&v[e], &u, 4); a = fmaxf(a, fabsf(v[e])); }
                if ((int)h_scale_even(a) == target) { for (int e = 0; e < 32; ++e) h[e] = bf16(v[e]); break; }
            }
        }
    } else if (edge) {  // per row class: m%4 == 0 clean; 1 tiny scales / zero groups; 2 huge scales; 3 one inf and one NaN group, -0 groups
        for (int m = 0; m < MMAX; ++m)
            for (int64_t g = 0; g < G; ++g) {
                const int cls = m % 4;
                int kind = -1;
                const uint32_t r = rnd();
                if (cls == 1 && r % 4 == 0) kind = (int)(r / 4 % 4);           // 0 1 2 3
                if (cls == 2 && r % 4 == 0) kind = 4 + (int)(r / 4 % 3);       // 4 5 6
                if (cls == 3) kind = g == 3 ? 7 : (g == G / 2 ? 8 : (r % 8 == 0 ? 9 : -1));
                if (kind < 0) continue;
                ++nedge;
                for (int e = 0; e < 32; ++e) {
                    uint16_t& h = hx[((size_t)m * G + g) * 32 + e];
                    float v; uint32_t u = (uint32_t)h << 16; memcpy(&v, &u, 4);
                    switch (kind) {
                        case 0: v = ldexpf(v, -118); break;  // sb ~ 7..10
                        case 1: v = ldexpf(v, -124); break;  // sb ~ 1..4: crosses the lower edge
                        case 2: v = ldexpf(v, -130); break;  // subnormal x
                        case 3: v = 0.f; break;              // all-zero group
                        case 4: v = ldexpf(v, 112); break;   // sb ~ 240..243
                        case 5: v = ldexpf(v, 118); break;   // sb ~ 246..250: above the upper edge
                        case 6: v = ldexpf(v, 126); break;   // sb 254, overflow in the QDQ
                        case 7: if (e == 5) v = INFINITY; break;
                        case 8: if (e == 9) v = NAN; break;
                        case 9: v = -0.f; break;
                    }
                    h = bf16(v);
                }
            }
    }
    uint8_t *dw[COPIES], *ds[COPIES]; uint16_t* dx; float* dact; float* dout; float* dref; void* dscr;
    ik_bf16* dxb; uint4* dxi; float* dxs; uint8_t* dfl;
    for (int c = 0; c < COPIES; ++c) {
        HC(hipMalloc(&dw[c], wb)); HC(hipMalloc(&ds[c], sb));
        HC(hipMemcpy(dw[c], hw.data(), wb, hipMemcpyHostToDevice)); HC(hipMemcpy(ds[c], hs.data(), sb, hipMemcpyHostToDevice));
    }
    HC(hipMalloc(&dx, hx.size() * 2)); HC(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dact, 4)); float one = 1.f; HC(hipMemcpy(dact, &one, 4, hipMemcpyHostToDevice));  // W4A4
    HC(hipMalloc(&dout, (size_t)MMAX * N * 4)); HC(hipMalloc(&dref, (size_t)MMAX * N * 4));
    HC(hipMalloc(&dscr, (size_t)MMAX * K * 2 + 4096));
    HC(hipMalloc(&dxb, (size_t)MMAX * K * 2)); HC(hipMalloc(&dxi, (size_t)MMAX * K)); HC(hipMalloc(&dxs, (size_t)MMAX * G * 4)); HC(hipMalloc(&dfl, 256));
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
    const Bufs bufs{reinterpret_cast<const ik_bf16*>(dx), dact, dxb, dxi, dxs, dfl};
    for (int m = 0; m < MMAX; ++m) prod1(m, 0, dref + (size_t)m * N);
    HC(hipDeviceSynchronize());
    std::vector<float> ref((size_t)MMAX * N), got((size_t)MMAX * N);
    HC(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost));
    long nanref = 0, infref = 0, finref = 0;
    for (float v : ref) { if (v != v) ++nanref; else if (std::isinf(v)) ++infref; else ++finref; }
    hipEvent_t e0, e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    struct Arm { const char* name; int id; bool exact; bool m1only; };
    const Arm arms[] = {{"prod1", 0, true, true},          {"rows (prod)", 1, true, false},  {"r2 nofb* p1", 10, false, false},
                        {"r2 nofb* p4", 2, false, false},  {"r2 fb4", 3, true, false},       {"r2 fb4 pf", 4, true, false},
                        {"r1 fb4", 7, true, false},        {"r1 fb4 pf", 5, true, false},    {"r4 fb4 pf", 6, true, false},
                        {"m1 lds r2", 8, true, true},      {"m1 lds r1", 9, true, true}};
    auto run = [&](int id, int M, int c, float* o) {
        switch (id) {
            case 0: for (int m = 0; m < M; ++m) prod1(m, c, o + (size_t)m * N); break;
            case 1: rows(M, c, o); break;
            case 10: i8_by_m<2, 0, false>(M, 1, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 2: i8_by_m<2, 0, false>(M, 4, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 3: i8_by_m<2, 4, false>(M, 5, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 4: i8_by_m<2, 4, true>(M, 5, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 7: i8_by_m<1, 4, false>(M, 5, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 5: i8_by_m<1, 4, true>(M, 5, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 6: i8_by_m<4, 4, true>(M, 5, bufs, dw[c], ds[c], o, N, (int)K); break;
            case 8: m1_launch<2>(bufs, dw[c], ds[c], o, N, (int)K); break;
            case 9: m1_launch<1>(bufs, dw[c], ds[c], o, N, (int)K); break;
        }
    };
    printf("## mxr1 %ldx%ld iters=%d rounds=%d edge=%d (edge groups %d of %ld; ref outputs finite %ld inf %ld NaN %ld); us per call incl. pre-pass, min over rounds [max]; x = ratio to prod1 M=1; bitwise vs prod1 per row\n",
           (long)N, (long)K, iters, rounds, edge, nedge, (long)(MMAX * G), finref, infref, nanref);
    double t1 = 0;
    long total_bad = 0;
    for (const Arm& a : arms) {
        if ((a.id == 8 || a.id == 9) && K > 8192) continue;
        printf("%-14s", a.name);
        for (int M : Ms) {
            if (a.m1only && M > 1) { printf(" |"); continue; }
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
            if (a.exact || !edge) total_bad += bad;
            printf(" | M=%2d %7.1f [%7.1f] x%.2f %s", M, best, worst, t1 > 0 ? best / t1 : 0.0, bad ? ("MISMATCH " + std::to_string(bad)).c_str() : "ok");
        }
        printf("\n");
    }
    printf("## total mismatches %ld\n", total_bad);
    return total_bad ? 4 : 0;
}
