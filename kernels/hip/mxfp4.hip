// matmul_mxfp4: out[M,N] = q(x)[M,K] @ W^T with W[n,k] = e2m1(code) * 2^(scale[n,k/32] - 127) (OCP MXFP4, AMD Quark).
//   w     u8  [N, K/2]   two codes per byte, element 2i in the low nibble
//   scale u8  [N, K/32]  e8m0, plain row-major (checkpoint layout)
//   act   f32 [1]        0: q(x) = x (W4A16); 1: q(x) = dynamic MXFP4 QDQ of x per 32-element group along K (W4A4, the
//                        scheme Quark declares; scale rule "even", see crates/intarsia-core/src/quant/mxfp4.rs)
//   *_gemv  M <= 8: warp-per-output-row, one 32-element group (16 weight bytes + 1 scale byte + 32 activations) per
//           lane-iteration. With a scratch (the registered kernel declares one) the activations are QDQ'd once by the
//           pre-pass below; without, every warp QDQs the groups it reads (each group once per output row: N times).
//   *_mma   any M: the activations are QDQ'd (or copied) once into scratch (bf16 [M, K]), then the dequantise-to-bf16 WMMA
//           GEMM of ik_dq_gemm.h runs on them (e2m1 * 2^e is exact in bf16; f32 accumulate).
// Deterministic (fixed reduction order). Activations bf16; output bf16 or f32. Same structure as nvfp4.hip.
#include "ik_dq_gemm.h"

#include <stdio.h>

namespace {

constexpr int kWarps = 8;
constexpr int kG = 32;  // elements per e8m0 scale

// e8m0 byte -> 2^(b - 127); 0 -> 2^-127 (f32 subnormal), 255 -> NaN
IK_DEVI float mx_e8m0(uint32_t b) {
    return b == 0u ? __uint_as_float(0x00400000u) : (b == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(b << 23));
}

// Quark "even" shared scale of a group with largest magnitude amax (>= 0): biased e8m0 byte
IK_DEVI uint32_t mx_scale_even(float amax) {
    if (amax != amax) return 255u;
    const uint32_t r = (__float_as_uint(amax) + (1u << 21)) & 0x7F800000u;
    const int biased = (int)(r >> 23);
    if (biased == 0) return 0u;
    if (biased == 255) return 254u;
    int e = biased - 129;
    e = e < -127 ? -127 : (e > 127 ? 127 : e);
    return (uint32_t)(e + 127);
}

// round to the nearest e2m1 value, ties to the even code, saturating at +-6 (crates/intarsia-core quant::nvfp4::f32_to_fp4)
IK_DEVI float mx_fp4_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.f;
    else if (a < 3.5f) r = 3.f;
    else if (a <= 5.f) r = 4.f;
    else r = 6.f;
    return copysignf(r, v);
}

// QDQ one group of 32 activations in place
IK_DEVI void mx_qdq32(float* f) {
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < kG; ++e) amax = fmaxf(amax, fabsf(f[e]));
    const uint32_t sb = mx_scale_even(amax);
    const float s = mx_e8m0(sb), inv = mx_e8m0(254u - sb);  // 1 / 2^(sb-127) = 2^(127-sb), exact
#pragma unroll
    for (int e = 0; e < kG; ++e) f[e] = mx_fp4_round(f[e] * inv) * s;
}

// ---- GEMV --------------------------------------------------------------------------------------------------------

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                  const float* __restrict__ act, OutT* __restrict__ out, int64_t N, int K) {
    IK_FP4_LUT(lut);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;  // whole warp exits together
    const bool qdq = act != nullptr && act[0] != 0.f;  // null: x was QDQ'd (or copied) into scratch by the pre-pass
    const uint4* wr = reinterpret_cast<const uint4*>(w + n * (K / 2));
    const uint8_t* sr = sc + n * (K / kG);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int u = lane; u < K / kG; u += IK_WARP) {
        const uint4 wv = ik_ldcs(wr + u);
        const float s = mx_e8m0(sr[u]);
        float wf[kG];
        ik_fp4x8_to_f32(wv.x, lut, wf);
        ik_fp4x8_to_f32(wv.y, lut, wf + 8);
        ik_fp4x8_to_f32(wv.z, lut, wf + 16);
        ik_fp4x8_to_f32(wv.w, lut, wf + 24);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * kG);
            float xf[kG];
            ik_unpack8(xp[0], xf);
            ik_unpack8(xp[1], xf + 8);
            ik_unpack8(xp[2], xf + 16);
            ik_unpack8(xp[3], xf + 24);
            if (qdq) mx_qdq32(xf);
            float t = 0.f;
#pragma unroll
            for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
            acc[m] = fmaf(t, s, acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float t = ik_warp_sum(acc[m]);
        if (lane == 0) ik_store(out + (int64_t)m * N + n, t);
    }
}

// M == 1 (decode): the block stages x once in LDS (bf16 [K]), QDQ'd per 32-element group when act[0] != 0, so neither a
// pre-pass launch nor a per-row QDQ is needed. The grid is sized to the device (blocks per CU, INTARSIA_HIP_MXFP4_LDS)
// and each warp walks rows n, n + 8 * gridDim.x, ... so the staging is paid once per block, not once per 8 rows. Per
// output row the arithmetic is the one of mxfp4_gemv_kernel<1> on a pre-passed x (same lane-to-group map, same fma
// chain, same warp sum; the QDQ'd values are exact in bf16): bitwise equal.
template <typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_gemv1_lds_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, const uint8_t* __restrict__ sc0,
                       OutT* __restrict__ out0, int64_t N0, const uint8_t* __restrict__ w1, const uint8_t* __restrict__ sc1,
                       OutT* __restrict__ out1, int64_t N1, const float* __restrict__ act, int K) {
    // rows [0, N0) of (w0, sc0, out0), then rows [0, N1) of (w1, sc1, out1) (MatmulMxfp4Pair: both from one staged x)
    const int64_t N = N0 + N1;
    auto row = [&](int64_t r, const uint8_t*& w, const uint8_t*& sc, int64_t& rr) {
        if (r < N0) { w = w0; sc = sc0; rr = r; } else { w = w1; sc = sc1; rr = r - N0; }
    };
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* xs = reinterpret_cast<uint4*>(smem);  // [4][G] x 8 bf16: chunk i of group g at xs[i * G + g] (lanes read
                                                 // consecutive 16-byte words: no LDS bank conflicts)
    IK_FP4_LUT(lut);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t stride = (int64_t)gridDim.x * kWarps;
    int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    const int G = K / kG;
    const bool qdq = act[0] != 0.f;
    constexpr int UN = 8;  // groups per lane with their loads in flight together
    // the first batch of the first row is in flight while x is staged
    uint4 wv[UN];
    uint32_t sb[UN];
    if (n < N) {
        const uint8_t *w, *sc;
        int64_t rr;
        row(n, w, sc, rr);
        const uint4* wr = reinterpret_cast<const uint4*>(w + rr * (K / 2));
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            const int u = lane + j * IK_WARP;
            if (u < G) {
                wv[j] = ik_ldcs(wr + u);
                sb[j] = sc[rr * G + u];
            }
        }
    }
    const uint4* xg = reinterpret_cast<const uint4*>(x);
    for (int g = threadIdx.x; g < G; g += kWarps * IK_WARP) {
        uint4 v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = xg[g * 4 + i];
        if (qdq) {
            float f[kG];
#pragma unroll
            for (int i = 0; i < 4; ++i) ik_unpack8(v[i], f + 8 * i);
            mx_qdq32(f);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = ik_pack8(f + 8 * i);  // exact: e2m1 * 2^e fits bf16
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) xs[i * G + g] = v[i];
    }
    __syncthreads();
    bool first = true;
    for (; n < N; n += stride) {  // whole warp walks the same rows
        const uint8_t *w, *sc;
        int64_t rr;
        row(n, w, sc, rr);
        const uint4* wr = reinterpret_cast<const uint4*>(w + rr * (K / 2));
        const uint8_t* sr = sc + rr * G;
        float acc = 0.f;
        for (int base = lane; base < G; base += UN * IK_WARP) {
#pragma unroll
            for (int j = 0; j < UN && !first; ++j) {
                const int u = base + j * IK_WARP;
                if (u < G) {
                    wv[j] = ik_ldcs(wr + u);
                    sb[j] = sr[u];
                }
            }
            first = false;
#pragma unroll
            for (int j = 0; j < UN; ++j) {
                const int u = base + j * IK_WARP;
                if (u < G) {  // same group order per lane as mxfp4_gemv_kernel
                    const float s = mx_e8m0(sb[j]);
                    float wf[kG];
                    ik_fp4x8_to_f32(wv[j].x, lut, wf);
                    ik_fp4x8_to_f32(wv[j].y, lut, wf + 8);
                    ik_fp4x8_to_f32(wv[j].z, lut, wf + 16);
                    ik_fp4x8_to_f32(wv[j].w, lut, wf + 24);
                    float xf[kG];
                    ik_unpack8(xs[u], xf);
                    ik_unpack8(xs[G + u], xf + 8);
                    ik_unpack8(xs[2 * G + u], xf + 16);
                    ik_unpack8(xs[3 * G + u], xf + 24);
                    float t = 0.f;
#pragma unroll
                    for (int e = 0; e < kG; ++e) t = fmaf(wf[e], xf[e], t);
                    acc = fmaf(t, s, acc);
                }
            }
        }
        const float t = ik_warp_sum(acc);
        if (lane == 0) ik_store((n < N0 ? out0 : out1) + rr, t);
    }
}

constexpr int64_t kLdsMax = 16 * 1024;  // x of K <= 8192 in LDS; a larger x (down_proj, K = 17408) costs occupancy:
                                        // the pre-pass + plain GEMV is faster there (measured on gfx1201)

// INTARSIA_HIP_MXFP4_LDS=<blocks per CU> for the M == 1 LDS kernel (default 4); 0: the pre-pass + plain GEMV path
static int mx_lds_blocks() {
    static const int v = [] {
        int per = 4;
        if (const char* e = getenv("INTARSIA_HIP_MXFP4_LDS")) per = atoi(e);
        if (per <= 0) return 0;
        int dev = 0, cus = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || cus <= 0)
            cus = 64;
        return per * cus;
    }();
    return v;
}

template <typename OutT>
int lds_launch(const ik_bf16* x, const uint8_t* w0, const uint8_t* s0, OutT* o0, int64_t N0, const uint8_t* w1,
               const uint8_t* s1, OutT* o1, int64_t N1, const float* act, int64_t K, hipStream_t s) {
    const int64_t full = (N0 + N1 + kWarps - 1) / kWarps;
    const unsigned grid = (unsigned)(full < mx_lds_blocks() ? full : mx_lds_blocks());
    mxfp4_gemv1_lds_kernel<OutT><<<grid, kWarps * IK_WARP, (size_t)K * 2, s>>>(x, w0, s0, o0, N0, w1, s1, o1, N1, act, (int)K);
    return ik_launch_status();
}

template <typename OutT>
int gemv_by_m(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, const float* act, OutT* out, int64_t N,
              int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
#define IK_MX_GEMV(MM) mxfp4_gemv_kernel<MM, OutT><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, sc, act, out, N, (int)K)
    switch (M) {
        case 1: IK_MX_GEMV(1); break;
        case 2: IK_MX_GEMV(2); break;
        case 3: IK_MX_GEMV(3); break;
        case 4: IK_MX_GEMV(4); break;
        case 5: IK_MX_GEMV(5); break;
        case 6: IK_MX_GEMV(6); break;
        case 7: IK_MX_GEMV(7); break;
        case 8: IK_MX_GEMV(8); break;
        default: return IK_ERR_SHAPE;
    }
#undef IK_MX_GEMV
    return ik_launch_status();
}

// ---- row-exact multi-row GEMV (_rows launchers, M = 2..16) ------------------------------------------------------------
// Output row m of an M-row call is bitwise the M == 1 launch on row m alone: per (output row n, activation row m) the lane
// walks the groups u = lane, lane + 32, ... in order, t = fma chain over the 32 elements of the group, acc = fma(t, s, acc),
// then the xor-butterfly warp sum, exactly as mxfp4_gemv_kernel<1>. What differs from mxfp4_gemv_kernel<M> is the register
// shape only. That kernel hoists the loads and bf16 unpacks of all M activation rows (~33 VGPR per row) and spills from
// M = 6 (256 VGPR, 126 spilled at M = 8: 19x the one-row time). Here a warp computes kRowsR = 2 weight rows, so the
// activation loads and unpacks of a group are shared by two rows, and each activation row is loaded, unpacked and consumed
// 8 elements at a time behind a scheduler barrier, so the registers do not grow with M (104 VGPR at M = 2, 134 at M = 16).
// x is the QDQ'd (or copied) pre-pass output, plain bf16.
constexpr int kRowsR = 2;  // weight rows per warp (measured against 1, 3 and 4: docs/results/2026-10-25-mr-hip-mxfp4-rows.md)

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps * IK_WARP)
mxfp4_gemv_rows_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                       OutT* __restrict__ out, int64_t N, int K) {
    constexpr int R = kRowsR;
    IK_FP4_LUT(lut);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n0 = ((int64_t)blockIdx.x * kWarps + warp) * R;
    if (n0 >= N) return;  // whole warp exits together
    const int G = K / kG;
    const uint4* wr[R];
    const uint8_t* sr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < N ? n0 + r : N - 1;  // a tail row past N re-reads the last row and is not stored
        wr[r] = reinterpret_cast<const uint4*>(w + n * (K / 2));
        sr[r] = sc + n * G;
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    for (int u = lane; u < G; u += IK_WARP) {
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
        for (int m = 0; m < M; ++m) {
            __builtin_amdgcn_sched_barrier(0);  // keeps the loads of row m + 1 from being hoisted above the fmas of row m
            const uint4* xp = reinterpret_cast<const uint4*>(x + (int64_t)m * K + (int64_t)u * kG);
            float t[R];
#pragma unroll
            for (int r = 0; r < R; ++r) t[r] = 0.f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float xf[8];
                ik_unpack8(xp[c], xf);
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int e = 0; e < 8; ++e) t[r] = fmaf(wf[r][8 * c + e], xf[e], t[r]);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r][m] = fmaf(t[r], s[r], acc[r][m]);
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

template <int M, typename OutT>
void rows_launch(const ik_bf16* x, const uint8_t* w, const uint8_t* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps * kRowsR - 1) / (kWarps * kRowsR));
    mxfp4_gemv_rows_kernel<M, OutT><<<grid, kWarps * IK_WARP, 0, s>>>(x, w, sc, out, N, (int)K);
}

template <typename OutT>
int rows_by_m(int M, const ik_bf16* x, const uint8_t* w, const uint8_t* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
#define IK_MX_ROWS(MM) case MM: rows_launch<MM, OutT>(x, w, sc, out, N, K, s); break
    switch (M) {
        IK_MX_ROWS(2); IK_MX_ROWS(3); IK_MX_ROWS(4); IK_MX_ROWS(5); IK_MX_ROWS(6); IK_MX_ROWS(7); IK_MX_ROWS(8);
        IK_MX_ROWS(9); IK_MX_ROWS(10); IK_MX_ROWS(11); IK_MX_ROWS(12); IK_MX_ROWS(13); IK_MX_ROWS(14); IK_MX_ROWS(15);
        IK_MX_ROWS(16);
        default: return IK_ERR_SHAPE;
    }
#undef IK_MX_ROWS
    return ik_launch_status();
}

// ---- activation pre-pass (mma): xs[M, K] = act ? QDQ(x) : x, one thread per 32-element group --------------------

__global__ void mxfp4_xprep_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act, ik_bf16* __restrict__ xs,
                                   int64_t groups) {
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const uint4* xp = reinterpret_cast<const uint4*>(x) + g * 4;
    uint4* dp = reinterpret_cast<uint4*>(xs) + g * 4;
    if (act[0] == 0.f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) dp[i] = xp[i];
        return;
    }
    float f[kG];
#pragma unroll
    for (int i = 0; i < 4; ++i) ik_unpack8(xp[i], f + 8 * i);
    mx_qdq32(f);
#pragma unroll
    for (int i = 0; i < 4; ++i) dp[i] = ik_pack8(f + 8 * i);  // exact: e2m1 * 2^e fits bf16
}

// ---- MMA policy ---------------------------------------------------------------------------------------------------

struct Mxfp4Policy {
    IK_DEVI bool skip(const DqArgs&) { return false; }
    struct State {
        const float* lut;
    };
    struct Raw {
        uint2 w;     // 16 fp4 codes
        uint32_t s;  // e8m0 group scale
    };
    IK_DEVI State init(const DqArgs&) {
        IK_FP4_LUT(lut);
        State s;
        s.lut = lut;
        return s;
    }
    IK_DEVI Raw load(const DqArgs& a, int64_t n, int k) {
        Raw r;
        r.w = *reinterpret_cast<const uint2*>(a.w + n * (a.K / 2) + k / 2);
        r.s = a.bs[n * (a.K / kG) + k / kG];
        return r;
    }
    IK_DEVI Raw zero() {
        Raw r;
        r.w = make_uint2(0, 0);
        r.s = 127;  // 1.0, times zero codes
        return r;
    }
    IK_DEVI void dequant(const State& st, const Raw& r, uint4& lo, uint4& hi) {
        const float s = mx_e8m0(r.s);
        float f[16];
        ik_fp4x8_to_f32(r.w.x, st.lut, f);
        ik_fp4x8_to_f32(r.w.y, st.lut, f + 8);
#pragma unroll
        for (int e = 0; e < 16; ++e) f[e] *= s;  // exact: e2m1 * 2^e
        lo = ik_pack8(f);
        hi = ik_pack8(f + 8);
    }
    IK_DEVI uint4 xprep(const State&, uint4 v) { return v; }
    IK_DEVI float out_scale(const State&) { return 1.f; }
};

// ---- matmul_mxfp4_mma v2 (lane mx): the same bits as Mxfp4Policy + dq_gemm_run (v1), faster ---------------------------
//
// Bitwise by construction: every output is the same chain of bf16 WMMAs (k steps of 16 in order from k = 0, acc from 0)
// over the same bf16 operands; only the tile (which outputs a block owns) and the way the bf16 weights are built change.
// tests/mxfp4_v2.rs checks v2 against v1 bit for bit (shapes, tiles, every scale byte), scripts/mx/mxbench.hip too.
//
// Dequant without the LDS table and the f32 multiply / bf16 rounding: for a scale byte s in [3, 250] every e2m1 * 2^(s-127)
// is a normal f32 with a 2-bit significand, so its bf16 is exact and is built directly: magnitude code m >= 2 ->
// (m << 6) + ((s - 1) << 7), m == 1 (0.5) -> (s - 1) << 7, m == 0 -> 0, sign bit from the code. Other scale bytes (2^-127
// subnormals, NaN at 255) take v1's path.
struct Mxfp4PolicyV2 : Mxfp4Policy {
    IK_DEVI static uint32_t bf2(uint32_t c2, uint32_t e) {  // two codes (bits 0-3, 16-19) -> two bf16 in one dword
        const uint32_t m = c2 & 0x00070007u;
        const uint32_t lo = m & 0xffffu, hi = m >> 16;
        const uint32_t vlo = lo >= 2u ? (lo << 6) + e : (lo == 1u ? e : 0u);
        const uint32_t vhi = hi >= 2u ? (hi << 6) + e : (hi == 1u ? e : 0u);
        return (vlo | (vhi << 16)) | ((c2 & 0x00080008u) << 12);
    }
    IK_DEVI void dequant(const State& st, const Raw& r, uint4& lo, uint4& hi) {
        if (r.s < 3u || r.s > 250u) {
            Mxfp4Policy::dequant(st, r, lo, hi);
            return;
        }
        const uint32_t e = (r.s - 1u) << 7;
        // element j of a word is nibble j; output dword q holds elements 2q (low half) and 2q + 1 (high half)
        const uint32_t w0 = r.w.x, w1 = r.w.y;
        const uint32_t a0 = (w0 & 0xfu) | ((w0 << 12) & 0xf0000u), a1 = ((w0 >> 8) & 0xfu) | ((w0 << 4) & 0xf0000u);
        const uint32_t a2 = ((w0 >> 16) & 0xfu) | ((w0 >> 4) & 0xf0000u), a3 = ((w0 >> 24) & 0xfu) | ((w0 >> 12) & 0xf0000u);
        const uint32_t b0 = (w1 & 0xfu) | ((w1 << 12) & 0xf0000u), b1 = ((w1 >> 8) & 0xfu) | ((w1 << 4) & 0xf0000u);
        const uint32_t b2 = ((w1 >> 16) & 0xfu) | ((w1 >> 4) & 0xf0000u), b3 = ((w1 >> 24) & 0xfu) | ((w1 >> 12) & 0xf0000u);
        lo = make_uint4(bf2(a0, e), bf2(a1, e), bf2(a2, e), bf2(a3, e));
        hi = make_uint4(bf2(b0, e), bf2(b1, e), bf2(b2, e), bf2(b3, e));
    }
};

// Tile override for sweeps: INTARSIA_HIP_MXFP4_TILE=WN,WM,TN,TM (read once; any listed tile gives the same bits). The
// bench (scripts/mx/mxbench.hip, -DIK_MX_BENCH) sets it per call through ik_mxfp4_bench_set().
static int g_mx_tile[4] = {-1, 0, 0, 0};
static int g_mx_v1 = -1;  // INTARSIA_HIP_MXFP4_V1=1: matmul_mxfp4_mma runs v1 (kill switch)
static void mx_cfg_init() {
    if (g_mx_tile[0] == -1) {
        g_mx_tile[0] = 0;
        const char* e = getenv("INTARSIA_HIP_MXFP4_TILE");
        if (e && sscanf(e, "%d,%d,%d,%d", &g_mx_tile[0], &g_mx_tile[1], &g_mx_tile[2], &g_mx_tile[3]) != 4) g_mx_tile[0] = 0;
    }
    if (g_mx_v1 == -1) {
        const char* e = getenv("INTARSIA_HIP_MXFP4_V1");
        g_mx_v1 = (e && e[0] == '1') ? 1 : 0;
    }
}

template <class P, typename OutT>
void mx_gemm_run(const DqArgs& a, OutT* out, hipStream_t s) {
    const int64_t M = a.M, N = a.N;
#define IK_MX_TILE(WN, WM, TN, TM)                                                                                          \
    {                                                                                                                      \
        dim3 grid((unsigned)((M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((N + WN * TN * 16 - 1) / (WN * TN * 16))); \
        dq_gemm_kernel<P, WN, WM, TN, TM, OutT><<<grid, WN * WM * 32, 0, s>>>(a, out);                                     \
        return;                                                                                                            \
    }
#define IK_MX_IS(A, B, C, D) (t[0] == A && t[1] == B && t[2] == C && t[3] == D)
    const int* t = g_mx_tile;
    if (t[0] > 0) {
        if (IK_MX_IS(4, 2, 2, 4)) IK_MX_TILE(4, 2, 2, 4)
        if (IK_MX_IS(4, 4, 4, 2)) IK_MX_TILE(4, 4, 4, 2)
        if (IK_MX_IS(4, 4, 2, 4)) IK_MX_TILE(4, 4, 2, 4)
        if (IK_MX_IS(2, 2, 4, 4)) IK_MX_TILE(2, 2, 4, 4)
        if (IK_MX_IS(4, 2, 4, 4)) IK_MX_TILE(4, 2, 4, 4)
        if (IK_MX_IS(2, 4, 4, 4)) IK_MX_TILE(2, 4, 4, 4)
        if (IK_MX_IS(4, 4, 2, 2)) IK_MX_TILE(4, 4, 2, 2)
        if (IK_MX_IS(2, 4, 4, 2)) IK_MX_TILE(2, 4, 4, 2)
    }
#undef IK_MX_IS
    if (M <= 16) IK_MX_TILE(4, 1, 2, 1)
    if (M <= 32) IK_MX_TILE(4, 1, 2, 2)
    if (M <= 64) IK_MX_TILE(2, 2, 4, 2)
    IK_MX_TILE(4, 2, 2, 4)
#undef IK_MX_TILE
}

// ---- launchers ---------------------------------------------------------------------------------------------------

int launch(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, const ik_tensor* act, ik_tensor* out,
           const ik_scratch* scratch, void* stream, bool gemv, bool rows = false, bool v1 = false) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(sc, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % kG != 0 || w->shape[1] != K / 2 || sc->shape[0] != N || sc->shape[1] != K / kG || act->shape[0] != 1 ||
        out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (K == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), IK_STREAM(stream)) == hipSuccess
                   ? IK_OK
                   : IK_ERR_LAUNCH;
    if (M > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* sp = (const uint8_t*)sc->data;
    const float* ap = (const float*)act->data;
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (gemv) {
        if (M > (rows ? 16 : 8)) return IK_ERR_SHAPE;
        if (M == 1 && K * 2 <= kLdsMax && mx_lds_blocks() > 0) {
            return f32o ? lds_launch(xp, wp, sp, (float*)out->data, N, (const uint8_t*)nullptr, (const uint8_t*)nullptr,
                                     (float*)nullptr, 0, ap, K, s)
                        : lds_launch(xp, wp, sp, (ik_bf16*)out->data, N, (const uint8_t*)nullptr, (const uint8_t*)nullptr,
                                     (ik_bf16*)nullptr, 0, ap, K, s);
        }
        if (scratch && scratch->data && scratch->bytes >= M * K * 2 && !(reinterpret_cast<uintptr_t>(scratch->data) & 15u)) {
            // QDQ (or copy) the activations once, then the GEMV reads them as plain bf16
            ik_bf16* xs = (ik_bf16*)scratch->data;
            const int64_t groups = M * K / kG;
            mxfp4_xprep_kernel<<<(unsigned)((groups + 255) / 256), 256, 0, s>>>(xp, ap, xs, groups);
            IK_TRY(ik_launch_status());
            xp = xs;
            ap = nullptr;
        }
        if (rows && M >= 2) {  // register-shaped kernels: x must be the plain pre-pass output
            if (!(scratch && scratch->data && scratch->bytes >= M * K * 2 && !(reinterpret_cast<uintptr_t>(scratch->data) & 15u)))
                return IK_ERR_INVALID_ARG;
            return f32o ? rows_by_m((int)M, xp, wp, sp, (float*)out->data, N, K, s)
                        : rows_by_m((int)M, xp, wp, sp, (ik_bf16*)out->data, N, K, s);
        }
        return f32o ? gemv_by_m((int)M, xp, wp, sp, ap, (float*)out->data, N, K, s)
                    : gemv_by_m((int)M, xp, wp, sp, ap, (ik_bf16*)out->data, N, K, s);
    }
    if (!scratch || !scratch->data || scratch->bytes < M * K * 2) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    ik_bf16* xs = (ik_bf16*)scratch->data;
    const int64_t groups = M * K / kG;
    mxfp4_xprep_kernel<<<(unsigned)((groups + 255) / 256), 256, 0, s>>>(xp, ap, xs, groups);
    IK_TRY(ik_launch_status());
    DqArgs a;
    a.x = xs;
    a.w = wp;
    a.bs = sp;
    a.sc = nullptr;
    a.xq = nullptr;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    mx_cfg_init();
    if (v1 || g_mx_v1) {
        if (f32o) dq_gemm_run<Mxfp4Policy>(a, (float*)out->data, s);
        else dq_gemm_run<Mxfp4Policy>(a, (ik_bf16*)out->data, s);
    } else {
        if (f32o) mx_gemm_run<Mxfp4PolicyV2>(a, (float*)out->data, s);
        else mx_gemm_run<Mxfp4PolicyV2>(a, (ik_bf16*)out->data, s);
    }
    return ik_launch_status();
}

// ---- matmul_mxfp4_mma_fold (lane mx): OPT-IN (INTARSIA_HIP_MXFP4_FOLD=1), changes numerics -----------------------
//
// W4A4 MXFP4 on the fp8 x fp8 WMMA at the speed of ik_f8_gemm.h's W8A8 GEMM. Both operands are folded into e4m3 relative to
// their row's largest group scale: weight row n with ref Wn = max scale byte of the row, activation row m with ref Am = max
// scale byte over its nonzero groups (after today's QDQ, mx_qdq32). A code of magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} in a
// group with scale byte s becomes e4m3(mag * 2^t), t = s - ref + 6, which is EXACT for t in [-8, 6] (a row's scales spanning
// at most 14 octaves; the 27B's weights span at most 10, scripts/mx/scales.py) and rounds to nearest even below (tiny groups
// of activation rows with a wider span). Then out = acc * 2^(Wn + Am - 266), one fp8 GEMM over the whole K.
// Against matmul_mxfp4_mma: the products are the same exact values; the sums differ in the last bits (the fp8 WMMA adds
// in its own order and rounding, scripts/mx/wmma_eq.hip) and the rounded tiny groups. Opt-in only, gated on the 27B KL
// goldens (docs/results/2026-10-25-mx-hip-27b-prefill.md). W4A16 (act == 0) runs matmul_mxfp4_mma v2 inside.
// Fragments and k layout as ik_f8_gemm.h (the k order inside a 32-k pair is permuted the same way for both operands).

typedef int mxf8_v2i __attribute__((ext_vector_type(2)));
IK_DEVI ik_dq_v8f mxf8_wmma(uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi, ik_dq_v8f c) {
    const uint2 a = make_uint2(alo, ahi), b = make_uint2(blo, bhi);
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(mxf8_v2i, a), __builtin_bit_cast(mxf8_v2i, b),
                                                                c);
}

// byte m of row t + 8 = e4m3(mag[m] * 2^t), t = -8 .. 6 (exact; checked when written, scripts/mx/fold_table.py)
__constant__ uint32_t kMxFoldTab[15][2] = {
    {0x03020100u, 0x0c080604u}, {0x06040200u, 0x14100c08u}, {0x0c080400u, 0x1c181410u}, {0x14100800u, 0x24201c18u},
    {0x1c181000u, 0x2c282420u}, {0x24201800u, 0x34302c28u}, {0x2c282000u, 0x3c383430u}, {0x34302800u, 0x44403c38u},
    {0x3c383000u, 0x4c484440u}, {0x44403800u, 0x54504c48u}, {0x4c484000u, 0x5c585450u}, {0x54504800u, 0x64605c58u},
    {0x5c585000u, 0x6c686460u}, {0x64605800u, 0x74706c68u}, {0x6c686000u, 0x7c787470u},
};

// the 8-byte magnitude table for shift t: exact rows for -8 <= t <= 6, else e4m3 RNE of each magnitude (rare)
IK_DEVI void mx_fold_tab(int t, uint32_t& t0, uint32_t& t1) {
    if (t >= -8 && t <= 6) {
        t0 = kMxFoldTab[t + 8][0];
        t1 = kMxFoldTab[t + 8][1];
        return;
    }
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    uint32_t b[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) b[m] = ik_f_to_e4m3(ldexpf(mag[m], t));
    t0 = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
    t1 = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24);
}

// 8 e2m1 codes (element j = nibble j) -> 8 e4m3 bytes (element j = byte j) through the magnitude table, sign bit 3 -> 7
IK_DEVI uint2 mx_fold8(uint32_t w, uint32_t t0, uint32_t t1) {
    const uint32_t ev = w & 0x0F0F0F0Fu, od = (w >> 4) & 0x0F0F0F0Fu;
    const uint32_t be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
    const uint32_t bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
    return make_uint2(__builtin_amdgcn_perm(bo, be, 0x05010400u), __builtin_amdgcn_perm(bo, be, 0x07030602u));
}

// weight row refs: wref[n] = max scale byte of row n (one thread per row)
__global__ void mx_fold_wref_kernel(const uint8_t* __restrict__ ws, uint8_t* __restrict__ wref, int64_t N, int KG) {
    const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    uint32_t mx = 0;
    const uint8_t* p = ws + n * KG;
    for (int g = 0; g < KG; ++g) mx = max(mx, (uint32_t)p[g]);
    wref[n] = (uint8_t)mx;
}

// activations, one block per row: act != 0 -> xc[m, :] = folded e4m3 of mx_qdq32's codes, aref[m]; act == 0 -> xs = x (bf16)
constexpr int kFoldMaxG = 1024;  // groups per row (K <= 32768)
__global__ void __launch_bounds__(256) mx_fold_actq_kernel(const ik_bf16* __restrict__ x, const float* __restrict__ act,
                                                           uint8_t* __restrict__ xc, uint8_t* __restrict__ aref,
                                                           ik_bf16* __restrict__ xs, int K) {
    const int64_t row = blockIdx.x;
    const int G = K / kG, tid = threadIdx.x;
    const uint4* xp = reinterpret_cast<const uint4*>(x + row * K);
    if (act[0] == 0.f) {
        uint4* dp = reinterpret_cast<uint4*>(xs + row * K);
        for (int i = tid; i < K / 8; i += blockDim.x) dp[i] = xp[i];
        return;
    }
    __shared__ uint8_t sb_s[kFoldMaxG];
    __shared__ uint32_t red[8];
    uint32_t mymax = 0;
    for (int g = tid; g < G; g += blockDim.x) {
        float f[kG];
#pragma unroll
        for (int i = 0; i < 4; ++i) ik_unpack8(xp[g * 4 + i], f + 8 * i);
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < kG; ++e) amax = fmaxf(amax, fabsf(f[e]));
        const uint32_t sb = mx_scale_even(amax);
        sb_s[g] = (uint8_t)sb;
        if (amax > 0.f && sb < 255u) mymax = max(mymax, sb);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mymax = max(mymax, (uint32_t)__shfl_xor((int)mymax, o, 32));
    if ((tid & 31) == 0) red[tid >> 5] = mymax;
    __syncthreads();
    uint32_t ar = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) ar = max(ar, red[i]);
    if (ar == 0u) ar = 127u;  // all-zero row: any ref
    if (tid == 0) aref[row] = (uint8_t)ar;
    for (int g = tid; g < G; g += blockDim.x) {
        float f[kG];
#pragma unroll
        for (int i = 0; i < 4; ++i) ik_unpack8(xp[g * 4 + i], f + 8 * i);
        const uint32_t sb = sb_s[g];
        const float inv = mx_e8m0(254u - sb);
        const int t = (int)sb - (int)ar + 6;
        uint32_t d[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            uint32_t v = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) v |= (uint32_t)ik_f_to_e4m3(ldexpf(mx_fp4_round(f[q * 4 + b] * inv), t)) << (8 * b);
            d[q] = v;
        }
        uint4* dp = reinterpret_cast<uint4*>(xc + row * K + g * kG);
        dp[0] = make_uint4(d[0], d[1], d[2], d[3]);
        dp[1] = make_uint4(d[4], d[5], d[6], d[7]);
    }
}

constexpr int kFoldBK = 64;           // k per stage: two groups
constexpr int kFoldLD = kFoldBK + 16;  // LDS row stride in bytes (ik_f8_gemm.h's kF8LD)

template <int WN, int WM, int TN, int TM, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32)
mx_fold_gemm_kernel(const uint8_t* __restrict__ xc, const uint8_t* __restrict__ aref, const uint8_t* __restrict__ w,
                    const uint8_t* __restrict__ ws, const uint8_t* __restrict__ wref, const float* __restrict__ act,
                    OutT* __restrict__ out, int M, int64_t N, int K, int vec_out) {
    if (act[0] == 0.f) return;  // W4A16: the bf16 kernel does this launch
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int CHW = BN * 2, CHX = BM * 4;  // 16-byte chunks of one stage: weights one group (32 codes), x 16 bytes
    constexpr int CW = (CHW + NT - 1) / NT, CX = (CHX + NT - 1) / NT;
    __shared__ __align__(16) uint8_t sw[2][BN * kFoldLD];
    __shared__ __align__(16) uint8_t sx[2][BM * kFoldLD];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wn = wave % WN, wm = wave / WN;
    const int m_blk = blockIdx.x * BM;
    const int64_t n_blk = (int64_t)blockIdx.y * BN;
    const int nk = K / kFoldBK, KG = K / kG;

    uint4 rw[CW], rx[CX];
    int rt[CW];  // fold shift of the chunk's group
    int wr[CW];
#pragma unroll
    for (int c = 0; c < CW; ++c) {
        const int64_t n = n_blk + ((tid + c * NT) >> 1);
        wr[c] = (n < N) ? (int)wref[n] : 127;
    }
    auto gload = [&](int k0) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT, r = idx >> 1, ch = idx & 1;
            const int64_t n = n_blk + r;
            if (idx < CHW && n < N) {
                rw[c] = *reinterpret_cast<const uint4*>(w + n * (K / 2) + (k0 + ch * kG) / 2);
                const uint32_t s = ws[n * KG + k0 / kG + ch];
                rt[c] = s == 255u ? 1000 : (int)s - wr[c] + 6;
            } else {
                rw[c] = make_uint4(0, 0, 0, 0);
                rt[c] = 0;
            }
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int m = m_blk + r;
            rx[c] = (idx < CHX && m < M) ? *reinterpret_cast<const uint4*>(xc + (int64_t)m * K + k0 + ch * 16) : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) {
                const int r = idx >> 1, ch = idx & 1;
                uint32_t t0, t1;
                if (rt[c] == 1000) {
                    t0 = t1 = 0x7f7f7f7fu;  // scale byte 255: NaN
                } else {
                    mx_fold_tab(rt[c], t0, t1);
                }
                const uint2 a = mx_fold8(rw[c].x, t0, t1), b = mx_fold8(rw[c].y, t0, t1);
                const uint2 cc = mx_fold8(rw[c].z, t0, t1), d = mx_fold8(rw[c].w, t0, t1);
                uint8_t* p = &sw[buf][r * kFoldLD + ch * kG];
                *reinterpret_cast<uint4*>(p) = make_uint4(a.x, a.y, b.x, b.y);
                *reinterpret_cast<uint4*>(p + 16) = make_uint4(cc.x, cc.y, d.x, d.y);
            }
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHX) *reinterpret_cast<uint4*>(&sx[buf][(idx >> 2) * kFoldLD + (idx & 3) * 16]) = rx[c];
        }
    };

    ik_dq_v8f acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    gload(0);
    sstore(0);
    __syncthreads();
    const int lr = lane & 15, lh = (lane >> 4) * 16;
    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) gload((kt + 1) * kFoldBK);
#pragma unroll
        for (int kk = 0; kk < kFoldBK; kk += 32) {
            uint4 af[TN], bf[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i)
                af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kFoldLD + kk + lh]);
#pragma unroll
            for (int j = 0; j < TM; ++j)
                bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kFoldLD + kk + lh]);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = mxf8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
                    acc[i][j] = mxf8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
                }
        }
        if (kt + 1 < nk) sstore(buf ^ 1);
        __syncthreads();
    }

    const int ln = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m_blk + wm * TM * 16 + j * 16 + lr;
            const int64_t n = n_blk + wn * TN * 16 + i * 16 + ln;
            if (m >= M || n >= N) continue;
            const int ea = (int)aref[m] - 266;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = ldexpf(acc[i][j][e], ea + (n + e < N ? (int)wref[n + e] : 0));
            OutT* o = out + (int64_t)m * N + n;
            if (vec_out) {
                if (sizeof(OutT) == 2) {
                    *reinterpret_cast<uint4*>(o) = ik_pack8(v);
                } else {
                    reinterpret_cast<float4*>(o)[0] = make_float4(v[0], v[1], v[2], v[3]);
                    reinterpret_cast<float4*>(o)[1] = make_float4(v[4], v[5], v[6], v[7]);
                }
            } else {
                for (int e = 0; e < 8 && n + e < N; ++e) ik_store(o + e, v[e]);
            }
        }
    }
}

template <typename OutT>
void mx_fold_run(const uint8_t* xc, const uint8_t* aref, const uint8_t* w, const uint8_t* ws, const uint8_t* wref,
                 const float* act, OutT* out, int M, int64_t N, int K, int vec_out, hipStream_t s) {
#define IK_FOLD_TILE(WN, WM, TN, TM)                                                                                      \
    {                                                                                                                    \
        dim3 grid((unsigned)(((int64_t)M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((N + WN * TN * 16 - 1) / (WN * TN * 16))); \
        mx_fold_gemm_kernel<WN, WM, TN, TM, OutT><<<grid, WN * WM * 32, 0, s>>>(xc, aref, w, ws, wref, act, out, M, N, K, vec_out); \
        return;                                                                                                          \
    }
    const int* t = g_mx_tile;
    if (t[0] > 0) {
        if (t[0] == 4 && t[1] == 4 && t[2] == 4 && t[3] == 2) IK_FOLD_TILE(4, 4, 4, 2)
        if (t[0] == 4 && t[1] == 4 && t[2] == 2 && t[3] == 4) IK_FOLD_TILE(4, 4, 2, 4)
        if (t[0] == 4 && t[1] == 2 && t[2] == 2 && t[3] == 4) IK_FOLD_TILE(4, 2, 2, 4)
        if (t[0] == 2 && t[1] == 2 && t[2] == 4 && t[3] == 4) IK_FOLD_TILE(2, 2, 4, 4)
    }
    // ik_f8_gemm.h's f8_gemm_run table (weights x tokens)
    if (M <= 64) IK_FOLD_TILE(4, 1, 2, 2)
    if (M >= 2048 && N >= 8192) IK_FOLD_TILE(4, 4, 4, 2)
    if (M >= 2048 && ((N + 127) / 128) * (((int64_t)M + 255) / 256) >= 64) IK_FOLD_TILE(4, 4, 2, 4)
    IK_FOLD_TILE(4, 2, 2, 4)
#undef IK_FOLD_TILE
}

// matmul_mxfp4_mma's W4A16 half for the fold launch: exits when act != 0
struct Mxfp4PolicyV2A16 : Mxfp4PolicyV2 {
    IK_DEVI bool skip(const DqArgs& a) { return a.sc[0] != 0.f; }
};

// scratch: [0, 2MK) the bf16 copy (W4A16) or the folded codes [0, MK) (W4A4); then aref [M], wref [N]
int launch_fold(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, const ik_tensor* act, ik_tensor* out,
                const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(sc, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % kFoldBK != 0 || K / kG > kFoldMaxG || M == 0 || N == 0) return launch(x, w, sc, act, out, scratch, stream, false);
    if (w->shape[1] != K / 2 || sc->shape[0] != N || sc->shape[1] != K / kG || act->shape[0] != 1 || out->shape[0] != M ||
        out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (!scratch || !scratch->data || scratch->bytes < M * K * 2 + M + N) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    hipStream_t s = IK_STREAM(stream);
    uint8_t* xc = (uint8_t*)scratch->data;
    uint8_t* aref = xc + M * K * 2;
    uint8_t* wref = aref + M;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* sp = (const uint8_t*)sc->data;
    const float* ap = (const float*)act->data;
    mx_cfg_init();
    mx_fold_wref_kernel<<<(unsigned)((N + 255) / 256), 256, 0, s>>>(sp, wref, N, (int)(K / kG));
    IK_TRY(ik_launch_status());
    mx_fold_actq_kernel<<<(unsigned)M, 256, 0, s>>>((const ik_bf16*)x->data, ap, xc, aref, (ik_bf16*)xc, (int)K);
    IK_TRY(ik_launch_status());
    const int vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    const bool f32o = out->dtype == IK_DTYPE_F32;
    if (f32o) mx_fold_run<float>(xc, aref, wp, sp, wref, ap, (float*)out->data, (int)M, N, (int)K, vec_out, s);
    else mx_fold_run<ik_bf16>(xc, aref, wp, sp, wref, ap, (ik_bf16*)out->data, (int)M, N, (int)K, vec_out, s);
    IK_TRY(ik_launch_status());
    DqArgs a;
    a.x = (const ik_bf16*)xc;
    a.w = wp;
    a.bs = sp;
    a.sc = ap;
    a.xq = nullptr;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = vec_out;
    if (f32o) mx_gemm_run<Mxfp4PolicyV2A16>(a, (float*)out->data, s);
    else mx_gemm_run<Mxfp4PolicyV2A16>(a, (ik_bf16*)out->data, s);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_mxfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                         ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return launch(x, w, scales, act, out, scratch, stream, true);
}
// MatmulMxfp4Pair: out0 = matmul_mxfp4(x, w0, s0, act), out1 = matmul_mxfp4(x, w1, s1, act). M == 1 and K <= 8192: one
// launch of the LDS kernel over the rows of both (each row bitwise as in the single op); otherwise the two single ops.
static int pair_impl(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1,
                     const ik_tensor* s1, const ik_tensor* act, ik_tensor* out0, ik_tensor* out1,
                     const ik_scratch* scratch, void* stream, bool rows) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w0, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(w1, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(s0, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(s1, IK_DTYPE_U8, 2, 1));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out0 || !out1 || out0->dtype != out1->dtype || (out0->dtype != IK_DTYPE_BF16 && out0->dtype != IK_DTYPE_F32))
        return IK_ERR_DTYPE;
    IK_TRY(ik_check(out0, out0->dtype, 2));
    IK_TRY(ik_check(out1, out1->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (K % kG != 0 || w0->shape[1] != K / 2 || w1->shape[1] != K / 2 || s0->shape[0] != N0 || s1->shape[0] != N1 ||
        s0->shape[1] != K / kG || s1->shape[1] != K / kG || act->shape[0] != 1 || out0->shape[0] != M ||
        out1->shape[0] != M || out0->shape[1] != N0 || out1->shape[1] != N1)
        return IK_ERR_SHAPE;
    if (M == 1 && K > 0 && K * 2 <= kLdsMax && mx_lds_blocks() > 0 && N0 > 0 && N1 > 0) {
        hipStream_t s = IK_STREAM(stream);
        const ik_bf16* xp = (const ik_bf16*)x->data;
        const float* ap = (const float*)act->data;
        const uint8_t *a0 = (const uint8_t*)w0->data, *b0 = (const uint8_t*)s0->data;
        const uint8_t *a1 = (const uint8_t*)w1->data, *b1 = (const uint8_t*)s1->data;
        return out0->dtype == IK_DTYPE_F32
                   ? lds_launch(xp, a0, b0, (float*)out0->data, N0, a1, b1, (float*)out1->data, N1, ap, K, s)
                   : lds_launch(xp, a0, b0, (ik_bf16*)out0->data, N0, a1, b1, (ik_bf16*)out1->data, N1, ap, K, s);
    }
    IK_TRY(launch(x, w0, s0, act, out0, scratch, stream, true, rows));
    return launch(x, w1, s1, act, out1, scratch, stream, true, rows);
}
int ik_matmul_mxfp4_pair_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1,
                              const ik_tensor* s1, const ik_tensor* act, ik_tensor* out0, ik_tensor* out1,
                              const ik_scratch* scratch, void* stream) {
    return pair_impl(x, w0, s0, w1, s1, act, out0, out1, scratch, stream, false);
}
int ik_matmul_mxfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                        ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return launch(x, w, scales, act, out, scratch, stream, false);
}
// Row-exact twins (OpCall::exact_rows, speculative verify, batched decode): M <= 16, row r bitwise the M == 1 launch on row r
// alone (tests/rows_exact.rs holds the bitwise proof on the device). M == 1 is the gemv launcher itself; M = 2..16 runs
// mxfp4_gemv_rows_kernel: the same per-row arithmetic (lane-to-group map, fma chain, warp sum; the QDQ is per group of
// 32), with a register shape that does not grow with M.
int ik_matmul_mxfp4_gemv_rows(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                              ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return launch(x, w, scales, act, out, scratch, stream, true, true);
}
int ik_matmul_mxfp4_pair_gemv_rows(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1,
                                   const ik_tensor* s1, const ik_tensor* act, ik_tensor* out0, ik_tensor* out1,
                                   const ik_scratch* scratch, void* stream) {
    return pair_impl(x, w0, s0, w1, s1, act, out0, out1, scratch, stream, true);
}
// matmul_mxfp4_mma_v1 (lane mx): matmul_mxfp4_mma as on main before v2 (the bitwise reference of tests/mxfp4_v2.rs)
int ik_matmul_mxfp4_mma_v1(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                           ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return launch(x, w, scales, act, out, scratch, stream, false, false, true);
}
// matmul_mxfp4_mma_fold (lane mx, OPT-IN, changes numerics): W4A4 folded onto the fp8 WMMA (launch_fold above). Scratch
// M * K * 2 + M + N bytes, 16-byte aligned.
int ik_matmul_mxfp4_mma_fold(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                             ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return launch_fold(x, w, scales, act, out, scratch, stream);
}
#ifdef IK_MX_BENCH
// bench hook: tile (0 = the table) and v1 for the following calls
void ik_mxfp4_bench_set(int wn, int wm, int tn, int tm, int v1) {
    g_mx_tile[0] = wn;
    g_mx_tile[1] = wm;
    g_mx_tile[2] = tn;
    g_mx_tile[3] = tm;
    g_mx_v1 = v1;
}
#endif

}  // extern "C"
