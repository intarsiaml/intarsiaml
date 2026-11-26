// Lossless 12-bit bf16 weights (P12, format crates/intarsia-core/src/quant/p12.rs) on HIP (lane ink-hip-p12): the decode / verify
// GEMVs of Inkling-2's dense weights (wo, the unembed, the in-projection). Ports of the CUDA kernels (kernels/cuda/inkling2_p12.cu)
// onto the HIP bf16 twins' loop shapes. Each kernel decodes every weight element to its exact bf16 bits in registers (the in-place
// 12-bit code plus the row base, then the row's escapes patched in place), so the words it hands to the arithmetic are the words
// the bf16 kernel loads from the bf16 weight; the lane -> chunk mapping, the fma chain and the reduction are the bf16 kernel's:
//
//   p12_gemv_kernel<M, R, U, OutT>   == gemv_kernel<M, R, U, OutT> (matmul.hip: matmul_bf16_gemv[_rows] at every M it takes,
//                                       logits_bf16_gemv[_rows] at M <= 8; the one-row unembed in pieces as gemv_vec_split)
//   p12_gemv_wide_kernel<M, OutT>    == gemv_rows_wide_kernel<M, OutT> (logits_bf16_gemv_rows at M >= 9)
//   p12_in_proj_kernel<M>            == in_proj_gemv_kernel<M> (ink_proj.hip: ink_in_proj_bf16_gemv[_rows])
//
// In every HIP bf16 GEMV a (weight row, activation row) output is lane L's fma chain over chunks L, L + 32, L + 64, ... (8 fmas per
// chunk in element order) and then ik_warp_sum; R (weight rows per wave), U (chunk loads in flight per lane), the block and the
// piece only change the schedule. The defaults below keep the bf16 twins' exact loop shapes (even their masked tail iterations, which
// add 0 * 0) or the same masked-tail pattern, so even a signed zero matches; the ik_hip_p12_set variants are the same chain with other
// schedules.
//
// Operands: K % 16 == 0, x bf16 16-byte aligned, codes [N, 3K/2] u8 16-byte aligned (plane A at row byte 0, 8 bytes per 8-element
// chunk; plane B at row byte K, 4 bytes per chunk), aux [N + 1 + E] (u32 as i32).
#include "ik_hip.h"
#include "ik_hip_abi_inkp12.h"

namespace {

constexpr int kWarps = 8;  // waves per block: matmul.hip's and ink_proj.hip's GEMVs

IK_DEVI void store_out(float* p, float v) { *p = v; }
IK_DEVI void store_out(ik_bf16* p, float v) { *p = ik_f2b(v); }

// streaming loads of the codes (read once per step): the 8-byte plane A word pair and the 4-byte plane B word
typedef unsigned int ik_u32x2 __attribute__((ext_vector_type(2)));
IK_DEVI uint2 ldcs2(const uint2* p) {
    const ik_u32x2 v = __builtin_nontemporal_load(reinterpret_cast<const ik_u32x2*>(p));
    return make_uint2(v[0], v[1]);
}
IK_DEVI uint32_t ldcs1(const uint32_t* p) { return __builtin_nontemporal_load(p); }

// The wave's index in its block, as a scalar: every row-dependent address (aux, the side list, the segment) is then wave-uniform,
// so the row header and the escape entries load through the scalar unit.
IK_DEVI int wave_in_block() { return __builtin_amdgcn_readfirstlane((int)(threadIdx.x / IK_WARP)); }

// One packed weight row: plane A, plane B, the base term, its escapes, and this lane's cursor into them.
struct P12Row {
    const uint2* a;
    const uint32_t* b;
    uint32_t bb;           // base * 0x00800080
    const uint32_t* side;  // this row's escapes (col << 16 | bf16 bits), columns ascending
    uint32_t ns;           // how many (wave-uniform)
    uint32_t e;            // cursor: the first escape not before the chunk last decoded (a lane decodes its chunks ascending)
    uint32_t nx;           // that escape's entry (valid while e < ns)
};

// Row r of a P12 weight with nrows rows (the global row count: the side list starts after aux[nrows]).
IK_DEVI P12Row p12_row(const uint8_t* codes, const uint32_t* aux, int64_t nrows, int64_t r, int64_t K) {
    P12Row w;
    const uint8_t* base = codes + r * (3 * K / 2);
    w.a = reinterpret_cast<const uint2*>(base);
    w.b = reinterpret_cast<const uint32_t*>(base + K);
    const uint32_t m0 = aux[r], m1 = aux[r + 1];
    w.bb = (m0 & 0xFFu) * 0x00800080u;
    w.side = aux + nrows + 1 + (m0 >> 8);
    w.ns = (m1 >> 8) - (m0 >> 8);
    w.e = 0;
    w.nx = w.ns ? w.side[0] : 0u;
    return w;
}

// The raw code words of chunk c (loads only), decoded later by p12_dec: a lane issues all its loads in flight before decoding.
struct P12Raw {
    uint2 a;
    uint32_t b;
};
IK_DEVI P12Raw p12_raw(const P12Row& w, int64_t c) { return P12Raw{ldcs2(w.a + c), ldcs1(w.b + c)}; }
IK_DEVI P12Raw p12_zero() { return P12Raw{make_uint2(0u, 0u), 0u}; }

// Chunk c (elements 8c .. 8c + 7) as the four bf16 pair words a bf16 weight stores, escapes patched. A lane must decode the chunks
// of one row in ascending order (the cursor only moves forward); the escape of column col lives in chunk col >> 3.
IK_DEVI uint4 p12_dec(P12Row& w, const P12Raw& r, int64_t c) {
    const uint2 a = r.a;
    const uint32_t b = r.b;
    const uint32_t q3 = ((a.x >> 11) & 0x000F000Fu) | ((a.y >> 7) & 0x00F000F0u) | ((b >> 3) & 0x07000700u) | ((b << 1) & 0x80008000u);
    uint32_t v[4] = {(a.x & 0x87FF87FFu) + w.bb, (a.y & 0x87FF87FFu) + w.bb, (b & 0x87FF87FFu) + w.bb, q3 + w.bb};
    if (w.ns != 0) {  // wave-uniform: 84 % of the unembed's rows (and most in_proj / wo rows) have no escape and skip this
        // skip the escapes of earlier chunks, patch this chunk's
        while (w.e < w.ns && (int64_t)(w.nx >> 19) <= c) {
            const uint32_t ent = w.nx;
            const int64_t ec = (int64_t)(ent >> 19);
            ++w.e;
            if (w.e < w.ns) w.nx = w.side[w.e];
            if (ec < c) continue;
            const uint32_t col = ent >> 16;
            const uint32_t j = (col >> 1) & 3u, sh = (col & 1u) * 16u;
            const uint32_t m = 0xFFFFu << sh;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                if ((uint32_t)q == j) v[q] = (v[q] & ~m) | ((ent & 0xFFFFu) << sh);
        }
    }
    return make_uint4(v[0], v[1], v[2], v[3]);
}

IK_DEVI uint4 p12_load(P12Row& w, int64_t c) { return p12_dec(w, p12_raw(w, c), c); }

// ---- MatmulP12: matmul.hip gemv_body<M, R, U, OutT> with the weight loads replaced by P12 loads ---------------------------------
// Rows [r0, r1) of the N-row weight (a piece of the one-row unembed, or all of it); out[m * ldo + n] for global row n.
template <int M, int R, int U, typename OutT>
__device__ __forceinline__ void p12_gemv_body(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ codes,
                                              const uint32_t* __restrict__ aux, OutT* __restrict__ out, int64_t ldo, int64_t N,
                                              int64_t r0, int64_t r1, int64_t K) {
    const int warp = wave_in_block(), lane = threadIdx.x % IK_WARP;
    const int64_t n0 = r0 + ((int64_t)blockIdx.x * kWarps + warp) * R;
    if (n0 >= r1) return;  // whole wave exits together; no block-level sync below
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);
    P12Row wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < r1 ? n0 + r : r1 - 1;  // tail rows re-read the last row; their result is dropped
        wr[r] = p12_row(codes, aux, N, n, K);
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

    for (int64_t base = 0; base < nv; base += (int64_t)IK_WARP * U) {
        P12Raw raw[R][U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
#pragma unroll
            for (int r = 0; r < R; ++r) raw[r][u] = ok ? p12_raw(wr[r], idx) : p12_zero();
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
            uint4 wv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) wv[r] = ok ? p12_dec(wr[r], raw[r][u], idx) : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[8];
                ik_unpack8(ok ? ik_ldg(x8 + m * nv + idx) : make_uint4(0, 0, 0, 0), xf);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float wf[8];
                    ik_unpack8(wv[r], wf);
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[r][m] += wf[e] * xf[e];
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float s = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < r1) store_out(out + m * ldo + n0 + r, s);
        }
}

template <int M, int R, int U, typename OutT>
__global__ void __launch_bounds__(kWarps* IK_WARP)
p12_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ codes, const uint32_t* __restrict__ aux,
                OutT* __restrict__ out, int64_t ldo, int64_t N, int64_t r0, int64_t r1, int64_t K) {
    p12_gemv_body<M, R, U, OutT>(x, codes, aux, out, ldo, N, r0, r1, K);
}

// matmul.hip gemv_rows_wide_kernel<M, OutT> (the lm_head's rows twin at M >= 9): 4 weight rows per wave, one chunk per lane per
// step, each activation row loaded, unpacked and consumed behind a scheduler barrier so the registers do not grow with M.
constexpr int kWideR = 4;

template <int M, typename OutT>
__global__ void __launch_bounds__(kWarps* IK_WARP)
p12_gemv_wide_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ codes, const uint32_t* __restrict__ aux,
                     OutT* __restrict__ out, int64_t ldo, int64_t N, int64_t r0, int64_t r1, int64_t K) {
    constexpr int R = kWideR;
    const int warp = wave_in_block(), lane = threadIdx.x % IK_WARP;
    const int64_t n0 = r0 + ((int64_t)blockIdx.x * kWarps + warp) * R;
    if (n0 >= r1) return;
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);
    P12Row wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < r1 ? n0 + r : r1 - 1;
        wr[r] = p12_row(codes, aux, N, n, K);
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    for (int64_t base = 0; base < nv; base += IK_WARP) {
        const int64_t idx = base + lane;
        const bool ok = idx < nv;
        float wf[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) ik_unpack8(ok ? p12_load(wr[r], idx) : make_uint4(0, 0, 0, 0), wf[r]);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            __builtin_amdgcn_sched_barrier(0);  // keeps the loads of row m + 1 from being hoisted above the fmas of row m
            float xf[8];
            ik_unpack8(ok ? ik_ldg(x8 + m * nv + idx) : make_uint4(0, 0, 0, 0), xf);
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[r][m] += wf[r][e] * xf[e];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float sum = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < r1) store_out(out + m * ldo + n0 + r, sum);
        }
}

// ---- launch shape (ik_hip_p12_set; every value the same bits) ------------------------------------------------------------------
int g_gemv_r = -1, g_gemv_u = -1, g_inproj_u = -1;

struct RU {
    int r, u;
};
// the instantiated (R, U) of p12_gemv_kernel at M = 1 / at M = 2..8 (the defaults first: the bf16 twins' shapes)
constexpr bool ru_one(int r, int u) {
    return (r == 1 && (u == 8 || u == 4 || u == 9 || u == 12 || u == 16)) || (r == 2 && (u == 4 || u == 8));
}
constexpr bool ru_many(int r, int u) { return (r == 2 && (u == 4 || u == 8)) || (r == 1 && (u == 4 || u == 8)); }

// Defaults (window 1 microbench, fleetq #12128, rotated copies; docs/results/2026-11-26-inkling2-hip-p12.md): one row R = 1, U = 16 (the
// unembed 1100 us against 1214 at the twin's 1, 8; wo 22.9 against 22.1), 2..4 rows R = 1, U = 4 (4 rows: wo 23.7 against 25.4, the
// unembed 1282 against 1319 at 2, 4), 5..8 rows the twin's 2, 4. Each keeps its twin's masked tail iterations where the twin has them
// (U = 16 at nv = 288 / 384 like the twin's U = 8; U = 4 like the twin's U = 4), so even a signed zero matches.
template <int M>
RU gemv_ru() {
    const RU d = M == 1 ? RU{1, 16} : (M <= 4 ? RU{1, 4} : RU{2, 4});
    if (g_gemv_r < 0 || g_gemv_u < 0) return d;
    const bool ok = M == 1 ? ru_one(g_gemv_r, g_gemv_u) : ru_many(g_gemv_r, g_gemv_u);
    return ok ? RU{g_gemv_r, g_gemv_u} : d;
}

// One launch of rows [r0, r1); false when no instantiated variant matches (nothing launched).
template <int M, typename OutT>
bool p12_gemv_go(const ik_bf16* x, const uint8_t* codes, const uint32_t* aux, OutT* out, int64_t ldo, int64_t N, int64_t r0,
                 int64_t r1, int64_t K, hipStream_t s) {
    const unsigned block = kWarps * IK_WARP;
    if constexpr (M >= 9) {
        const unsigned grid = (unsigned)((r1 - r0 + (int64_t)kWarps * kWideR - 1) / ((int64_t)kWarps * kWideR));
        p12_gemv_wide_kernel<M, OutT><<<grid, block, 0, s>>>(x, codes, aux, out, ldo, N, r0, r1, K);
        return true;
    } else {
        const RU c = gemv_ru<M>();
#define IK_P12_RU(RV, UV)                                                                                          \
    if (c.r == RV && c.u == UV) {                                                                                  \
        const unsigned grid = (unsigned)((r1 - r0 + (int64_t)kWarps * RV - 1) / ((int64_t)kWarps * RV));           \
        p12_gemv_kernel<M, RV, UV, OutT><<<grid, block, 0, s>>>(x, codes, aux, out, ldo, N, r0, r1, K);             \
        return true;                                                                                               \
    }
        if constexpr (M == 1) {
            IK_P12_RU(1, 8) IK_P12_RU(1, 4) IK_P12_RU(1, 9) IK_P12_RU(1, 12) IK_P12_RU(1, 16) IK_P12_RU(2, 4) IK_P12_RU(2, 8)
        } else {
            IK_P12_RU(2, 4) IK_P12_RU(2, 8) IK_P12_RU(1, 4) IK_P12_RU(1, 8)
        }
#undef IK_P12_RU
        return false;
    }
}

// Estimated microseconds of an M-row call over [N, K] P12 codes (the split rule's input): the code bytes at 627 GB/s (the R9700's
// measured stream rate, as gemv_vec_split), times a row factor for the rows past 4 (the fma work per byte grows with M).
double p12_est_us(int64_t M, int64_t N, int64_t K) {
    const double bytes = (double)N * (double)K * 1.5;
    const double rows = M <= 4 ? 1.0 : (double)M / 4.0;
    return bytes / 627000.0 * rows;
}

// INTARSIA_HIP_INK_DEC3 bit 4 (dec3's unembed split, matmul.hip gemv_vec_split): a call estimated over the R9700's dispatch stall
// line runs as consecutive pieces of output rows. Each piece is the same kernel on rows [a, b) with the global N (the side list
// starts after aux[N]) and the full output stride, so a row's loads, fma chain and warp sum do not change: the same bits. Unlike
// the bf16 split this takes any M (the output stride is passed).
template <typename OutT>
int p12_gemv_split(const ik_bf16* x, const uint8_t* codes, const uint32_t* aux, OutT* out, int64_t M, int64_t N, int64_t K,
                   hipStream_t s) {
    const int64_t unit = (int64_t)kWarps * 4;  // whole blocks of every instantiated R
    const int pieces = (ik_hip_dec3_mask() & 4) ? ik_split_pieces(p12_est_us(M, N, K), N / unit) : 1;
    const int64_t per = pieces <= 1 ? N : ((N + pieces - 1) / pieces + unit - 1) / unit * unit;
    for (int64_t a = 0; a < N; a += per) {
        const int64_t b = N - a < per ? N : a + per;
        bool ok = false;
        switch (M) {
#define IK_P12_M(MV) \
    case MV: ok = p12_gemv_go<MV, OutT>(x, codes, aux, out, N, N, a, b, K, s); break;
            IK_P12_M(1) IK_P12_M(2) IK_P12_M(3) IK_P12_M(4) IK_P12_M(5) IK_P12_M(6) IK_P12_M(7) IK_P12_M(8) IK_P12_M(9) IK_P12_M(10)
            IK_P12_M(11) IK_P12_M(12) IK_P12_M(13) IK_P12_M(14) IK_P12_M(15) IK_P12_M(16)
#undef IK_P12_M
            default: return IK_ERR_SHAPE;
        }
        if (!ok) return IK_ERR_INVALID_ARG;  // never report OK for a launch that did not happen
    }
    return ik_launch_status();
}

// codes [N, 3K/2] u8 and aux [N + 1 + E] (i32 holding u32) for an [N, K] weight
int check_p12(const ik_tensor* codes, const ik_tensor* aux, int64_t N, int64_t K) {
    IK_TRY(ik_check(codes, IK_DTYPE_U8, 2));
    IK_TRY(ik_check(aux, IK_DTYPE_I32, 1));
    if (codes->shape[0] != N || codes->shape[1] != 3 * K / 2) return IK_ERR_SHAPE;
    if (aux->shape[0] < N + 1) return IK_ERR_SHAPE;
    if (!ik_aligned16(codes->data) || ((uintptr_t)aux->data & 3) != 0) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

int matmul_p12(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    if (out->ndim != 2) return IK_ERR_SHAPE;
    const int64_t M = x->shape[0], K = x->shape[1], N = out->shape[1];
    if (out->shape[0] != M || M < 1 || M > 16 || K % 16 != 0 || K > 65536) return IK_ERR_SHAPE;
    IK_TRY(check_p12(codes, aux, N, K));
    if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    if (N == 0) return IK_OK;
    if (N > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* cp = (const uint8_t*)codes->data;
    const uint32_t* ap = (const uint32_t*)aux->data;
    if (out->dtype == IK_DTYPE_F32) return p12_gemv_split<float>(xp, cp, ap, (float*)out->data, M, N, K, s);
    if (out->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    return p12_gemv_split<ik_bf16>(xp, cp, ap, (ik_bf16*)out->data, M, N, K, s);
}

// ---- InkInProj over P12 codes: ink_proj.hip in_proj_gemv_kernel<M> with the weight loads replaced by P12 loads ------------------
// One wave per row of the stacked weight [Nq + Nk + Nv, D], its segment's activation plane and output.
struct GemvSegs {
    const ik_bf16* x[3];
    ik_bf16* out[3];
    int n[3];
    int r0[4];
};

template <int M>
__global__ void __launch_bounds__(kWarps* IK_WARP)
p12_in_proj_kernel(GemvSegs sg, const uint8_t* __restrict__ codes, const uint32_t* __restrict__ aux, int64_t N, int64_t K) {
    const int warp = wave_in_block(), lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;
    const int j = n < sg.r0[1] ? 0 : (n < sg.r0[2] ? 1 : 2);
    const int64_t nn = n - sg.r0[j];
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(sg.x[j]);
    P12Row wr = p12_row(codes, aux, N, n, K);
#pragma unroll 4
    for (int64_t i = lane; i < nv; i += IK_WARP) {
        float wf[8];
        ik_unpack8(p12_load(wr, i), wf);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float xf[8];
            ik_unpack8(ik_ldg(x8 + m * nv + i), xf);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[m] += wf[e] * xf[e];
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) sg.out[j][(int64_t)m * sg.n[j] + nn] = ik_f2b(s);
    }
}

// The same chain with U chunk loads in flight per lane (the gemv body's schedule). With U = nv / 32 (9 at D = 2304) the loop has no
// masked iterations, exactly the stride loop's adds.
template <int M, int U>
__global__ void __launch_bounds__(kWarps* IK_WARP)
p12_in_proj_u_kernel(GemvSegs sg, const uint8_t* __restrict__ codes, const uint32_t* __restrict__ aux, int64_t N, int64_t K) {
    const int warp = wave_in_block(), lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;
    const int j = n < sg.r0[1] ? 0 : (n < sg.r0[2] ? 1 : 2);
    const int64_t nn = n - sg.r0[j];
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(sg.x[j]);
    P12Row wr = p12_row(codes, aux, N, n, K);
    for (int64_t base = 0; base < nv; base += (int64_t)IK_WARP * U) {
        P12Raw raw[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            raw[u] = idx < nv ? p12_raw(wr, idx) : p12_zero();
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
            float wf[8];
            ik_unpack8(ok ? p12_dec(wr, raw[u], idx) : make_uint4(0, 0, 0, 0), wf);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[8];
                ik_unpack8(ok ? ik_ldg(x8 + m * nv + idx) : make_uint4(0, 0, 0, 0), xf);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[m] += wf[e] * xf[e];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) sg.out[j][(int64_t)m * sg.n[j] + nn] = ik_f2b(s);
    }
}

// Default: up to 4 rows with nv a multiple of 9 * 32 (D = 2304: 9 chunks per lane), all nine chunk loads of a lane in flight
// (p12_in_proj_u_kernel<M, 9>: no masked iterations, exactly the stride loop's adds; window 1: 53.8 / 55.0 us at 1 / 4 rows against
// 57.1 / 58.7 for the stride loop and 69.4 / 70.0 for bf16); otherwise the stride loop (at 16 rows U = 9 spills: 1464 us).
template <int M>
void in_proj_go(const GemvSegs& sg, const uint8_t* cp, const uint32_t* ap, int64_t N, int64_t D, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    const unsigned block = kWarps * IK_WARP;
    const int u = g_inproj_u >= 0 ? g_inproj_u : (M <= 4 && (D / 8) % (9 * IK_WARP) == 0 ? 9 : 0);
    switch (u) {
        case 4: p12_in_proj_u_kernel<M, 4><<<grid, block, 0, s>>>(sg, cp, ap, N, D); break;
        case 8: p12_in_proj_u_kernel<M, 8><<<grid, block, 0, s>>>(sg, cp, ap, N, D); break;
        case 9: p12_in_proj_u_kernel<M, 9><<<grid, block, 0, s>>>(sg, cp, ap, N, D); break;
        default: p12_in_proj_kernel<M><<<grid, block, 0, s>>>(sg, cp, ap, N, D); break;
    }
}

int in_proj_p12(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv,
                void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    ik_tensor* o[3] = {pq, pk, pv};
    const int64_t S = x->shape[0], T = x->shape[1], D = x->shape[2];
    if (!(S == 1 || S == 3) || T < 1 || T > 16 || D % 16 != 0 || D > 65536) return IK_ERR_SHAPE;
    GemvSegs sg{};
    int64_t nsum = 0;
    const ik_bf16* xp = (const ik_bf16*)x->data;
    for (int j = 0; j < 3; ++j) {
        IK_TRY(ik_check(o[j], IK_DTYPE_BF16, 2));
        if (o[j]->shape[0] != T) return IK_ERR_SHAPE;
        sg.x[j] = xp + (j < S - 1 ? j : S - 1) * T * D;
        sg.out[j] = (ik_bf16*)o[j]->data;
        sg.n[j] = (int)o[j]->shape[1];
        sg.r0[j] = (int)nsum;
        nsum += o[j]->shape[1];
    }
    sg.r0[3] = (int)nsum;
    if (nsum > 0x7fffffff) return IK_ERR_SHAPE;
    IK_TRY(check_p12(codes, aux, nsum, D));
    if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    if (nsum == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    const uint8_t* cp = (const uint8_t*)codes->data;
    const uint32_t* ap = (const uint32_t*)aux->data;
    switch (T) {
#define IK_IPP_CASE(R) \
    case R: in_proj_go<R>(sg, cp, ap, nsum, D, s); break;
        IK_IPP_CASE(1) IK_IPP_CASE(2) IK_IPP_CASE(3) IK_IPP_CASE(4) IK_IPP_CASE(5) IK_IPP_CASE(6) IK_IPP_CASE(7) IK_IPP_CASE(8)
        IK_IPP_CASE(9) IK_IPP_CASE(10) IK_IPP_CASE(11) IK_IPP_CASE(12) IK_IPP_CASE(13) IK_IPP_CASE(14) IK_IPP_CASE(15) IK_IPP_CASE(16)
#undef IK_IPP_CASE
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    return matmul_p12(x, codes, aux, out, stream);
}
int ik_matmul_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    return matmul_p12(x, codes, aux, out, stream);
}
int ik_matmul_p12_logits_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    return matmul_p12(x, codes, aux, out, stream);
}
int ik_matmul_p12_logits_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    return matmul_p12(x, codes, aux, out, stream);
}
int ik_ink_in_proj_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                            ik_tensor* pv, void* stream) {
    return in_proj_p12(x, codes, aux, pq, pk, pv, stream);
}
int ik_ink_in_proj_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                                 ik_tensor* pv, void* stream) {
    return in_proj_p12(x, codes, aux, pq, pk, pv, stream);
}
void ik_hip_p12_set(int gemv_r, int gemv_u, int inproj_u) {
    g_gemv_r = gemv_r;
    g_gemv_u = gemv_u;
    g_inproj_u = inproj_u;
}

}  // extern "C"
