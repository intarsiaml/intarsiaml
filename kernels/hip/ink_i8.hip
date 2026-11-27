// Int8 weights with one f32 scale per output row on HIP (lane ink-hip-i8, opt-in, lossy; format crates/intarsia-core/src/quant/i8row.rs,
// docs/results/2026-11-26-inkling2-hip-i8.md): the decode / verify GEMVs of Inkling-2's dense weights (wo, the unembed, the
// in-projection), W8A16. Contract: out[m, n] = rA(scale[n] * sum_k x[m, k] * q[n, k]).
//
// Each wave owns R weight rows. Lane L reads 16-byte chunks (16 int8 weights) L, L + 32, L + 64, ... of the row (U loads in flight),
// sign-extends and converts each byte to f32, and runs the fma chain acc[r][m] += w * x in element order over the chunks it owns;
// ik_warp_sum then adds the 32 lanes, and lane 0 multiplies the sum by the row's scale and rounds. The chain of one (row, activation
// row) pair depends neither on M, nor on R, nor on U (only masked tail iterations, which add +0, differ), so a row of an M-row call is
// the one-row call's: rows-exact.
//
//   i8_gemv_kernel<M, R, U, OutT>   MatmulI8 (wo bf16 out, unembed f32 out), M 1..16
//   i8_in_proj_kernel<M, U>         InkInProj with i8 weights: one wave per row of the stacked [Nq + Nk + Nv, D] weight
//
// Operands: K % 16 == 0, x bf16 16-byte aligned, q [N, K] i8 16-byte aligned, scales [N, 1] f32 4-byte aligned.
#include "ik_hip.h"
#include "ik_hip_abi_inki8.h"

namespace {

constexpr int kWarps = 8;  // waves per block: the bf16 twins'

IK_DEVI void store_out(float* p, float v) { *p = v; }
IK_DEVI void store_out(ik_bf16* p, float v) { *p = ik_f2b(v); }

IK_DEVI int wave_in_block() { return __builtin_amdgcn_readfirstlane((int)(threadIdx.x / IK_WARP)); }

// streaming load of the weight bytes (read once per step)
IK_DEVI uint4 ldw(const uint4* p) { return ik_ldcs(p); }

// the four signed bytes of a word as f32 (sign-extend, convert)
IK_DEVI void unpack_i8x4(uint32_t w, float* f) {
    f[0] = (float)(int)(int8_t)(w & 0xFFu);
    f[1] = (float)(int)(int8_t)((w >> 8) & 0xFFu);
    f[2] = (float)(int)(int8_t)((w >> 16) & 0xFFu);
    f[3] = (float)(int)(int8_t)(w >> 24);
}
IK_DEVI void unpack_i8x16(const uint4& u, float* f) {
    unpack_i8x4(u.x, f);
    unpack_i8x4(u.y, f + 4);
    unpack_i8x4(u.z, f + 8);
    unpack_i8x4(u.w, f + 12);
}

// ---- MatmulI8 --------------------------------------------------------------------------------------------------------------------
template <int M, int R, int U, typename OutT>
__global__ void __launch_bounds__(kWarps* IK_WARP)
i8_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ q, const float* __restrict__ scales, OutT* __restrict__ out,
               int64_t N, int64_t K) {
    const int warp = wave_in_block(), lane = threadIdx.x % IK_WARP;
    const int64_t n0 = ((int64_t)blockIdx.x * kWarps + warp) * R;
    if (n0 >= N) return;  // whole wave exits together
    const int64_t nv = K / 16;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);  // 8 bf16 per uint4: chunk c of row m is x8[m * (K / 8) + 2c], + 1
    const int64_t xs = K / 8;
    const uint4* wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int64_t n = n0 + r < N ? n0 + r : N - 1;  // tail rows re-read the last row; their result is dropped
        wrow[r] = reinterpret_cast<const uint4*>(q + n * K);
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

    for (int64_t base = 0; base < nv; base += (int64_t)IK_WARP * U) {
        uint4 raw[R][U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
#pragma unroll
            for (int r = 0; r < R; ++r) raw[r][u] = idx < nv ? ldw(wrow[r] + idx) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
            float wf[R][16];
#pragma unroll
            for (int r = 0; r < R; ++r) unpack_i8x16(raw[r][u], wf[r]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[16];
                ik_unpack8(ok ? ik_ldg(x8 + m * xs + 2 * idx) : make_uint4(0, 0, 0, 0), xf);
                ik_unpack8(ok ? ik_ldg(x8 + m * xs + 2 * idx + 1) : make_uint4(0, 0, 0, 0), xf + 8);
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int e = 0; e < 16; ++e) acc[r][m] += wf[r][e] * xf[e];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float sc = n0 + r < N ? scales[n0 + r] : 0.f;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float s = ik_warp_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N) store_out(out + m * N + n0 + r, s * sc);
        }
    }
}

// ---- launch shape (ik_hip_i8_set; every value the same bits up to a signed zero) ---------------------------------------------------
int g_gemv_r = -1, g_gemv_u = -1, g_inproj_u = -1;

struct RU {
    int r, u;
};
// instantiated (R, U): M 1..4 R 1..2, U 2/4/8; M 5..16 R 1..2, U 1/2
constexpr bool ru_few(int r, int u) { return (r == 1 || r == 2) && (u == 2 || u == 4 || u == 8); }
constexpr bool ru_many(int r, int u) { return (r == 1 || r == 2) && (u == 1 || u == 2); }

template <int M>
RU gemv_ru() {
    const RU d = M == 1 ? RU{1, 8} : (M <= 4 ? RU{1, 4} : RU{2, 2});
    if (g_gemv_r < 0 || g_gemv_u < 0) return d;
    const bool ok = M <= 4 ? ru_few(g_gemv_r, g_gemv_u) : ru_many(g_gemv_r, g_gemv_u);
    return ok ? RU{g_gemv_r, g_gemv_u} : d;
}

template <int M, typename OutT>
bool i8_gemv_go(const ik_bf16* x, const uint8_t* q, const float* sc, OutT* out, int64_t N, int64_t K, hipStream_t s) {
    const unsigned block = kWarps * IK_WARP;
    const RU c = gemv_ru<M>();
#define IK_I8_RU(RV, UV)                                                                                     \
    if (c.r == RV && c.u == UV) {                                                                            \
        const unsigned grid = (unsigned)((N + (int64_t)kWarps * RV - 1) / ((int64_t)kWarps * RV));           \
        i8_gemv_kernel<M, RV, UV, OutT><<<grid, block, 0, s>>>(x, q, sc, out, N, K);                          \
        return true;                                                                                         \
    }
    if constexpr (M <= 4) {
        IK_I8_RU(1, 2) IK_I8_RU(1, 4) IK_I8_RU(1, 8) IK_I8_RU(2, 2) IK_I8_RU(2, 4) IK_I8_RU(2, 8)
    } else {
        IK_I8_RU(1, 1) IK_I8_RU(1, 2) IK_I8_RU(2, 1) IK_I8_RU(2, 2)
    }
#undef IK_I8_RU
    return false;
}

template <typename OutT>
int i8_gemv_dispatch(const ik_bf16* x, const uint8_t* q, const float* sc, OutT* out, int64_t M, int64_t N, int64_t K, hipStream_t s) {
    bool ok = false;
    switch (M) {
#define IK_I8_M(MV) \
    case MV: ok = i8_gemv_go<MV, OutT>(x, q, sc, out, N, K, s); break;
        IK_I8_M(1) IK_I8_M(2) IK_I8_M(3) IK_I8_M(4) IK_I8_M(5) IK_I8_M(6) IK_I8_M(7) IK_I8_M(8) IK_I8_M(9) IK_I8_M(10) IK_I8_M(11)
        IK_I8_M(12) IK_I8_M(13) IK_I8_M(14) IK_I8_M(15) IK_I8_M(16)
#undef IK_I8_M
        default: return IK_ERR_SHAPE;
    }
    if (!ok) return IK_ERR_INVALID_ARG;  // never report OK for a launch that did not happen
    return ik_launch_status();
}

// q [N, K] i8 and scales [N, 1] f32
int check_i8(const ik_tensor* q, const ik_tensor* sc, int64_t N, int64_t K) {
    IK_TRY(ik_check(q, IK_DTYPE_I8, 2));
    IK_TRY(ik_check(sc, IK_DTYPE_F32, 2));
    if (q->shape[0] != N || q->shape[1] != K) return IK_ERR_SHAPE;
    if (sc->shape[0] != N || sc->shape[1] != 1) return IK_ERR_SHAPE;
    if (!ik_aligned16(q->data) || ((uintptr_t)sc->data & 3) != 0) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

int matmul_i8(const ik_tensor* x, const ik_tensor* q, const ik_tensor* sc, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    if (out->ndim != 2) return IK_ERR_SHAPE;
    const int64_t M = x->shape[0], K = x->shape[1], N = out->shape[1];
    if (out->shape[0] != M || M < 1 || M > 16 || K % 16 != 0 || K > 65536) return IK_ERR_SHAPE;
    IK_TRY(check_i8(q, sc, N, K));
    if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    if (N == 0) return IK_OK;
    if (N > 0x7fffffffLL) return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* qp = (const uint8_t*)q->data;
    const float* sp = (const float*)sc->data;
    if (out->dtype == IK_DTYPE_F32) return i8_gemv_dispatch<float>(xp, qp, sp, (float*)out->data, M, N, K, s);
    if (out->dtype != IK_DTYPE_BF16) return IK_ERR_DTYPE;
    return i8_gemv_dispatch<ik_bf16>(xp, qp, sp, (ik_bf16*)out->data, M, N, K, s);
}

// ---- InkInProj over i8 codes -------------------------------------------------------------------------------------------------------
struct GemvSegs {
    const ik_bf16* x[3];
    ik_bf16* out[3];
    int n[3];
    int r0[4];
};

template <int M, int U>
__global__ void __launch_bounds__(kWarps* IK_WARP)
i8_in_proj_kernel(GemvSegs sg, const uint8_t* __restrict__ q, const float* __restrict__ scales, int64_t N, int64_t K) {
    const int warp = wave_in_block(), lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarps + warp;
    if (n >= N) return;
    const int j = n < sg.r0[1] ? 0 : (n < sg.r0[2] ? 1 : 2);
    const int64_t nn = n - sg.r0[j];
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const int64_t nv = K / 16, xs = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(sg.x[j]);
    const uint4* wrow = reinterpret_cast<const uint4*>(q + n * K);
    for (int64_t base = 0; base < nv; base += (int64_t)IK_WARP * U) {
        uint4 raw[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            raw[u] = idx < nv ? ldw(wrow + idx) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t idx = base + (int64_t)u * IK_WARP + lane;
            const bool ok = idx < nv;
            float wf[16];
            unpack_i8x16(raw[u], wf);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[16];
                ik_unpack8(ok ? ik_ldg(x8 + m * xs + 2 * idx) : make_uint4(0, 0, 0, 0), xf);
                ik_unpack8(ok ? ik_ldg(x8 + m * xs + 2 * idx + 1) : make_uint4(0, 0, 0, 0), xf + 8);
#pragma unroll
                for (int e = 0; e < 16; ++e) acc[m] += wf[e] * xf[e];
            }
        }
    }
    const float sc = scales[n];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) sg.out[j][(int64_t)m * sg.n[j] + nn] = ik_f2b(s * sc);
    }
}

// U chunk loads in flight per lane: D = 2304 has 144 chunks (4.5 per lane): U 5 = one masked pass; U 3 / 2 / 1 sweep
template <int M>
bool in_proj_go(const GemvSegs& sg, const uint8_t* qp, const float* sp, int64_t N, int64_t D, hipStream_t s) {
    const unsigned grid = (unsigned)((N + kWarps - 1) / kWarps);
    const unsigned block = kWarps * IK_WARP;
    const int u = g_inproj_u > 0 ? g_inproj_u : (M <= 4 ? 5 : 2);
    switch (u) {
        case 1: i8_in_proj_kernel<M, 1><<<grid, block, 0, s>>>(sg, qp, sp, N, D); return true;
        case 2: i8_in_proj_kernel<M, 2><<<grid, block, 0, s>>>(sg, qp, sp, N, D); return true;
        case 3: i8_in_proj_kernel<M, 3><<<grid, block, 0, s>>>(sg, qp, sp, N, D); return true;
        case 4: i8_in_proj_kernel<M, 4><<<grid, block, 0, s>>>(sg, qp, sp, N, D); return true;
        case 5: i8_in_proj_kernel<M, 5><<<grid, block, 0, s>>>(sg, qp, sp, N, D); return true;
        default: return false;
    }
}

int in_proj_i8(const ik_tensor* x, const ik_tensor* q, const ik_tensor* sc, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream) {
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
    IK_TRY(check_i8(q, sc, nsum, D));
    if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    if (nsum == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    const uint8_t* qp = (const uint8_t*)q->data;
    const float* sp = (const float*)sc->data;
    bool ok = false;
    switch (T) {
#define IK_I8IP_CASE(R) \
    case R: ok = in_proj_go<R>(sg, qp, sp, nsum, D, s); break;
        IK_I8IP_CASE(1) IK_I8IP_CASE(2) IK_I8IP_CASE(3) IK_I8IP_CASE(4) IK_I8IP_CASE(5) IK_I8IP_CASE(6) IK_I8IP_CASE(7) IK_I8IP_CASE(8)
        IK_I8IP_CASE(9) IK_I8IP_CASE(10) IK_I8IP_CASE(11) IK_I8IP_CASE(12) IK_I8IP_CASE(13) IK_I8IP_CASE(14) IK_I8IP_CASE(15)
        IK_I8IP_CASE(16)
#undef IK_I8IP_CASE
        default: return IK_ERR_SHAPE;
    }
    if (!ok) return IK_ERR_INVALID_ARG;
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_i8_gemv(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return matmul_i8(x, q, scales, out, stream);
}
int ik_matmul_i8_gemv_rows(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return matmul_i8(x, q, scales, out, stream);
}
int ik_matmul_i8_logits_gemv(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return matmul_i8(x, q, scales, out, stream);
}
int ik_matmul_i8_logits_gemv_rows(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* out, void* stream) {
    return matmul_i8(x, q, scales, out, stream);
}
int ik_ink_in_proj_i8_gemv(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv,
                           void* stream) {
    return in_proj_i8(x, q, scales, pq, pk, pv, stream);
}
int ik_ink_in_proj_i8_gemv_rows(const ik_tensor* x, const ik_tensor* q, const ik_tensor* scales, ik_tensor* pq, ik_tensor* pk,
                                ik_tensor* pv, void* stream) {
    return in_proj_i8(x, q, scales, pq, pk, pv, stream);
}
void ik_hip_i8_set(int gemv_r, int gemv_u, int inproj_u) {
    g_gemv_r = gemv_r;
    g_gemv_u = gemv_u;
    g_inproj_u = inproj_u;
}

}  // extern "C"
