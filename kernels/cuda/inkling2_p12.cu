// Lossless 12-bit bf16 weights (P12, format crates/intarsia-core/src/quant/p12.rs) on CUDA: the GEMVs of Inkling-2's dense
// decode weights (lane ink-cuda-p12; the Metal kernels are kernels/metal/p12.metal). Each kernel decodes every weight element
// to its exact bf16 bits in registers (the in-place 12-bit code plus the row base, then the row's escapes patched in place),
// so the words it hands to the arithmetic are the words the bf16 kernel loads from the bf16 weight. The lane -> element
// mapping, the multiply-add chain and the reduction tree are the bf16 kernel's, so the outputs are the same bits:
//
//   p12_gemv_kernel<M, OutT>  == gemv_kernel<M, OutT, true> (matmul.cu: matmul_bf16_gemv[_rows], logits_bf16_gemv[_rows];
//                                the rows twin's row-group split runs the same body per row, so it has the same bits too)
//   p12_in_proj_kernel        == ink_mma_gemv_kernel (ink_proj.cu: ink_in_proj_bf16_mma[_rows])
//
// Operands: K % 16 == 0 (8-byte plane-A chunks), x 16-byte aligned, bf16 x; codes [N, 3K/2] u8, aux [N + 1 + E] u32.
#include "ik_launch.h"
#include "ik_internal.h"
#include "ik_quant.cuh"
#include "ik_cuda_abi_inkp12.h"

namespace {

constexpr int kWarpsPerBlock = 8;  // matmul.cu's GEMV
constexpr int kProjWarps = 4;      // ink_proj.cu's tensor-core GEMV

IK_DEVI void store_out(float* p, float v) { *p = v; }
IK_DEVI void store_out(ik_bf16* p, float v) { *p = ik_f2b(v); }

// One packed weight row: plane A (a uint2 per 8-element chunk), plane B (a u32 per chunk), the base term, its escapes.
struct P12Row {
    const uint2* a;
    const uint32_t* b;
    uint32_t bb;            // base * 0x00800080
    const uint32_t* side;   // this row's escapes (col << 16 | bf16 bits), columns ascending
    uint32_t ns;            // how many
    uint32_t e;             // cursor: the first escape not before the chunk last loaded (a lane loads its chunks ascending)
};

IK_DEVI P12Row p12_row(const uint8_t* codes, const uint32_t* aux, int64_t nrows, int64_t r, int64_t K) {
    P12Row w;
    const uint8_t* base = codes + r * (3 * K / 2);
    w.a = reinterpret_cast<const uint2*>(base);
    w.b = reinterpret_cast<const uint32_t*>(base + K);
    const uint32_t m0 = __ldg(aux + r), m1 = __ldg(aux + r + 1);
    w.bb = (m0 & 0xFFu) * 0x00800080u;
    w.side = aux + nrows + 1 + (m0 >> 8);
    w.ns = (m1 >> 8) - (m0 >> 8);
    w.e = 0;
    return w;
}

// Chunk c (elements 8c .. 8c + 7) as the four bf16 pair words a bf16 weight stores, escapes patched. The chunks of one row
// must be loaded in ascending order (the cursor only moves forward); the escape of column col lives in chunk col >> 3.
IK_DEVI uint4 p12_load(P12Row& w, int64_t c) {
    const uint2 a = __ldcs(w.a + c);
    const uint32_t b = __ldcs(w.b + c);
    const uint32_t q3 = ((a.x >> 11) & 0x000F000Fu) | ((a.y >> 7) & 0x00F000F0u) | ((b >> 3) & 0x07000700u) | ((b << 1) & 0x80008000u);
    uint32_t v[4] = {(a.x & 0x87FF87FFu) + w.bb, (a.y & 0x87FF87FFu) + w.bb, (b & 0x87FF87FFu) + w.bb, q3 + w.bb};
    // escapes: rare (most rows have none); skip those of earlier chunks, patch this chunk's
    while (w.e < w.ns) {
        const uint32_t ent = __ldg(w.side + w.e);
        const int64_t ec = (int64_t)(ent >> 19);
        if (ec > c) break;
        ++w.e;
        if (ec < c) continue;
        const uint32_t col = ent >> 16;
        const uint32_t j = (col >> 1) & 3u, sh = (col & 1u) * 16u;
        const uint32_t m = 0xFFFFu << sh;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if ((uint32_t)q == j) v[q] = (v[q] & ~m) | ((ent & 0xFFFFu) << sh);
    }
    return make_uint4(v[0], v[1], v[2], v[3]);
}

// ---- MatmulP12: matmul.cu gemv_body<M, OutT, VEC = true> with the weight loads replaced by p12_load ------------------------
template <int M, typename OutT>
__global__ void __launch_bounds__(kWarpsPerBlock * IK_WARP)
p12_gemv_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ codes, const uint32_t* __restrict__ aux,
                OutT* __restrict__ out, int64_t N, int64_t K) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int64_t n = (int64_t)blockIdx.x * kWarpsPerBlock + warp;
    if (n >= N) return;
    P12Row wr = p12_row(codes, aux, N, n, K);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const int64_t nv = K / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);
#pragma unroll 4
    for (int64_t i = lane; i < nv; i += IK_WARP) {
        float wf[8];
        ik_unpack8(p12_load(wr, i), wf);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float xf[8];
            ik_unpack8(__ldg(x8 + m * nv + i), xf);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[m] += wf[e] * xf[e];
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float s = ik_warp_sum(acc[m]);
        if (lane == 0) store_out(out + m * N + n, s);
    }
}

template <typename OutT>
int p12_gemv(const ik_bf16* x, const uint8_t* codes, const uint32_t* aux, OutT* out, int64_t M, int64_t N, int64_t K, cudaStream_t s) {
    const unsigned grid = (unsigned)((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const unsigned block = kWarpsPerBlock * IK_WARP;
#define IK_P12_CASE(R) \
    case R: p12_gemv_kernel<R, OutT><<<grid, block, 0, s>>>(x, codes, aux, out, N, K); break;
    switch (M) {
        IK_P12_CASE(1) IK_P12_CASE(2) IK_P12_CASE(3) IK_P12_CASE(4) IK_P12_CASE(5) IK_P12_CASE(6) IK_P12_CASE(7) IK_P12_CASE(8)
        IK_P12_CASE(9) IK_P12_CASE(10) IK_P12_CASE(11) IK_P12_CASE(12) IK_P12_CASE(13) IK_P12_CASE(14) IK_P12_CASE(15) IK_P12_CASE(16)
        default: return IK_ERR_SHAPE;
    }
#undef IK_P12_CASE
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

// ---- InkInProj over P12 codes: ink_proj.cu ink_mma_gemv_kernel with the weight loads replaced by p12_load ---------------
struct P12Seg {
    const ik_bf16* x;
    ik_bf16* out;
    int n;   // rows of this segment
    int r0;  // its first row in the stacked weight
};
struct P12Segs {
    P12Seg s[3];
    int nseg;
    int t0[4];
};

IK_DEVI void mma_add(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    float t[4] = {0.f, 0.f, 0.f, 0.f};
    ik_mma_bf16_16816(t, a0, a1, a2, a3, b0, b1);
#pragma unroll
    for (int v = 0; v < 4; ++v) c[v] += t[v];
}

IK_DEVI uint4 ld_act(const ik_bf16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

__global__ void __launch_bounds__(kProjWarps * IK_WARP)
p12_in_proj_kernel(P12Segs sg, const uint8_t* __restrict__ codes, const uint32_t* __restrict__ aux, int64_t nrows, int T, int D) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int tile = blockIdx.x * kProjWarps + warp;
    int j = 0;
    while (j + 1 < sg.nseg && tile >= sg.t0[j + 1]) ++j;
    if (tile >= sg.t0[sg.nseg]) return;
    const P12Seg s = sg.s[j];
    const int n0 = (tile - sg.t0[j]) * 16;
    const int m0 = blockIdx.y * 16;
    const int g = lane >> 2, c = lane & 3;
    const int ra = min(n0 + g, s.n - 1), rb = min(n0 + g + 8, s.n - 1);  // rows past n are read, never stored
    const bool two = T - m0 > 8;
    const int xa = m0 + g, xb = m0 + 8 + g;
    const bool ha = xa < T, hb = two && xb < T;
    P12Row wa = p12_row(codes, aux, nrows, (int64_t)s.r0 + ra, D);
    P12Row wb = p12_row(codes, aux, nrows, (int64_t)s.r0 + rb, D);
    const ik_bf16* pa = s.x + (int64_t)(ha ? xa : 0) * D + 16 * c;
    const ik_bf16* pb = s.x + (int64_t)(hb ? xb : 0) * D + 16 * c;
    float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4 z = make_uint4(0, 0, 0, 0);
    const int kmain = D & ~63;
    // the bf16 kernel's lane loads elements kb + 16c .. kb + 16c + 15 of its rows: chunks (kb + 16c) / 8 and the next
    const int cc = 2 * c;
    if (kmain > 0) {
        uint4 A0 = p12_load(wa, cc), A1 = p12_load(wa, cc + 1), B0 = p12_load(wb, cc), B1 = p12_load(wb, cc + 1);
        uint4 X0 = ha ? ld_act(pa) : z, X1 = ha ? ld_act(pa + 8) : z;
        uint4 Y0 = hb ? ld_act(pb) : z, Y1 = hb ? ld_act(pb + 8) : z;
        for (int kb = 0; kb < kmain; kb += 64) {
            const uint32_t a[8] = {A0.x, A0.y, A0.z, A0.w, A1.x, A1.y, A1.z, A1.w};
            const uint32_t b[8] = {B0.x, B0.y, B0.z, B0.w, B1.x, B1.y, B1.z, B1.w};
            const uint32_t x[8] = {X0.x, X0.y, X0.z, X0.w, X1.x, X1.y, X1.z, X1.w};
            const uint32_t y[8] = {Y0.x, Y0.y, Y0.z, Y0.w, Y1.x, Y1.y, Y1.z, Y1.w};
            const int kn = kb + 64;
            if (kn < kmain) {
                const int ck = kn / 8 + cc;
                A0 = p12_load(wa, ck), A1 = p12_load(wa, ck + 1), B0 = p12_load(wb, ck), B1 = p12_load(wb, ck + 1);
                if (ha) X0 = ld_act(pa + kn), X1 = ld_act(pa + kn + 8);
                if (hb) Y0 = ld_act(pb + kn), Y1 = ld_act(pb + kn + 8);
            }
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                mma_add(c0, a[2 * st], b[2 * st], a[2 * st + 1], b[2 * st + 1], x[2 * st], x[2 * st + 1]);
                if (two) mma_add(c1, a[2 * st], b[2 * st], a[2 * st + 1], b[2 * st + 1], y[2 * st], y[2 * st + 1]);
            }
        }
    }
    // tail: 16-wide k steps (D % 16 == 0), lane c's four k are kb + 4c .. kb + 4c + 3: half (c & 1) of chunk (kb + 4c) / 8
    for (int kb = kmain; kb < D; kb += 16) {
        const int o = kb - 16 * c + 4 * c;  // the activation pointers are offset by 16c
        const int ch = (kb + 4 * c) / 8;
        const uint4 Aw = p12_load(wa, ch), Bw = p12_load(wb, ch);
        const uint2 A = (c & 1) ? make_uint2(Aw.z, Aw.w) : make_uint2(Aw.x, Aw.y);
        const uint2 B = (c & 1) ? make_uint2(Bw.z, Bw.w) : make_uint2(Bw.x, Bw.y);
        const uint2 X = ha ? __ldg(reinterpret_cast<const uint2*>(pa + o)) : make_uint2(0, 0);
        const uint2 Y = hb ? __ldg(reinterpret_cast<const uint2*>(pb + o)) : make_uint2(0, 0);
        mma_add(c0, A.x, B.x, A.y, B.y, X.x, X.y);
        if (two) mma_add(c1, A.x, B.x, A.y, B.y, Y.x, Y.y);
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        if (nt == 1 && !two) break;
        const float* cr = nt ? c1 : c0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int m = m0 + nt * 8 + 2 * c + (e & 1);
            const int n = n0 + g + (e >> 1) * 8;
            if (m < T && n < s.n) s.out[(int64_t)m * s.n + n] = ik_f2b(cr[e]);
        }
    }
}

}  // namespace

extern "C" {

int ik_matmul_p12_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    if (out->ndim != 2) return IK_ERR_SHAPE;
    const int64_t M = x->shape[0], K = x->shape[1], N = out->shape[1];
    if (out->shape[0] != M || M < 1 || M > 16 || K % 16 != 0 || K > 65536) return IK_ERR_SHAPE;
    IK_TRY(check_p12(codes, aux, N, K));
    if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    if (N == 0) return IK_OK;
    if (N > 0x7fffffffLL * kWarpsPerBlock / 2) return IK_ERR_SHAPE;
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* cp = (const uint8_t*)codes->data;
    const uint32_t* ap = (const uint32_t*)aux->data;
    if (out->dtype == IK_DTYPE_F32) return p12_gemv<float>(xp, cp, ap, (float*)out->data, M, N, K, s);
    if (out->dtype == IK_DTYPE_BF16) return p12_gemv<ik_bf16>(xp, cp, ap, (ik_bf16*)out->data, M, N, K, s);
    return IK_ERR_DTYPE;
}
int ik_matmul_p12_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    return ik_matmul_p12_gemv(x, codes, aux, out, stream);
}
int ik_matmul_p12_logits_gemv(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    return ik_matmul_p12_gemv(x, codes, aux, out, stream);
}
int ik_matmul_p12_logits_gemv_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* out, void* stream) {
    return ik_matmul_p12_gemv(x, codes, aux, out, stream);
}

int ik_ink_in_proj_p12_mma(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                           ik_tensor* pv, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    ik_tensor* o[3] = {pq, pk, pv};
    const int64_t S = x->shape[0], T = x->shape[1], D = x->shape[2];
    if (!(S == 1 || S == 3) || D % 16 != 0 || D > 65536 || T > 0x7fffffff) return IK_ERR_SHAPE;
    int64_t nsum = 0;
    for (int j = 0; j < 3; ++j) {
        IK_TRY(ik_check(o[j], IK_DTYPE_BF16, 2));
        if (o[j]->shape[0] != T) return IK_ERR_SHAPE;
        nsum += o[j]->shape[1];
    }
    IK_TRY(check_p12(codes, aux, nsum, D));
    if (!ik_aligned16(x->data)) return IK_ERR_INVALID_ARG;
    if (T == 0 || nsum == 0) return IK_OK;
    if (nsum > 0x7fffffff) return IK_ERR_SHAPE;
    P12Segs sg{};
    sg.nseg = 3;
    int r0 = 0;
    const ik_bf16* xp = (const ik_bf16*)x->data;
    for (int j = 0; j < 3; ++j) {
        const int64_t xi = j < S - 1 ? j : S - 1;
        sg.s[j] = P12Seg{xp + xi * T * D, (ik_bf16*)o[j]->data, (int)o[j]->shape[1], r0};
        r0 += (int)o[j]->shape[1];
    }
    sg.t0[0] = 0;
    for (int j = 0; j < 3; ++j) sg.t0[j + 1] = sg.t0[j] + (sg.s[j].n + 15) / 16;
    const int tiles = sg.t0[3];
    if (tiles == 0) return IK_OK;
    const dim3 grid((unsigned)((tiles + kProjWarps - 1) / kProjWarps), (unsigned)((T + 15) / 16));
    if (grid.y > 65535) return IK_ERR_SHAPE;
    p12_in_proj_kernel<<<grid, kProjWarps * IK_WARP, 0, IK_STREAM(stream)>>>(sg, (const uint8_t*)codes->data,
                                                                            (const uint32_t*)aux->data, nsum, (int)T, (int)D);
    return ik_launch_status();
}
int ik_ink_in_proj_p12_mma_rows(const ik_tensor* x, const ik_tensor* codes, const ik_tensor* aux, ik_tensor* pq, ik_tensor* pk,
                                ik_tensor* pv, void* stream) {
    return ik_ink_in_proj_p12_mma(x, codes, aux, pq, pk, pv, stream);
}

}  // extern "C"
