// Inkling-2-Mini attention in-projection (Op::InkInProj) and the tensor-core GEMV it runs on (lane ink-cuda-mixmoe).
//
// The GEMV: out[m, n] = rA(x[m] . w[n]) for up to three segments (weight rows, activation block, output) in ONE launch. A warp
// owns 16 weight rows and 16 activation rows of a segment and runs mma.sync m16n8k16 (bf16 in, f32 accumulate) with the
// weights as the A operand and the activations as B, so the weights stream from memory once whatever the row count, with no
// unpacking: the dot product is over k, so k may be permuted as long as A and B use the same permutation. Lane (g, c) =
// (lane / 4, lane % 4) loads 16 CONTIGUOUS elements (two 16-byte loads) of weight rows g and g + 8 and of activation row g at
// physical k = kb + 16c .. kb + 16c + 15 of a 64-wide k block; for the block's mma step s (0..3) its four logical k slots
// {2c, 2c+1, 2c+8, 2c+9} are the physical kb + 16c + 4s + {0, 1, 2, 3}, which over the quad is a bijection onto the step's 16 k
// values. The registers then are exactly the fragments (a bf16 pair per 32-bit word).
//
// Rows-exact: every output element is the same chain of mma steps over the same k blocks whatever T (an activation row's
// column in the n-tile does not change its sum: the tensor core computes each column independently), so row m of a T-row call
// is bitwise the one-row call on that row; rows past T are zeros that are never stored. grid.y walks 16-row chunks of T, so
// any T is rows-exact (the in-projection's prefill takes cuBLAS instead: it re-reads the weights per chunk).
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "ik_cuda_abi_ink.h"
#include "ink_common.cuh"

namespace {

constexpr int kWarps = 4;  // warps per block (one 16 x 16 tile each)

IK_DEVI uint4 ld_stream(const ik_bf16* p) { return __ldcs(reinterpret_cast<const uint4*>(p)); }
IK_DEVI uint4 ld_act(const ik_bf16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// one k16 step formed from zero, then added to the f32 accumulator with ordinary fp32 adds: accumulating the whole K inside the
// mma (acc as its C operand) loses low bits in the tensor core's accumulate step (moe_grouped.cu measured it; the admission
// sweep here saw rel_l2 9e-4 that way against 1e-4 for cuBLAS)
IK_DEVI void mma_add(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    float t[4] = {0.f, 0.f, 0.f, 0.f};
    ik_mma_bf16_16816(t, a0, a1, a2, a3, b0, b1);
#pragma unroll
    for (int v = 0; v < 4; ++v) c[v] += t[v];
}

__global__ void __launch_bounds__(kWarps * IK_WARP) ink_mma_gemv_kernel(InkSegs sg, int T, int D) {
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int tile = blockIdx.x * kWarps + warp;
    int j = 0;
    while (j + 1 < sg.nseg && tile >= sg.t0[j + 1]) ++j;
    if (tile >= sg.t0[sg.nseg]) return;
    const InkSeg s = sg.s[j];
    const int n0 = (tile - sg.t0[j]) * 16;
    const int m0 = blockIdx.y * 16;
    const int g = lane >> 2, c = lane & 3;
    const int ra = min(n0 + g, s.n - 1), rb = min(n0 + g + 8, s.n - 1);  // rows past n are read, never stored
    const bool two = T - m0 > 8;
    const int xa = m0 + g, xb = m0 + 8 + g;
    const bool ha = xa < T, hb = two && xb < T;
    const ik_bf16* wa = s.w + (int64_t)ra * D + 16 * c;
    const ik_bf16* wb = s.w + (int64_t)rb * D + 16 * c;
    const ik_bf16* pa = s.x + (int64_t)(ha ? xa : 0) * D + 16 * c;
    const ik_bf16* pb = s.x + (int64_t)(hb ? xb : 0) * D + 16 * c;
    float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4 z = make_uint4(0, 0, 0, 0);
    const int kmain = D & ~63;
    // main loop: 64-wide k blocks, the next block's loads issued before this block's mma
    if (kmain > 0) {
        uint4 A0 = ld_stream(wa), A1 = ld_stream(wa + 8), B0 = ld_stream(wb), B1 = ld_stream(wb + 8);
        uint4 X0 = ha ? ld_act(pa) : z, X1 = ha ? ld_act(pa + 8) : z;
        uint4 Y0 = hb ? ld_act(pb) : z, Y1 = hb ? ld_act(pb + 8) : z;
        for (int kb = 0; kb < kmain; kb += 64) {
            const uint32_t a[8] = {A0.x, A0.y, A0.z, A0.w, A1.x, A1.y, A1.z, A1.w};
            const uint32_t b[8] = {B0.x, B0.y, B0.z, B0.w, B1.x, B1.y, B1.z, B1.w};
            const uint32_t x[8] = {X0.x, X0.y, X0.z, X0.w, X1.x, X1.y, X1.z, X1.w};
            const uint32_t y[8] = {Y0.x, Y0.y, Y0.z, Y0.w, Y1.x, Y1.y, Y1.z, Y1.w};
            const int kn = kb + 64;
            if (kn < kmain) {
                A0 = ld_stream(wa + kn), A1 = ld_stream(wa + kn + 8), B0 = ld_stream(wb + kn), B1 = ld_stream(wb + kn + 8);
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
    // tail: 16-wide k steps (D % 16 == 0), lane c's four k are kb + 4c .. kb + 4c + 3
    for (int kb = kmain; kb < D; kb += 16) {
        const int o = kb - 16 * c + 4 * c;  // the row pointers are offset by 16c
        const uint2 A = __ldcs(reinterpret_cast<const uint2*>(wa + o)), B = __ldcs(reinterpret_cast<const uint2*>(wb + o));
        const uint2 X = ha ? __ldg(reinterpret_cast<const uint2*>(pa + o)) : make_uint2(0, 0);
        const uint2 Y = hb ? __ldg(reinterpret_cast<const uint2*>(pb + o)) : make_uint2(0, 0);
        mma_add(c0, A.x, B.x, A.y, B.y, X.x, X.y);
        if (two) mma_add(c1, A.x, B.x, A.y, B.y, Y.x, Y.y);
    }
    // c[0]: (row g, col 2c), c[1]: (g, 2c+1), c[2]: (g+8, 2c), c[3]: (g+8, 2c+1); rows = weight rows, cols = activation rows
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        if (nt == 1 && !two) break;
        const float* cc = nt ? c1 : c0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int m = m0 + nt * 8 + 2 * c + (e & 1);
            const int n = n0 + g + (e >> 1) * 8;
            if (m < T && n < s.n) s.out[(int64_t)m * s.n + n] = ik_f2b(cc[e]);
        }
    }
}

}  // namespace

// shared with ink_mix.cu (the controller's h projection)
int ink_mma_gemv(const InkSegs& in, int T, int D, cudaStream_t st) {
    InkSegs sg = in;
    sg.t0[0] = 0;
    for (int j = 0; j < sg.nseg; ++j) sg.t0[j + 1] = sg.t0[j] + (sg.s[j].n + 15) / 16;
    const int tiles = sg.t0[sg.nseg];
    if (tiles == 0 || T == 0) return IK_OK;
    if (D % 16 != 0) return IK_ERR_SHAPE;
    const dim3 grid((unsigned)((tiles + kWarps - 1) / kWarps), (unsigned)((T + 15) / 16));
    if (grid.y > 65535) return IK_ERR_SHAPE;
    ink_mma_gemv_kernel<<<grid, kWarps * IK_WARP, 0, st>>>(sg, T, D);
    return ik_launch_status();
}

namespace {

// mode 1: the tensor-core GEMV at any T; 2: cuBLAS
int in_proj(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, int mode, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 2));
    ik_tensor* o[3] = {pq, pk, pv};
    const int64_t S = x->shape[0], T = x->shape[1], D = x->shape[2];
    if (!(S == 1 || S == 3) || w->shape[1] != D) return IK_ERR_SHAPE;
    int64_t nsum = 0;
    for (int j = 0; j < 3; ++j) {
        IK_TRY(ik_check(o[j], IK_DTYPE_BF16, 2));
        if (o[j]->shape[0] != T) return IK_ERR_SHAPE;
        nsum += o[j]->shape[1];
    }
    if (nsum != w->shape[0]) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    cudaStream_t st = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const ik_bf16* wp = (const ik_bf16*)w->data;
    const bool mma_ok = D % 16 == 0 && ik_aligned16(x->data) && ik_aligned16(w->data) && D <= 0x7fffffff && T <= 0x7fffffff;
    if (mode == 1) {
        if (!mma_ok) return IK_ERR_SHAPE;
        InkSegs sg{};
        sg.nseg = 3;
        int64_t r0 = 0;
        for (int j = 0; j < 3; ++j) {
            const int64_t xi = j < S - 1 ? j : S - 1;
            sg.s[j] = InkSeg{xp + xi * T * D, wp + r0 * D, (ik_bf16*)o[j]->data, (int)o[j]->shape[1]};
            r0 += o[j]->shape[1];
        }
        return ink_mma_gemv(sg, (int)T, (int)D, st);
    }
    // prefill: three cuBLAS GEMMs on views
    int64_t r0 = 0;
    for (int j = 0; j < 3; ++j) {
        const int64_t xi = j < S - 1 ? j : S - 1;
        ik_tensor xv = *x, wv = *w;
        xv.data = (void*)(xp + xi * T * D);
        xv.ndim = 2;
        xv.shape[0] = T, xv.shape[1] = D, xv.stride[0] = D, xv.stride[1] = 1;
        wv.data = (void*)(wp + r0 * D);
        wv.shape[0] = o[j]->shape[1];
        IK_TRY(ik_matmul_bf16(&xv, &wv, o[j], stream));
        r0 += o[j]->shape[1];
    }
    return IK_OK;
}

}  // namespace

extern "C" {

int ik_ink_in_proj_bf16_mma(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream) {
    return in_proj(x, w, pq, pk, pv, 1, stream);
}
int ik_ink_in_proj_bf16_mma_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream) {
    return in_proj(x, w, pq, pk, pv, 1, stream);
}
int ik_ink_in_proj_bf16_cublas(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream) {
    return in_proj(x, w, pq, pk, pv, 2, stream);
}

}  // extern "C"
