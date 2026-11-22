// Prefill (rows > 16) gate + norm of the Inkling-2-Mini loop blend on the M5 neural accelerators (lane ink-prefill; launcher
// src/inkprefill.rs, hooked from inkmix.rs `blend_rows`). Same math as inkling2_mix.metal's `ink_blend_gate` + `ink_blend_norm`:
//   gn[s,t,c] = gain[s,c,:] . h[s,t,:],  fg[s,t,c] = forget[s,c,:] . h[s,t,:]          (f32, K = G)
//   y = (1 - tanh(fg)) * d + tanh(gn) * m;  out[s,t] = rA(y * inv(y) * norm[s])
// `ink_blend_gf_nax`: the two K = G products as one accelerator pass per (stream, 64-row x 64-channel tile), read straight
// from device memory (h [S, rows, G], gain / forget [S, D, G], all bf16), f32 results to scratch. `ink_blend_yn`: one
// threadgroup per (stream, row): y, the sum of squares, the norm, the bf16 output. Not bitwise the decode kernels (the
// accelerator's K order): admitted against the f64 reference like the other NAX prefill paths.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace metal;
using namespace mpp::tensor_ops;

struct InkBlendP {
    uint d, rows, s, sd;
    uint g, nsl, cs, t0;
    uint ocount, dcount, mcount, pad0;
    uint dt0, mt0, pad1, pad2;
    float eps, fpad0, fpad1, fpad2;
    uint sk[8];              // leading bf16 elements: gain, forget, norm, out
};

// bind: 0 h bf16 [S, rows, G], 1 gain bf16 [S, D, G], 2 forget, 3 gn f32 [S, rows, D], 4 fg f32 [S, rows, D], 5 InkBlendP.
// Grid (S * ceil(D / 64), ceil(rows / 64)) threadgroups of 128 threads (4 simdgroups, 2 x 2 sub-tiles of 32 x 32).
kernel void ink_blend_gf_nax(device bfloat* h [[buffer(0)]], device bfloat* gain [[buffer(1)]], device bfloat* forget [[buffer(2)]],
                             device float* gn [[buffer(3)]], device float* fg [[buffer(4)]], constant InkBlendP& p [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const int D = int(p.d), R = int(p.rows), G = int(p.g), td = (D + 63) / 64, s = int(tg.x) / td;
    const int m0 = int(tg.y) * 64 + int(sg / 2) * 32, n0 = (int(tg.x) % td) * 64 + int(sg % 2) * 32;
    if (m0 >= R || n0 >= D) return;
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tH(h + (ulong)s * R * G, dextents<int, 2>(G, R));
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tG(gain + p.sk[0] + (ulong)s * D * G, dextents<int, 2>(G, D));
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tF(forget + p.sk[1] + (ulong)s * D * G, dextents<int, 2>(G, D));
    tensor<device float, dextents<int, 2>, tensor_inline> tGN(gn + (ulong)s * R * D, dextents<int, 2>(D, R));
    tensor<device float, dextents<int, 2>, tensor_inline> tFG(fg + (ulong)s * R * D, dextents<int, 2>(D, R));
    constexpr auto desc = matmul2d_descriptor(32, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroup> op;
    auto sH = tH.slice(0, m0);
    auto sG = tG.slice(0, n0);
    auto sF = tF.slice(0, n0);
    auto oG = tGN.slice(n0, m0);
    auto oF = tFG.slice(n0, m0);
    op.run(sH, sG, oG);
    op.run(sH, sF, oF);
}

// bind: 0 dest f32 [Sd, dcount, D], 1 msg f32 [S, mcount, D], 2 gn f32, 3 fg f32, 4 norm bf16 [S, D], 5 out bf16
// [S, ocount, D], 6 InkBlendP. Grid S * rows threadgroups of 256 threads.
kernel void ink_blend_yn(device const float* dest [[buffer(0)]], device const float* msg [[buffer(1)]],
                         device const float* gn [[buffer(2)]], device const float* fg [[buffer(3)]],
                         device const ushort* nrm [[buffer(4)]], device ushort* out [[buffer(5)]],
                         constant InkBlendP& p [[buffer(6)]], uint gidx [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const uint s = gidx / p.rows, r = gidx % p.rows, d = p.d;
    device const float* dr = dest + ((ulong)(p.sd == 1 ? 0 : s) * p.dcount + p.dt0 + r) * d;
    device const float* mr = msg + ((ulong)s * p.mcount + p.mt0 + r) * d;
    device const float* gr = gn + ((ulong)s * p.rows + r) * d;
    device const float* fr = fg + ((ulong)s * p.rows + r) * d;
    constexpr uint NE = 12;  // D <= 256 * 12 = 3072
    float y[NE];
    float ss = 0.f;
    for (uint j = 0; j < NE; ++j) {
        const uint c = tid + 256u * j;
        y[j] = 0.f;
        if (c < d) {
            y[j] = (1.0f - precise::tanh(fr[c])) * dr[c] + precise::tanh(gr[c]) * mr[c];
            ss = fma(y[j], y[j], ss);
        }
    }
    ss = simd_sum(ss);
    if (lane == 0) red[sgi] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (uint k = 0; k < 8; ++k) tot += red[k];
    const float inv = precise::rsqrt(tot / float(d) + p.eps);
    device ushort* orow = out + p.sk[3] + ((ulong)s * p.ocount + p.t0 + r) * d;
    for (uint j = 0; j < NE; ++j) {
        const uint c = tid + 256u * j;
        if (c < d) orow[c] = f2bf(y[j] * inv * bf2f(nrm[p.sk[2] + (ulong)s * d + c]));
    }
}
