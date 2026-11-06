// DFlash2 draft kernels on Metal (lane dm; hip twin kernels/hip/dflash.hip, semantics crates/intarsia-core/src/ops.rs).
//
// dyn_conv_bf16 (`Op::DynConv`): the draft's causal conv along the block rows with per-row dynamic taps.
//   x [T,H], k [T,K*G] (row t: K taps x G groups, tap-major), base [K,H] -> out [T,H]
//   out[t,c] = sum over taps o < K with t >= o of (base[o,c] + k[t, o*G + c / (H/G)]) * x[t-o, c]
// f32 accumulation (taps in ascending order), one rounding to bf16. One thread per (t, c): the op is tiny (T <= 16 rows of
// H = 5120 in use), so there is no tiling.
//
// Kernel arguments (`OpP`, common.h): a = T * H, b = H, c = K (taps), d = G. s0..s3 = leading elements to skip of x, k, base, out.

#include "common.h"

kernel void dyn_conv_bf16(device const ushort* x [[buffer(0)]],
                          device const ushort* k [[buffer(1)]],
                          device const ushort* base [[buffer(2)]],
                          device ushort* out [[buffer(3)]],
                          constant OpP& p [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a) return;
    const uint h = p.b, taps = p.c, g = p.d;
    const uint t = gid / h, c = gid % h;
    const uint grp = c / (h / g);
    const uint n = min(t + 1, taps);
    float acc = 0.0f;
    for (uint o = 0; o < n; ++o)
        acc += (bf2f(base[p.s2 + (ulong)o * h + c]) + bf2f(k[p.s1 + (ulong)t * taps * g + (ulong)o * g + grp])) *
               bf2f(x[p.s0 + (ulong)(t - o) * h + c]);
    out[p.s3 + gid] = f2bf(acc);
}

// ---- logits_rows_scalar_m<M>: M-row `Logits` (f32 out) for weights or activations that are not 16-byte aligned ---------------
// The 1-row kernel for those operands is `gemv_f32o_scalar` (gemv_f32out.metal): lane l of the simdgroup that owns output row n
// sums fma(w[n, i], x[i], acc) over i = l, l + 32, ... in ascending order, then simd_sum. The Qwen3.8-27B lm_head sits at byte
// 8 mod 16 of its file, so every head call takes that kernel, and a verify / draft head of M rows was M such launches (M x 5.8 ms).
// This kernel keeps that per-row order exactly (output row r is bitwise the 1-row launch on activation row r) and reads each
// weight once for all M rows: the activations are staged in threadgroup memory per K tile of LRS_T elements (a multiple of 32,
// so element i stays with lane i % 32), each simdgroup owns LRS_P output rows. GemvParams as gemv_f32out.metal (y_skip 0).
// Grid: ceil(N / (LRS_SG * LRS_P)) threadgroups of LRS_SG * 32 threads.

struct LrsParams {
    uint k;
    uint n;
    uint w_skip;
    uint x_skip;
    uint y_skip;
    uint pad0;
    uint pad1;
    uint pad2;
};

constant constexpr uint LRS_T = 512;
constant constexpr uint LRS_SG = 8;
constant constexpr uint LRS_P = 4;

template <uint M>
kernel void logits_rows_scalar(device const ushort* W [[buffer(0)]],
                               device const ushort* X [[buffer(1)]],
                               device float* Y [[buffer(2)]],
                               constant LrsParams& p [[buffer(3)]],
                               uint tg [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ushort xs[M * LRS_T];
    const uint row0 = (tg * LRS_SG + sg) * LRS_P;
    device const ushort* w[LRS_P];
    for (uint q = 0; q < LRS_P; ++q) w[q] = W + p.w_skip + (ulong)min(row0 + q, p.n - 1) * p.k;
    float acc[LRS_P][M];
    for (uint q = 0; q < LRS_P; ++q)
        for (uint r = 0; r < M; ++r) acc[q][r] = 0.0f;
    for (uint t0 = 0; t0 < p.k; t0 += LRS_T) {
        const uint tn = min(LRS_T, p.k - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < M * LRS_T; e += LRS_SG * 32) {
            const uint r = e / LRS_T, i = e % LRS_T;
            xs[e] = i < tn ? X[p.x_skip + (ulong)r * p.k + t0 + i] : (ushort)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < tn; i += 32) {
            float wv[LRS_P];
            for (uint q = 0; q < LRS_P; ++q) wv[q] = bf2f(w[q][t0 + i]);
            for (uint r = 0; r < M; ++r) {
                const float xv = bf2f(xs[r * LRS_T + i]);
                for (uint q = 0; q < LRS_P; ++q) acc[q][r] = fma(wv[q], xv, acc[q][r]);
            }
        }
    }
    for (uint q = 0; q < LRS_P; ++q) {
        for (uint r = 0; r < M; ++r) {
            const float s = simd_sum(acc[q][r]);
            if (lane == 0 && row0 + q < p.n) Y[p.y_skip + (ulong)r * p.n + row0 + q] = s;
        }
    }
}

#define LRS_INST(M) \
    template [[host_name("logits_rows_scalar_m" #M)]] kernel void logits_rows_scalar<M>( \
        device const ushort*, device const ushort*, device float*, constant LrsParams&, uint, uint, uint, uint);
LRS_INST(2)
LRS_INST(3)
LRS_INST(4)
LRS_INST(5)
LRS_INST(6)
LRS_INST(7)
LRS_INST(8)
LRS_INST(9)
LRS_INST(10)
LRS_INST(11)
LRS_INST(12)
LRS_INST(13)
LRS_INST(14)
LRS_INST(15)
LRS_INST(16)
