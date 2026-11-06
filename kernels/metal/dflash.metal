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

