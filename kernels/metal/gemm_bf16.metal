// BF16 GEMM for M > 1 (and any M): Y[M,N] = X[M,K] . W[N,K]^T, bf16 in, f32 accumulate, bf16 or f32 out.
//
// Straightforward tiling, correctness first (speed is phase B): a 32x32 output tile per threadgroup of 4 simdgroups,
// each simdgroup owns a 16x16 quadrant (2x2 simdgroup_matrix<float,8,8>). K advances in steps of 32; both operand tiles
// are staged in threadgroup memory as f32 (bf16 -> f32 is exact), zero padded at the M / N / K edges, so the 8x8
// multiply-accumulates need no bounds checks. The K order is fixed, so results are run-to-run identical.
//
// bind: 0 X [M,K], 1 W [N,K], 2 Y [M,N]. a = M, b = N, c = K. s0 / s1 / s2 skips (Y: f32 output has none).

#include "common.h"

constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;

template <typename OutT>
kernel void gemm_bf16_t(device const ushort* X [[buffer(0)]],
                        device const ushort* W [[buffer(1)]],
                        device OutT* Y [[buffer(2)]],
                        constant OpP& p [[buffer(3)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[BM * BK];
    threadgroup float Bs[BN * BK];
    threadgroup float Cs[4 * 16 * 16];
    const uint M = p.a, N = p.b, K = p.c;
    const uint n0 = tgp.x * BN, m0 = tgp.y * BM;
    const uint sm = sg / 2, sn = sg % 2;

    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 128) {
            const uint r = e / BK, c = e % BK;
            const uint gk = k0 + c;
            const uint gm = m0 + r, gn = n0 + r;
            As[e] = (gm < M && gk < K) ? bf2f(X[p.s0 + (ulong)gm * K + gk]) : 0.0f;
            Bs[e] = (gn < N && gk < K) ? bf2f(W[p.s1 + (ulong)gn * K + gk]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK / 8; ++kk) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], As + (sm * 16 + i * 8) * BK + kk * 8, BK);
            // Bs is [n][k]; the multiply wants B as [k][n], i.e. the transposed load
            for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], Bs + (sn * 16 + j * 8) * BK + kk * 8, BK, ulong2(0, 0), true);
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* cs = Cs + sg * 256;
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], cs + (i * 8) * 16 + j * 8, 16);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 256; e += 32) {
        const uint gm = m0 + sm * 16 + e / 16, gn = n0 + sn * 16 + e % 16;
        if (gm < M && gn < N) store_out(Y, p.s2 + (ulong)gm * N + gn, cs[e]);
    }
}

template [[host_name("gemm_bf16_o_bf16")]] kernel void gemm_bf16_t<ushort>(
    device const ushort*, device const ushort*, device ushort*, constant OpP&, uint2, uint, uint, uint);
template [[host_name("gemm_bf16_o_f32")]] kernel void gemm_bf16_t<float>(
    device const ushort*, device const ushort*, device float*, constant OpP&, uint2, uint, uint, uint);
