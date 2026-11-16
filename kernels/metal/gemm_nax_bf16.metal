// bf16 GEMM on the M5 neural accelerators: C[M,N] = A[M,K] . W[N,K]^T, bf16 in, bf16 out, one rounding at the store.
// Compiled -std=metal4.0 into intarsia_metal4.metallib (build.rs METAL4_SOURCES); it needs Metal Performance
// Primitives (mpp::tensor_ops::matmul2d), as MLX's NAX GEMM does (mlx/backend/metal/kernels/steel/gemm/nax.h).
//
// Operands are read straight from device memory by the primitive (no threadgroup staging). W is read as stored
// ([N,K], transpose_right), nothing is repacked. M and N edges go through sliced tensors (the primitive bounds-checks
// them against the tensor extents), K is dynamic and its tail is zero-filled. The tile (BM x BN) is a property of the
// entry point; the P1 spike measured every tile bitwise equal (the K reduction order does not depend on the tile), so
// the host may pick the tile by shape without a row's result depending on the call it arrives in.
//
// Strict precision only: matmul2d_descriptor::relaxed_precision = false. The spike measured `true` bitwise equal to
// `false` on bf16 operands and no faster (docs/results/2026-10-14-metal-prefill-p1-spike.md), so it is not built.
//
// bind (gemm_nax_t): 0 A [M,K], 1 W [N,K], 2 C [M,N], 3 NaxP. Tensors are bound at 4-byte aligned offsets and the
// leftover 0 or 1 bf16 element comes in as a skip (same convention as the other dense kernels), so operands may sit at
// any 2-byte address. Grid: (ceil(N/BN) << swz_log, ceil(ceil(M/BM) / 2^swz_log)) threadgroups of NSG*32 threads.
// bind (gemm_nax_pair_t, Op::MatmulPair for M > 1): 0 A, 1 W0 [N0,K], 2 W1 [N1,K], 3 C0, 4 C1, 5 NaxPairP; one grid of
// (tiles_n0 + tiles_n1, tiles_m) threadgroups, the first tiles_n0 columns of threadgroups do W0 / C0. Bitwise two
// gemm_nax_t launches.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct NaxP {
  uint m, n, k, swz_log;
  uint s0, s1, s2, pad;
};

struct NaxPairP {
  uint m, n0, n1, k;
  uint s_a, s_w0, s_w1, s_c0;
  uint s_c1, pad0, pad1, pad2;
};

// One BM x BN output tile at (m0, n0) of C[M,N] = A[M,K] . W[N,K]^T by NSG simdgroups; matmul2d does the K loop and
// the in-tile layout and writes the bf16 destination tensor. (A float cooperative destination rounded to bf16 in the
// kernel and stored element by element gave bit-identical results and was slower, see the spike doc.)
template <int BM, int BN, int NSG>
inline void nax_tile(device bfloat* A, device bfloat* W, device bfloat* C, int M, int N, int K, int m0, int n0) {
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A, dextents<int, 2>(K, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W, dextents<int, 2>(K, N));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tC(C, dextents<int, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  if (m0 + BM <= M && n0 + BN <= N) {
    // interior tile: static extents, no bounds checks
    auto sA = tA.template slice<dynamic_extent, BM>(0, m0);
    auto sW = tW.template slice<dynamic_extent, BN>(0, n0);
    auto sC = tC.template slice<BN, BM>(n0, m0);
    op.run(sA, sW, sC);
  } else {
    auto sA = tA.slice(0, m0);
    auto sW = tW.slice(0, n0);
    auto sC = tC.slice(n0, m0);
    op.run(sA, sW, sC);
  }
}

template <int BM, int BN, int NSG>
kernel void gemm_nax_t(device bfloat* A [[buffer(0)]], device bfloat* W [[buffer(1)]], device bfloat* C [[buffer(2)]],
                       constant NaxP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), N = int(p.n), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int ty = int((tg.y << p.swz_log) + (tg.x & ((1u << p.swz_log) - 1)));
  const int tx = int(tg.x >> p.swz_log);
  if (tx >= tiles_n || ty >= tiles_m) return;
  nax_tile<BM, BN, NSG>(A + p.s0, W + p.s1, C + p.s2, M, N, K, ty * BM, tx * BN);
}

template <int BM, int BN, int NSG>
kernel void gemm_nax_pair_t(device bfloat* A [[buffer(0)]], device bfloat* W0 [[buffer(1)]],
                            device bfloat* W1 [[buffer(2)]], device bfloat* C0 [[buffer(3)]],
                            device bfloat* C1 [[buffer(4)]], constant NaxPairP& p [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM;
  const int tiles_n0 = (int(p.n0) + BN - 1) / BN, tiles_n1 = (int(p.n1) + BN - 1) / BN;
  const int tx = int(tg.x), ty = int(tg.y);
  if (tx >= tiles_n0 + tiles_n1 || ty >= tiles_m) return;
  if (tx < tiles_n0) {
    nax_tile<BM, BN, NSG>(A + p.s_a, W0 + p.s_w0, C0 + p.s_c0, M, int(p.n0), K, ty * BM, tx * BN);
  } else {
    nax_tile<BM, BN, NSG>(A + p.s_a, W1 + p.s_w1, C1 + p.s_c1, M, int(p.n1), K, ty * BM, (tx - tiles_n0) * BN);
  }
}

#define NAX_T(NAME, PAIR, BM, BN, NSG) \
  template [[host_name(NAME)]] kernel void gemm_nax_t<BM, BN, NSG>( \
      device bfloat*, device bfloat*, device bfloat*, constant NaxP&, uint2); \
  template [[host_name(PAIR)]] kernel void gemm_nax_pair_t<BM, BN, NSG>( \
      device bfloat*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, constant NaxPairP&, uint2);

// Keep in step with NAX_VARIANTS in src/gemm_nax.rs.
NAX_T("gemm_nax_t64x64w4", "gemm_nax_pair_t64x64w4", 64, 64, 4)
NAX_T("gemm_nax_t128x64w4", "gemm_nax_pair_t128x64w4", 128, 64, 4)
// P4/P5 tiles for the 128 <= M <= 512 chunks (docs/results/2026-10-15-metal-prefill-p45.md): 32x32 on one simdgroup for
// the narrow k + v pair, 32 rows on two simdgroups for M tails that leave a 64-row tile mostly empty, 64x64 on two
// simdgroups otherwise.
NAX_T("gemm_nax_t32x32w1", "gemm_nax_pair_t32x32w1", 32, 32, 1)
NAX_T("gemm_nax_t32x128w2", "gemm_nax_pair_t32x128w2", 32, 128, 2)
NAX_T("gemm_nax_t64x64w2", "gemm_nax_pair_t64x64w2", 64, 64, 2)

// Lane mgemm (docs/results/2026-11-16-mgemm.md), form (b): K-split. One dispatch per K range [k0, k1); the partial sum of
// the output tile is carried between dispatches in an f32 buffer P [M,N]. The first pass multiplies, the next ones load
// P into the float cooperative destination and multiply-accumulate, the last one rounds to bf16 into C (the same RNE
// store as the float-destination variant the P1 spike measured bitwise equal to gemm_nax_t). A and W keep their
// [M,K] / [N,K] layouts: a K range is a strided tensor (row stride K) starting at column k0.
// bind: 0 A, 1 W, 2 C bf16, 3 P f32, 4 NaxKsP. flags: bit 0 = load P (accumulate), bit 1 = last pass (store C, else P).
struct NaxKsP {
  uint m, n, k, swz_log;
  uint k0, k1, flags, pad;
};

template <int BM, int BN, int NSG, bool ACC, typename TA, typename TW, typename TP, typename TC>
METAL_FUNC void ks_tile(TA sA, TW sW, TP sP, TC sC, bool last) {
  constexpr auto mode = ACC ? matmul2d_descriptor::mode::multiply_accumulate : matmul2d_descriptor::mode::multiply;
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false, mode);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto ct = op.template get_destination_cooperative_tensor<TA, TW, float>();
  if constexpr (ACC) {
    ct.load(sP);
  }
  op.run(sA, sW, ct);
  if (last) {
    auto cb = op.template get_destination_cooperative_tensor<TA, TW, bfloat>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
      if (ct.is_valid_element(i)) cb[i] = static_cast<bfloat>(ct[i]);
    }
    cb.store(sC);
  } else {
    ct.store(sP);
  }
}

template <int BM, int BN, int NSG>
kernel void gemm_nax_ks_t(device bfloat* A [[buffer(0)]], device bfloat* W [[buffer(1)]], device bfloat* C [[buffer(2)]],
                          device float* P [[buffer(3)]], constant NaxKsP& p [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), N = int(p.n), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int ty = int((tg.y << p.swz_log) + (tg.x & ((1u << p.swz_log) - 1)));
  const int tx = int(tg.x >> p.swz_log);
  if (tx >= tiles_n || ty >= tiles_m) return;
  const int m0 = ty * BM, n0 = tx * BN, k0 = int(p.k0), kl = int(p.k1) - int(p.k0);
  const bool acc = (p.flags & 1u) != 0, last = (p.flags & 2u) != 0;
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A + k0, dextents<int, 2>(kl, M), array<int, 2>{1, K});
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W + k0, dextents<int, 2>(kl, N), array<int, 2>{1, K});
  tensor<device float, dextents<int, 2>, tensor_inline> tP(P, dextents<int, 2>(N, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tC(C, dextents<int, 2>(N, M));
  if (m0 + BM <= M && n0 + BN <= N) {
    auto sA = tA.template slice<dynamic_extent, BM>(0, m0);
    auto sW = tW.template slice<dynamic_extent, BN>(0, n0);
    auto sP = tP.template slice<BN, BM>(n0, m0);
    auto sC = tC.template slice<BN, BM>(n0, m0);
    if (acc) ks_tile<BM, BN, NSG, true>(sA, sW, sP, sC, last);
    else ks_tile<BM, BN, NSG, false>(sA, sW, sP, sC, last);
  } else {
    auto sA = tA.slice(0, m0);
    auto sW = tW.slice(0, n0);
    auto sP = tP.slice(n0, m0);
    auto sC = tC.slice(n0, m0);
    if (acc) ks_tile<BM, BN, NSG, true>(sA, sW, sP, sC, last);
    else ks_tile<BM, BN, NSG, false>(sA, sW, sP, sC, last);
  }
}

#define NAX_KS_T(NAME, BM, BN, NSG) \
  template [[host_name(NAME)]] kernel void gemm_nax_ks_t<BM, BN, NSG>( \
      device bfloat*, device bfloat*, device bfloat*, device float*, constant NaxKsP&, uint2);

NAX_KS_T("gemm_nax_ks_t64x64w2", 64, 64, 2)
NAX_KS_T("gemm_nax_ks_t128x64w4", 128, 64, 4)
NAX_KS_T("gemm_nax_ks_t32x128w2", 32, 128, 2)
NAX_KS_T("gemm_nax_ks_t64x64w4", 64, 64, 4)
