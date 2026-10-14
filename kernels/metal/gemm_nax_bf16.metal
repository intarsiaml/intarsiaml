// bf16 GEMM on the M5 neural accelerators: C[M,N] = A[M,K] . W[N,K]^T, bf16 in, bf16 out, one rounding at the store.
// Compiled -std=metal4.0 into intarsia_metal4.metallib (build.rs METAL4_SOURCES); it needs Metal Performance
// Primitives (mpp::tensor_ops::matmul2d), as MLX's NAX GEMM does (mlx/backend/metal/kernels/steel/gemm/nax.h).
//
// Operands are read straight from device memory by the primitive (no threadgroup staging). W is read as stored
// ([N,K], transpose_right), nothing is repacked. M and N edges go through sliced tensors (the primitive bounds-checks
// them against the tensor extents), K is dynamic. One tile configuration per entry point and the same for all M, so a
// row's result does not depend on the call's M.
//
// bind: 0 A [M,K], 1 W [N,K], 2 C [M,N], 3 NaxP. Tensors are bound at 4-byte aligned offsets and the leftover 0 or 1
// bf16 element comes in as a skip (same convention as the other dense kernels), so operands may sit at any 2-byte
// address. Grid: (ceil(N/BN) << swz_log, ceil(ceil(M/BM) / 2^swz_log)) threadgroups of NSG*32 threads.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct NaxP {
  uint m, n, k, swz_log;
  uint s0, s1, s2, pad;
};

// One output tile of BM x BN per threadgroup of NSG simdgroups; matmul2d does the K loop and the in-tile layout and
// writes the bf16 destination tensor. (A float cooperative destination rounded to bf16 in the kernel and stored
// element by element gave bit-identical results and was slower, see docs/results/2026-10-14-metal-prefill-p1-spike.md.)
template <int BM, int BN, int NSG, bool RELAXED, class TA, class TW, class TC>
inline void run_tile(TA sA, TW sW, TC sC) {
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, RELAXED);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  op.run(sA, sW, sC);
}

template <int BM, int BN, int NSG, bool RELAXED>
kernel void gemm_nax_t(device bfloat* A [[buffer(0)]], device bfloat* W [[buffer(1)]], device bfloat* C [[buffer(2)]],
                       constant NaxP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), N = int(p.n), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int ty = int((tg.y << p.swz_log) + (tg.x & ((1u << p.swz_log) - 1)));
  const int tx = int(tg.x >> p.swz_log);
  if (tx >= tiles_n || ty >= tiles_m) return;
  const int m0 = ty * BM, n0 = tx * BN;

  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A + p.s0, dextents<int, 2>(K, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W + p.s1, dextents<int, 2>(K, N));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tC(C + p.s2, dextents<int, 2>(N, M));
  if (m0 + BM <= M && n0 + BN <= N) {
    // interior tile: static extents, no bounds checks
    run_tile<BM, BN, NSG, RELAXED>(tA.template slice<dynamic_extent, BM>(0, m0), tW.template slice<dynamic_extent, BN>(0, n0),
                                   tC.template slice<BN, BM>(n0, m0));
  } else {
    run_tile<BM, BN, NSG, RELAXED>(tA.slice(0, m0), tW.slice(0, n0), tC.slice(n0, m0));
  }
}

#define NAX_T(NAME, BM, BN, NSG, RELAXED) \
  template [[host_name(NAME)]] kernel void gemm_nax_t<BM, BN, NSG, RELAXED>( \
      device bfloat*, device bfloat*, device bfloat*, constant NaxP&, uint2);

// Keep in step with NAX_VARIANTS in src/gemm_nax.rs.
NAX_T("gemm_nax_t64x64w4", 64, 64, 4, false)
NAX_T("gemm_nax_t128x64w4", 128, 64, 4, false)
NAX_T("gemm_nax_t64x128w4", 64, 128, 4, false)
NAX_T("gemm_nax_t128x128w8", 128, 128, 8, false)
NAX_T("gemm_nax_t64x64w4_r", 64, 64, 4, true)
NAX_T("gemm_nax_t128x64w4_r", 128, 64, 4, true)
NAX_T("gemm_nax_t64x128w4_r", 64, 128, 4, true)
NAX_T("gemm_nax_t128x128w8_r", 128, 128, 8, true)
