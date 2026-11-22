// InkInProj prefill on the M5 neural accelerators (lane ink-m-mix): y_j[T, n_j] = x[min(j, S-1)] . W_j^T for the three
// stacked weight segments in one grid, bf16 in, bf16 out, one rounding at the store. The tile is gemm_nax_bf16.metal's
// nax_tile (same descriptor, strict precision), so each segment is bitwise a `matmul_bf16_nax` launch on that segment.
// Compiled -std=metal4.0 into intarsia_metal4.metallib (build.rs METAL4_SOURCES).
//
// bind: 0 X [S, T, K], 1 W [n0+n1+n2, K], 2 Y0, 3 Y1, 4 Y2, 5 InkNaxP. Grid: (tiles_n0 + tiles_n1 + tiles_n2, tiles_m)
// threadgroups of NSG * 32 threads. Tensors at 4-byte aligned offsets with a leftover-element skip.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct InkNaxP {
  uint m, k, n0, n1;
  uint n2, xo1, xo2, s_x;
  uint s_w, s_y0, s_y1, s_y2;
};

template <int BM, int BN, int NSG>
inline void ink_nax_tile(device bfloat* A, device bfloat* W, device bfloat* C, int M, int N, int K, int m0, int n0) {
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A, dextents<int, 2>(K, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W, dextents<int, 2>(K, N));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tC(C, dextents<int, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  if (m0 + BM <= M && n0 + BN <= N) {
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
kernel void ink_in_proj_nax_t(device bfloat* X [[buffer(0)]], device bfloat* W [[buffer(1)]], device bfloat* Y0 [[buffer(2)]],
                              device bfloat* Y1 [[buffer(3)]], device bfloat* Y2 [[buffer(4)]],
                              constant InkNaxP& p [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM;
  const int t0 = (int(p.n0) + BN - 1) / BN, t1 = (int(p.n1) + BN - 1) / BN, t2 = (int(p.n2) + BN - 1) / BN;
  const int tx = int(tg.x), ty = int(tg.y);
  if (tx >= t0 + t1 + t2 || ty >= tiles_m) return;
  device bfloat* A = X + p.s_x;
  device bfloat* Wb = W + p.s_w;
  if (tx < t0) {
    ink_nax_tile<BM, BN, NSG>(A, Wb, Y0 + p.s_y0, M, int(p.n0), K, ty * BM, tx * BN);
  } else if (tx < t0 + t1) {
    ink_nax_tile<BM, BN, NSG>(A + p.xo1, Wb + (ulong)p.n0 * K, Y1 + p.s_y1, M, int(p.n1), K, ty * BM, (tx - t0) * BN);
  } else {
    ink_nax_tile<BM, BN, NSG>(A + p.xo2, Wb + (ulong)(p.n0 + p.n1) * K, Y2 + p.s_y2, M, int(p.n2), K, ty * BM,
                              (tx - t0 - t1) * BN);
  }
}

#define INK_NAX_T(NAME, BM, BN, NSG) \
  template [[host_name(NAME)]] kernel void ink_in_proj_nax_t<BM, BN, NSG>( \
      device bfloat*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, constant InkNaxP&, uint2);

// The gemm_nax_bf16.metal tiles (src/gemm_nax.rs NAX_VARIANTS), so gemm_nax::pick_variant chooses the tile.
INK_NAX_T("ink_in_proj_nax_t64x64w4", 64, 64, 4)
INK_NAX_T("ink_in_proj_nax_t128x64w4", 128, 64, 4)
INK_NAX_T("ink_in_proj_nax_t32x32w1", 32, 32, 1)
INK_NAX_T("ink_in_proj_nax_t32x128w2", 32, 128, 2)
INK_NAX_T("ink_in_proj_nax_t64x64w2", 64, 64, 2)
