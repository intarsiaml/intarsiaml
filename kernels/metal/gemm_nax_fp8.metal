// fp8 projection GEMM on the M5 neural accelerators: C[M,N] = (A[M,K] . Wb[N,K]^T) * (ws * is), A = q(x) as bf16 (the
// `fp8_xprep_bf16` pre-pass, exact), Wb = the e4m3 weights decoded to bf16 by `fp8_w_to_bf16` (fp8.metal; exact), both in
// the plan scratch, C bf16 or f32. Compiled -std=metal4.0 into intarsia_metal4.metallib (build.rs METAL4_SOURCES); the
// matmul2d setup is gemm_nax_bf16.metal's (device operands read by the primitive, W as stored [N,K] with transpose_right,
// edges through sliced tensors, K dynamic with a zero-filled tail, strict precision).
//
// What differs from the bf16 NAX GEMM: the destination is a float cooperative tensor, scaled by the weight / input scale
// product in registers and then stored (bf16 round-to-nearest-even or f32), where the tiled `matmul_fp8_mma` scales the
// float accumulator the same way (cs[e] * scale) before `store_out`. Products of the bf16 operands are exact, so the only
// change against `matmul_fp8_mma` is the order in which the accelerator accumulates the K sum (docs/results/2026-10-27-
// nxp-metal-nax-prefill.md).
//
// bind: 0 A [M,K] bf16, 1 Wb [N,K] bf16, 2 scales f32 [2] (ws, is; is > 0 W8A8 static), 3 C [M,N], 4 NaxFp8P. Grid and
// threadgroup as gemm_nax_t: (ceil(N/BN) << swz_log, ceil(ceil(M/BM) / 2^swz_log)) threadgroups of NSG*32 threads.
// The `gemm_nax_fp8x_*` instances (XSEL, `matmul_fp8_nax` only; lane ink-fp8 brief 22) also bind 5 X [M,K] bf16, the
// call's x: A is q(x) only for W8A8 (`fp8_xprep_bf16_a8` writes nothing for W8A16, where q(x) = x), so W8A16 reads x
// itself: the same operand bits without the copy. MXFP4 / NVFP4 use the plain instances (their A / scales differ).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct NaxFp8P {
  uint m, n, k, swz_log;
  uint s_c, s_x, pad1, pad2;
};

// run the product into a float cooperative destination, scale it in registers and store it to sC (f32 as is, bf16 through a
// bf16 cooperative tensor of the same layout, round-to-nearest-even like `f2bf`)
template <bool F32, typename Op, typename TA, typename TW, typename TC>
METAL_FUNC void epilogue(thread Op& op, TA sA, TW sW, TC sC, float scale) {
  auto ct = op.template get_destination_cooperative_tensor<TA, TW, float>();
  op.run(sA, sW, ct);
  if constexpr (F32) {
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
      if (ct.is_valid_element(i)) ct[i] *= scale;
    }
    ct.store(sC);
  } else {
    auto cb = op.template get_destination_cooperative_tensor<TA, TW, bfloat>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
      if (ct.is_valid_element(i)) cb[i] = static_cast<bfloat>(ct[i] * scale);
    }
    cb.store(sC);
  }
}

template <typename OutT, int BM, int BN, int NSG, bool XSEL>
kernel void gemm_nax_fp8_t(device bfloat* A [[buffer(0)]], device bfloat* W [[buffer(1)]],
                           device const float* sc [[buffer(2)]], device OutT* C [[buffer(3)]],
                           constant NaxFp8P& p [[buffer(4)]], device bfloat* X [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), N = int(p.n), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int ty = int((tg.y << p.swz_log) + (tg.x & ((1u << p.swz_log) - 1)));
  const int tx = int(tg.x >> p.swz_log);
  if (tx >= tiles_n || ty >= tiles_m) return;
  const int m0 = ty * BM, n0 = tx * BN;
  const float is = sc[1];
  const float scale = sc[0] * (is > 0.0f ? is : 1.0f);
  // W8A16: `fp8_xprep_bf16_a8` wrote nothing (q(x) = x), read x itself
  device bfloat* Ae = (!XSEL || is > 0.0f) ? A : X + p.s_x;

  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(Ae, dextents<int, 2>(K, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W, dextents<int, 2>(K, N));
  tensor<device OutT, dextents<int, 2>, tensor_inline> tC(C + p.s_c, dextents<int, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  if (m0 + BM <= M && n0 + BN <= N) {
    // interior tile: static extents, no bounds checks
    auto sA = tA.template slice<dynamic_extent, BM>(0, m0);
    auto sW = tW.template slice<dynamic_extent, BN>(0, n0);
    auto sC = tC.template slice<BN, BM>(n0, m0);
    epilogue<is_same_v<OutT, float>>(op, sA, sW, sC, scale);
  } else {
    auto sA = tA.slice(0, m0);
    auto sW = tW.slice(0, n0);
    auto sC = tC.slice(n0, m0);
    epilogue<is_same_v<OutT, float>>(op, sA, sW, sC, scale);
  }
}

#define NAX_FP8_T(NAME, OUT, BM, BN, NSG) NAX_FP8_TX(NAME, OUT, BM, BN, NSG, false)
#define NAX_FP8_TX(NAME, OUT, BM, BN, NSG, XSEL) \
  template [[host_name(NAME)]] kernel void gemm_nax_fp8_t<OUT, BM, BN, NSG, XSEL>( \
      device bfloat*, device bfloat*, device const float*, device OUT*, constant NaxFp8P&, device bfloat*, uint2);

// Keep in step with NAX_FP8_SHADERS in src/fp8.rs (one entry per gemm_nax::NAX_VARIANTS tile, bf16 and f32 output).
NAX_FP8_T("gemm_nax_fp8_bf16_t64x64w4", bfloat, 64, 64, 4)
NAX_FP8_T("gemm_nax_fp8_bf16_t128x64w4", bfloat, 128, 64, 4)
NAX_FP8_T("gemm_nax_fp8_bf16_t32x32w1", bfloat, 32, 32, 1)
NAX_FP8_T("gemm_nax_fp8_bf16_t32x128w2", bfloat, 32, 128, 2)
NAX_FP8_T("gemm_nax_fp8_bf16_t64x64w2", bfloat, 64, 64, 2)
NAX_FP8_T("gemm_nax_fp8_f32_t64x64w4", float, 64, 64, 4)
NAX_FP8_T("gemm_nax_fp8_f32_t128x64w4", float, 128, 64, 4)
NAX_FP8_T("gemm_nax_fp8_f32_t32x32w1", float, 32, 32, 1)
NAX_FP8_T("gemm_nax_fp8_f32_t32x128w2", float, 32, 128, 2)
NAX_FP8_T("gemm_nax_fp8_f32_t64x64w2", float, 64, 64, 2)
// Keep in step with NAX_FP8X_SHADERS in src/fp8.rs.
NAX_FP8_TX("gemm_nax_fp8x_bf16_t64x64w4", bfloat, 64, 64, 4, true)
NAX_FP8_TX("gemm_nax_fp8x_bf16_t128x64w4", bfloat, 128, 64, 4, true)
NAX_FP8_TX("gemm_nax_fp8x_bf16_t32x32w1", bfloat, 32, 32, 1, true)
NAX_FP8_TX("gemm_nax_fp8x_bf16_t32x128w2", bfloat, 32, 128, 2, true)
NAX_FP8_TX("gemm_nax_fp8x_bf16_t64x64w2", bfloat, 64, 64, 2, true)
NAX_FP8_TX("gemm_nax_fp8x_f32_t64x64w4", float, 64, 64, 4, true)
NAX_FP8_TX("gemm_nax_fp8x_f32_t128x64w4", float, 128, 64, 4, true)
NAX_FP8_TX("gemm_nax_fp8x_f32_t32x32w1", float, 32, 32, 1, true)
NAX_FP8_TX("gemm_nax_fp8x_f32_t32x128w2", float, 32, 128, 2, true)
NAX_FP8_TX("gemm_nax_fp8x_f32_t64x64w2", float, 64, 64, 2, true)

// ---- InkInProj, e4m3 stacked weights (lane ink-fp8, brief 22) --------------------------------------------------------
// y_j[T, n_j] = (x[min(j, S-1)] . Wb_j^T) * ws_j for the three stacked segments in one grid; Wb = the e4m3 stack decoded to
// bf16 once per call by `fp8_w_to_bf16` (exact) into the plan scratch; x read directly (W8A16: the InkInProj e4m3 form has
// no input scale). Per segment the tile is `gemm_nax_fp8_t`'s (same descriptor, same `epilogue`, scale ws_j * 1.0f = ws_j),
// so each segment is bitwise a `matmul_fp8_nax` launch on that segment (tests/inkmix_f8_nax.rs).
//
// bind: 0 X [S, T, K] bf16, 1 Wb [n0+n1+n2, K] bf16 (16-byte aligned), 2 scales f32 [3], 3 Y0, 4 Y1, 5 Y2, 6 InkNaxF8P.
// Grid: (tiles_n0 + tiles_n1 + tiles_n2, tiles_m) threadgroups of NSG * 32 threads.
struct InkNaxF8P {
  uint m, k, n0, n1;
  uint n2, xo1, xo2, s_x;
  uint s_y0, s_y1, s_y2, pad0;
};

template <int BM, int BN, int NSG>
inline void ink_f8_tile(device bfloat* A, device bfloat* W, device bfloat* C, int M, int N, int K, int m0, int n0,
                        float scale) {
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A, dextents<int, 2>(K, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W, dextents<int, 2>(K, N));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tC(C, dextents<int, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  if (m0 + BM <= M && n0 + BN <= N) {
    auto sA = tA.template slice<dynamic_extent, BM>(0, m0);
    auto sW = tW.template slice<dynamic_extent, BN>(0, n0);
    auto sC = tC.template slice<BN, BM>(n0, m0);
    epilogue<false>(op, sA, sW, sC, scale);
  } else {
    auto sA = tA.slice(0, m0);
    auto sW = tW.slice(0, n0);
    auto sC = tC.slice(n0, m0);
    epilogue<false>(op, sA, sW, sC, scale);
  }
}

template <int BM, int BN, int NSG>
kernel void ink_in_proj_f8_nax_t(device bfloat* X [[buffer(0)]], device bfloat* W [[buffer(1)]],
                                 device const float* sc [[buffer(2)]], device bfloat* Y0 [[buffer(3)]],
                                 device bfloat* Y1 [[buffer(4)]], device bfloat* Y2 [[buffer(5)]],
                                 constant InkNaxF8P& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM;
  const int t0 = (int(p.n0) + BN - 1) / BN, t1 = (int(p.n1) + BN - 1) / BN, t2 = (int(p.n2) + BN - 1) / BN;
  const int tx = int(tg.x), ty = int(tg.y);
  if (tx >= t0 + t1 + t2 || ty >= tiles_m) return;
  device bfloat* A = X + p.s_x;
  // the pack launch's scale expression: ws * (is > 0 ? is : 1) with is = 0
  if (tx < t0) {
    ink_f8_tile<BM, BN, NSG>(A, W, Y0 + p.s_y0, M, int(p.n0), K, ty * BM, tx * BN, sc[0] * 1.0f);
  } else if (tx < t0 + t1) {
    ink_f8_tile<BM, BN, NSG>(A + p.xo1, W + (ulong)p.n0 * K, Y1 + p.s_y1, M, int(p.n1), K, ty * BM, (tx - t0) * BN,
                             sc[1] * 1.0f);
  } else {
    ink_f8_tile<BM, BN, NSG>(A + p.xo2, W + (ulong)(p.n0 + p.n1) * K, Y2 + p.s_y2, M, int(p.n2), K, ty * BM,
                             (tx - t0 - t1) * BN, sc[2] * 1.0f);
  }
}

#define INK_F8_NAX_T(NAME, BM, BN, NSG) \
  template [[host_name(NAME)]] kernel void ink_in_proj_f8_nax_t<BM, BN, NSG>( \
      device bfloat*, device bfloat*, device const float*, device bfloat*, device bfloat*, device bfloat*, \
      constant InkNaxF8P&, uint2);

// Keep in step with inkmix.rs F8_NAX_SHADERS (one per gemm_nax::NAX_VARIANTS tile, same order).
INK_F8_NAX_T("ink_in_proj_f8_nax_t64x64w4", 64, 64, 4)
INK_F8_NAX_T("ink_in_proj_f8_nax_t128x64w4", 128, 64, 4)
INK_F8_NAX_T("ink_in_proj_f8_nax_t32x32w1", 32, 32, 1)
INK_F8_NAX_T("ink_in_proj_f8_nax_t32x128w2", 32, 128, 2)
INK_F8_NAX_T("ink_in_proj_f8_nax_t64x64w2", 64, 64, 2)
