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
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct NaxFp8P {
  uint m, n, k, swz_log;
  uint s_c, pad0, pad1, pad2;
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

template <typename OutT, int BM, int BN, int NSG>
kernel void gemm_nax_fp8_t(device bfloat* A [[buffer(0)]], device bfloat* W [[buffer(1)]],
                           device const float* sc [[buffer(2)]], device OutT* C [[buffer(3)]],
                           constant NaxFp8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), N = int(p.n), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int ty = int((tg.y << p.swz_log) + (tg.x & ((1u << p.swz_log) - 1)));
  const int tx = int(tg.x >> p.swz_log);
  if (tx >= tiles_n || ty >= tiles_m) return;
  const int m0 = ty * BM, n0 = tx * BN;
  const float is = sc[1];
  const float scale = sc[0] * (is > 0.0f ? is : 1.0f);

  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A, dextents<int, 2>(K, M));
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

#define NAX_FP8_T(NAME, OUT, BM, BN, NSG) \
  template [[host_name(NAME)]] kernel void gemm_nax_fp8_t<OUT, BM, BN, NSG>( \
      device bfloat*, device bfloat*, device const float*, device OUT*, constant NaxFp8P&, uint2);

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
