// Router logits of Inkling-2-Mini prefill chunks on the M5 neural accelerators (lane ink-prefill; launcher
// src/inkprefill.rs `launch_logits_nax`): `Op::Logits` C[M,N] f32 = A[M,K] bf16 . W[N,K]^T bf16, for M > 16. gemm_nax_bf16.metal's
// tile with an f32 destination tensor (no rounding at the store). The K reduction is the accelerator's, so the result is not
// bitwise `logits_bf16`: admitted against the f64 reference at the Logits f32 tolerance.
// bind: 0 A [M,K], 1 W [N,K], 2 C [M,N] f32, 3 NaxP (gemm_nax_bf16.metal's layout; s0 / s1 bf16 skips, s2 f32 skip).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct NaxP {
  uint m, n, k, swz_log;
  uint s0, s1, s2, pad;
};

template <int BM, int BN, int NSG>
kernel void ink_logits_nax(device bfloat* A [[buffer(0)]], device bfloat* W [[buffer(1)]], device float* C [[buffer(2)]],
                           constant NaxP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int M = int(p.m), N = int(p.n), K = int(p.k);
  const int m0 = int(tg.y) * BM, n0 = int(tg.x) * BN;
  if (m0 >= M || n0 >= N) return;
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A + p.s0, dextents<int, 2>(K, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W + p.s1, dextents<int, 2>(K, N));
  tensor<device float, dextents<int, 2>, tensor_inline> tC(C + p.s2, dextents<int, 2>(N, M));
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

// Keep in step with ROUTER_TILES in src/inkprefill.rs.
template [[host_name("ink_logits_nax_t32x32w1")]] kernel void ink_logits_nax<32, 32, 1>(device bfloat*, device bfloat*, device float*, constant NaxP&, uint2);
template [[host_name("ink_logits_nax_t64x64w4")]] kernel void ink_logits_nax<64, 64, 4>(device bfloat*, device bfloat*, device float*, constant NaxP&, uint2);
