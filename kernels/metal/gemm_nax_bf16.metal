#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

[[host_name("gemm_nax_probe")]] kernel void nax_trivial(device bfloat* A [[buffer(0)]], device bfloat* W [[buffer(1)]], device bfloat* C [[buffer(2)]],
                        constant uint3& mnk [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = mnk.x, N = mnk.y, K = mnk.z;
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A, dextents<int, 2>(K, M));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tW(W, dextents<int, 2>(K, N));
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tC(C, dextents<int, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, tgid.y * 64);
  auto mW = tW.slice(0, tgid.x * 32);
  auto mC = tC.slice(tgid.x * 32, tgid.y * 64);
  op.run(mA, mW, mC);
}
