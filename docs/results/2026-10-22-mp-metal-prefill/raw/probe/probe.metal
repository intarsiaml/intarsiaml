#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// out[(which * 32 + lane) * 64 + i] = (valid << 16) | (row << 8) | col   for element i of the coop tensor
template <typename CT>
inline void dump(thread CT& ct, device uint* out, uint which, uint lane) {
    for (ushort i = 0; i < 64; ++i) {
        uint v = 0xffffffff;
        if (i < ct.get_capacity()) {
            auto ix = ct.get_multidimensional_index(i);
            v = ((uint)ix[1] << 8) | (uint)ix[0];
        }
        out[(which * 32 + lane) * 64 + i] = v;
    }
}
kernel void probe(device uint* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
    {
        constexpr auto d = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<d, execution_simdgroup> op;
        auto a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        auto b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
        auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
        dump(a, out, 0, lane); dump(b, out, 1, lane); dump(c, out, 2, lane);
    }
    {
        constexpr auto d = matmul2d_descriptor(16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<d, execution_simdgroup> op;
        auto a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        auto b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
        auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
        dump(a, out, 3, lane); dump(c, out, 5, lane);
    }
    {
        constexpr auto d = matmul2d_descriptor(16, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<d, execution_simdgroup> op;
        auto a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        auto b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
        auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
        dump(a, out, 6, lane); dump(b, out, 7, lane); dump(c, out, 8, lane);
    }
    {
        constexpr auto d = matmul2d_descriptor(16, 128, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<d, execution_simdgroup> op;
        auto a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        auto b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
        auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
        dump(a, out, 9, lane); dump(c, out, 10, lane);
    }
}
