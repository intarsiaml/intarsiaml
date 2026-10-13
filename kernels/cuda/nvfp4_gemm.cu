// matmul_nvfp4_gemm: same op and numerics as matmul_nvfp4_mma (nvfp4.cu), prefill-shaped batches, with the memory
// schedule of wq_gemm.cuh (64-byte-per-row cp.async runs, staged and dequantised once per tile and k step, 3-slot A
// ring, full-row output stores). Needs K % 128 == 0.
#include "wq_gemm.cuh"

namespace {

template <typename OutT>
int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs, ik_tensor* out, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    cudaStream_t s = IK_STREAM(stream);
    const int vec = (sizeof(OutT) == 2 && N % 8 == 0 && ik_aligned16(out->data)) ? 1 : 0;
    wq::gemm_by_m<wq::FMT_NVFP4, OutT>((const ik_bf16*)x->data, nullptr, (const uint8_t*)w->data, (const uint8_t*)bs->data,
                                       (const float*)gs->data, (OutT*)out->data, (int)M, (int)N, (int)K, vec, s);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_nvfp4_gemm(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                         const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(block_scales, IK_DTYPE_FP8_E4M3, 2, 4));
    IK_TRY(ik_check_a(global_scale, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 128 != 0 || w->shape[1] != K / 2 || block_scales->shape[0] != N || block_scales->shape[1] != K / 16 ||
        global_scale->shape[0] != 1 || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    return out->dtype == IK_DTYPE_F32 ? run<float>(x, w, block_scales, global_scale, out, stream)
                                      : run<ik_bf16>(x, w, block_scales, global_scale, out, stream);
}

}  // extern "C"
