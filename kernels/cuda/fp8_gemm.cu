// matmul_fp8_gemm: same op and numerics as matmul_fp8_mma (fp8.cu), with the memory schedule of wq_gemm.cuh.
// W8A8 (input_scale > 0): the activations are first quantised to e4m3-valued bf16 into scratch (one pass over x), the
// GEMM then reads those. Needs K % 64 == 0.
#include "wq_gemm.cuh"

namespace {

__global__ void __launch_bounds__(256)
fp8_actq_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ xq, const float* __restrict__ sc, int64_t n8) {
    const float s_in = sc[1];
    if (!(s_in > 0.f)) return;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    float f[8];
    ik_unpack8(reinterpret_cast<const uint4*>(x)[i], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = ik_e4m3_to_f(ik_f_to_e4m3(__fdiv_rn(f[e], s_in)));
    reinterpret_cast<uint4*>(xq)[i] = ik_pack8(f);
}

template <typename OutT>
int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const ik_bf16* xq, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    cudaStream_t s = IK_STREAM(stream);
    const int vec = (sizeof(OutT) == 2 && N % 8 == 0 && ik_aligned16(out->data)) ? 1 : 0;
    wq::gemm_by_m<wq::FMT_FP8, OutT>((const ik_bf16*)x->data, xq, (const uint8_t*)w->data, nullptr, (const float*)sc->data,
                                     (OutT*)out->data, (int)M, (int)N, (int)K, vec, s);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_fp8_gemm_weight_only_internal(const void* x, const void* w, const void* scales, void* out, int out_is_f32, int64_t M,
                                     int64_t N, int64_t K, void* stream) {
    cudaStream_t s = IK_STREAM(stream);
    const uint8_t* wp = (const uint8_t*)w;
    const float* sc = (const float*)scales;
    if (out_is_f32) {
        wq::gemm_by_m<wq::FMT_FP8, float>((const ik_bf16*)x, nullptr, wp, nullptr, sc, (float*)out, (int)M, (int)N, (int)K, 0, s, 1);
    } else {
        const int vec = (N % 8 == 0 && ik_aligned16(out)) ? 1 : 0;
        wq::gemm_by_m<wq::FMT_FP8, ik_bf16>((const ik_bf16*)x, nullptr, wp, nullptr, sc, (ik_bf16*)out, (int)M, (int)N, (int)K, vec, s, 1);
    }
    return ik_launch_status();
}

int ik_matmul_fp8_gemm(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                       const ik_scratch* scratch, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_FP8_E4M3, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 128 != 0 || w->shape[1] != K || scales->shape[0] != 2 || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    // scratch: the quantised activations [M, K] bf16 (always reserved; used when input_scale > 0, known only on device)
    const int64_t xq_bytes = M * K * 2;
    if (!scratch || !scratch->data || scratch->bytes < xq_bytes) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    ik_bf16* xq = (ik_bf16*)scratch->data;
    cudaStream_t s = IK_STREAM(stream);
    const int64_t n8 = M * K / 8;
    fp8_actq_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((const ik_bf16*)x->data, xq, (const float*)scales->data, n8);
    IK_TRY(ik_launch_status());
    return out->dtype == IK_DTYPE_F32 ? run<float>(x, w, scales, out, xq, stream) : run<ik_bf16>(x, w, scales, out, xq, stream);
}

}  // extern "C"
