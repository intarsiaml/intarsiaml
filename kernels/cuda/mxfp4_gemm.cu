// matmul_mxfp4_mma: prefill-shaped W4A16 GEMM for OCP MXFP4 (AMD Quark): out[M,N] = x[M,K] @ W^T on tensor cores with
//   w     u8  [N, K/2]   two e2m1 codes per byte, element 2i in the low nibble
//   scale u8  [N, K/32]  e8m0 (2^(b - 127)), plain row-major (checkpoint layout)
// The weight tile is dequantised to bf16 in shared memory (exact: e2m1 * a power of two), the activations stay bf16 (W4A16,
// no activation quantisation), and bf16 mma.sync accumulates in f32 over k in a fixed order. It is the wq_gemm.cuh pipeline
// (FMT_MXFP4; the cp.async weight copy, the 3-slot activation ring and the full-row stores of the NVFP4 / FP8 prefill GEMMs).
// Row invariance: an output element is the same k-ordered chain of m16n8k16 mma steps whatever M, tile shape (BM = 16, 32 or
// 64 rows) or position in the call, so a row's bits do not depend on the rows around it. Not bitwise `matmul_mxfp4_gemv`
// (that one sums each 32-group in f32 FMA order, then scales), which is why the registry routes only ordinary calls here
// (`quantmoe.rs`): verify and batched steps (`exact_rows`) keep the GEMV's bits through `matmul_mxfp4_gemv_rows`.
// The `act` input is accepted for the matmul_mxfp4 signature and not read: the caller routes W4A16 calls only (W4A4 / W4A8
// quantise x, which this kernel does not). K % 128 == 0 (a raw unit is 64 bytes = 128 codes = 4 scale bytes).
#include "wq_gemm.cuh"

namespace {

template <typename OutT>
int run(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, void* stream) {
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    cudaStream_t s = IK_STREAM(stream);
    const int vec = (sizeof(OutT) == 2 && N % 8 == 0 && ik_aligned16(out->data)) ? 1 : 0;
    wq::gemm_by_m<wq::FMT_MXFP4, OutT>((const ik_bf16*)x->data, nullptr, (const uint8_t*)w->data, (const uint8_t*)sc->data,
                                       nullptr, (OutT*)out->data, (int)M, (int)N, (int)K, vec, s);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_matmul_mxfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act, ik_tensor* out,
                        void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_U8, 2, 2));
    IK_TRY(ik_check_a(act, IK_DTYPE_F32, 1, 4));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % 128 != 0 || w->shape[1] != K / 2 || scales->shape[0] != N || scales->shape[1] != K / 32 || act->shape[0] != 1 ||
        out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M == 0 || N == 0 || K == 0) return IK_OK;  // K % 128 == 0 and K == 0: nothing to accumulate (callers never pass it)
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (out->dtype == IK_DTYPE_F32) return run<float>(x, w, scales, out, stream);
    return run<ik_bf16>(x, w, scales, out, stream);
}

}  // extern "C"
