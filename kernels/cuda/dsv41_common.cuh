/* Shared helpers of the DeepSeek-V4.1 kernels (dsv41_*.cu): typed float loads / stores over bf16 / f16 / f32 and the host
 * dtype dispatch. Written from the op contract (docs/design/dsv41-ops.md) and DeepSeek's official inference code (MIT). */
#ifndef IK_DSV41_COMMON_CUH
#define IK_DSV41_COMMON_CUH

#include "ik_launch.h"
#include "ik_cuda_abi_dsv41.h"
#include <cuda_fp16.h>

namespace dsv41 {

template <typename T> IK_DEVI float ld(const T* p, int64_t i);
template <> IK_DEVI float ld<float>(const float* p, int64_t i) { return p[i]; }
template <> IK_DEVI float ld<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
template <> IK_DEVI float ld<__half>(const __half* p, int64_t i) { return __half2float(p[i]); }

template <typename T> IK_DEVI void st(T* p, int64_t i, float v);
template <> IK_DEVI void st<float>(float* p, int64_t i, float v) { p[i] = v; }
template <> IK_DEVI void st<__nv_bfloat16>(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16_rn(v); }
template <> IK_DEVI void st<__half>(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }

static inline bool is_float_dt(int dt) { return dt == IK_DTYPE_F32 || dt == IK_DTYPE_BF16 || dt == IK_DTYPE_F16; }

/* dense float tensor of rank `ndim` (any of bf16 / f16 / f32) */
static inline int check_float(const ik_tensor* t, int ndim) {
    if (!t) return IK_ERR_INVALID_ARG;
    if (!is_float_dt(t->dtype)) return IK_ERR_DTYPE;
    return ik_check(t, t->dtype, ndim);
}

static inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline unsigned grid1d(int64_t n, int threads) {
    int64_t b = cdiv(n, threads);
    if (b < 1) b = 1;
    if (b > 65535LL * 32) b = 65535LL * 32;
    return (unsigned)b;
}

}  // namespace dsv41

/* Call `F<T>(args...)` with T the C++ type of float dtype `dt`; returns IK_ERR_DTYPE for anything else. */
#define DSV_FLOAT_DISPATCH(dt, T, ...)                 \
    switch (dt) {                                      \
        case IK_DTYPE_F32: {                           \
            typedef float T;                           \
            __VA_ARGS__;                               \
            break;                                     \
        }                                              \
        case IK_DTYPE_BF16: {                          \
            typedef __nv_bfloat16 T;                   \
            __VA_ARGS__;                               \
            break;                                     \
        }                                              \
        case IK_DTYPE_F16: {                           \
            typedef __half T;                          \
            __VA_ARGS__;                               \
            break;                                     \
        }                                              \
        default:                                       \
            return IK_ERR_DTYPE;                       \
    }

#endif /* IK_DSV41_COMMON_CUH */
