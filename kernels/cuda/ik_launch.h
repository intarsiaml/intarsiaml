/* Host-side helpers shared by the launchers: tensor validation, status codes. */
#ifndef IK_LAUNCH_H
#define IK_LAUNCH_H

#include "../abi/ik_abi.h"
#include "../common/ik_portable.h"

static inline int64_t ik_numel(const ik_tensor* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; ++i) n *= t->shape[i];
    return n;
}

static inline bool ik_dense(const ik_tensor* t) {
    int64_t expect = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        if (t->shape[i] > 1 && t->stride[i] != expect) return false;
        expect *= t->shape[i];
    }
    return true;
}

/* null / dtype / rank / dense-row-major check; IK_OK or an IK_ERR_* */
static inline int ik_check(const ik_tensor* t, int dtype, int ndim) {
    if (!t || (!t->data && ik_numel(t) > 0)) return IK_ERR_INVALID_ARG;
    if (t->ndim < 0 || t->ndim > IK_MAX_DIMS) return IK_ERR_INVALID_ARG;
    if (t->dtype != dtype) return IK_ERR_DTYPE;
    if (t->ndim != ndim) return IK_ERR_SHAPE;
    if (!ik_dense(t)) return IK_ERR_STRIDE;
    return IK_OK;
}

/* optional device-side position: null, or an i32 tensor with one element (the launcher's host `pos0` is then only a
 * bound for validation and grid sizing; the kernel reads the position from the tensor when it runs) */
static inline int ik_check_pos(const ik_tensor* t) {
    if (!t) return IK_OK;
    if (!t->data || t->dtype != IK_DTYPE_I32 || ik_numel(t) != 1) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

#define IK_TRY(expr)                 \
    do {                             \
        int _ik_s = (expr);          \
        if (_ik_s != IK_OK) return _ik_s; \
    } while (0)

/* Map a failed kernel launch to IK_ERR_LAUNCH (does not synchronise). */
static inline int ik_launch_status() { return cudaGetLastError() == cudaSuccess ? IK_OK : IK_ERR_LAUNCH; }

static inline bool ik_aligned16(const void* p) { return ((uintptr_t)p & 15u) == 0; }

static inline unsigned ik_round_up(unsigned v, unsigned m) { return (v + m - 1) / m * m; }

#define IK_STREAM(s) (reinterpret_cast<cudaStream_t>(s))

/* silu(g) = g * sigmoid(g). The one expression silu_glu uses (elementwise.cu): the fused kernels that fold a silu_glu
 * into their epilogue (nvfp4.cu glu, norm.cu gated rmsnorm) must produce its bits. */
IK_DEVI float ik_silu(float g) { return g / (1.f + expf(-g)); }

#endif /* IK_LAUNCH_H */
