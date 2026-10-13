/*
 * IntarsiaML kernel ABI (torch-free). ARCHITECTURE.md section 3.3.
 *
 * Every kernel pack (native, tensorfold, radiance, vendor wrappers) exposes
 * launchers with this C ABI. Kernels are compiled by nvcc/hipcc into a static
 * library and bound from Rust with `extern "C"`.
 *
 * Rules for every launcher:
 *   - raw tensors in, status out, explicit stream, NO allocation inside;
 *   - launchers are asynchronous on `stream` (a cudaStream_t / hipStream_t
 *     passed as void*); they never synchronise the device;
 *   - tensors are described by ik_tensor; a launcher validates dtype, rank and
 *     strides and returns IK_ERR_* instead of crashing;
 *   - name: ik_<op>_<variant>, e.g. ik_rmsnorm_bf16, ik_matmul_bf16_gemv.
 *     The Rust KernelImpl::name is the launcher name without the `ik_` prefix
 *     (`<op>_<variant>`, e.g. "matmul_bf16_gemv"), and the op part matches the
 *     Op id (see crates/intarsia-core/src/ops.rs, Op::as_str).
 *
 * The numeric values of IK_DTYPE_* must equal intarsia_core::DType (repr i32);
 * a Rust unit test parses this header and checks them.
 */
#ifndef IK_ABI_H
#define IK_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IK_ABI_VERSION 1
#define IK_MAX_DIMS 6

/* ---- dtypes (== intarsia_core::DType) ---------------------------------- */
enum ik_dtype {
    IK_DTYPE_F32 = 0,
    IK_DTYPE_F16 = 1,
    IK_DTYPE_BF16 = 2,
    IK_DTYPE_FP8_E4M3 = 3,
    IK_DTYPE_FP8_E5M2 = 4,
    IK_DTYPE_I32 = 5,
    IK_DTYPE_I8 = 6,
    IK_DTYPE_U8 = 7
};

/* ---- status codes ------------------------------------------------------- */
enum ik_status {
    IK_OK = 0,
    IK_ERR_INVALID_ARG = 1,   /* null pointer, bad rank, bad attribute */
    IK_ERR_DTYPE = 2,         /* dtype combination not supported by this launcher */
    IK_ERR_SHAPE = 3,         /* shape precondition violated (the Rust predicate should have caught it) */
    IK_ERR_STRIDE = 4,        /* non-contiguous layout not supported */
    IK_ERR_UNSUPPORTED = 5,   /* arch / feature not compiled in */
    IK_ERR_LAUNCH = 6         /* the runtime reported a launch error (cudaGetLastError != success) */
};

/* ---- tensors ------------------------------------------------------------ */
/* Strides are in ELEMENTS. In Phase 0 every tensor is dense row-major
 * (stride[i] = prod(shape[i+1..])); launchers may reject anything else with
 * IK_ERR_STRIDE. `data` already includes any byte offset of the view. */
typedef struct {
    void*   data;
    int32_t dtype;            /* enum ik_dtype */
    int32_t ndim;
    int64_t shape[IK_MAX_DIMS];
    int64_t stride[IK_MAX_DIMS];
} ik_tensor;

/* ---- quantisation metadata ---------------------------------------------- */
enum ik_quant_format {
    IK_QUANT_NONE = 0,
    IK_QUANT_FP8_BLOCK = 1,   /* reserved */
    IK_QUANT_NVFP4 = 2,       /* reserved */
    IK_QUANT_MXFP4 = 3,       /* reserved */
    IK_QUANT_GGUF_Q4K = 4,    /* reserved */
    IK_QUANT_EXL3 = 5         /* reserved */
};

typedef struct {
    const void* scales;
    const void* zeros;
    int32_t     group;        /* elements per scale group */
    int32_t     format;       /* enum ik_quant_format */
} ik_quant;

/* ---- launchers (Phase 0: dense BF16 weights, no ik_quant) --------------- *
 * All tensors are dense row-major. `stream` is cudaStream_t/hipStream_t.
 * In/out shapes are given per launcher; attributes are plain scalars.
 *
 * ik_embedding_*     ids[T] i32, table[V,H]           -> out[T,H]
 * ik_rmsnorm_*       x[R,H], w[H], eps                -> out[R,H]
 * ik_rmsnorm_add_*   x[R,H], res[R,H], w[H], eps      -> out[R,H] = rmsnorm(x+res)*w, sum[R,H] = x+res
 * ik_add_*           a[R,H], b[R,H]                   -> out[R,H]
 * ik_qk_norm_*       x[T,N,D], w[D], eps              -> out[T,N,D]   (rmsnorm over D per head)
 * ik_rope_*          x[T,N,D], pos0, theta            -> out[T,N,D]   (neox/rotate-half, positions pos0..pos0+T-1)
 * ik_matmul_*        x[M,K], w[N,K]                   -> out[M,N]     (out = x @ w^T, f32 accumulate)
 * ik_kv_append_*     new[T,Nkv,D], cache[Nkv,cap,D], pos0          (writes rows pos0..pos0+T-1 in place)
 * ik_attn_prefill_*  q[T,N,D], kc[Nkv,cap,D], vc[Nkv,cap,D], pos0, scale -> out[T,N,D]  (causal, GQA)
 * ik_attn_decode_*   same with T == 1                                    (kv_len = pos0 + 1)
 * ik_silu_glu_*      gate[T,I], up[T,I]               -> out[T,I] = silu(gate)*up
 * ik_argmax_*        logits[R,V]                      -> out[R] i32
 */

/* Phase 0 launcher set (BF16 activations/weights; the cuda pack implements all of them).
 * Every launcher returns an enum ik_status. Zero-element tensors are a no-op (IK_OK). */
int ik_embedding_bf16(const ik_tensor* ids, const ik_tensor* table, ik_tensor* out, void* stream);
int ik_rmsnorm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, ik_tensor* out, void* stream);
int ik_rmsnorm_add_bf16(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps,
                        ik_tensor* out, ik_tensor* sum_out, void* stream);
int ik_add_bf16(const ik_tensor* a, const ik_tensor* b, ik_tensor* out, void* stream);
int ik_qk_norm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, ik_tensor* out, void* stream);
int ik_rope_bf16(const ik_tensor* x, int64_t pos0, float theta, ik_tensor* out, void* stream);
/* matmul: cuBLAS (any M, "fast") and a native GEMV (M <= 4, deterministic, run-to-run identical). */
int ik_matmul_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
int ik_matmul_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
/* logits: the same math with an f32 output tensor */
int ik_logits_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
int ik_logits_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
/* kv_append: `cache` dtype may be BF16 or FP8_E4M3 (saturating round-to-nearest-even) */
int ik_kv_append_bf16(const ik_tensor* new_kv, int64_t pos0, ik_tensor* cache, void* stream);
int ik_attn_prefill_bf16(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                         float scale, ik_tensor* out, void* stream);
int ik_attn_decode_bf16(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                        float scale, ik_tensor* out, void* stream);
int ik_silu_glu_bf16(const ik_tensor* gate, const ik_tensor* up, ik_tensor* out, void* stream);
/* argmax: logits dtype F32 or BF16; lowest index wins ties; out is I32 [R] */
int ik_argmax_f32(const ik_tensor* logits, ik_tensor* out, void* stream);
int ik_argmax_bf16(const ik_tensor* logits, ik_tensor* out, void* stream);

/* Optional pack introspection (each pack exports one; used by `intarsia doctor`). */
typedef struct {
    int32_t     abi_version;  /* IK_ABI_VERSION the pack was built against */
    const char* pack;         /* "native", "tensorfold", ... */
    const char* version;      /* pack version / git hash */
    const char* arch;         /* "sm_121" */
} ik_pack_info;

#ifdef __cplusplus
}
#endif
#endif /* IK_ABI_H */
