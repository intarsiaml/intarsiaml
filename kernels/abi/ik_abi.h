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

/* ---- scratch workspace ----------------------------------------------------
 * "NO allocation inside" holds for kernels that need temporary global memory
 * too (split-KV attention partials, split-K reductions, MoE routing tables):
 * the caller allocates it once at plan time and hands it to every launch.
 *
 *   - the requirement is declared on the Rust side, per shape: KernelImpl::scratch
 *     is a function ShapeInfo -> bytes (crates/intarsia-core/src/registry.rs). The
 *     plan takes the maximum over every shape of the model and reserves one buffer
 *     for all ops (ops run in stream order, so they may share it);
 *   - a launcher that uses scratch takes `const ik_scratch* scratch` as the
 *     parameter immediately before `stream` (nullptr is legal when it declared
 *     none); `scratch->data` is 256-byte aligned, uninitialised, valid only for
 *     this launch's stream order; `scratch->bytes` >= the declared size;
 *   - a launcher returns IK_ERR_INVALID_ARG when scratch is null or too small
 *     for the shape it was given; it never allocates to make up for it;
 *   - contents do not survive the launch: write before read, in every launch.
 */
typedef struct {
    void*   data;
    int64_t bytes;
} ik_scratch;

/* ---- device-side position (CUDA graphs) -----------------------------------
 * ik_rope_*, ik_kv_append_* and ik_attn_decode_* take an optional `const ik_tensor* pos` (immediately before `stream`;
 * nullptr = use the host `pos0`). When given it is an I32 tensor with one element holding the position, read by the
 * kernel when it RUNS, so a captured graph can be replayed at every position by writing that element between
 * replays. The host `pos0` is then only a bound used for validation. The result is bit-identical to launching with
 * the same value as host `pos0`. The caller guarantees pos + T <= cap (a launcher cannot check a device value).
 */

/* ---- launchers (Phase 0: dense BF16 weights, no ik_quant) --------------- *
 * All tensors are dense row-major. `stream` is cudaStream_t/hipStream_t.
 * In/out shapes are given per launcher; attributes are plain scalars.
 *
 * ik_embedding_*     ids[T] i32, table[V,H]           -> out[T,H]
 * ik_rmsnorm_*       x[R,H], w[H], eps, w_offset      -> out[R,H]     (scale = w + w_offset; 1.0 for zero-centred norms)
 * ik_rmsnorm_add_*   x[R,H], res[R,H], w[H], eps      -> out[R,H] = rmsnorm(x+res)*w, sum[R,H] = x+res
 * ik_add_*           a[R,H], b[R,H]                   -> out[R,H]
 * ik_qk_norm_*       x[T,N,D], w[D], eps              -> out[T,N,D]   (rmsnorm over D per head)
 * ik_rope_*          x[T,N,D], pos0, theta, rotary_dim -> out[T,N,D]  (neox/rotate-half over the first rotary_dim values, 0 = all; positions pos0..pos0+T-1)
 * ik_gate_split_*    x[T,N,2D]                        -> q[T,N,D], gate[T,N,D]  (per head: D query values, then D gate values)
 * ik_sigmoid_mul_*   a[R,X], g[R,X]                   -> out[R,X] = a * sigmoid(g)
 * ik_gdn_conv_*      x[T,C], taps[C,K], state[C,K-1] f32 (in/out)  -> y[T,C] = silu(causal depthwise conv)
 * ik_gdn_delta_rule_* qkv[T,2*Hk*dk+Hv*dv], a[T,Hv], b[T,Hv], A_log[Hv], dt_bias[Hv], eps, state[Hv,dv,dk] f32 (in/out)
 *                                                    -> o[T,Hv,dv]   (gated delta rule, crates/intarsia-core/src/gdn.rs)
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
int ik_rmsnorm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, float w_offset, ik_tensor* out, void* stream);
int ik_rmsnorm_add_bf16(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, float w_offset,
                        ik_tensor* out, ik_tensor* sum_out, void* stream);
int ik_add_bf16(const ik_tensor* a, const ik_tensor* b, ik_tensor* out, void* stream);
int ik_qk_norm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, float w_offset, ik_tensor* out, void* stream);
int ik_rope_bf16(const ik_tensor* x, int64_t pos0, float theta, int64_t rotary_dim, ik_tensor* out,
                 const ik_tensor* pos, void* stream);
/* gated attention / GatedDeltaNet helpers (bf16 activations; conv taps, A_log and dt_bias bf16; recurrent state f32) */
int ik_gate_split_bf16(const ik_tensor* x, ik_tensor* q, ik_tensor* gate, void* stream);
int ik_sigmoid_mul_bf16(const ik_tensor* a, const ik_tensor* g, ik_tensor* out, void* stream);
int ik_gdn_conv_bf16(const ik_tensor* x, const ik_tensor* taps, ik_tensor* y, ik_tensor* state, void* stream);
/* head_dim (state[2]) in {32, 64, 128, 256}; deterministic */
int ik_gdn_delta_rule_bf16(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                           const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream);
/* same contract, head_dim in {64, 128}: several lanes share a state row (prefill: the one-thread-per-row kernel is
 * latency bound). Deterministic; sums in a different (fixed) order than ik_gdn_delta_rule_bf16. */
int ik_gdn_delta_rule_bf16_lanes(const ik_tensor* qkv, const ik_tensor* a, const ik_tensor* b, const ik_tensor* a_log,
                                 const ik_tensor* dt_bias, float eps, ik_tensor* o, ik_tensor* state, void* stream);
/* matmul: cuBLAS (any M, "fast") and a native GEMV (M <= 4, deterministic, run-to-run identical). */
int ik_matmul_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
int ik_matmul_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
/* logits: the same math with an f32 output tensor */
int ik_logits_bf16(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
int ik_logits_bf16_gemv(const ik_tensor* x, const ik_tensor* w, ik_tensor* out, void* stream);
/* kv_append: `cache` dtype may be BF16 or FP8_E4M3 (saturating round-to-nearest-even) */
int ik_kv_append_bf16(const ik_tensor* new_kv, int64_t pos0, ik_tensor* cache, const ik_tensor* pos, void* stream);
int ik_attn_prefill_bf16(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                         float scale, ik_tensor* out, void* stream);
int ik_attn_decode_bf16(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                        float scale, ik_tensor* out, const ik_tensor* pos, void* stream);
/* split-KV (flash-decoding) attention for few query rows: T == 1 (decode) or T <= 16 (verify); uses scratch,
 * head_dim in {32, 64, 128, 256}, q heads per kv head <= 8. The scratch size is a function of the shape
 * (cuda backend: splitkv_parts in crates/intarsia-backend-cuda/src/kernels.rs). */
int ik_attn_decode_bf16_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos,
                                void* stream);
int ik_attn_prefill_bf16_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                 float scale, ik_tensor* out, const ik_scratch* scratch, void* stream);
/* flash-style tensor-core prefill (causal, GQA), head_dim in {64, 128, 256}, no scratch */
int ik_attn_prefill_bf16_flash(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                               float scale, ik_tensor* out, void* stream);
int ik_silu_glu_bf16(const ik_tensor* gate, const ik_tensor* up, ik_tensor* out, void* stream);
/* argmax: logits dtype F32 or BF16; lowest index wins ties; out is I32 [R] */
int ik_argmax_f32(const ik_tensor* logits, ik_tensor* out, void* stream);
int ik_argmax_bf16(const ik_tensor* logits, ik_tensor* out, void* stream);

/* Quantised weights and MoE (docs/quant-moe.md). Activations BF16; outputs BF16 or F32 where stated.
 *   matmul_fp8_*    x[M,K] bf16, w[N,K] e4m3, scales f32[2] = (weight_scale, input_scale) -> out[M,N]
 *                   input_scale > 0: W8A8 (x quantised to e4m3 with that static scale), <= 0: W8A16.
 *                   _gemv: M <= 4 (x staged in shared memory); _mma: any M (bf16 tensor cores). K % 16 == 0.
 *   matmul_nvfp4_*  x[M,K] bf16, w u8[N,K/2] (2 x e2m1, low nibble first), block_scales e4m3[N,K/16], global f32[1]
 *                   -> out[M,N]; _gemv: M <= 8, _mma: any M. K % 16 == 0 (W4A16).
 *   moe_topk_bf16   logits[T,E] bf16 -> weights f32[T,K] (bf16-rounded softmax weights, renormalised over the selected
 *                   experts when norm != 0), ids i32[T,K]; ties: lowest expert index first; E <= 512, K <= 32.
 *   moe_experts_bf16  x[T,H], ids i32[T,K], weights f32[T,K], gate_up_w u8[E,2I,H/2], gate_up_s e4m3[E,2I,H/16],
 *                   down_w u8[E,H,I/2], down_s e4m3[E,H,I/16], gscales f32[E,3] (gate, up, down weight_scale_2)
 *                   -> out[T,H] = sum_k w * down(silu(gate(x)) * up(x)); uses scratch (intarsia_core::quant::ops::
 *                   moe_experts_scratch). H, I % 16 == 0, E <= 1024.
 *   matmul_fp8_gemm, matmul_nvfp4_gemm  same ops for prefill-shaped M: weights copied with cp.async in 64-byte row runs,
 *                   dequantised once per tile (kernels/cuda/wq_gemm.cuh). K % 128 == 0; fp8 uses scratch (M*K*2 bytes,
 *                   the W8A8-quantised activations).
 *   moe_experts_mma_bf16  moe_experts_bf16 for batches: pairs grouped by expert, tensor-core tiles, scratch (intarsia_core::
 *                   quant::ops::moe_experts_mma_scratch). H % 128 == 0, I % 128 == 0, E <= 1024.
 *   moe_experts_decode_bf16  moe_experts_bf16 for T = 1: two launches, bit-identical results (scratch: ..._decode_scratch).
 *   moe_combine_bf16  routed[T,H], shared[T,H], gate_logit[T,1] -> out[T,H] = routed + sigmoid(gate) * shared. */
int ik_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream);
int ik_matmul_fp8_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream);
/* HIP decode "v2" twins (docs/results/2026-10-13-hip-decode.md): same arguments and the same results as the kernel they
 * are named after, restructured for latency; defined by the HIP backend only. */
int ik_matmul_fp8_gemv_v2(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream);
int ik_matmul_fp8_gemm(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                       const ik_scratch* scratch, void* stream);
/* matmul_fp8_gemm_f8: same op, W8A8 on e4m3 tensor cores (mma m16n8k32): no weight dequantisation. Same arguments and
 * scratch as matmul_fp8_gemm; weight-only activations (input_scale <= 0) fall back to the bf16 path inside. */
int ik_matmul_fp8_gemm_f8(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, ik_tensor* out,
                          const ik_scratch* scratch, void* stream);
int ik_matmul_nvfp4_gemm(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                         const ik_tensor* global_scale, ik_tensor* out, void* stream);
int ik_matmul_nvfp4_gemv(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                         const ik_tensor* global_scale, ik_tensor* out, void* stream);
int ik_matmul_nvfp4_gemv_v2(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                            const ik_tensor* global_scale, ik_tensor* out, void* stream);
int ik_matmul_nvfp4_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                        const ik_tensor* global_scale, ik_tensor* out, void* stream);
int ik_moe_topk_bf16(const ik_tensor* logits, int top_k, int norm, ik_tensor* weights, ik_tensor* ids, void* stream);
int ik_moe_topk_bf16_v2(const ik_tensor* logits, int top_k, int norm, ik_tensor* weights, ik_tensor* ids, void* stream);
int ik_moe_experts_bf16_v2(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                           const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                           const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                           void* stream);
int ik_moe_experts_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                        const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                        const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                        void* stream);
int ik_moe_experts_mma_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                            const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                            const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out, const ik_scratch* scratch,
                            void* stream);
int ik_moe_experts_decode_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                               const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                               const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                               const ik_scratch* scratch, void* stream);
/* moe_experts_decode_v2_bf16: the same op and the same bits as moe_experts_decode_bf16 (CUDA), with the down projection
 * restructured: a warp per expert slot instead of per output row (docs/results/2026-10-14-cuda-decode-fusion.md). */
int ik_moe_experts_decode_v2_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* weights,
                                  const ik_tensor* gate_up_w, const ik_tensor* gate_up_s, const ik_tensor* down_w,
                                  const ik_tensor* down_s, const ik_tensor* gscales, ik_tensor* out,
                                  const ik_scratch* scratch, void* stream);
int ik_moe_combine_bf16(const ik_tensor* routed, const ik_tensor* shared, const ik_tensor* gate_logit, ik_tensor* out,
                        void* stream);

/* Fused decode launchers (HIP: docs/results/2026-10-14-hip-decode-2.md; CUDA: docs/results/2026-10-14-cuda-decode-fusion.md). Each replaces a short chain of the launchers
 * above and produces the same bits as that chain, so a backend without them keeps the unfused path (the model asks the
 * registry whether a kernel exists before it uses one). M = 1 (one decode row) except gated_rmsnorm.
 *   matmul_pair_bf16_gemv   x[1,K], w0[N0,K], w1[N1,K] (bf16, K % 8 == 0) -> out0[1,N0], out1[1,N1]: two matmul_bf16_gemv.
 *   matmul_fp8_pair_gemv    x[1,K], (w0, scales0), (w1, scales1) fp8 -> out0, out1 (bf16): two matmul_fp8_gemv_v2.
 *   matmul_nvfp4_glu_gemv   x[1,K], (w, block scales, global) for gate and for up, [N,K] each -> out[1,N] (bf16)
 *                           = silu_glu(matmul_nvfp4_gemv_v2(gate), matmul_nvfp4_gemv_v2(up)).
 *   gated_rmsnorm_bf16      x[R,D], z[R,D], w[D], eps, w_offset -> out[R,D] = silu_glu(z, rmsnorm(x, w)); out may be x.
 *   rmsnorm_add_matmul_pair_bf16_gemv
 *                           x[1,K], res[1,K], w[K] (bf16, K % 8 == 0, K <= 8192), eps, w_offset, w0[N0,K], w1[N1,K] ->
 *                           normed[1,K], sum[1,K], out0[1,N0], out1[1,N1] = rmsnorm_add_bf16(x, res, w) then matmul_pair_bf16_gemv
 *                           on `normed`. normed / sum must not overlap x / res (HIP and CUDA).
 *   gated_rmsnorm_matmul_fp8_gemv
 *                           x[R,128], z[R,128], w[128] (bf16), eps, w_offset, fp8 weights [N,R*128] + scales -> out[1,N] (bf16)
 *                           = matmul_fp8_gemv_v2(gated_rmsnorm_bf16(x, z, w) read as one row, ...) (HIP and CUDA). */
int ik_matmul_pair_bf16_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* out0,
                             ik_tensor* out1, void* stream);
int ik_matmul_fp8_pair_gemv(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* scales0, const ik_tensor* w1,
                            const ik_tensor* scales1, ik_tensor* out0, ik_tensor* out1, void* stream);
int ik_matmul_nvfp4_glu_gemv(const ik_tensor* x, const ik_tensor* w_gate, const ik_tensor* block_scales_gate,
                             const ik_tensor* global_scale_gate, const ik_tensor* w_up, const ik_tensor* block_scales_up,
                             const ik_tensor* global_scale_up, ik_tensor* out, void* stream);
int ik_gated_rmsnorm_bf16(const ik_tensor* x, const ik_tensor* z, const ik_tensor* w, float eps, float w_offset,
                          ik_tensor* out, void* stream);
int ik_gated_rmsnorm_matmul_fp8_gemv(const ik_tensor* x, const ik_tensor* z, const ik_tensor* nw, float eps, float w_offset,
                                     const ik_tensor* w, const ik_tensor* scales, ik_tensor* out, void* stream);
int ik_rmsnorm_add_matmul_pair_bf16_gemv(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps,
                                         float w_offset, const ik_tensor* w0, const ik_tensor* w1, ik_tensor* normed,
                                         ik_tensor* sum, ik_tensor* out0, ik_tensor* out1, void* stream);

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
