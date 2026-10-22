/* CUDA-only launchers of the Gemma 4 extended attributes (OpAttrs::window, rope_prop, scale on Add / Embedding, GeluGlu):
 * declared here beside kernels/cuda/ik_cuda_abi.h. Separate launchers, so the plain ones keep their exact bits. The CUDA
 * backend's `every_launcher_is_declared_in_the_abi_header` test reads this file too. */
#ifndef IK_CUDA_ABI_G4_H
#define IK_CUDA_ABI_G4_H

#include "abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* gelu_glu_bf16: out = gelu_tanh(gate) * up (gelu_pytorch_tanh), f32 inside, one bf16 rounding (kernels/cuda/elementwise.cu) */
int ik_gelu_glu_bf16(const ik_tensor* gate, const ik_tensor* up, ik_tensor* out, void* stream);

/* rope_prop_bf16: proportional rope (Gemma 4 global layers). x[T,N,D]: pair (j, j + D/2) for j < rotary_dim/2 is rotated by
 * pos * theta^(-2j/D) (denominator the full head dim D), every other value is copied; rotary_dim 0 = D. x and out may alias. */
int ik_rope_prop_bf16(const ik_tensor* x, int64_t pos0, float theta, int64_t rotary_dim, ik_tensor* out,
                      const ik_tensor* pos, void* stream);

/* add_scaled_bf16 / embedding_scaled_bf16: out = round_bf16(f32(a + b) * scale), round_bf16(f32(table row) * scale) */
int ik_add_scaled_bf16(const ik_tensor* a, const ik_tensor* b, ik_tensor* out, float scale, void* stream);
int ik_embedding_scaled_bf16(const ik_tensor* ids, const ik_tensor* table, ik_tensor* out, float scale, void* stream);

/* attn_{prefill,decode}_bf16_ext: attention.cu's SIMT kernels with head_dim up to 512 and a sliding window (the query at
 * absolute position p attends keys max(0, p + 1 - window)..=p, window 0 = all), kernels/cuda/attention_ext.cu */
int ik_attn_prefill_bf16_ext(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                             int64_t window, ik_tensor* out, void* stream);
int ik_attn_decode_bf16_ext(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                            int64_t window, ik_tensor* out, const ik_tensor* pos, void* stream);

/* attn_{decode,prefill}_bf16_splitkv_ext: the split-KV kernels (head_dim 32/64/128/256, group <= 8, scratch as the plain ones)
 * with the sliding window; kernels/cuda/attention_splitkv.cu */
int ik_attn_decode_bf16_splitkv_ext(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                    float scale, int64_t window, ik_tensor* out, const ik_scratch* scratch,
                                    const ik_tensor* pos, void* stream);
int ik_attn_prefill_bf16_splitkv_ext(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                     float scale, int64_t window, ik_tensor* out, const ik_scratch* scratch,
                                     void* stream);

#ifdef __cplusplus
}
#endif

#endif /* IK_CUDA_ABI_G4_H */
