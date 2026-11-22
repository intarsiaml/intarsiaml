/* Inkling-2-Mini attention launchers (CUDA only), kernels/cuda/inkling2_attn.cu (lane ink-cuda-attn). Semantics: the op
 * contract in crates/intarsia-core/src/inkling2/ops.rs (InkKvPrologue 70, RelSinkAttn 71, InkAttnDecode 75); same inputs /
 * outputs in the same order, the attributes in `ik_ink_attrs`. Every tensor is bf16 and dense row-major. `pos` is the optional
 * device position (i32 [1], null = attrs->pos0); `scratch` is the workspace the Rust registration sizes (inkattn.rs). */
#ifndef IK_CUDA_ABI_INK_H
#define IK_CUDA_ABI_INK_H

#include "abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* mirrored by `IkInkAttrs` in crates/intarsia-backend-cuda/src/inkattn.rs (repr(C)) */
typedef struct {
    int64_t n_heads, n_kv_heads, head_dim, d_rel;
    int64_t pos0, window, tau_floor;
    int32_t dyn_conv, kv_ring;
    float qk_clip, rel_clip, eps, tau_alpha, scale;
    int32_t split_keys; /* keys per split of the split-KV attention (0 = the default) */
} ik_ink_attrs;

/* InkKvPrologue: pq [T,Nq], pk / pv [T,Ckv(+Hk*K)], k / v taps [K,Ckv], q scale [H,Dh], r scale [H,dr], rel proj [dr,E] ->
 * q [T,H,Dh], rel [T,H,E], k / v [T,Hk,Dh], k / v conv rings [Rr,Ckv] (in place). */
int ik_ink_kv_prologue_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                            const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_ink_attrs* a, ik_tensor* q,
                            ik_tensor* rel, ik_tensor* k, ik_tensor* v, ik_tensor* kring, ik_tensor* vring, const ik_tensor* pos,
                            void* stream);

/* RelSinkAttn: q [T,H,Dh], k / v [T,Hk,Dh], k / v cache [Hk,Cr,Dh] (inputs; the same buffers as the outputs), rel [T,H,E],
 * pq [T,Nq] -> o [T,H*Dh], k / v cache (in place). Scratch: f32 partials (see inkattn.rs `attn_scratch`). */
int ik_rel_sink_attn_bf16(const ik_tensor* q, const ik_tensor* k, const ik_tensor* v, const ik_tensor* kc_in, const ik_tensor* vc_in,
                          const ik_tensor* rel, const ik_tensor* pq, const ik_ink_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc,
                          const ik_scratch* scratch, const ik_tensor* pos, void* stream);

/* InkAttnDecode: RelSinkAttn(InkKvPrologue(..)) for 1..16 rows, bitwise the pair of the two launchers above. Inputs as the
 * prologue's plus the caches; outputs o, k / v cache, k / v conv ring (all in place but o). */
int ik_ink_attn_decode_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                            const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_tensor* kc_in, const ik_tensor* vc_in,
                            const ik_ink_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc, ik_tensor* kring, ik_tensor* vring,
                            const ik_scratch* scratch, const ik_tensor* pos, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* IK_CUDA_ABI_INK_H */
