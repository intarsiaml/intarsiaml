/* Launchers of the Inkling-2-Mini attention ops: InkKvPrologue (70), RelSinkAttn (71), InkAttnDecode (75) on HIP
 * (lane ink-hip-attn; crates/intarsia-backend-hip/src/inkattn.rs holds the table, kernels.rs checks that every launcher of that table is
 * declared here). Contract: crates/intarsia-core/src/inkling2/ops.rs. Every launcher returns an ik_status, never synchronises
 * and never allocates. Every tensor is bf16 and dense row-major; `pos` is the optional device position (i32 [1], null =
 * attrs->pos0). Same operands and attributes as the CUDA launchers (kernels/cuda/ik_cuda_abi_ink.h). */
#ifndef IK_HIP_ABI_INKATTN_H
#define IK_HIP_ABI_INKATTN_H

#include "../abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* mirrored by `IkInkAttnAttrs` in crates/intarsia-backend-hip/src/inkattn.rs (repr(C)); field order as CUDA's ik_ink_attrs */
typedef struct {
    int64_t n_heads, n_kv_heads, head_dim, d_rel;
    int64_t pos0, window, tau_floor;
    int32_t dyn_conv, kv_ring;
    float qk_clip, rel_clip, eps, tau_alpha, scale;
    int32_t split_keys; /* keys per split of the split-KV attention (0 = the default, 64) */
    int32_t flash;      /* RelSinkAttn: prefill calls (T > 16, Dh >= 16) on the WMMA flash kernel (INTARSIA_HIP_INK_ATTN_FLASH; not bitwise) */
    int32_t diag;       /* INTARSIA_INK_DEC_DIAG (op 75 timing diagnostics, wrong output): bit 0 no prologue jobs, bit 1 no merge */
} ik_ink_attn_attrs;

/* InkKvPrologue: pq [T,Nq], pk / pv [T,Ckv(+Hk*K)], k / v taps [K,Ckv], q scale [H,Dh], r scale [H,dr], rel proj [dr,E] ->
 * q [T,H,Dh], rel [T,H,E], k / v [T,Hk,Dh], k / v conv rings [Rr,Ckv] (in place). Dh <= 256, dr <= 32, K <= 4, H / Hk <= 4. */
int ik_ink_kv_prologue_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                            const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_ink_attn_attrs* a, ik_tensor* q,
                            ik_tensor* rel, ik_tensor* k, ik_tensor* v, ik_tensor* kring, ik_tensor* vring, const ik_tensor* pos,
                            void* stream);

/* RelSinkAttn (scalar split-KV, every T; with attrs->flash, calls of more than 16 rows and Dh >= 16 on the WMMA flash kernel): q [T,H,Dh], k / v [T,Hk,Dh], k / v cache [Hk,Cr,Dh] (inputs; the same buffers as the
 * outputs), rel [T,H,E], pq [T,Nq] -> o [T,H*Dh], k / v cache (in place). Dh in {8,16,32,64,128}, H / Hk in 1..3. Scratch: f32
 * partials of one slice of at most 64 rows, min(T,64) * H * 32 * (Dh + 2) * 4 bytes (inkattn.rs `attn_scratch`). */
int ik_rel_sink_attn_bf16(const ik_tensor* q, const ik_tensor* k, const ik_tensor* v, const ik_tensor* kc_in, const ik_tensor* vc_in,
                          const ik_tensor* rel, const ik_tensor* pq, const ik_ink_attn_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc,
                          const ik_scratch* scratch, const ik_tensor* pos, void* stream);

/* InkAttnDecode (op 75, 1..16 rows): RelSinkAttn(InkKvPrologue(..)) in one dispatch, bitwise the pair of the two launchers
 * above. Inputs as the prologue's plus the caches; outputs o, k / v cache, k / v conv ring (all in place but o). Scratch: the
 * partials T * H * 32 * (Dh + 2) * 4 bytes, then the staged K / V rows 2 * T * Hk * Dh * 2 bytes (inkattn.rs `decode_scratch`). */
int ik_ink_attn_decode_bf16(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                            const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_tensor* kc_in, const ik_tensor* vc_in,
                            const ik_ink_attn_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc, ik_tensor* kring, ik_tensor* vring,
                            const ik_scratch* scratch, const ik_tensor* pos, void* stream);

/* InkAttnDecode row-table form (batched decode, lane ink-hip-batch; `OpCall::rows`): row r (T <= 16 rows) is the T = 1 call of
 * sequence r at position pos[r] (i32 [T], required) on the caches / conv rings at entries index .. index + 3 of row r of tab
 * (u8 [T, stride, 8] device addresses) plus roff[0..4] (the carriers' byte offsets: k cache, v cache, k ring, v ring); kc / vc /
 * kring / vring are row 0's carriers (shapes). attrs->pos0 is not read. Scratch: T * H * 32 * (Dh + 2) * 4 bytes. */
int ik_ink_attn_decode_bf16_rows(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt, const ik_tensor* vt,
                                 const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp, const ik_tensor* kc_in, const ik_tensor* vc_in,
                                 const ik_ink_attn_attrs* a, ik_tensor* o, ik_tensor* kc, ik_tensor* vc, ik_tensor* kring, ik_tensor* vring,
                                 const ik_scratch* scratch, const ik_tensor* pos, const ik_tensor* tab, int64_t stride, int64_t index,
                                 const int64_t* roff, void* stream);

#ifdef __cplusplus
}
#endif

#endif
