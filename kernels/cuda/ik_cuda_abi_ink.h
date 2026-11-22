/* Inkling-2-Mini launchers (CUDA only; lane ink-cuda-mixmoe): the in-projection, the depth mix / conv / controller ops and the
 * experts (kernels/cuda/ink_proj.cu, ink_mix.cu, ink_moe.cu). Semantics: the op contract in crates/intarsia-core/src/inkling2/ops.rs
 * (one launcher per `Op`, the op's inputs / outputs in order, attributes as scalars). Activations are bf16 (the model's dtype);
 * weights bf16; queries / msg / dest f32 as the contract says. An absent optional tensor is a zero-element tensor. Positioned
 * launchers take an optional device position `pos` (i32 [1], null = the host `pos0`). Every launcher is rows-exact: row t of a
 * T-row call is bitwise the one-row call on row t (with the state the earlier rows left), except where a line says otherwise. */
#ifndef IK_CUDA_ABI_INK_H
#define IK_CUDA_ABI_INK_H

#include "abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ink_in_proj: x [S,T,D] (S = 1 or 3), w [Nq+Nk+Nv, D] -> pq [T,Nq], pk [T,Nk], pv [T,Nv]; segment j reads x[min(j, S-1)].
 * _mma: one launch of the tensor-core GEMV (any T, in 16-row chunks) (16 weight rows x 16 activation rows per warp, the
 * weight read once per chunk; every row's sum the same mma chain whatever T, so rows-exact). */
int ik_ink_in_proj_bf16_mma(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);
/* the same launcher (the rows-exact registration) */
int ik_ink_in_proj_bf16_mma_rows(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);
/* prefill: three cuBLAS GEMMs */
int ik_ink_in_proj_bf16_cublas(const ik_tensor* x, const ik_tensor* w, ik_tensor* pq, ik_tensor* pk, ik_tensor* pv, void* stream);

/* sconv_res: x [T,C], taps [K,C] -> y [T,C], ring [Rr,C] (in place), sum [T,C] (with res; addend optional). */
int ik_sconv_res_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend, int64_t pos0,
                      const ik_tensor* pos, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, void* stream);

/* attn_res_mix: src [N,R,D], last [R,D], addend [R,D] or empty, queries f32 [Q,D], norm [Q,D] or empty -> out [Q,R,D]
 * (bf16 with norm, f32 without). */
int ik_attn_res_mix_bf16(const ik_tensor* src, const ik_tensor* last, const ik_tensor* addend, const ik_tensor* queries,
                         const ik_tensor* norm, float eps, ik_tensor* out, void* stream);

/* conv_mix: SconvRes, then the depth mix over src [N,T,D] with queries [Q,D] (Q may be 0) and either the norm or, with msg
 * [S,T,D] f32, the controller blend (down [S,G,2D], gain / forget [S,D,G], norm [S,D]). Scratch (blend only): see
 * `ink_blend_scratch_bytes`. One launch without the blend when T + K - 1 <= Rr (else + the deferred ring write); the blend
 * adds three (h GEMV, gate, norm). */
int ik_conv_mix_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend,
                     const ik_tensor* src, const ik_tensor* queries, const ik_tensor* norm, const ik_tensor* msg,
                     const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, int64_t pos0, const ik_tensor* pos,
                     float eps, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, ik_tensor* mix, const ik_scratch* scratch,
                     void* stream);

/* loop_blend: dest f32 [Sd,R,D] (Sd 1 or S), msg f32 [S,R,D], down [S,G,2D], gain / forget [S,D,G], norm [S,D] -> out [S,R,D].
 * Scratch: `ink_blend_scratch_bytes`. */
int ik_loop_blend_bf16(const ik_tensor* dest, const ik_tensor* msg, const ik_tensor* down, const ik_tensor* gain,
                       const ik_tensor* forget, const ik_tensor* norm, float eps, ik_tensor* out, const ik_scratch* scratch,
                       void* stream);

/* Scratch bytes of the blend for S streams, R rows, width D, rank G (mirrored by the Rust side). */
int64_t ik_ink_blend_scratch_bytes(int64_t s, int64_t r, int64_t d, int64_t g);

/* moe_experts (NVFP4, ink.situ): x [T,H] bf16, ids i32 [T,K], weights f32 [T,K], gate_up w u8 [E,2I,H/2], scales e4m3
 * [E,2I,H/16], down w u8 [E,H,I/2], scales e4m3 [E,H,I/16], global scales f32 [E,3] -> out [T,H] (bf16 or f32).
 * Union-grouped: each expert's rows are read once per call for all the pairs routed to it. Scratch: the core
 * `moe_experts_scratch` layout. */
int ik_moe_experts_situ_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                             const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                             float gate_cap, float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream);
/* the same launcher (the rows-exact registration) */
int ik_moe_experts_situ_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                                  const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                                  float gate_cap, float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* moe_experts_mxfp8: x [T,H] bf16, ids, weights, w13 e4m3 [E,2I,H], w13 scales u8 (E8M0) [E,2I,H/32], w2 e4m3 [E,H,I], w2
 * scales u8 [E,H,I/32] -> out [T,H]. act: 0 silu, 1 gelu_tanh, 2 situ (gate_cap, up_cap). Scratch as above. */
int ik_moe_experts_mxfp8_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                              const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                              float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream);
int ik_moe_experts_mxfp8_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                                   const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                                   float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream);

/* Benchmarks only: 1 launches only the gate+up kernel of an experts call, 2 only the down kernel (the output is garbage). */
void ik_ink_moe_probe(int v);

#ifdef __cplusplus
}
#endif

#endif /* IK_CUDA_ABI_INK_H */
