/* DeepSeek-V4.1 launchers (CUDA only), kernels/cuda/dsv41_*.cu. Semantics: the op contract docs/design/dsv41-ops.md (one
 * launcher per `Op`, same inputs / outputs in the same order, attributes as scalars). Every float tensor may be bf16, f16 or
 * f32 unless a line says otherwise; every tensor is dense row-major. Positioned launchers take an optional device-side
 * position `pos` (i32 [1], null = the host `pos0`) where the op's shapes do not depend on it. */
#ifndef IK_CUDA_ABI_DSV41_H
#define IK_CUDA_ABI_DSV41_H

#include "abi/ik_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* hc_expand: x [T,D] -> y [T,S,D], S copies of each row. */
int ik_hc_expand(const ik_tensor* x, ik_tensor* y, void* stream);

/* hc_mixes: x [T,S,D], fn f32 [(2+S)S, S*D], scale f32 [3], base f32 [(2+S)S] -> pre f32 [T,S], post f32 [T,S], comb f32
 * [T,S,S]; S <= 4. mix = (fn @ flatten(x)) * rsqrt(mean(flatten(x)^2) + eps), then the split and the Sinkhorn of
 * hc_split_sinkhorn (kernel.py:406-475), all in f32. Scratch: f32 partial dots [T, nsplit, (2+S)S + 1] (see dsv41_hc.cu,
 * `hc_nsplit`; the Rust side mirrors it). */
int ik_hc_mixes(const ik_tensor* x, const ik_tensor* fn, const ik_tensor* scale, const ik_tensor* base, float eps,
                int64_t iters, float hc_eps, ik_tensor* pre, ik_tensor* post, ik_tensor* comb, const ik_scratch* scratch,
                void* stream);

/* hc_pre: x [T,S,D], mix f32 [T,S] -> y [T,D] = sum_s mix[s] x[s] (f32 sum in stream order). */
int ik_hc_pre(const ik_tensor* x, const ik_tensor* mix, ik_tensor* y, void* stream);

/* hc_post: x [T,D], res [T,S,D], post f32 [T,S], comb f32 [T,S,S] -> y [T,S,D], y[k] = post[k] x + sum_j comb[j,k] res[j]. */
int ik_hc_post(const ik_tensor* x, const ik_tensor* res, const ik_tensor* post, const ik_tensor* comb, ik_tensor* y,
               void* stream);

/* hc_sublayer: the fused sublayer entry (Op::HcSublayer). x [T,S,D], fn / scale / base as hc_mixes, pre_in f32 [T,S] (the
 * previous sublayer's pre-mix), norm_w [D] (x's dtype or f32), then post_x [T,D], post_mix f32 [T,S], post_comb f32 [T,S,S]
 * (the fold: h = hc_post(post_x, x, post_mix, post_comb), stored to h_out [T,S,D]) or post_x null / empty -> n [T,D] =
 * rmsnorm(hc_pre(h, pre_in), norm_w), pre / post f32 [T,S], comb f32 [T,S,S] = hc_mixes(h). S <= 4, S*D % 4 == 0, D <= 12288.
 * Scratch: f32 partials [T, nsplit, (2+S)S + 1] (`hc_sub_nsplit`, mirrored on the Rust side). Two launches. */
int ik_hc_sublayer(const ik_tensor* x, const ik_tensor* fn, const ik_tensor* scale, const ik_tensor* base, const ik_tensor* pre_in,
                   const ik_tensor* norm_w, const ik_tensor* post_x, const ik_tensor* post_mix, const ik_tensor* post_comb, float eps,
                   float w_offset, int64_t iters, float hc_eps, ik_tensor* n, ik_tensor* pre, ik_tensor* post, ik_tensor* comb,
                   ik_tensor* h_out, const ik_scratch* scratch, void* stream);

/* moe_route: logits f32 [T,E], bias f32 [E] -> weights f32 [T,K], ids i32 [T,K]. score_func 0 softmax, 1 sigmoid, 2
 * sqrt(softplus); experts = top-K of score + bias (ties to the lower index), weights = score[experts] (normalised by their
 * sum + 1e-20 when norm_topk and K > 1) * scale. E <= 1024, K <= 32. */
int ik_moe_route(const ik_tensor* logits, const ik_tensor* bias, int64_t top_k, float scale, int32_t norm_topk,
                 float temperature, int32_t score_func, ik_tensor* weights, ik_tensor* ids, void* stream);

/* silu_glu_clamp: gate, up [R,I] -> silu(min(gate, l)) * clamp(up, -l, l); limit 0 = no clamp. */
int ik_silu_glu_clamp(const ik_tensor* gate, const ik_tensor* up, float limit, ik_tensor* out, void* stream);

/* act_qdq: x [.., N] -> y (same shape and dtype): per block of `block` elements of the last dim quantise and dequantise.
 * fmt 0 e4m3, 1 e2m1; scale_fmt 0 ue8m0, 1 e4m3, 2 f32 (act_quant / fp4_act_quant of kernel.py, the cpu reference's
 * `qdq_block` bit for bit). block: a power of two <= 32 or a multiple of 32. */
int ik_act_qdq(const ik_tensor* x, int32_t fmt, int64_t block, int32_t scale_fmt, ik_tensor* y, void* stream);

/* rope_tail: x [T,N,Dh], inv_freq f32 [rd/2] -> out: the last rd elements rotated as adjacent (re, im) pairs, row t at
 * position (pos ? *pos : pos0) + t * pos_stride, angle = f32(pos) * inv_freq[i] in f32; inverse rotates by -angle. */
int ik_rope_tail(const ik_tensor* x, const ik_tensor* inv_freq, int64_t pos0, int64_t pos_stride, int32_t inverse,
                 ik_tensor* out, const ik_tensor* pos, void* stream);

/* window_kv: kv [T,D] -> win [W+T,D] (rows < W from ring slots p % Wc of positions pos0-W+r, zero below 0; then the chunk),
 * ring [Wc,D] updated in place with the chunk's last min(T,Wc) rows. kv, win and ring share one dtype. Two launches (the
 * ring is read before it is written). */
int ik_window_kv(const ik_tensor* kv, int64_t window, int64_t pos0, ik_tensor* win, ik_tensor* ring, const ik_tensor* pos,
                 void* stream);

/* attn_sparse: q [T,H,D], win [W+T,D], cmp [Nc,D], idx i32 [T,Kc] (-1 = none), sink f32 [H] -> o [T,H,D]. Query t attends
 * the window rows t+1..t+W (positions >= 0 only) and cmp[idx[t,j]] for idx >= 0; one vector is key and value (MQA); the
 * sink joins the softmax denominator only; a query with no keys gives zeros. Scratch: split-KV partials for T <= 16 (see
 * dsv41_attn.cu `attn_nsplit`), none above. q, win, cmp and o share one dtype. */
int ik_attn_sparse(const ik_tensor* q, const ik_tensor* win, const ik_tensor* cmp, const ik_tensor* idx,
                   const ik_tensor* sink, int64_t window, int64_t pos0, float scale, ik_tensor* out,
                   const ik_scratch* scratch, const ik_tensor* pos, void* stream);

/* compress_pool: kv f32 [T,D], score f32 [T,D] -> out [G,D] (groups completing in the chunk, softmax-gated per channel over
 * each group's `ratio` rows; rows before pos0 from the rings at slot p % Rc), kv ring f32 [Rc,D], score ring f32 [Rc,D]
 * (in place: then the chunk's last min(T,Rc) rows). Two launches. */
int ik_compress_pool(const ik_tensor* kv, const ik_tensor* score, int64_t ratio, int64_t pos0, ik_tensor* out,
                     ik_tensor* kv_ring, ik_tensor* score_ring, void* stream);

/* indexer_blocks: q [T,Hi,Di], k [Nk,Di], w [T,Hi] -> mask u8 [T, ceil(Nk/block)]: the `keep` best blocks (block score =
 * best row score, the query's newest visible block pinned in, -inf blocks dropped, ties to the lower block).
 * indexer_topk: the same q, k, w plus candidates u8 [T, ceil(Nk/block)] or empty [0] -> idx i32 [T, min(top_k, Nk)]: the
 * best rows (ties to the lower index; rows outside kept candidate blocks are -inf), sorted by position, picks past the
 * query's visible rows -1. score[t,c] = scale * sum_h w[t,h] relu(q[t,h] . k[c]) for c < (pos0+t+1)/ratio. Hi <= 32,
 * Hi*Di <= 8192, Di % 8 == 0, 64 % block == 0. Scratch: f32 [rows, Nk] for a tile of `rows` = scratch / (4 Nk) queries
 * (at least one; the launcher runs the chunk tile by tile), never a per-head score. */
int ik_indexer_blocks(const ik_tensor* q, const ik_tensor* k, const ik_tensor* w, int64_t keep, int64_t block,
                      int64_t ratio, int64_t pos0, float scale, ik_tensor* mask, const ik_scratch* scratch, void* stream);
int ik_indexer_topk(const ik_tensor* q, const ik_tensor* k, const ik_tensor* w, const ik_tensor* cand, int64_t top_k,
                    int64_t block, int64_t ratio, int64_t pos0, float scale, ik_tensor* idx, const ik_scratch* scratch,
                    void* stream);

/* transpose01 (dsv41_engram.cu): x [A,B,C] -> y [B,A,C], any 1, 2 or 4 byte dtype (pure byte movement, exact). */
int ik_transpose01(const ik_tensor* x, ik_tensor* y, void* stream);

/* engram_gather: records u8 [T,C,Dh + Dh/32] (per row Dh e4m3 bytes, then Dh/32 E8M0 scales) -> y float [T, C*Dh],
 * e4m3 payload * 2^(scale - 127) in f32 (cpu reference's arithmetic). */
int ik_engram_gather(const ik_tensor* records, ik_tensor* y, void* stream);

/* logits_f16w / matmul_f16w (dsv41_dense.cu): x [M,K] (f32 / bf16 / f16), w f16 [N,K] -> y [M,N] (f32 / bf16 / f16),
 * y = x w^T, f32 accumulation; M <= 65535. The router `ffn.gate` and the indexer `weights_proj`, which the checkpoint keeps
 * in f16. */
int ik_logits_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream);
int ik_matmul_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream);

/* engram_merge: h [T,S,D], kv [T,(S+1)*D] (S keys, then one value), q_w [S,D], k_w [S,D] (same dtype) -> out [T,S,D] (any
 * float dtypes, f32 math): per stream s, dot = sum_d h q_w k_w key * rsqrt(ms(h) + eps) * rsqrt(ms(key) + eps) / sqrt(D),
 * gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))), out = h + gate * value. */
int ik_engram_merge(const ik_tensor* h, const ik_tensor* kv, const ik_tensor* q_w, const ik_tensor* k_w, float eps, ik_tensor* out,
                    void* stream);

/* dense_f16w: x float [M,K], w f16 [N,K] -> y float [M,N] = x w^T, f32 accumulation (the f16 router / indexer weights). */
int ik_dense_f16w(const ik_tensor* x, const ik_tensor* w, ik_tensor* y, void* stream);

#ifdef __cplusplus
}
#endif

#endif /* IK_CUDA_ABI_DSV41_H */
