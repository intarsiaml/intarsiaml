# DeepSeek-V4.1 Flash: op contract, state and graph (lane 1a)

This is the contract that lanes 1b (cpu reference ops), 2 (CUDA) and the Engram lane implement against. Lane 1a appended
every op DeepSeek-V4.1 needs to `Op` once (discriminants 44..=61, `crates/intarsia-core/src/ops.rs`, never insert above
them), added the state slots, the Engram tier interface, the typed config and the model graph. The model does not run yet:
`intarsia plan show --model <dir> --backend cpu` lists the full op sequence of the checkpoint from its `config.json` and
safetensors headers, and loading it for generation refuses with a pointer to `plan show`.

Plan: docs/plans/2026-10-31-dsv41-unpark.md. Design and Engram variant A2: docs/research/2026-10-20-dsv41-flash-scope.md.

References are to DeepSeek's official inference code (MIT), as checked out in `.worktrees/dsv41-official/`:
`model.py`, `kernel.py`, `engram.py`. Line numbers are of that checkout.

## Where things live

| What | File |
|---|---|
| Op attributes (`DsvAttrs` in `OpAttrs::dsv`), call checks, shape facts | `crates/intarsia-core/src/dsv41/ops.rs` |
| Admission sweep cases | `crates/intarsia-core/src/dsv41/cases.rs` |
| cpu kernels (both `ref_*` and `cpu_f32_*`), `STUBS` | `crates/intarsia-backend-cpu/src/dsv41_ref.rs` |
| Typed config (HF nested and official flat form) | `crates/intarsia-core/src/dsv41/config.rs` |
| Graph IR and builder, state layout | `crates/intarsia-core/src/dsv41/graph.rs` |
| `plan show` recorder and renderer | `crates/intarsia-core/src/dsv41/plan.rs` |
| Header-only weight shapes, detection, load refusal | `crates/intarsia-models/src/dsv41.rs` |
| Engram hash, `EngramTier`, `MemEngramTier` | `crates/intarsia-core/src/dsv41/engram.rs` |
| Ring / Shared state slots | `crates/intarsia-core/src/state.rs` (`SlotSpec::Ring`, `SlotSpec::Shared`) |
| Recipe (matching, layout names) | `recipes/deepseek_v41.toml` |
| Tests | `crates/intarsia-models/tests/dsv41.rs`, `crates/intarsia-backend-cpu/tests/{admission,ops}.rs`, unit tests in `dsv41/*` |

## Conventions

All tensors are dense row-major. A chunk is `T` rows at absolute positions `pos0 .. pos0 + T`, for every op. The official
code runs only a prefill from position 0 or a one-token decode step; the rules below are the general form and reduce to
those two cases (lane 1b's fixtures should cover both, plus a chunked prefill). "Positioned" ops (`RopeTail`, `WindowKv`,
`AttnSparse`, `CompressPool`, `IndexerBlocks`, `IndexerTopk`, and `ActQdq` in its row-offset form) also accept a device-side
position (`OpCall::pos`, `i32 [1]`) for graph replay, with capacity-sized shapes ("Device position" below). `D` = 512 (head_dim), `S` = 4 (hc_mult), `H` = 64 heads, `W` = 128 (window).

Outputs never alias inputs: the graph ping-pongs the residual (`x0` / `x1`) and gives every intermediate its own
workspace name, so a kernel may write an output before it has read all of an input (`HcPost` mixes across streams,
`EngramMerge` reads every stream). The only in-place tensors are the state rings of `WindowKv` and `CompressPool`, which
are outputs only (tested in `crates/intarsia-models/tests/dsv41.rs`).

Activation dtype is bf16 unless stated; "float" means bf16 / f16 / f32. Ops that the official code runs in fp32 take or
produce f32 where it matters (hyper-connection mixes, router logits, compressor inputs).

## The ops

Status: **ref** = cpu reference in `dsv41_ref.rs` (both kernel sets run it, admission compares them), **stub** = kernels
registered so plans resolve, every launch returns `unsupported: <op>: not implemented (lane 1b)`, and admission rejects
them with that error (`crates/intarsia-backend-cpu/tests/admission.rs` asserts both).

| # | Op | Inputs -> outputs | Attributes | Mirrors | Status |
|---|---|---|---|---|---|
| 44 | `HcExpand` | x `[T,D]` -> y `[T,S,D]` (copies) | | `Transformer.forward` `model.py:1261` | ref |
| 45 | `HcMixes` | x `[T,S,D]` float, fn `[(2+S)S, S*D]` f32, scale `[3]` f32, base `[(2+S)S]` f32 -> pre `[T,S]`, post `[T,S]`, comb `[T,S,S]` (all f32) | `eps` (norm_eps, the RMS over the flattened `S*D` row), `dsv.iters` (20), `dsv.hc_eps` (1e-6) | `Block.hc_mixes` `model.py:950`, `hc_split_sinkhorn` `kernel.py:406-475` | ref |
| 46 | `HcPre` | x `[T,S,D]`, mix `[T,S]` f32 -> y `[T,D]` = sum_s mix[s] x[s] | | `Block.hc_pre` `model.py:959` | ref |
| 47 | `HcPost` | x `[T,D]`, res `[T,S,D]`, post `[T,S]` f32, comb `[T,S,S]` f32 -> y `[T,S,D]`, y[k] = post[k] x + sum_j comb[j,k] res[j] | | `Block.hc_post` `model.py:964` | ref |
| 48 | `MoeRoute` | logits `[T,E]` f32, bias `[E]` f32 -> weights `[T,K]` f32, ids `[T,K]` i32 | `top_k`, `scale` (route_scale 1.5), `norm_topk`, `temperature` (gate_temp, 1), `dsv.score_func` (sqrt-softplus) | `Gate.forward` `model.py:811`: experts chosen by `score + bias`, weights are the unbiased scores, normalised (+1e-20) then scaled | ref |
| 49 | `SiluGluClamp` | gate `[R,I]`, up `[R,I]` -> `[R,I]` = silu(min(gate, l)) * clamp(up, -l, l) | `dsv.limit` (swiglu_limit 10; 0 = no clamp) | `Expert.forward` `model.py:832` | ref |
| 50 | `ActQdq` | x `[.., N]` float -> same shape and dtype: quantise per block of the last dim, dequantise | `dsv.qdq_fmt` (e4m3 / e2m1), `dsv.qdq_block`, `dsv.qdq_scale` (ue8m0 power of two, e4m3, f32) | `act_quant` `kernel.py:98` (amax floor 1e-4, `fast_round_scale`), `fp4_act_quant` `kernel.py:184` (floors 6*2^-126 / 6*2^-9 for e4m3 scales) | ref |
| 51 | `RopeTail` | x `[T,N,Dh]`, inv_freq `[rd/2]` f32 -> rotated copy: the last `rd` elements as adjacent (re, im) pairs, row `t` at position `pos0 + t*pos_stride`, angle `pos * inv_freq[i]` (f32) | `pos0`, `dsv.pos_stride` (ratio for compressed latents: a group stands for its first token), `dsv.inverse` (de-rotation of the attention output) | `apply_rotary_emb` `model.py:392`; tables `precompute_freqs_cis` `model.py:369` (`Dsv41Config::inv_freq`) | ref |
| 52 | `WindowKv` | kv `[T,D]` -> win `[W+T,D]`, ring `[Wc,D]` (in place, `Wc >= W`) | `window`, `pos0` | `Attention._window_kv` `model.py:701` | ref |
| 53 | `AttnSparse` | q `[T,H,D]`, win `[W+T,D]`, cmp `[Nc,D]`, idx `[T,Kc]` i32, sink `[H]` f32 -> o `[T,H,D]` | `window`, `pos0`, `scale` (head_dim^-0.5) | `sparse_attn` `kernel.py:311-404`, window rows `get_window_topk_idxs` `model.py:410` | ref |
| 54 | `CompressPool` | kv `[T,D]` f32, score `[T,D]` f32 -> out `[G,D]`, kv ring `[Rc,D]` f32, score ring `[Rc,D]` f32 (rings in place) | `dsv.ratio` (>= 2), `pos0` | `Compressor.forward` `model.py:429-486` (ratio > 1) | ref |
| 55 | `IndexerBlocks` | q `[T,Hi,Di]`, k `[Nk,Di]`, w `[T,Hi]` -> mask `[T, ceil(Nk/B)]` u8 | `top_k` (candidate_topk_blocks 2048), `dsv.block` (8), `dsv.ratio`, `pos0`, `scale` | `select_candidate_blocks` `model.py:584` on the score of `Indexer.forward` `model.py:527` | ref |
| 56 | `IndexerTopk` | q, k, w as above, candidates `[T, ceil(Nk/B)]` u8 or empty `[0]` -> idx `[T, min(top_k, Nk)]` i32 | `top_k` (index_topk 512), `dsv.block`, `dsv.ratio`, `pos0`, `scale` | `Indexer.forward` `model.py:527-582` | ref |
| 57 | `GroupedMatmul` | x `[T,G*K]`, w `[G,N,K]` -> `[T,G*N]`, group g: x[:, gK..] @ w[g]^T | | `einsum("bsgd,grd->bsgr")` `model.py:789` | ref |
| 58 | `Transpose01` | x `[A,B,C]` -> `[B,A,C]` (any dtype) | | layout helper: per-group EXL3 `wo_a` slices need group-major rows | ref |
| 59 | `MatmulFp8Block` | x `[M,K]` float, w `[N,K]` e4m3, w_scale `[ceil(N/b), K/b]` u8 (E8M0) -> `[M,N]` | `dsv.block` (32; the activation is quantised to e4m3 per `b` elements with a ue8m0 scale first) | `linear` `model.py:181`, `fp8_gemm` `kernel.py:208-308` | **stub** |
| 60 | `EngramGather` | records `[T,C,Dh+Dh/32]` u8 (per row: `Dh` e4m3 bytes, then `Dh/32` E8M0 scale bytes; 264 B for Dh 256) -> `[T, C*Dh]` = payload * 2^(scale-127) | | `ParallelEngramEmbedding.forward` `model.py:296-326`; record layout from the plan review (2026-10-31) | ref |
| 61 | `EngramMerge` | h `[T,S,D]`, kv `[T,(S+1)D]` (S keys then one value), q_w `[S,D]`, k_w `[S,D]` -> `[T,S,D]` | `eps` (norm_eps) | `Engram.forward` `model.py:328-367` | ref |
| 62 | `HcSublayer` | x `[T,S,D]`, fn, scale, base (as `HcMixes`), pre_in `[T,S]` f32, norm_w `[D]` (x's dtype or f32), then post_x `[T,D]`, post_mix `[T,S]` f32, post_comb `[T,S,S]` f32 **or** three `[0]` -> n `[T,D]`, pre `[T,S]`, post `[T,S]`, comb `[T,S,S]` (f32), h_out `[T,S,D]` **or** `[0]`. h = x, or with the fold h = `HcPost(post_x, x, post_mix, post_comb)` stored to h_out; (pre, post, comb) = `HcMixes(h)`; n = `RmsNorm(HcPre(h, pre_in), norm_w)` | `eps` (norm_eps: both the mixes' RMS and the norm), `w_offset`, `dsv.iters`, `dsv.hc_eps` | `Block.forward` `model.py:983-995` (`hc_post`, `hc_mixes`, `hc_pre`, `attn_norm` / `ffn_norm`) | ref (fused; added by lane cuda-attn) |

### Exact rules the references implement

- **WindowKv**: `win` row `r < W` holds position `pos0 - W + r` taken from the ring slot `p % Wc` (zero when `p < 0`), rows
  `W..W+T` are the chunk. The chunk's last `min(T, Wc)` rows are then written to the ring at `p % Wc`. With `Wc = W` this
  is the official ring of exactly `window` slots.
- **AttnSparse**: query `t` attends the window rows `t+1 ..= t+W` of `win` (positions `pos0+t-W+1 ..= pos0+t`, skipping
  positions below 0) plus the compressed rows `idx[t, j] >= 0` (`-1` = none). One vector is key and value (MQA, one KV
  head). Softmax over the attended keys with `exp(sink[h] - max)` added to the denominator only. A query with no keys
  gives zeros (the kernel's finite running-max convention, `kernel.py:355`).
- **CompressPool**: group `g` covers positions `g*r .. (g+1)*r`; the groups completing inside the chunk are
  `pos0/r .. (pos0+T)/r` (`G` of them, possibly 0). Rows of a group before `pos0` come from the rings (slot `p % Rc`);
  the output is the per-channel softmax over the group's `r` scores of `kv * softmax(score)`. Afterwards the chunk's last
  `min(T, Rc)` rows go into the rings. Ratio 1 is not this op: it is a plain projection (`Compressor.forward` line 461).
- **IndexerTopk / IndexerBlocks** score: `score[t,c] = scale * sum_h w[t,h] * relu(q[t,h] . k[c])` for the compressed rows
  query `t` can see (`c < (pos0+t+1)/ratio`), `-inf` past them. Topk keeps the `min(top_k, Nk)` best, sorted by
  position; unreachable picks become `-1`. With candidates, rows outside kept blocks are `-inf` first. Blocks: a block's
  score is its best row; the block holding the query's newest visible row is pinned in; `top_k` blocks are kept, `-inf`
  picks dropped. Ties go to the lower index (`torch.topk` leaves them unspecified; this matters only for exactly equal
  scores, and for `-inf` fill-ins when fewer than `top_k` rows are reachable inside the candidates, which cannot happen
  with the shipped 2048 blocks x 8 >= 512).
- **HcMixes**: `mix = (fn @ flatten(x)) * rsqrt(mean(flatten(x)^2) + eps)`; `pre = sigmoid(mix[0:S]*scale[0] + base) + hc_eps`;
  `post = 2 sigmoid(mix[S:2S]*scale[1] + base)`; `comb` = row softmax + hc_eps, column normalise, then `iters - 1` rounds of
  row and column normalisation, each dividing by `sum + hc_eps`.
- **EngramMerge**: per stream `s`: `dot = sum_d h[s,d] q_w[s,d] k_w[s,d] key[s,d] * rsqrt(ms(h[s])+eps) * rsqrt(ms(key[s])+eps) / sqrt(D)`,
  `gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6)))`, `out[s] = h[s] + gate * value`.
- **MoeRoute**: `score = sqrt(softplus(logit / temperature))` (or sigmoid / softmax), experts = top-k of `score + bias`
  (ties to the lower index), weights = `score[experts]`, divided by their sum + 1e-20 when `norm_topk` and `K > 1`, times
  `scale`. Image-span tokens use `gate.bias_vl` instead of `gate.bias` (`Gate.forward` line 822); text-only passes `bias`.

- **HcSublayer**: exactly the op sequence it replaces, every intermediate in the dtype the separate ops store it in: h
  (fold) rounded to h_out's dtype, then `HcMixes(h)`, then y = `HcPre(h, pre_in)` rounded to n's dtype, then
  `RmsNorm(y, norm_w)` with `eps` / `w_offset`. The cpu reference composes the separate references and
  `crates/intarsia-backend-cpu/tests/dsv41_hc_sublayer.rs` asserts the fused outputs equal the sequence bit for bit. The
  fold is all-or-nothing: post_x, post_mix, post_comb and h_out are all present, or all `[0]`. No output aliases an
  input (pre / post / comb must not be the pre_in / post_mix / post_comb buffers).

### Device position (CUDA graphs)

A captured decode / verify pass cannot change shapes or host scalars, so the positioned ops (`dsv41::ops::POSITIONED`:
`RopeTail`, `WindowKv`, `ActQdq`, `CompressPool`, `IndexerBlocks`, `IndexerTopk`, `AttnSparse`) also run with the pass's
first position `P` in device memory (`OpCall::pos`, `i32 [1]`) and with fixed, capacity-sized shapes. With `pos` set,
`attrs.pos0` is ignored entirely (semantics and checks); without it every op is exactly the host form above. `T` = leading
dim of the first input, `R` = `attrs.dsv.ratio` (>= 1, **the graph must set it** on every device-position call below that
uses it, including `RopeTail`, `ActQdq` and `AttnSparse`, whose host forms ignore it).

| Op | Device-position form |
|---|---|
| `RopeTail` | row `t` at position `floor(P / R) * R + t * pos_stride` (R = 1: `P + t * pos_stride`; the compressed latent / index-key ropes set R = the compressor ratio, whose host `pos0` is `(pos0 / r) * r`) |
| `WindowKv` | the host op at `pos0 = P` |
| `ActQdq` | x `[G,N]` rank 2, y `[Ry,N]` same dtype (Ry may differ from G): y row `floor(P / R) + g` = qdq(x row `g`) for every `g < G` with `floor(P / R) + g < Ry`; every other y row untouched (new compressed latents / index keys into the shared slot, y = the whole slot) |
| `CompressPool` | kv, score f32 `[T,D]`, out `[Gmax,D]` (any Gmax): with `G = (P+T)/r - P/r`, out rows `g < min(G, Gmax)` = the host result at `pos0 = P`, rows `>= G` untouched; ring writes = the host's at `pos0 = P`. kv / score rows `>= T` are never read |
| `IndexerBlocks` | q `[T,Hi,Di]`, k `[Ncap,Di]` (a capacity view of the shared index keys), w `[T,Hi]`, mask u8 `[T, ceil(Ncap/B)]`: with `N = min((P+T)/R, Ncap)`, `mask[:, :ceil(N/B)]` = the host result (`pos0 = P`, keys `k[:N]`), `mask[:, ceil(N/B):] = 0` |
| `IndexerTopk` | k `[Ncap,Di]`; cand empty or u8 `[T, ceil(Ncap/B)]` (only its first `ceil(N/B)` columns are read); idx i32 `[T,Kw]`, `Kw = min(top_k, Ncap)`: with `Kh = min(top_k, N)`, `idx[:, :Kh]` = the host result (`pos0 = P`, `k[:N]`, `cand[:, :ceil(N/B)]`), `idx[:, Kh:] = -1`; `N = 0` gives an all `-1` idx |
| `AttnSparse` | cmp `[Ncap,D]` (Ncap may be 0), idx i32 `[T,Kw]`: when `Ncap > 0`, `N = min((P+T)/R, Ncap)`, `Kc = min(Kw, N)` and the output = the host result (`pos0 = P`, `cmp[:N]`, `idx[:, :Kc]`); when `Ncap = 0`, `Kc = Kw`. idx columns `>= Kc` are never read |

**Bitwise rule.** A device-position call produces exactly the bits of the equivalent host call on the same GPU, outside the
documented untouched / `-1` / `0` regions. The CUDA kernels get there by making every host-side decision from the shapes
only (grids, tiles, the tensor-core vs SIMT choice, the fast multi-block vs one-block select, scratch layout) and computing
`P`, `N`, `G`, `Kc`, `Kh` on the device: `AttnSparse` sizes its grid from `Kw` and computes the split count of the
decode / verify path (`T <= 16`: `nsplit = min(cdiv(W + Kc, tile), 32)`, tile 32 keys for bf16) from `Kc` exactly as the
host computes it from its idx width; blocks of splits `>= nsplit` exit, one split writes the output directly, more go
through the merge with the same partial layout. Work is bounded by `N`: score blocks past `N` exit before loading the
query, the selects run over `N` entries with `k = min(top_k, N)` (blocks: `min(keep, ceil(N/B))`), the multi-block select
bounds its segments by `N`. The indexer keeps the host's static choice of the multi-block select by the capacity length
(`>= 16384` rows): both selects are exact, so the picks are the same. Each query's visible rows are clamped to `N`.

The cpu reference (`dsv41_ref::run_dev`) runs the host computation on the leading rows at `pos0 = P` into temporaries
and scatters / pads into the capacity outputs; the cpu backend passes the position of these ops through unresolved.
Admission cases carry the position in `DsvAttrs::host_dev_pos` (`exec::run_host` uploads it as `OpCall::pos`; nothing
else reads it). `crates/intarsia-backend-cuda/tests/dsv41_dpos.rs` compares every device-position call with its host call
bitwise on the GPU, replays a captured graph of the device calls at many positions, and (`INTARSIA_DSV41_DPOS_BENCH=1`)
prints the device vs host timing.

Scratch for capacity shapes: `index_scratch` with `kv_len = Ncap` (k's rows) and `m = T` (one tile of `[T, Ncap]` f32
scores, plus the multi-block select's buffers when `T <= 16` and `Ncap >= 16384`); `attn_scratch` does not depend on the
index count (`T * H * 32 * (D + 2) * 4` bytes for `T <= 16`). Both come from `ShapeInfo::from_call` of the capacity call.

### Ops that existing ones cover

| Need | Op used | Note |
|---|---|---|
| Linear layers of the EXL3 checkpoint (q, kv, compressor, indexer, `wo_a` slices, `wo_b`, shared expert, Engram `wkv`, head) | `MatmulExl3` | f32 output where the official code computes in f32 (compressor ratio > 1, head) |
| Router logits (f16 weight, f32 out) | `Logits` | `linear(x.float(), weight.float())` |
| Indexer `weights_proj` (f16 weight) | `Matmul` | its `softmax_scale * n_heads^-0.5` factor is the indexer ops' `scale` |
| Norms (eps 1e-20) | `RmsNorm` | |
| Routed + shared expert sum | `Add` | the official sum is in f32 before the cast; one rounding apart |
| Token embedding | `Embedding` | |
| Mia's routed EXL3 experts (384 per layer, top 6, clamped SwiGLU) | `MoeExperts`, EXL3 encoding (`quant::exl3_moe`, from `dsv41/cuda-moe` `8e2b8b54b`): an i16 input 3 selects it; inputs x `[T,H]`, ids `[T,K]` i32, weights `[T,K]` f32, arena `[W]` i16 (every expert's gate / up / down tiles), scales `[E, 3H+3I]` f16, meta `[E, 8]` i32 (16-byte arena offsets and width per matrix), codebook `[1]` i32, limit `[1]` f32 (swiglu_limit) | f64 cpu reference in `exl3_ref`; the graph sizes the arena as `E x` expert 0's tiles (one width per stack in Mia, verified over all of `tensor_storage`) |
| Base checkpoint's FP4 routed experts | `MoeExperts` (MXFP4: e2m1 + E8M0 per 32) | needs `dsv.limit` honoured by its kernels (an attribute, not a new op; lane 1b) |

`MatmulFp8` does not cover the base's FP8 linears: it takes a per-tensor scale and a static activation scale, the base has
32x32 E8M0 block scales and dynamic per-32 activation quantisation. Hence `MatmulFp8Block`.

## State

| Slot | Per | Shape | Truncation |
|---|---|---|---|
| `Ring { kind: WindowKv }` | every layer (40) | `[128 + 16, 512]` act dtype | by length within `margin` 16 (`graph::ROLLBACK_MARGIN`: DSpark blocks of 5, n-gram drafts); a deeper cut is refused (`Unsupported`), except back to 0 |
| `Ring { kind: CompressTail, planes: 2 }` | KV sources with ratio > 1 (2, 8, 14) | 2 planes `[18, 512]` f32 (kv, score): a multiple of the ratio holding `ratio - 1 + 16` tokens (`graph::tail_rows`) | same |
| `Shared { kind: CompressedKv }` | KV sources (2, 8, 14, 20) | `[ceil(cap/ratio), 512]` | by length (rows past `len/ratio` are stale) |
| `Shared { kind: IndexK }` | KV sources (2, 8, 14, 20) | `[ceil(cap/ratio), 128]` | same |

Each `Shared` slot records its writer layer and its reader layers. The readers of a source are the layers whose latest KV
source at or before them with the same ratio is that layer (`Dsv41Config::kv_source`):
compressed KV 2 -> 2..7, 8 -> 8..13, 14 -> 14..19, 20 -> 20..39; index keys 2 -> 2, 8 -> 8, 14 -> 14,
20 -> 20, 24, 28, 32, 36. Per-pass carries (not state): the top-k index list of the latest index source
(`topk_idxs.<src>`, layers 3..7 use layer 2's, 21..23 layer 20's, 25..27 layer 24's, ...) and layer 20's candidate mask.
At 128K tokens the slots take about 405 MiB per sequence in bf16 (window rings 5 MiB, compressed KV 3 x 64 MiB + 128 MiB,
index keys 80 MiB).

The prefix store and batched decode row tables refuse ring / shared slots for now (`Unsupported`).

## Multi-stream residual

The residual is `[T, 4, 5120]` ("x" in the graph). Per layer: (Engram) -> `HcMixes(hc_attn)` -> `HcPre(prev pre-mix)` ->
norm -> attention -> `HcPost(attn post, comb)` -> `HcMixes(hc_ffn)` -> `HcPre(attn pre-mix)` -> norm -> MoE ->
`HcPost(ffn post, comb)`. The pre-mix of a sub-block is the one the *previous* sub-block computed (`Block.forward`
`model.py:970`): layer 0's attention uses a constant one-hot `[1,0,0,0]` (`make_identity_pre_mix` `model.py:1162`), the
head collapses with the last FFN's pre-mix (`Transformer.forward` line 1271). `plan show` reports
`residual streams 4`. `ParallelHead.forward` (`model.py:1010`) is a plain f32 linear of the last row: no further mix.

### Wiring the fused `HcSublayer` (for the graph owner)

The graph still emits the separate ops; switching is local to `build` in `graph.rs` and changes no buffer or slot:

- Attention sub-block of layer `l`: replace `HcMixes(hc_attn)` + `HcPre(attn)` + `RmsNorm(attn_norm)` by
  `HcSublayer(x, hc_attn_fn, hc_attn_scale, hc_attn_base, pre_mix, attn_norm.weight, [0], [0], [0])` ->
  (`n`, `hc_attn.pre`, `hc_attn.post`, `hc_attn.comb`, `[0]`), attrs `eps = norm_eps`, `dsv.iters`, `dsv.hc_eps`.
- MoE sub-block: fold the attention's `HcPost`: drop `HcPost(attn)` and emit
  `HcSublayer(x, hc_ffn_*, attn_pre, ffn_norm.weight, attn_out, attn_post, attn_comb)` -> (`n`, `hc_ffn.pre`,
  `hc_ffn.post`, `hc_ffn.comb`, `flip(x)`), then `x = flip(x)` as before.
- Next layer's attention may fold the FFN's `HcPost` the same way, except when something reads the streams between them:
  an Engram layer (`EngramMerge` runs on the post-`HcPost` streams before the mixes) keeps the separate `HcPost`. The
  DSpark tap reads the streams too; with the fold it must come after the `HcSublayer` (it reads `h_out`). The head keeps
  `HcPost` + `HcPre(head)` + `RmsNorm` (no mixes there).
- Fold at decode / verify only (`T <= 16`). On a 4096-row prefill chunk the fold is slower than the separate `HcPost`
  (sm_121: fused 5.4 ms vs 5.05 ms separate) while the unfolded op still wins (2.76 ms vs 3.25 ms), so prefill graphs keep
  `HcPost` separate and emit `HcSublayer` without the fold. Decode (sm_121, per rank): 16.4 us unfolded (separate 20.8),
  18.4 us folded (separate 24.9).
- The CUDA kernel takes `S <= 4`, `S*D % 4 == 0`, `D <= 12288`; its scratch is `hc_sub_scratch` in
  `crates/intarsia-backend-cuda/src/dsv41.rs` (the plan reserves it like any kernel scratch).

DSpark tap: before the attention of each MTP target layer (37, 38, 39) the graph records `HcPre` of the stream with the
constant mix `1/S` into `main_hidden.{0,1,2}` (`h.mean(dim=2)`, `Transformer.forward` line 1269). Nothing reads it until
the drafting lane lands; it is there so the hook is part of the frozen graph (plan review: "design DSpark hooks in 1a").
Every per-sequence state is a length-truncatable slot (window and compressor rings within their margin, shared caches by
length); the Engram token history is `SequenceState::tokens`; candidates and top-k lists are per-pass carries.

## Engram tier

`crates/intarsia-core/src/dsv41/engram.rs`. The host hashes token ids (`NgramHasher`, the rolling XOR hash of
`NgramHashState.forward` `engram.py:129`; layout primes `EngramLayout` `engram.py:87` derived and checked against
`engram_num_embeddings`; multipliers from numpy PCG64, generated by `tools/gen_dsv41_engram_params.py` with a hash fixture
the tests compare against). The row fetch is issued as soon as the token ids are known (`EngramTier::issue`, never blocks),
waited on before the upload (`EngramTier::wait` writes packed records `[n, 264]`: 256 e4m3 payload bytes then 8 E8M0
scales, the layout `EngramGather` reads; a row outside the rank's share is a zero payload with unit scales). The share is
`EngramTier::owned_rows`, which defaults to `EngramShard::range` (`ParallelEngramEmbedding`'s `ceil(rows / world)` split);
the NVMe tier (`intarsia-engram`, `NvmeEngramTier`) splits at a hash-column boundary to match the table copies on disk,
and its `shard()` then only carries rank and world, so mask with `owned_rows`, not `shard().range`. `touch` (wake the
drive ~1 ms before the ids are known) and `prefetch` (top-k candidates) are optional. `MemEngramTier` is the in-memory test tier.
The plan review (2026-10-31) replaced the scope doc's resident scales (3 GB per rank) with these 264 B records, 15 per
4 KiB page, so nothing table-sized stays in host memory. Not here: the compressed token map (`build_compressed_token_map` `engram.py:17`, a tokenizer
normaliser pass that must yield exactly 99,092 ids) and the NVMe io_uring tier (Engram lane).

## Graph

`dsv41::graph::build(config, shapes, GraphOpts { t, pos0, act, cap })` returns stages (embed, 40 layers, head), each a list
of `Step { op, what, inputs, outputs, attrs }` whose tensors name a checkpoint tensor, a workspace buffer, rows of a state
slot, an engine constant or a host input, with dtype and shape. Every step is a valid `OpCall` on shape-only tensors
(`Step::phantom_call`). Linear layers resolve from the headers: `{p}.trellis` -> `MatmulExl3`, an e4m3 `{p}.weight` with
`{p}.scale` -> `MatmulFp8Block`, otherwise `Matmul` / `Logits`. Mia stores `wo_a` as 8 per-group EXL3 slices
(`wo_a.slice.{g}`): `Transpose01` to group-major rows, 8 `MatmulExl3`, `Transpose01` back; a dense `wo_a` (the base after
`convert.py`) is one `GroupedMatmul`. Routed experts stack per layer at load (`{layer}.ffn.experts.*.w1.trellis`
`[384, ..]`); the graph checks experts 0 and 383 agree. The MTP / DSpark blocks are found and described, not built.

Decode step of the Mia checkpoint (`plan show`, cpu, position 4095), steps per stage:

| Stage | Steps | What differs |
|---|---|---|
| embed | 2 | `Embedding`, `HcExpand` |
| layers 0, 3-7, 9-13, 15-19, 21-23, 25-27, 29-31, 33-35 | 38 | window (+ shared compressed KV and indices from their sources on 3-39) |
| layer 1 | 41 | + Engram (`EngramGather`, `MatmulExl3`, `EngramMerge`) |
| layers 2, 8 | 53 | + compressor ratio 2 (2 f32 `MatmulExl3`, `CompressPool`), index keys, indexer |
| layer 14 | 56 | 2 / 8 plus Engram |
| layer 20 | 52 | compressor ratio 1 (one `MatmulExl3`), index keys, `IndexerBlocks`, indexer |
| layers 24, 28, 32, 36 | 43 | indexer in layer 20's candidates (no own keys) |
| layers 37-39 | 39 | + DSpark tap (`HcPre` with the mean mix) |
| head | 3 | `HcPre`, `RmsNorm`, `MatmulExl3` (f32 logits) |

1653 steps per decode step; 622 of them `MatmulExl3`, 40 `MoeExperts` (EXL3 encoding). No stub op on this checkpoint.

## Final discriminant list (frozen)

Decided 2026-10-31 (coordinator): the routed EXL3 experts use the EXL3 encoding of the existing `MoeExperts`, so the
`MoeExpertsExl3` variant lane 1a first proposed was dropped before it reached any lane that numbers ops. Lanes refer to
ops by name; the numbers below are what `Op as usize` returns and what admission seeds by.

| # | Op | # | Op | # | Op |
|---|---|---|---|---|---|
| 44 | `HcExpand` | 50 | `ActQdq` | 56 | `IndexerTopk` |
| 45 | `HcMixes` | 51 | `RopeTail` | 57 | `GroupedMatmul` |
| 46 | `HcPre` | 52 | `WindowKv` | 58 | `Transpose01` |
| 47 | `HcPost` | 53 | `AttnSparse` | 59 | `MatmulFp8Block` |
| 48 | `MoeRoute` | 54 | `CompressPool` | 60 | `EngramGather` |
| 49 | `SiluGluClamp` | 55 | `IndexerBlocks` | 61 | `EngramMerge` |
|  |  |  |  | 62 | `HcSublayer` (fused, appended later) |

`Op::ALL` has 63 entries; `HcSublayer` (62, the fused hyper-connection sublayer entry, appended by lane cuda-attn at the
coordinator's request) is last. The next op anyone appends is 63.

## Where the official code differs from the scope doc

- Index keys are owned by the KV sources only (2, 8, 14, 20; `Indexer.owns_k` `model.py:500`). The indexers on 24, 28,
  32, 36 compute queries and weights and read layer 20's keys. Non-index-source layers also reuse the top-k index list of
  their latest index source, not only its KV.
- Only layer 20 compresses at ratio 1, and there it is a plain projection (no `wgate`, no pooling, no tail state); layers
  21-39 read its cache. The ratio-2 compressors are on 2, 8 and 14 only. Layers 37-39 are ratio 1; the window-only layers
  are 0, 1 and the three MTP blocks.
- Layer 20's own top-k is not masked by its candidates; only the later index sources (24..36) mask.
- Hyper-connections need four ops (expand, mixes, pre, post), and each sub-block collapses with the previous sub-block's
  pre-mix.
- Activation fake-quantisation is part of the math: the window KV (whole 512-wide row) goes through e4m3 per 32 (ue8m0),
  the compressed KV (rotated first, whole 512-wide row) through e2m1 per 16 with e4m3 scales, index keys and queries
  through e2m1 per 32 (ue8m0): `ActQdq`. The attention output goes through e4m3 per 32 before `wo_a` **only for the FP8
  base checkpoint** (`model.py:785`, `quant == "official"` in the fixtures); the Mia EXL3 checkpoint, TensorFold and the
  committed fixtures do not (layer-0 `o_lora` disagrees with the fixture by 2.5e-2 with it, 2.5e-6 without), so the
  graph inserts that step for `Dsv41Quant::Fp8` only. The attention output is de-rotated before `wo_a`. YaRN changes only
  the frequencies (no mscale).
- The Mia EXL3 checkpoint has no Engram tables (`config.json` `engram_table_dir: null`, no `engram.embed` tensors): they
  have to come from the base (FP8 + E8M0, 2 x ~98 GB). Its other non-EXL3 tensors: norms (bf16), hc weights, attention
  sinks, router bias (f32), router weight and `weights_proj` (f16), embedding (bf16).
- `gate.bias_vl` is the vision-language routing bias for image-span tokens (`Gate.forward`), unused text-only.
- `MatmulFp8` does not cover the base's FP8 (block E8M0 scales, dynamic activation quantisation): new `MatmulFp8Block`.

## Parity with the official code (lane 1b)

Every cpu reference op is checked against DeepSeek's official `model.py` / `kernel.py` / `engram.py` through the fixtures in
`golden/dsv41/` (generator `tools/dsv41/fixtures.py`, which runs the official code on CPU with torch emulations of the
tilelang kernels; the emulations are not bit-checked against the GPU kernels). Run `cargo test -p intarsia-backend-cpu
--test dsv41_parity --test dsv41_chunked -- --nocapture` for the `PARITY ...` rows; tolerance is `max |got - ref| / max |ref|`.

| Op | Fixture | Result (max abs error / max abs ref) | Tolerance |
|---|---|---|---|
| `ActQdq` (fp8 b32 ue8m0, fp4 b32 e8m0, fp4 b16 e4m3) | `ops` (4096 values each) | bit-exact | 0 |
| `HcMixes` (sigmoid split, 20 Sinkhorn rounds) | `ops` `sinkhorn/*` (32 rows) | pre 7.2e-8, post 9e-8, comb 1.4e-7 | 1e-6 |
| `HcPre`, `HcPost` | `ops` `hc_pre`/`hc_post` (bf16); layer-2 taps (prefill and decode); fp32 chain | bf16 6.4e-9 and 3.0e-3 (6 of 81,920 values round to the neighbouring bf16: one ulp); layer-2 taps <= 4.9e-5; fp32 chain 1.3e-7 | 4e-3 / 1e-5 |
| `MoeRoute` (`bias`, `bias_vl`) | layer-2 router logits, 16 tokens (7 image rows, all 7 route differently under `bias_vl`) | all expert sets equal; weights 1.0e-7 | 1e-5 |
| `SiluGluClamp`, `GroupedMatmul`, `Transpose01` | naive transcriptions of `Expert.forward` / `einsum` | exact | 1e-6 |
| `RopeTail` | `ops` rope plain + compressed_yarn: tables at positions up to 1,048,575, rows at 5, 6, 7, 70000, forward and inverse; layer-2 `o_sparse -> o_unrot` | tables 3.0e-8 (cos/sin), rows bit-exact | 1e-5 / 4e-3 |
| `RopeTail` then `ActQdq` on compressed rows | layer-2 `latent_pre_rope -> latent` taps | bit-exact (e2m1 per 16 with e4m3 scales over the whole 512-wide row, after the rotation) | 0 |
| `WindowKv`, `CompressPool` (ratio 2, partial groups) | naive `_window_kv` / `Compressor.forward`; chunks 5,20,1,1,13 and 7,1,1,2 | exact | 1e-6 |
| `IndexerBlocks` | `ops` `select_candidate_blocks` (rows of 20,000 / 15,000 / 3,000 / 9) | identical kept blocks | exact |
| `IndexerTopk` | layer-2 selections (16 prefill rows + 2 decode steps); tie rule below | identical | exact |
| `AttnSparse` | `sparse_attn_4k_{bf16,fp32}` (real layer-2 data: 10 / 8 rows at positions 0 .. 4094, window 128 + up to 512 compressed rows, 64 heads, sink); synthetic with `-1` and an all-masked row | bf16 5.3e-3, fp32 6.7e-7, synthetic 4.8e-3 (all-masked row exactly 0) | 8e-3 / 1e-5 / 8e-3 |
| `EngramMerge` / `EngramGather` | layer-1 taps (`engram.in/kv/out`, `engram.rows`) | 4.1e-7 (6 of 327,680 values differ, by 3.8e-6 absolute: bf16 rounding of near-zero values) / exact | 1e-6 / 0 |
| layers 0, 1, 2 (graph + ops; routed experts and Engram rows taken from the fixture) | `layers_{bf16,fp32}`, prefill 16 + decode at 16 and 17, real Mia weights | bf16 1.4e-3 .. 5.5e-3 per layer output (rms 1.5e-3 .. 3.2e-3); fp32 1.9e-7 .. 6.3e-6 | 1e-2 / 1e-4 |

Why the bf16 tolerances are not zero: the official sparse attention rounds the softmax probabilities to bf16 before `P @ V`
(`kernel.py:380`; the reference keeps them in f64 and rounds the output once), and the bf16 hidden states round at every
sub-block, so two correct implementations differ by a few bf16 ulps of the largest value. fp32 flows agree to ~1e-6 (the ops
compute in f64 and round to f32; torch accumulates in f32).

Behaviour the tests pin that the op table leaves implicit:

- `AttnSparse` always attends the query's own window key (`window >= 1`, the key at position `pos0 + t`), so an all-masked row
  (zeros) can only come from a call that leaves the window out, as the official kernel's synthetic fixture does; the
  reference returns zeros for a row without keys (`keys.is_empty()`).
- The compressed latent is `RmsNorm -> RopeTail (pos_stride = ratio, first position of the group) -> ActQdq (e2m1, 16, e4m3)`,
  index keys `wk -> RmsNorm -> RopeTail -> ActQdq (e2m1, 32, ue8m0)`; `attn.idxs` of the official code indexes
  `[window ring 128 | chunk | compressed]` (compressed row `idx - 144` in a 16-row prefill, `idx - 129` in a decode step).
- Chunked == whole, decode-after-prefill == prefill-all and a rollback of up to the ring margin (16) are bit-identical
  for the op chain window KV -> compressor (ratio 1, 2, 3) -> index compressor -> indexer top-k (with and without candidate
  blocks) -> sparse attention -> de-rotation (`dsv41_chunked.rs`), and layer 2 through the real graph and weights is
  bit-identical across chunk schedules (`dsv41_layers.rs`). The official code is not (bf16 matmul batch shape: 6.9e-3 relative
  between prefill(N) and prefill(N-3)+3 decode); the engine's kernels must be, because speculative verification compares
  the verify pass with the decode it replaces. A ring without the rollback margin is detected as unsafe by the same tests.

### Indexer top-k tie-breaking

`torch.topk(sorted=False)` leaves equal scores in an unspecified order; on the 4,095-token layer-2 run its picks differ from a
stable descending sort in 1,013 of 4,095 rows, all of them equal-score swaps at the cut. The engine's rule is explicit and
shared by `IndexerTopk`, `IndexerBlocks` (block scores) and `MoeRoute` (expert scores): **the `top_k` highest scores win;
equal scores are resolved to the lowest row (block, expert) index; the winners are returned in ascending row order, unreachable
fill-ins (`-inf`, row `>= (pos0 + t + 1) / ratio`) as `-1` after the real picks.** The result then depends only on the
scores, not on the chunking, the thread count or the sort algorithm. The tests compare against an independent stable sort
(`indexer_topk_tie_break_is_lowest_index_and_deterministic`, data tied at the cut) and against the official picks on the
layer-2 fixture rows with fewer visible rows than `index_topk` (every visible row selected, `indexer_topk_matches_layer2_selection`).
Not yet done: the SET comparison against the official `topk(sorted=False)` on the 4,095-row run needs the real indexer
scores (`tools/dsv41/fixtures.py indexer4k` writes them with both selections; it needs ~35 GB and the Mia weights on
spark1, queue slot not obtained in the lane's time; the test then asserts equal sets whenever the official pick has no tie at
the cut, and for the 1,013 tie rows that both picks are optimal and differ only inside the tied value).
