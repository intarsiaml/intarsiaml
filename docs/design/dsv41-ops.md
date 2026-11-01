# DeepSeek-V4.1 Flash: op contract, state and graph (lane 1a)

This is the contract that lanes 1b (cpu reference ops), 2 (CUDA) and the Engram lane implement against. Lane 1a appended
every op DeepSeek-V4.1 needs to `Op` once (discriminants 44..=62, `crates/intarsia-core/src/ops.rs`, never insert above
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
`AttnSparse`, `CompressPool`, `IndexerBlocks`, `IndexerTopk`) also accept a device-side position (`OpCall::pos`,
`i32 [1]`) for graph replay. `D` = 512 (head_dim), `S` = 4 (hc_mult), `H` = 64 heads, `W` = 128 (window).

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
| 60 | `MoeExpertsExl3` | x `[T,H]`, ids `[T,K]` i32, weights `[T,K]` f32, then for w1 (gate), w3 (up), w2 (down): trellis `[E,K/16,N/16,16*bits]` i16, suh `[E,K]` f16, svh `[E,N]` f16, codebook `[1]` i32 -> y `[T,H]` = sum_k weights[k] * w2_e(SiluGluClamp(w1_e x, w3_e x)) | `dsv.limit`, `top_k` | `MoE.forward` / `Expert.forward` `model.py:856-907` (the official code scales before `w2`: same math, one rounding apart) | **stub** |
| 61 | `EngramGather` | rows `[T,C,Dh]` e4m3, scales `[T,C,Dh/32]` u8 (E8M0) -> `[T, C*Dh]` = rows * 2^(scale-127) | | `ParallelEngramEmbedding.forward` `model.py:296-326` | ref |
| 62 | `EngramMerge` | h `[T,S,D]`, kv `[T,(S+1)D]` (S keys then one value), q_w `[S,D]`, k_w `[S,D]` -> `[T,S,D]` | `eps` (norm_eps) | `Engram.forward` `model.py:328-367` | ref |

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

### Ops that existing ones cover

| Need | Op used | Note |
|---|---|---|
| Linear layers of the EXL3 checkpoint (q, kv, compressor, indexer, `wo_a` slices, `wo_b`, shared expert, Engram `wkv`, head) | `MatmulExl3` | f32 output where the official code computes in f32 (compressor ratio > 1, head) |
| Router logits (f16 weight, f32 out) | `Logits` | `linear(x.float(), weight.float())` |
| Indexer `weights_proj` (f16 weight) | `Matmul` | its `softmax_scale * n_heads^-0.5` factor is the indexer ops' `scale` |
| Norms (eps 1e-20) | `RmsNorm` | |
| Routed + shared expert sum | `Add` | the official sum is in f32 before the cast; one rounding apart |
| Token embedding | `Embedding` | |
| Base checkpoint's FP4 routed experts | `MoeExperts` (MXFP4: e2m1 + E8M0 per 32) | needs `dsv.limit` honoured by its kernels (an attribute, not a new op; lane 1b) |

`MatmulFp8` does not cover the base's FP8 linears: it takes a per-tensor scale and a static activation scale, the base has
32x32 E8M0 block scales and dynamic per-32 activation quantisation. Hence `MatmulFp8Block`.

## State

| Slot | Per | Shape | Truncation |
|---|---|---|---|
| `Ring { kind: WindowKv }` | every layer (40) | `[128, 512]` act dtype | by length within `margin` (0 today: the official ring); a cut past it is refused (`Unsupported`) |
| `Ring { kind: CompressTail, planes: 2 }` | KV sources with ratio > 1 (2, 8, 14) | 2 planes `[ratio, 512]` f32 (kv, score) | same |
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
`residual streams 4`.

## Engram tier

`crates/intarsia-core/src/dsv41/engram.rs`. The host hashes token ids (`NgramHasher`, the rolling XOR hash of
`NgramHashState.forward` `engram.py:129`; layout primes `EngramLayout` `engram.py:87` derived and checked against
`engram_num_embeddings`; multipliers from numpy PCG64, generated by `tools/gen_dsv41_engram_params.py` with a hash fixture
the tests compare against). The row fetch is issued as soon as the token ids are known (`EngramTier::issue`, never blocks),
waited on before the upload (`EngramTier::wait` writes `[n, 256]` e4m3 payload and `[n, 8]` E8M0 scales, zeros for rows
outside the rank's shard, `EngramShard` = `ParallelEngramEmbedding`'s `ceil(rows / world)` split). `MemEngramTier` is the
in-memory test tier. "Scales resident" (A2) means the tier keeps the E8M0 scale array (`[rows, 8]`, about 3 GB per rank)
in host memory and gathers it by row index in `wait`, beside the payload rows read from NVMe; `EngramGather` therefore takes
per-row scales and never sees the table. Not here: the compressed token map (`build_compressed_token_map` `engram.py:17`, a tokenizer
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
| layers 0, 3-7, 9-13, 15-19, 21-23, 25-27, 29-31, 33-35, 37-39 | 39 | window (+ shared compressed KV and indices from their sources on 3-39) |
| layer 1 | 42 | + Engram (`EngramGather`, `MatmulExl3`, `EngramMerge`) |
| layers 2, 8 | 54 | + compressor ratio 2 (2 f32 `MatmulExl3`, `CompressPool`), index keys, indexer |
| layer 14 | 57 | 2 / 8 plus Engram |
| layer 20 | 53 | compressor ratio 1 (one `MatmulExl3`), index keys, `IndexerBlocks`, indexer |
| layers 24, 28, 32, 36 | 44 | indexer in layer 20's candidates (no own keys) |
| head | 3 | `HcPre`, `RmsNorm`, `MatmulExl3` (f32 logits) |

1650 steps per decode step; 622 of them `MatmulExl3`, 40 `MoeExpertsExl3` (the only stub on this checkpoint).

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
- Activation fake-quantisation is part of the math: the window KV and the attention output go through e4m3 per 32
  (ue8m0), the compressed KV through e2m1 per 16 with e4m3 scales, index keys and queries through e2m1 per 32 (ue8m0):
  `ActQdq`. The attention output is de-rotated before `wo_a`. YaRN changes only the frequencies (no mscale).
- The Mia EXL3 checkpoint has no Engram tables (`config.json` `engram_table_dir: null`, no `engram.embed` tensors): they
  have to come from the base (FP8 + E8M0, 2 x ~98 GB). Its other non-EXL3 tensors: norms (bf16), hc weights, attention
  sinks, router bias (f32), router weight and `weights_proj` (f16), embedding (bf16).
- `gate.bias_vl` is the vision-language routing bias for image-span tokens (`Gate.forward`), unused text-only.
- `MatmulFp8` does not cover the base's FP8 (block E8M0 scales, dynamic activation quantisation): new `MatmulFp8Block`.
