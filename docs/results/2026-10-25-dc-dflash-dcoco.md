# DFlash parallel-draft speculation and the DCOCO corrector (`--spec dflash [--dcoco]`), lane dc, 2026-10-25

Branch `lane-dc/dflash-dcoco`. Opt-in only; with the flags off nothing is loaded and `plan show` plus the 256 greedy ids
of the 8B on CUDA are bitwise main. Raw metadata for this note is in `2026-10-25-dc-dflash-dcoco/raw/`.

Status: **phase 1 (scope note, no code)**. Sections 6 onwards are filled in by later phases.

## 0. Decision and sources

User decision 2026-10-25 19:39 ("go dcoco"): add DFlash, then the DCOCO corrector on top, both default-off.

| source | what we took | licence |
|---|---|---|
| arXiv 2610.12570, "Contextual Correction of Parallel Drafts for Speculative Decoding" | the corrector math (section 3), selection rule, stochastic rule (App. G), reported numbers | paper |
| `TD-Labs/Qwen3-8B-DCOCO` (HF, sha 4080a604, 2026-10-14) | config and tensor names/shapes (section 2) | **Apache-2.0** (HF card) |
| github.com/two-delta/DCOCO | README only (vLLM image + `vllm serve` recipe). **No code read, none copied.** | **no licence**: GitHub API `license: null`, `LICENSE` is 404, so all rights reserved. Implementation is from the paper and the checkpoint only |
| vLLM `v1/spec_decode/dflash.py`, `model_executor/models/qwen3_dflash.py`, `v1/spec_decode/utils.py` (main) | DFlash semantics: query layout, context K/V, non-causal attention | Apache-2.0 |
| z-lab/dflash MLX reference, vendored in `/opt/code/TensorFold/src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` | the reference forward and the spec loop; usable as an independent numerical reference | MIT (Z Lab) |

## 1. DFlash in one page

Target Qwen3-8B (36 layers, H=4096, 32 q heads, 8 kv heads, head_dim 128). Draft: 5 dense Qwen3 layers (same shapes,
`intermediate_size` 12288, qk-norm, rope theta 1e6 neox), all `full_attention`, **non-causal**, plus `fc` and `hidden_norm`.

**Features from the target.** `target_layer_ids = [0, 8, 16, 24, 32]` are 0-based layer indices. The tap is the *output* of
layer i, i.e. the full residual stream after that layer (attention and MLP adds both applied), before any final norm.
(z-lab: `_LayerHook` stores the layer output; vLLM's `eagle_aux_hidden_state_layer_ids = [1, 9, 17, 25, 33]` is the same
five tensors seen as the input of layer i+1.) For every target position t, `feat_t = concat(out_0, out_8, out_16, out_24, out_32)`
in that order, 20480 values. Prefill positions and every verified position produce features.

**Context projection (shared by all 5 draft layers).**
`h_ctx = hidden_norm(fc(feat))`, `fc` is [4096, 20480] no bias, `hidden_norm` an RMSNorm.
For draft layer l: `K_ctx = rope(k_norm(k_proj_l(h_ctx)), pos)`, `V_ctx = v_proj_l(h_ctx)`, at the target's absolute
positions. Note the context rows do **not** pass through the layer's `input_layernorm`, only the shared `hidden_norm`.
These K/V persist in the draft's own KV cache (5 layers x 8 kv heads x 128 x 2 x bf16 = 20 KiB per token).

**Block decode (one forward per round).**
Let P be the number of context rows, equal to the position of the anchor token `b` (the last committed token, which the
target has not yet been fed). Query tokens are `[b, MASK x 15]` (`mask_token_id` 151669) at positions P..P+15, 16 rows.
Per layer: `x = embed(tokens)`; `x += o_proj(attn(input_layernorm(x)))`; `x += mlp(post_attention_layernorm(x))`; the
query rows' own K/V (same projections, qk-norm, rope at P..P+15) are appended **after** the context K/V and attended with
**no causal mask** (every query row sees all context and all 16 block rows). Final `norm`, then `lm_head` on rows 1..15
only (row 0, the anchor, is not read). Argmax of row i gives the draft for position P+i. **15 drafts, 16 verify rows.**
The block K/V are scratch: the next round overwrites positions P'.. with real context K/V.

**Verify (unchanged from MTP/ngram).** Target forward on `[b, d1..d15]` (16 rows = `LOGIT_ROWS`), longest-prefix greedy
match, commit accepted + 1 bonus, roll the target KV back. The next round's context is the features of the rows the verify
produced for the committed positions (anchor row plus the accepted rows).

Geometry note: the checkpoint's `dflash_config.block_size = 15` counts drafted slots (paper: B = 15 excluding the anchor;
README `num_speculative_tokens: 15`; vLLM `num_query_per_req = 1 + num_speculative_tokens`). z-lab's MLX `block_size`
includes the anchor (16). `sample_from_anchor: true` and `causal: false` are in the config; `causal: false` matches the
above, `sample_from_anchor` is **open** (section 8).

## 2. Checkpoint (metadata only; weights not downloaded yet)

`TD-Labs/Qwen3-8B-DCOCO`: one `model.safetensors` of 4,591,038,706 bytes, 70 tensors, all BF16, 2,295,515,649 parameters
(raw: `raw/TD-Labs_Qwen3-8B-DCOCO.safetensors-header.tsv`, read with two HTTP range requests of the safetensors header).
`architectures = ["DCOCODraftModel"]`, `model_type = qwen3`, `vocab_size = draft_vocab_size = 151936`,
`tie_word_embeddings = false`.

| tensor | shape | use |
|---|---|---|
| `layers.{0..4}.input_layernorm.weight`, `post_attention_layernorm.weight` | [4096] | RMSNorm, eps 1e-6 |
| `layers.{i}.self_attn.q_proj / o_proj` | [4096,4096] | |
| `layers.{i}.self_attn.k_proj / v_proj` | [1024,4096] | also applied to `h_ctx` |
| `layers.{i}.self_attn.q_norm / k_norm.weight` | [128] | per-head RMSNorm |
| `layers.{i}.mlp.gate_proj / up_proj` | [12288,4096] | SwiGLU |
| `layers.{i}.mlp.down_proj` | [4096,12288] | |
| `fc.weight` | [4096, 20480] | feature fusion |
| `hidden_norm.weight`, `norm.weight` | [4096] | context norm, final norm |
| `embed_tokens.weight` | [151936,4096] | draft's own copy (2.4 GB). DCOCO freezes it, so it is probably the target's table: **verify row 151669 (mask) and a sample of rows against Qwen3-8B in phase 2; use the draft's table if they differ** |
| `lm_head.weight` | [151936,4096] | separate from the embedding (not tied), draft's own |
| `dcoco.cand_proj.weight` | [128,4096] | W_cand |
| `dcoco.ctx_h_proj.weight` | [128,4096] | W_ctx |
| `dcoco.cand_ctx_proj.weight` | [128,256] | W_fuse |
| `dcoco.ctx_out_norm.weight / .bias` | [128] | affine LayerNorm after W_fuse |
| `dcoco.pair_proj.weight` | [128,8192] | W_pair |
| `dcoco.trans_mlp.0.weight / .bias` | [256,386] / [256] | MLP layer 1 |
| `dcoco.trans_mlp.2.weight / .bias` | [1,256] / [1] | MLP layer 2 (scalar) |

Checks: draft sizes 5 x 193.4M (layers) + 83.9M (`fc`) + 2 x 622.1M (embed, lm_head) = 2,295.3M, and the 10 `dcoco.*`
tensors sum to exactly 2,229,505 parameters, the paper's number for Qwen3-8B (App. A). `dcoco_candidate_topk = 16`,
`dcoco_emb_dim = dcoco_state_dim = 128`, `dcoco_mlp_dim = 256`.

Draft weights read per round on a 273 GB/s Spark: 5 layers + `fc` (2.1 GB) + `lm_head` (1.24 GB) = about 3.4 GB, against
16.4 GB for the BF16 8B target. The draft costs about 20% of a verify step, so a round is about 1.2x a plain decode step.

## 3. Corrector math (from the paper, section 4.2-4.3 and App. A, G)

Per drafted slot i = 1..B (B = 15): hidden state `h_i` (4096), top-K draft candidates `c_ik` and their draft logits
`l_i(k)`, K = 16. Let E be the (frozen) token embedding, `LN` a non-affine LayerNorm, `LN_gb` an affine one, r = 128.

```text
b_ik = LN(W_cand E[c_ik])                     W_cand  [r, d]        = dcoco.cand_proj
g_i  = LN(W_ctx  h_i)                         W_ctx   [r, d]        = dcoco.ctx_h_proj
z_ik = LN_gb(W_fuse [b_ik ; g_i])             W_fuse  [r, 2r]       = dcoco.cand_ctx_proj, ctx_out_norm
s_p  = W_pair [h_p ; h_{p+1}]                 W_pair  [r, 2d]       = dcoco.pair_proj        (p = 1..B-1, once per slot pair)
S_p(a,b) = l_{p+1}(b) + MLP([s_p ; z_pa ; z_{p+1,b} ; l_p(a) ; l_{p+1}(b)])      MLP: 386 -> 256, GELU, 256 -> 1
```

The MLP's output layer was zero-initialised in training, so at initialisation S_p(a,b) = l_{p+1}(b) and the selection below
reproduces DFlash's argmax. Selection (prefix-first greedy, eq. 7): `k_1 = argmax_k l_1(k)`, then
`k_{p+1} = argmax_b S_p(k_p, b)`. The draft block is `c_{p,k_p}`. Complexity: (B-1) K^2 = 3584 scores in the paper's
batched pass, O(B K) for the walk. Verify is the unchanged linear verify.

Implementation plan, no extra drafter pass:
- The walk only ever needs row `a = k_p` of S_p, so it can be run lazily: 14 steps x 16 candidates = 224 MLP evaluations
  instead of 3584. cpu-ref and the first CUDA version do exactly that.
- MLP layer 1 is linear in its concatenated input, so split `W0 = [Ws | Wa | Wb | w_la | w_lb]` (128, 128, 128, 1, 1
  columns; **column order is an assumption**, section 8): `pre = Ws s_p + Wa z_pa + Wb z_{p+1,b} + w_la l_p(a) + w_lb l_{p+1}(b) + bias`.
  The three 128->256 projections are done once per slot or per candidate (Matmul), the pair grid is a broadcast add, so the
  per-transition work is a 256-wide GELU and a dot. This reuses existing ops; no new GEMM kernel is needed.
- Greedy exactness: the corrector only changes *which* tokens are proposed. Verify is the target's own greedy argmax, so
  the committed ids are the target's regardless of the corrector's numerics (the paper's lossless claim, and the same
  argument as MTP). Stochastic (App. G): the proposal is a point mass `p_j = 1[v = y_j]`, accept with probability
  `q_j(y_j)` (min(1, q/p) with p = 1), on rejection resample from `q` restricted to `v != y_j` and renormalised. We only
  ship greedy, so this stays out of scope until the engine has stochastic spec.
- Paper numbers to compare against (Qwen3-8B, B200, vLLM, T=0, thinking off, ten-suite average, concurrency 1): vanilla
  265.6 tok/s, DFlash 783.1 (2.95x, tau 3.913), DCOCO 959.7 (3.62x, tau 4.985); gain +18-23% at batch 1, narrowing with
  concurrency. tau there = accepted drafts + 1.

## 4. Mapping onto this repo

Grounded in `crates/intarsia-core` at origin/main cf84efa43 (read in this phase; line numbers will drift).

| piece | existing seam | what is new |
|---|---|---|
| mode | `spec.rs` `SpecMode {Off,Ngram,Mtp}`, `SpecConfig::parse`, `SPEC_K_MAX = 6` | `SpecMode::Dflash`. `SPEC_K_MAX` is 6 because of the 8-row GEMV cliff; DFlash needs k up to 15 (16 verify rows = `LOGIT_ROWS`, `VERIFY_MAX_ROWS = 16`). Dflash gets its own `DFLASH_K_MAX = 15`, others unchanged |
| drafter | `model/mtp.rs` (`mtp_draft`, own `MtpState` with a pending queue and its own KV) | `model/dflash.rs`: loads the draft, owns a 5-slot draft `SequenceState`, ingest queue, block forward |
| verify, accept, rollback | `Model::forward_verify`, `engine.rs` `spec_verify / spec_accept / spec_cut`, `st.truncate` for attention-only models | **unchanged** |
| gate / back-off | `SpecGate`, `CostGate` (`spec.rs`) | the tap must keep recording on back-off (plain decode) steps; ingest is lazy (queued features), so a back-off step costs one 40 KB/token copy and the context never has holes |
| target tap | `Model::run_layers` loop (`model.rs`, two ping-pong residual buffers, pending-add, fused norms) | copy the post-layer residual (pending add included) of layers 0, 8, 16, 24, 32 into a `[rows, 20480]` staging buffer. Needed on prefill chunks, decode, verify, the captured decode graph, and the fused paths. Off unless dflash is loaded, so the default op stream is unchanged |
| attention | `AttnPrefill/AttnDecode`, contract "causal GQA" (`ops.rs`), `OpAttrs` has `window` only | new attr `causal: bool` (default true, not printed by `plan show` when true) so the default plan is byte-identical; non-causal variant for q_len = 16 over kv_len = P + 16 on cpu-ref and CUDA |
| draft layers | no generic small-transformer runner (`mtp.rs` is a bespoke MoE layer) | dense Qwen3 `Block::Attn + Block::Dense` already loads via `recipes/qwen3.toml`; reuse those blocks with weights from the draft dir, plus `fc`, `hidden_norm`, and the K/V-from-`h_ctx` projection (existing Matmul + QkNorm + Rope + KvAppend ops) |
| lm_head / top-K | `Logits`, `Argmax` ops | phase 2 needs `Argmax` on 15 rows; phase 3 needs top-16 values and ids per row (new op, cpu-ref + CUDA) |
| corrector | none | `dcoco.*` weights, host-side lazy walk first, GPU later only if profiling asks for it |
| batched | `--spec mtp` is refused with `--max-batch > 1` (LOG 2026-10-25 12:57) | same refusal for dflash; "4 streams" needs a decision (section 9) |

Costs and risks specific to the engine:
- **16-row verify exactness on CUDA.** The MTP doc's bitwise result covered at most 4 rows, and the `SPEC_K_MAX` comment
  records an 8-row GEMV cliff. The exactness gate (dflash ids == spec-off ids) is only as good as 16-row verify
  == 1-row decode, bit for bit. First step of phase 2 is a probe (below), before any drafter code.
- **Tap off-by-one is silent.** A wrong layer index or a pre-add tap still passes exactness and just lowers acceptance. Gate:
  per-slot argmax and tau against an independent reference.
- **Prefix reuse and reset.** Draft KV is per sequence; any prefix-cache hit, truncate or sequence reset must reset or
  truncate the draft KV to the same position (MTP has the same obligation in `MtpState`).
- **Chunked prefill.** Features of each prefill chunk are ingested as the chunk finishes (never stored for the whole prompt:
  8k tokens x 40 KB = 320 MB).

## 5. CLI / recipe surface and gates

```text
intarsia run|serve --model <qwen3-8b dir> --spec dflash --spec-draft <dir> [--spec-k N (1..=15, default 15)] [--dcoco]
```
- `--spec-draft` is required with `--spec dflash`; the draft's `config.json` (`dflash_config`) supplies layer ids, mask id,
  block size, top-K, corrector widths. No recipe change: the target's recipe is unchanged, the draft is validated against it
  (hidden size, vocab, layer ids < n_layers, fc width = 5 x H). `--dcoco` requires the `dcoco.*` tensors and `--spec dflash`,
  else a load error. Missing `--spec` leaves every existing path untouched.
- Env (diagnostics only): `INTARSIA_DFLASH_TRACE` (per round: anchor, drafts, accepted), `INTARSIA_DFLASH_CHECK` (dump
  per-slot argmax and top-16 for the reference comparison).
- Same refusals as MTP: `--max-batch > 1`.
- Model store for the checkpoint: `/home/nvidia/.cache/huggingface/hub/models--TD-Labs--Qwen3-8B-DCOCO` on spark1 (the HF
  cache the 8B already lives in; 278 GB free), downloaded with `fleetq run --load net,disk`.

Gates (defined before measuring; "no" on a hard gate stops the phase):

| id | gate | phase | hard |
|---|---|---|---|
| G0 | flags off: `plan show` json+text md5 and 256 greedy ids on CUDA 8B equal main | all | yes |
| G1 | 16-row CUDA verify probe: ngram k=15 (or spec-off continuation as drafts) gives ids and per-row argmax bitwise equal to 1-row decode; time verify16 vs decode1 | 2 | yes (else kernel work, see hours) |
| G2 | cpu-ref: `--spec dflash` ids == spec-off ids on the golden prompts; random-weight tiny draft equals a numpy/python re-implementation | 2 | yes |
| G3 | CUDA 8B: `--spec dflash` ids == spec-off on the golden prompts (short and 4k), acceptance counts logged | 2 | yes |
| G4 | draft correctness vs an independent reference (z-lab MLX on endor with `$HOME/models/Qwen3-8B`, MIT; or vLLM 0.31 if an arm64 image runs on spark1): per-slot argmax and top-16 on a fixed context, then tau on chat-templated math/code/chat prompts, thinking off | 2 | tau within 5% of the reference, else the tap/draft has a bug |
| G5 | go/no-go: tau (accepted+1) >= 2.5 and tok/s >= 1.5x spec-off at batch 1 on the same prompts (break-even tau is about 1.3) | 2 | yes, else stop and report |
| G6 | `--dcoco` with the MLP forced to zero reproduces DFlash's drafts exactly (unit test); ids == spec-off | 3 | yes |
| G7 | go/no-go: DCOCO tau >= 1.10 x DFlash tau and tok/s >= 1.05 x DFlash at batch 1; measured spec-off / DFlash / DFlash+DCOCO interleaved x3 via localeval at 1 and 4 clients | 3 | yes, else report, no ports |

## 6. Hours (agent hours, including fleetq waits)

| step | h |
|---|---|
| Phase 1 scope note, PR, LOG rows | 2 (this) |
| 2a G1 16-row CUDA verify probe | 1.5 |
| 2b loader, draft config validation, `SpecMode::Dflash`, CLI/serve flags, recipe checks | 3 |
| 2c target feature tap (prefill, decode, verify, captured graph, fused/pending paths) | 4 |
| 2d `causal` attr, cpu-ref non-causal, CUDA non-causal kernel | 4 |
| 2e drafter: ingest, block forward, KV truncate/reset, gate interplay | 6 |
| 2f cpu-ref tests (tiny random draft vs python reference; z-lab MLX numerics on endor) | 4 |
| 2g CUDA gates G3-G5, reference run | 5 |
| **phase 2 total** | **about 28** (+8 contingency if 16-row verify is not bitwise: row-exact kernels for M = 9..16) |
| 3 corrector loader, top-16 op (cpu-ref, CUDA), host-side lazy walk, unit tests | 8 |
| 3 localeval 3-way x 2 concurrencies x 3 interleaved, doc | 4 |
| 3 optional GPU transition kernel (only if the walk shows in the profile) | 3 |
| **phase 3 total** | **about 12-15** |
| 4 HIP port (non-causal attn, tap, top-K) / Metal port | about 16 / 16 each, only if phase 3 pays |
| batched DFlash (real multi-stream), only if wanted | 20+ |

## 7. Not done in phase 1

No weights downloaded, no code. Status of later sections: phase 2 results, phase 3 results, ports: pending.

## 8. Open items (resolved empirically in phase 2/3, not guessed)

1. `sample_from_anchor: true` in the config: vLLM's `dflash` method samples only the mask rows and the layout above matches
   the paper's 14 transitions; the flag may belong to the DCOCO vLLM overlay. Check by tau against the reference.
2. Whether the draft's `embed_tokens` row 151669 (mask) and the rest equals Qwen3-8B's table.
3. Corrector unknowns the paper and tensor list do not pin: LayerNorm eps, GELU erf vs tanh, whether `h_i` is the draft's
   final-norm output or the pre-norm residual, the concatenation order in the 386-wide MLP input (paper lists
   `[s_p ; z_pa ; z_{p+1,b} ; l_p(a) ; l_{p+1}(b)]`, used as the default). Tests: Δ ≡ 0 reproduces DFlash (G6), and DCOCO tau
   must beat DFlash tau (G7); the variants are cheap to try in cpu-ref.
4. Whether the tap should include the pending add for layer 32 on the last-layer-fused path (it must; checked by G4).

## 9. Decisions needed from the user

- "4 streams": batched speculation does not exist (MTP refuses `--max-batch > 1`). Option A: 4 concurrent clients against
  the serial server (what localeval can do today; throughput is then bounded by serial service). Option B: real batched
  DFlash, a separate 20+ h item. Default taken unless told otherwise: A.
