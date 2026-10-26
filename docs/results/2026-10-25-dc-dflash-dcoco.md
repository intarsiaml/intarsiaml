# DFlash parallel-draft speculation and the DCOCO corrector (`--spec dflash [--dcoco]`), lane dc, 2026-10-25

Branch `lane-dc/dflash-dcoco`. Opt-in only; with the flags off nothing is loaded and `plan show` plus the 256 greedy ids
of the 8B on CUDA are bitwise main. Raw metadata for this note is in `2026-10-25-dc-dflash-dcoco/raw/`.

Status: phase 1 (scope note) accepted by the coordinator; **phase 2 done, phase 3 in progress** (section 10, 11).

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

**Block decode (one forward per round). Corrected after the MLX reference run (section 10.3).**
Let P be the number of context rows, equal to the position of the anchor token `b` (the last committed token, which the
target has not yet been fed). The query block is `[b, MASK x 14]` (`mask_token_id` 151669), **15 rows** (`dflash_config.block_size
= 15`) at positions P..P+14. Per layer: `x = embed(tokens)`; `x += o_proj(attn(input_layernorm(x)))`; `x += mlp(post_attention_layernorm(x))`;
the query rows' own K/V (same projections, qk-norm, rope at P..P+14) are appended **after** the context K/V and attended
with **no causal mask** (every query row sees all context and all block rows). Final `norm`, then `lm_head` on **all 15
rows, the anchor row included** (`sample_from_anchor: true`): row r predicts the token at position P+r+1, so row 0 (the
anchor) drafts the token right after the anchor. **15 drafts, 16 verify rows.** (The z-lab convention of taking rows 1..15
gives tau 1.04 with this checkpoint: it is the wrong layout here.) The block K/V are scratch: the next round overwrites
positions P'.. with real context K/V.

**Verify (unchanged from MTP/ngram).** Target forward on `[b, d1..d15]` (16 rows = `LOGIT_ROWS`), longest-prefix greedy
match, commit accepted + 1 bonus, roll the target KV back. The next round's context is the features of the rows the verify
produced for the committed positions (anchor row plus the accepted rows).

Geometry note: `block_size = 15` is the number of query rows and of drafted slots; `sample_from_anchor: true` means the
anchor row is sampled too; `causal: false` is the non-causal block attention. (Whether the query block has 15 or 16 rows
(one extra unused mask row) is a statistical tie on the reference, tau 3.298 vs 3.261; we use 15, which is also cheaper.)

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

## 7. Phase 1 scope

Phase 1 downloaded metadata only and wrote no code. Phase 2 results are in section 10; phase 3 results and ports: pending.

## 8. Open items (resolved empirically in phase 2/3, not guessed)

1. `sample_from_anchor: true`: resolved by the MLX reference (section 10.3): rows 0..14 including the anchor row.
2. Whether the draft's `embed_tokens` row 151669 (mask) and the rest equals Qwen3-8B's table.
3. Corrector unknowns the paper and tensor list do not pin: LayerNorm eps, GELU erf vs tanh, whether `h_i` is the draft's
   final-norm output or the pre-norm residual, the concatenation order in the 386-wide MLP input (paper lists
   `[s_p ; z_pa ; z_{p+1,b} ; l_p(a) ; l_{p+1}(b)]`, used as the default). Tests: Δ ≡ 0 reproduces DFlash (G6), and DCOCO tau
   must beat DFlash tau (G7); the variants are cheap to try in cpu-ref.
4. Whether the tap should include the pending add for layer 32 on the last-layer-fused path (it must; checked by G4).

## 9. Decisions (coordinator, 2026-10-25 after the scope note was accepted)

- "4 streams": phases 2 and 3 are gated **at batch 1 only**. The 4-client run against the serial server is report-only,
  labelled "serial server, not batched speculation", and gates nothing.
- Batched DFlash (real multi-stream speculation) is out of scope: **follow-up item, about 20 h, not started**.
- Embedding mismatch (draft vs Qwen3-8B, mask row included) would have stopped the lane before any CUDA code; it does not
  (section 10.2).
- Corrector details the paper leaves open (LayerNorm eps, GELU variant, concat order) are chosen by acceptance rate on the
  cpu-ref and logged in LOG.md.
- Stop rule: if the 16-row probe (G1) shows verify is not bitwise, report before spending the extra ~8 h.

## 10. Phase 2 results

Raw logs: `2026-10-25-dc-dflash-dcoco/raw/`. Scripts: `scripts/dc/spark1.sh` (sync, build, probe, fetch; every call is a fleetq
job), `tools/dc/tensor_cmp.py`. Probe test: `crates/intarsia-models/tests/verify16_cuda_real.rs`.

### 10.1 G1: 16-row CUDA verify probe (Qwen3-8B BF16, spark1 GB10, fleetq #5441, #5447)

`Model::verify_check` runs each id window through q single-row decode steps and through one `forward_verify`, from the
same state, and compares the f32 logits. Windows: q = 2..16 at contexts 17, 130, 513, 1024, 1900, 3000 (random ids) plus 24
windows of 16 ids that are the model's own greedy continuation (all accepted case). 1194 rows in all.

| config | windows with a logit difference | worst abs logit diff | argmax mismatches | verify16 / decode1 (len 1024, 12 passes, probe timing only) |
|---|---|---|---|---|
| default (multi-row `AttnPrefill` in verify) | 78 of 120 | 3.79 | 3 (random ids) | 79.3 / 66.9 ms = 1.19 |
| `INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=1` (one `AttnDecode` call per row) | **0** | **0** | 0 | 88.7 / 67.0 ms = 1.32 |

(`raw/g1-verify16-probe.log`, `raw/g1-verify16-probe-perrow-attn.log`.) Finding: on the dense 8B the CUDA multi-row verify
attention is **not** bitwise the decode step from about 100 context tokens on (exact up to length 17 for every q, and at 130
for every q except 16; from 513 even q = 2 differs), while the `_rows` GEMV kernels are exact (all differences vanish when
only attention is changed). So the existing CUDA `--spec ngram` on the dense 8B can in principle flip a greedy id at a
near-tie; this is a pre-existing property, the 35B MTP gate did not see it. The fix is the existing HIP mechanism
(`Backend::verify_attn_per_row`: one decode call per row, exact by construction). CUDA now answers it from an opt-in
env var (default off, so nothing changes for the existing modes); DFlash will switch it on for its own verify through a
Model-level flag, so the ~8 h row-exact-kernel contingency is not needed. Cost: verify16 is 1.32x a decode step instead of
1.19x, so the break-even tau rises from about 1.4 to about 1.55 (draft included); G5's threshold (tau >= 2.5) is unchanged.

### 10.2 Draft embedding and lm_head versus Qwen3-8B (cpu byte comparison on spark1, fleetq #5443)

`TD-Labs/Qwen3-8B-DCOCO` downloaded to `/home/nvidia/.cache/huggingface/hub/models--TD-Labs--Qwen3-8B-DCOCO/snapshots/4080a604.../`
(fleetq #5442; 4,591,038,706 bytes, sha256 `df240fec...cb12c` equals the `x-linked-etag`). `tools/dc/tensor_cmp.py` compares
tensors byte for byte: `embed_tokens.weight` and `lm_head.weight` are **IDENTICAL** to Qwen3-8B's (all 151936 rows, so the
mask row 151669 too); the controls `layers.0.self_attn.q_proj` and `norm.weight` differ, as they must (`raw/embed-cmp.log`).
Decision: the draft **shares the target's embedding and lm_head**; the draft's own copies (2.5 GB) are not loaded.

### 10.3 Independent reference: z-lab MLX DFlash on this Mac (Apple M5 Max, Qwen3-8B bf16), 16 chat-templated prompts

Harness `tools/dc/ref/` (delegated worker, reviewed here), prompts `tools/dc/prompts_dc.jsonl` (6 math, 5 code, 5 chat; Qwen3
chat template, thinking off), greedy, up to 192 tokens, results in `raw/ref/` (README.txt has commands and versions).
The corrector in `dcoco_ref.py` is written from the paper only (no code from the unlicensed repo).

| run | pooled tau (accepted + 1), 904-920 rounds |
|---|---|
| plain DFlash, 15-row block, rows 0..14 | 3.298 |
| plain DFlash, 16-row block | 3.261 |
| plain DFlash, z-lab layout (rows 1..15) | 1.041 (wrong layout for this checkpoint) |
| DCOCO default (eps 1e-5, erf GELU, order `[s;za;zb;la;lb]`, h after final norm, raw logits) | 3.360 (+1.9%) |
| DCOCO eps 1e-6 / tanh GELU / both | 3.360 (identical accepted counts) |
| DCOCO logits log-softmaxed or minus top-1 | 3.353 |
| DCOCO concat order `[za;zb;s;la;lb]` / `[s;zb;za;lb;la]` | 3.259 / 3.026 |
| DCOCO h before the final norm | 3.133 |

Findings: (1) the draft rows are 0..14 including the anchor row; the tap is the output of 0-based layer i for
i in 0, 8, 16, 24, 32 (worked first try). (2) The corrector structure is confirmed by teacher-forced accuracy of slot p+1
given the true slot p (7379 transitions): argmax 0.587, default corrector 0.748, all the wrong variants 0.54-0.58, so the
concat order, the h tap and the fuse order are pinned; eps and GELU variant are indistinguishable (0.7477 vs 0.7479) and
change drafts in 3-18 of 904 rounds without changing any accepted count. They stay at the PyTorch defaults (eps 1e-5, erf
GELU). (3) The in-loop gain of the corrector on these prompts is only +1.9% pooled (math/code prompts with tau 4-8.5
gain, chat prompts with tau 1.4-2.6 gain nothing), against +27% in the paper's ten suites (3.913 -> 4.985); the paper's
suites are math/code heavy and its drafter is trained on its own corpus, so this may be the prompt mix; phase 3 measures
it on our own runs and G7 decides. (4) Speculative output equals plain greedy for 9 of 16 prompts; the other 7 first differ
at a bf16 near-tie (0 to 2 ulps, 4 exact ties) of the MLX 16-row verify versus 1-row decode: the MLX GPU verify is not
row-exact either, which is why our own gate is the exactness in section 10.1.
Fixtures for the Rust cpu-ref tests (not committed, 10-14 MB each): `/opt/code/intarsiaml/.worktrees/dc-ref/out/fixture_{math2,code1,chat2}.npz`.

### 10.4 Phase 2 gates on CUDA (Qwen3-8B BF16, `--spec dflash`, spark1 GB10, greedy, 192 tokens)

Code: `crates/intarsia-core/src/model/dflash.rs` (draft, ingest, taps via `run_layers`), `spec.rs` (`SpecMode::Dflash`,
`DFLASH_K_MAX = 15`), `engine.rs` (`spec_dflash`), `state.rs` (`DflashState`), CLI `--spec dflash --spec-draft DIR`.
Scripts: `scripts/dc/gates.sh`, `scripts/dc/spark1.sh`. Logs in `raw/`.

| gate | result |
|---|---|
| G0 flags off vs main `5300bfde6` (`raw/g0-noop.log`) | PASS: `plan show` json (8758 B) and text (1478 B) byte-identical, 256 greedy ids identical for the short and the 4k prompt |
| G2 cpu-ref (`tests/dflash_cpu.rs`, 6 tests) | PASS: tiny Qwen3 + random draft, ids equal spec-off for k = 1, 3, 7 with `spec_proposed > 0`, rollback and prefix reuse, partial acceptance (near-identity draft), f64 host reference of the draft (rel-L2 1.1e-2, 0 argmax flips), bad geometry refused |
| draft vs MLX reference, real weights (`tests/dflash_fixture.rs`, `raw/g4-fixture-cuda.log`) | 45/45 drafts equal for reference features and for the real target prefill through the taps; block output rel-L2 0.9-3.0e-2 (bf16) |
| G3 ids == spec-off, 16 chat-templated prompts, gate off (`raw/g4-tau-16.log`), plus 6 with the default gate (`raw/g3-ids.log`) | **PASS 16/16 and 6/6**, with verify passes 24-136 and 54-167 accepted drafts per prompt (drafting really ran) |
| G4 acceptance vs the MLX reference, same 16 prompts, gate off | pooled tau (accepted + 1) **3.288 engine vs 3.298 reference** (-0.3%); per prompt 0.92-1.02 except the short-output chat prompts (1.29, 1.15: the reference run ends earlier or differs by a near-tie) |
| G5 go/no-go (tau >= 2.5, >= 1.5x spec-off at batch 1) | **PASS**: pooled decode speedup **2.12x** (gate off, 16 prompts, 192 tokens, spec-off 15.4 tok/s): math 3.3-5.1x, code 1.4-4.0x, chat 0.91-1.54x |

Reference note: the reference is the z-lab MLX DFlash on this Mac with the same checkpoint (section 10.3), not vLLM. A vLLM 0.31
arm64 image run on spark1 was not attempted (the production vLLM services share the box; the MLX comparison already pins the
layout to 0.3%). The paper's own numbers (tau 3.913 over its ten suites, 2.95x) are on B200 with math/code heavy suites; our
pooled 3.29 is on 6 math, 5 code, 5 chat prompts.

Finding: with the back-off gate off, chat prompts with tau < 1.5 (chat4: 1.40) run slower than spec-off (0.91x); the default
`SpecGate` (backs off after consecutive passes with nothing accepted) is what protects them; its effect is part of the localeval
runs in section 11.

### 10.5 `--spec ngram` on CUDA (coordinator question: is it exact on the dense 8B?)

Not reachable: `--spec ngram` is refused on CUDA at load ("--spec Ngram needs the Metal backend without tensor parallelism
(backend cuda, tp 1)", `crates/intarsia-models/src/load.rs`; 18 of 18 runs of the 16 prompts plus the short and 4k prompts
exited with that error, `raw/ngram-k4-ids.log`). So no CUDA user can hit the multi-row verify attention through ngram, and
no ngram PR is needed. Metal is not affected by the CUDA attention-kernel finding of section 10.1 (it has its own row-exact
verify contract, `docs/results/2026-10-15-metal-spec-g1.md`).

The finding of section 10.1 does matter for the other speculative mode that is allowed on CUDA, `--spec mtp`: its verify
passes take the same `AttnPrefill` multi-row call. The MTP gates on the hybrid 35B (10 attention layers of 40) were bitwise at
128 and 1024 context; a model with more attention layers or longer contexts is the case the dense 8B probe says can flip a
near-tie argmax. Not measured here (no dense MTP checkpoint on spark1 besides the 27B, which belongs to another lane). Cheap
mitigation if wanted: switch `Backend::verify_attn_per_row` on for CUDA when `--spec mtp` is on (cost measured in section 10.1:
verify16 1.32x vs 1.19x a decode step).

## 11. Phase 3: DCOCO corrector (`--spec dflash --spec-draft DIR --dcoco`)

Code: `crates/intarsia-core/src/model/dcoco.rs` (top-16 per slot on the device via `topk_rows`, `W_cand E[c]`, `W_ctx h` and the
two halves of `W_pair` as bf16 projections, host-side lazy prefix-first walk of 14 x 16 transitions). Written from the paper and the
checkpoint's tensor list; the unlicensed upstream repository was not read.

### 11.1 Correctness gates (CUDA 8B)

| gate | result |
|---|---|
| fixture: corrected drafts vs the MLX corrector (`tests/dflash_fixture.rs`, `raw/g6-fixture-dcoco-cuda.log`) | PASS 45/45 slots equal (real prefill through the taps, block rel-L2 1.4-3.0e-2); the corrected drafts differ from the plain argmax drafts, so the corrector is active |
| G6 greedy ids == spec-off, 16 prompts, gate off (`raw/g6-dcoco.log`) | **PASS 16/16** |
| G6 zero-MLP (`INTARSIA_DCOCO_ZERO=1`) reproduces plain DFlash | **PASS**: proposed/accepted counts identical to the plain DFlash run on all 4 prompts tried (math2 430/161, code1 516/154, chat2 1398/93, math5 544/154), ids identical to spec-off |

### 11.2 Does it pay? (same 16 prompts, 192 tokens, gate off, spec-off 15.4 tok/s)

| | pooled tau (accepted + 1) | pooled decode time vs plain DFlash |
|---|---|---|
| DFlash | 3.288 | 1.000 |
| DFlash + DCOCO | 3.347 (**+1.8%**) | 1.078 (**7% slower**: a draft pass costs 25 ms against 16 ms, the corrector runs partly on the host) |

Per prompt, DCOCO raises tau on 8 prompts, keeps it on 5 and lowers it on 3 (math3 7.96 -> 7.64, code5 6.33 -> 6.16, chat1 2.09 ->
2.04); it is slower than plain DFlash on 15 of 16 prompts (only math6 gains, +9%). Full table: the analysis in `raw/g6-dcoco.log`
against `raw/g4-tau-16.log`. The independent MLX run of section 10.3 gave the same +1.9%, so an implementation bug in either is unlikely.

**G7 (DCOCO tau >= 1.10x DFlash and tok/s >= 1.05x): FAIL.** Coordinator rule (>= 5% pooled served gain and no category
regressing): not met, so `--dcoco` stays an experimental opt-in flag. The paper reports tau 3.913 -> 4.985 (+27%) on ten
suites with its own drafter training mix; our 16 single-turn prompts do not reproduce that. The assumption that might be wrong
is that the checkpoint's corrector gain transfers from the paper's suites to these prompts; two independent implementations
agree on the small gain. Phase 4 (HIP and Metal ports of DCOCO) is not warranted. DFlash itself on HIP/Metal is a follow-up
(out of scope now).
