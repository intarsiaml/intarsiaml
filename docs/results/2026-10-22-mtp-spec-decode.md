# MTP speculative decoding for Qwen3.6-35B-A3B (`--spec mtp`), lane mtp, 2026-10-22

Branch `lane-mtp/spec-decode`. Opt-in only: `--spec mtp [--spec-k K]` loads the checkpoint's multi-token-prediction head
(recipe key `[mtp]` in `recipes/qwen3_5_moe.toml`) and uses it as the speculative drafter. Without the flag nothing under
`mtp.*` is read, and the model, plan and memory are unchanged.

## 1. Design

### 1.1 The head in the checkpoint

`config.json` (text config): `mtp_num_hidden_layers = 1` and `mtp_use_dedicated_embeddings = false`, so the head shares
the model's embedding and lm_head. In the NVFP4 checkpoint `/Users/josef/models/Qwen3.6-35B-A3B-NVFP4` (shard 3),
`hf_quant_config` lists `mtp*` as ignored, and the head's 19 tensors are all bf16:

| tensor | shape |
|---|---|
| `mtp.fc` | [2048, 4096] |
| `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.norm` | [2048] |
| `mtp.layers.0.input_layernorm`, `.post_attention_layernorm` | [2048] |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj` | [8192, 2048], [512, 2048] x2, [2048, 4096] (gated q, as the main model's) |
| `mtp.layers.0.self_attn.{q,k}_norm` | [256] |
| `mtp.layers.0.mlp.gate` | [256, 2048] |
| `mtp.layers.0.mlp.experts.gate_up_proj` | [256, 1024, 2048] (fused: 512 gate rows, then 512 up rows per expert) |
| `mtp.layers.0.mlp.experts.down_proj` | [256, 2048, 512] |
| `mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj`, `shared_expert_gate` | [512, 2048] x2, [2048, 512], [1, 2048] |

The norms are zero-centred like the main model's: weight means are 1.93, -0.73, -0.51 and -0.10 for `mtp.norm`, the two
`pre_fc` norms and `input_layernorm`, against 1.63 for the main final norm. So `norm_offset = 1` applies to them too.

### 1.2 Forward (vLLM `Qwen3NextMultiTokenPredictor`, `vllm/model_executor/models/qwen3_next_mtp.py`)

```text
x = fc(cat([pre_fc_norm_embedding(embed(token_{p+1})), pre_fc_norm_hidden(hidden_p)]))   embedding first
r = x; r += attn(input_layernorm(r))                 gated full attention at rope position p, its own KV cache
r += moe(post_attention_layernorm(r))
g = norm(r); draft_{p+2} = argmax(lm_head(g))        the main model's lm_head (NVFP4 here)
```

`hidden_p` is the main model's final normed hidden state of position p, which is what the vLLM target model returns and
the proposer feeds. Position convention, from vLLM's EAGLE/MTP proposer: head row p reads `hidden_p` and the token of
position p + 1, and uses rope position p. So the head's KV cache covers every committed position, prompt included.
Chained drafts (k >= 2) run the same layer again on its own output `g` and the previous draft, as vLLM does when
`num_speculative_tokens` exceeds `mtp_num_hidden_layers`.

Implementation (`crates/intarsia-core/src/model/mtp.rs`):

- `fc` is split at load into its two [2048, 2048] column halves. The forward runs two GEMVs plus an `Add`, so there is no
  concat buffer and no new op.
- The fused bf16 routed experts are quantised to NVFP4 at load, with ModelOpt RTN (`quant::nvfp4::quantize_rows`, one
  global scale per expert matrix, 16-element block scales). This puts them in the stacked layout of `quant::moe::Experts`,
  so the head runs on the same MoE kernels as the main model on every backend. No backend has a bf16 routed-expert
  kernel. This costs only draft quality, never output (see 1.3). On the 35B, quantisation takes about 0.45 GB of device
  memory instead of 1.6 GB.
- Router, shared expert, attention projections and `fc` stay bf16.

### 1.3 Verify and greedy identity

- Each step runs the existing verify pass, `Model::forward_verify`, over `[next, d_1..d_k]` (k + 1 rows, `exact_rows` set).
- Acceptance is the existing exact-argmax rule (`spec::accept`). The emitted tokens are always argmaxes of verify rows, so
  the drafter's quality moves only the acceptance rate.
- Output is token-identical to `--spec off` exactly when every verify row's argmax equals the one-row decode step's.
  That holds where each op of the verify pass has a `rows_exact` kernel. The cpu backend loops over rows, so it holds there.
- Metal registers `rows_exact` kernels only for bf16 GEMV / pair / logits and split-KV attention. CUDA registers none.
  So on the 35B (FP8 projections, NVFP4 experts and head, GatedDeltaNet), `--spec mtp` on a GPU emits the verify path's
  argmax. Where that differs from the M = 1 path, the output would differ. On 12 (prompt, k) cases, 3,072 tokens, it
  never did; the row-level mismatch rate is not yet measured (3.1).
  The logit-level comparison was deferred.
- The GatedDeltaNet conv and delta rule run one row at a time inside a verify pass (below). That keeps those two ops
  bitwise equal to decode on every backend.

### 1.4 Rolling back recurrent state

A GatedDeltaNet slot is updated in place (conv state [C, 3] f32 plus delta state [32, 128, 128] f32). On the 35B that is
30 layers x about 2.1 MB = 63 MB per sequence. A rejected draft cannot be truncated away the way KV rows can. Options:

| option | per verify pass | on rejection | notes |
|---|---|---|---|
| A. snapshot before verify, restore, re-run the accepted rows through the whole model | copy 63 MB | restore 63 MB + a full forward of j + 1 rows | almost doubles the cost of every rejected pass |
| B. snapshot before verify, restore, re-run only conv + delta rule on saved per-layer inputs | copy 63 MB + save qkv/a/b rows | restore 63 MB + 30 x (conv + delta) over j + 1 rows | more ops and buffers than C |
| **C. (chosen) per-row states: conv + delta one row per call, slot copied after each of the first k rows** | k x 63 MB device copies | copy one saved row state (63 MB) | no new kernels; t = 1 calls are the decode launches, so bitwise to decode |

C is `SpecRows` in `crates/intarsia-core/src/state.rs` and the verify branch of `gdn_block`. Rollback is
`SequenceState::spec_rollback(keep)`. It is also used by `Seq::finish` when a stop id, stop string or `max_tokens` lands
inside an accepted run, so a finished request leaves the state where sequential decode leaves it, and prefix reuse
stays exact. The per-row copies cost k x 63 MB of device traffic per pass, about 0.5 ms per row at Spark's ~270 GB/s (63 MB read plus 63 MB written; an earlier estimate of 0.25 ms counted the read only) (read
plus write). Measured cost is in section 3.

The head's own KV cache is attention-only and rolls back by truncation. Its queue of pending hidden rows
(`MtpState::pending`) is cut to the kept length. Recurrent prefix-reuse snapshots also record the head's position and
pending row, so a request that reuses a snapshotted prefix drafts from the right state.

### 1.5 Scheduling, k, exclusions

- **Per step:** the head catches up over the queued rows. After a pass that accepted j drafts this is j + 1 rows in one
  call, and its last row gives draft 1. Then k - 1 chained one-row head passes run, then the verify pass.
  After a plain step there is one queued row; after a prefill chunk, all but the last row are caught up at once.
- **Engine:** `LocalEngine` only (B = 1). The batching engine refuses `--spec mtp`.
- **Requests:** greedy only, no constraint (J1), no logit trace. Spec stays off inside the `min_tokens` window and never
  drafts past `max_tokens`.
- **Exclusions:** token feedback (T1) stays excluded: `feedback_ok` requires `!spec_on`. Tensor parallelism is refused.
- **Gate:** the existing drafter gate (`spec::GateParams`, back-off after 6 empty passes) applies unchanged.
- **k:** default `--spec-k` is the n-gram default (4). Section 3 sweeps k = 1..3 and picks one.

## 2. CPU reference and tests

`crates/intarsia-models/tests/mtp.rs` uses the tiny hybrid MoE of `tp_hybrid.rs` plus a random head in the 35B layout.
The tests:

- `mtp_head_matches_a_host_reference`: the head's logits (f32 activations) against an f64 host implementation of the
  vLLM head, written independently in the test, with the same NVFP4 round trip on the routed experts. Max relative logit
  difference is under 2e-4, and the argmax matches on every row. HF transformers has no MTP forward (it drops `mtp.*`),
  and vLLM is not installed on endor, so this is the reference that was obtainable.
- `mtp_is_not_loaded_without_the_flag`: no head is loaded without the flag. A missing head is an error only with it.
  Plain decode with the head loaded is identical.
- `mtp_head_drafts_keep_greedy_output_identical_over_256_tokens`: the head's own drafts at k = 1, 2, 3 on 3 prompts, 256
  greedy tokens, identical to spec off. With random weights the head's drafts are rejected, so this exercises the
  reject-all rollback.
- `scripted_drafts_hit_every_acceptance_pattern_and_stay_identical`: the drafts are the true continuation with an error
  at a pseudo-random offset, or none. The head still runs, so its state machinery is exercised. This covers full, partial
  and zero acceptance through the GatedDeltaNet rollback at k = 1..3. It also covers `max_tokens` and stop ids inside a
  run, and follow-up requests that reuse the prefix. All are identical to spec off.
  - k = 1: 677 passes, 323 of 677 drafts accepted.
  - k = 2: 549 passes, 407 of 1088 accepted.
  - k = 3: 459 passes, 551 of 1344 accepted.

These prove the engine logic (drafting, verify, rollback, finish, reuse), not a GPU backend's numerics.

## 3. CUDA on spark1 (35B NVFP4)

### 3.0 Preliminary: shared (non-exclusive) slot, before the kernel/driver upgrade

fleetq #4296, commit a6d5ad19, before spark1's kernel 6.17.0-1032 / driver 580.178.04 upgrade (#4301). `intarsia generate`, a
chat prompt plus `--no-think`, 256 greedy tokens with `--ignore-eos`, four prompts. The slot was shared and vLLM was up, so
these are **not** the gate numbers. The exclusive localeval window (3.2) replaces them.
Raw: `2026-10-22-mtp-spec-decode/raw/identity-nonexcl/`. Script: `scripts/identity.sh`.

| prompt | off tok/s | k=1 tok/s (acc.) | k=2 tok/s (acc.) | k=3 tok/s (acc.) |
|---|---|---|---|---|
| p0 code | 82.6 | 95.9 (96.2%) | 110.5 (92.7%) | 103.0 (85.1%) |
| p1 prose | 83.1 | 89.9 (81.4%) | 94.5 (70.1%) | 84.1 (53.6%) |
| p2 list | 84.1 | 93.4 (88.9%) | 101.1 (76.6%) | 93.9 (62.6%) |
| p3 arithmetic | 83.9 | 97.3 (97.7%) | 110.7 (91.1%) | 119.3 (90.7%) |

**Token identity.** All 12 MTP runs (4 prompts x k = 1..3) produce the same 256 greedy tokens as `--spec off`.
This is empirical evidence, not a structural property (3.1). Acceptance is per draft token; tokens per pass = 1 + accepted / passes.

### 3.0b Identity re-run after the driver upgrade (shared slot)

fleetq #4349, commit 8e1fa857, CUDA build 56239424 bytes, GB10 on driver 580.178.04 / kernel 6.17.0-1032. Same script
and prompts as 3.0. All 12 MTP runs produce the same 256 greedy tokens as `--spec off` (empirical, 3.1).
Raw: `raw/identity/`.

| prompt | off tok/s | k=1 tok/s (acc.) | k=2 tok/s (acc.) | k=3 tok/s (acc.) |
|---|---|---|---|---|
| p0 code | 85.8 | 100.5 (96.2%) | 115.1 (92.7%) | 117.5 (85.1%) |
| p1 prose | 86.1 | 92.5 (81.4%) | 96.2 (70.1%) | 85.7 (53.6%) |
| p2 list | 86.0 | 96.2 (88.9%) | 101.4 (76.6%) | 96.1 (62.6%) |
| p3 arithmetic | 85.0 | 100.4 (97.7%) | 112.9 (91.1%) | 121.7 (90.7%) |

Acceptance rates are identical to 3.0 to the decimal. Drafts and verify argmaxes did not change with the driver.

**spark1 crashed during this job.** The first `INTARSIA_MTP_VERIFY_CHECK` run started at 22:59:13, and its stderr stayed
empty. The box then rebooted without a clean shutdown and was back up at 23:01:52. The journal of that boot ends at
22:49:14, so its last ten minutes were never flushed. spark1 went down without a shutdown twice more: the next boot's
journal ends at 23:07:32 (back up 23:23:31), and the one after at 23:38:24 (back up 23:45:11). None of my jobs had a
process on the box at those times. The cause is not established: the new driver, the diagnostic run, or both.

### 3.1 What the identity runs show, and what they don't (logit-level check deferred)

**Claim.** `--spec mtp` on CUDA is *self-consistent greedy over the verify rows*: every emitted token is the argmax of a
verify row. It was *empirically identical* to `--spec off` on 12 (prompt, k) cases, 3,072 emitted tokens. The
*row-level argmax mismatch rate against the M = 1 path is not yet measured.*

- **Not structural.** CUDA has no row-exact verify kernels. Only the GatedDeltaNet conv and delta rule run as t = 1
  launches inside a verify pass. Attention runs `attn_prefill` (split-KV) over q rows, the MoE runs its grouped multi-row
  path, and the projections run GEMM where decode runs GEMV. A different accumulation order can flip a near-tie argmax,
  after which the output diverges from `--spec off`. It stays valid greedy output over the verify path.
- **What the evidence is.** 3.0 and 3.0b ran the same 12 cases (4 short prompts x k = 1..3, 256 tokens) twice and gave
  byte-identical acceptance counts. So "24/24" is 12 independent cases run twice: 3,072 emitted tokens per run set, 2,496
  verify passes over both sets. Short low-entropy prompts and 256 tokens cannot bound a near-tie flip rate. A 4k-token
  prose generation could diverge.
- **Deferred.** `INTARSIA_MTP_VERIFY_CHECK=1` measures it: each verify pass's rows are re-run as single-row decode steps
  from the same state, with max |dlogit| and argmax agreement per row. Its first run was in progress when spark1 crashed
  (3.0b), so the coordinator deferred it and it was not run on spark2. To run it later: >= 2k tokens on a high-entropy
  prompt; report the max |dlogit| distribution, the argmax mismatch count, and the count of rows with a top-2 margin
  under 1e-2.
- **Diagnostic caveats.** It takes a recurrent snapshot per pass, which evicts prompt-chunk snapshots, so prefix reuse
  differs while it is on. Its output per row is `max|dlogit|:argmax-agrees:top-2 margin:decides` (single-row logits; `decides` = every earlier
  draft was accepted, so the row chose an emitted token).

### 3.4 What a pass costs, and when a draft pays

From the committed logs. Off-arm decode is 2963-2999 ms for 256 steps, about **11.6 ms per step**. 256 of 257 steps are
graph replays (`raw/identity/p*-off.err`).

MTP ms per pass is decode time / verify passes in `raw/identity/p*-mtp*.err` (fleetq #4349, spark1, post-upgrade). It is
the same across all 4 prompts to within 0.5 ms:

| k | ms per pass | from the spark2 trace: draft (head) / verify + rollback, median ms |
|---|---|---|
| 1 | 19.5-19.7 | 1.83 / 17.99 |
| 2 | 24.9-25.1 | 3.60 / 21.41 |
| 3 | 29.8-30.4 | 5.36 / 24.84 |

The trace column is `raw/phases/trace-mtp*.err`, `INTARSIA_SPEC_TRACE=1`, spark2, fleetq #4371. Each phase ends in a
token download, so host time is device time there.

- **Fit:** pass(k) ≈ **14.3 + 5.3·k ms**. The Fable review derived 14.4 + 5.0·k from the same logs.
- **Fixed part.** About 2.7 ms over a replayed decode step: the verify pass and the head run eagerly (`graph replays=2`
  per run).
- **Per draft:** about 1.8 ms of head pass (one gated-attention + MoE layer, the full-vocabulary lm_head, the argmax
  download), plus about 3.4 ms for the extra verify row. The extra row costs its share of the MoE expert reads (up to 8
  new experts x 40 layers), 30 per-row GatedDeltaNet state copies (126 MB of traffic), and 30 more conv/delta launches.
- **Break-even.** Drafting at depth i pays only if P(accept_i | accepted all earlier) > 5.3 / 11.6 ≈ **0.43-0.46**. That
  is why k = 3 wins on the code and arithmetic prompts (85-91% total acceptance) and loses on prose (54-63%).
- **The localeval window must confirm this** with the 128/1k/4k numbers (3.2). The default k = 2 is the best fixed choice
  under this cost. Idea C (section 4) makes the fixed choice moot.

### 3.2 localeval window: decode tok/s at 128 / 1k / 4k, k sweep (spark2, quiet slot, retro down)

- **Setup.** fleetq #4399 on spark2, commit 0211d16f. `retro` vLLM was stopped for the window (00:42:05-01:03:38, 21
  min, `raw/downtime.log`), so the slot was quiet. `intarsia serve --backend cuda --max-seq 20480 --mem-budget 30
  --profile-every 0 [--spec mtp --spec-k K]` and `localeval speed --sizes 128,1024,4096 --gen 128 --reps 3 --thinking
  off`.
- **Order.** Three passes, arms interleaved: p1 off, 1, 2, 3; p2 3, 2, 1, off; p3 2, off, 3, 1. Each cell is the median
  of the three passes' medians, with the range across passes.
- **Lanes.** The MTP arms are labelled `spec-mtp-k<K>-p<N>` (composer: spec lane). The off arm is plain decode.
- **Raw.** `raw/speed/localeval-*.txt`, `metrics-*.txt` (server spec counters) and `serve-*.log`.

| arm | 128 | 1k | 4k | acceptance (all sizes) | tokens / pass |
|---|---|---|---|---|---|
| off | 84.3 (84.1-84.4) | 83.8 (83.6-83.8) | 81.2 (81.2-81.3) | - | 1 |
| k=1 | 86.0 (85.4-86.1) | 84.7 (83.3-86.5) | 87.2 (87.0-88.2) | 76.9% | 1.77 |
| k=2 | 86.3 (85.2-86.7) | 87.8 (82.9-89.3) | 88.5 (83.3-93.3) | 62.8% | 2.25 |
| k=3 | 75.3 (74.7-79.8) | 80.4 (80.3-82.3) | 87.0 (77.9-90.8) | 51.6% | 2.54 |

- **Gain at k=2:** +2.4% at 128, +4.8% at 1k, +9.0% at 4k against off.
- **Noise.** Under the composer's 5% decode floor at 128 and 1k, with overlapping ranges at 1k: a **tie** at 128 and 1k.
  A small win at 4k, where plain decode slows with context but a verify pass amortises its attention over 2.25 tokens.
- **Why so much less than the code / arithmetic prompts of 3.0b** (+16-33% at k=2): localeval's speed prompts accept far
  less. 62.8% of drafts at k=2 against 70-93%, so 2.25 tokens per pass against 2.4-2.85. At pass(2) ≈ 25 ms vs 11.6 ms
  per plain step (3.4), break-even is 2.16 tokens per pass. 2.25 predicts about +4%, as measured.
- **The gain depends on content.** It needs acceptance above about 0.45 per draft position. Ideas A and C (section 4)
  lower that threshold.
- **localeval run ids** (`/opt/code/deepeval/results/`, all also logged to unanimis):
  - off: 20261023-004820-adhoc-mtp-lane-plain-off-p1, -005554-…-p2, -005758-…-p3
  - k=1: 20261023-004928-adhoc-spec-mtp-k1-p1, -005457-…-p2, -010007-…-p3
  - k=2: 20261023-005029-adhoc-spec-mtp-k2-p1, -005351-…-p2, -005702-…-p3
  - k=3: 20261023-005136-adhoc-spec-mtp-k3-p1, -005244-…-p2, -005900-…-p3
- **k default.** k = 2 is the best fixed choice. It is never worse than k = 1 by more than the noise, wins at 1k and 4k,
  and k = 3 loses at 128 and 1k. `MTP_K_DEFAULT = 2` stays.
- **First attempt.** A first window (#4371 at ef518fd7, 10 min of retro downtime) aborted on its first MTP arm. The
  server refused the head's `Add` with "no kernel … [admitted kernels only]", because the plan dry-run never ran the head
  (review finding 5). The window script's guard stopped it before any wrong number was recorded. Fixed in 56e453b3.

### 3.3 No-op: Qwen3-8B and 35B with `--spec off`, `plan show` + 256 greedy ids vs main a92267b0 (pending)

## 4. Next (not in this PR)

Ideas from the Fable review of #110 (`scratchpad/reviews/fable-pr110-mtp-review.md`), with its estimates. The coordinator
will open a follow-up lane for A + C.

| | Idea | Expected | Notes |
|---|---|---|---|
| A | Graph-capture the verify pass (per q in 2..=k+1) and the one-row head pass | ~+20% tok/s (pass(k=2) 25 -> ~20 ms) | Removes the eager floor. Needs `spec_rows` preallocated (done), a device-side head position, fixed ids/hidden buffers |
| B | Zero-copy rollback: out-of-place recurrent state in verify rows (`state_in` / `state_out`) | ~5% at k=2, ~8% at k=3 | Same arithmetic, so bitwise. Accept-all then costs nothing, a rejection one 63 MB copy. Each backend's GDN kernels gain a pointer |
| C | Adaptive depth from a per-position acceptance EMA, or the draft's top-2 logit margin | +5-10% average | Threshold = marginal cost / step (0.43 now, ~0.2 after A + B). Removes the need for a fixed `MTP_K_DEFAULT` |
| D | Draft-only reduced-vocabulary lm_head (top ~32k tokens) | ~3-4% at k=2 | The verify pass keeps the full head, so output is unchanged. Check the acceptance loss on code identifiers |
| E | Tree / multi-candidate drafting | not worth it here | Each extra verify row's cost is dominated by MoE expert reads |

Also open:
- The logit-level verify-row check (3.1).
- One GPU multi-turn run (a stop id inside a run, then prefix reuse after a rollback) to cover the real-buffer
  `truncate` / snapshot path that only the CPU tests exercise.
- A Metal / HIP run.

