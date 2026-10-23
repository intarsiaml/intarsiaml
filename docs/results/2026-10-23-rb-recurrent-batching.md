# Continuous batching for models with recurrent state (Qwen3.6-35B-A3B): design and CPU correctness prototype, lane rb, 2026-10-23

**Status: design plus an opt-in CPU prototype. Draft PR, do not merge.** Branch `lane-rb/recurrent-batching`, base `origin/main` a312a794. No GPU was used and no
GPU kernel was written. Every number below is one of: **measured** (cited to a file or run id), **derived** (arithmetic shown, from measured inputs), or
**estimate** (a judgement; the assumption is named). Section 12 is the prototype's status and test output.

## 0. Summary

**The gap.** The composer (same quant, Spark GB10, Qwen3.6-35B-A3B NVFP4) has IntarsiaML at 74.2 tok/s at 1 stream, 75.2 at 4 and 75.2 at 16
(`20261022-050829-adhoc-fx3-35b-intarsia-p1-conc`): it serves one sequence at a time. vLLM is 105.5 / 105.8 at 4 / 16 (+40.3% / +40.7%; both rows carry the
`drop:gsm8k` quality flag, peer-not-production, and the production vLLM server runs `--max-num-seqs 2`, so it plateaus at about 105 from 4 streams on:
composer catalog entry `sglang-continuous-batching-uncapped-running-requests`, quoting `2026-10-22-fx3-spark-standings.md`). SGLang, run with
`--max-running-requests 16`, is 157.5 / 219.9 (quality ok). SGLang is the honest target; on the 8B a vLLM given `--max-num-seqs 16` reached 156 tok/s at 16 streams against SGLang's 91, so a properly
configured vLLM would not stop at 105 either.

**Root cause is wider than one line.** `engine/batch.rs:374` is the first of seven places that refuse or silently break hybrid batching (section 1). The one that
decides the speed-up is not a refusal at all: **the speed-up is bounded by what a batched step can share, and on this model that is only the dense weights** (section 2).

**The arithmetic (section 2).** One decode step reads about 2.4 GB at N = 1: 1.69 GB of dense weights that N sequences share (GDN and attention projections,
shared expert, router, lm_head), plus 0.57 GB of routed experts (8 of 256 per layer), plus 0.13 GB of recurrent-state read-and-write and the KV that each
sequence owns. Experts and state do not amortise. The bandwidth ceiling for N streams, anchored on the measured 12.4 ms graph step and the 74.2 tok/s serve rate,
is **x2.2 at 4 streams (162-165 tok/s) and x3.1-3.6 at 16 (229-265 tok/s)**. SGLang's measured 157.5 / 219.9 sits at 95% / 83% of that ceiling, which is the validation
that the model is not off by a factor.

**Expected result of the plan (estimate, aggregate tok/s on the composer's concurrent metric):**

| streams | now | ceiling (section 2) | expected after stages S1-S5 | vLLM (flagged) | SGLang |
|---|---|---|---|---|---|
| 1 | 74.2 | 74.2 | 74.2 (B=1 not slower, gate) | 69.7 | 68.1 |
| 2 | ~74 (serial) | 116 | 95-105 | 105.9 | n/a |
| 4 | 75.2 | 162-165 | **130-145 (x1.75-1.95; +24..+37% over vLLM)** | 105.5 | 157.5 |
| 8 | n/a | 201-214 | 160-180 | n/a | n/a |
| 16 | 75.2 | 229 (265 with expert grouping) | **185-205 (x2.5-2.75; +75..+94% over vLLM)**, 210-235 after S6 | 105.8 | 219.9 |

Matching vLLM at 4 streams needs only x1.40, so the first stage that shares the dense weights across rows (S3) already closes the stated gap at 4 streams.

**Staged plan (section 11).** S1 engine plumbing for hybrids (1.5 d, prototyped here on CPU), S2 CUDA bring-up that is exact by construction, no speed-up (4 d),
S3 multi-row dense GEMVs and the fused decode ops (5 d, first speed-up, the gap at 4 streams closes), S4 N-token MoE decode, batched GDN, batched attention tables (5 d),
S5 CUDA graphs per batch bucket (3 d), S6 expert grouping, HIP, Metal, mixed fold, prefix store and MTP at B > 1 (separate lanes). Critical path S1-S5: about 18
working days of one engineer or agent lane (estimate).

**Open questions, most important first (section 13):** (1) exactness policy on CUDA: the CUDA backend registers no `rows_exact` kernels, so a bitwise-to-solo
batch needs new kernels; is a measured-but-not-bitwise mode acceptable if the kernels turn out not to be row-exact? (2) raise the batch cap from 8 to 16 for CUDA;
(3) when `--max-batch > 1` is on, the solo step loses its fused decode ops unless they get multi-row kernels; (4) MTP and the prefix store stay B = 1 until after S5.

## 1. What refuses or breaks today (not only `batch.rs:374`)

| # | Where (origin/main a312a794) | What |
|---|---|---|
| 1 | `crates/intarsia-core/src/engine/batch.rs:374` | `BatchedEngine::new` returns `Unsupported("continuous batching on a model with recurrent state")` |
| 2 | `crates/intarsia-core/src/model/batch.rs:45-54` | `enable_batch` accepts dense attention-only models: `Block::Gdn`, `Block::Moe` and gated attention are rejected |
| 3 | `crates/intarsia-core/src/model/batch.rs:100-107` | `check_batch_row` rejects any state with `has_recurrent()` |
| 4 | `crates/intarsia-core/src/model.rs:1554-1558` and `:1569-1573` | `run_layers` with `Seqs::Batch` returns `Unsupported` at a GDN block and at a MoE block |
| 5 | `crates/intarsia-core/src/model/batch.rs:216` | `attn_block_batch` asserts there is no gate and passes `None` to `attn_out_proj`; the 10 attention layers of the 35B are gated, so the output gate would be dropped silently |
| 6 | `crates/intarsia-core/src/model/batch.rs:22`, `scheduler.rs:47`, `model.rs:39` | `BATCH_MAX_ROWS` = `MAX_BATCH_CAP` = 8 (an operator decision for Metal's exact kernels); the 16-stream composer row needs 16 (`LOGIT_ROWS` is already 16) |
| 7 | `crates/intarsia-core/src/model/batch.rs:56-61` | CUDA has no `rows_exact` kernels (`batch_capable()` is false), so `enable_batch` fails on CUDA before it gets to the model shape |

Four more defects are in the plumbing and would fail or diverge **after** lifting 1-4 (all found by reading, the prototype fixes them for the CPU path):

* **Prefix reuse fails the request.** `Scheduler::pick_slot` (`scheduler.rs:683-696`) uses `common_prefix(..).min(plen-1)` and then `truncate(slot, cached)`. A recurrent
  state can only return to its current length, a snapshot or 0 (`state.rs:191-197, 209-249`); `LocalEngine` goes through `restorable_len` (`engine.rs:899-904`), the
  scheduler does not. Result today: `Unsupported`, request failed, for any reuse that is not a snapshot length.
* **The prefill schedule differs from solo.** `SlotModel::prefill_chunk` (`engine/batch.rs:123-129`) calls `Model::forward` per scheduler chunk. `Seq::prefill_step`
  (`engine.rs:1122-1153`) additionally snapshots after every non-final chunk and runs the last prompt token alone after a snapshot at `plen-1`. Without that the solo
  reference is not the batched run's chunk schedule (a 1-row decode kernel in one, a bigger chunk in the other) and no snapshot exists for later reuse.
* **Speculation flag.** `BatchedEngine::submit` calls `Seq::build(.., false)` (`engine/batch.rs:436`), so `recurrent` is always false and an n-gram `--spec` request
  would draft on a recurrent state (`engine.rs:512-513` refuses that only in `LocalEngine`).
* **Memory plan.** `enable_batch` sizes `kv_cap_tokens` from the KV budget alone (`model/batch.rs:62-84`); recurrent slots (63 MB each plus snapshots) are not in it.

## 2. Roofline: what N sequences can share

Decode on this model is weight-bandwidth bound: nsys of one graph-replayed 35B step is 12 406 us for 724 launches with 98.6 us idle
(`docs/results/2026-10-14-cuda-decode-fusion.md:73-91`), and every group runs at 185-220 GB/s of GB10's 273 GB/s peak (fp8 GEMV: 1.28 GB in 5.79 ms = 221 GB/s; GDN delta
rule: 126 MB read+write in 0.673 ms = 187 GB/s). Bytes per step, from the checkpoint headers (`2026-10-23-rb-recurrent-batching/wbytes.txt`; script in the same directory):

| group | bytes per decode token | shared by N rows? |
|---|---|---|
| GDN projections in_qkv/in_z/out/a/b/conv (30 layers, fp8) | 1 016.5 MB | yes (one weight read) |
| attention q/k/v/o (10 layers, fp8) | 272.7 MB | yes |
| shared expert (40 layers, nvfp4), router (bf16) | 70.8 + 41.9 MB | yes |
| lm_head (nvfp4 + fp8 scales) | 286.1 MB | yes |
| **dense subtotal** | **1 688 MB** | **once per step** |
| routed experts, 8 of 256 per layer: 1.769 MB per expert x 8 x 40 | 566 MB | only the experts two rows both pick |
| GDN state read + write: 2 x 30 x (98 304 + 2 097 152) B | 131.7 MB per sequence | no |
| KV read, 10 layers x 2 x 2 x 256 x 2 B = 20 KiB per token | 21 MB per sequence at 1k context | no |

Step bytes at N = 1: 2 407 MB; at the measured 12.406 ms that is an effective **194 GB/s (71% of peak)**. The 74.2 tok/s serve rate adds 1.07 ms of host time per step
(13.48 ms per token against 12.41 ms of GPU), assumed to grow 0.05 ms per extra row (sampling, streaming).

Routed experts per layer touched by N rows, uniform routing: `256 x (1 - (248/256)^N)` = 8.0 / 15.8 / 30.5 / 57.4 / 102.0 at N = 1 / 2 / 4 / 8 / 16. A real router is skewed, so
this overstates distinct experts (understates sharing). Two expert modes are modelled: **grouped** (each distinct expert is read once) and **per-pair** (every (row, expert)
pair reads its expert, which is what the existing decode kernel does if it is simply given N x 8 pairs). Aggregate tok/s, 1k context, anchored on the 194 GB/s and 74.2 tok/s
(`2026-10-23-rb-recurrent-batching/roofline.py`, output in `roofline.txt`):

| N | step bytes, grouped (MB) | agg tok/s, grouped | x over N=1 | step bytes, per-pair (MB) | agg tok/s, per-pair | x over N=1 |
|---|---|---|---|---|---|---|
| 1 | 2 407 | 74.2 | 1.00 | 2 407 | 74.2 | 1.00 |
| 2 | 3 108 | 116.7 | 1.57 | 3 126 | 116.1 | 1.56 |
| 4 | 4 460 | 165.2 | 2.23 | 4 564 | 161.7 | 2.18 |
| 8 | 6 974 | 214.1 | 2.89 | 7 439 | 201.2 | 2.71 |
| 16 | 11 348 | 265.3 | 3.58 | 13 191 | 229.2 | 3.09 |

At 4096 context the same model gives 156.8 (grouped) / 153.6 (per-pair) at N = 4 and 244 / 213 at N = 16.

Four consequences that shape the design:

1. **Share the dense weights first; grouping experts is worth +2% at 4 streams and +16% at 16.** With uniform routing 4 rows pick 30.5 distinct experts of 32 pairs, so
   per-pair is within 2% of grouped at 4 streams. An expert-grouped batched MoE kernel is stage S6, not the first thing to build.
2. **GDN state traffic is the second largest per-row cost and cannot be shared**: 2.1 GB per step at N = 16, 18% of the step. Any scheme that moves the state
   (gather/scatter, section 4) adds to it. The state must be read and written in place.
3. **A batched GDN launch buys launch count, not bandwidth.** The existing single-sequence delta-rule kernel already runs at 187 GB/s, the step's effective bandwidth, so
   N launches cost the same device time as one launch over N states (section 4). The device-side state table that graph capture needs is the real requirement.
4. **Validation.** SGLang's measured 157.5 (x2.31 over its own 1 stream) and 219.9 (x3.23) against this model's grouped ceiling 165.2 (x2.23) and 265.3 (x3.58): 95% and 83%.
   vLLM's flat 105.5 / 105.8 from 2 to 16 streams is its `--max-num-seqs 2` production setting, not a bandwidth limit, and carries a quality flag; do not size against it beyond the +40% headline.

Not modelled and called out as risks: (a) multi-row GEMV compute at 16 rows (Metal measured the bf16 multi-row GEMV at 1.2x a 1-row launch at 8 rows and 2.6x at
16, `docs/research/continuous-batching-design.md` section 4; the CUDA fp8/nvfp4 GEMVs have not been measured with more than 1 row); (b) per-row attention launches (3 per
row-layer, 30 per row); (c) host time per step beyond 0.05 ms per row.

## 3. State layout and the scheduler

**Layout (unchanged from the Metal design, `docs/research/continuous-batching-design.md` section 3.2 option A): N independent `SequenceState`s, one per slot.** A slot
holds that sequence's KV (`AttnKvState`, contiguous), its recurrent slots (`RecurrentState`, one buffer per GDN layer), its token list and up to
`RECURRENT_SNAPSHOTS` = 4 snapshots (`state.rs:15`). `SlotModel` already owns `Vec<SequenceState>` (`engine/batch.rs:78-90`), `forward_batch` already takes
`&mut [&mut SequenceState]`, and the recurrent buffers are allocated once per state (`SequenceState::new`, `state.rs:92-119`), so **slot addresses are fixed for the engine's lifetime**.
That fact is what makes a device-side pointer table possible without an arena (section 4).

**Memory at 63 MB per sequence** (derived: 30 layers x (98 304 B conv + 2 097 152 B delta state) = 65 863 680 B = 62.8 MiB; `gdn_block.rs:18-24`):

| N | live recurrent state | + 4 snapshots per slot (worst case) | KV at 8k tokens per slot (20 KiB/token) | per-step state traffic (read + write) |
|---|---|---|---|---|
| 2 | 126 MiB | 628 MiB | 320 MiB | 263 MB |
| 4 | 251 MiB | 1.23 GiB | 640 MiB | 527 MB |
| 8 | 502 MiB | 2.45 GiB | 1.25 GiB | 1.05 GB |
| 16 | 1 005 MiB | 4.9 GiB | 2.5 GiB | 2.1 GB |

Against about 23 GB of weights on a 128 GB machine none of this is binding. Two things still need code: the memory plan must count the recurrent term
(`enable_batch` today divides the KV budget by `N x bytes_per_token` only), and snapshots must be bounded under batching (see section 7: `RECURRENT_SNAPSHOTS` per slot becomes a
configuration, 2 at N >= 8, until the prefix store takes over snapshot storage). With `--spec mtp` each sequence additionally holds `k` `SpecRows` buffers of 63 MB (PR #110, `state.rs`
`SpecRows`), which is one reason MTP stays B = 1 (section 8).

**Scheduler.** `scheduler::BatchEngine` needs no structural change: it already has slots, FIFO admission, decode-first iterations, a separate prefill sub-pass and idle-slot
prefix reuse (`scheduler.rs:699-918`). Three hooks on the `BatchModel` trait are new for recurrent models:

* `restorable_len(slot, want) -> usize` (default `want`): `pick_slot` asks the model for the length a slot can actually return to, so admission picks the slot with the longest *restorable* common prefix.
* prefill for recurrent slots mirrors `Seq::prefill_step`: plain chunks of the scheduler's chunk size, a snapshot after each non-final chunk, and the final chunk split as `plen - pos - 1` tokens + snapshot + the last token alone. `SlotModel::prefill_chunk` does it, so the scheduler's chunk accounting is unchanged and the solo reference is the same schedule.
* the slot count limit `MAX_BATCH_CAP` becomes backend-dependent (8 Metal, 16 CUDA proposed).

## 4. Batched decode for GDN: gather/scatter, N launches, or a state table

One decode token for N sequences means N independent T = 1 recurrences, one per slot: `conv_state_r <- shift(conv_state_r, qkv_r)`, `S_r <- decay(S_r, a_r) + beta_r k_r (v_r - S_r^T k_r)`,
`core_r = S_r^T q_r` (`gdn.rs`, `docs/hybrid-models.md`). The projections that feed it (`in_qkv`, `in_z`, `in_b`, `in_a`) and the gated norm and `out_proj` after it are ordinary
N-row GEMVs. Only the conv and delta rule touch per-sequence state. Per layer, per sequence: read and write 2.1 MB (`conv` 98 KB + `S` 2.1 MB).

| option | state bytes moved per step | launches per step (30 layers) | graph-capturable | exact vs solo |
|---|---|---|---|---|
| **A. gather into `[N, ...]`, one batched kernel, scatter back** | 126 MB read+write by the kernel **plus** copy-in (r+w 126 MB) **plus** copy-out (r+w 126 MB) = **378 MB per sequence = 3.0x** | 30 + 60 copies | yes | yes (same kernel arithmetic) |
| **B. N per-row launches on each slot's own buffers (the MTP verify loop and the attention rows do this)** | 126 MB per sequence (in place) | 30 x N | no unless the kernel args are static: a graph binds one `SequenceState::id` (`graph_step.rs:178-180`) | yes by construction (the T = 1 call is the decode call) |
| **C. one launch over N states: kernel takes a device table `slot_of_row[N]` + static `slot_base[]` pointers, `blockIdx.z = row`** | 126 MB per sequence (in place) | 30 | yes (the table is rewritten per step like `ws.pos`) | yes (per row the same arithmetic, same CTA shape) |

Step cost at N = 16 (derived): A moves 6.05 GB of state against 2.1 GB for B/C, +3.9 GB on an 11.3 GB step = **+35% step bytes**, a 26% throughput loss; ruled out. B and C move
the same bytes, and the single-sequence kernel is already at bandwidth (126 MB in 673 us = 187 GB/s, `2026-10-14-cuda-decode-fusion.md`), so **B's device time equals C's** (16 x 673 us = 10.8 ms,
the same as 2.1 GB / 194 GB/s). What C removes is 30 x (N - 1) launches (450 at N = 16, about 1-2 ms inside a graph, 5-8 ms eager) and the underfilled-GPU tail of each T = 1 launch
(8 x 32 = 256 CTAs of 32 threads per layer, `gdn.cu:635-647`).

**Recommendation.** Never A. Stage S2 uses B (it is exact by construction and needs no new kernel). Stage S4 builds the *table* version of the existing launches, which is what
graph buckets require (the kernel reads its state address from a device table, so one captured graph serves any assignment of rows to slots); the single `blockIdx.z` launch (C) is then a
small step (the CTAs have no cross-coupling) and is taken only if the profile shows the 450 extra graph nodes matter. An arena (one buffer `[slots, ...]` per layer) would give the same
addresses with a slot index instead of a pointer table; it is not needed because slot buffers are already allocated once, and it would change `SequenceState` allocation, snapshot
copies and the prefix-store restore path. Not recommended unless the table proves a bottleneck.

What the table must cover for graphs to be state-independent, per layer: GDN conv state and `S` base per slot; K and V cache base per slot (attention); and per row the position
(`ws.pos` becomes `[N]`), the token (`ws.ids` `[N]`) and the output rows (`ws.logits` / `ws.tok` `[N]`). A bucketed graph (N in 1, 2, 3, 4, 6, 8, 12, 16) is padded with a dummy slot whose
outputs are discarded; one spare `SequenceState` is the cost (63 MB + a small KV).

## 5. The rest of a batched hybrid step

A decode layer of the 35B is norm -> block -> norm -> MoE. For N rows:

* **Dense projections** (GDN `in_qkv`/`in_z`/`out`, attention `q`/`k`/`v`/`o`, shared expert, router, `lm_head`): multi-row GEMVs, one weight read for N rows. CUDA has fp8 `matmul_fp8_gemv` for M in 1..=4
  and nvfp4 `matmul_nvfp4_gemv` for M in 1..=8; the fused pair kernels are M = 1 only (`quantmoe.rs:139, 217, 318`). 16 rows need new or extended kernels.
* **Gated attention** (10 layers): q + gate projection once for N rows, `GateSplit` once, then **per row** QK-norm/rope/KV-append and `AttnDecode` on that row's cache (`attn_rows`, `model/batch.rs:205-235`), then
  the output gate (`SigmoidMul`) and `o_proj` once for N rows. The gate must be threaded through `attn_block_batch` (defect 5 in section 1).
* **GDN** (30 layers): section 4.
* **MoE** (40 layers): router for N rows, top-8 per row, routed experts, gated shared expert, combine. The T = 1 CUDA path is a dedicated pair of kernels (`moe_experts_decode_v2_bf16`,
  gate_up on a `(pair, I/8)` grid, down on `(token, H/8)`; `quantmoe.rs:92`, `DECODE_MAX_TOKENS = 1`); T in 2..31 uses the grouped warp-per-row kernel and T >= 32 the tensor-core kernel,
  whose output differs by up to 3.5e-4 rel L2 (`docs/quant-moe.md:75-80`). **Whether the T in 2..31 kernel is bitwise equal to the T = 1 kernel has not been shown**, so an exact batched step should
  not route N rows to it; the first step is the decode kernel extended to N tokens (its grid is over pairs, each pair independent, so per-token arithmetic does not change). That is the "per-pair" row of the table in section 2.
* **Head**: `lm_head` for N rows into `ws.logits` (16 rows exist) and a row-wise argmax, as `batch_head` does.

## 6. Mixed prefill + decode iterations

Policy unchanged from the Metal scheduler (`continuous-batching-design.md` section 2.4, decisions 2 and 12): decode-first; the prefill chunk is a **separate forward pass** after the batched decode step; busy chunk
256, solo chunk 2048. Decode rows never share a GEMM with prefill rows, which is what keeps every decode row bitwise the solo row.

For a hybrid this is not a new problem: a prefilling sequence P owns its slot's `SequenceState`; the GDN scan for its chunk (`gdn_delta_rule_bf16_split`, tokens sequential inside the CTA, state in place) reads and writes P's
recurrent buffers only, and the decode rows' GDN steps read and write theirs. The two passes are serialised on the stream, so there is no hazard. The costs (estimate) are what a chunk of 256 tokens costs on the
35B: it touches most of the 256 experts per layer (2 048 expert picks), so it is close to a full 18 GB expert read, about 60-90 ms. Decode streams see one such stall per iteration while a prompt is prefilling, the same
shape as Metal's 70 ms stall. The scheduler's `busy_chunks_per_iter` and the EWMA step-time hook already bound it.

A Sarathi-style fold (decode rows inside the prefill GEMM) would save about one weight pass per mixed iteration but makes decode rows depend on the tile choice of the prefill GEMM (not bitwise, `continuous-batching-design.md`
4.4/4.5) and, for GDN, needs a ragged varlen GDN scan kernel (rows of different sequences with different state pointers inside one launch). Out of scope until S6.

## 7. Prefix reuse and the recurrent snapshots

State machine today (`state.rs`): a sequence holds the live state at `len`, and up to 4 snapshots, each valid for `tokens[..snap.len]`. `restorable_len(want)` is the largest of
{current length, a snapshot length <= want, 0}. `truncate(len)` restores the snapshot taken at exactly `len` (63 MB device copy) or zeroes the state for 0. Solo prefill snapshots after each
non-final chunk and at `plen - 1` (`engine.rs:1139-1153`).

Under batching, with slots:

* **Admission.** `pick_slot` takes the free slot with the longest *restorable* common prefix (new hook, section 3), ties to least recently used, exactly the current policy with a different distance.
  `truncate(slot, cached)` then restores a snapshot into the slot's live buffers (63 MB copy, 0.65 ms at 194 GB/s, small against the prefill it saves).
* **Prefill.** Per slot, the snapshot after each non-final chunk and at `plen - 1`; the last prompt token runs alone. This is the solo schedule, so batched and solo prefill are the same sequence of passes.
* **Decode.** Decode writes the live state in place and never snapshots; a finished request leaves its slot idle with live state at its final length. The next prompt reuses it only if the prompt
  continues the held tokens (multi-turn without the Qwen3.6 template dropping earlier `<think>` blocks) or hits a snapshot.
* **Memory.** 4 snapshots per slot is 4 x 63 MB x N (4.9 GiB at N = 16). The prototype keeps `RECURRENT_SNAPSHOTS` = 4. For production the number per slot should be a scheduler setting (2 at N >= 8), because the
  prefix store (below) is the right home for long-lived snapshots.

**Interaction with the in-flight prefix-cache lane** (`origin/lane-pc/prefix-store`, worktree `/opt/code/wt-pc`; 26 files, +3 792 -14; I read its diff of `state.rs`, `engine.rs`, `model.rs`, `plan.rs`, `load.rs` and the head of
`prefix_store.rs`). It is a `LocalEngine` (B = 1) feature and refuses `--max-batch > 1` at load (`load.rs`: "the prefix store needs the single-sequence engine"); its diff of `engine/batch.rs` and `scheduler.rs` is empty.
Collisions with this design, all small:

| pc lane change | effect on this design |
|---|---|
| `SequenceState.clean` (prefill-clean watermark), `SequenceState::load_prefix`, literal `clean: 0` in `new` and `plan.rs::phantom_state` | any code building `SequenceState { .. }` literally must add the field; batched decode must not advance `clean` (it does not: `commit` leaves it alone) |
| `Seq::new` rewritten (`local_reuse`, `restore`), `Seq::prefill_step` chunking by store boundaries, `store_at` after chunks | the batched path does not call either (it uses `Seq::build` + `admit` and `SlotModel::prefill_chunk`); **if the store is later allowed at B > 1 the batched prefill must use the same chunk planner (`SeqStore::chunk_end`)** or snapshots land at different boundaries than solo |
| `GenRequest.prewarm`, `EngineMeta.prefix_store` | new fields on structs `BatchedEngine` constructs; trivial conflicts only |
| `PrefixStore` is `Arc<Mutex<..>>`, entries are host-memory blobs of one recurrent state (about 63 MB) plus shared KV pages | usable per slot: `insert`/`restore` take one `&SequenceState`; one shared store across N slots works; its 8 GiB host budget is independent of the N x 63 MB live state |
| unrelated opt-in dead-tail skip in `forward_inner`/`run_layers`, guarded to `Seqs::One` | does not touch `Seqs::Batch`; a textual conflict in `run_layers` is possible, none semantically |

Merge order: whichever lands second rebases; my `run_layers` change is a three-line arm each for `Block::Gdn` and `Block::Moe` under `Seqs::Batch`, and the new code is in new files
(`model/gdn_batch.rs`).

## 8. Speculative decoding (`--spec mtp`, PR #110)

PR #110 (`lane-mtp/spec-decode`, draft, last commit 2026-10-22 22:32) runs a verify pass of `k + 1` rows with the GDN conv + delta rule **one row at a time** (T = 1 calls, so each row is the decode launch), copying the
layer's recurrent slot into `st.spec_rows.bufs[r]` after each of the first `k` rows (`SpecRows`, `k x 63 MB` per sequence), and rolls back with `SequenceState::spec_rollback(keep)`. It refuses the batched engine
(`BatchedEngine::new`: "--spec mtp with the batching engine (use --max-batch 1)") and still has `Seqs::Batch` + GDN unsupported.

Design for the pair of features:

* **This lane does not change #110's contract.** Batching enabled for a recurrent model refuses `--spec` (`meta.spec.enabled()`), exactly as #110 does for MTP, and passes `recurrent = true` to `Seq::build` so no n-gram draft is ever
  built on a recurrent state. Both are one-line guards in `BatchedEngine::new` / `submit`.
* **Code collisions.** #110 moves the body of `gdn_block` into a `conv_delta` helper and adds a verify branch; this lane adds `gdn_block_batch` in a new file and does not edit `gdn_block`. The shared surface is the `conv_delta`
  helper: after #110 lands, `gdn_block_batch` should call it (a three-line change) instead of the op sequence it duplicates today.
* **Where they meet later (S6).** MTP is a low-load latency feature; batching is a load feature. B = 1: MTP as in #110. B >= 2: spec off for every sequence (the existing "busy" rule, `scheduler.rs:807`, `SPEC_OFF_AT_DEFAULT`).
  A sequence that was speculating and shares a step with others must keep the MTP head's queue of pending hidden rows consistent: `forward_batch` would have to return each row's final normed hidden state for sequences that carry an
  `MtpState`. That interface (a per-row hidden output from the batched pass) is the real work and is S6. The data from Metal says it may not pay (`continuous-batching-design.md` decision 16: batched verify no-go at +1.3%).
* **Memory.** With spec on, N sequences x `k` x 63 MB of `SpecRows` (k = 2: 126 MB per sequence, 2 GiB at N = 16) is another reason to keep it B = 1.

## 9. Per-backend kernel work

### CUDA (Spark, first)

No `rows_exact` kernel is registered on CUDA (`kernels.rs`; PR #110 confirms: "CUDA registers none"). Registering them is the honest cost of an exact batch. Work items, in priority order by bytes shared (section 2):

| item | what | why / evidence | est. |
|---|---|---|---|
| K1 | fp8 GEMV for M = 2..16 rows, **row r bitwise equal to the M = 1 kernel** (same K-split, same reduction tree per row); plus the fused fp8 pair and `SigmoidMulMatmulFp8`, `GatedRmsnormMatmulFp8`, `MatmulPair`, `RmsNormAddMatmulPair` multi-row forms | 47% of the step (5.8 ms); today fp8 M <= 4, pair fused M = 1 only | 4 d |
| K2 | nvfp4 GEMV (lm_head, shared expert) for M up to 16, row-exact | 10.6% + shared expert; exists M <= 8 | 1 d |
| K3 | MoE decode: `moe_experts_decode_v2` for N tokens (grid over `N x 8` pairs; token dimension in gate_up and down grids, router top-k per row) | 23% of the step; `DECODE_MAX_TOKENS = 1` | 2 d |
| K4 | GDN conv + delta rule reading `state` from a device table, `row -> slot` index, `blockIdx.z = row`; T = 1 | 5.4% at N = 1, 18% at N = 16; same CTA shape per row so per-row arithmetic is unchanged | 2 d |
| K5 | `QkNormRopeKv` and `AttnDecode` with per-row position and slot tables (no per-row launches) | 10 layers x N x 3 launches otherwise | 3 d |
| K6 | graph buckets: `ws.ids`, `ws.pos`, `ws.logits`, `ws.tok` become `[N]`, one `Captured` per bucket, dummy-slot padding | `graph_step.rs:33,43,178-180` caches one state-bound recording | 3 d |
| K7 | (S6) expert-grouped decode MoE: each CTA loads an expert tile once and loops over its rows | +2% at 4 streams, +16% at 16 | 4 d |

CUDA graphs and fusion interplay: `fused_ok` (`model.rs:1618-1653`) drops a fused op from the **solo** step when `batch_rows >= 2` and the op has no kernel at every row count (the batch-consistency rule, `plan::check_batch_ops`).
On CUDA today that removes the four phase-1 fusions from B = 1 once batching is enabled (944 launches and 13.95 ms instead of 724 and 12.41 ms, -12%, `2026-10-14-cuda-decode-fusion.md`). Either K1 supplies the multi-row forms (this is why
they are in K1), or the rule is relaxed on CUDA. This is why S3 includes them, and why B = 1 with `--max-batch > 1` is a gate (section 10).

### HIP

GDN conv and delta rule exist (`kernels/hip/gdn.hip`, fused `gdn_fused.hip` for dk = 128, K = 4, T <= 32). Same state convention (a base buffer), so K4's table form ports mechanically. HIP MoE is deterministic but **not batch-invariant from 16 tokens**
(`docs/quant-moe.md:78-81`, `docs/backends/hip.md`), so an exact batch on HIP is limited to the GEMV-style regime (T < 16 and T x top-k <= 16 as documented; for top-8 that may be as low as T = 2, to be measured) or needs `INTARSIA_HIP_MOE_WMMA=0` (MoE 3x slower at T = 256). Estimate: 5-6 days after CUDA.

### Metal

`kernels/metal/gdn.metal` has conv (`gdn_conv_bf16_k*`) and delta rule (`gdn_delta_rule_t<DK>`) with state as buffer + offset, T = 1 supported; Metal already has the row-exact bf16 GEMVs and split-KV rows. Missing for the 35B on Metal: row-exact multi-row forms of its quantised
GEMVs and MoE (the `m35` lanes), and the same per-row GDN loop (option B costs nothing in correctness). Estimate: 5-7 days after the CUDA design is proven; the Metal 35B decode is 8 064 us per step
(`docs/results/2026-10-22-md-metal-35b-decode-profile.md`), so the same roofline method applies with that machine's bandwidth.

## 10. Accuracy gates

The contract is the one the Metal scheduler states (`scheduler.rs:13-35`): a request's token stream must not depend on what runs beside it; the reference is the same request alone, same chunk schedule, same cached length, same
decode-written/prefill-written history. Gates, in the order a stage must pass them:

| gate | what | where | status |
|---|---|---|---|
| G-A engine logic | batched (N = 2..4) greedy output token-identical to solo over >= 128 tokens; slot reuse, prefix reuse on and off, cancel, error isolation | CPU backend, tiny hybrid fixture, every PR | **built (section 12)** |
| G-B kernel row exactness | for each batched kernel, row r of an N-row call equals the 1-row call **bitwise**, N = 1..16, random inputs, edge positions | CUDA unit tests on spark (`--load cpu` is not enough) | per kernel K1-K5 |
| G-C model bitwise | `forward_batch` logits equal solo `forward` logits bitwise per row, for >= 128 steps, N = 2, 4, 8, 16, mixed lengths (`tests/common/batch_identity.rs`, the harness Metal uses) | CUDA, tiny hybrid + 35B | S2/S3/S4 |
| G-D end-to-end identity | >= 200 requests, staggered arrivals, greedy and sampled, `min_tokens`, stop strings, EOS, cancels, prefix hits, mixed prompt lengths: every output equals its solo run at the same chunk schedule and cached length, 0 mismatches | CUDA, 35B | before default-on |
| G-E nothing else moved | default path byte-identical (`plan show`, 256 greedy ids vs main); the existing qualify / KL / GSM8K gates unchanged at B = 1; B = 1 with `--max-batch 4` within 2% of B = 1 without (the fused-op rule above) | CUDA | S3 |
| G-F speed | same-slot interleaved A/B on an exclusive slot, 3 rounds: >= x1.4 at 4 streams in S3, >= x1.75 at 4 and >= x2.5 at 16 after S5 | spark1, fleetq exclusive | per stage |

**What MTP proved and did not.** The MTP lane's CUDA result is 12 of 12 runs token-identical over 256 greedy tokens and it says in its own text "evidence, not proof"; CUDA has no `rows_exact` kernels, so the verify path's argmax can differ from the
decode path's. For batching the same weakness would be worse (more rows, more kernels), so G-B and G-C are **bitwise** gates on real kernels, not token-identity runs. If a kernel cannot be made row-exact, the fallback is a measured KL budget
(open question 1), not a silent loosening.

## 11. Staged plan, estimates, expected speed-up

Working days of one engineer or agent lane, estimates. Speed-ups are on the composer's concurrent metric against 74.2 / 75.2.

| stage | content | days | gate | expected 4 / 16 streams |
|---|---|---|---|---|
| S0 | this doc; CPU prototype (opt-in `INTARSIA_RECURRENT_BATCH=1`, CPU only), G-A | done tonight | G-A | n/a |
| S1 | engine plumbing for hybrids production-ready: `restorable_len` hook, recurrent prefill mirror, spec guards, memory plan term, per-slot snapshot budget, `MAX_BATCH_CAP` per backend, server `ModelInfo.max_batch` | 1.5 | G-A in CI, all existing batch tests green | CPU only |
| S2 | CUDA bring-up, exact by construction: `enable_batch` for CUDA hybrids, GDN rows via option B (per-row T = 1 launches), MoE per row, attention per row, dense projections per row with the existing 1-row kernels except where a multi-row kernel is already proven. Harness: G-C on spark | 4 | G-C bitwise, G-D short | x1.0 (correctness harness; no sharing) |
| S3 | K1 + K2: multi-row dense GEMVs and fused forms; dense weights shared | 5 | G-B, G-C, G-E, G-F | **x1.4-1.6 at 4 (105-120), x1.8-2.0 at 16** |
| S4 | K3 + K4 + K5: N-token MoE decode, GDN/attention tables, no per-row launches | 5 | G-B, G-C, G-F | x1.75-1.9 at 4, x2.2-2.5 at 16 |
| S5 | K6: graph buckets, dummy-slot padding, B = 1 with batching enabled equals B = 1 | 3 | G-D full, G-E, G-F | **x1.75-1.95 at 4 (130-145), x2.5-2.75 at 16 (185-205)** |
| S6 | K7 expert grouping; HIP; Metal; prefix store at B > 1; MTP at low batch; mixed prefill/decode fold | separate lanes | | 16 streams 210-235 |

Critical path S1-S5: 18.5 days. The fraction of the ceiling assumed for S5 is 80-90% of the per-pair ceiling (SGLang's measured 95% / 83% of the grouped ceiling is the existence proof; the 16-stream
number is the less certain one because of per-row attention and host time). Break-evens worth knowing: the vLLM +40% gap is closed at x1.40 on 4 streams, which S3 reaches if K1's 4-row fp8 path is as fast as its 1-row path
(it must be, the weights are 1.28 GB and the GEMV is memory bound).

**What would change the plan.** (a) If a multi-row fp8/nvfp4 GEMV cannot be made row-exact at no speed cost, G-B fails at K1: take the open-question-1 fallback. (b) If CUDA graph replay of the bucket loses the 98.6 us idle advantage (kernel args via tables
add indirection), S5's launch saving shrinks and S4's single-launch kernels matter more. (c) If real routing is much more skewed than uniform, K7 is worth more than +16% and moves up.

## 12. CPU prototype

Opt-in, cpu backend only, default off. Nothing in it is a GPU kernel; it is the engine logic of stage S1 and the model-level structure of S2, proven on the cpu backend, whose multi-row kernels compute every row with the
one-row loop (so a batched row is the solo row by construction there; the tests prove the plumbing, state handling and scheduling, not GPU numerics).

**Switch.** `Model::allow_recurrent_batch()` (cpu only, before `enable_batch`) and `INTARSIA_RECURRENT_BATCH=1`, which `load_engine` reads (`crates/intarsia-models/src/load.rs`) and which lets `--max-batch N` through on the cpu backend. The
switch is a field on `Model`, not an environment read inside the engine, because tests of one binary run in parallel threads. Without it every refusal of section 1 is unchanged.

**What changed (files).**

| file | change |
|---|---|
| `crates/intarsia-core/src/model/gdn_batch.rs` (new) | `gdn_block_batch`: four projections and the gated norm + out-projection on all rows; conv + delta rule once per row as the T = 1 calls of the solo step, on that row's own recurrent slot, in place |
| `crates/intarsia-core/src/model.rs` | `recurrent_batch` field; `run_layers` arms for `Seqs::Batch` at a GDN block (new path) and at a MoE block (allowed under the switch) |
| `crates/intarsia-core/src/model/batch.rs` | `allow_recurrent_batch`/`recurrent_batch`; `enable_batch` and `check_batch_row` accept hybrid blocks under the switch; `attn_block_batch` passes the output gate through and makes the solo step's per-row fused gate-split + prologue call |
| `crates/intarsia-core/src/plan.rs` | the batch-consistency check treats `GdnConv`/`GdnDeltaRule`/`GdnConvDelta` as per-row ops and accepts `GatedRmsnormMatmulFp8` either once (multi-row kernel) or once per row (one-row kernel only) |
| `crates/intarsia-core/src/scheduler.rs`, `engine/batch.rs` | `BatchModel::restorable_len` (used by `pick_slot`); `SlotModel` prefills recurrent slots with the solo schedule (snapshot after each non-final chunk, last prompt token alone after a snapshot); `BatchedEngine` refuses `--spec` on a recurrent model and passes `recurrent = true` to `Seq::build` |
| `crates/intarsia-core/src/model/gdn_block.rs` | visibility only (`state_s_offset`, the GDN workspace fields), no behaviour change; `gdn_block` is untouched so PR #110's restructuring does not conflict |
| `docs/tuning.md` | the `INTARSIA_RECURRENT_BATCH` row (the env-var scanner requires it) |

**Tests** (`crates/intarsia-models/tests/batch_recurrent.rs`, `batch_recurrent_load.rs`; run on two tiny hybrids: the checked-in dense-FFN `tiny_qwen3_5` fixture, and the same fixture with a MoE FFN built the way the 35B's is: fp8 attention/GDN
projections, NVFP4 routed and shared experts, a shared-expert gate):

| test | what it pins down |
|---|---|
| `hybrid_batching_is_refused_without_the_opt_in` | with the switch off `enable_batch` still refuses the hybrid model ("dense attention-only"), the switch must precede `enable_batch`, and no engine can be built over an unbatched model |
| `batched_hybrid_decode_is_bitwise_the_solo_step` | `Model::forward_batch`, N = 2, 3, 4 sequences, 130 consecutive batched steps each: every row's f32 logits equal the solo step's bit for bit (greedy rows on both `LastLogits` and the on-device `LastArgmax` path, seeded-sampled rows), and tokens, lengths and the later steps that read the recurrent state agree |
| `batched_hybrid_decode_beside_prefill_chunks` | batched steps interleaved with chunked prefill of another sequence: decode rows still equal solo, and the prefilled state equals the same prompt prefilled alone |
| `batched_engine_equals_solo_over_128_tokens` | `BatchedEngine` with 2, 3 and 4 slots against `LocalEngine`, 9 requests each (prompts 3 to 64 tokens, 130 to 144 generated tokens, greedy and seeded-sampled), more requests than slots so slots are recycled: ids, text, finish reason and counts identical |
| `batched_engine_with_prefix_reuse_equals_solo` | the same with idle-slot prefix reuse on |
| `prefix_reuse_restores_recurrent_state_from_a_snapshot` | whole-prompt reuse, a shared prefix ending between chunk boundaries, an extended prompt and a short prefix, over 2 and 3 slots, with at least 2 requests per run reusing a prefix: output identical to cold solo |
| `load_engine_batches_a_hybrid_model_only_with_the_env_opt_in` | the CLI path: a model directory with a `tokenizer.json`, `load_engine` with `--max-batch 3`: refused without the variable (and with `0`), served with `1`, three concurrent requests equal the single-sequence engine |

**Negative controls** (each mutation made in the worktree, the failing test noted, then reverted): GDN rows all read `sts[0]`'s state -> `batched_hybrid_decode_is_bitwise_the_solo_step` fails at the first step
("logits differ at 96 of 96 entries"); `pick_slot` ignores `restorable_len` -> `prefix_reuse_restores_recurrent_state_from_a_snapshot` fails with "cannot roll recurrent state back to 33 tokens (no snapshot there; length 169)"
(the request fails: the defect section 1 predicted); the recurrent prefill takes no snapshots -> the same test fails with "only 0 of 5 requests reused a prefix". Not observable on the cpu backend: whether the last prompt token runs
alone (a bigger chunk and a 1-row call give the same bits there); that rule is a GPU gate (G-C/G-D).

**What the batch-consistency check found on the first run** (it fails the load, as designed, when a batched step would not be the solo step's op multiset): (1) the solo step folds the gated-attention split into the per-row prologue
(`GateSplitQkNormRopeKv`), which the batched step did not, so `attn_block_batch` now makes that call per row; (2) the GDN conv and delta rule are per-row ops in a batched step; (3) the solo step's fp8 out-projection fold
(`GatedRmsnormMatmulFp8`) has a **one-row kernel only**, so a 2-row batched step ran two separate ops where the solo step runs one. The prototype runs the fold per row in that case (the same op, the weights read once per row). This is
the CPU instance of the CUDA finding in section 9: today's fused decode ops are one-row kernels, and every one of them needs a multi-row form (K1) or a per-row run before a batched step is both exact and fast.

**Default path.** With the switch off the flag-gated code is not reached: `enable_batch` and `check_batch_row` keep their old predicates, `gdn_block` and its callers are untouched, the new `run_layers` arm is under `Seqs::Batch`
and the switch, `restorable_len` defaults to the old behaviour, and `attn_block_batch` builds its fused gate prologue only for a gated block (an ungated block takes the call sequence it always did, including the `exact_rows` mark).
Evidence:

| check | result |
|---|---|
| `cargo test -p intarsia-core -p intarsia-models -p intarsia-cli --no-fail-fast` (cpu, 46 test binaries incl. `batch`, `batch_engine`, `hybrid`, `plan`, `spec`, `docs_env_vars`) | 454 passed, 0 failed, 2 ignored |
| re-run of `batch`, `batch_engine`, `hybrid`, `batch_recurrent`, `batch_recurrent_load` after the last `attn_block_batch` guard | all pass |
| **Metal default path**, `INTARSIA_TEST_REAL_MODEL=1 cargo test -p intarsia-models --release --test batch_metal` on this Mac (not a fleet box), Qwen3-8B BF16, `~/Models/Qwen3-8B` | pass: 138 (row, step) pairs bitwise identical to solo, 48 decode rows bitwise identical beside 256-token prefill chunks |
| `plan show` + 256 greedy ids, origin/main vs HEAD at `--max-batch 2` (G-E) | **not run**; the Metal batched bitwise gate above is the check on the changed default path |

**Test output.**

```
$ cargo test -p intarsia-models --test batch_recurrent --test batch_recurrent_load
     Running tests/batch_recurrent.rs (<scratch>/target/debug/deps/batch_recurrent-313990ed38fa7ce3)
test hybrid_batching_is_refused_without_the_opt_in ... ok
test batched_hybrid_decode_beside_prefill_chunks ... ok
test prefix_reuse_restores_recurrent_state_from_a_snapshot ... ok
test batched_hybrid_decode_is_bitwise_the_solo_step ... ok
test batched_engine_with_prefix_reuse_equals_solo ... ok
test batched_engine_equals_solo_over_128_tokens ... ok
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 44.37s
     Running tests/batch_recurrent_load.rs (<scratch>/target/debug/deps/batch_recurrent_load-a4d63c441f98c58e)
test load_engine_batches_a_hybrid_model_only_with_the_env_opt_in ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.99s
```

## 13. Risks, open questions, decisions

Risks: (1) row-exactness of the CUDA multi-row kernels (largest); (2) 16 rows of GEMV compute (untested on CUDA); (3) the B = 1 fused-op regression when batching is enabled; (4) rebase churn with #110 and the pc lane in `run_layers` / `gdn_block`
(mitigated by new files); (5) 4 snapshots per slot (4.9 GiB at N = 16).

**Open questions for the user**

1. **Exactness on CUDA.** Bitwise-to-solo needs K1-K5 row-exact kernels (about 17 days of kernel work, section 9). If a kernel resists, accept a non-bitwise batch mode behind a KL gate (as HIP's f8 mma default), or hold the stage? The plan assumes bitwise with a KL fallback only by your decision.
2. **Cap 16 on CUDA.** `MAX_BATCH_CAP` and `BATCH_MAX_ROWS` are 8, an operator decision for Metal's exact kernels. The 16-stream row needs 16 on CUDA only; OK to make the cap per backend?
3. **B = 1 cost of enabling batching.** Keep the batch-consistency rule (solo gives up its fusions until K1 supplies multi-row forms; -12% at B = 1 with `--max-batch > 1`) or relax it on CUDA and gate batched-vs-solo bitwise equality separately?
4. **MTP and the prefix store at B > 1.** Both are refused at B > 1 in this design until after S5. Agree?
5. **Which rival number is the target?** vLLM's rows carry a quality flag (`drop:gsm8k`, peer-not-production) and come from a server capped at `--max-num-seqs 2`; this plan treats SGLang (157.5 / 219.9) as the bar and vLLM's +40% as a floor. Worth re-running vLLM with `--max-num-seqs 16` before the next composer pass.

**Decisions made in this lane** (rows added to `docs/decisions/LOG.md`): slots as independent `SequenceState`s with in-place state access (no gather/scatter); GDN batched as per-row T = 1 semantics, per-row launches first and a
device-table kernel later; prefill stays a separate pass; hybrid prefix reuse through `restorable_len` with the solo snapshot schedule; MTP and the prefix store stay B = 1; opt-in flag, CPU only, default off.

## 14. Reproducing the numbers

`docs/results/2026-10-23-rb-recurrent-batching/`: `wbytes.py` + `wbytes.txt` (checkpoint bytes by tensor class, from the safetensors headers of `~/Models/Qwen3.6-35B-A3B-NVFP4`), `roofline.py` + `roofline.txt` (section 2),
`cells.txt` (composer cells for GB10 / 35B). Sources: composer `regimes.json` (`/opt/code/model-benchmarks/composer/data`), `docs/results/2026-10-14-cuda-decode-fusion.md`, `docs/quant-moe.md`, `docs/research/continuous-batching-design.md`,
PR #110 (`/opt/code/wt-mtp`), `lane-pc/prefix-store` (`/opt/code/wt-pc`).
