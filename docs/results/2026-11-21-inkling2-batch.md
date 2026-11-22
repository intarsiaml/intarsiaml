# Inkling-2-Mini batched decode (lane ink-batch, 2026-11-21)

Batched decode of 2..=16 sequences in one step (`Model::forward_batch`) for Inkling-2-Mini, CPU reference and Metal.
Branch `ink/batch` from `ink/metal` 1dacac3a8.

## Briefs received

- 2026-11-21 (main, after the design note): design approved, proceed. (1) Memory: count the full-attention KV of the 8
  full visits per sequence at max context as well as the rings; `enable_batch` (and the server) must cap B so everything
  fits and refuse cleanly rather than over-allocate. (2) Solo path: single-stream decode tok/s and G2 short / long
  unchanged before vs after the merge (bitwise logits on the short set preferred). (3) MTP later: one row per sequence now;
  do not design out several rows of one sequence (draft verify across streams), do not build it; one line on what it needs.
  (4) Throughput through localeval (local-model-bench) at 1/4/8/16 streams under `fleetq --quiet gpu,cpu`, oMLX's 94 tok/s
  at 16 streams in the same table.
- 2026-11-21 (coordinator): Metal `generate` defaults to `--profile-every 16` (every 16th step waits on 419 command buffers,
  about 4 ms/token of fake gap; real single-stream greedy is about 15.76 ms = 63.4 tok/s). Use `--profile-every 0` for every
  engine-side decode timing, the solo no-regression check included; localeval through serve is unaffected.
- 2026-11-21 (coordinator): endor's quiet-GPU line is 14+ jobs (about 9 h). Correctness runs (#10810 mixed-context
  batched == solo, the G2-bitwise part of #10795) go as default-gpu-load jobs with short honest `--expect` (they share the box
  and backfill); only tok/s (#10773, the tok/s of #10795, #10811) stays `--quiet gpu,cpu`. ink-metal fixed the rows_exact pin
  list at a0751f302: merge ink/metal.
- 2026-11-21 (coordinator, after 173.7 tok/s at 16): (1) do the 9..16-row single-read work now: dense row GEMVs (in_proj, wo,
  unembed) read the weights once for all rows instead of two 8-row chunks, then the experts read each expert's weights once
  per step over the rows routed to it; target 16 streams >= 230 tok/s, every row still bitwise its solo step (the tiny tests
  and #10844's mixed-context check are the gate); new kernels opt-in by row count first; if a 16-row form cannot stay
  bitwise with the 8-row one, ask before changing the contract. (2) Confirm from code that serve / the engine route a single
  live sequence through the solo path, not a B = 1 batched step (B = 1 batch_step 17.5 ms vs 15.86 ms solo); if not, make
  them. (3) Keep #10844 / #10845 / #10811 queued; tiny / cpu tests for new work on spark2 (`--load cpu --mem 30`, copies in
  ~/intarsiaml-ink-batch); endor only for Metal admission and timing.
- 2026-11-21 (coordinator): `intarsia serve` on Metal refused Inkling (require_admitted: no admitted moe_experts kernel; the
  plan-shape case builder dropped `attrs.ink`, so the SiTU kernels were never admitted). Fixed in ink-metal 761a1561b
  (quant/cases.rs, test `situ_experts_case_keeps_the_ink_attrs`). Merge ink/metal at or after 761a1561b now; #10811's script
  must build from that tip at job start, not use a prebuilt binary.
- 2026-11-21 (coordinator, for the 8-vs-16 profile): ink-m-mix measured InkInProj at the real shape: 1 row 69.8 us, 8 rows
  83.3 us, 16 rows 168 us (241 GB/s, arithmetic-limited; the three-Matmul form is no faster at 16 rows, 260 GB/s), about
  +4 ms per step from 8 to 16 rows over 48 visits. Their idea 3, a matrix-unit GEMV for 9..16 rows bitwise equal to the 1-row
  kernel, is my call after the profile: build it only if InkInProj is the biggest share and it can stay bitwise per row.
  ink-metal is merging ink/m-mix 47570539e (InkInProj up to 8 rows on the GEMV instead of NAX): re-merge ink/metal before the
  next batch timing.

## 1. Design note (contract delta, written before any kernel)

### 1.1 Shape of a batched step

A batched step of B sequences is ONE pass of the visit table over B rows (row r = sequence r's next token), exactly the
loop of a solo B-row pass (`Model::ink_pass`), with two differences:

* every row-local op (`Embedding`, `RmsNorm`, `InkInProj`, `Matmul` wo, the router `Matmul` / `Logits`, `MoeRoute`,
  `MoeExperts` / `MoeExpertsMxfp8`, `AttnResMix`, `LoopBlend`) runs once over the B rows under `exact_rows` (the row-exact
  multi-row kernels that are bitwise the 1-row kernel per row: the property solo 2..=16-row passes already rely on,
  `ink_rows_exact`). The workspace planes are `[B, D]` exactly as for a B-row chunk. **MoE union grouping across
  sequences needs nothing new**: `MoeExperts` sees B rows and does not know or care which sequence a row is from.
* every op that touches per-sequence state, `SconvRes` (69), `InkKvPrologue` (70), `RelSinkAttn` (71), `ConvMix` (74) and
  `InkAttnDecode` (75), runs in a new **row-table form**: one call over the B rows, row r being the 1-row (T = 1) call
  on sequence r's own state at sequence r's own position.

### 1.2 Contract delta (appended, optional; the existing calls are unchanged bit for bit)

The row-table form reuses the existing batched-decode mechanism (`OpCall::rows: Option<RowTab>`, R3a; today defined for
`GdnConv`, `GdnDeltaRule`, `KvAppend`, `AttnDecode`, `QkNormRopeKv`) and extends it to ops 69, 70, 71, 74, 75:

* `call.rows = Some(RowTab { ptrs: u8 [B, stride, 8], rows: B, stride, index, byte_off: 0 })` and `call.pos = Some(i32 [B])`
  (per-row positions; required with a row table). A row-table call has `T = B` rows in every row-indexed tensor
  (`[T, ..]` activations, and row r of each plane of `[N, T, D]` / `[S, T, D]` sources, messages and the `[S, T, D]` mix).
* **Semantics.** Row r of the call is bitwise the T = 1 call at position `pos[r]` (the device-position form of the solo
  step) whose row-indexed tensors are row r of the call's and whose state tensors are sequence r's. No row reads or writes
  another row's state; there are no "in-call" history rows (each row's history is its own ring / cache).
* **State addressing.** The state tensors of the call (the rings and caches, in their output order) describe row 0's
  buffers and are shape / dtype / offset carriers: the j-th state tensor of row r lives at
  `tab[r, index + j] + carrier.offset` (`carrier.offset` = the tensor's byte offset in row 0's buffer; every sequence's
  slots have the same layout). State tensors per op: `SconvRes` / `ConvMix`: the ring (j = 0). `InkKvPrologue`: k ring,
  v ring (j = 0, 1). `RelSinkAttn`: k cache, v cache (j = 0, 1). `InkAttnDecode`: k cache, v cache, k ring, v ring
  (j = 0..4).
* **Row table entries** (`Model::upload_row_table`): per slot two entries, as for `AttnKv` (k, v): an Inkling-2 `Ring`
  slot (`RingKind::InkKv` / `InkConv`) gets `(buffer address, buffer address)`, so both planes of a ring are reached by
  the one entry plus their carrier offsets, and `index = 2 * slot` everywhere. `InkAttnDecode` needs two slots
  (KV, then the K/V conv ring) behind one `index`: the loader guarantees `kvconv_slot == kv_slot + 1` (asserted at load),
  so entries `2*kv .. 2*kv + 4` are k cache, v cache, k ring, v ring. DeepSeek-V4.1's ring kinds and shared slots keep
  their "unsupported" error.
* **Checks** (`OpCall::check`, `inkling2::ops::check`): a row table on 69/70/71/74/75 needs `pos` `i32 [B]`,
  `rows == T`, `index + n_state <= stride`; `InkAttnDecode` keeps its 1..=16 rows. A linear cache's `pos0 + T <= Cr`
  check is the device-position one (skipped, as today with `pos`).
* **Registration.** A row-table kernel is registered `.row_table()` (selected only for calls with a table) and named
  `<base>_rows` so it takes its base kernel's admission (`Registry::effective_admission`). The plan's batch check counts
  a row-table call as B of the solo step's calls (`OpCall::row_weight`), so 69/70/71/74/75 join `plan::PER_ROW_OPS`.
* **Fused vs pair.** Chosen per the 1-row rule (`ink_attn_fused_rule(1, .., ctx)`) over the batch's largest context;
  both forms give the same bits, only time moves.

### 1.3 The 144 per-sequence slots in a batch

Nothing moves: every sequence keeps its own `SequenceState` (48 visits x (KV, K/V conv ring, h conv ring) = 144 slots).
The host writes one row table per step (`u8 [B, 2 * 144, 8]`, 36 KiB at B = 16) and the position vector; a captured
graph of the step can then replay with any set of sequences. Memory: the rings are per sequence and fixed size (about
170 MB per sequence on the real model: 40 SWA rings of 1040 x 1024 x 2 planes bf16, plus 48 x 2 conv rings), held once in
the plan's workspace today; `enable_batch` reserves ring bytes x (B - 1) out of the KV budget like recurrent state, and
shrinks the per-sequence full-attention KV capacity accordingly.

### 1.4 Model / scheduler / engine

* `run_layers(Seqs::Batch)` on an Inkling-2 model runs the visit loop with B rows; the stateful call sites go through one
  helper that, in a batch, swaps in row 0's carriers, the row table and the position vector (no fork of `ink_pass`).
* `enable_batch` accepts `Block::Ink` when every row-table Inkling-2 op the step runs has a registered `row_table` kernel
  (cpu: the decomposition reference below; Metal: the new `_rows` kernels); `row_tables_wanted` gets an Inkling-2 clause.
* Head parity: the solo step softcaps logits on the host (`softcap_logits`); the batched step's `LastLogits` rows and
  device top-k values get the same treatment (argmax is unaffected, the cap is monotone).
* A failed batched pass resets every Inkling-2 state of the batch (half-written rings), as `forward` does for one.
* Several rows of one sequence (batched speculative verify across streams, later): the row-table form would take a
  segment table (row -> (sequence, k-th row of its run)) and the in-call history rule of the solo T-row call within each
  run; nothing in the entry layout or positions above prevents it. Not built.
* Memory (main's note 1): `enable_batch` reserves rings x (B - 1) and divides the remaining KV budget by B x the
  full-visit bytes per token, refusing (`Error::Memory`) when fewer than 16 tokens per sequence would remain, as today.
* Engine / scheduler: nothing Inkling-specific (`engine/batch.rs` drives `forward_batch`); graphs of the batched step stay
  off for Inkling-2 (the solo graph path is not there either).

### 1.5 CPU reference of the row-table form

The cpu backend registers `cpu_f32_<op>_rows` (and `ref_<op>_rows`) for 69/70/71/74/75: the call is decomposed into B
T = 1 sub-calls (row r's slices gathered per plane into temporaries, row r's state tensors as views of the addressed
buffers, `pos0 = pos[r]`), each run by the ordinary kernel, results scattered back. Bitwise the solo row by construction;
it is the reference the Metal `_rows` kernels are tested against. `device_address` on cpu returns the buffer's host
address (owned buffers only); `supports_row_tables` stays false for every other model.

## 2. What was built

| Part | Where | What |
|---|---|---|
| Contract | `crates/intarsia-core/src/inkling2/ops.rs` ("Row tables"), `ops.rs` (`OpCall::check`), `plan.rs` (`PER_ROW_OPS`) | `ROW_TABLE_OPS`, `row_roles` (Row / Planes / State(j) / Fixed per tensor), `row_states`; a row-table call needs `pos` i32 `[B]`, `rows == T`, `index + states <= stride`; the five ops count as `B` solo calls in the plan's batch check. Calls without a table: unchanged code path. |
| Model | `model.rs` (`run_layers` Ink arm), `model/inkling2.rs` (`InkBatch`, `sx` / `rowtab` helpers), `model/batch.rs` | `Seqs::Batch` runs the visit loop over B rows; the 5 stateful call sites go through one helper that adds the row table + positions in a batch (identical call otherwise). `enable_batch` accepts `Block::Ink`, reserves rings x (B - 1); row-table entries `(buf, buf)` for Ink rings; softcap on batched logits / top-k values (solo parity); a failed batched pass resets every state of the batch; batched graphs off for Ink. A batch takes fused `InkAttnDecode` when the backend has no row-table pair (both give the same bits). |
| cpu | `crates/intarsia-backend-cpu/src/rowtab.rs`, `mem.rs` (`Raw` view, `address`), `lib.rs` | `ref_<op>_rows` / `cpu_f32_<op>_rows`: the call run as B one-row sub-calls on the addressed buffers (the reference of the form); registered after the base kernels (admission takes an op's first reference kernel). |
| Metal | `kernels/metal/inkling2_attn.metal` (`ink_attn_dec<GP, RT>`, `ink_attn_comb<RT>`), `inkling2_mix.metal` (`MX_ROWTAB`), `inkattn.rs` (`launch_ink_decode_rows`), `inkmix.rs` (`conv_mix_bf16_rows`), `dense.rs` (`dispatch_ind`) | Templated RT flag on the existing kernels: row r reads its caches / rings through the table (address in the params, resident via the indirect set), its own position, its own T = 1 scratch block; per-row split choice unchanged (grid over the most splits, the combine returns for a row with one split). The non-RT instantiations compile to the same code as before. |
| Recipe | `recipes/inkling2.toml` | `max_batch = 16` (the cap of `--max-batch`; without the flag one sequence at a time, as before). |
| Bench | `crates/intarsia-models/examples/batch_step.rs` | model name from the directory; `--ctx-spread N` (sequence i at ctx + i N). Asserts batched tokens == solo tokens every step. |

## 3. Tests

`crates/intarsia-models/tests/inkling2_batch.rs`, tiny fixture (window 8, rel extent 12, 2-layer blocks, 3 passes), cpu and
`INTARSIA_INKLING2_BACKEND=metal`; logits compared bitwise (softcapped) against the solo decode step of each sequence:

| Test | What | cpu | Metal | Metal, `INTARSIA_INK_SPLIT_KEYS=4` |
|---|---|---|---|---|
| `batch_equals_solo_2_5_16` | 2, 5, 16 sequences at lengths 1..64 (before / at / past the window, ring wrapped many times), 6 steps | pass | pass | pass |
| `batch_argmax_equals_solo` | device argmax rows | pass | pass | pass |
| `batch_membership_changes` | one finishes mid-batch, rejoin, shuffled row order, single-row step | pass | pass | pass |
| `batch_rollback_per_sequence` | two sequences cut back by length (3, 2) while the third continues | pass | pass | pass |
| `batch_failed_pass_resets_every_sequence` | injected failure in a row-table `ConvMix`: every sequence reset to 0, then prefill + batched decode bitwise again | pass | pass | pass |
| `batched_engine_equals_solo_engine` | `BatchedEngine` (4 slots, 9 requests at once, one seeded-sampled) token for token = `LocalEngine`; batched passes ran | pass | pass | pass |

`INTARSIA_INK_SPLIT_KEYS=4` forces several splits per row at tiny contexts, so the row-table combine and rows with different
split counts in one launch are exercised.

Other models (same branch): `intarsia-core` 232 + doc tests, `intarsia-backend-cpu` (admission quick sweep included),
`intarsia-models` `batch`, `batch_recurrent` (15, cpu), `batch_metal`, `inkling2` (8, cpu and Metal), `intarsia-backend-metal`
lib (kernel-list golden gains `ink_attn_decode_bf16_rows` only; `conv_mix_bf16_rows` is an Inkling op, outside the golden) and
its `inkling2_attn` (1), `inkmix` (3), `row_tables` (8), `rows_exact` (10 of 11) tests: pass (#10815). The 11th,
`rows_exact_registrations_are_pinned`, fails identically on the base a88db4620 (#10818): ink/m-mix's
`ink_in_proj_bf16_gemv_rows` is `rows_exact` but not in the pin list; not touched here (integration lane). `plan show --model ~/models/Qwen3-8B --backend metal`:
stdout and stderr byte-identical to a88db4620 (#10785).

## 4. Speed (Metal, endor, real checkpoint)

`batch_step` (`crates/intarsia-models/examples/batch_step.rs`), fleetq #10773 `--quiet gpu,cpu --mem 25`, checkpoint b85c3cfa
(W4A16, bf16), every sequence at context 1024 (+64 decoded), 4 rounds x 16 steps, solo and batched arms alternating;
`profile_every: 0`. Each step asserts the batched tokens equal the solo tokens (held for every B).

| streams | B solo steps ms | 1 batched step ms | speedup | solo agg tok/s | batched agg tok/s | per-stream tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.30 | 17.53 | 0.99x | 57.8 | 57.0 | 57.0 |
| 2 | 36.83 | 23.29 | 1.58x | 54.3 | 85.9 | 43.0 |
| 4 | 71.49 | 32.55 | 2.20x | 56.0 | 122.9 | 30.7 |
| 8 | 146.64 | 58.38 | 2.51x | 54.6 | 137.0 | 17.1 |
| 16 | 292.58 | 92.13 | 3.18x | 54.7 | **173.7** | 10.9 |

oMLX on endor at 16 streams: 94 tok/s aggregate; Fable's estimate for this engine: about 250 (perf-strategy section 7).
Step time grows 5.8 ms from 1 to 2 rows, then about 2.3 ms per row to 8 and 4.2 ms per row from 8 to 16: the 8 -> 16 step
is the one that does not amortise (perf-strategy section 7: the chunked row GEMVs read the weights twice above 8 rows, and
the expert union grows). Engine-path numbers through `intarsia serve --max-batch 16` (`bench speed`, localeval
`conc-plain-1k`): #10811, queued.
