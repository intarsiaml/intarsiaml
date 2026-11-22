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
- 2026-11-21 (coordinator): ink/metal now has m-mix 47570539e (merge 606fd36e4, tip f55bfe8b7), no default flips: re-merge
  before the opt-in A/B starts.
- 2026-11-21 (coordinator): when #10845 shows solo tok/s base vs branch within noise (<= 1 % slower, or faster), ink/batch at
  that tip merges into ink/metal (ink-metal does the merge); report #10845 with the tip hash as soon as it lands. A solo
  regression above 1 %: find it before anything else.
- 2026-11-21 (coordinator): hypothesis for the ~17 ms not in GPU time: host softcap 0.67 ms per 200058-wide row (ink-hostgap)
  plus a host full-vocab pass ~0.2 ms+, per row, if the batched step downloads B logits rows and softcaps / argmaxes them on the
  host: 16 x ~1 ms. (1) Check from code what happens per row on the host in a greedy batched step. (2) If so, greedy rows on
  the device (the solo path's softcap + argmax per row, one small readback of B ids). (3) Sampled rows keep host logits; the
  per-row softcap on the host pool waits for hostgap's a2f2b87b9. Report before building; record this brief.
- 2026-11-21 (coordinator, rule for all Inkling lanes): this session and every worktree run ON endor; a local cargo
  build / check / test or shader compile loads endor's CPU and skews every quiet timing job there (a possible cause of noisy
  A/Bs). No local builds or tests on endor outside a fleetq slot. macOS compile checks: bespin
  (`fleetq run --host bespin --load cpu --mem 6 --expect <min>`, a `git archive` copy in ~/intarsiaml-ink-batch, toolchain in
  ~/.cargo/bin; no GPU timing or real model there). CPU tests: spark2. GPU work: endor fleetq jobs. Reading, git and editing
  locally are fine.
  Lane audit: every build and test of this lane ran inside an endor fleetq slot except one `cargo check -p intarsia-core`
  at 21:1x, before any timing job of this lane. Every quiet timing job queued now (#10900 A/B, solo A/B, #10811) builds inside
  its own slot before timing, so nothing outside a slot overlaps it.
- 2026-11-21 (coordinator): in ink-metal #10890 `intarsia bench speed` failed at its calibration call ("no tokens received")
  against the Metal serve, while chat and localeval streaming worked; ink-metal is fixing it. #10946 must not end on that
  failure: merge their fix before #10946 builds (hash to follow) or skip the bench step. Done: #10946 now runs localeval
  `conc-plain-1k` first, then the bench as a non-fatal step with a 30 min watchdog. Solo baseline from #10890: at
  `--max-batch 1` `conc-plain-1k` is serialized, 50.8-51.6 tok/s aggregate at 1 / 4 / 16 streams; the `--max-batch 16` run is
  compared against it.
- 2026-11-22 (coordinator): the `bench speed` calibration fix is ink/metal b9dd0d6b4 (crates/intarsia-bench/src/speed.rs only;
  Inkling's first generated token is a block marker the text stream drops, so a max_tokens 1 probe returned usage only and
  counted as "no tokens received"). Cherry-pick it before #10946 builds; keep the non-fatal bench step. Done: cherry-picked as
  6b5ff520f before #10946 started (it builds the tip at job start); the bench step stays non-fatal.
- 2026-11-22 (coordinator, after #10914): ink-metal merges ink/batch bbc0bec00 into ink/metal after #10901, right after
  ink/m-mix 52b337aa9 (both touch the mix shader), and messages the merge hash. (1) Get the 0.6 % back: from that ink/metal
  tip make the row-table branch compile-time (Metal function constant or a separate pipeline variant), so the solo path
  compiles without MX_ROWTAB; shader / pipeline only (ink-dfuse owns the inkmix.rs row kernels for fusion: touch no more
  than needed). (2) Correctness: rows-exact and batched == solo at B 2 / 5 / 16, run on endor in a default-load slot; macOS
  compile checks on bespin; no local builds. (3) Timing: if the two-pass curve job has not started, fold a solo ABAB
  (merged tip vs variant, 6 reps, --profile-every 0) into it; else one short quiet job. (4) Merge criterion: solo >= the
  merged tip and batched output identical; report the hash.
- 2026-11-22 (ink-metal, coordinator): ink/m-mix merged into ink/metal at 331238a4e, ink/batch bbc0bec00 at 45887198b, tip
  8546d143f (docs only); no textual conflicts. m-mix changed `mix_geom` (NE 4 from D 2048) and `BLEND_RB` 16 -> 4, so the
  row-table mix path now runs at NE 4 (md5s differ from before: reduction order). Merge gate #11048 (default load) reruns
  tiny B 2 / 5 / 16, rows_single, rows_exact, real-model batch_step mixed contexts, G2 / G3. Branch the compile-time
  MX_ROWTAB variant from 8546d143f; its solo ABAB's base arm is 8546d143f (not f55bfe8b7).
- 2026-11-22 (coordinator): pid 53955 on 8099 was an orphan of ink-dfuse's finished #10882; killed (with parent 53944),
  8099 free. Keep the port-pick + owner check; no other change.
- 2026-11-22 (coordinator, relaying ink-dfuse): ink-dfuse's hoist switch (INTARSIA_INK_MIX_HOIST) picked x vs ring from the
  row index instead of the in-call index + xoff under row tables; fixed on ink/dfuse ac2371c81 (contiguous path unchanged).
  Run the row-table rows-exact tests and tiny B 2 / 5 / 16 batched == solo with INTARSIA_INK_MIX_HOIST=1 against ink/dfuse
  ac2371c81 in a default-load slot (folded into #11056 if not started, else after ink/dfuse merges into ink/metal); report
  pass / fail to the coordinator and ink-dfuse (a194e917d8f92b542).
- 2026-11-22 (coordinator, on the B 8 / 16 regression A/B): the same m-mix merge (NE 4, BLEND_RB) is the main suspect in
  ink-metal's decode-mode G2 regression on the merged tip (teacher-forced decode rows 2.6e-3 -> 1.1e-2, drift growing row to
  row: looks like 1-row decode writing bad state); their bisect #11142 runs NE=3 / BLEND_RB=16 arms. Send the five-arm B 8 / 16
  table to ink-metal (af43898d63e5b5190) and the coordinator. If ink-metal reverts NE 4 for accuracy: rerun B 8 / 16 on the
  fixed tip first (a batched-only override may be moot). If accuracy is fine at NE 4 and only batched speed needs NE 3 /
  RB 16: approved, a batched-only override in mix_geom / blend_geom, written by ink-dfuse (a194e917d8f92b542, owns the
  inkmix.rs row kernels now; m-mix lane closed): send dfuse the exact lines, solo defaults stay. Solo rtvar: inconclusive,
  stays unmerged; retry in a later quiet slot only if cheap (0.6 % at best).
- 2026-11-22 (coordinator, relaying ink-hostgap): a side worktree whose target/ was cloned (cp -cR, APFS clone) keeps the
  origin's mtimes, so cargo can keep old crates as fresh and build stale code. If a target was cloned: touch every tracked
  file under crates/ before each build (or cargo clean -p intarsia-core -p intarsia-backend-metal), and each arm's log must
  show those crates compiling; for the HOIST=1 check, confirm the hoisted kernel dispatched.
  Done: `cp -cR` here preserves mtimes (checked). Every side target of this lane was cloned (ink-batch/target from ink-metal
  at the start; base-, rtvar-, dfuse-target from it). The queued five-arm A/B and #11104 now touch all tracked files under
  crates/, kernels/, recipes/ of each tree before its build and grep the `Compiling intarsia-*` lines; #11104's dfuse tree
  carries an uncommitted scratch-only `eprintln` that names the first hoisted mix dispatch with and without a row table
  (tests run with --nocapture). Exposure of past results: ink-batch/target was cloned before this lane's first build, from
  a newer ink-metal tip, so the very first tiny runs (before the 21:5x ink/metal merge) may have mixed in ink-metal code
  (they were all rerun after the merge); #11057's rtvar build (its target cloned at job start, 15 s build) is treated as
  unverified: the A/B's arm C rebuilds it from touched sources.
- 2026-11-22 (coordinator, build hazard 2 from pack12 #11066): `cargo test -p intarsia-cli --test <x>` rebuilds
  target/<profile>/intarsia without `--features metal` over the backend binary; later CLI steps fail ("built without the
  Metal backend") and md5 of two empty outputs reads SAME. Rules: build the backend binary last (or tests in their own
  --target-dir); count md5 / self-golden matches only when both runs exit 0 with non-empty output. Reply only if affected.
  Checked: no script of this lane runs `cargo test -p intarsia-cli`; intarsia-models has no metal feature (the backend is a
  plain dependency), so its test builds do not change batch_step. Past matches all had non-empty output and exit 0 (G2:
  EXIT 0 x 4, tensors compared one by one; generate outputs 1025 / 946 bytes). Not affected. Hardened anyway: #11104 compares
  digests only when both runs produced all three, and the timing job's text comparison requires non-empty files.
- 2026-11-22 (coordinator): (1) when #11104 starts, the coordinator runs ink-metal's G2 bisect inside its slot
  (FLEETQ_JOB=11104, ~25 min, 20 GB, correctness only): do not release #11104 early; extra GPU load during it is expected
  (bitwise / correctness tests only). (2) The five-arm regression hunt #11164 hunts the same suspects (m-mix NE 4 / BLEND_RB
  16 -> 4, InkMixP 112 -> 144): when the bisect lands (~04:15) read its table in
  /opt/code/intarsiaml/.worktrees/ink-metal/.scratch/bisect_job.log; if it names the culprit, cut #11164 to the arms it does
  not settle for batched B 8 / 16, or cancel it.

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

## 5. Single-read 9..16-row dense GEMVs (opt-in)

Above 8 rows the row-exact bf16 GEMV (`matmul_bf16_gemv_rows`: Inkling-2's wo) and the logits GEMV (`logits_bf16_gemv_rows`: the
unembed, the router) ran as two chunk launches (`row_chunks`), so their weights came from DRAM twice. The MoE experts were
already read once per step for all rows (`Policy::Auto`: union gate+up and union down above one row) and InkInProj already
takes 1..=16 rows in one dispatch, so these two are the only re-reads.

`<base>_mr8t<T>` (kernels/metal/gemv_multirow.metal, T = 1..8, families 16-byte / 8-byte bf16 and f32-out logits): one launch of
8 + T rows, as an 8-row pass and a T-row pass inside each threadgroup (simdgroup for logits) on the same weight row, the
weights read from DRAM once. Each (weight row, activation row) accumulator is the mirrored 1-row kernel's (same pass helpers,
lane mapping and reduction order). Opt-in `INTARSIA_METAL_ROWS_SINGLE=1` (default off; only 9..=16-row calls of those two
kernels; `with_rows_single` test hook). G-B: `crates/intarsia-backend-metal/tests/rows_single.rs`, 9..=16 rows, wo shape on
both load families, a K off the 16-byte block, logits at N 8195 and 128: every row bitwise the one-row launch and the two-chunk
launch (#10877, pass); `rows_exact` 11 / 11 and the backend lib tests 18 / 18 in the same run.

Expected gain: about 0.9 GB of re-read per step each for wo (48 visits) and the unembed, about 3 ms at 16 rows. A/B on the
batched step: queued (quiet slot).

## 6. Real-checkpoint correctness (#10844, default gpu load)

* **Solo path unchanged**: `intarsia accuracy --backend metal --max-seq 4096 --write-self-golden` on the G2 oracle goldens,
  base a88db4620 vs this branch (bf6eac934 + docs): every tensor of the self-goldens bitwise identical (short 48 / 48,
  long 60 / 60; the files differ only in the `built_unix` / `self_meta` metadata). G2 short KL21 2.380e-3 (gate 4e-3), the
  integration lane's figure; both arms exit 0 on short and long.
* **Batched == solo on the real model at mixed positions**: `batch_step --ctx 600 --ctx-spread 150` (sequence i at
  600 + 150 i: 600..2850, so the 1040-row SWA rings of the later sequences have wrapped while the early ones have not, and the
  full visits read up to 2.9k positions), B = 2, 5, 16, 8 steps: batched tokens == solo tokens every step (asserted).
  Its timings were taken under a shared (non-quiet) slot and are not reported as speed.
* Tiny tests on the same build: cpu 6 / 6, Metal 6 / 6; `rows_exact_registrations_are_pinned` pass after the ink/metal fix.

## 7. Engine path (serve)

First attempt #10811 (22:26) exited 1: `serve --max-batch 16` refused at load, `moe_experts` without an admitted kernel at the
batch phase's shapes (`Batch t=10..`). It ran before the merge that brought ink/metal 761a1561b (the plan-shape SiTU case,
merged at 22:27:45) and on the old script that used a prebuilt binary: the known failure, not a batch-specific one. Log kept
in `.scratch/serve_batch-10811.log` / `serve-10811-prefix/`. Rerun queued with the script that builds the tip at job start; a
default-load admission probe (`serve --max-batch 16` on the tip until `/v1/models` answers, one chat request) runs first.

## 8. Where the 8 -> 16 step goes (#10885, kernel-mode trace, default load: shares only)

`batch_prof --ctx 1024`, `INTARSIA_METAL_PROFILE=kernel` (one command buffer per launch, so absolute times are inflated and
launch-count changes are over-weighted), cut by step marks, median over 10 steps; GPU ms per step:

| op | B 8 | B 16 | delta | launches at 16 |
|---|---:|---:|---:|---:|
| matmul_bf16_gemv_rows (wo, two chunks above 8) | 2.87 | 6.50 | +3.63 | 96 |
| ink_in_proj_bf16_gemv_rows | 8.19 | 11.80 | +3.61 | 48 |
| moe_experts_situ_bf16_rows (gate+up / down / gather) | 29.05 | 33.91 | +4.86 | 132 |
| ink_attn_decode_bf16_rows | 10.59 | 12.08 | +1.48 | 48 |
| logits_bf16_gemv_rows (unembed) | 3.47 | 4.63 | +1.16 | 2 |
| logits_bf16_gemv_rows (router, two chunks above 8) | 1.06 | 1.83 | +0.76 | 96 |
| moe_experts_mxfp8 (the bf16 / mxfp8 visits) | 3.06 | 4.37 | +1.31 | 12 |
| everything else | 9.73 | 9.02 | -0.71 | |
| **sum** | **67.0** | **84.1** | **+17.1** | |

Summed GPU time grows 17 ms while the real step (`batch_step`, quiet) grows 34 ms (58.4 -> 92.1). Before choosing a kernel
lever the command-buffer timeline (GPU busy vs wall per step, B 2 / 8 / 16) is measured, to tell real GPU growth from host
gaps (queued). The coordinator's InkInProj figure (+4 ms) matches this table; it is one of three similar shares, not the
largest (the MoE situ path is).

## 9. A/B of `INTARSIA_METAL_ROWS_SINGLE` (#10900, quiet, tip 76cedfc93)

`batch_step --ctx 1024 --rounds 2 --steps 16 --batches 8,16`, arms alternating off / on, 2 reps; batched step ms
(aggregate tok/s). Batched tokens == solo tokens in every arm (asserted, exit 0), so the one-launch form is also exact on the
real model.

| rep | B 8 off | B 8 on | B 16 off | B 16 on |
|---|---:|---:|---:|---:|
| 1 | 39.18 (204.2) | 38.03 (210.4) | 71.20 (224.7) | 70.33 (227.5) |
| 2 | 39.47 (202.7) | 38.21 (209.4) | 78.45 (204.0) | 78.28 (204.4) |

The flag cannot change a B = 8 step (it only acts on 9..16-row calls), yet B 8 "on" is 1.1-1.3 ms faster in both reps: that is
the run-order bias of this A/B (on always runs second). The B 16 difference (0.9 and 0.2 ms) is inside it. **No measurable
gain; the flag stays opt-in and off.** (Expected about 3 ms; the re-read is mostly served from the SLC, it seems.)

**Bigger finding: the whole curve moved.** Against #10773 (section 4, ink/metal 1dacac3a8-era code), the same harness at this
tip gives solo 61 tok/s (was 57.8), B 8 39.2 ms (was 58.4), B 16 71.2 ms = **224.7 tok/s** (was 92.1 ms = 173.7) in rep 1;
rep 2 of B 16 is 78 ms (204 tok/s), a 10 % spread between reps of the same setting. The likely cause is the merged ink/metal
work since then, chiefly m-mix 47570539e (InkInProj up to 8 rows on the GEMV instead of the matrix units; the 8-row InkInProj
was 8.2 ms per step in #10885); unrelated local CPU load on endor during #10773 (the coordinator's build rule) may also have
inflated it. A full 1 / 2 / 4 / 8 / 16 curve at the tip, two passes, is queued to replace section 4's table.

## 10. Solo path speed: ink/metal f55bfe8b7 vs ink/batch 177cfd2f5 (#10914, quiet)

`intarsia generate --backend metal --max-seq 4096 --max-tokens 256 --temperature 0 --profile-every 0`, both built in the slot,
6 reps per arm alternating order (base/new, new/base, ...); output text identical.

| arm | mean tok/s | median | min | max |
|---|---:|---:|---:|---:|
| base ink/metal f55bfe8b7 | 64.13 | 64.09 | 63.92 | 64.40 |
| ink/batch 177cfd2f5 | 63.73 | 63.64 | 63.47 | 64.11 |

Branch -0.62 % (mean), -0.70 % (median): inside the <= 1 % merge criterion, but consistent (the branch is slower in 5 of 6
pairs), about 0.1 ms per token. Suspects, all in code the solo step runs: the `MX_ROWTAB` runtime branch and the now-mutable
ring pointer in the mix kernel's conv loop, the 32 bytes added to `InkP` / `InkMixP` (set_bytes), and the per-call
`row-table` checks in `ink_pass` / `OpCall::check`. Not chased here (inside the criterion); a compile-time row-table variant of
the mix shader would remove the first two if it matters.

## 11. Host gap of the batched step (#10956, default load: shares only) and the serve probe (#10947)

`batch_prof --ctx 1024`, `INTARSIA_METAL_PROFILE=cb` (command-buffer timeline; the trace costs time itself, so the wall
times are above the untraced run beside each), GPU busy = union of the command buffers inside each step, median of 12:

| B | wall ms | GPU busy ms | host gap ms | command buffers / step |
|---:|---:|---:|---:|---:|
| 2 | 30.91 | 30.45 | 0.46 | 7 |
| 8 | 43.19 | 42.54 | 0.65 | 7 |
| 16 | 83.05 | 82.22 | 0.83 | 7 |

The batched step is GPU-bound: under 1 ms of host gap at every B. So the 17 ms that the kernel-mode trace (#10885) did not
account for between B 8 and B 16 is GPU time the per-launch trace under-counts (one command buffer per launch changes
overlap and cache state), not host work; the coordinator's host-softcap hypothesis is out on this evidence too (greedy rows
never come back as logits, section 8 note). The levers above 8 rows are GPU kernels.

Serve probe #10947 at ink/batch 38e487a5d: `intarsia serve --backend metal --max-batch 16 --max-seq 8192` starts in 62 s
("continuous batching: up to 16 sequences (--max-batch), busy prefill chunk 256") and answers a chat request; the admission
refusal of #10811 is gone with ink/metal 761a1561b.

## 12. #10946 is invalid: it measured another lane's server

#10946's own `intarsia serve` loaded, then failed to bind (`bind 127.0.0.1:8099: Address already in use`, supervisor exit
78). Port 8099 was held by ink-dfuse's `intarsia serve ... --max-batch 1 --port 8099` (pid 53955, running since 23:07), so
the script's readiness probe answered and localeval `conc-plain-1k` (50.8 / 52.1 / 52.6 tok/s at 1 / 4 / 16 streams,
deepeval results/20261122-004010-adhoc-endor-intarsia-metal-ink-batch, unanimis 9be1ea95) measured that `--max-batch 1`
server, not this lane's: **not a `--max-batch 16` result.** (It matches #10890's serialized 50.8-51.6.) Peak RSS in the log
is of the failed start. Fix: the script now takes a free port from 8131 up, and after readiness checks that the listener is
this job's binary (and still the same pid before the bench); rerun queued. The serve probe #10947 (port 8097, "listening on
http://127.0.0.1:8097" in its own log) was its own server and stands.

## 13. Compile-time row-table mix variant (ink/batch-rtvar) timing (#11057, quiet)

Variant: ink/batch-rtvar fc76eb318 (+ 2b740dee7, `batch_step --digest`, example only) on the merged ink/metal tip 8546d143f.
`ink_mix_row<NE, NS, QB, RT>`: the solo instantiations (`ink_mix_row_e*`, RT = false) carry no row-table code, the batched
calls take `ink_mix_row_rt_e*`; host: `mix_row_shader(.., rt)` and the two dispatch sites only. bespin compile check: Rust
compiles; bespin's Metal toolchain cannot build the shaders (empty metallib, `attention.metal` fails there), so the shader
compile is checked on endor (#11104).

**Solo ABAB** (6 reps each, alternating, 256 tokens, `--profile-every 0`; output text identical):

| arm | mean tok/s | median | min | max |
|---|---:|---:|---:|---:|
| merged tip 8546d143f | 62.81 | 62.77 | 61.60 | 63.59 |
| rtvar 2b740dee7 | 62.56 | 62.81 | 61.31 | 63.14 |

Inconclusive: median +0.06 %, mean -0.4 %, and this run's rep spread (3 %, both arms) is several times the 0.6 % being
recovered (#10914's spread was 0.8 %). The whole box was slower in this slot (both arms about 2 % under #10914).

**Curve on rtvar** (2 passes, ctx 1024, batched tokens == solo tokens every step):

| B | pass 1 step ms | agg tok/s | pass 2 step ms | agg tok/s |
|---:|---:|---:|---:|---:|
| 1 | 16.94 | 59.0 | 16.42 | 60.9 |
| 2 | 22.12 | 90.4 | 21.34 | 93.7 |
| 4 | 28.80 | 138.9 | 27.89 | 143.4 |
| 8 | 48.88 | 163.7 | 46.26 | 172.9 |
| 16 | 79.10 | 202.3 | 80.79 | 198.1 |

B 8 is 18-25 % slower than #10900 at ink/batch 76cedfc93 (39.2 ms) and B 16 about 12 % (71.2 ms in #10900 rep 1; rep 2
was 78), while solo moved only about 2-4 %. The batched step got slower between ink/batch 76cedfc93 and the merged tip (or
with the variant). Candidates: m-mix's new defaults (mix_geom NE 4 from D 2048, `BLEND_RB` 16 -> 4) applied to the row-table
mix path, other ink/metal changes in the merge (25 files), or rtvar itself. Quiet A/B queued: ink/batch vs 8546d143f vs
8546d143f with `INTARSIA_INK_MIX_NE=3` / `INTARSIA_INK_BLEND_RB=16` vs rtvar, B 1 / 8 / 16, forward then reversed order.
