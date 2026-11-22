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
