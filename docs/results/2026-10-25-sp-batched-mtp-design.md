# sp: batched MTP speculation on the Qwen3.8-27B (HIP, R9700) - design note

Lane sp, 2026-10-25. Design only: no code, no fleet slot. Base: `origin/main` 5241377b (contains #142, #143 and mr's #145 MXFP4 row-exact GEMV).
`--mtp-draft-vocab` stays opt-in (the user's call); this note does not touch it.

## 0. Summary

**Revision 2 (same day).** The first version of this note fitted the batched plain-decode cost on localeval aggregates that include prefill, and from that
inferred a 4.9 ms per-row `lm_head` cost and recommended fixing it first. mr's measurement (#147, unmerged, `docs/results/2026-10-25-mr-hip-27b-batch-profile.md`)
refutes that: the unprofiled 27B step is 30.3 / 36.8 / 39.2 / 42.1 / 59.5 / 82.7 / 110.9 ms at 1 / 2 / 3 / 4 / 8 / 12 / 16 rows, and `logits_bf16_gemv_rows` is flat at
4.3 to 4.4 ms up to 8 rows, with its cliff only at 9 to 16 rows (7.8 ms at 12, 15.0 at 16). **There is no per-row head cost to fix below 8 rows, so the "head
fix first" recommendation is withdrawn**; the 9 to 16 row `lm_head` kernel is mr3's work (fleetq #5287/#5288) and is not duplicated here. Section 2 is
redone on mr's measured curve.

- **Feasible, and most of the machinery exists.** The single-sequence step already has everything batched speculation needs: a row-exact verify
  pass (`Model::forward_verify`), per-row verify attention on HIP, per-row recurrent state saves and rollback (`SpecRows`,
  `SequenceState::spec_rollback`), a per-sequence MTP head (`SequenceState::mtp`), and a batched decode path whose row-exact kernels are proven for the
  27B (`Model::forward_batch`, mr's `batch_cuda_real` pass at N = 2..16). What is missing is the glue: a ragged multi-sequence verify pass, a batched
  draft head, a scheduler step that carries drafts, and `load.rs` stops refusing `--spec mtp` with `--max-batch > 1`.
- **The gain is real but narrow: 2 to 4 streams.** Modelled against the measured batched step (section 2): about **+48% aggregate at 2 streams, +27% at 3, +13% at
  4, +5% at 5, nothing from 6** (k=2 at 2 to 3 streams, k=1 at 4 and 5). A 10% lower acceptance takes 2 streams to +33%, 4 streams to +2%. A flatter lm_head at 9 to
  16 rows (mr3) only changes the 8-stream and wider cases, which stay below 1.0, so it does not extend the range. Above 5 streams the verify rows cost more GEMV
  time (`R(8) = 1.94`) than the tokens they buy; that is a property of the GEMV, and the break-even moves up only with a flatter small-row MXFP4 GEMV (the L1 lane).
- **Recommendation:** worth building as a B <= 4 feature (45 to 55 engineering hours) only if 2 to 4 concurrent streams on the 27B is a case the user cares
  about; it is a +50% aggregate there and nothing at 8 and 16 streams, where Radiance's short-draft gain has no counterpart on this GEMV. W0 (a 45-minute window) now
  only has to re-measure solo MTP after mr and read the state size; the "per-row head" measurement it was meant to take is answered by mr's data.

## 1. What exists today (code facts)

| piece | where | what it does now |
|---|---|---|
| refusal | `crates/intarsia-models/src/load.rs:292` | `--spec mtp` with `--max-batch > 1` returns `Unsupported` before weights are read |
| scheduler hook | `scheduler.rs` `BatchSeq::prepare(model, slot, spec_allowed, busy)` | a speculating sequence may run its own verify on its own slot and return `SeqStep::Ready`; `spec_allowed = B < spec_off_at` (default 4); a verify pass here is "a whole extra weight pass beside the batched step" |
| multi-token runs | `engine::Seq` `pending` + `SeqStep::Ready` | a verified run's tokens are delivered one per scheduler iteration |
| single-sequence step | `Model::spec_step` (captured) / `mtp_draft` + `forward_verify` (eager) | draft (catch-up over n queued rows, then k-1 chain rows) and verify (k+1 rows, row-exact kernels, per-row attention on HIP) |
| state rollback | `SequenceState::spec_rollback(keep)` | KV `len` shortens; each recurrent slot copies the state saved after verify row `keep-base-1` from `SpecRows.bufs`; the head's KV truncates and its pending queue is recomputed |
| batched decode | `Model::forward_batch`, `Seqs::Batch`, `upload_row_table` | one row per sequence, row-exact kernels, bitwise the solo step; 27B on HIP proven for N = 2..16 (mr, `docs/results/2026-10-25-mr-hip-mxfp4-rows.md`) |
| MXFP4 rows GEMV | mr #145 | `matmul_mxfp4_gemv_rows`, M = 1..16 (`ROWS_MAX`), row `m` bitwise the one-row launch; 17408x5120: 80 us at M=1, 87 (M=3), 93 (M=4), 156 (M=8), 272 (M=16) |

Measured 27B facts this note leans on (all from the results docs; the ones marked *model* are fitted, not measured):

- solo spec off 32.8 to 33.1 tok/s (30.5 ms/token); solo MTP k=2 pass 53.5 ms, 2.44 to 2.83 tokens/pass at 77 / 72 / 92 / 87% acceptance
  (128 / 1k / 4k / 16k), i.e. 1.39x to 1.60x (sp doc sections 2 and 4); k=3 is slower than k=2 (71 ms per pass).
- lm_head 248320 x 5120 bf16: 4.25 ms per sweep, flat for M = 1..4 (sp doc section 5; M > 4 not measured).
- 27B batched plain decode on HIP after mr: the unprofiled step is 30.3 / 42.1 / 59.5 / 110.9 ms at 1 / 4 / 8 / 16 rows (mr #147, table in section 2.1); the first version of this
  note used localeval aggregates here, which include prefill, and was wrong.

## 2. Pass-cost model and expected gain

### 2.1 Inputs (measured)

mr, #147 (draft, measurement only), 27B MXFP4, one R9700, 1k context, after #145, unprofiled:

| rows B | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| plain batched step, ms | 30.3 | 36.8 | 39.2 | 42.1 | 46.2 | 50.4 | 59.5 | 82.7 | 110.9 |
| aggregate tok/s | 33.0 | 54.4 | 76.6 | 95.0 | 108.1 | 119.2 | 134.5 | 145.1 | 144.3 |

Per-op (profiled; B = 2 / 4 / 8 / 12 / 16): MXFP4 pair GEMV 21.4 / 23.2 / 31.7 / 42.6 / 52.6 ms, MXFP4 GEMV 12.6 / 14.4 / 18.8 / 24.4 / 30.3, `logits` 4.26 / 4.29 / 4.42 / 7.78 / 15.0,
`gdn_delta_rule` 2.2 / 2.9 / 4.8 / 6.8 / 8.8 (0.55 ms per sequence, at the state byte floor), `attn_decode` 0.9 / 1.3 / 2.0 / 2.8 / 3.6, everything else 9.1 to 9.9.
Solo (sp doc): spec off 30.5 ms/token; MTP k=2 pass 53.5 ms with the old GEMV, 2.44 to 2.83 tokens/pass.

### 2.2 The model

```
plain(B)     = mr's measured step (interpolated)
verify(B,q)  = plain(M) - 0.75*(M-B) + B*((q-1)*0.26 + 0.3)            M = B*q rows; a verify row has no sequence of its own (GDN state + attention
                                                                         0.75 ms per sequence), pays a state save per extra row (149 MiB write = 0.26) and
                                                                         ~0.3 ms expected rollback copy per sequence
draft(B,k)   = headlayer(2.5*B rows) + lm_head(B) + (k-1)*(headlayer(B) + lm_head(B))        head layer 2.4 ms flat to ~6 rows; lm_head from mr's curve
spec(B,k)    = draft + verify(B,k+1)
gain(B,k)    = tau(k) * plain(B) / spec(B,k)       tau = 1.8 / 2.5 / 2.95 tokens per pass at k = 1 / 2 / 3 (measured k=2 mean 2.44 to 2.83, rounded down)
```

Checks (not claims): at B = 1, k = 2 the model gives a 51.8 ms pass and gain 1.46, against the measured 1.39 to 1.60x with the old kernels (the GEMV improvement
of #145 is inside `plain(3)` here; solo MTP after mr is not yet measured, W0 does that). Sensitivity: the per-sequence term 0.75 varied over 0.4 to 1.2 moves
the 2/4/8-stream gains by at most 0.04.

### 2.3 Expected gain (modelled; k per row = best uniform with `B*(k+1) <= 16`)

| streams | plain step, ms | plain agg tok/s | best k | gain | spec agg tok/s | gain at tau -10% |
|---|---|---|---|---|---|---|
| 1 | 30.3 | 33 | 2 (3 ties) | 1.46 | 48 | 1.32 |
| 2 | 36.8 | 54 | 2 | **1.48** | 80 | 1.33 |
| 3 | 39.2 | 77 | 2 | 1.27 | 97 | |
| 4 | 42.1 | 95 | 1 | **1.13** | 107 | 1.02 |
| 5 | 46.2 | 108 | 1 | 1.05 | 113 | |
| 6 | 50.4 | 119 | none | 0.99 | | |
| 8 | 59.5 | 134 | none | 0.88 | | |

- Speculation helps while the extra verify rows are almost free: the GEMV at M = 4 to 6 is 1.16 to 1.58x the one-row time, so a k=2 pass at B = 2 (6 rows) streams the
  weights once for 2.5 tokens per sequence. By M = 8 the GEMV costs 1.94x and the extra rows no longer pay; Radiance's `--draft-short-from 8` gain
  (+12% at 8, +18% at 16) comes from a GEMV that stays flat there. With mr3's `lm_head` fix (flat to 16) the 4-stream gain moves from 1.13 to 1.15 and the 8-stream
  one from 0.88 to 0.96: still no gain.
- **Short-draft policy:** k=2 up to B = 3, k=1 at B = 4 and 5, plain from 6 (`row_budget` 8, `spec_off_at` 6; both knobs). The scheduler's default `spec_off_at` is 4; it
  would be raised to 6 for MTP only when this is built and measured.
- The numbers are modelled from op-level data; the measured claim to check in W2 is the 2-stream +45 to +50% and the 4-stream +10 to +15% (localeval aggregate, spec on
  against main's batched spec off, same window).

## 3. Design

### 3.1 Scheduler interface

New work for the scheduler is one step kind and two model calls. `prepare` stays the per-sequence planning hook, so the scheduler's structure (plan every
row, one model step, accept in admission order) is unchanged.

```rust
pub enum SeqStep<Mode> {
    Plain(u32, Mode),
    Ready,
    /// Speculate: `next` is the token to feed; draft up to `k` tokens and verify them in the batched pass.
    Spec { next: u32, k: usize },
}

pub struct SpecRow { pub slot: usize, pub next: u32, pub k: usize }      // k = 0 is a plain row (argmax)
pub struct SpecOut { pub drafts: Vec<u32>, pub verify: Vec<u32> }        // verify.len() == drafts.len() + 1

trait BatchModel {
    fn spec_capable(&self) -> bool { false }                              // true for the HIP hybrid 27B/35B with an MTP head
    /// Draft and verify for all rows in one batched pass; each slot grows by (accepted+1) after `accept_spec`, not here.
    fn spec_step(&mut self, rows: &[SpecRow]) -> Result<Vec<Result<SpecOut>>>;
    /// Shorten `slot` to `keep` tokens inside the last `spec_step` (KV length, recurrent state from SpecRows, head KV + queue).
    fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<()>;
}

trait BatchSeq<M> {
    /// Host-side acceptance on this sequence's own output (the same function the single-sequence engine uses), pushes the
    /// emitted tokens onto its pending queue, calls `model.spec_rollback(slot, keep)` for the rejected rows. Returns the token count.
    fn accept_spec(&mut self, model: &mut M, slot: usize, out: SpecOut) -> Result<usize>;
    fn prepare(...)  // may now return SeqStep::Spec
}
```

Scheduler changes (`decode_phase`):

1. `prepare` as today; plain rows and `Spec` rows are collected separately. If any `Spec` row exists *and* the model is `spec_capable`, plain rows join as `SpecRow { k: 0 }`
   (a `k = 0` row is the decode row of today, so one pass serves all); otherwise the old path (`decode_step`) runs unchanged. With no speculating
   sequence in the iteration nothing new runs, which makes the no-op gate with spec off a code-path statement and not only a measurement.
2. `spec_step(rows)`; per-row inner `Err` fails that request alone (as `decode_step`), outer `Err` is fatal (design 2.5, unchanged).
3. For each row: `accept_spec` (acceptance + rollback), then the existing `token_done` loop, repeated while the sequence still has pending run tokens, so
   a run is delivered in the iteration it was verified and the sequence is not left out of step with the others (today's one-token-per-iteration drain
   would make sequences drift out of phase and leave `Ready`-only iterations with fewer rows). A stop, `max_tokens` or a cancelled client ends the
   loop; `finish(.., cancelled)` already carries the contract "roll back anything unaccounted in `slot` (speculative rows)", which covers a stop in the middle of a run.
4. `spec_allowed = B < spec_off_at` becomes a *row budget* decision: `k_i` is chosen in `prepare` from `(B, budget)`; `spec_off_at` remains as a hard
   ceiling and default (4, matches the table above).
5. Tokens never depend on `k_i`: verify is exact. This is the same latitude the `busy` and `spec_allowed` arguments already have, and it does not break the
   trait's exactness contract (a function of the request, `cached` and the outputs handed to it): the *schedule* depends on the batch, the *tokens* do not.

`load.rs` lifts the refusal for backends that report `spec_capable` (HIP hybrid) and keeps it for the rest (CUDA, Metal, CPU stay refused in v1; section 3.8).

### 3.2 How verify rows mix across sequences

One flattened row list, `rows = ⊕_i [next_i, d_i1 .. d_ik_i]`, `M = Σ(k_i+1) <= row_budget <= ROWS_MAX = 16`. Every op sees the `M` rows:

- **Dense projections, MLP, norms, embeddings, lm_head:** plain multi-row kernels in `exact_rows` mode, i.e. mr's row-exact GEMV family; row `m` is bitwise the one-row
  launch, which is what makes row mixing safe. No sequence-aware code: these ops do not know row ownership. The `lm_head` runs once over `M` rows
  (`verify_head`-style), then `Argmax` per row, and one download of `4*M` bytes.
- **Row table:** the `upload_row_table` machinery gets a second form: entry `r = (seq_index, position)` with consecutive positions per sequence, replacing
  the one-entry-per-sequence table of `forward_batch`. `ws.max_tokens` must cover `M`; `Model::enable_batch` / `enable_verify` are sized for `Σ` (new
  `enable_spec_batch(rows)`).
- **Attention (section 3.5)** and **GatedDeltaNet (section 3.5)** are the two sequence-aware ops.
- A sequence with `k_i = 0` contributes one row and nothing else changes for it. Prefilling sequences are untouched: prefill chunks keep their own
  passes (decode-first scheduling is unchanged), so there is no mixing of prefill and verify rows in v1.

Row order inside the pass is by slot, rows of one sequence adjacent; the per-sequence results are read back by offset.

### 3.3 Batched drafting

Drafting per sequence (as `Seq::prepare` would do in the single-slot hook) costs a head layer plus an `lm_head` sweep per draft per sequence: 13 ms per
sequence per step at k=2, which is the whole gain at B = 4. So the draft is batched too:

1. **Catch-up:** each sequence has `n_i` queued hidden rows (its accepted tokens from the last step; mean about 2.5). One head-layer pass over the ragged
   `Σ n_i` rows (row table with the head's own per-sequence positions and KV), argmax of each sequence's last row -> `d_i1`.
2. **Chain:** `k-1` head passes of one row per still-drafting sequence (rows of sequences with `k_i = 1` drop out), each reading its previous draft.
3. **Verify** as in section 3.2.

The head layer and its attention reuse the single-sequence code with the row table. The head keeps one KV cache and one pending queue per sequence
(`SequenceState::mtp`, already per slot state). The draft `lm_head` is one `[B, V]` projection plus per-row argmax.

**Acceptance equality needs a row-exact head.** The tokens do not depend on the drafts, but "acceptance equal to solo MTP" does: the batched head must
produce the solo head's drafts bitwise. Today the solo head runs on the ordinary kernels (multi-row catch-up included); the batched head would run
ragged multi-row passes. Either the ordinary bf16 matmul/rmsnorm/rope kernels are already row-exact at M = 1..16 on HIP (W0 checks this with a 20-minute
test: ordinary M-row against M one-row calls, bytes compared), or **both** solo and batched head run under `exact_rows` (solo acceptance then shifts by noise
against main, which has to be re-baselined once; the cost is the head's `proj` kernels at M = 2..3, which the model puts under 0.3 ms per pass).

### 3.4 Per-sequence accept and rollback

Acceptance is host-side and per sequence, exactly as in the single-sequence engine (one download of the `M` verify tokens and the drafts, compare, `n_acc`).
The sequence's `accept_spec` then calls `spec_rollback(slot, keep)` with `keep = base + n_acc + 1`, which is `SequenceState::spec_rollback` for that slot's state,
unchanged:

- **Attention KV:** `len = keep`; rows past it are never read (attention reads keys `<= pos`) and are overwritten by the next append. Nothing is copied.
- **Recurrent state (GDN, conv):** verify saves the state after each row but the last into the slot's `SpecRows.bufs` (`q-1` copies); rollback copies
  buffer `keep-base-1` back into the live state. No copy when everything is accepted. This is today's logic per slot; the batched pass writes each
  sequence's saves into its own `SpecRows`. Cost: about 149 MiB per saved row per sequence (mr #147: the 27B's GDN state is 149 MiB read+write per sequence; the 63 MB in `load.rs` is the 35B figure; W0 reads it from the plan), i.e. `B*k*149 MiB`
  of extra device memory, 1.2 GB at B = 4, k = 2, which the plan accounting (`plan.rs` workspace) must include so `--max-seq` is not overcommitted. Idea B of the plan
  (out-of-place state, no copies, about +5%) stays an optimisation after correctness.
- **MTP head:** `m.kv.truncate`, `pending_rows = keep - m.kv.len`; the verify pass queues all of its hidden rows and the rollback drops the rejected ones, as in the single-sequence code.
- **Failure:** a row error fails that request; an outer error is fatal. As in `forward_verify`, an error part-way through leaves recurrent state matching
  no prefix, so the failing slot is `reset` (the request ends anyway).
- **Stop in the middle of a run:** `finish` truncates to the stop point through the same call.

Slots are independent, so rollback of one sequence never reads another's buffers; the only shared state is the workspace, which is rewritten every step.

### 3.5 Per-row verify attention and GatedDeltaNet

- **Attention (HIP):** HIP's multi-row split-KV attention partitions keys by the call's row count, so its row `t` is not bitwise the decode step's. That is why
  `Backend::verify_attn_per_row` (HIP) makes the single-sequence verify run **one decode call per row**. Batched verify does the same: one decode
  attention call per row `r` with `(seq, pos)` from the row table, KV append for the row's token first (rows of one sequence are appended in order,
  launches on one stream are ordered). Cost is linear in `M`: 2.5% of the solo pass at M=3; the 27B has 16 full-attention layers, so at M = 8 that is
  `16*8 = 128` launches per step, ~0.5 ms. The captured single-sequence step already does exactly this; the batched pass reuses its per-row launch
  helper with a table of `M` entries.
- **GatedDeltaNet (48 layers):** state is per sequence and the delta rule runs row by row. v1 does **not** add a ragged kernel: per layer it launches the
  existing verify GDN path once per sequence on that sequence's rows (conv + delta over `q_i` rows, saving the per-row states into the sequence's
  `SpecRows`), and the rest of the layer (projections, norms) stays flattened. Launches: `48 * B * ~3`, 1150 at B = 8; the step is GPU-bound (30 to 90 ms)
  so enqueue is hidden. A ragged GDN kernel (rows of all sequences in one launch) is a later optimisation.
- Both choices keep every row's arithmetic the single-sequence verify's, which is the single-sequence step's lossless argument carried over unchanged.

### 3.6 Draft-length policy

`k_i = f(B, row_budget, history)`: start from `k_max = 2`; while `Σ(k_i+1) > row_budget` reduce the longest drafts first; stop speculating when `B >= spec_off_at`.
A cheaper acceptance-driven policy (per-sequence moving average of accepted/proposed; drop to k=1 under 50%) is a refinement for the 35B and not needed on
the 27B, where k=2 wins at every size (sp doc section 2). The measured-cost gate of #143 stays off.

### 3.7 Memory and plan

Extra device memory: `SpecRows` `B*k*149 MiB`, the head's per-slot KV (one layer, small) and pending queue, a workspace for `M <= 16` rows, `verify_tok`/`batch_tok` for
`M`. `plan::resolve` / `phantom_state` already model the single-sequence spec buffers; they need the per-slot form, and the dry run (`--dry-run` plan json) must
list them so a model that fits at `--max-batch 4 --spec off` but not with spec on is refused at load, not mid-run.

### 3.8 Not in scope

- CUDA and Metal batched speculation: the generic code is backend-agnostic and their multi-row attention is bitwise per row, so enabling them later is
  a `spec_capable` flag and a gate run; v1 keeps them refused so their paths do not change. Shared files still change (`scheduler.rs`, `engine/batch.rs`,
  `model.rs`, `state.rs`), so the CUDA and Metal no-op gates run (section 4).
- Captured (graph) batched speculation: on the 27B the eager single-sequence step is within 0.4 to 1.4% of the captured one (sp doc section 4), so v1 is eager. A captured
  form needs a graph per `(sequence set, k vector, n vector)`, which explodes with B; revisit only with a measured gap.
- Device-side acceptance / next draft without a host round trip: the sp doc measured the gap at 0.3 ms per 53 ms pass (under 1%), below the 5% bar; unchanged for B > 1 (the host work
  is per sequence, B times the per-sequence host work of 0.004 ms).
- Tensor parallel (TP2) speculation: `spec_rollback` refuses TP states; unchanged.

## 4. Lossless gate

Definition: with batched speculation on, every sequence's output equals its **solo, spec off** output bitwise (the real reference: greedy output is a
function of the model, not of the schedule), and with the draft length fixed its per-step proposed/accepted counts equal its **solo MTP** counts.

| # | check | where |
|---|---|---|
| G1 | batched spec ids == solo spec-off ids, 256 greedy tokens, 27B natural-v1 128 / 1k / 4k / 16k, B = 2, 3, 4 (and 8 with `row_budget` 16 for stress), prompts of different lengths and different kinds in one batch | HIP window (`scripts/sp`) |
| G2 | with `k` fixed (policy off) and the same prompts, each sequence's `(passes, proposed, accepted)` equals its solo MTP k=2 run; this needs the row-exact head (section 3.3) | HIP window |
| G3 | scripted drafts (`DraftScript` in `intarsia-models/tests/mtp.rs`) covering every acceptance pattern, **different scripts per sequence in one batch**, B = 2..4, on the CPU backend with `INTARSIA_RECURRENT_BATCH=1` and the per-row-attention flag forced, hybrid fixture model: batched == solo spec-off | `cargo test`, no GPU |
| G4 | sequences join and leave mid-run (a new request is admitted while others are mid-run; one finishes on `max_tokens` inside a draft run; one is cancelled; one stops on a stop string inside a run; one fails alone via `row_fault`) and the survivors' ids still equal solo | `scheduler.rs` tests (mock `BatchModel`) and `fault_inject` |
| G5 | `ws`/row-table independence: the same sequence run beside different neighbours (permuted slots, different k vectors) gives the same ids and, at fixed k, the same counts | G3 and HIP window |
| G6 | spec off, any `--max-batch`: plan json and text byte-identical to main, 256 ids identical (8B, 35B, 27B, short and 4k) | HIP no-op; CUDA (spark1) and Metal (endor) no-op, because shared crates change |
| G7 | CUDA and Metal MTP equality tests at `--max-batch 1` (spec on, as in #142/#143) | spark1, endor |
| G8 | existing HIP gates: `kernels test --backend hip --quick`, `batch_recurrent --test-threads=1`, 35B `batch_cuda_real` with the hip backend including `--ignored`, the 27B `real_35b_batched_decode_is_bitwise_the_solo_step` run mr used | HIP window |
| G9 | `INTARSIA_MTP_VERIFY_CHECK` (verify-vs-decode bitwise) extended to a batched pass: after a verify pass, replay each row as a solo decode step on a cloned state and compare logits bytes; run at B = 2..4 | HIP window, debug only |

Not allowed to ship if any of G1, G3, G4, G6 fails. If G2 fails but G1 holds, the output is still lossless and the issue is only head equality; that case ships with the head under
`exact_rows` in both modes (section 3.3).

## 5. Work plan, windows, hours

Order matters: stage 0 can change what is worth building.

| stage | work | engineering h |
|---|---|---|
| 0 | W0 (below): solo MTP on main after mr, state size; no head fix (refuted, section 0) | 1 (analysis) |
| 1 | `Model::spec_batch`: ragged row table (verify rows), per-row attention with the table, per-sequence GDN launch + `SpecRows`, row-exact `lm_head` + argmax, `enable_spec_batch` sizing and plan accounting | 14 to 18 |
| 2 | batched draft head (ragged catch-up, chain), row-exact head switch (if W0 says needed) | 8 to 10 |
| 3 | scheduler (`SeqStep::Spec`, `BatchModel::spec_step/spec_rollback`, `accept_spec`, run delivery), `SlotModel` / `SeqAdapter` glue, `load.rs`, flags, metrics (`intarsia_spec_*` per batch size), docs rows | 8 to 10 |
| 4 | policy (row budget, k per B), tuning knobs (`--spec-row-budget`; `INTARSIA_*` documented in `docs/tuning.md`) | 3 to 4 |
| 5 | tests G3 to G5 (CPU, mock scheduler, fault injection), gate scripts for G1, G2, G9 | 10 to 12 |
| | **total core (stages 1 to 5)** | **43 to 54** (plus stage 0: 7 to 11) |

Fleet windows (tatooine = both R9700s, radiance down, restored and confirmed `active` before release; each <= 45 min per slot, split in two slots where noted):

| window | purpose | slot length |
|---|---|---|
| W0 (before any code) | (a) solo 27B MTP k=1 / 2 / 3 on current main (post-mr) at 128 / 1k / 4k / 16k, md5 vs spec off, ms per pass (`INTARSIA_SPEC_TRACE`): calibrates the model's B = 1 point; (b) plan json at `--max-batch 4` with and without MTP: state memory; the batched `lm_head` rows curve is mr's (#147), not re-measured; (c, a unit test in stage 2) ordinary bf16 matmul/rmsnorm M-row vs one-row bytes for head row-exactness | 45 min, both cards (radiance fills them) |
| W1 | G1, G2, G5 correctness at B = 2, 3, 4 (8 stress), 27B natural-v1, ids + counts; G9 | 45 min, 2 cards in parallel |
| W2 | localeval speed: single stream 128 / 1k / 4k / 16k, aggregate 1 / 4 / 16 streams (and 2, 8 added since that is where the table says the gain is), spec on vs main spec off and vs main batched, same window; plus the 35B quick check | 2 x 45 min |
| W3 | final gates G6 and G8 on the PR head (27B/35B/8B no-op, `kernels test --quick`, `batch_recurrent`, `batch_cuda_real` incl. `--ignored`, 27B batched equality) | 45 min |
| spark1, endor | G6 + G7 no-op and MTP equality, builds outside windows | 2 x 30 min each (before and after review fixes) |

Tatooine total: **5 slots, about 3.7 hours** if nothing fails, **5.5 to 6.5 hours** with one re-run of W1 and W3. Calendar: stage 0 in the next window opening;
the build itself is 6 to 8 working days of engineering at the hours above, with W1 usable from about day 4 once stages 1 to 3 compile.

## 6. Risks and decisions

1. **The gain range is narrow (2 to 4 streams).** Decision for the coordinator and the user: build it only if 2 to 4 concurrent 27B streams on one R9700 is
   a case that matters. The earlier "head fix first" idea was withdrawn (section 0); the model is calibrated on mr's measured curve, not on a fit.
2. **GEMV shape.** Everything above `M = 8` is limited by mr's `R(M)` (1.94 at 8, 3.39 at 16). A flatter small-row GEMV (the L1 lane) would move the break-even
   from B = 5 to roughly B = 8; the row budget is a knob so this needs no redesign.
3. **Head row-exactness** (section 3.3) is the one thing that can change solo MTP numbers slightly (acceptance noise, re-baseline); everything else
   leaves the single-sequence path bitwise unchanged.
4. **Memory:** `SpecRows` is `B*k*149 MiB` (mr's figure for the 27B state; W0 cross-checks it from the plan); at `--max-batch 16` with spec on this is the first place a load can run out; the plan check must refuse it with a message.
5. **Verify-row cost per sequence is bursty:** a sequence with a long accepted run delivers many tokens in one iteration (its ITL gap is the step), so the
   streaming smoothness of the batched path changes; tokens per iteration up to `k+1`. It is the same as solo MTP today; flagged for the server's SSE pacing.
6. **Out of scope and unchanged:** `--mtp-draft-vocab` default (user), the measured-cost gate (off), CUDA/Metal batched speculation.

Appendix: the model (stdlib Python; `plain` is mr's table interpolated):

```python
PL={1:30.3,2:36.8,3:39.2,4:42.1,5:46.2,6:50.4,8:59.5,12:82.7,16:110.9}; LOG={1:4.26,2:4.26,4:4.29,8:4.42,12:7.78,16:15.01}
hl=lambda m: 2.4*max(1.0,m/6)
verify=lambda B,q: plain(B*q)-0.75*(B*q-B)+B*((q-1)*0.26+0.3)
spec=lambda B,k: hl(2.5*B)+lm(B)+(k-1)*(hl(B)+lm(B))+verify(B,k+1)
TAU={1:1.8,2:2.5,3:2.95}; gain=lambda B,k: TAU[k]*plain(B)/spec(B,k)
```
