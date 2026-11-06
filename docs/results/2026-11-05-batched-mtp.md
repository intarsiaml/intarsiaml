# bm: batched MTP speculation (`--spec mtp` with `--max-batch` > 1)

Lane `lane-bm/batched-mtp`, 2026-11-05. Base: `origin/main` 602a1fcfe.

## Approval

The user approved starting this work on 2026-11-05 at 17:08 EST ("start 2"), reversing their 2026-10-25 deferral (design note
`docs/results/2026-10-25-sp-batched-mtp-design.md` on `lane-sp/batched-mtp-design`, revision 2).

## Coordinator direction (2026-11-05, during the lane)

Copied from the coordinator's message: Fable's engine review (`.worktrees/perf-ideas/fable-engine/report.md`, idea 1) recommends two phases.
Do phase A first, push it as soon as it passes, and report it before phase B.

- **Phase A:** inside the batching engine, run solo MTP whenever only one slot is active (hooks in `engine/batch.rs`, `spec_off_at` exists).
  A lone user on a `--max-batch` server would get solo-MTP speed: Spark 35B 4k 80.8 -> 114, HIP 27B 56 -> 93, Metal 35B 124 -> 160 (plain batched -> solo MTP).
  Fable's estimate 12-20 h, bitwise, cheap kill test: a CPU tiny-hybrid equality check.
- **Phase B:** the multi-row (ragged) verify for 2-4 streams, as in the design note.

## Phase A: what was built (branch `lane-bm/batched-mtp`)

Opt-in `INTARSIA_BATCHED_MTP=N` (default unchanged: `--spec mtp` with `--max-batch` > 1 is still refused without it). With it, at `--tp 1`:

- **Per-step choice.** `load.rs` and `BatchedEngine::new` accept `--spec mtp` (also on recurrent / hybrid models). The scheduler's existing hook does
  the rest: while at most `N` sequences decode (`spec_off_at = min(--spec-off-at, N + 1)`), a sequence runs its own solo draft + verify pass on its
  slot inside `prepare` (the single-sequence `Seq::prepare` code: captured `spec_step` where it applies, else `mtp_draft` + `forward_verify`, the same
  `spec_accept` / rollback, so the sampled-spec lane's accept rule plugs into the one place it already lives); wider steps are plain batched decode.
  The table behind the choice is the measured one below: `N` is where batched MTP stops paying.
- **MTP heads stay in sync across plain batched steps.** `Model::forward_batch` now queues each row's final-normed hidden row (left in `ws.n` by the
  eager and the captured step) for that sequence's head without running it; the next draft catches the head up over every queued row in one pass
  (`Model::mtp_after_rows`; a nearly full queue is caught up in place). So a wide batch pays one 4 KB device copy per sequence per step, not a head
  pass. Packed prefill (`forward_packed`) used to refuse a state with a head; it now feeds each prompt's head and catches up, as a solo chunk does.
- **Memory.** `enable_batch` reserves each extra slot's drafter state (`k` recurrent verify-row copies + the head's hidden-row queue,
  `Model::mtp_slot_bytes`) out of the KV budget, with a plan note; 0 without the head, so `plan show` without MTP is unchanged.
- **Prefix reuse** is off in the batched-MTP engine (a reused prefix would need the head's cache to cover it; on CUDA / HIP / Metal it is already
  off for the hybrid 35B, `batch_snapshots() == 0`).
- Not changed: the decode twin (`INTARSIA_OVERLAP`, opt-in) refuses a model with a head, so overlap falls back to off with its existing log line;
  `--spec auto`, DFlash and tensor parallelism stay refused with `--max-batch` > 1.

## Bit-identity (CPU tiny hybrid, `crates/intarsia-models/tests/mtp.rs`)

The fixture is the 35B-shaped tiny hybrid (GatedDeltaNet + gated attention + NVFP4 MoE + MTP head). Each test compares every sequence's greedy ids,
finish reason and token counts with its solo **spec-off** run:

| test | what it interleaves | result |
|---|---|---|
| `batched_mtp_interleaved_sequences_equal_solo_with_every_acceptance_pattern` | 4 sequences (prompts 13 / 6 / 21 / 30 tokens, max_tokens 60 / 83 / 106 / 129) on 4 slots, drafting at B <= 4, scripted drafts with a per-sequence accept/reject pattern, k = 1, 2, 3 | pass; k=1: 21 passes, 12 of 21 accepted; k=2: 25 of 45; k=3: 106 of 183 |
| `batched_mtp_switches_between_plain_batched_steps_and_drafting` | 6 requests on 4 slots (slot reuse), drafting only at B <= 2, so 3- and 4-row steps are plain batched and drafting resumes after the head catch-up; head drafts and scripted drafts | pass; batch sizes 1 to 4 all occur (hist 33 / 9 / 27 / 96 steps); 1 of 24 and 20 of 50 accepted |
| `batched_mtp_with_packed_prefill_equals_solo` | 5 prompts of 17 to 40 tokens, packed prefill on (2 packed passes), drafting at B <= 4 | pass; 19 of 28 accepted |

The cpu kernels compute every row with the one-row loop, so this proves the engine logic (rollback per slot, head queue / catch-up, plain-draft switch,
packed feed), not GPU numerics; the CUDA greedy-ids gate below covers the device.

## Phase B (opt-in `INTARSIA_BATCHED_VERIFY=1`, built while the fleet queue was full)

Phase A's solo passes cost a whole weight pass per drafting sequence, so they cannot pay at 2 or more streams. Phase B batches the verify:

- `Model::forward_verify_batch(states, ids)`: the verify rows of several sequences as one ragged list through the existing packed path
  (`Seqs::Packed`, a segment per sequence) under `exact_rows`. Projections, MoE, norms and the head run once over all rows on the row-exact
  kernels; attention and the GatedDeltaNet recurrence run per segment through the solo verify code (per-row attention where the backend needs it;
  the delta rule row by row with the per-row state saves, now also in the multi-segment branch of `gdn_block_segs`). Each state ends as after its
  own `forward_verify` (committed rows, `spec_rows.saved`, head queue). Passes are capped at 16 rows (`VERIFY_MAX_ROWS`); the engine splits larger
  sets into several passes.
- Scheduler: `SeqStep::Verify(ids)`, `BatchModel::verify_step`, `BatchSeq::accept_verify`. A sequence that drafts while others decode returns its
  `[tok, drafts]` (`Seq::prepare_with(.., defer_verify)`: the draft is still the solo `mtp_draft`), the scheduler runs one `verify_step` for all of
  them, then each sequence accepts and rolls back through the same `Seq::verify_pass` / `spec_accept` as a solo pass (the seam the sampled-spec lane
  changes). At B = 1 nothing changes (the captured solo step runs).
- Verified runs are delivered in the iteration they were verified (`BatchSeq::next_ready`), so the drafting sequences stay in phase and verify
  together; without this the passes mostly held one sequence.
- The busy-batch drafter check (`SpecGate::allow(busy)`) does not apply to a deferred verify (it exists because a solo pass beside a batched step is
  a whole extra weight pass).
- Still per sequence: the draft (one head pass per drafting sequence). A batched head is the next step if the measurement says the draft dominates.

CPU tests (`crates/intarsia-models/tests/mtp.rs`):

| test | result |
|---|---|
| `batched_verify_pass_equals_each_sequences_solo_verify` (model level: 3 states, 24 rounds of segment lengths 1..4, drafts accepted and rejected; verify tokens, head drafts after rollback, and a later decode step's logits bitwise equal the solo states') | pass, 77 of 105 drafts accepted |
| `batched_verify_engine_equals_solo` (engine: 4 interleaved sequences, k = 1..3, scripted per-sequence patterns; then 6 requests on 4 slots drafting at B <= 2) | pass; verify passes by sequences per pass, k=1: 1 / 20 / 17 / 32 passes of 1 / 2 / 3 / 4 sequences; k=2: 5 / 15 / 28 / 13; k=3: 0 / 21 / 11 / 14 |

## Gates

| gate | where | result |
|---|---|---|
| `cargo test --workspace --no-fail-fast` | endor, fleetq #8608, tree 29e123586 (phase A + B) | 1327 passed, 0 failed, 29 ignored (230 test binaries) |
| CUDA `plan show` (35B NVFP4, `--max-seq 8192 --mem-budget 40`, json), branch vs main 602a1fcfe | spark1, fleetq #8614 (`raw/gates-cuda2/`) | byte-identical (15570 B; the recipe's tp1 MTP default is on here, so the MTP plan is covered too) |
| CUDA `kernels test --backend cuda --quick` | spark1, #8614 | 106 kernels, 918 cases, 0 not admitted / failed |
| CUDA greedy ids, 6 concurrent prompts of different lengths and `max_tokens` (`scripts/bm/ids.py`) vs the solo spec-off server, 35B NVFP4 tp1 | spark1, #8614 (`raw/gates-cuda2/`; repeats `raw/gates-cuda3/`, scripts `scripts/bm/gates2.sh` .. `gates4.sh`) | see below |
| Metal no-op, goldens, ids, speed | endor, fleetq #8616 (queued) | pending |

CUDA greedy ids (6 sequences; `ALL` = every sequence equals its solo spec-off ids):

| server | drafted / accepted | result |
|---|---|---|
| phase A, `INTARSIA_BATCHED_MTP=4 --max-batch 4` | 101 / 81 | ALL |
| phase A, `INTARSIA_BATCHED_MTP=1 --max-batch 16` | 3 / 3 | ALL |
| phase B, `=5 INTARSIA_BATCHED_VERIFY=1 --max-batch 8`, k=2 | 305 / 227 | sequence 2 differs from token 2 on, **identical to the plain batched run below** |
| phase B, `=8 ... --max-batch 16`, k=3 | 1312 / 839 | the same sequence 2 difference, identical to plain batched |
| phase B, as above with `INTARSIA_PACK_PREFILL=0` | 314 / 235 | ALL |
| plain batched `--max-batch 16`, branch | 0 | sequence 2 differs (2 of 2 runs) |
| plain batched `--max-batch 16`, **main 602a1fcfe** | 0 | ALL in 2 runs, sequence 2 differs from token 2 in 2 runs (4 runs) |
| plain batched, branch, `INTARSIA_PACK_PREFILL=0` | 0 | ALL |

Finding (not this lane's): on main, the CUDA batched engine with packed prefill on (the CUDA default since SRV16) intermittently gives one of
six concurrent prompts a different greedy continuation from token 2 on; with packing off it matched in every run here. Which prompts pack
together depends on arrival timing, so it is schedule-dependent. Batched MTP (A and B) reproduces the plain batched engine's tokens exactly in
those runs, so the verify path adds no difference of its own. Raw: `raw/gates-cuda2/` (first set), `raw/gates-cuda3/` (main and no-pack repeats).

## Metal 35B NVFP4 on endor (fleetq #8616, one `--exclusive` slot, tree 7fb2a067d)

No-op vs main 602a1fcfe (`raw/metal/noop.txt`): `plan show` text and json byte-identical for Qwen3-8B and the 35B; 256 greedy ids identical on the
short and ~3.5k prompts for both models; `kernels test --backend metal` 49 kernels, 980 cases, all admitted; goldens qwen3-8b-fp32 and -long PASS
(bf16 class). Batched-MTP greedy ids, 6 concurrent prompts vs the solo spec-off server: phase A (`=4`, 36 drafted / 26 accepted), phase B (`=5` +
batched verify, 254 / 197) and plain `--max-batch 8` all ALL identical.

localeval speed, 1k prompt, 256 generated, thinking off, 3 reps, 2 rounds ABBA (`--max-batch 8`; plain = no spec, bmtp1 = phase A drafting only at
B = 1, bver5 = phase B drafting at B <= 5 with batched verify, k = 2):

| arm | streams | aggregate r1 / r2 | mean | per-stream (median, r1 / r2) | TTFT p50 s (r1 / r2) |
|---|---|---|---|---|---|
| bmtp1 | 1 | 129.6 / 127.7 | 128.7 | 145.4 / 142.8 | 0.22 / 0.22 |
| bver5 | 1 | 137.7 / 140.6 | 139.1 | 155.4 / 159.2 | 0.22 / 0.22 |
| plain | 1 | 111.1 / 111.2 | 111.2 | 122.2 / 122.4 | 0.22 / 0.22 |
| bmtp1 | 2 | 163.0 / 162.4 | 162.7 | 91.4 / 91.0 | 0.34 / 0.34 |
| bver5 | 2 | 170.2 / 166.1 | 168.1 | 96.8 / 93.8 | 0.34 / 0.34 |
| plain | 2 | 166.0 / 169.0 | 167.5 | 92.9 / 94.8 | 0.33 / 0.33 |
| bmtp1 | 4 | 223.7 / 227.4 | 225.6 | 64.1 / 65.3 | 0.58 / 0.58 |
| bver5 | 4 | 173.3 / 175.6 | 174.4 | 48.5 / 49.9 | 0.57 / 0.57 |
| plain | 4 | 231.2 / 232.5 | 231.8 | 66.2 / 66.6 | 0.56 / 0.56 |
| bmtp1 | 8 | 287.8 / 293.8 | 290.8 | 42.3 / 43.2 | 1.04 / 1.03 |
| bver5 | 8 | 281.7 / 280.0 | 280.9 | 42.4 / 42.0 | 1.19 / 1.19 |
| plain | 8 | 292.1 / 291.2 | 291.6 | 42.8 / 42.7 | 1.01 / 1.01 |

acceptance bmtp1: 59.7% / 56.9%
acceptance bver5: 61.9% / 60.8%

run ids bmtp1: 20261105-200027-adhoc-bm-metal-bmtp1-r1, 20261105-200222-adhoc-bm-metal-bmtp1-r2
run ids bver5: 20261105-200105-adhoc-bm-metal-bver5-r1, 20261105-200143-adhoc-bm-metal-bver5-r2
run ids plain: 20261105-195949-adhoc-bm-metal-plain-r1, 20261105-200253-adhoc-bm-metal-plain-r2

Reading: drafting pays only for a lone stream. At 1 stream phase A gives +16 % aggregate (111.2 -> 128.7) and phase B's arm +25 % (139.1; at B = 1
both run the same solo MTP step, so the gap between them is run-to-run and spec-gate state, not the batched verify). At 2 streams the batched verify
is break-even (168.1 vs 167.5) and at 4 streams it loses 25 % (174.4 vs 231.8): on Metal a 4-sequence verify pass of 12 rows plus 4 per-sequence
head drafts costs more than the ~1.6 tokens per sequence it yields over a 4-row decode step. At 8 streams the B <= 5 rule falls back to plain and
the arms tie (281 to 292). Phase A at 2 to 8 streams equals plain (it does not draft there).

## Status

Phase A built and CPU-tested; CUDA gates and the spark1 speed run are in progress.
