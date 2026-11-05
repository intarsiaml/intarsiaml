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

## Status

Phase A built and CPU-tested; CUDA gates and the spark1 speed run are in progress.
