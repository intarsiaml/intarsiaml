# bm: batched MTP speculation (`--spec mtp` with `--max-batch` > 1)

Lane `lane-bm/batched-mtp`, 2026-11-05. Base: `origin/main` 602a1fcfe.

## Approval

The user approved starting this work on 2026-11-05 at 17:08 EST ("start 2"), reversing their 2026-10-25 deferral (design note
`docs/results/2026-10-25-sp-batched-mtp-design.md` on `lane-sp/batched-mtp-design`, revision 2).

## User decision: N=1 default (2026-11-05, about 19:45 EST)

Relayed by the coordinator after #309 merged (opt-in, main 196e768b6): the user decided "N=1 default". Implemented on branch
`lane-bm/batched-mtp-default` as a recipe default, `[tuning.cuda] batched_mtp = 1` and `[tuning.metal] batched_mtp = 1` in `recipes/qwen3_5_moe.toml`
(the measured model), read only at tp = 1 in-process; HIP and every tp > 1 stay off; `INTARSIA_BATCHED_MTP` wins (`0` off);
`INTARSIA_BATCHED_VERIFY` stays off. On CUDA the recipe's `--spec mtp` default now also applies to a `--max-batch` > 1 server; on Metal the 35B has
no `--spec mtp` default, so N = 1 applies to an explicit `--spec mtp --max-batch` > 1 without the env var. Other recipes (the 27B, the 8B) are unchanged.

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

## CUDA 35B NVFP4 tp1 on spark1 (fleetq #8654; binary built at 1b72bad12, code identical to the branch head)

localeval speed (`scripts/bm/le-window.sh w1 2 ...`), 1k prompt, 256 generated, thinking off, 3 reps, 2 rounds ABBA, all `--max-batch 16 --max-seq 8192
--mem-budget 40`, spark1-vllm up (MemAvailable 51 GB with our server up). Arms: plain (no spec), bmtp1 (phase A, drafting at B = 1), bmtp1d (bmtp1 +
`INTARSIA_BATCHED_VERIFY=1`, which at B = 1 only adds the verified-run drain), bver5 / bver8 (phase B, batched verify, drafting at B <= 5 / 8), k = 2.
Raw: `raw/w1/` (localeval outputs, `/metrics` spec counters, server logs).

| arm | streams | aggregate r1 / r2 | mean | per-stream (median, r1 / r2) | TTFT p50 s (r1 / r2) |
|---|---|---|---|---|---|
| bmtp1 | 1 | 106.0 / 99.4 | 102.7 | 117.0 / 107.2 | 0.23 / 0.19 |
| bmtp1d | 1 | 98.1 / 95.4 | 96.8 | 105.7 / 102.5 | 0.19 / 0.19 |
| bver5 | 1 | 104.7 / 91.4 | 98.1 | 113.3 / 98.2 | 0.19 / 0.19 |
| bver8 | 1 | 103.0 / 98.4 | 100.7 | 127.1 / 108.7 | 0.47 / 0.25 |
| plain | 1 | 80.1 / 80.3 | 80.2 | 85.1 / 85.1 | 0.19 / 0.19 |
| bmtp1 | 2 | 121.8 / 122.7 | 122.2 | 65.6 / 66.4 | 0.30 / 0.32 |
| bmtp1d | 2 | 123.0 / 121.4 | 122.2 | 66.0 / 65.2 | 0.29 / 0.29 |
| bver5 | 2 | 112.4 / 112.9 | 112.7 | 60.6 / 60.3 | 0.29 / 0.29 |
| bver8 | 2 | 111.2 / 114.6 | 112.9 | 59.2 / 63.6 | 0.29 / 0.32 |
| plain | 2 | 122.1 / 122.8 | 122.4 | 65.5 / 65.8 | 0.28 / 0.28 |
| bmtp1 | 3 | 149.0 / 150.8 | 149.9 | 53.8 / 55.2 | 0.40 / 0.45 |
| bmtp1d | 3 | 151.3 / 151.4 | 151.4 | 54.7 / 54.8 | 0.40 / 0.40 |
| bver5 | 3 | 124.0 / 125.9 | 125.0 | 43.6 / 46.2 | 0.32 / 0.48 |
| bver8 | 3 | 125.3 / 131.7 | 128.5 | 46.1 / 47.7 | 0.40 / 0.40 |
| plain | 3 | 151.1 / 157.5 | 154.3 | 54.3 / 56.9 | 0.38 / 0.38 |
| bmtp1 | 4 | 173.6 / 174.2 | 173.9 | 47.7 / 47.8 | 0.53 / 0.52 |
| bmtp1d | 4 | 173.6 / 173.7 | 173.6 | 47.6 / 47.7 | 0.53 / 0.52 |
| bver5 | 4 | 134.2 / 138.5 | 136.3 | 36.9 / 37.5 | 0.53 / 0.53 |
| bver8 | 4 | 142.4 / 140.0 | 141.2 | 39.1 / 37.9 | 0.54 / 0.53 |
| plain | 4 | 183.2 / 173.6 | 178.4 | 50.4 / 47.5 | 0.51 / 0.52 |
| bmtp1 | 8 | 234.5 / 228.8 | 231.7 | 32.4 / 31.6 | 0.82 / 0.82 |
| bmtp1d | 8 | 230.4 / 232.5 | 231.4 | 31.9 / 32.1 | 0.83 / 0.82 |
| bver5 | 8 | 229.4 / 231.7 | 230.6 | 31.8 / 32.2 | 0.86 / 0.84 |
| bver8 | 8 | 137.1 / 138.1 | 137.6 | 18.6 / 18.8 | 0.84 / 0.84 |
| plain | 8 | 234.4 / 234.7 | 234.6 | 32.4 / 32.5 | 0.81 / 0.82 |
| bmtp1 | 16 | 305.9 / 307.3 | 306.6 | 21.6 / 21.8 | 1.50 / 1.58 |
| bmtp1d | 16 | 304.0 / 303.2 | 303.6 | 21.6 / 21.4 | 1.59 / 1.59 |
| bver5 | 16 | 305.8 / 305.0 | 305.4 | 21.8 / 21.7 | 1.62 / 1.62 |
| bver8 | 16 | 311.7 / 301.6 | 306.6 | 22.5 / 21.6 | 1.72 / 1.69 |
| plain | 16 | 308.1 / 305.4 | 306.8 | 21.9 / 21.6 | 1.56 / 1.56 |

acceptance bmtp1: 61.0% / 60.8%
acceptance bmtp1d: 59.9% / 65.8%
acceptance bver5: 59.3% / 60.2%
acceptance bver8: 60.7% / 61.5%

run ids bmtp1: 20261105-203038-adhoc-bm-bmtp1-r1, 20261105-204216-adhoc-bm-bmtp1-r2
run ids bmtp1d: 20261105-203539-adhoc-bm-bmtp1d-r1, 20261105-203715-adhoc-bm-bmtp1d-r2
run ids bver5: 20261105-203217-adhoc-bm-bver5-r1, 20261105-204038-adhoc-bm-bver5-r2
run ids bver8: 20261105-203357-adhoc-bm-bver8-r1, 20261105-203852-adhoc-bm-bver8-r2
run ids plain: 20261105-202856-adhoc-bm-plain-r1, 20261105-204349-adhoc-bm-plain-r2

Reading:

- **1 stream: phase A closes most of the gap.** 80.2 -> 102.7 tok/s aggregate (+28 %), per-stream 85 -> 112. The scoreboard's solo MTP 114 is from another
  window and a `--max-batch 1` server, so this run does not show whether the last ~10 % is the batching engine's overhead or window-to-window.
- **The drain does not help on CUDA** (bmtp1d 96.8 vs bmtp1 102.7 at 1 stream): the Metal 1-stream gap between bver5 and bmtp1 is not explained by
  it here; it stays unexplained.
- **2 to 16 streams: phase A ties plain** within 0 to 3 % (2: 122.2 vs 122.4; 3: 149.9 vs 154.3; 4: 173.9 vs 178.4; 8: 231.7 vs 234.6; 16: 306.6 vs 306.8).
- **Batched verify loses on CUDA at every width where it drafts:** 2 streams -8 %, 3 -17 %, 4 -21 to -24 %, and bver8 at 8 streams -41 %
  (137.6 vs 234.6). The verify rows run the row-exact GEMV kernels (`*_gemv_rows`), whose cost grows about linearly with rows, so on the MoE 35B a
  3-row verify per sequence costs close to 3 decode rows while yielding ~2.2 tokens; the per-sequence head drafts come on top. The design note's
  +48 % at 2 streams on the HIP 27B does not carry to this model and these kernels. Acceptance is ~60 % in every arm.

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

Reading: drafting pays only for a lone stream. At 1 stream phase A gives +16 % aggregate (111.2 -> 128.7) and phase B's arm +25 % (139.1). At B = 1
both run the same solo MTP step; the one code difference is the verified-run drain (`BatchSeq::next_ready`, on whenever batched verify is on), which
delivers a whole accepted run in one scheduler iteration instead of one token per iteration. The gap held in both rounds (137.7 / 140.6 vs
129.6 / 127.7), so it is not noise; the CUDA run has an arm (`bmtp1d`: N = 1 with the flag) that isolates the drain. At 2 streams the batched verify
is break-even (168.1 vs 167.5) and at 4 streams it loses 25 % (174.4 vs 231.8): on Metal a 4-sequence verify pass of 12 rows plus 4 per-sequence
head drafts costs more than the ~1.6 tokens per sequence it yields over a 4-row decode step. At 8 streams the B <= 5 rule falls back to plain and
the arms tie (281 to 292). Phase A at 2 and 4 streams does not draft, yet costs about 3 % against plain in both rounds (2 streams 163.0 / 162.4 vs 166.0 / 169.0; 4 streams
223.7 / 227.4 vs 231.2 / 232.5); likely the per-step head-queue copy and the head catch-up passes (unverified). At 8 streams it ties.

## Recommendation

- **Default-on for the lone-user case: `INTARSIA_BATCHED_MTP=1` (draft only while one sequence decodes) on CUDA tp1 and Metal**, where `--spec mtp` would
  be the single-sequence default. It gives +28 % (CUDA) and +16 % (Metal) at 1 stream, costs 0 to 3 % at 2 to 4 streams, and nothing at 8 to 16.
  Greedy output is bit-identical per sequence. Flipping the default is the coordinator's call; this branch leaves it opt-in.
- **Batched verify (`INTARSIA_BATCHED_VERIFY`) stays opt-in, effectively off:** it loses 8 to 41 % on CUDA and is break-even to -25 % on Metal at 2 to 8
  streams. It is worth revisiting only with verify kernels whose cost per extra row is well under a decode row (a real multi-row GEMM for the
  MoE / NVFP4 projections under a row-exactness guarantee) and a batched draft head.
- Not run: HIP (tatooine, out of scope); CUDA at tp2; a solo `--max-batch 1` MTP arm in the same window.

## Status

Done for this round: phase A and phase B built, gated and measured. Open: the Metal 1-stream bver5 vs bmtp1 gap (not the drain on CUDA); the CUDA
packed-prefill flake on main (separate task); rebase onto `lane-ss/sampled-spec` once it merges (then gate the deferred verify path on greedy requests,
since `forward_verify_batch` returns argmax only).
