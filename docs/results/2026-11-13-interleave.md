# Lane interleave: cap the 2-stream ITL tail (prefill vs decode interleave)

Date: 2026-11-13. Branch `prefill-decode-interleave`, off origin/main `10b235785` (#353).

## Brief (copied verbatim-in-substance so it survives compaction)

**Goal.** With 2 streams (`--max-batch 2 --prefix-store on`), Qwen3.6-35B-A3B-NVFP4 on a DGX Spark GB10 matches vLLM on TTFT and decode, but the inter-token tail is worse.

Window 3 baseline (48-request interleaved replay, seed 0, 2 streams, spark2):

| | ITL p99 median | ITL p99 worst | TTFT sum | decode/stream | combined | wall |
|---|---|---|---|---|---|---|
| vLLM | 304 ms | 884 ms | 77.8 s | 39.8 tok/s | 58.9 tok/s | 129 s |
| intarsiaml | 370 ms | 1412 ms | 73.4 s | 39.8 tok/s | 57.4 tok/s | 132 s |

Data: branch `ab35c`, `docs/results/2026-11-13-ab35c-tools-store-sampled.md` "Window 3", `raw/D-store-*`, `raw/serve-D-store.log` (worktree `/opt/code/intarsiaml/.worktrees/ab35c-lane`).

Cause: worst gaps = the other stream's uncached prefill running while this stream waits (c2t1 p99 1412 ms vs c3t1 8k-uncached prefill TTFT 1.49 s; c1t3 1312 ms vs ~5k uncached TTFT 1.31 s). Second source: store settle after Done, 170-410 ms per settle (256-602 MiB at ~1.5 GB/s).

**Target:** worst ITL p99 <= ~900 ms, median p99 <= ~310 ms; TTFT sum no worse than +5%; decode tok/s not lower; batched store-on output bitwise equal to single-engine store-on.

**Scope.**
1. Find how the batched engine schedules prefill chunks vs the other slot's decode; measure chunk size / per-chunk time with store on (packed + macro prefill are off with store on).
2. Cheapest fix capping the stall (decode step between prefill chunks / smaller chunk while another slot decodes / decode rows in the chunk). Keep chunk boundaries deterministic (bitwise gate; CUDA is chunk-sensitive). Document trade-off.
3. Reduce settle stall (split across decode steps or move download off scheduler thread). Fold in Fable #353 follow-ups: (a) `fail_active` (scheduler.rs ~905-916) never settles when it empties `active`; (b) request cancelled while queued (`slot == None`) still triggers settle (batch.rs ~683-691): skip it.
4. Conflicts: `prefix_store.rs` is being changed on `store-writer-bound` (keep edits small; rebase if it merges first). Lane `batch-fu` changes batch.rs startup/macro workspace, serve.rs, docs.

**Gates.**
- CPU: `cargo test -p intarsia-models --test batch_engine`, `--test prefix_store`, `-p intarsia-core --test scheduler`, `-p intarsia-core --lib`. New CPU test: decoding slot advances during other slot's long prefill.
- CUDA spark1: build via `REPO=/opt/code/intarsiaml/.worktrees/interleave ABL='~/intarsia-interleave' BHOST=spark1 FLEETQ_OWNER=claude-interleave fleetq run --host spark1 --load cpu --mem 24 --expect 20 --label "interleave: build" -- bash /opt/code/intarsiaml/.worktrees/ab35c/build.sh`; then `INTARSIA_TEST_REAL_MODEL=1 INTARSIA_PREFIX_STORE_DEBUG=1 cargo test --release -p intarsia-models --test prefix_store_cuda` under `fleetq run --host spark1 --quiet gpu --mem 45` (see `docs/results/2026-11-13-batch-store.md`). Never stop spark1 vLLM.
- Replay window spark2: <=35 min vLLM down, template `scripts/ab35c/window3.sh` (branch ab35c) copied to `scripts/interleave/window.sh`. Arms: main `10b235785` vs this branch, both `--max-batch 2 --prefix-store on`, replay2 interleaved seed 0 K=2 (K=3 if time). `FLEETQ_OWNER=claude-interleave fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 45 ...`; trap restores vLLM. Standing OK for <=60 min windows.

**Rules.** Real checks; tool calls <=4 min; fleetq status first, always `--expect`, no `--force`/`--no-queue`; GB10 memory watch (kill if avail <10 GB); commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR body ends with Claude Code line. Finish: DRAFT PR against main, no merge; report before/after table, gates, PR link.

## Progress log

### 1. What the scheduler does today (orientation, 2026-11-13 morning)

- `Scheduler::step` (crates/intarsia-core/src/scheduler.rs) already runs one decode step, then one prefill chunk, per iteration (`busy_chunks_per_iter` = 1). With the store on, the chunk length is the request's own store schedule (`SeqAdapter::chunk_len`: the model chunk 2048, cut at message boundaries). So decode **does** run between chunks; the ab35c note "the batched engine does not run a decode step between prefill chunks" is wrong.
- Window 3 server metrics (`raw/serve-D-store.metrics` minus `.metrics.before`, branch ab35c): 253 chunks, 271,856 tokens (mean 1074 rows per chunk); chunk *call* time 20–50 ms for most (the call only enqueues: CUDA runs it asynchronously and the next decode step's sync absorbs it); `intarsia_batch_itl_seconds` (server, between consecutive tokens of one request): 0 gaps above 0.5 s, 104 in 0.25–0.5 s, 61 in 0.15–0.25 s, 7383 at or below 0.15 s.
- **The 1.2–1.4 s client "p99"s are a metric artifact, not 1.4 s stalls.** localeval's `itl_stats` (`/opt/code/deepeval/src/localeval/client.py`) divides each SSE chunk's delta by a character-count weight (`n_tokens * chars / total_chars` when chunks != tokens), and its p99 is `ordered[int(0.99 n)]`, the second-largest of ~159 values. A ~330 ms stall that lands on a 1-character chunk (weight ~0.23) reads as ~1.4 s. So worst-request p99 ≈ (largest real gaps) / (weight of the token they land on): to get worst <= ~900 ms reliably, every server-side gap must be about 200 ms or less.
- The real gap sources: (1) one store chunk's GPU time (2048 rows ≈ 330 ms; boundary-cut chunks shorter), (2) settle after `Done` on the scheduler thread (170–410 ms), which also sits in front of the next admission (TTFT).

### 2. Design (chosen)

- **Layer-boundary pause of a prefill chunk** while another slot decodes: the chunk's thread parks between layers (residual stream saved and restored, model pass flags saved/cleared/restored), the scheduler runs the partner's decode step on the same stream, then the chunk resumes. Chunk boundaries, kernels and inputs are unchanged, so bits cannot change; pause points may even depend on timing without breaking reproducibility. Rejected: smaller busy chunk for store requests (changes bits vs single-engine store-on and splits chunk-keyed entries), decode rows in the chunk (changes GEMM row counts on a chunk-sensitive backend), the overlap twin (refused with the store; rb3 measured chunks ~55% slower beside decode, would break the TTFT gate).
- **Settle in slices**: `finish` only marks a settle pending; the scheduler demotes a bounded number of bytes per iteration after a decode step's sync (stream idle) and only in iterations with no prefill in progress, and settles fully when idle (covers Fable follow-up (a): `fail_active` emptying `active`). Queued-cancel (`slot == None`) never marks one (follow-up (b)).

### 3. Implementation and CPU gates

- `crates/intarsia-core/src/model/pause.rs` (new): `Model::with_layer_pause` installs a thread-local hook; `run_layers` calls `layer_pause` before evenly spaced layers of a single-sequence pass longer than `rows` (one park per `rows` rows). Around the hook: `ws.h[cur]` (and `ws.blk` when pending) copied to a save buffer and back on the pass's stream; 8 pass-scoped model flags (`in_layers`, `tail_dead`, `prefill_inv`, `exact_rows`, `head_rows_attn`, `head_whole`, `kv_only_skip_q`, `noncausal_rows`) cleared and restored. Off for TP, remote ranks, Gemma, plan recordings.
- `scheduler.rs`: `BatchModel::{pause_rows, prefill_begin_paused, prefill_park, prefill_resume, housekeep}`, `Housekeep::{Step, Pause, Idle}`, `Scheduler::prefill_paused` (decode step at every park; same planned chunk), `BatchMetrics::paused_steps` (server: `intarsia_batch_paused_steps_total`). Housekeeping: `Step` after a decode step when no request is prefilling; `Pause` inside a parked chunk; `Idle` when nothing is active or waiting (covers `fail_active` emptying `active`, follow-up (a)).
- `engine/batch.rs`: paused chunk on its own thread with a park/go channel pair (the overlap path's slot hand-off); `INTARSIA_PREFILL_PAUSE_ROWS` (default 768, 0 off; 0 with the overlap twin), `INTARSIA_SETTLE_SLICE_MB` (default 48). `finish` marks a settle pending instead of settling; not for a request that never held a slot (follow-up (b)). Pause-time settle only when the deferred footprint passes half of `DEFER_CAP`.
- `prefix_store.rs` (small): `enforce`'s demote loop body extracted to `demote_one` (behaviour unchanged), new `settle_some` (bounded demote ignoring `defer`, recycles full slabs as they empty, full `settle` at the end) and `deferred_bytes`; `DEFER_CAP` made `pub`.
- Tests: scheduler mock pause + `batched_equals_solo_randomized_with_pause`, `decoding_slot_advances_during_a_long_chunk` (0 partner steps in a 1500-row chunk without pauses, 4 with `pause_rows` 300; same tokens), `housekeeping_points`; real CPU models `batched_{hybrid,dense}_multi_turn_paused_chunks_are_bitwise` (pause 2 rows on 8-row chunks, one-object settle slices; bitwise the single engine store-on, `paused_steps` > 0, device tier empty once idle). Negative control: dropping the `ws.h` restore makes both real-model tests fail.
- CPU gates (macOS, this branch): `batch_engine` 9/9, `prefix_store` 31/31, `scheduler` 43/43, `intarsia-core --lib` 201 passed (2 ignored); `docs_env_vars` 2/2, server metrics test 1/1.

### Metric note (coordinator, 2026-11-13, from stream-burst lane PR #358)

localeval's `itl_stats` spreads a chunk's delta over its characters when SSE chunks do not line up with tokens, which inflates p99 (stream-burst: 50.9 ms reported vs 35.7 ms raw on the same marks). The reasoning parser (thinking on) and MTP (up to 3 tokens per verify) distort it further. This lane's replay (`scripts/ab35b/replay.py`, used by replay2) runs **thinking off, temperature 0**, and the store refuses MTP, so only the character weighting applies here. Raw engine-side gaps are reported next to the client numbers: the server's `intarsia_batch_itl_seconds` histogram (time between consecutive tokens of one request, scheduler side), diffed over each arm (`raw/serve-*.metrics` minus `.metrics.before`). Probe for raw per-chunk gaps if needed: `/opt/code/intarsiaml/.worktrees/stream-burst/docs/results/2026-11-13-stream-burst/sse_probe.py`.

### Pending step (coordinator, 2026-11-13): merge main before the PR

#357 (batch-fu) merged to main as 53a8c500d (touches `scheduler.rs` validate-before-restore / reject directly, `engine/batch.rs` own_chunk / check_constraint, `load.rs`, `serve.rs`); #358 (docs) merged too. Plan: finish the spark1 `prefix_store_cuda` gate and the spark2 window on the build in hand (branch 9cae1a9d1 code), then `git merge origin/main`, resolve, rerun the four CPU gates; if the merge touched the changed code paths (scheduler prefill/housekeep, batch.rs SlotModel/finish), rebuild on spark1 and rerun `prefix_store_cuda`. Then open the draft PR.

## 4. spark2 window (2026-11-13 07:43–08:00 EST)

vLLM (`missioncntrl-guidance`) down 1003 s (16.7 min, cap 35), restored with `/v1/models` 200 at 08:00:33; earlyoom active; MemAvailable ≥ 85 GB at every arm start. fleetq: `--takes-down retro --quiet gpu,cpu --mem 58 --expect 45`, owner claude-interleave. Script `scripts/interleave/window.sh 35 k3`, window 3's plan and trace (copied to `raw/`), replay2 interleaved seed 0 (48 requests, thinking off, temperature 0), every arm `--max-batch 2 --prefix-store on` with `INTARSIA_PREFIX_STORE_DEBUG=1`. Binaries built on spark2 by `scripts/interleave/build.sh`: E-main = main `10b235785`, E-il = this branch `bc0fb6c54` (code = `9cae1a9d1`; defaults `INTARSIA_PREFILL_PAUSE_ROWS` 768, `INTARSIA_SETTLE_SLICE_MB` 48). Table: `python3 scripts/interleave/table.py docs/results/2026-11-13-interleave/raw <arm prefixes>`. Server gaps = `intarsia_batch_itl_seconds` diff (engine side, raw, no character weighting).

| arm | ITL p99 median / worst (client, localeval) | TTFT sum | decode/stream median | combined | wall | server gaps > 0.15 / > 0.25 / > 0.5 s | paused steps | settles whole / slices |
|---|---|---|---|---|---|---|---|---|
| vLLM (window 3, not re-measured in this window) | 304 / 884 ms | 77.8 s | 39.8 tok/s | 58.9 tok/s | 129.1 s | n/a | n/a | n/a |
| D-store (window 3, #353) | 370 / 1412 ms | 73.4 s | 39.8 tok/s | 57.4 tok/s | 132.2 s | 165 / 104 / 0 | n/a | 47 / 0 |
| **E-main** (main 10b235785) K=2 | 384 / 1479 ms | 72.5 s | 40.1 tok/s | 57.9 tok/s | 131.1 s | 162 / 103 / 0 of 7548 | n/a | 47 / 0 |
| **E-il** (this branch) K=2 | **182** / 1201 ms | 77.0 s (+6.2%) | 39.4 tok/s (-1.7%) | 57.1 tok/s | 133.0 s | 116 / **0** / 0 of 7548 | 190 | 18 / 193 |
| E-main K=3 | 361 / 2287 ms | 158.1 s | 38.6 tok/s | 57.8 tok/s | 131.4 s | 180 / 109 / 0 | n/a | 48 / 0 |
| E-il K=3 | 179 / 1384 ms | 169.3 s (+7.1%) | 38.8 tok/s | 57.2 tok/s | 132.6 s | 113 / 0 / 0 | 203 | 23 / 195 |

(K=3 still serves 2 slots; the third stream queues, so its TTFT includes queueing.)

Against the targets (K=2):
- median ITL p99 <= ~310 ms: **met** (384 -> 182 ms, below vLLM's 304).
- worst ITL p99 <= ~900 ms: **not met** (1479 -> 1201 ms). The engine has no gap above 0.25 s any more (103 -> 0), so the remaining client worst is the character-weighting artifact on a ~0.15–0.25 s gap (116 such gaps remain: a 768-row segment plus its decode step, chunk ends with their capture enqueue, settle slices). Getting under 900 ms reliably needs every gap at roughly 150 ms or less, i.e. more parks.
- TTFT sum within +5%: **not met** (+6.2%, 72.5 -> 77.0 s).
- decode tok/s not lower: **not met by 1.7%** (40.1 -> 39.4 median per stream; window 3's D-store on #353 measured 39.8, so run-to-run spread is about 1%).

Why TTFT rose: **not established.** My first reading (each park adds the partner's ~24 ms decode step to the prefilling request, and the sliced settle saved nothing because main's settle was not on the admission path) did not survive an offline check (advisor review):
- Paired E-main / E-il requests by (conversation, turn): same prompts, same uncached rows (48/48), and **identical replies 48/48** (temperature 0).
- Regressing per-request ΔTTFT on estimated parks (store chunks of up to 2048 rows, `ceil(len/768) - 1` parks each): **slope -0.4 ms per park, intercept +97 ms per request** (ΔTTFT sum +4.54 s, median +52 ms). So the TTFT cost does not scale with parks; it is a roughly constant per-request cost whose cause is not identified. Prime suspect: the sliced settle (slices run only in iterations with nothing prefilling, so a new request can wait behind a slice, and slices stall both decoding streams); the parks themselves look nearly free.
- Server side (metrics diffs): `batch_itl` sum 188.7 s (main) vs 188.3 s (branch) over the same 7548 gaps, `batch_step` sum 125.1 s / 3893 steps vs 121.4 s / 3832, `batch_ttft` sum 71.1 s vs 75.7 s. Total decode time is unchanged; the client decode median moved 40.1 -> 39.4 tok/s (means 41.22 -> 40.90, -0.8%).
- Consequently the earlier knob projections (1024 / 512 rows) are withdrawn, and whether a setting meets all four targets is open. The next measurement that would settle it: one window with pause on / slicing off and pause off / slicing on as separate arms.

**Defaults changed after this check:** `INTARSIA_PREFILL_PAUSE_ROWS` and `INTARSIA_SETTLE_SLICE_MB` both default to 0 (off) on this branch, so by default the engine behaves as main (whole settle right after `Done`), plus follow-ups (a) (idle settle when `fail_active` empties `active`) and (b) (no settle for a request cancelled while queued). The window above ran the earlier defaults (768 / 48), i.e. both features on. CPU gates rerun after the change: `prefix_store` 32/32, `batch_engine` 9/9, `scheduler` 44/44, `intarsia-core --lib` 201 (2 ignored), `docs_env_vars` 2/2.

## 5. CUDA gate (spark1) and the merge of main

- spark1, build `9cae1a9d1` (`~/intarsia-interleave`, built by `scripts/ab35c`'s build.sh under fleetq #9234), fleetq #9238 (`--quiet gpu --mem 45`, vLLM up): `INTARSIA_TEST_REAL_MODEL=1 INTARSIA_PREFIX_STORE_DEBUG=1 INTARSIA_PREFILL_PAUSE_ROWS=256 INTARSIA_CB_TRACE=1 cargo test --release --locked -p intarsia-models --test prefix_store_cuda -- --test-threads=1 --nocapture prefix_store_`: **2 passed** (180 s). `prefix_store_batched_is_single_engine_cuda`: all 4 batched replies bitwise the single-sequence engine's (store on), ids equal, cached 1871 / 1168 match. Pauses forced to 256 rows so the gate exercises them: the trace shows 2 parked chunks with 7 decode steps of the partner inside them, and 5 sliced / idle settles. Lowest MemAvailable 59 GB (sampled every 10 s).
  - An earlier attempt (#9236) never ran the test (`cd` inside a backgrounded subshell, cargo started in `~`), and its orphaned memory-sampling loop held the job open for 26 min until I killed it under that job.
- Merge of origin/main `53a8c500d` (#357 batch-fu, #358): one conflict, `docs/tuning.md` (main's `INTARSIA_PREFIX_STORE` / `_DEBUG` rows taken, this lane's settle wording reapplied). Main's code changes are admission-time `BatchSeq::validate`, `own_chunk` for the `PrefillStart` trace event, and a comment in `store_combinations`; none touches the pause path, `housekeep`, `finish`/settle or the model, so the CUDA gate above (pre-merge build) was not rerun. CPU gates on the merge `b338dfc50`: `batch_engine` 9/9, `prefix_store` 32/32, `scheduler` 44/44, `intarsia-core --lib` 201 (2 ignored), `docs_env_vars` 2/2.


## 6. PR

Draft PR https://github.com/intarsiaml/intarsiaml/pull/359 (not merged). Host build dirs left in place: spark1 `~/intarsia-interleave`, spark2 `~/intarsia-interleave` and `~/intarsia-il-main`.

## 7. Brief: second window, TTFT attribution (coordinator, 2026-11-13)

One more spark2 window to find the cause of the +6% TTFT. All arms on the current HEAD build, `--max-batch 2 --prefix-store on`, replay2 interleaved seed 0, K=2: (1) both switches off, (2) pause only 768, (3) slice only 48, (4) both (dropped if 4 do not fit). Main's arm reused from window 1 (E-main) unless time allows a rerun. vLLM down <= 35 min, same trap/restore; `fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 45`. Report per arm: TTFT sum, decode per stream, median ITL p99, server gap counts; plus per-request paired TTFT delta against the switches-off arm (separates the off path's own cost from each switch's). Do not change engine code until Fable's #359 findings arrive (forwarded by the coordinator). Also: fix the ab35c lane doc's claim that decode never runs between chunks (`/opt/code/intarsiaml/.worktrees/ab35c-lane`, branch `ab35c`, `docs/results/2026-11-13-ab35c-tools-store-sampled.md`, "Cause (checked against the per-request trace)"), commit and push on `ab35c`.

## 8. Fable review of #359 (forwarded 2026-11-13): merge after fixes — to do after the split window

Fable: pause path and sliced settle against in-flight captures check out; with both switches off the PR is main plus the queued-cancel skip. Fixes:
1. (Medium) Idle settle after `fail_active` does nothing: `housekeep` returns unless `self.settle` is `Some`, which only `finish` with `settle_slice > 0` sets; `fail` never does. Make `Idle` settle whenever `store.deferring()` (give `SlotModel` a store handle). Real-model CPU test: fail -> idle settle, deferred bytes to 0 with no further admission.
2. (Medium, before `store-writer-bound` merges) `settle_some` never calls `poll_writer`: call it at the top.
3. (Low, opt-in) Slicing with pause off has no relief valve (`Step` slices only when nothing prefills): simplest of whole settle when deferred bytes pass DEFER_CAP/2 and a prefill is due, or document that `INTARSIA_SETTLE_SLICE_MB` needs `INTARSIA_PREFILL_PAUSE_ROWS`.
4. TTFT attribution from `serve-E-il.log`: 51 slices inside parked chunks (1.93 s, direct TTFT cost; consider not slicing in a park unless near the cap); one capture hit DEFER_CAP and went synchronous (156 ms); insert alloc 225 -> 437 ms; the rest most likely the 190 parked decode steps (~24 ms). Use actual park counts from the trace, not `ceil(len/768)-1`.
5. After the fixes: CPU gates and spark1 `prefix_store_cuda` on the final HEAD; confirm in the doc that the switches-off arm covers "replay identical to main" for the shipped defaults. Push, update the PR body, report HEAD sha, window table, gates. Do not merge.

