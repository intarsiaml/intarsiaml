# DSv4.1 TP=2: three failure paths (lane dsv41/failpaths, 2026-11-03)

Branch `dsv41/failpaths` from origin/main `5786176be`, local only. The unit tests run the real `Worker` and `Dsv41Runtime` on the cpu backend on endor (macOS), with graphs from the Mia checkpoint's headers and no weights. The Spark run is in section 6: fleetq job #8376, 2026-11-04, binary `f4e6fed89` against baseline `03f8dc2a3`.

| | Claim | Verdict | Fix |
|---|---|---|---|
| 1 | Rank 1 stays up when the primary dies with work parked | **Verified.** The "about 2 s" is too low, see below | `drain()` for `Dsv41Runtime`, on stream 0 |
| 2 | A near-cap greedy request kills both ranks | **Verified, and wider than claimed** | No drafting round where its block does not fit |
| 3 | A rank-0-only error silently desynchronises the group | **Verified in the code path.** The wrong sums or the hang are inferred, not reproduced | Any error after the pass was sent fails the group. Also: a failed `reset` |

## 1. Rank 1 does not leave when the primary dies

Trace. `impl RankModel for Dsv41Runtime` had no `drain()`, so `drain_failure` (`tp_remote.rs:641`) returned `None` and `serve_inner` returned `Ok` on the end of the stream, then dropped the sequences. `forward_rank` is `self.forward(st, ids, false)`, so `Want::Nothing`: `finish_pass` returns without a sync. `draft_rank` is also `Want::Nothing` on a rank other than 0. `verify_rank` is `Want::Ids`, a download, so it blocks inside the pass, where `watch_peer` acts. So non-greedy decode (spec needs `temperature <= 0`, the API default is 1.0) is enqueue-only and leaves `watch_peer` nothing to act on. Greedy has `draft_rank` and the k = 0 pass exposed, verify passes covered.

Why the earlier tests missed it: `failtest.sh`'s `req_json` always sends `temperature: 0`, so a2 and a3 (exit 87 at 3.4 and 2.2 s, section 5 of `2026-11-02-dsv41-rdma.md`) ran greedy. #290 fixed `Model` and its test named the DSv4.1 runtime's passes "synchronous" (`a_rank_with_nothing_queued_does_not_wait`), which is not true of these passes. That text is corrected.

Stream 0 is right, checked by reading: collectives take stream 0 (`extra_inner`), a replay is `graph_replay(h, 0)`, `join_lanes` makes stream 0 wait for the lanes, and the prefetch stream 1 is joined with `stream_wait(0, 1)` at the end of a segment. The serving thread is the only one enqueuing and it is the one waiting, so nothing races the drain.

Timing correction. The drain limit is 2 s and `abort_group_and_exit` then waits up to 3 s for `ncclCommAbort`, which does not return with the other rank gone. So expect 5 s or more, not 2 s; the same path on the 35B took 7.2 to 7.8 s. The "34 to 180 s" before the fix is from reading (34 s link retry window in section 5, 180 s default ring deadline in `docs/tuning.md`), not measured for this case.

## 2. A greedy request near the capacity

Trace, at capacity `cap`, block 5 (`dspark_block_size`), sequence length `L` with the next token not yet in it. `next_rows`: `room = cap - (L + 1)`, speculation when `room >= 2`, so `L <= cap - 3`. `Worker::draft` sends `start_draft`, then `draft_steps(st, first, steps)`. **Nothing clamps:** `steps` only clamps the Markov steps, and the check uses the block, `pos0 + t > cap` with `pos0 = st.len`. So `L = cap - 4` and `L = cap - 3` give `InvalidCall` on both ranks. Rank 1 answers the failure and leaves (exit 87). Rank 0's reader sees the failure and marks the engine failed, so rank 0 exits 87 too.

Wider than the claim: both `cap - 4` and `cap - 3` fail, and it does not take a near-cap prompt. `generate` admits any `prompt + max_tokens <= cap`, so a greedy reply growing towards the capacity can land on those positions: from `L = cap - 5` a round with k = 0 goes to `cap - 4`. Whether it lands there depends on the k and rejections of the rounds before.

Fix: `Dsv41Runtime::draft_fits(len)` is now the one check, used by `draft_steps` and by `next_rows` (`room >= 2 && draft_fits(len)`). Near the capacity a step is the plain one-row pass that `L = cap - 2` already took.

## 3. An error on rank 0 only, mid-pass

Trace. `Worker::forward`, `draft` and `verify` send the pass, then run rank 0's half. `settle` handled `Error::Backend` by marking the peer unusable, and `group_failed` then fails the group. Any other error drained the answer, which is OK because rank 1 only enqueued (`forward_rank` replies as soon as the launches are queued). `group_failed` returned false, `reset` ran and the server stayed healthy, with rank 1 parked in the collectives of a pass rank 0 will never join. A multi-chunk prefill is the same: rank 1 got all the ids and enqueued all the chunks.

Which errors can do this on rank 0 alone, after the send:

| Error | Source | Rank-local? |
|---|---|---|
| `Error::Memory` | `cudaMalloc` (`intarsia-backend-cuda/src/device.rs:269`) from `grow_scratch` (`run_stage`, and `capture` before its recording) | Yes: memory pressure on that box; also rank 0 alone runs the Markov steps |
| `Error::Io` | Engram read on that rank's NVMe: `store.rs` is `io::Result`, `tier.rs` `s.poll(t)?` and `wait_records(..)?`, reached from `engram_records` and `issue_engram`. **Not in the claim** | Yes |
| `Error::Model`, `InvalidCall`, `NoKernel` | `bind`, `view`, kernel argument checks | Deterministic and usually on both ranks, but the Markov steps are rank 0's alone |
| `BadRequest`, `InvalidCall`, `Unsupported` | the runtime's own checks (capacity, block) | No: rank 1 raises the same and leaves, so the group was already dead |

So no error after the send ever left the group usable: either rank 1 failed too and left, or it parked. Failing the group on all of them gives up nothing.

The consequence as far as reading goes, not reproduced: `reset` sends `Drop` and `New`. On rank 1 the drop's `cudaFree` waits for the parked stream (the stack #290 saw). Rank 0's next pass pairs with the parked collectives, which frees rank 1, whose new pass then pairs with rank 0's remaining ones, and the two stay shifted. Equal sizes give wrong sums without an error, and different sizes give a hang or an NCCL error.

Fix: `settle` marks the peer unusable and does not wait, for any error from this rank's half, and returns a non-backend error as a backend error, so `group_failed` fires with its code unchanged. The call sites did not change.

Same class, found on the way: `reset` sets `remote = None`, then allocates a new state while still holding the old one. If that allocation fails, the engine stayed healthy with no `remote`, and the next request would run its collectives with rank 1 never told. A failed `reset` now fails the group.

What still does not fail the group, and how that was established:

- **Request validation** (`generate`, before the first send). Test: the far end records no pass from the refused requests, `failed` stays empty, and the next turn reuses the cached sequence (`a_request_refused_before_anything_was_sent_changes_nothing`).
- **A rejected truncate.** The acknowledgement says rank 1 changed nothing (`ack_status`), and the reader only treats `ACK_FAILED` as failure. By reading.
- **Errors between passes** (constraint, decoder). By reading: rank 0's last pass read its logits or ids back, which is a device sync, so all its collectives completed and rank 1's matching ones can. The whole warm-up and decode path reads back. Prefill chunks before the last do not: a cancel between chunks can leave their collectives running, all of them paired, because rank 1 has answered each chunk it was told. The next pass queues behind them on both ranks. That they pair is by reading; the Spark case `cancel` is what can show it.
- **A failed graph capture on one rank** is still swallowed (graphs off, eager from then on, as before). Nothing runs during a recording and the eager pass issues the same collectives from the same graph. By reading only.

## 4. Second review (`dsv41_engine.rs`, `state.rs`, the server)

A second review gave 14 findings, 11 distinct defects once duplicates are merged. Each was fixed only after reading the code it points at; none was refuted. One is left open on purpose.

| Finding | Effect before | Fix | Test (fails with the fix reverted) |
|---|---|---|---|
| Ring rollback counted from `len`, not the furthest row written | Chained cuts read overwritten window and compressor rows: **wrong output, no error**. Two requests are enough, greedy or not | `SequenceState::ring_high`; a cut must stay within `margin` of `len` and `margin + 1` of the mark, otherwise reuse is refused | `after_dropped_draft_rows_the_band_of_18_or_more_is_refused`, `two_requests_without_speculation_are_refused_too` |
| The engine never ticked `steps_total` | The stall watchdog (`--watchdog-s`) fired on any request longer than the limit | `telemetry.step()` after every pass; the prompt now goes one chunk a pass so a prefill ticks too | `every_pass_ticks_the_watchdog_each_prefill_chunk_too` (see section 5 for what a tick means) |
| A failed request left `running = 1` | An idle server read as stalled | `request_finished("error", ..)` in `run` | `a_failed_request_closes_its_telemetry` |
| A validation error reset the cached sequence | The next turn lost its prefix | `BadRequest` from before the first send keeps it | `a_request_refused_before_anything_was_sent_changes_nothing` |
| Cancel not honoured while queued or during prefill | A departed client held the engine for the whole prefill | Checked on entry and between chunks | `a_request_cancelled_while_queued_touches_nothing`, `a_cancel_between_prefill_chunks_leaves_both_ranks_in_step` |
| `stop_sequence` missing or wrong for a match over several tokens | Wrong field in the reply | `stops.matched()`, as `engine.rs` | `the_stop_sequence_reported_is_the_one_that_fired` |
| `ignore_eos` also turned off `stop_ids` | The reply ran past a requested stop id | `stop_ids` always stop, as `Seq::build` | `ignore_eos_keeps_stop_ids` |
| Stop strings active before `min_tokens` | The reply ended early | No matching until token `min_tokens + 1`, as the main engine | `stop_strings_wait_for_min_tokens` |
| `max_tokens` above the room left was an error | HTTP 500 for a client's fixed limit | Clamped, finish reason `length`, as `Seq::admit` | `max_tokens_is_clamped_to_the_room_left` |
| A taken port restarted a supervised server for ever | The whole model reloaded on every restart | The child exits `EXIT_FATAL` on a bind failure, after the retry of section 5 | `a_port_that_stays_taken_stops_the_supervisor_after_the_retry` |

The ring limit comes from the kernels, not from reading: `the_ring_check_accepts_exactly_what_the_kernels_survive` runs the reference kernels at ratios 1, 2 and 3 and compares them with `SequenceState`'s check at 1 to 24 rows below the mark: 17 is bit-identical to never drafting, 18 is not, and the check agrees at each.

**Left open:** finding 7, overlapping lane scratch slices (`runtime.rs:1138`). `wq_a` and `wkv` in one fork need different scratch sizes, so their declared slices can overlap on a GPU with more than 50 SMs. It is reachable only with a build for another architecture: the default build targets `sm_121` and the device check refuses others, and GB10 at 48 SMs has a 7168-byte gap at one row. This comes from the review's reading and arithmetic; I did not run it. It needs a fix before DSv4.1 runs on such a GPU.

## 5. Third review

An Opus review of this branch, also by reading.

**Prefill ticks and cancels followed the enqueue.** Chunks before the last are `Want::Nothing`, so the ticks added in section 4 said what the host queued. At 18 to 28 ms an enqueue and 0.84 s a 512-row chunk (the review's numbers), a 65000-token prompt then blocks about 106 s in the last read-back, past the 60 s watchdog. Whether launches queue that deep cannot be read from the code, so the fix does not depend on it: `Worker::prefill` waits for stream 0 every `PREFILL_SYNC_ROWS` = 4096 rows.
- Why 4096: 8 default chunks, about 6.7 s at 0.84 s a chunk, so a device nine times slower still passes the watchdog. A larger `INTARSIA_DSV41_CHUNK` waits every chunk.
- Why not a non-blocking query: `Backend` has none, and adding one touches every backend.
- Cost: by arithmetic, at most one enqueue of device idle per wait, about 0.4 %. **Not measured.**
- Rank-local: no collective is added, and `a_long_prefill_ticks_with_the_device_not_the_enqueue` pins that the same 127 passes reach rank 1 with the same cuts. It runs on a virtual device clock, so it models the device rather than measuring one. Spark case `supervised` is the real check; it reports INCONCLUSIVE if the prefill stays under the watchdog.

**Every bind failure was fatal.** A supervised server now retries `EADDRINUSE` and `EADDRNOTAVAIL` for 120 s (`INTARSIA_BIND_RETRY_S`) with the model loaded, then exits 78. 120 s outlasts the supervisor's own 30 s grace for a stopping server. Other bind errors and an unsupervised server fail at once. The restart case is tested with a listener that holds the port and lets go, not with a real restart. The watchdog now starts before the bind, so during the retry a failed group still exits 87 and an orphan still exits 88; a retry that ends in the watchdog's 2 s flush also exits 87, not 78.

**An engine panic could hang rank 1.** The panic handler now calls `group_failed`, so the peer is dead before `Worker` drops and no `T_DROP` is sent. Rank 1 leaves on the hang-up through `drain`. Warm-up runs on the loader's thread, so it has its own catch with the same route. `group_failed` changes its state first and its log line cannot panic on a closed stderr (a child with a dead stderr pipe tests it). Tested at the message level: no free reaches the far end. Rank 1's actual exit after a panic is not tested.

**Recorded, not fixed:** `ring_high` does not count draft-block writes (`graph.rs:903`, `runtime.rs:640`, `state.rs:270`). `draft_steps` writes the DSpark stage rings without a commit. The same is true on main. By the review's reading it only affects draft acceptance, since verify rows decide the tokens; I read the lines and did not test it.

**Tests the review called weak:** `next_rows_drafts_exactly_where_the_block_fits...` spells the limit out and goes through `next_rows`; the ring test now calls the check; the cancel test is renamed to what it shows, the message order, and the pairing is left to Spark case `cancel`.

Each fix reverted on the final code (#8306): without the wait, 3 tests fail, one with 106.6 s without a tick against the 60 s watchdog; without the retry, all 3 of `serve_bind`; without the panic route, the panic test; without the ring check, 3 tests, one with the check accepting 18 rows the kernels do not survive. All pass with the fixes: 32 `failpath_tests`, 3 `serve_bind`, 9 `dsv41_chunked`, 40 server `api` tests.

Last round (#8335, #8336), each fix reverted: warm-up panic route, closed-stderr log, watchdog before the bind, and the 87 at the end of a retry each fail their own test.

## 6. Spark run (#8376, 2026-11-04 08:29 to 09:17 EST)

Main at `03f8dc2a3` (PR 296) was merged first; fix binary `f4e6fed89` (the later commits change a test and docs only), baseline `intarsia.base` built from `03f8dc2a3`. Both at `INTARSIA_DSV41_CHUNK=512`. No KILLED line from the 5 GiB watcher; lowest MemAvailable 6.7 GiB on spark1, 6.5 GiB on spark2. Raw: `.scratch/failpaths-run1/` (not committed).

| Case | Expected | Result | Verdict |
|---|---|---|---|
| near | 200, ranks up at cap-5 to cap-2 | All 7 requests HTTP 200, finish `length`, /health 200, both ranks up. **Baseline at cap-4: HTTP 500, /health 503, ranks exit 87 at 7.0 and 7.4 s** | Pass; defect 2 reproduced |
| force | 500, 503, 87, 87 | 3 of 3. Rank 0 exits at 5.0 to 5.6 s, rank 1 at 10.0 to 10.7 s (from the request's start) | Pass |
| ring | Fix's ids equal a cold run's | Fix: identical, reuse refused (cached 0). **Baseline: reuses 199 tokens and differs from the cold run at the first token (237 against 225)**: the wrong output of section 4, on real kernels | Pass; defect reproduced |
| cancel | Up, 200, ids equal a cold run's | Retry after a mid-prefill cancel: cached 2048, identical to cold, /health 200, both ranks up | Pass |
| supervised | 200, no restart | 65000-token prefill 88 s, 3500-token reply 97 s, both past the 60 s watchdog, HTTP 200, 0 restarts | Pass; the baseline control (`BASE_SUPERVISED`) was not run |
| golden | Identical | 7 texts (greedy, 2 seeded at temperature 0.7, a 9000-word prefill) and 2 token-id runs (200 and 12000 tokens) identical, non-empty | Pass |
| kill | 87, about 8 s | Fix, 20 kills at temperature 0.7: **18 exit 87 at 6.8 to 7.5 s, 2 exit 0 at 2.1 and 2.9 s**. Baseline: exit 0 at 183.3 and 182.7 s | Fix holds; defect 1 reproduced; see below |

**Kill: 2 of 20 differ from the header's "87".** Those workers log `rank 0 disconnected` with no failure: the kill landed with nothing parked, the drain finished, and they exit 0. That is the rule #290 set ("a primary that left nothing stuck still gives exit 0", decision of 2026-11-03 17:45), and the GPU is free sooner. The script header's expectation was too narrow. A process manager that restarts rank 1 only on a non-zero exit would not restart it in those runs. The baseline holds the GPU for 183 s, about the 180 s ring deadline, and also exits 0.

**Prefill cost of the device wait: none measurable.** 24000-token cold prompts, four alternating rank starts, 4 runs an arm: baseline mean 32.09 s (32.06 to 32.13), fix 32.10 s (32.08 to 32.16), +0.03 %, inside the spread. This includes the one message per chunk. The chunk comes from the environment on both; the response's `prefill_chunk` field is empty on this engine.

**Interactions with PR 296.** Conflicts in `server/lib.rs`, `tests/api.rs` and `LOG.md`; all kept.
- Watchdog: main's `wants_watchdog` decides whether it starts and this branch keeps it before the bind, so `exit_on_failed` covers the retry too.
- Sticky faults: `fatal()` and `is_sticky_fault` are in `LocalEngine`; `Dsv41Engine` has its own `group_failed` and does not call them. At world 1 a `Backend` error on DSv4.1 still does not fail the engine. No conflict, but the two rules differ.
- Vocabulary: main refuses out-of-range ids in `LocalEngine::submit`, which DSv4.1 does not use. With this branch such an id past the send would fail the group, so `generate` now refuses it first. Tested with the recorder, not on a Spark.

## Tests

`crates/intarsia-models/src/dsv41_engine.rs`, `mod failpath_tests`, 30 tests after the third review (section 5), no GPU: a real `Worker` and `Dsv41Runtime` facing a real `RemotePeer` and `tp_remote::serve`, whose far end records what it was told. The fake backend adds a failing `alloc` and a parked stream 0. Rank 1's exit runs in a child process through the real `serve_group`. Each row reverts one fix on the final code, with the tests in place (job #8290; `INTARSIA_TP_REMOTE_TIMEOUT_S=3`, because the old `settle` waits for a reply).

| Fix reverted | Tests failing without it | Message | With the fix |
|---|---|---|---|
| #1 `drain()` | `rank_1_leaves_the_group_when_the_primary_is_gone_and_stream_0_is_parked`, `the_runtime_drains_stream_0` | exit 42 (the loop returned) instead of 87 | pass |
| #2 `draft_fits` in `next_rows` | `near_the_capacity_a_greedy_step_is_a_plain_pass_not_a_drafting_round` | `[New, Draft]` instead of `[New, Forward]` | pass |
| #3 `settle` | `a_rank_local_error_after_the_pass_was_sent_fails_the_group`, `a_failure_inside_the_runtime_fails_the_group`, `settle_does_not_wait_for_the_other_rank_after_a_failure_and_the_error_is_a_backend_error` | `failed` stays empty | pass |
| #3 failed `reset` | `a_reset_that_fails_fails_the_group` | `failed` stays empty | pass |
| none | | | 24 of 24 at that run; see section 5 for the final count |

The section 4 fixes were reverted the same way, one at a time, in jobs #8277 and #8278; each test in its table failed without its fix.

The tests no row names pin what must not change. The one exception to "the test catches the defect end to end": the near-cap test checks what rank 1 is told, not an actual exit 87, because the far end is a recorder. The refusal itself is pinned by `next_rows_drafts_exactly_where_the_block_fits...`, and the exit is what the Spark case `near` shows.

`cargo test --workspace --no-fail-fast`: 1291 passed, 0 failed, 29 ignored (#8374, exit 0, at `2f533701e`, after merging main). Earlier: #8368 did not compile (the merge dropped a closing brace in `tests/api.rs`, fixed); #8266 failed on this lane's own missing `docs/tuning.md` rows, fixed; #8308 was stopped by fleetq (exit -15) and gives no result.

## The normal path

`git diff --stat 5786176be -- crates`: 9 files changed, 1042 insertions(+), 41 deletions(-). Of those, 697 lines are `failpath_tests`, and `dsv41_chunked.rs` and `serve_bind.rs` are tests too. No file under `kernels/`, no `graph.rs`, no `tp_remote.rs` beyond a test's message.

Normal-path lines that changed, and the evidence for each:

| Line | Evidence |
|---|---|
| `settle`'s `Ok` arm: `r.finish()`, was `let rr = r.finish(); if out.is_ok() { rr }` | Same call in the same order. `settle_on_a_good_pass_waits_for_the_answer_as_before` |
| `next_rows`: `room >= 2 && draft_fits(len)` | `where_the_block_fits...`: rank 1 is told to draft at 1, 2, 8, 32, cap-6, cap-5. `next_rows_drafts_exactly_where_the_block_fits...`: through `next_rows` at every position below 64, a round exactly at 1 to 59, and `draft_steps` refuses the limit spelled out in the test |
| `draft_steps`: `!draft_fits(pos0)`, was `pos0 == 0 \|\| pos0 + t > cap` | The same test |
| `forward`: `test_fault().and_then(..)` | Unset it is a match on `None`, then the same call. Read in `load` after the warm-up, on rank 0 only |
| Prompt sent chunk by chunk (`Worker::prefill`) | `Dsv41Runtime::forward` cuts `min(left, chunk)` and asks logits only of the last, so each rank runs the same passes; only the messages between the ranks change. `a_prompt_is_cut_into_the_runtimes_chunks_and_only_the_last_reads_logits` pins the cuts. Each chunk now also waits for rank 1's answer: **the prefill speed effect is not measured** |
| `ring_cut_ok` in `restorable_len` and `check_ring_cut` | `the_normal_paths_are_unchanged`: one cut as before, 2000 rounds of verify rejections never refused. Reuse is refused only in the band, which costs a full prefill there |
| `telemetry.step()` after each pass | Counter only |

Only DSv4.1's `graph.rs` emits `SlotSpec::Ring`, so `ring_cut_ok` never runs for another model: `ring_margin()` is `None` there. For those models the evidence is that plus the workspace suite. The real `compress_ratios` are 0 and 2 in the Mia config, so the ring test's ratios 1, 2 and 3 cover the one in use.

This is control flow, and those tests stop before any kernel runs: **no bitwise numeric check has run.** The gates that would, `speculative_greedy_equals_plain_greedy` and the rest of `tests/dsv41_runtime.rs`, skip without a rank cache and fixtures, which endor does not have. The Spark case `golden` compares fix and main.

## Spark script

`scripts/dsv41-tp2/failpaths-test.sh`: with no argument it prints the two builds and the fleetq command and runs nothing. Under `run-full.sh` as its `HOOK`, vLLM goes down and is always restored, with memfine and the watcher at the 5 GiB line. Cases: `near` (#2), `force` (#3, `INTARSIA_DSV41_TEST_RANK0_FAIL_PASS`), `ring` (chained cuts), `cancel`, `supervised`, `golden` (main against fix), `kill` (#1: 20 SIGKILLs of rank 0 at random times mid-decode, temperature 0.7, default deadline, rank 1's exit code and seconds in `kill.tsv`, plus 2 on main), and optionally `kill35`.

`build.sh` supports `BIN_NAME` and its `--delete` only touches `src/`; `run-full.sh` restores vLLM from an EXIT trap, so an `exit 1` here still restores it (both by reading).

Before the first real run, checked: `bash -n`, the printed commands, `--dry-run` (all cases through a stubbed ssh and curl, with `bash -n` on every remote command it produced, and an assertion that the forced fault stays in case `force`), and `fleetq when` accepting the job's flags (estimated start Wed 07:05). It has now run (section 6): about 40 s a rank start, not the guessed 4.5 min.

## Not checked

- The supervised case's control: the baseline against the same 65000-token prefill.
- `kill35`: not in this run.
- A real `cudaMalloc` failure on rank 0. The force case injects one level above `grow_scratch`.
- Rank 1's exit after an engine panic, on a GPU.
- `ring_high` not counting draft-block writes (section 5): recorded, not tested.
- Finding 7, lane scratch overlap (section 4): open, only reachable on another architecture.
- Bitwise numerics beyond golden's 9 outputs, and any chunk other than 512.

Also not done: no 35B change, and PR #293's files were left alone (`engine.rs`, `model.rs`, `tp_hybrid.rs` untouched here).

## Gates (no-op check against origin/main 809674eb9; branch head 3dc076b23)

No-op gates for the shared-file changes (`plan.rs`, `state.rs`, `tp_remote.rs`, `serve.rs`, server `lib.rs`). Scripts are the engine-localfail ones with paths and SHAs changed (`raw/gates/scripts/`), plus two differences: `gates-lf.sh` has its `serve` section removed (see Not run), and `gates-cuda.sh` waits for its remote builds with `jobs -r` (the original polled zsh-indexed PIDs under `bash -s`). Logs in `raw/gates/`.

| gate | job | result |
|---|---|---|
| Metal builds, `cargo build --release --locked -p intarsia-cli --features metal`, own target dir per binary (branch head, detached `origin/main` worktree) | #8395, endor, `--load cpu` | ok |
| `plan show --backend metal` text and `--json`, 35B (Qwen3.6-35B-A3B-NVFP4), 8B (Qwen3-8B), 27B (Qwen3.8-27B-Quark-AWQ-MXFP4), base vs new | #8397, endor, `--exclusive --mem 24` | **byte-identical, 6 of 6** |
| `kernels test --backend metal`, base and new | #8397 | 49 kernels, 980 cases, **0 FAIL** both; logs differ only in the wall-time line (31.4 s vs 31.2 s) |
| `generate` greedy, 256 ids (`--ignore-eos --spec off`), short chat prompt and 4K prompt, 27B and 8B | #8397 | **ids and text identical, 4 of 4 cells** |
| 8B goldens `qwen3-8b-fp32` (16 decode positions) and `qwen3-8b-long-fp32` (64), `accuracy --mode both` against the metal baselines | #8397 | both floor PASS, baseline PASS, golden gate PASS; `.out` of base and new byte-identical |
| CUDA on spark1: `plan show` 35B and 8B, text and `--json`; `kernels test --backend cuda --quick` | #8399, `--load cpu --mem 60` | `plan show` identical 4 of 4; 106 kernels, 918 cases, **0 FAIL** both; logs identical apart from wall time |
| `cargo test --workspace --no-fail-fast` at 3dc076b23 | #8396, endor, `--load cpu` | exit 0: 1292 passed, 0 failed, 29 ignored |

CUDA job #8394 (first attempt) is not a result: the script's wait loop used zsh array indices under `bash -s`, so it read the binaries before the branch build finished (`No such file`, exit 127 on the new side). Kept in `raw/gates/cuda-run1-failed/` and `raw/gates/logs/cuda.log`; #8399 is the same script with the loop fixed.

Not run: `serve` (LocalEngine and batched, HTTP and out-of-vocab checks) was not part of this request, and the failure paths on a Spark are covered by section 6, not here; a base-vs-new comparison on other models than 35B, 8B and 27B.
