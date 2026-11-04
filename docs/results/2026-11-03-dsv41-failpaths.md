# DSv4.1 TP=2: three failure paths (lane dsv41/failpaths, 2026-11-03)

Branch `dsv41/failpaths` from origin/main `5786176be`, local only. **Nothing here has run on a GPU or a Spark.** The tests run the real `Worker` and `Dsv41Runtime` on the cpu backend on endor (macOS), with graphs from the Mia checkpoint's headers and no weights. `scripts/dsv41-tp2/failpaths-test.sh` is written and dry-run only.

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

- **Request validation** (`generate`, before the first send). Test: the far end records no pass, `failed` stays empty, and the peer still works.
- **A rejected truncate.** The acknowledgement says rank 1 changed nothing (`ack_status`), and the reader only treats `ACK_FAILED` as failure. By reading.
- **Errors between passes** (constraint, decoder). By reading: rank 0's last pass read its logits or ids back, which is a device sync, so all its collectives completed and rank 1's matching ones can. The whole warm-up and decode path reads back; only the non-last prefill chunks do not, and they are inside `rt.forward`.
- **A failed graph capture on one rank** is still swallowed (graphs off, eager from then on, as before). Nothing runs during a recording and the eager pass issues the same collectives from the same graph. By reading only.

## Tests

`crates/intarsia-models/src/dsv41_engine.rs`, `mod failpath_tests`, 13 tests, no GPU: a real `Worker` and `Dsv41Runtime` facing a real `RemotePeer` and `tp_remote::serve`, whose far end records what it was told. The fake backend adds a failing `alloc` and a parked stream 0. Rank 1's exit runs in a child process through the real `serve_group`. Each row reverts one fix with the tests in place (`--expect 12`, job #8265; `INTARSIA_TP_REMOTE_TIMEOUT_S=3`, because the old `settle` waits for a reply).

| Fix reverted | Tests failing without it | Message | With the fix |
|---|---|---|---|
| #1 `drain()` | `rank_1_leaves_the_group_when_the_primary_is_gone_and_stream_0_is_parked`, `the_runtime_drains_stream_0` | exit 42 (the loop returned) instead of 87 | pass |
| #2 `draft_fits` in `next_rows` | `near_the_capacity_a_greedy_step_is_a_plain_pass_not_a_drafting_round` | at 60 of 64 tokens: `[New, Draft]` instead of `[New, Forward]` | pass |
| #3 `settle` | `a_rank_local_error_after_the_pass_was_sent_fails_the_group`, `a_failure_inside_the_runtime_fails_the_group`, `settle_does_not_wait_for_the_other_rank_after_a_failure_and_the_error_is_a_backend_error` | `failed` stays empty | pass |
| #3 failed `reset` | `a_reset_that_fails_after_a_refused_request_fails_the_group` | `failed` stays empty | pass |
| none | | | 13 of 13 |

The other 6 pass in every arm, on purpose, because they pin what must not change. The one exception to "the test catches the defect end to end": the near-cap test checks what rank 1 is told, not an actual exit 87, because the far end is a recorder. The refusal itself is pinned by `draft_fits_is_exactly_what_the_runtime_accepts...`, and the exit is what the Spark case `near` shows.

`cargo test --workspace --no-fail-fast` (#8266): 1241 passed, 29 ignored, 1 failed. The failure was this lane's own: `docs/tuning.md` rows for the two new variables (`docs_env_vars`). Fixed and that test re-run green (#8273).

## The normal path

`git diff --stat -- crates`: 3 files changed, 504 insertions(+), 17 deletions(-), of which all but about 60 lines are the test module. No file under `kernels/`, no `graph.rs`, no `tp_remote.rs` beyond a test's message.

Normal-path lines that changed, and the evidence for each:

| Line | Evidence |
|---|---|
| `settle`'s `Ok` arm: `r.finish()`, was `let rr = r.finish(); if out.is_ok() { rr }` | Same call in the same order. `settle_on_a_good_pass_waits_for_the_answer_as_before` |
| `next_rows`: `room >= 2 && draft_fits(len)` | `where_the_block_fits...`: rank 1 is told to draft at 1, 2, 8, 32, cap-6, cap-5. `draft_fits_is_exactly...`: at every position 0..=64, `draft_fits` is exactly what `draft_steps` accepts, and every position where a round could run before still runs one |
| `draft_steps`: `!draft_fits(pos0)`, was `pos0 == 0 \|\| pos0 + t > cap` | The same test |
| `forward`: `test_fault().and_then(..)` | Unset it is a match on `None`, then the same call. Read in `load` after the warm-up, on rank 0 only |

This is control flow, and those tests stop before any kernel runs: **no bitwise numeric check has run.** The gates that would, `speculative_greedy_equals_plain_greedy` and the rest of `tests/dsv41_runtime.rs`, skip without a rank cache and fixtures, which endor does not have. The Spark case `golden` compares fix and main.

## Spark script

`scripts/dsv41-tp2/failpaths-test.sh`: with no argument it prints the two builds and the fleetq command and runs nothing. Under `run-full.sh` as its `HOOK`, vLLM goes down and is always restored, with memfine and the watcher at the 5 GiB line. Cases: `near` (#2), `force` (#3, `INTARSIA_DSV41_TEST_RANK0_FAIL_PASS`), `golden` (main against fix), `kill` (#1: 20 SIGKILLs of rank 0 at random times mid-decode, temperature 0.7, default deadline, rank 1's exit code and seconds in `kill.tsv`, plus 2 on main), and optionally `kill35`.

Checked: `bash -n`, the printed commands, `--dry-run` (all cases through a stubbed ssh and curl, with `bash -n` on every remote command it produced), and `fleetq when` accepting the job's flags (estimated start Wed 07:05). **It has never run on a Spark**, so its timings (about 4.5 min a run, `--expect 180`) are guesses, and at that length it cannot fit before vLLM is due back on a morning that needs it.

## Not checked

Only Sparks can show:

- Rank 1's real exit code and time on a parked CUDA stream: that `cudaStreamSynchronize` blocks rather than erroring, with NCCL and with the RDMA ring, and how often the watch gets there first.
- Numerics of the fix against main (`golden`).
- A real near-cap request end to end, including the plain pass's numbers at the last positions.
- The forced rank-0 failure end to end, and how rank 1 leaves when its parked collectives never pair. The wrong sums and the hang of section 3 are inference; the fix is meant to make them unreachable, and no run has shown them happening.
- Whether a real `cudaMalloc` failure on rank 0 arrives as `Error::Memory` under memory pressure, rather than the watcher or earlyoom killing the rank first. The injection is one level above `grow_scratch`.
- The tests on Linux. They ran on macOS, where `watch_peer` is compiled out, so the watch thread never ran beside these tests.

Also not done: no 35B change, and PR #293's files were left alone (`engine.rs`, `model.rs`, `tp_hybrid.rs` untouched here).
