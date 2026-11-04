# Single-sequence engine and server failure paths (2026-11-03)

Branch `engine/localfail`, from origin/main `1d6a20799`. Findings: review 3 (`.scratch/review3-findings.json`, code reading
only). Each finding was treated as a claim: a test that fails before the fix, then the fix. Tests run on the cpu backend:
`Flaky` (a cpu backend that fails launches on demand) in `crates/intarsia-models/tests/engine_localfail.rs`, the 2-rank
host-communicator TP setup for the group cases, and a mock engine in `crates/intarsia-server/tests/api.rs`.

| # | Finding | Verdict | Fails before the fix | Fix |
|---|---|---|---|---|
| 0, 11 | Prompt id >= vocab aborts the TP communicator for good; `/health` stays 200 | verified | `out_of_vocab_prompt_leaves_the_tp_group_working` ("communicator aborted (another rank failed)" on the next normal request); `out_of_vocab_prompt_is_a_bad_request` (submit accepted the id) | `LocalEngine::submit` and `Seq::new` reject ids >= vocab as `BadRequest` before any forward or cache change; `start()` already maps a submit `BadRequest` to 400, so the non-streaming 500 and the streaming 200-plus-error-event of finding 6 end too |
| 6, 13 | `as u32` wraps ids >= 2^32 (4294967301 ran as token 5) | verified | `completions_token_ids_beyond_u32_are_a_400` (200 before) | `parse_prompt` uses `u32::try_from`, 400 otherwise. The 500 half of finding 6 is the engine half above; `completions_out_of_vocab_id_is_a_400_stream_or_not` pins the 400 for stream and non-stream (it passes before too: the mock models the engine's submit refusal, the real one is the engine test) |
| 1 | `Error::Backend` is not fatal in `LocalEngine` (permanent 500 loop, healthy `/health`) | verified (the repro's "mock Backend" is a cpu wrapper here; Metal's error status may not recur, as the finding notes) | `device_error_fails_the_engine`, `queued_requests_drain_after_a_device_error` | `run_job` sets `Shared.failed` on `Error::Backend`, before the error event; queued jobs drain through the existing branch. `memory_error_on_one_device_is_not_fatal` and `bad_requests_are_not_fatal` pin that a `Memory` error without a group and a `BadRequest` stay non-fatal |
| 12 | An aborted in-process TP group leaves the engine "healthy" | verified, with the finding's own corrections: under the supervisor `exit_on_stall` already makes the watchdog exit on a failed engine, so `exit_on_failed` matters only for `--no-supervisor`; `INTARSIA_TP_TEST_BUG` is a scale hook, so the test uses a failing backend instead | `aborted_tp_group_fails_the_engine` (rank 0 failing with `Error::Memory`: error is not a `Backend` error, engine stayed healthy) | `Model::tp_group_aborted` (set by `forward_tp`, `forward_verify_tp`, `forward_remote` when a pass fails and aborts the communicator); `run_job` fails the engine on it whatever the error. `serve` sets `exit_on_failed` for `--tp > 1` as for `--tp-remote`. Beside PR #293: that one marks a lost remote rank from the reader thread (`get_or_insert`, same `Shared.failed`); this one covers a failed pass. The edited `Shared.failed` line is untouched, so the two merge. The remote branch of the flag is not covered by a test (it would sit in `tests/tp_hybrid.rs`, which #293 changes) |
| 7 | Graceful shutdown hangs while an `/events` client is connected | verified | `graceful_shutdown_ends_the_events_stream` (server still waiting 5 s after the signal; checked by disabling the end of the stream) | `/events` ends on the shutdown signal (`Shutdown::after`, used by `serve`); generations in flight still finish. The bind path is untouched |
| 2 | Cancelled queued request wipes the warm cache | verified | `cancelled_queued_request_keeps_the_warm_cache` (follow-up reused 0 tokens) | `generate_one` returns the cancelled outcome before `Seq::new` |
| 3 | Logprobs post-processing on the 2-thread runtime | verified; severity perf, and the finding's literal trigger does not stall on defaults (single-sequence engine serializes the two requests), as the verdict says | `completions_logprobs_do_not_stall_the_async_threads` (current-thread runtime: the timer stood still for the whole request) | the block is `logprobs_json`, run through `blocking_as`. `completions_logprobs_shape` checks the reply is unchanged in shape |

Not touched, by instruction: 4, 5, 8, 9, 10, 14, 15, 16, 17.

Left undone: `forward_logits_each`, `forward_logits_decode` and `forward_script` (accuracy tooling, not server paths) do not check ids
against the vocabulary; `Error::Memory` is still non-fatal in the batched scheduler (`is_fatal`); HIP graph replays skip the host
`aborted` check (finding 12's note).

## Test runs (endor, cpu, own target dir, via fleetq #8280)

- `cargo test -p intarsia-models --test engine_localfail`: 8 passed, 0 failed.
- `cargo test -p intarsia-server`: 18 + 42 + 2 passed, 0 failed (`tests/api.rs` 42).
- `cargo test -p intarsia-core`: 168 + 2 + 37 passed, 0 failed.
- `cargo test --workspace --no-fail-fast` at the last code commit: 1243 passed, 0 failed, 29 ignored.
