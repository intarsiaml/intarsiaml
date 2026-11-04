# Single-sequence engine and server failure paths (2026-11-03)

Branch `engine/localfail`, from origin/main `1d6a20799`. Findings: review 3 (`.scratch/review3-findings.json`, code reading
only). Each finding was treated as a claim: a test that fails before the fix, then the fix. Tests run on the cpu backend:
`Flaky` (a cpu backend that fails launches on demand) in `crates/intarsia-models/tests/engine_localfail.rs`, the 2-rank
host-communicator TP setup for the group cases, and a mock engine in `crates/intarsia-server/tests/api.rs`.

| # | Finding | Verdict | Fails before the fix | Fix |
|---|---|---|---|---|
| 0, 11 | Prompt id >= vocab aborts the TP communicator for good; `/health` stays 200 | verified | `out_of_vocab_prompt_leaves_the_tp_group_working` ("communicator aborted (another rank failed)" on the next normal request); `out_of_vocab_prompt_is_a_bad_request` (submit accepted the id) | `LocalEngine::submit` rejects ids >= vocab as `BadRequest` before the request queues, any forward or any cache change (the `Seq::new` copy of the check was removed in round 2: `submit` is the one place every request passes, and no test could tell the copy from its absence); `start()` already maps a submit `BadRequest` to 400, so the non-streaming 500 and the streaming 200-plus-error-event of finding 6 end too |
| 6, 13 | `as u32` wraps ids >= 2^32 (4294967301 ran as token 5) | verified | `completions_token_ids_beyond_u32_are_a_400` (200 before) | `parse_prompt` uses `u32::try_from`, 400 otherwise. The 500 half of finding 6 is the engine half above; `out_of_vocab_prompt_id_is_a_400_from_a_real_engine` (a real `LocalEngine` over the tiny hybrid fixture, `crates/intarsia-server/tests/prefix_store.rs`) pins the 400 for stream and non-stream; the mock-based test of round 1 passed without the fix, because the mock did the refusing, and is gone |
| 1 | `Error::Backend` is not fatal in `LocalEngine` (permanent 500 loop, healthy `/health`) | verified (the repro's "mock Backend" is a cpu wrapper here; Metal's error status may not recur, as the finding notes) | `device_error_fails_the_engine`, `queued_requests_drain_after_a_device_error` | `run_job` sets `Shared.failed` on `Error::Backend`, before the error event; queued jobs drain through the existing branch. `memory_error_on_one_device_is_not_fatal` and `bad_requests_are_not_fatal` pin that a `Memory` error without a group and a `BadRequest` stay non-fatal |
| 12 | An aborted in-process TP group leaves the engine "healthy" | verified, with the finding's own corrections: under the supervisor `exit_on_stall` already makes the watchdog exit on a failed engine, so `exit_on_failed` matters only for `--no-supervisor`; `INTARSIA_TP_TEST_BUG` is a scale hook, so the test uses a failing backend instead | `aborted_tp_group_fails_the_engine` (rank 0 failing with `Error::Memory`: error is not a `Backend` error, engine stayed healthy) | `Model::tp_group_aborted` (set by `forward_tp`, `forward_verify_tp`, `forward_remote` when a pass fails and aborts the communicator); `run_job` fails the engine on it whatever the error. `serve` sets `exit_on_failed` for `--tp > 1` as for `--tp-remote`. Beside PR #293: that one marks a lost remote rank from the reader thread (`get_or_insert`, same `Shared.failed`); this one covers a failed pass. The edited `Shared.failed` line is untouched, so the two merge. The remote branch of the flag is not covered by a test (it would sit in `tests/tp_hybrid.rs`, which #293 changes) |
| 7 | Graceful shutdown hangs while an `/events` client is connected | verified | `graceful_shutdown_ends_the_events_stream` (server still waiting 5 s after the signal; checked by disabling the end of the stream) | `/events` ends on the shutdown signal (`Shutdown::after`, used by `serve`); generations in flight still finish. The bind path is untouched |
| 2 | Cancelled queued request wipes the warm cache | verified | `cancelled_queued_request_keeps_the_warm_cache` (follow-up reused 0 tokens) | `generate_one` returns the cancelled outcome before `Seq::new` |
| 3 | Logprobs post-processing on the 2-thread runtime | verified; severity perf, and the finding's literal trigger does not stall on defaults (single-sequence engine serializes the two requests), as the verdict says | `completions_logprobs_do_not_stall_the_async_threads` (current-thread runtime: the timer stood still for the whole request) | the block is `logprobs_json`, run through `blocking_as`. `completions_logprobs_shape` checks the reply is unchanged in shape |

Not touched, by instruction: 4, 5, 8, 9, 10, 14, 15, 16, 17.

Left undone: `forward_logits_each`, `forward_logits_decode` and `forward_script` (accuracy tooling, not server paths) do not check ids
against the vocabulary; `Error::Memory` is still non-fatal in the batched scheduler (`is_fatal`); HIP graph replays skip the host
`aborted` check (finding 12's note).

## Test runs, round 1 (endor, cpu, own target dir, via fleetq #8280)

- `cargo test -p intarsia-models --test engine_localfail`: 8 passed, 0 failed.
- `cargo test -p intarsia-server`: 18 + 42 + 2 passed, 0 failed (`tests/api.rs` 42).
- `cargo test -p intarsia-core`: 168 + 2 + 37 passed, 0 failed.
- `cargo test --workspace --no-fail-fast` at the last code commit: 1243 passed, 0 failed, 29 ignored.

## Review round 2 (Opus review of the branch)

Each item below was a claim from reading code; none was found wrong. Fixes are one commit each (`git log engine/localfail`).

| Item | Verdict | Test that fails without the fix | Fix |
|---|---|---|---|
| 1. A one-off backend error (Metal command buffer) now fails `LocalEngine` | verified: `spec_pipe` resets after one, `fatal` made it a reload | `a_transient_backend_error_does_not_fail_the_engine` | `Backend::is_sticky_fault(&Error)` (default: every `Error::Backend`; Metal: none); `run_job` asks the model's backend |
| 2. Launcher codes that are not device faults arrive as `Error::Backend` | verified by reading `cuda/device.rs`, `hip/device.rs`, `device_quantmoe.rs`, `exl3.rs`: all four mapped any nonzero `ik_status` to `Backend` | `only_a_launch_error_is_a_device_fault` (unit test of the classifier in `intarsia-core`) | `backend::launch_status_error`: `ik_status` 1 to 4 (invalid arg, dtype, shape, stride) is `InvalidCall`, 5 (unsupported) is `Unsupported`, 6 (the runtime's launch error) and unknown codes are `Backend`; the buffer bounds checks (`cuda`/`hip` `ptr`) are `InvalidCall` |
| 3. `tp_aborted` missed the MTP and batched TP paths | verified | `a_failed_mtp_pass_marks_the_group_aborted` (draft and catch-up), `a_failed_tp_batched_step_marks_the_group_aborted` | `Model::note_tp_pass` at every site that aborts the group (list below) |
| 4. `--no-supervisor --tp 2 --watchdog-s 0` never exits | verified: the watchdog was not started, and it performs the `exit_on_failed` exit | `a_failed_engine_exit_needs_the_watchdog_even_without_a_stall_limit` (the predicate; the exit itself calls `process::exit` and is not run) | `wants_watchdog` also true for `exit_on_failed`; `observability.md` stands as written |

How the classification is carried (item 2): no new error type. The existing variants already say what the error is (`InvalidCall` and `Unsupported` for a refused call, `Backend` for a device fault), so the CUDA / HIP launch wrappers return the right one, and the engine asks the backend whether a remaining `Backend` error is sticky (item 1). CUDA runtime codes from `cuda_err` / `hip_err` (cudaMalloc, copies, stream sync) all stay `Backend`, hence fatal: a table of which are non-sticky was not built, so a non-sticky runtime code (an invalid value, say) still fails the engine. The batched scheduler's `is_fatal` is unchanged: a Metal one-off error still fails a batched engine, but a launcher refusal is no longer `Backend` there either.

Abort sites of the communicator (item 3): `model.rs` `forward_tp`, `forward_verify_tp` and `forward_remote` (flag set); `model/mtp.rs` `mtp_catch_up` and `mtp_draft_tp`, and `model/batch.rs` `forward_batch_tp` (flag set now); `tp_remote.rs` reader thread (lost remote rank: the `on_lost` hook of PR 293 marks the engine) and `abort_group_and_exit` (the process exits); `dsv41_engine.rs` (the dsv41/failpaths lane).

Tests that held without their fix, repaired:

- `completions_out_of_vocab_id_is_a_400_stream_or_not` (mock): replaced by `out_of_vocab_prompt_id_is_a_400_from_a_real_engine` (above).
- `aborted_tp_group_fails_the_engine`, rank 1: both ranks now run on `Flaky` and rank 0's backend calls its errors transient, so only the group flag can fail the engine, and the test asserts the "tensor-parallel group aborted" reason.
- The `Seq::new` vocab check: removed.

Checked: the batched engine (`SeqAdapter::admit`) rejected an out-of-vocab id after `submit` had accepted it, so the client got a 500 (non-streaming: `collect` makes every `GenEvent::Error` an internal error) or a 200 stream carrying an error event. `BatchedEngine::submit` now refuses it as a `BadRequest` (400); `admit` keeps its check. Test: `cancel_and_errors_leave_others_alone` (batch_engine.rs, now expects the `BadRequest` from `submit`).

Mutation check (every fix reverted in turn with its tests kept; all failed as listed in `.scratch`, not committed): `a_transient_backend_error_does_not_fail_the_engine`, the two out-of-vocab engine tests, `aborted_tp_group_fails_the_engine`, `a_failed_mtp_pass_marks_the_group_aborted`, `a_failed_tp_batched_step_marks_the_group_aborted`, `cancel_and_errors_leave_others_alone`, the watchdog predicate test, and the real-engine HTTP test.

What the CPU backend could not exercise: Metal's `is_sticky_fault` (a one-line `false`) is covered by the `Flaky` stand-in and by reading; the CUDA / HIP wrappers are covered by the classifier unit test and by the CI check-only cfgs run here (`INTARSIA_CUDA_CHECK_ONLY=1 cargo check --locked -p intarsia-backend-cuda --all-targets` and `-p intarsia-cli --features cuda`, then the same with `INTARSIA_HIP_CHECK_ONLY=1` and `hip`, as `.github/workflows/ci.yml` runs them): they type-check, no kernel was built or run. `cargo check -p intarsia-backend-metal -p intarsia-cli` passes. Not covered anywhere: that a real wedged CUDA / HIP context returns `Backend` rather than one of the codes now reclassified.

Known limits, left as they are:

- An exit within about 2.5 s of the engine failing can drop the response of another request that has just finished (`health.rs`, the watchdog sleeps 2 s before `die`).
- `queued_requests_drain_after_a_device_error` and `completions_logprobs_shape` are weak: the first accepts a rejected submit or an error event for the queued jobs, the second pins only the shape of the reply on a mock engine with one traced token.
- `dsv41_engine.rs` and `state.rs` are untouched.

Test runs, round 2 (endor, cpu, via fleetq #8305 and #8310, no slot held after): `cargo test -p intarsia-models --test engine_localfail --test tp_mtp --test tp_batch --test batch_engine`: 9 + 8 + 15 + 9 passed; `-p intarsia-server`: 19 + 41 + 3 passed; `-p intarsia-core`: 169 + 2 + 37 passed; `cargo test --workspace --no-fail-fast`: 1248 passed, 0 failed, 29 ignored.

## Gates (no-op check against origin/main 1d6a20799; branch head 97cd1f2e8)

Both binaries `cargo build --release --locked -p intarsia-cli --features metal`, own target dirs, from the branch head and a detached `origin/main` worktree (the branch's merge-base is `origin/main`); build #8311 (`--load cpu`, `raw/scripts/build.sh`, `raw/build.log`). Gates on endor, #8312 (`raw/scripts/gates-lf.sh`, log `raw/gates-summary.log`, data `raw/gates/`) and #8316 (`raw/scripts/acc-long.sh`, `raw/acc-long.log`). Recipe and models from the m27 hs-rowtable gates (#8300). No `INTARSIA_METAL_*` env set.

| gate | result |
|---|---|
| `plan show --backend metal` text and `--json`, 35B (Qwen3.6-35B-A3B-NVFP4), 8B (Qwen3-8B), 27B (Qwen3.8-27B-Quark-AWQ-MXFP4), base vs new | **byte-identical, 6 of 6** |
| `kernels test --backend metal`, base and new | 49 kernels, 980 cases, **0 FAIL** both; logs differ only in the wall-time line (36.0 s vs 31.2 s) |
| `generate` greedy, 256 ids (`--ignore-eos --spec off`), short chat prompt and 4K prompt (`prompt-4096.txt`), 27B and 8B, single stream | **ids and text identical, 4 of 4 cells** |
| `serve --max-batch 1` (LocalEngine) and `--max-batch 4` (batched engine), 27B, 256 ids per request: short and 4K one at a time, then short, 4K, 900 and 1800 characters at once | **ids identical, 6 of 6 requests at each batch size** (each equal to base) |
| 8B goldens (`/opt/code/intarsiaml/golden`, `accuracy --mode both` against `golden/baselines/qwen3-8b-fp32` and `-long-fp32` metal baselines) | both floor PASS, baseline PASS, golden gate PASS; `.out` of base and new **byte-identical** (the first long run used `--decode-positions 16`, the baseline has 64: refused by both binaries identically, rerun with 64 in #8316) |
| HTTP, new binary, server up (27B, both batch sizes) | normal completion 200 (256 ids); `{"prompt":[999999999]}` to `/v1/completions` **400** `bad request: token id 999999999 >= vocab 248320`; `/health` 200 before and after; next normal request 200 with ids identical to the first |

Not run: base binary with the out-of-vocab prompt (contrast, not a gate); workspace tests (passed earlier on this head, see above).
