# Metal continuous batching: fatal-failure injection on the real GPU

Date 2026-10-17. Branch `metal-cb-fault-inject` off `spike/metal` d793300 (local, not pushed, nothing merged). Machine endor (Apple M5 Max), Qwen3-8B BF16, `intarsia serve --backend metal --max-seq 8192 --max-batch 8`. Closes the last open M2 item: D3 "error paths", fatal half (`docs/research/continuous-batching-design.md` 2.5, row D3 of `docs/results/2026-10-16-metal-cb-m2-closeout.md`), which had been tested on the mock only (`failure_isolation_and_fatal`, `engine_queue_bound_and_fatal`).

## Verdict

**PASS, both modes, both process models.** All four cases (`err` and `panic` injected, each under the default supervisor and under `--no-supervisor`) passed all 26 checks. The code matches the design text; nothing was changed in the engine, server or scheduler behaviour, and no bug was found.

| Check | err | panic |
|---|---|---|
| Every in-flight and queued request ends with a clean error, none hangs, none truncated (11 of 11) | pass | pass |
| Streamed requests: HTTP 200, then an `error` SSE event, then `[DONE]`; plain requests: HTTP 500 JSON body | pass | pass |
| Later submits refused with 503 JSON, not hung (plain, streamed, chat) | pass | pass |
| `/health` 503 `status: failed`, `running 0 waiting 0` | pass | pass |
| `/metrics` outcome counters match the clients (details below) | pass | pass |
| `--no-supervisor`: process stays up refusing (checked 8 s after the failure) | pass | pass |
| Default supervisor: child exits 87, supervisor restarts, new child serves | pass | pass |
| Fresh restart without the fault: normal plain and streamed request succeed | pass | pass |
| No server process or listener left behind | pass | pass |

Gates: `cargo test -p intarsia-core` 71+2+21 passed, and with `--features fault-inject` 75+5+2+21; `-p intarsia-server` 5+21; `-p intarsia-cli` 80+1; `cargo check --workspace --all-targets` clean with no feature, `--features intarsia-cli/metal`, `intarsia-cli/fault-inject`, and both; `cargo test -p intarsia-backend-metal --release` 110 passed, 0 failed, plus the new `tests/fault_inject.rs` 2 passed with `--features fault-inject`. The `cbidentity` run was not needed, see "Feature off".

## Commits

| Commit | What | Touches `crates/*/src` |
|---|---|---|
| `58c52a9` | the `fault-inject` feature: `crates/intarsia-core/src/fault.rs`, the scheduler hook, the Metal `synchronize` hook, feature wiring in three `Cargo.toml`, tests | **yes**: `intarsia-core/src/{fault,lib,scheduler}.rs`, `intarsia-backend-metal/src/lib.rs`, all `#[cfg(feature = "fault-inject")]` |
| `e9c969a` | test-only fix to `crates/intarsia-backend-metal/tests/fault_inject.rs`: the two tests share the process-wide armed flag and an `upload`'s `synchronize` could consume the other test's fault, so they now run behind one lock (found in review; the first run in #2276 happened to pass) | no (`tests/` only) |
| `28dab7f` | `tools/cbfault_e2e.py` (driver), `tools/cbfault_slot.sh` (slot wrapper) | no |
| this doc and `docs/results/raw/2026-10-17-metal-cb-fault-inject/` | results | no |

## How the hook works

A cargo feature, off by default: `intarsia-core/fault-inject` (the `fault` module is not compiled without it), `intarsia-backend-metal/fault-inject`, and `intarsia-cli/fault-inject` which turns on both (`cargo build --release -p intarsia-cli --features metal,fault-inject`). Armed by the environment variable `INTARSIA_FAULT_INJECT`, read once when the scheduler is built; a bad spec makes the engine refuse to start.

Spec: `<mode>:<trigger>[,waiting=Q][,pos=P]`.

* mode `err`: the Metal backend returns `Error::Backend("metal: command buffer failed: injected fault (fault-inject)")`. `panic`: it panics.
* trigger `after=N`: the decode step after N completed ones. Trigger `mixed=R`: the first decode step with at least R decoding rows while some request is mid-prefill (its next chunk runs in the same iteration).
* `waiting=Q`: also at least Q requests queued behind the full slot set. `pos=P`: also a prefilling request that has already prefilled at least P prompt tokens.

Mechanism: the scheduler owns a `Plan`. Right before each `decode_step` (`scheduler.rs`, the `#[cfg]` block before `self.model.decode_step`) it shows the plan the row count, the prefilling count, the queue length and the furthest prefill position. On a match it prints `fault-inject: arming ...` to stderr and sets a process-wide one-shot flag (`fault::arm`). The Metal `Backend::synchronize` (`intarsia-backend-metal/src/lib.rs`) consumes the flag after its real waits: `err` puts the error where the first failed command buffer's error would be (`first`, the value `synchronize` already returns for a buffer in status Error), so the code above it is the real code; `panic` panics after the stream lock is dropped (no poisoned mutex). The injected failure is therefore a real `Err` from `forward_batch` inside `SlotModel::decode_step`, with the GPU work of that step done, not a mock return. The plan fires once.

Where it surfaces: in the first `synchronize` of the mixed iteration's decode step, before that iteration's prefill chunk runs. The 4k prompt then holds 512 tokens of KV (two 256-token busy chunks) and no first token. Armed at iteration 19 to 22 (`decode rows 7, prefilling 1, furthest prefill 512 tokens, waiting 3`). A genuine GPU error would surface at the same place (`synchronize` / `download` after the step's command buffers); this does not exercise a GPU that is actually broken.

### Feature off

Everything is `#[cfg(feature = "fault-inject")]`: no field, call or branch exists in a normal build, so the engine step path is unchanged and `tools/cbidentity.py` was not rerun. The release binary built without the feature (`cargo build --release -p intarsia-cli --features metal`, `58c52a9`) has the same size as the one built from d793300: **18,200,192 bytes both**. It is not byte-identical: 210 bytes differ, in the Mach-O `LC_UUID`, a metallib identifier string, `core::panic::Location` line numbers (the `scheduler.rs` and `lib.rs` lines moved), and the ad-hoc code signature. `strings` finds none of `INTARSIA_FAULT_INJECT`, `fault-inject`, `injected fault` in it (4 in the feature build, 18,199,968 bytes). Scratch binaries: `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/faultinj/intarsia-{fi,off,base}`.

## Tests

* Unit, no GPU: `fault.rs` (spec parsing, `after`, `mixed` with `waiting` and `pos`, one-shot arm/take); `crates/intarsia-core/tests/fault_inject.rs` (mock scheduler: `err` fails all 6 requests of a 3 decoding + 1 prefilling + 2 queued load, refuses later submits, counters; `after=N` fires on step N+1 only; unset plan inert; `panic` via `INTARSIA_FAULT_INJECT` through `BatchEngine` and its `catch_unwind`; bad spec refused). Run with `cargo test -p intarsia-core --features fault-inject`.
* Metal, GPU: `crates/intarsia-backend-metal/tests/fault_inject.rs` (armed `err` fails the next `synchronize` and a `download`, once; armed `panic` unwinds out of `synchronize` and the backend is usable afterwards).
* End to end: `tools/cbfault_e2e.py` through `tools/cbfault_slot.sh`.

## End-to-end scenario

Server: `intarsia serve --model ~/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 --max-batch 8 --no-ui` (plus `--no-supervisor` in the `nosup` cases), `INTARSIA_FAULT_INJECT=<mode>:mixed=7,waiting=2,pos=512`. First one warm-up request (a 1-row decode must not trip the plan: it did not). Then:

1. 7 decoding streams (4 streamed, 3 plain; `ignore_eos`, 3000 max tokens), waited until `/health` shows `running 7` and the streamed ones have 3 tokens.
2. The 4k prompt (streamed, 4096 tokens by calibration), waited until `/health` shows `running 8` (it holds the 8th slot).
3. Three more requests (2 streamed, 1 plain) that queue behind the full house.

The fault fires about 0.2 s after the 4k prompt is sent, on the first decode step with 7 rows, 3 queued and the 4k prompt at position 512. Every client has a 40 s read timeout and a 120 s deadline.

### What the 11 clients saw (identical in all four cases)

| Client | HTTP | Ended with | Tokens before the error |
|---|---|---|---|
| 4 streamed decoders | 200 | `error` SSE event, then `[DONE]` | 6 to 13 each |
| 3 plain decoders | 500 | JSON error body | none |
| 4k prompt (streamed, mid-prefill) | 200 | `error` SSE event, then `[DONE]` | 0 |
| 2 streamed queued | 200 | `error` SSE event, then `[DONE]` | 0 |
| 1 plain queued | 500 | JSON error body | none |
| 3 late probes after the failure (plain, streamed, chat) | 503 | JSON error body | none |

All 11 clients had ended 0.7 to 0.9 s after the first decoder was sent. The body, same shape in stream events, 500s and 503s: `{"error": {"message": "engine failed: backend error: metal: command buffer failed: injected fault (fault-inject)", "type": "server_error", "param": null, "code": 500}}` (code 503 for the refusals). In `panic` mode the message is `engine failed: scheduler panicked: injected fault (fault-inject) in Metal synchronize`, and the default panic hook also prints `thread 'intarsia-batch' panicked at crates/intarsia-backend-metal/src/lib.rs` to the server log.

### Counters against the clients (after the failure, deltas from after the warm-up)

| Counter | Value | Clients |
|---|---|---|
| `intarsia_batch_requests_failed_total` | +11 | 11 got the error (8 admitted + 3 queued) |
| `intarsia_batch_requests_admitted_total` | +8 | 7 decoders + the 4k prompt |
| `intarsia_batch_requests_finished_total` | +0 | none finished |
| `intarsia_requests_total` / `requests_finished_total{reason="error"}` | +8 / +8 | telemetry counts started (admitted) requests; the 3 queued ones never start |
| `intarsia_batch_running`, `intarsia_batch_queue_depth`, `intarsia_requests_running`, `intarsia_requests_waiting` | 0 | no leaked slot or request |
| `intarsia_engine_failed` | 1 | `/health` 503 `failed` |

Same in `err-sup` and `panic-sup` (read before the child exited).

### Process behaviour

* `--no-supervisor` (`exit_on_stall` is `args.child`, so the watchdog does not exit): the process stays up, `/health` 503, submits refused with 503, still the same 8 s later. This is the code's behaviour and consistent with the design (restart is the supervisor's job).
* Default supervisor: the watchdog logs `[watchdog] engine failed (...); exiting for restart`, the child exits 87 (`EXIT_FAILED`), the supervisor logs `server died after 56 s: exit 87 (engine worker failed); restart #1 in 1.0s`, and the new child (new pid) was healthy again after 53 s (model load) and served a plain and a streamed request (24 tokens, `finish_reason length`). The restarted child inherits `INTARSIA_FAULT_INJECT`, so it is armed again; the post-restart requests are single-row, which `mixed=7` ignores.
* The fresh restart without the fault (`nosup` cases) also served both request types.
* The driver kills only its own process group and checked afterwards that no process and no listener on the port remained; the slot wrapper checked again.

## Observations (not bugs; nothing changed)

* `intarsia_tokens_generated_total` is only incremented when a request completes normally: the 6 to 13 tokens each streamed decoder had already sent before the failure are in no token counter (still 9 after the failure, from the warm-up). The failed-request counters are right; a token counter that excludes failed requests is a choice, noted here for whoever reads the dashboard. **Fixed afterwards** on `metal-cb-tokcount`: a failed request now adds the tokens it generated before the failure, in both the batched and the single-sequence engine. Unit test: `failed_request_counts_its_generated_tokens`. Not re-measured on the GPU.
* Queued requests that die in `fail_all` are counted in `intarsia_batch_requests_failed_total` but not in `intarsia_requests_total` (telemetry starts at admission). Fine, as the batch counters are the outcome counters.

## Reproduce

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo build --release -p intarsia-cli --features metal,fault-inject
cargo test -p intarsia-backend-metal --release --no-run
cargo test -p intarsia-backend-metal --release --features fault-inject --test fault_inject --no-run
fleetq run --host endor --exclusive --mem 24 --label "cb fault-inject e2e" -- tools/cbfault_slot.sh docs/results/raw/<dir>
# env: BIN (default target/release/intarsia), MODES=err,panic, CASES=nosup,sup, SKIP_CARGO=1
```

fleetq jobs on endor: **#2287** (rerun of `tests/fault_inject.rs` in the Metal crate alone, 5 times, 2 passed each time, after the test-only lock fix `e9c969a`; this one test ran outside the main slot, the e2e results and the 110 existing Metal tests in #2276 stand as they are; log `cargo-metal-fault-rerun.log`), **#2276** (the full run: four cases, then both cargo test runs; all of the above), after two smoke runs, #2270 (`err`/`nosup`, ended on a harness bug in the cleanup, `os.killpg(pgid, 0)` raising `PermissionError` on macOS, fixed) and #2271 (`panic`/`sup` and `nosup`, 26/26 each). Raw: `docs/results/raw/2026-10-17-metal-cb-fault-inject/` (`slot.log`, `e2e/<mode>-<sup|nosup>/result.json` with every client's record and the metrics before and after, the server logs, `cargo-metal*.log`, `hygiene.txt`).

## Not done

* Not run: `tools/cbidentity.py` (hook is compiled out; see "Feature off"). `fault-inject` is not wired into the CPU backend: the CPU path is covered by the mock tests only.
* The injection is a synthetic error at the right place, not a GPU that really faults; it does not cover a hung command buffer (that is the watchdog's `stalled` path, untouched here).
* The `after=N` trigger ran on the mock only; on hardware only `mixed=R` was used. `after=N` counts every decode step, including 1-row ones, not only batched steps of 2 or more rows.
* Only the decode step is a trigger point. A prefill-chunk failure takes the same `fail_all` route (`is_fatal` for `Error::Backend`, covered by the mock), but was not injected on hardware.
* Not covered: the Anthropic-flavour endpoint (`/v1/messages`) error shape; only `/v1/completions` and `/v1/chat/completions` were driven.
