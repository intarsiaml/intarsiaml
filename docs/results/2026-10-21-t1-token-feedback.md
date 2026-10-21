# T1: device-side token feedback for greedy decode

Lane T1 of the quota-burn plan (`docs/plans/2026-10-21-quota-burn-plan.md`), approved on 2026-10-21 at 12:25 EDT (item 2 of G3's decisions).

- **Base:** origin/main `c56c1292`, which already includes G3's PR #45 (`FIRST_COMMIT`). Re-checked after merging main `ba41a3ae` (J1 structured output, M6 TUI, 35B TP2 HIP): section 3b.
- **Branch:** `perf/t1-token-feedback`. The main gates ran on `b4a300b3` (squashed into `6712b10c`, same `crates/` tree); the post-merge re-check ran on `9ac9e2cb`.
- **Host:** endor (Apple M5 Max), Qwen3-8B BF16.
- **Raw data:** [raw/2026-10-21-t1-token-feedback/](raw/2026-10-21-t1-token-feedback/).

## Headline

- **The idle gap between decode steps is gone on Metal.**
  - 0.263 → 0.000 ms at 4k, and 0.254 → 0.000 ms at a short context. These are medians of 6 cb-traced runs per arm, against the new main with G3's first-commit lever.
  - The step period falls from 28.47 to 28.15 ms (−1.1 %) at 4k and from 27.41 to 27.20 ms (−0.8 %) short.
- **Decode tok/s** (interleaved A/B):
  - plain `generate` at 4k: +0.7 %, with feedback ahead in each of 6 adjacent pairs. Over the runs G3's state rule admits (≥1300 MHz, CPU ≤0.30) the gain is +0.85 %.
  - served `bench speed`, admitted runs only: +0.85 % at 4k and +1.3 % at 128.
- **Bitwise:**
  - greedy 256 ids are byte-identical to base, on the short prompt, the 4k prompt, and with profiled steps interleaved;
  - real-model stop conditions are byte-identical;
  - goldens are byte-identical;
  - `qualify` PASS (10 stages run, 2 skipped).
- **CUDA and HIP: no-ops.** Greedy ids are identical, and neither backend ever takes the new path.
- **On by default on Metal**, as the spec's rule requires (every gate bitwise, and faster). `INTARSIA_TOKEN_FEEDBACK=0` turns it off.
- **The cost of a late stop is bounded.** A stop is seen one step late. The step already submitted after the last token runs only its first 16 dispatches (about 1 ms); the rest is held and dropped.
  - The first version had no hold. Back-to-back short EOS-terminated chat requests then took +28 ms each (81 → 109 ms), because the discarded step landed in the next request's prefill.
  - With the hold they match base (76.3 vs 76.0 ms).
- **Shared footprint:** `engine.rs`, `model.rs`, `backend.rs` and the cpu backend. PR opened, not merged.

## 1. What changed

A greedy decode step is encoded and submitted before the host reads the previous step's token. The token stays on device and feeds the next step's embedding lookup.

- **Ring.**
  - Step N's argmax goes to one of two i32 slots in `ws.tok` (slots 1 and 2; slot 0 stays the plain path's).
  - Step N+1's `Embedding` reads that slot instead of `ws.ids`, and its argmax goes to the other slot.
  - With one step of lookahead, no slot is overwritten before the host has read it.
- **Fences.**
  - `Backend::fence` commits the open command buffer without waiting and returns a fence.
  - `Backend::read_after(fence, ..)` waits only for the buffers up to that fence, so step N+1 keeps running while the host reads step N's token.
  - Metal implements both. The defaults (`synchronize`, then `download`) keep every other backend on exactly the old behaviour.
  - On Metal, `upload` and `download` each drain the whole queue. The feedback step therefore uploads nothing: the ids come from the ring, and the position is the host-side `pos0` attr, as on the eager path.
- **Hold and discard.** The speculative step is not submitted whole.
  - `Backend::hold(after = 16)` commits its first 16 dispatches and keeps the rest encoded but uncommitted.
    - The 16 dispatches are about 1 ms of GPU work, queued behind step N. They keep the GPU busy through the host's wake and token handling (about 0.3 ms).
  - If the step is needed, `release` commits the rest. If a stop came, `discard` drops it unrun.
  - `INTARSIA_TOKEN_FEEDBACK_HOLD=<n>` sets the count; 0 submits the whole step.
  - A `synchronize` while a step is held (a profiled step, the one-off bandwidth probe) commits the held work, and that step runs whole. That is still correct: the rollback never depends on the discard.
- **Same kernels, same inputs.** Only the buffer the embedding reads its id from changes, plus where command buffers end. G3 showed that moving buffer boundaries is bitwise.
- **Rollback.**
  - A step whose input token has not been read yet is committed to the `SequenceState` with a placeholder (`UNREAD`) in `tokens`. The host replaces it once it reads the token.
  - A stop (EOS or stop id, stop string, cancel, a closed client) is seen one step late.
  - The speculative step is then cut off by `truncate` to `plen + generated - 1`. That is exactly what the plain loop holds (every emitted token but the last, fed), and it is the rule `finish` already applies after speculative decoding. The same cut runs after an error.
  - For attention-only state, truncation is bookkeeping. The KV row the dropped step may have written lies past `len`, and the next use overwrites it.
- **No step past `max_tokens`.** The next step is submitted only when the token being read is not the last one `max_tokens` (or the context window) allows. A forced-length request never runs a discarded step.
- **Where it applies:** `LocalEngine` (the single-sequence engine), greedy decode with the head on device (`head_mode == LastArgmax`), when all of these hold:
  - a request with `min_tokens` has passed the stop-id mask; it switches over at that point;
  - no structured-output constraint (J1). A lazy constraint (tool calls under `tool_choice: auto`) starts inactive, on the on-device argmax path, and activates on a trigger token. Inside the feedback loop the step after the trigger would already be speculated unmasked, so any request carrying a constraint keeps the old loop;
  - no speculative decoding (drafting needs the tokens on the host);
  - attention-only state (no recurrent slots);
  - one device (no tensor-parallel peers);
  - no captured graph replay;
  - not a dry run.
- **Old loop kept for:** stochastic sampling, logit tracing, structured output, `BatchedEngine` (`--max-batch > 1`), spec decode, hybrid recurrent models, TP, CUDA and HIP.
- **Switch:**
  - `Backend::token_feedback() -> (supported, default)`: Metal `(true, true)`; cpu `(true, false)`, for the engine tests; every other backend `(false, false)`;
  - `INTARSIA_TOKEN_FEEDBACK=0|1` overrides the default;
  - `Model::set_token_feedback` sets it from code.
- **G3's first-commit lever** (`FIRST_COMMIT = 4`, PR #45) fires on the first buffer after a `synchronize`.
  - Feedback removes that sync between decode steps, so the lever no longer fires in the pipelined loop. It still fires for the host-fed first step.
  - It is superseded, not lost. G3's lever shortened the gap; feedback removes it, because step N+1's start is already queued behind step N.
  - Measured against the new main, the base gap is 0.26 ms (G3 reported 0.295 ms after its lever), and with feedback it is 0.000 ms.

## 2. Speed (Metal, endor, Qwen3-8B BF16)

All runs compare new main `c56c1292` (base) with the final dev build, interleaved, in exclusive endor slot #3122. "fb" is feedback on, the new default.

**2a. Step anatomy.** cb traces of the last 128 of 129 decode steps, 6 runs per arm (`raw/.../gates/cbsum.txt`):

| context | arm | period ms | GPU busy ms | idle gap ms | buffers/step | tok/s from period |
|---|---|---:|---:|---:|---:|---:|
| 4k | base | 28.473 | 28.187 | 0.263 | 4 | 35.12 |
| 4k | fb | 28.153 | 28.152 | **0.000** | 2 | 35.52 (+1.1 %) |
| short | base | 27.413 | 27.155 | 0.254 | 4 | 36.48 |
| short | fb | 27.200 | 27.199 | **0.000** | 2 | 36.76 (+0.8 %) |

GPU busy time per step is unchanged within noise (28.19 vs 28.15 ms at 4k). The saving is the gap.

**2b. Plain `generate`.** 4k prompt, 256 tokens, macmon at 100 ms (`gates/ab-gen`, `gates/ab-gen2`, `gate.py`):

| run set | base tok/s | fb tok/s | delta | admitted runs (≥1300 MHz, CPU ≤0.30) |
|---|---:|---:|---:|---|
| ×3 (B F F B B F) | 35.37 | 35.32 | −0.1 % | base 3 runs, 35.37; fb 1 run, 35.70 |
| ×6 (B F F B B F F B B F F B) | 35.16 | 35.41 | **+0.7 %**; fb ahead in 6 of 6 adjacent pairs | base 4 runs, 35.19; fb 2 runs, 35.51 (+0.9 %) |
| all admitted runs, both sets | 35.27 (7 runs) | 35.57 (3 runs) | **+0.85 %** | |

Feedback runs sit 15–40 MHz lower, plausibly because the GPU no longer idles between steps. The ≥1300 MHz rule therefore rejects them more often (3 admitted against 7), which biases the admitted-only comparison against feedback. The 6/6 paired wins and the deterministic gap removal are the stronger evidence.

**2c. Served.** `bench speed --at-most`, gen 256, 3 reps, A/B ×3, plus 20 back-to-back short EOS-terminated chat requests per arm (`servesum.txt`):

| arm (final build) | decode 128 | decode 4k | TTFT 4k s | EOS request wall ms | GPU MHz | CPU | admitted |
|---|---:|---:|---:|---:|---:|---:|---|
| base-1 | 36.8 | 34.9 | 1.213 | 78.5 | 1335 | 0.06 | yes |
| fb-1 | 37.2 | 35.5 | 1.121 | 77.5 | 1364 | 0.08 | yes |
| fb-2 | 37.1 | 35.5 | 1.122 | 75.5 | 1357 | 0.05 | yes |
| base-2 | 36.8 | 35.4 | 1.102 | 75.3 | 1360 | 0.06 | yes |
| base-3 | 36.4 | 35.3 | 1.067 | 75.2 | 1332 | 0.07 | yes |
| fb-3 | 36.5 | 35.3 | 1.096 | 75.0 | 1294 | 0.07 | no |
| **mean, admitted runs** | 36.67 → **37.15 (+1.3 %)** | 35.20 → **35.50 (+0.85 %)** | | 76.3 → 76.5 (all runs: 76.3 → 76.0) | | | |

**The same A/B for the first version, without the hold** (`gates-nohold/`, same base):
- Decode was +1.7 % at 128 and +0.4 % at 4k. Only one run per arm was admitted.
- The short EOS-terminated requests took **81.2 → 109.2 ms** (+28 ms: one discarded step each).
- That penalty is what the hold removed; per-token decode did not change.
- The 4k TTFT in that round was bimodal on both arms (1.1 s, or 1.7–2.0 s), so it says nothing about feedback.

## 3. Gates

| gate | result |
|---|---|
| greedy 256 ids, Qwen3-8B BF16, short and 4k prompt: base (main `c56c1292`), dev with feedback off (`INTARSIA_TOKEN_FEEDBACK=0`), dev with feedback; also with `--profile-every 16`, where profiled steps interleave with feedback steps | **byte-identical** in every comparison, both rounds (`gates/`, `gates-nohold/`). After the default flip, the default build against base, short and 4k: byte-identical, idle gap 0.000 ms (`gates/default-on/`) |
| stop conditions on the real model (`--chat`): natural EOS; a stop string; `min_tokens` 12; a longer EOS-terminated answer; max_tokens 7. The answer ends on EOS at 6 tokens, so the last case is an EOS stop too. `max_tokens` is covered by the forced 256-token runs and the cpu tests | **byte-identical**, base vs feedback, 5 of 5 |
| goldens `qwen3-8b-fp32` and `qwen3-8b-long-fp32` (`accuracy --mode both`) | logs **byte-identical**, base vs feedback |
| `qualify`, feedback on | **PASS**: 10 stages run, 0 failed (kernels 28/374, gate, acc-short, bench, acc-long, acc-4k, envelope, bench-long). The ab and tp stages are skipped (no reference server, no TP). acc-4k: prefill top-1 99.28 %, KL 2.44e-4 (`gates/qualify/`) |
| Metal tests: `cargo test -p intarsia-backend-metal --release` with the real model, and `cargo test --workspace --features intarsia-cli/metal` | on the final head `0487bc30` with feedback **on by default** (#3160, `final-tests/`): **117 passed, 0 failed** and **550 passed, 0 failed**. The run on the hold commit before the flip, with feedback off by default: 117/0 and 549/0. They include G3's `fill.rs` / `qkv_fold.rs` dispatch-count tests and the new `tests/token_feedback.rs`, which checks: `read_after` waits for its own fence only; a dropped fence falls back to `download`; hold, release and discard; a sync while holding |
| CPU: `cargo check --workspace --tests` (cpu, and `--features intarsia-cli/metal`); `cargo test --workspace --exclude intarsia-backend-metal` | pass; 432 passed, 0 failed |
| EOS mid-batch: `BatchedEngine` over a model with feedback switched on. Three concurrent requests, one stopping on a stop id at its 4th token while the others continue, then one more request; compared with feedback off | identical (ids, `Done`, KV length). `BatchedEngine` never takes the feedback loop (`batched_engine_is_unchanged_with_feedback_on`, #3160) |
| stop-condition tests (cpu backend, `crates/intarsia-models/tests/token_feedback.rs`, 6 tests, the batched one, and a lazy-constraint one); every request runs on a feedback engine and a plain one, see below | 8/8 after the main merge (#3171); 7/7 on head `0487bc30` (#3160); 6/6 with and without the hold (tatooine container, #3083 and #3141). The first run (#3079, #3080) failed on two bugs in the tests themselves: the probe request ran on one engine only, and the held-back UTF-8 tail event was counted as a token |
| mutants on the rollback logic and the exclusions (`raw/.../mutants.py`, 8 mutants, listed below) | **8/8 killed** after the main merge (#3171, `mutants/mutants.txt`); the first 7 also before and after the hold (#3085, #3141). The discard itself is not mutated: it is an optimisation with no correctness role, covered by the Metal hold test and the EOS-latency A/B |
| CUDA no-op (spark1, non-exclusive, vLLM up): greedy 256 ids, Qwen3-8B, short and 4k prompt; base vs dev, and dev with `INTARSIA_TOKEN_FEEDBACK=1`, which must change nothing because CUDA does not support feedback | pre-rebase head (#3080, `cuda/pre-rebase/`): **byte-identical**, 4 of 4; 255 graph replays and 0 eager steps, so the feedback path is never taken. Final head `b4a300b3` vs main `c56c1292` (#3153): **byte-identical**, 4 of 4; 0 eager steps; dev CPU tests 217 passed, 0 failed (`cuda/`). After the main merge, head `9ac9e2cb` vs main `ba41a3ae` (#3172): **byte-identical**, 4 of 4; dev CPU tests 229 passed, 0 failed (`cuda/postmerge/`) |
| HIP no-op (tatooine, #3086, radiance down 50 s, coordinator-approved): greedy 256 ids, Qwen3-8B, GPU 0, short and 4k prompt, base vs dev | **byte-identical**, 2 of 2; 38.7 / 37.1 tok/s on both (`hip/`). This ran on the pre-rebase, pre-hold head, and today's single window is used. Since then the only HIP-reachable changes are the no-op trait defaults (`hold` / `release` / `discard`) and calls on the feedback path, which HIP never takes |

**What the stop-condition tests cover:**
- an EOS or stop id at each of tokens 0 to 5;
- a queue of requests, each stopped mid-run;
- the `min_tokens` window, then feedback;
- stop strings;
- cancel;
- `max_tokens` of 1, 2, 3, 4 and 17;
- the context cap.

For each request the two engines must agree on the ids, the text, the `Done` event, the KV length left behind, and two follow-up requests that reuse the KV. The feedback engine must run exactly one discarded step after a stop that comes later than the first token, and none after a length stop.

**The 8 mutants:**
1. no rollback;
2. rollback keeps the extra row;
3. rollback cuts one row too many;
4. no token patch;
5. speculate at `max_tokens`;
6. read the wrong ring slot;
7. feed the stale host token;
8. let constrained requests take the feedback loop (drop the J1 exclusion).

The lazy-constraint test picks a trigger token that first appears at generated token k >= 2, so the constraint activates inside the loop. It failed under mutant 8 and passes with the exclusion. A trigger at token 0 or 1 would be seen before the loop starts and prove nothing (the first version of the test did that, and mutant 8 survived it).

## 3b. Post-merge re-check (main `ba41a3ae`, head `9ac9e2cb`)

Main gained J1 (structured output, in the shared `engine.rs`), M6 (TUI) and 35B TP2 on HIP while this PR was open. After merging it and adding the constraint exclusion, gates 1–5 were rerun on endor (#3173, `gates-postmerge/`) with base rebuilt from the new main:

| gate | result |
|---|---|
| Metal tests: metal crate (real model on), workspace + metal | **117 / 0** and **631 / 0** failed |
| greedy 256 ids, short and 4k: base vs dev off, base vs dev feedback; short with `--profile-every 16` | **byte-identical**, 5 of 5 |
| real-model stop conditions (EOS, stop string, `min_tokens`, longer answer, max_tokens 7) | **byte-identical**, 5 of 5 |
| goldens, short and long | logs **byte-identical** |
| step anatomy (6 cb-traced runs per arm) | idle gap 0.264 → **0.000** ms at 4k, 0.256 → **0.000** ms short; period 28.05 → 27.75 ms (−1.1 %) at 4k, 27.08 → 26.84 ms (−0.9 %) short |
| CUDA no-op (#3172) | **byte-identical**, 4 of 4 |

`qualify` and the speed A/B were not rerun: the merge changes no Metal decode code, and the per-step result reproduces section 2a.

## 4. Effect on each backend

- **Metal: on by default.**
  - Every single-sequence greedy decode with the head on device takes the pipelined loop. That covers `serve` without `--max-batch`, `generate` and the speed benches.
  - Steps lose the 0.26 ms idle gap; numerics are unchanged.
  - Sampling, logit tracing, `--max-batch > 1`, `--spec` and hybrid models keep the old loop.
  - Cost: about 1 ms of GPU work after each stop (the held step's first 16 dispatches), and one more committed command buffer per step.
- **CUDA and HIP: no change, by construction and as measured.**
  - Both inherit `token_feedback() = (false, false)`, so `Model::feedback_ok` is false and the engine runs the old loop. They also replay captured decode graphs, which `feedback_ok` excludes in any case.
  - The shared-code changes they do run:
    - `run_layers` gained an `ids_at: None` argument;
    - `head_to_device` calls `head_to_device_into` with the same `ws.tok` view (the same ops, in the same order);
    - `Model` has one more `AtomicBool` and a `usize`;
    - `Backend` has new default methods they never call.
  - Greedy ids are identical to base on both (section 3).
- **Do CUDA and HIP have the gap?**
  - The graph step makes the same host round trip: `upload(ids)`, `upload(pos)`, `graph_replay`, then a blocking `download(tok)`.
  - So the device idles from the end of one replay until the next one launches: the host's wake, two small uploads and one graph launch.
  - This gap was not profiled here. The existing profiles measure inside a step (HIP: 3 us between kernels; CUDA: 99 us idle inside a 13.95 ms graph), not between steps.
  - Estimate: tens of us per step, under 0.5 % of an 8B step on either backend. Closing it would also need the position advanced on device inside the captured graph. Left alone without a measurement.
- **cpu:** supports feedback but is off by default. Launches are synchronous, so the default fence and read are exact. It exists only to test the engine logic.

## 5. Files

**Shared footprint** (no merge without the coordinator's no-op checks):
- `crates/intarsia-core/src/backend.rs`: `token_feedback`, `fence`, `read_after`, `hold`, `release`, `discard`, each with an old-path default.
- `crates/intarsia-core/src/model.rs`: the `ids_at` argument of `run_layers`, `head_to_device_into`, the switch and hold count.
- `crates/intarsia-core/src/model/feedback.rs` (new): `feedback_step`, `feedback_release`, `feedback_discard`, `feedback_read`, `feedback_ok`, the ring.
- `crates/intarsia-core/src/model/batch.rs`: the new argument (`None`).
- `crates/intarsia-core/src/engine.rs`: `Seq::feedback_ok`, `Seq::run_feedback` (the loop and the rollback), one branch in `run_seq`.
- `crates/intarsia-backend-cpu/src/lib.rs`: `token_feedback() = (true, false)`.
- `crates/intarsia-models/tests/token_feedback.rs` (new): 8 tests (6 on `LocalEngine`, 1 on `BatchedEngine`, 1 with a lazy structured-output constraint).

**Metal:**
- `crates/intarsia-backend-metal/src/lib.rs`:
  - `fence` / `read_after`, with a fence list per stream that `synchronize` clears;
  - `hold` / `release` / `discard`;
  - `token_feedback = (true, true)`.
- `crates/intarsia-backend-metal/tests/token_feedback.rs` (new): 2 tests.

**Scripts,** in `raw/2026-10-21-t1-token-feedback/`:
- endor: `build.sh`, `gates.sh`, `gate.py` (G3's state-admission rule), `cbgap.py` (G3's), `cbsum.py`, `servesum.py`, `eosreq.py`, `mutants.py`;
- tatooine and spark1: `hip-build.sh`, `hip-window.sh`, `cuda-run.sh`.

## 6. Downtime and slots

- **radiance (tatooine): one window, fleetq #3086.** Stopped at 13:00:51 EDT and answering again at 13:01:41 (50 s). Radiance had been serving for 13 min since #3066 (`raw/.../downtime.log`). No other role was taken down.
- **tatooine, GPUs hidden:** #3079, #3082, #3083, #3085 and #3141, all builds and CPU tests in the container.
- **spark1:** #3080, #3153 and #3172, non-exclusive, 24 GB, vLLM up.
- **endor, after the main merge:** #3169, #3170, #3171 (cpu tests and mutants, 8 GB); #3173 (exclusive, 24 GB, post-merge Metal re-check).
- **endor:**
  - #3090: builds and CPU checks, 8 GB;
  - #3091: exclusive; stopped by the deadlock below;
  - #3122: exclusive, 24 GB; every Metal gate and A/B.
- **Bug found in the first Metal run (#3091).** For a fence that a `synchronize` had dropped, `read_after` fell back to `download` while still holding the stream lock: a self-deadlock, hit by `--profile-every 16`. It was fixed before any result here was taken, and `tests/token_feedback.rs` now covers it.
- **#3122 and the G3 merge.** Partway through #3122 the coordinator reported G3's merge (PR #45). The slot's wrapper script was paused, the branch rebased onto `c56c1292`, and everything rebuilt and rerun inside the same job (`FLEETQ_JOB=3122`). The paused wrapper was then killed to release the slot. No fleet work ran outside a slot.

## 7. Decisions for the user

1. **Keep feedback on by default on Metal, or make it opt-in.**
   - It is on here, because the spec's rule says so: every gate is bitwise, and it is faster.
   - The evidence: the step gap is gone (−1.1 % per step at 4k), and admitted A/B runs are +0.85 % at 4k and +1.3 % at 128.
   - The A/B gain is small next to today's run-to-run noise, but feedback was ahead in all 6 adjacent plain-generate pairs.
   - Reverting is one line: `token_feedback() -> (true, false)` in the Metal backend.
2. **Merge.** The footprint is shared (`engine.rs`, `model.rs`, `backend.rs`, the cpu backend). The PR is open and not merged; the coordinator's pre-merge no-ops are the gate.
   - CUDA: rerun on the final head (section 3).
   - HIP: ran on the pre-hold head in today's single approved window. Since then: the hold, the merge of main (J1, M6, 35B TP2 HIP) and the constraint exclusion. HIP never takes the feedback path, but a rerun on the final head needs another radiance window.
3. **`BatchedEngine` (`--max-batch > 1`) is unchanged.** With a single active sequence it still pays the gap. Extending feedback to the batched step needs per-row stop handling: a separate lane, if wanted.
4. **CUDA and HIP feedback were not built.** The same gap exists between graph replays but was not measured. The estimate is tens of us per step, under 0.5 %, and closing it would also need the position advanced on device inside the captured graph. Measure first (nsys or rocprof between replays), then decide.
5. **G3's `FIRST_COMMIT` lever** no longer fires in the pipelined loop (section 1). Nothing to decide unless feedback is turned off; it then applies as before.
