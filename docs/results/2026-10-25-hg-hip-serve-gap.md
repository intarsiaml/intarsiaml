# hg: the served-vs-in-process decode gap on HIP (2026-10-25)

Lane hg, Qwen3.6-35B-A3B NVFP4 on one R9700 (tatooine, `serve --backend hip --max-batch 16`, base `origin/main` 73ac1bbf for the measurements, 2c2d9941 for the gates). Follows [hr3](2026-10-24-hr3-hip-batching-r3.md) (PR #127). Branch `lane-hg/hip-serve-gap`, PR #133.

Machine-readable trace: [trace.json](2026-10-25-hg-hip-serve-gap/trace.json). Raw logs and scripts: `2026-10-25-hg-hip-serve-gap/raw/`, `scripts/hg/`.

## 1. Result

**There is no host-overhead gap to close.** hr3 left 212.6 / 257.1 tok/s served at 4 / 16 streams against 294 / 385 in the in-process step test and put the difference down to "5-20 ms per step outside the model step". Measured with a per-phase host timeline of the serving scheduler, that is wrong:

| 16 streams decoding, 35B | served | in-process step test |
|---|---|---|
| decode-only iteration | **42.2 ms** (256-iteration window) | 41.6 ms |
| steady aggregate decode | **375.6 tok/s** (15 streams: 378-389) | 385 tok/s |
| GPU idle between two decode steps | **0.025-0.030 ms** | |
| host work outside the model step (prepare, accept, detokenize, SSE send, metrics) | **about 0.07 ms** per step (0.2%) | |

The remaining 2.4% between 375.6 and 385 is within the 15-stream run-to-run spread (378.1 / 382.3 / 387.5 / 389.4 on four server starts) and 16 against 15 rows.

The gap is the metric. `localeval speed --concurrency` divides all generated tokens by the wall time of the whole burst (`speed.py:133`), and that wall contains the prefill of every prompt. With the hr3 doc's own numbers (991-token prompts at about 6.1k tok/s prefill, 128 decode steps of 41.6 ms at 16 rows and 13.6 ms at 4 rows):

| streams | prefill (N x 991 / 6.1k tok/s) | decode (128 x step) | predicted wall | predicted tok/s | served (hr3) |
|---|---|---|---|---|---|
| 4 | 0.65 s | 1.74 s | 2.39 s | 214.3 | 212.6 |
| 16 | 2.59 s | 5.33 s | 7.92 s | 258.6 | 257.1 |

Both within 1%. The 16-stream burst spends 2.6 of 7.9 s in prefill, one prompt at a time.

The radiance lead (native 1.11.0, device-side speculative acceptance, cb5a69a) was not inspected at the commit; it targets speculative acceptance, and intarsiaml's greedy batched path has no per-step host sampling to move (argmax runs on the device, 64 bytes come back) and only 0.07 ms of host work per step. What exists on HIP: nothing of either. `INTARSIA_DEVICE_SAMPLING=1` (`model/batch.rs:159`) needs `Backend::supports_topk`, which only the CUDA backend overrides (`intarsia-backend-cuda/src/device.rs:295`; default false at `backend.rs:319`), so it is a no-op on HIP; `INTARSIA_OVERLAP` (`engine/batch.rs:59-61`) is documented CUDA only. Neither affects this greedy benchmark (argmax runs on the device) and neither was changed.

## 2. Host timeline (`INTARSIA_HOST_PROFILE=1`)

New, off by default, measures only. One `host-profile` line per 256 iterations of each class (decode-only, decode+prefill, prefill-only) on stderr: mean ms per iteration per phase. Decode-only iterations with 14.6-16 rows, 35B, graph replay, greedy (`raw/dev-prof.log`, `raw/host-profile-table.txt`):

| rows | wall ms | reap+admit | prepare | step-setup | upload | launch (graph replay call) | sync (wait for GPU) | step-tail | accept | metrics | on_token | detok | send | finish | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16.0 | 42.163 | 0.000 | 0.000 | 0.001 | 0.011 | 3.528 | 38.595 | 0.000 | 0.000 | 0.001 | 0.001 | 0.009 | 0.016 | 0.000 | 0.030 |
| 15.6 | 41.974 | 0.000 | 0.000 | 0.025 | 0.008 | 3.479 | 38.438 | 0.000 | 0.000 | 0.000 | 0.001 | 0.007 | 0.014 | 0.000 | 0.025 |
| 15.1 | 39.529 | 0.000 | 0.000 | 0.001 | 0.008 | 3.373 | 36.125 | 0.000 | 0.000 | 0.000 | 0.001 | 0.008 | 0.010 | 0.000 | 0.023 |
| 3.95 | 13.950 | 0.000 | 0.000 | 0.048 | 0.005 | 1.505 | 12.385 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.003 | 0.000 | 0.007 |

The `launch` column is host time inside the graph-replay call (the GPU starts on the first kernels while the host is still queuing the rest); `launch + sync` is the model step. `gap` is the time from the sync of one decode step returning to the next step's launch beginning: the span the GPU has nothing queued. Everything the brief listed as candidates (scheduler and admission, tokenizer and detokenize, sampling, copies, locks, SSE writes) is under 0.05 ms in total.

Prefill iterations: a 1k prompt prefills as one 125-145 ms chunk (7.3k tok/s) with the solo chunk of 4096; 16 prompts in a burst are 16 such iterations, one per scheduler iteration, interleaved with one decode step of the already-running rows ("decode+prefill" 151-169 ms: about 23 ms decode plus 114-142 ms prefill).

## 3. Stall when a prompt arrives during decode

`scripts/hg/stall.py`: 15 streams (1k prompt, 512 generated) decode; once all have their first token, a 1k and then a 4k prompt (8 generated tokens) arrive. The 15 streams' per-token latency from the arrival to the new request's first token, 35B, one R9700:

| `--busy-chunk` | steady ITL p50 / p99 (15 streams) | 1k arrival: TTFT, ITL mean / p99 / max of the 15 | 4k arrival: TTFT, ITL mean / p99 / max |
|---|---|---|---|
| 4096 (HIP default) | 38.6 / 54.5 ms | 0.22 s, 70.9 / 168 / 168 ms | 0.59 s, 130 / 430 / 430 ms |
| 1024 | 38.2 / 54.8 ms | 0.30 s, 76.1 / 170 / 170 ms | 0.79 s, 107 / 162 / 162 ms |
| 256 | 38.1 / 54.0 ms | 0.53 s, 83.4 / 107 / 107 ms | 1.87 s, 97.0 / 123 / 124 ms |

Steady aggregate with 15 streams: 378.1 / 382.3 / 387.5 tok/s (default / 1024 / 256), the same within the run-to-run spread.

The default stays 4096 on HIP: `load.rs:447-457` makes the busy chunk the solo chunk so that prefill beside decoding rows is bitwise the solo prefill (a smaller chunk changes the GEMM tiling; 1024 against 2048 measured KL 7.87e-3, 2.5x the ref-noise floor, on CUDA, [rb2](2026-10-23-rb2-cuda-batching.md) section 11). The table is the price of that: a 4k prompt arriving stalls every decoding stream for one ~430 ms chunk; `--busy-chunk 256` trades it for a 2.4-3.2x longer TTFT of the arriving prompt and is not bitwise (run `accuracy` before using it). The default stays in this PR; whether to change it is a user decision (open).

Not measured here: 16 simultaneously decoding streams plus an arrival (the 16th slot is the arrival); `stall-dev-16.json` in `raw/` is a first run whose arrivals queued behind 16 busy slots, only its steady figures (375.6 tok/s, ITL p50 40.9 ms) are used.

## 4. What the PR contains

* `INTARSIA_HOST_PROFILE=1` (`crates/intarsia-core/src/hostprof.rs`, spans in `scheduler.rs`, `engine.rs`, `model/batch.rs`, `model/graph_step.rs`). Off: one relaxed atomic load per span.
* `Telemetry::record_ops`: a graph replay counted its ~1,000 ops with one lock each; now one lock per replay. Telemetry only.
* `docs/tuning.md` rows (knob, stall table), LOG row, this doc, `scripts/hg/` (stall.py, steady.sh, window and gate scripts).

No kernel, plan, schedule or output changes. Batching off, nothing in the engine path differs.

## 5. Gates

Status at this commit:

| Gate | Where | Result |
|---|---|---|
| Batching-off no-op, plan show (json + text) and 256 greedy ids, 8B and 35B, short and 4k, vs origin/main 2c2d9941 | CUDA, **spark1** (the brief named spark2; spark2's queue was 2h43m, spark1 was free; same hardware class) fleetq #4994 | all byte-identical (`raw/cuda-spark1-job.log`, `raw/cuda/`) |
| same | Metal, endor #4995 | all byte-identical (`raw/metal-job.log`, `raw/metal/`) |
| Metal `kernels test` | endor | 44 kernels, 817 cases ADMITTED |
| Workspace tests on the branch | endor (#5007) | everything passes except `intarsia-backend-cuda` `rows_registry` (2 tests), which fails identically on origin/main (checked in a separate worktree): pre-existing, not from this PR (`raw/metal-workspace-tests.log`) |
| `cargo test -p intarsia-core --test scheduler` | local | 26 passed |
| HIP no-op, `batch_recurrent --test-threads=1`, 35B `batch_cuda_real` (+ `--ignored`), HIP `kernels test --quick` | tatooine | **not run yet** (queued behind other lanes' windows); this PR's code path is the same with the knob unset, so these are confirmation gates |

The HIP rows are filled in before the PR leaves draft.

## 6. Reproduce

```
# window (tatooine, fleetq job with --takes-down radiance --quiet gpu,cpu): scripts/hg/win.sh up hg <label> INTARSIA_HOST_PROFILE=1
# steady state and stall: python3 scripts/hg/stall.py --url http://tatooine:8294/v1 --streams 15 --gen 512 --inject 1024,4096
# no-op gates: scripts/hg/{hip-gates,cuda-noop,metal-noop}.sh (noop.sh run|cmp)
```

fleetq jobs: tatooine #4983 (window A: profile, stall); gates HIP, CUDA spark1 #4994, Metal endor #4995 and #5007.
