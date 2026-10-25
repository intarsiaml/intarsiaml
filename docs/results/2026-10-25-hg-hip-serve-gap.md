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

## 5. A/B against origin/main (same fleetq slot)

`scripts/hg/ab.sh`: server restarted per visit (`win.sh up`), arms `base` (origin/main), `pk` (the packed-prefill build of PR #137, knob unset: it contains this PR's code and nothing else active), `pkpack` (same build, `INTARSIA_PACK_PREFILL=1`). Per visit: 2 rounds of `localeval speed` (1k prompt, 128 generated, 1 / 4 / 16 streams; 4k prompt, 4 streams) and `scripts/hg/stall.py` steady-state decode with the client-side inter-token latency (ITL, SSE timestamps, per-stream, 2 s window after every stream has its first token). `localeval`'s concurrency rows do not report ITL, so ITL comes from `stall.py`. Tables: median over rounds x visits, (min-max, n).

The "1 stream" row is `localeval speed`'s single-stream decode figure (prefill excluded).

### 5.1 Window B2 (fleetq #5054, visits base pk pkpack pkpack pk base)

| load | origin/main | packed-prefill build, knob unset | INTARSIA_PACK_PREFILL=1 |
|---|---|---|---|
| 1 stream: tok/s | 162.8 (162.7-163.3, n=4) | 163.3 (159.7-165.9, n=4) | 162.5 (161.2-163.0, n=4) |
| 4 streams @1k: tok/s | 214.9 (200.9-216.4, n=4) | 210.8 (207.6-215.5, n=4) | 227.2 (227.1-229.3, n=4) |
| 16 streams @1k: tok/s | 259.7 (258.7-260.2, n=4) | 259.1 (256.2-260.6, n=4) | 288.2 (282.4-290.7, n=4) |
| 4 streams @4k: tok/s | 146.1 (145.8-146.3, n=4) | 142.8 (141.6-145.3, n=4) | 145.6 (145.4-145.8, n=4) |

| burst TTFT median s | origin/main | packed-prefill build, knob unset | INTARSIA_PACK_PREFILL=1 |
|---|---|---|---|
| 4 streams @1k | 0.37 (0.36-0.38, n=4) | 0.39 (0.37-0.41, n=4) | 0.46 (0.43-0.46, n=4) |
| 16 streams @1k | 1.35 (1.33-1.36, n=4) | 1.35 (1.34-1.39, n=4) | 0.97 (0.95-0.98, n=4) |
| 4 streams @4k | 1.02 (1.02-1.03, n=4) | 1.07 (1.02-1.08, n=4) | 1.03 (1.03-1.04, n=4) |

| steady decode (stall.py), n streams | origin/main | packed-prefill build, knob unset | INTARSIA_PACK_PREFILL=1 |
|---|---|---|---|
| 1 | 162.9 tok/s, ITL p50 6.1 / p99 12.4 ms | 163.8 tok/s, ITL p50 3.1 / p99 24.8 ms | 163.1 tok/s, ITL p50 6.1 / p99 11.4 ms |
| 4 | 285.4 tok/s, ITL p50 13.5 / p99 36.7 ms | 287.0 tok/s, ITL p50 13.5 / p99 25.5 ms | 283.2 tok/s, ITL p50 13.5 / p99 26.1 ms |
| 16 | 386.8 tok/s, ITL p50 40.9 / p99 57.7 ms | 383.6 tok/s, ITL p50 41.0 / p99 58.4 ms | 391.2 tok/s, ITL p50 39.8 / p99 56.6 ms |

### 5.2 Repeat (fleetq #5085, visits pk base pk base)

The first run read low for the knob-unset build in two places: the 1-stream burst row (127 against 136 tok/s in 3 of 4 pk rounds, 136 in the others) and 4 streams at 4k (142.8 against 146.1). The knob-on arm, which runs the same binary, did not show it, and nothing in the unset path differs from main's except the profile spans (one atomic load each). The repeat, four more visits in ABAB order, shows no difference:

| load | origin/main | packed-prefill build, knob unset | INTARSIA_PACK_PREFILL=1 |
|---|---|---|---|
| 1 stream: tok/s | 163.0 (162.8-163.1, n=4) | 163.1 (162.6-163.8, n=4) | - |
| 4 streams @1k: tok/s | 215.2 (212.4-217.8, n=4) | 215.0 (213.4-217.6, n=4) | - |
| 16 streams @1k: tok/s | 260.4 (259.7-260.9, n=4) | 260.1 (258.9-260.9, n=4) | - |
| 4 streams @4k: tok/s | 146.2 (145.7-146.4, n=4) | 146.1 (145.7-147.4, n=4) | - |

| burst TTFT median s | origin/main | packed-prefill build, knob unset | INTARSIA_PACK_PREFILL=1 |
|---|---|---|---|
| 4 streams @1k | 0.38 (0.37-0.39, n=4) | 0.38 (0.35-0.38, n=4) | - |
| 16 streams @1k | 1.33 (1.30-1.35, n=4) | 1.33 (1.32-1.36, n=4) | - |
| 4 streams @4k | 1.02 (1.02-1.03, n=4) | 1.02 (1.01-1.03, n=4) | - |

| steady decode (stall.py), n streams | origin/main | packed-prefill build, knob unset | INTARSIA_PACK_PREFILL=1 |
|---|---|---|---|
| 1 | 162.9 tok/s, ITL p50 6.1 / p99 11.7 ms | 163.2 tok/s, ITL p50 6.1 / p99 16.7 ms | - |
| 4 | 286.3 tok/s, ITL p50 13.4 / p99 30.1 ms | 284.2 tok/s, ITL p50 13.5 / p99 25.6 ms | - |
| 16 | 382.7 tok/s, ITL p50 41.2 / p99 57.2 ms | 382.1 tok/s, ITL p50 40.8 / p99 57.4 ms | - |

Read: knob-unset equals origin/main within 0.1% on every row of the repeat (single stream decode 163.1 vs 163.0, 16 streams 260.1 vs 260.4). The low readings of the first run were not the build: the 1-stream burst row was bimodal inside one server start (136 then 127 in visit 2), the repeat shows both builds at the same level, and the cause (GPU clock state or similar) was not isolated. Single-stream is within the 1% allowance in both runs (+0.3% and +0.1%).

## 6. Packed prefill (PR #137, separate, opt-in)

The burst's serial prefill is the measured gap, so lane 4 of the plan is built on top of this PR: `INTARSIA_PACK_PREFILL=1` runs the next chunks of several waiting prompts as one pass (row-wise kernels once over all rows; attention and the GatedDeltaNet recurrence per prompt). From 5.1, knob on against origin/main: 16 streams 259.7 -> 288.2 tok/s (+11%, TTFT median 1.35 -> 0.97 s), 4 streams 214.9 -> 227.2 (+5.7%), 1 stream and 4 streams at 4k unchanged (a 4k prompt is a single chunk). It is not part of this PR's code. Gates run for it (HIP window #5054, CUDA spark1 #5090, Metal endor #5089): HIP knob-unset no-op identical to origin/main (8 comparisons); `batch_recurrent` 11 passed on hip and cuda, including the packed-vs-solo test; `packed_rows` 5 passed; `real_35b_packed_prefill_equals_the_local_engine` passed (24 requests, 6 sampled, 8 and 16 slots, every chunk packed); CUDA and Metal no-op identical. Details in PR #137.

## 7. Gates

Status at this commit:

| Gate | Where | Result |
|---|---|---|
| Batching-off no-op, plan show (json + text) and 256 greedy ids, 8B and 35B, short and 4k, vs origin/main 2c2d9941 | CUDA, **spark1** (the brief named spark2; spark2's queue was 2h43m, spark1 was free; same hardware class) fleetq #4994 | all byte-identical (`raw/cuda-spark1-job.log`, `raw/cuda/`) |
| same | Metal, endor #4995 | all byte-identical (`raw/metal-job.log`, `raw/metal/`) |
| Metal `kernels test` | endor | 44 kernels, 817 cases ADMITTED |
| Workspace tests on the branch | endor (#5007) | everything passes except `intarsia-backend-cuda` `rows_registry` (2 tests), which fails identically on origin/main (checked in a separate worktree): pre-existing, not from this PR (`raw/metal-workspace-tests.log`) |
| `cargo test -p intarsia-core --test scheduler` | local | 26 passed |
| HIP batching-off no-op vs origin/main: plan show (json, text) and 256 greedy ids, 8B and 35B, short and 4k | tatooine, fleetq #5017, one R9700 per tree | all 8 comparisons identical (ids sha 5d882689 / 6203ecdb 8B, 0d079dea / d626e2f1 35B; plan json 8343 / 12961 B, text 1346 / 3184 B) (`raw/hip/gates-run.log`) |
| HIP `batch_recurrent --test-threads=1` | #5017 | 10 passed, 0 failed |
| HIP 35B `batch_cuda_real` (hip backend) | #5017 | 2 passed (35B batched decode bitwise solo, 35B engine equals LocalEngine) |
| same, `--ignored real_35b_batched_engine_long_prompts_equal_the_local_engine` | #5017 | passed (56.8 s) |
| `batch_cuda_real real_qwen3_8b_batched_decode_is_bitwise_the_solo_step` (dense 8B) | #5017, rerun with `INTARSIA_QWEN3_DIR` | **not applicable on HIP**: `Model::allow_recurrent_batch` refuses a dense model on hip (`hip_batch_needs_hybrid`, `model/batch.rs:781`, unchanged here); the refusing code is not touched by this PR, so the test cannot pass on origin/main either (not run there). The first run failed earlier on the unset env var (`raw/hip/g-8b.log`) |
| HIP `kernels test --quick` | #5017 | 62 kernels, 333 cases, all ADMITTED |
| HIP `packed_rows` (row-invariance of the 35B prefill kernels; for the packed-prefill PR) | #5017 | 5 passed: `matmul_nvfp4_mma`, `matmul_bf16`, `moe_topk_bf16_v2`, `moe_experts_bf16_v2` give each row the same bits packed (6 segments, 4,167 rows) as alone; a 15-row segment differs, 16 and up do not |

radiance was restored and answered (`active`, 30 s) before the slot was released.

## 8. Reproduce

```
# window (tatooine, fleetq job with --takes-down radiance --quiet gpu,cpu): scripts/hg/win.sh up hg <label> INTARSIA_HOST_PROFILE=1
# steady state and stall: python3 scripts/hg/stall.py --url http://tatooine:8294/v1 --streams 15 --gen 512 --inject 1024,4096
# no-op gates: scripts/hg/{hip-gates,cuda-noop,metal-noop}.sh (noop.sh run|cmp)
```

fleetq jobs: tatooine #4983 (window A: profile, stall); gates HIP, CUDA spark1 #4994, Metal endor #4995 and #5007.
