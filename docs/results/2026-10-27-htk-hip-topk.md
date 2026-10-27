# htk: device-side top-k for sampling on HIP (2026-10-27)

Lane htk, origin project intarsiaml. Qwen3.6-35B-A3B NVFP4 on one R9700 (tatooine, gfx1201, `serve --backend hip --max-batch 16`, tp1), `origin/main` c6722f79a plus this branch. Raw output: `2026-10-27-htk-hip-topk/raw/`, scripts `scripts/htk/`, mutants `scripts/hip-decode/mutate.py` (`htk-*`).

## Result

HIP now has the CUDA device-side top-k of sampling (finding B2 of the 2026-10-23 Fable review, `docs/results/2026-10-23-rb-recurrent-batching/fable-review.md`). It is bitwise the host sampler's candidate list, and with `INTARSIA_DEVICE_SAMPLING=1` the served 35B decodes sampled streams 7% (4 streams) and 9% (16 streams) faster, single stream and greedy unchanged. The switch stays opt-in, default off (user decision 2026-10-23 17:57, `docs/decisions/LOG.md`); this change only makes the knob work on HIP.

| Gate | Result |
|---|---|
| Kernel = host order, bit for bit | 16392 random + 1032 planted + 2764 edge (id, value) pairs equal to a host sort; V 248320 / 151936 / 1000 / 70, R <= 16, k 1..64, k == V, ties, +-0, +-inf, subnormals; sampler draw on the candidates = draw on the whole row |
| Planted mutants (3 or more) | 9 of 9 fail `tests/topk.rs` (table below) |
| `every_hip_kernel_has_a_quick_case` | passes (no GPU; the kernel is not an `Op`, like CUDA's) |
| `every_hip_kernel_matches_the_reference` | passes, 6 tests of the admission binary in 66 s |
| CPU tests | `cargo test --locked --workspace` without metal / cuda / hip crates: exit 0 |
| 35B tp1, device draws = host draws | 3360 (row, step) tokens equal: B 1 / 4 / 16 x 5 settings (t 0.7 k 20 p 1.0; t 1.0 k 40 p 0.9; t 0.8 k 64 p 0.5; t 1.3 k 8 p 0.95; t 0.6 k 1 p 1.0) x 32 steps (`real_35b_device_sampling_draws_the_host_tokens`) |
| 35B served, token ids identical | 21 of 21 cells (7 settings x 1 / 4 / 16 streams, seeded, 64 tokens each, 9408 token ids) between `INTARSIA_DEVICE_SAMPLING=0` and `=1`, same binary |
| Greedy untouched | the greedy cells of the served comparison are identical; greedy `localeval speed` 162.6 / 162.5 tok/s off, 162.6 / 162.6 on |
| Served multi-stream sampled, 4 streams | 208.9 -> 222.7 tok/s aggregate (1.07x) |
| Served multi-stream sampled, 16 streams | 258.5 -> 281.9 tok/s aggregate (1.09x) |
| Single stream no worse | sampled 153.5 / 153.7 -> 153.6 / 153.7 tok/s (1.00x) |
| `plan show` G0 on the other backends | cpu and metal: `plan show` text and `--json` byte-identical to `origin/main` for Qwen3-8B and the 35B (8 files). CUDA not run: no CUDA or core source is in the diff |

## What was built

* `kernels/hip/topk.hip`: `ik_topk_f32`, the CUDA kernel (`kernels/cuda/topk.cu`) with the same ABI (`ik_hip_abi.h`), the same two stages (32 slices x R blocks of 256 threads, each selecting its k best by k rounds "best element strictly after the previous pick in the total order"; one block per row merges the 32 x k candidates), the same scratch (`R * 32 * k * 8` bytes), the same limits (1 <= k <= 64, k <= V, R <= 65535) and the same NaN rule (a NaN never wins, the row's flag is set and the host samples that row from the whole logits). The code differs from the CUDA file only in the wave32 shuffles (`__shfl_xor(v, o, IK_WARP)`), `hipMemsetAsync` and a `V <= INT_MAX` check. No tuning: the 256-thread blocks are the CUDA shape; the HIP time per step is below.
* `Backend::supports_topk` / `topk_rows` for HIP (`crates/intarsia-backend-hip/src/device.rs`, `ffi.rs`). The model side (`HeadMode::LastTopK`, `Model::batch_topk`, `Seq::topk_for`) was already backend-neutral, so nothing in core changed. Which requests take it (unchanged): non-greedy, no constraint, no `trace_logits`, no stop-id masking, `top_k` in 1..=64. A request with only `temperature` / `top_p` (no `top_k`) samples on the host as before.
* `crates/intarsia-backend-hip/tests/topk.rs` (5 tests) is the kernel's admission. It is not in the registry sweep because the kernel is not an `Op`, exactly as on CUDA (`crates/intarsia-backend-cuda/tests/row_tables.rs`); the registry tests are unaffected and pass.
* `real_35b_device_sampling_draws_the_host_tokens` in `crates/intarsia-models/tests/batch_cuda_real.rs` (runs on CUDA and, with `INTARSIA_BATCH_TEST_BACKEND=hip`, HIP; skips unless `INTARSIA_DEVICE_SAMPLING=1`).
* `docs/tuning.md` row of `INTARSIA_DEVICE_SAMPLING` said "on (set 0 to disable)", stale since the 17:57 decision; it now says off, CUDA and HIP.

## Mutants

Each mutant is one edit of `kernels/hip/topk.hip` (or its launcher), built into its own `tests/topk.rs` binary (`scripts/htk/mutbuild.sh`) and run on the GPU.

| Mutant | Edit | Caught by |
|---|---|---|
| `htk-tiebreak` | ties to the highest index | 5 of 5 tests |
| `htk-slice-end` | stage 1 drops the last element of each slice | 5 of 5 |
| `htk-stage2-n` | stage 2 merges all slices but the last | 4 of 5 |
| `htk-nanflag` | a NaN is not flagged | NaN test |
| `htk-nonstrict` | a pick may repeat an equal earlier element | 4 of 5 |
| `htk-exhausted` | an exhausted slice's empty entries read as index 0 | 2 of 5 |
| `htk-nomemset` | NaN flags not cleared before the launch | 5 of 5 |
| `htk-kv` | the launcher accepts k > V | rejection test |
| `htk-scratch` | the launcher accepts a scratch one byte short | rejection test |

The first run of `htk-kv` survived: the rejection test's only k > V case also had k > 64, which the other check rejects. The test now has a k = 11, V = 10 case; the table is the run after that fix.

## Speed

Served, one 35B server per arm (`scripts/htk/window2.sh`, localeval 0.4.0 `speed`, 1024-token prompts, 128 forced tokens, 2 reps at one stream, one burst each at 4 and 16 streams; target `htk-sampled` sends temperature 0.8 and top_k 40, the suite pins top_p 1 and seed 0). Same binary, `INTARSIA_DEVICE_SAMPLING` 0 then 1.

| | host sampling (0) | device top-k (1) | ratio |
|---|---|---|---|
| sampled, 1 stream (decode tok/s, 2 reps) | 153.5, 153.7 | 153.6, 153.7 | 1.00 |
| sampled, 4 streams (aggregate tok/s) | 208.9 | 222.7 | 1.07 |
| sampled, 4 streams (per-stream peak decode, median) | 67.7 | 73.4 | 1.08 |
| sampled, 16 streams (aggregate tok/s) | 258.5 | 281.9 | 1.09 |
| greedy, 1 stream (decode tok/s, 2 reps) | 162.6, 162.5 | 162.6, 162.6 | 1.00 |

In-process, one batched step with 4 / 16 sampled rows including the host draw (`real_35b_batch_step_sampled_timing`, `INTARSIA_DEVICE_SAMPLING=1`): B = 16 44.7 ms host, 39.0 ms device (-13%); B = 4 59.0 ms host, 12.9 ms device. The B = 4 host figure is the first cell of the test and carries the warm-up, so read only B = 16 from it. The review estimated -12 to -22% at 16 streams on the CUDA side; the in-process step here is -13%, and the served burst figure above also holds the prefill of the 16 prompts.

## Limits of this measurement

* One server visit per arm, in the order off, on. Decode noise is +-5% (localeval notes); the 4-stream and 16-stream gains are 7% and 9% from one burst each and are consistent with the in-process step (-13% at B = 16), but they are not an a/b/b/a series. The first of the two allowed GPU windows ran for 16 s: the sync script had replaced `.bin` and the clean binaries were missing (only the mutant runs, one of them a clean-kernel run in effect, went through). So no second visit was made.
* The 35B draws test and the served comparison both draw with the engine's own `Sampler`; "equal to host sampling" means equal to the host path of the same engine at the same seeds, the same oracle as CUDA's `real_35b_batched_engine_equals_the_local_engine`.
* `INTARSIA_DEVICE_SAMPLING` stays off by default. Turning it on for HIP is a user decision; the data above supports it for requests that carry `top_k` (the Qwen3 recommended settings do: top_k 20). Requests that send only `temperature` and `top_p` do not take the device path.
* HIP now answers `supports_topk()`, and the dcoco corrector (`model/dcoco.rs`) uses `topk_rows` whenever the backend has it, without the `INTARSIA_DEVICE_SAMPLING` switch. HIP cannot reach it: `--dcoco` needs a DFlash (v1) draft, which `crates/intarsia-models/src/load.rs:529` admits on cpu and cuda only, and a DFlash2 draft (cpu, hip) refuses `--dcoco` (`crates/intarsia-core/src/model/dflash.rs:269`).
* No CUDA run for G0; CUDA's source and the core crate are not in the diff.

## Reproduce

```
scripts/htk/sync.sh HEAD
fleetq run --host tatooine --load cpu --expect 45 --label "htk build" -- ssh tatooine 'bash -s' < scripts/htk/build.sh
fleetq run --host tatooine --load cpu --expect 14 --label "htk mutants build" -- ssh tatooine 'bash -s' < scripts/htk/mutbuild.sh
fleetq run --host tatooine --takes-down radiance --mem 30 --quiet gpu,cpu --expect 30 --label "htk window" -- bash scripts/htk/window.sh
```
