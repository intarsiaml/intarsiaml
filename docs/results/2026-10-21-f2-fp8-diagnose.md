# F2: why `chat_reuse_4k` continue fails under the opt-in fp8 lt GEMM (INTERIM: the sweep has not run)

Lane F2, branch `lane-f2/fp8-diagnose`, 2026-10-21. Diagnosis only: no default changes, `INTARSIA_CUDA_FP8_GEMM_LT` stays opt-in.
Raw logs: `docs/results/2026-10-21-f2-fp8-diagnose/raw/`.

**Status.** The measuring job (fleetq #3295) sat in the spark1 queue for over 2.5 hours and had not started when this was
written: first behind #3271 (waiting on spark2's exclusive #3246), then behind #3276 (a3 GSM8K, 40 GB, running 56+ min) and #3285.
Everything below is from round 7's existing logs and a code audit. The sweep tool is built, tested and committed;
the numbers section is empty until #3295 runs.

## 1. What "3/32" is

Not 3 failing cases. `chat_reuse_4k` is ONE prompt; the `continue` row compares a warm stream (turn 1 + reply, then turn 2 reusing
that state) with a cold stream over 32 generated tokens. "agree 3/32" means the two greedy streams agree on the first 3 tokens
and split at token 3 (`raw/reusediag-lt.log`; with `--prefill-chunk 4096` the split is at token 5). The gate then judges the split
(`judge_streams`): it passes if the divergence is within a rounding tolerance (0.25 nats, KL 1e-2, calibrated on Qwen3-8B/HIP) or
the streams agree on at least 16 leading tokens. Here: warm took token 13 with a lead of 0.470 nats, cold took 321 with a lead of
0.067 nats, so the two runs' gap between the same two tokens differs by 0.54 nats and the row KL is 3.66e-2. Both exceed the tolerance, so the row FAILs.

## 2. Which hypotheses are already closed (code and existing logs)

- **Stale or per-call scale / amax: ruled out by construction.** `kernels/cuda/fp8_gemm_lt.cu` quantises with
  `x / input_scale` where `input_scale` is the checkpoint's static per-tensor scale, read from device memory (`sc[1]`), and passes
  `weight_scale`/`input_scale` to cuBLASLt as device scale pointers. There is no amax and nothing computed per call, so nothing can go stale
  across chunks or across the reuse boundary. This is vLLM's scheme. Per-row scaling is not available (the checkpoint has no per-row
  activation scales), so that test is not possible and was not built. `FAST_ACCUM` is off unless `INTARSIA_LT_FAST_ACCUM=1`.
- **Chunk- or call-dependent GEMM results: ruled out.** One cuBLASLt algorithm per (N, K, dtype) is pinned; the test
  `matmul_fp8_gemm_lt_rows_do_not_depend_on_the_call` passes bitwise for tail chunks and prefix-reuse row positions
  (round 7 §3.4).
- **Reuse-boundary state bug: ruled out.** `aligned` (reused length = multiple of the chunk, a hard bitwise logit check) and `restore`
  are 32/32 under lt.
- What is left in `continue`: the 15 reply tokens are decode-written (gemv, M = 1) in the warm run and prefill-written (lt GEMM) in the
  cold run. That is true for the default build too. Round 7's `--reuse-diag`: warm-vs-cold KL mean 4.96e-3 (lt) against 4.90e-3 (default),
  and default's max row KL (5.57e-2) exceeds lt's (4.22e-2). Chunk 2048 vs 1024 alone gives 4.85e-3 (lt) and 2.48e-3 (default).
  So the divergence is as large for default; default passed on this one prompt/split because no row tied closely enough. One sample per build.

## 3. The sweep (to run: `scripts/f2/run.sh`, summary: `scripts/f2/analyze.py`)

`accuracy --mode reuse --reuse-sweep N --reuse-sweep-from S --reuse-sweep-stride D` runs the gate's `continue` case at N different
splits of `chat_reuse_4k` through the engine's serving path, prints a `SPLIT` line per split (the gate's verdict and the `judge_divergence`
numbers) and a `ROW` line per row up to the first divergence (cold top-2 gap, warm gap for the same pair, row KL, argmax flip).
Arms: default and lt at chunk 2048 (42 splits from 2100, stride 59) and at chunk 4096 (41 splits from 4100, stride 13), same binary,
only the env differs. Question each arm answers: does default fail the same gate at other splits at a comparable count? If yes, the
failure is the generic warm/cold noise meeting near-ties, not an lt defect. If lt fails clearly more often with larger |gap shift|, there is an lt-specific
accumulation mismatch between the decode gemv and the prefill GEMM.

RESULTS: not yet measured.

## 4. Fix verdict

No clear bug found: nothing to fix. Two remedies exist and both are user decisions: (a) run the fp8 projections at decode M = 1 on the
same cuBLASLt accumulation (costs decode speed), or (b) set `reuse_logit_tol` / `reuse_kl_tol` for the 35B through the golden metadata or the recipe
(the built-in 0.25 nats / 1e-2 were calibrated on Qwen3-8B/HIP and fail default too once a split lands on a near-tie).

## 5. Checks run

- `cargo test -p intarsia-cli accuracy` (endor, fleetq #3296, separate target dir): 89 passed, 0 failed.
- `cargo check --workspace`-level build of `intarsia-cli` on endor: clean (this one ran outside a fleetq slot, before I noticed; it is a compile only).
- The sweep itself has NOT run on a GPU.
