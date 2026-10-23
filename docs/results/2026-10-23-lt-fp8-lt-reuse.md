# Lane lt: why `INTARSIA_CUDA_FP8_GEMM_LT=1` fails prefix-reuse `continue` rows and `acc-short` (2026-10-23)

Target: DGX Spark GB10 (sm_121, CUDA 13.0), Qwen3.6-35B-A3B-NVFP4, tp=1. Build: `main` 240e9d2a (`lt-base`, spark1 and spark2).
Follows PR #122 (`docs/results/2026-10-23-pf-cuda-fp8-prefill.md`), which left the mechanism unknown. Scripts: `scripts/lt/`.
Raw logs: `2026-10-23-lt-fp8-lt-reuse/raw/`. **Measurement only. No engine code changed, and no threshold or baseline changed.**

## Verdict

**The gates fail any change of e4m3 accumulation, lt included.** The lt GEMM has no M, chunk or reuse dependence. Another
accumulation order that is not lt (f8) trips both gates in the same way. lt (like f8) does move warm-vs-cold drift above the
default's (point 3), so (d) holds only in its gate form. Whether that extra drift moves away from the truth can be measured
only against an oracle, and on the one oracle-scored prompt it does not. Evidence:

1. **The same `continue` row fails without lt.** `INTARSIA_CUDA_FP8_GEMM_F8=1` is the repo's own e4m3 mma.sync GEMM: a third
   accumulation order, not cuBLASLt. It fails `chat_reuse_4k` `continue` at chunk 1024 (5/32, cold margin 0.073) by the same
   mechanism as lt (raw/r1/reuse-f81k.log). The default itself goes past the 0.25-nat tolerance on that row at chunk 3072.
   It diverges at token 22, and the shared rows before that differ by 1.09 nats. Only the 16-token grace rule saves it
   (raw/r1/reuse-def3072.log).
2. **The default's warm and cold runs already differ by more than the tolerance on every prompt.** The tolerance is consulted
   only when the greedy streams diverge. On the shared rows, the default's warm-vs-cold top-5 log-probability difference
   (`nu_max`, REUSE-MEASURE) is 1.08 to 6.49 nats on all 8 non-oracle prompt x chunk cells, against the fixed 0.25. lt
   ranges from 0.64 to 7.43 and f8 from 0.96 to 3.46: the same spread. Which arm fails is decided by whether a near-tie lands
   in the 16-token window. Failing cold margins: 0.067 and 0.281 (lt), 0.008 (lt, `long_code_4k`), 0.073 (f8).
3. **lt's teacher-forced warm-vs-cold KL cannot be told from f8's, and both sit above the default's** (same context, 32
   rows, table below). lt is above the default in 9 of 10 prompt x chunk cells (largest gaps: `long_docs_12k` at 1024,
   4.1e-2 against 3.7e-3; `long_code_4k` at 2048, 1.2e-2 against 4.0e-3). f8 is above the default in 6 of 10, and lt is
   above f8 in 5 of 10. The decode-written-KV component is the same: lt is above the default in 8 of 10 cells. One reason the
   default sits lowest may be that its prefill GEMM and decode GEMV share an accumulation chain. This is a hypothesis and was
   not tested. On `long_docs_4k`, the only prompt with a committed f64 CPU oracle, lt's warm run is no further from the oracle
   than its cold run (0.79x to 0.92x).
4. **(a) is ruled out at GEMM level and (b) within a run.** One pinned algorithm per (N, K) gives bit-identical rows for M = 4096
   against slices of 5, 15, 111, 425, 551 and 2048 rows, on all four 35B shapes. The new test is
   `matmul_fp8_gemm_lt_rows_do_not_depend_on_the_call_35b_shapes`. The shapes are those the `INTARSIA_LT_DEBUG` plan lines
   show: N x K = 8192x2048 (algo 67 tile 467), 4096x2048 (algo 35 tile 20), 2048x4096 (algo 67 tile 465) and 512x2048 (algo
   67 tile 32). Warm runs repeated within one process are bitwise identical, and so are cold runs (raw/d1, raw/s1, raw/s2).
5. **(c) is ruled out.** `restore` and `aligned` are bit-identical with lt in every run, and warm and cold reruns are bitwise
   identical, so neither prefix restore nor offsets interact with lt's workspace or scales.
6. **`acc-short`'s baseline gate fails for any accumulation change.** The baseline is main's own run: mean KL 5.870e-3, limit
   x1.15 = 6.750e-3. lt gives 6.825e-3, and `INTARSIA_LT_FAST_ACCUM=1` changes nothing on sm_121 (every reported KL identical to lt's). f8
   gives **7.806e-3**: it fails by more than lt. The move (+0.96e-3) is below vLLM's own run-to-run KL on this golden
   (2.5e-3 to 3.2e-3, `golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.json`). lt's teacher-forced decode KL
   halves (1.54e-2 to 6.99e-3), and so does f8's (7.14e-3). The absolute floor passes.

Changing lt's algorithm (another heuristic index, a tile, FAST_ACCUM) only to make these rows pass would move the near-ties to
other prompts. That would overfit the gate, so it was not done.

## Gate results with lt on (no change to code, thresholds or baselines)

| gate | default | lt | f8 (control) | source |
|---|---|---|---|---|
| prefix reuse, chunk 1024 | PASS 15/15 | FAIL (`chat_reuse_4k` continue 3/32) | **FAIL** (`chat_reuse_4k` continue 5/32) | #122 a3; raw/r1, raw/s2 |
| prefix reuse, chunk 1536 | PASS | PASS | not run | raw/r1 |
| prefix reuse, chunk 2048 | PASS | FAIL (`chat_reuse_4k` 3/32, `long_code_4k` 8/32) | PASS | #122 a3; raw/s2; raw/r1 |
| prefix reuse, chunk 3072 | PASS (beyond tolerance at token 22, grace) | PASS (beyond tolerance at token 22, grace) | not run | raw/r1 |
| prefix reuse, chunk 4096 | PASS 9/9 | FAIL (`chat_reuse_4k` 5/32) | not run | #122 a1 |
| prefix reuse, chunk 8192 | not run | PASS 6/6 (4k prompts skipped) | not run | #122 a1 |
| `acc-short` baseline gate (mean KL, limit 6.750e-3) | PASS 5.870e-3 | FAIL 6.825e-3 | FAIL 7.806e-3 | raw/as2 |
| `acc-short` absolute floor | PASS | PASS | PASS | raw/as2 |
| KL goldens short / long / 4k (floor 3e-2) | 4.67e-3 / 1.49e-2 / 1.22e-2 | 5.41e-3 / 1.42e-2 / 1.21e-2 | not run | #122 |
| lt GEMM tests (accuracy, row invariance x2) | n/a | 3/3 pass | n/a | raw/gputests-kernels-quick.txt |
| `kernels test --quick` | 0 FAIL (77 admitted) | 0 FAIL (76 admitted; `matmul_fp8_pair_gemm_reg` off with lt, 0 cases) | not run | raw/gputests-kernels-quick.txt |

## Teacher-forced warm vs cold, same context (`--reuse-diag --reuse-measure`)

Full table: `raw/diag-table.md`. Columns: shared-row `nu_max` (nats; the tolerance is 0.25), KL mean / max over 32 rows,
argmax flips.

| prompt | chunk | default | lt | f8 |
|---|---|---|---|---|
| chat_reuse_4k | 1024 | 1.18; 2.2e-3 / 2.5e-2; 0 | 0.90 (diverged); 2.6e-3 / 2.3e-2; 0 | 0.96 (diverged); 3.2e-3 / 1.7e-2; 0 |
| chat_reuse_4k | 2048 | 1.08; 4.9e-3 / 5.6e-2; 2 | 0.64 (diverged); 5.0e-3 / 4.2e-2; 0 | 1.17; 3.7e-3 / 3.4e-2; 0 |
| long_code_4k | 1024 | 2.42; 3.2e-3 / 2.8e-2; 0 | 2.58; 3.9e-3 / 5.2e-2; 0 | 2.29; 4.6e-3 / 5.7e-2; 0 |
| long_code_4k | 2048 | 4.06; 4.0e-3 / 5.1e-2; 0 | 1.72 (diverged); 1.2e-2 / 2.4e-1; 3 | 1.82; 2.7e-3 / 3.3e-2; 0 |
| long_docs_12k | 1024 | 1.63; 3.7e-3 / 2.7e-2; 0 | 1.48; 4.1e-2 / 1.22; 1 | 1.38; 1.5e-2 / 3.6e-1; 0 |
| long_docs_12k | 2048 | 6.49; 2.9e-2 / 7.5e-1; 0 | 6.97; 4.5e-2 / 1.24; 0 | 2.21; 1.6e-2 / 4.0e-1; 0 |
| long_code_12k | 1024 | 2.67; 2.1e-3 / 1.7e-2; 0 | 7.04; 2.2e-3 / 1.2e-2; 0 | 3.46; 1.6e-3 / 1.3e-2; 0 |
| long_code_12k | 2048 | 1.96; 2.4e-3 / 1.5e-2; 0 | 7.43; 1.9e-3 / 1.2e-2; 0 | 2.13; 2.8e-3 / 2.6e-2; 0 |
| long_docs_4k (oracle-judged) | 1024 | 1.4e-2 / 8.5e-2; 3 | 1.8e-2 / 8.3e-2; 2 | 2.0e-2 / 1.6e-1; 1 |
| long_docs_4k (oracle-judged) | 2048 | 1.3e-2 / 5.4e-2; 1 | 1.5e-2 / 7.3e-2; 3 | 1.8e-2 / 1.5e-1; 2 |

`long_docs_4k`, the one prompt with a committed f64 CPU `continue` oracle, passes the oracle-relative rule for all three arms
at every chunk run. Warm/cold oracle KL ratios: lt 0.79x to 0.92x, f8 1.02x to 1.46x, default 0.76x to 1.13x. The
`cold chunk A vs cold chunk B` lines of `--reuse-diag` are bitwise for some pairs on every arm (e.g. 2048 vs 4096). They look
capped by the engine's chunk and were not used as evidence.

## Speed

No engine code changed, so #122's same-window localeval (window w1, spark2, 2026-10-23 13:11-13:26) is still the measurement.
No new window was run. Prefill tok/s against vLLM: default -16.5 % / -9.6 % / -14.0 % (1k / 4k / 16k); lt chunk 2048
-9.5 % / +1.8 % / -3.8 %; lt chunk 8192 -9.2 % / **+8.6 %** / **+6.4 %**. Decode is unchanged.

## Build used

The measurements ran on `main` 240e9d2a. The PR base also has #120 (HIP hybrid batching). Its core changes (`fusion.rs`,
`model.rs`, `model/batch.rs`) affect only batched steps (`batch_rows >= 2`) through the HIP fusion table. None of them runs
on a CUDA single-stream reuse run, so the results stand on the PR base.

## No-op

This PR changes no engine code: one GPU test, `scripts/lt/`, this document and LOG rows. Every binary path is identical to
main, so batching-off and lt-off output is trivially main's.

## Open user decisions

1. **Reuse `continue` judge.** The fixed 0.25-nat stream rule fails any change of rounding on `chat_reuse_4k` (lt; f8 at 1024)
   and `long_code_4k` (lt at 2048). The default passes because its near-ties fall outside the 16-token window, while its own
   shared rows already differ by 1.1 to 6.5 nats. Options:
   - (A), the deciding experiment: commit f64 CPU `continue` oracles for `chat_reuse_4k` (split 4521) and `long_code_4k` (splits 2000, 3072), so these
     rows are judged by the existing oracle-relative rule (1.5x + 3e-3), as `long_docs_4k` already is. CPU cost on spark1:
     about 38 GB RSS per oracle (fx5c). This is the only measurement that settles whether lt's extra warm/cold drift (point 3)
     is error or noise.
   - (B) Waive the rule for lt.
   - (C) Keep lt opt-in.
   
   Lane lt did not change the rule or the tolerance.
2. **`acc-short` baseline.** The baseline gate is relative to main's exact numerics (x1.15). lt (+16 %) and f8 (+33 %) both
   exceed it, inside vLLM's own run-to-run noise and the absolute floor. Default-on lt would need the baseline re-recorded
   from the lt build, which is the user's call.
3. Default-on, and the chunk 8192 recipe change, remain the user's calls (LOG 2026-10-23 15:40).

## fleetq jobs (owner `claude-f31ccf91-lt`)

| job | what |
|---|---|
| #4670, #4671 | spark1 build and test binaries |
| #4681 | spark2 build |
| #4672 | diag d1 |
| #4674 | GPU tests and sweep s1 |
| #4680 | sweep s2 |
| #4683 | reuse controls r1 |
| #4679 | acc-short attempt, which errored: unstamped baseline |
| #4690 | acc-short as2 |
| #4691 | GPU tests, `docs_env_vars` and `kernels test --quick` |
