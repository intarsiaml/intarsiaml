# HIP WMMA small-row GEMV/GEMM family (lane wg), 2026-10-25

Draft. Opt-in behind `INTARSIA_HIP_WMMA_ROWS=1`; with the knob off nothing changes. A default flip is a separate user decision.
Raw logs: `2026-10-25-wg-hip-wmma-gemv/raw/`; machine-readable summary: `2026-10-25-wg-hip-wmma-gemv/trace.json`.
Measured on tatooine (R9700, gfx1201, one card), Qwen3.6-35B-A3B-NVFP4, tree stamp `1b16702a` (branch on origin/main `5e9cc6da`; the branch is
105 commits behind main now and has at least one conflict, `crates/intarsia-core/src/plan.rs`, so it needs a rebase and a re-run of the gates).

## Result

| 35B, served (localeval, 1k prompt, `--gen 512`, 3 reps, 3 interleaved passes) | knob off | option 1 | change |
|---|---|---|---|
| 1 stream, decode tok/s | 164.8 | 159.1 | -3.5 % |
| 4 streams, aggregate tok/s | 272.0 | 276.3 | +1.6 % |
| 8 streams, aggregate tok/s | 318.4 | 362.3 | +13.8 % |
| 16 streams, aggregate tok/s | 335.4 | 432.5 | +29.0 % |
| 16 streams, steady ITL (1000 / per-stream peak) | 35.97 ms | 30.44 ms | -15.4 % |
| 16 streams, steady aggregate (streams x peak) | 444.8 | 525.6 | +18.2 % |

Option 1 = `INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all`. Pass-to-pass spread was under 2 % in every cell
(`raw/served2/summary.txt`). The steady-ITL line is the 1000/peak figure, a noisier metric than the aggregate.

Model loop (`batch_cuda_real` `real_35b_batch_step_timing`, ms per step, interleaved, spread <= 0.2 ms):

| table | B=1 | B=4 | B=16 |
|---|---|---|---|
| off (= main) | 6.09 | 13.47 | 40.8 |
| option 1 (`fp8a8=all,bf16=all`) | 6.34 (+4.1 %) | 13.37 (-0.7 %) | 29.7 (-27 %) |
| `fp8a8=all,bf16=off` | 6.34 (+4.0 %) | 13.53 (+0.4 %) | 32.5 (-20 %) |
| `fp8a8=off,bf16=all` | 6.26 (+2.8 %) | 13.35 (-0.9 %) | 38.1 (-7 %) |

Earlier model-loop points (window D, x3): B=2 9.20 -> 9.35 ms (+1.7 %); profile pass B=8 -13 %. Per op at B=16: `matmul_fp8` 13.9 -> 5.6 ms
(130 calls), bf16 `matmul` 5.9 -> 3.1 ms (140 calls); `matmul_nvfp4` (7.7 ms) and `moe_experts` (13.7 ms) unchanged.

## Recommendation

The task's rule is that single stream may drop at most 1 %. Option 1 costs 3.5 % (served) to 4.1 % (model loop) at one stream, so it cannot become
the default without a waiver of that rule. The four choices:

* (a) knob off / W8A16-only table: no 35B change (the 35B dense projections are W8A8 and bf16). Safe; leaves the gain unclaimed.
* (b) option 1: +14 % at 8 streams, +29 % at 16, parity at 4, -3.5 % at 1. Batched == solo bitwise at every M; solo output changes once.
* (c) switch on at batch >= 4: dropped (breaks batched == solo bitwise at the switch; user: not waivable).
* (d) per kind: keeps batched == solo, but is dominated. fp8-only costs the same 4.0 % at B=1 for 74 % of the B=16 gain; bf16-only costs 2.8 % at B=1
  for 7 % at B=16.

Recommendation: ship (b) as the opt-in knob now (this PR, default unchanged = (a)), and ask the user for the default flip to (b) only if a 3.5 % single-stream
cost is acceptable for a batched-decode server (+29 % at 16 streams, +14 % at 8). If single stream must stay within 1 %, keep the knob off and take the M=1 tuning
candidates below first; they target the +0.25 ms.

## What is in the PR

* `kernels/hip/ik_f8_rows.h` (+ `fp8.hip`, `ik_hip_abi.h`, `ik_nf4_rows.h`, `ik_dq_gemm.h`, `ik_f8_gemm.h`): one WMMA kernel family for M = 1..64, W8A8 (fp8 WMMA) and
  W8A16 (bf16 WMMA), plus the bf16 router/gate and the decode folds (pair, norm+pair, gated, sigmoid). Row m is bitwise the same call on row m alone, whatever the other
  rows hold: K is cut into chunks of 32, chunk c belongs to wave c % 8, each wave adds its chunks in ascending order into one accumulator, the 8 waves are added in order
  0..7. Nothing in that order depends on M, the token tile or the window.
* `WmmaTable` (`quantmoe.rs`): per-kind row ranges, default `fp8a16=1-16` only. Env: `INTARSIA_HIP_WMMA_ROWS`, `INTARSIA_HIP_WMMA_TABLE`,
  `INTARSIA_HIP_WMMA_DIRECT_MIN` (`docs/tuning.md` rows added).
* Fragment-order fp8 weights behind `INTARSIA_HIP_FP8_FRAG=1` (approved layout exception, `docs/decisions/LOG.md`): bitwise equal to row-major in every reader, no step-time gain
  measured (opt1 6.30 vs opt1-frag 6.30 ms at B=1, 30.0 vs 30.0 ms at B=16). Kept off.
* Load log (`relayout:` line), plan JSON `weight_layout`, tests, and the window scripts (`scripts/wg/`, `scripts/wgserve/`, `scripts/wgfit/`).

Solo output changes once: WMMA rows are not bitwise equal to the GEMV one-row call (rel_l2 about 1e-6 to 3e-5 on random data). The hardware's fp8 WMMA sum is also not a
wide exact add (probe below), so no scalar emulation of it exists.

## B=1 chase: cause unknown

Option 1 costs +0.25 ms per B=1 step (6.08 -> 6.33 ms, x6 interleaved pairs, spread <= 0.02 ms). Ruled out:

* captured graphs: with `INTARSIA_BATCH_GRAPHS=0` the step is 6.07 / 6.08 (off) vs 6.32 / 6.34 ms (option 1), the same delta;
* noise: five pairs agree to 0.02 ms;
* per-op profile: the profiled ops add only 0.03 ms of the 0.25 ms (B=1: pair 0.235 -> 0.242, gated 0.141 -> 0.155, matmul_fp8 0.062 -> 0.065, sigmoid 0.047 -> 0.051, norm+pair
  0.034 -> 0.038 ms/step);
* extra launches: at M=1 the family is one STAGED launch, as the GEMV.

Not measurable here: the decode kernels never show up in `INTARSIA_HIP_TRACE` (graph replays and the batched path are not traced), and `rocprofv3` fails in the dev image
(`libdw.so.1` missing). Likely, unproven: the M=1 microbench losses per fold (+0.4 to +3.1 us: 2048x4096 +2.35, 4096x2048 +0.75, 512x2048 +0.43, bf16 32+32 pair +1.1 us),
about 150 folds a step. The per-kind runs are not additive (fp8 +0.24, bf16 +0.17, both +0.25 ms), so part of the cost may be shared; one more arm
(`ROWS=1` with fp8a8 and bf16 both off) would show it.

Fp8 ablation at M=1 (relative figures only; the ablation build is slower in absolute terms), us, full / no x staging / no cross-wave sum:
2048x2048 14.4 / 11.1 / 12.5; 2048x4096 20.2 / 18.3 / 20.3; 4096x2048 19.7 / 17.4 / 20.1; 12288x2048 46.8 / 43.7 / 46.3; 512x2048 6.0 / 5.6 / 6.2 (`raw/served/abl.log`).
Staging x (quantise, write LDS, barrier) is the largest removable item.

Later candidates, bits identical by construction (the summation order stays; split-k is out because it would change it), not built, no GPU run:
1. at M=1 write and sum only the 16 valid outputs per wave in the cross-wave step, same add order;
2. give the cross-wave buffer its own LDS region and drop one block barrier;
3. a 32-feature tile per block;
4. each wave quantises only its own K/8 slice into wave-private LDS, dropping the block barrier before the loop.

## Probe: the hardware sum is not a wide exact add

`scripts/wg/probe_wmma.cpp` drives one `v_wmma_f32_16x16x16_fp8_fp8` (16 products plus C) with crafted, random (2,000,000) and chained cases; `scripts/wgfit/probe_fit.py`
searches a model family (aligned wide add with `width`/truncation/rounding, and grouped adds). First dump (window E) was void: the result store sat inside a divergent branch and the
compiler moved the WMMA under partial lane activity. Fixed; window F sanity: layout check 0 of 256 differ, easy cases 0 of 11 differ, repeat run 0 differences.

* No model in the family is bitwise exact. Best on 100k random cases: `GR(8, 26, floor, rne)` 83.1 %, `AL(27, tz, rne)` 74.3 %; every best model misses the same 130 of 1955 crafted cases.
* Hardware result equals the exact RNE sum in 73.4 % of random cases and is within 1 ulp of it in 89.3 %.
* Cancellation (+2^16 and -2^16 plus tiny terms): 240 of 240 cases differ from the exact sum for each tiny exponent, so there is no single wide aligned add.
* k order: swapping adjacent pairs changes 0 of 4000 outputs; reversing or swapping halves changes 726 of 4000. The sum is pairwise first.
* Two chained instructions equal two separate ones (200000 cases, 0 differences).

Verdict: scalar emulation of the WMMA sum is not feasible with this model family. Not a correctness problem: the output is deterministic and row-invariant by construction.

## Gates

| gate | status |
|---|---|
| unit tests, row invariance M = 1..64 with negative controls (`rows_exact`: wmma fp8 / bf16 bitwise one-row call, folds bitwise vs unfused chain, default / all-direct / all-staged, fragment layout, layout knobs) | pass (window D, tree `aa6e3e16`; kernels unchanged since). Negative controls fail as intended (3 of 3). |
| `kernels test --quick` (hip) | not run on the final tree |
| `batch_recurrent --test-threads=1` | not run on the final tree |
| 35B `batch_cuda_real` hip incl. `--ignored` | timing, profile and op-listing tests run (pass); the identity / MTP-verify tests not run on this branch |
| accuracy: KL goldens, acc-short vs HIP baseline, qualify | not run |
| CUDA and Metal no-ops (shared crate `intarsia-core/src/plan.rs` changed) | not run |
| speed: localeval 1/4/8/16 streams, interleaved x3, same window | done (above); 1 stream and 4-16 streams in the same window; steady ITL reported |

The accuracy, identity and no-op gates are required before any default flip; this PR is knob-only.

## Method notes and limits

* One card (device 0), radiance down for each window and restored (`active`) before the slot was released; all windows were `fleetq` jobs with `--takes-down radiance`.
* The served runs use `--max-batch 16 --max-seq 8192 --mem-budget 29`, thinking off, one model, one prompt length (1024 tokens). Pass 3 ran in full (6 visits).
* The first served attempt failed (server `--max-seq 4096`, localeval calibration prompt is 4461 tokens); the numbers above are the rerun.
* Option 1's gain is at B >= 8 on this model; the NVFP4 experts (7.7 ms) and `moe_experts` (13.7 ms) at B=16 are untouched by this PR.
