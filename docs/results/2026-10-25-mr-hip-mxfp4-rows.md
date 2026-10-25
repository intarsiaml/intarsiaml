# mr: HIP MXFP4 multi-row GEMV without the register collapse (lane mr, 2026-10-25)

Branch `lane-mr/hip-mxfp4-rows` (base `main` c6c19896, which contains lane sp's `_rows` launchers). Draft PR. HIP only; kernels/hip/mxfp4.hip,
crates/intarsia-backend-hip (registration + tests), scripts/mr, docs. Nothing outside the HIP backend changed.

## Result

The row-exact MXFP4 launchers (`matmul_mxfp4_gemv_rows`, `matmul_mxfp4_pair_gemv_rows`, the kernels `OpCall::exact_rows` selects: MTP
verify, batched decode) now cover M = 1..16 (`ROWS_MAX`), and M = 2..16 runs a kernel whose registers do not grow with M. On the 27B's
17408x5120 MLP GEMV (us per call, ratio to M=1 in brackets): M=3 114 -> 87 (1.40 -> 1.08), M=4 155 -> 93 (1.91 -> 1.16), M=8 1535 -> 156
(18.9 -> 1.94), M=16 (no row-exact kernel before) 272 (3.4). Every output row is bitwise the one-row launch. M=1 is not touched (same launcher,
same bits, same time). The plain `matmul_mxfp4_gemv` (the non-row-exact registration, M <= 8) is untouched.

Consequence for the engine: the only op that stopped Qwen3.8-27B MXFP4 from batching on HIP was `matmul_mxfp4` at m = 9..16 (below). With
this PR the 27B batched decode equals the solo step for N = 2..16 sequences on the real checkpoint, and `serve --max-batch 16` runs.

## Step 0: what production selected and why the bench collapsed

* Registry (`crates/intarsia-backend-hip/src/quantmoe.rs`): `matmul_mxfp4_gemv` (priority 10) is selected for every M = 1..8; `matmul_mxfp4_mma`
  only above 8 or when gemv is excluded. Lane sp's `_rows` twins (priority -1, `rows_exact`) were the same launcher for M <= 8. So lane sp's
  timings are the production timings, not a forced bench. At M = 9..16 an `exact_rows` call had no MXFP4 kernel at all (mma is not row-exact).
* Cause of the collapse (`-Rpass-analysis=kernel-resource-usage`, gfx1201, raw/resource-usage-gemv-base.log): `mxfp4_gemv_kernel<M>` hoists the
  loads and bf16 unpacks of all M activation rows, about 33 VGPR per row.

  | M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|---|
  | VGPR | 111 | 144 | 177 | 210 | 243 | 256 | 256 | 256 |
  | spilled VGPR | 0 | 0 | 0 | 0 | 0 | 20 | 60 | 126 |
  | scratch bytes/lane | 0 | 0 | 0 | 0 | 0 | 84 | 216 | 348 |
  | occupancy (waves/SIMD) | 12 | 10 | 8 | 7 | 5 | 5 | 5 | 5 |

  The step from M=5 to M=8 is the spill (and occupancy 5 from M=5 on).

## The kernel

`mxfp4_gemv_rows_kernel<M>` (kernels/hip/mxfp4.hip). Per (output row n, activation row m) the arithmetic is the one-row kernel's: the lane
walks the groups u = lane, lane + 32, ... in order, t = fma chain over the 32 elements of the group, `acc = fma(t, s, acc)`, then the
xor-butterfly warp sum. Only the register shape differs:

* a warp computes 2 weight rows (`kRowsR`), so the activation loads and bf16 unpacks of a group are shared by both rows;
* each activation row is streamed (loaded, unpacked and consumed 8 elements at a time) behind `__builtin_amdgcn_sched_barrier(0)`, so the
  compiler cannot hoist the loads of the next row: 104 VGPR at M=2 to 134 at M=16, no spill, occupancy 10-12;
* x is the QDQ'd pre-pass output (the launcher runs the existing `mxfp4_xprep_kernel` first, as the old M > 1 path did), plain bf16.

Variants measured with the standalone harness (scripts/mr/mxbench.hip, raw/harness-variants-*.log, 17408x5120, us):

| variant | M=2 | M=3 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|
| 1 weight row / warp, barrier | 82 | 95 | 110 | 181 | 324 |
| 2 rows, no barrier | 86 | 91 | 101 | 162 | 289 |
| **2 rows, barrier (shipped)** | **84** | **88** | **97** | **149** | **260** |
| 4 rows, barrier | 87 | 91 | 96 | 143 (5120x17408) | spills |
| 3 rows, barrier (5120x17408) | 86 | 96 | 104 | 148 | 262 |
| 2 rows, 2 groups per iteration | 88 | 101 | 117 | 200 | 357 |
| 1 row, 2 / 4 groups per iteration | 88 / 128 | 110 / 165 | 134 / 191 | 232 / 315 | 411 / 600 |

Four rows per warp gains nothing at M <= 4 and spills above 6; unrolling groups only adds registers. M = 3 and 4 are limited by ALU (fma plus
bf16 unpack: at M=16 the kernel sits at about 12 us per extra row, near the VALU rate of the 89M-weight GEMV), not by weight bytes.

## Microbenchmark (production launcher, `rows_exact timing_mxfp4_rows`, card 0, us per call, ratio to M=1 in brackets)

| shape N x K | kernel | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 | M=12 | M=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 17408x5120 | before (`matmul_mxfp4_gemv`) | 81 (1.00) | 85 (1.05) | 114 (1.40) | 155 (1.91) | 217 (2.67) | n/a | n/a | 1535 (18.88) | n/a | n/a |
| | after (`_rows`) | 80 (1.00) | 83 (1.04) | 87 (1.08) | 93 (1.16) | 108 (1.35) | 127 (1.58) | 141 (1.76) | 156 (1.94) | 215 (2.68) | 272 (3.39) |
| 5120x17408 | before (`matmul_mxfp4_gemv`) | 81 (1.00) | 86 (1.06) | 110 (1.35) | 144 (1.78) | 208 (2.55) | n/a | n/a | 1551 (19.07) | n/a | n/a |
| | after (`_rows`) | 80 (1.00) | 87 (1.09) | 95 (1.19) | 105 (1.31) | 118 (1.48) | 133 (1.66) | 142 (1.78) | 156 (1.95) | 216 (2.70) | 273 (3.41) |
| 12288x5120 | before (`matmul_mxfp4_gemv`) | 60 (1.00) | 63 (1.04) | 82 (1.35) | 110 (1.83) | 157 (2.60) | n/a | n/a | 917 (15.18) | n/a | n/a |
| | after (`_rows`) | 60 (1.00) | 63 (1.05) | 66 (1.11) | 70 (1.17) | 78 (1.32) | 92 (1.54) | 103 (1.73) | 115 (1.92) | 157 (2.63) | 197 (3.31) |
| 5120x6144 | before (`matmul_mxfp4_gemv`) | 36 (1.00) | 41 (1.16) | 50 (1.41) | 62 (1.75) | 86 (2.41) | n/a | n/a | 464 (13.01) | n/a | n/a |
| | after (`_rows`) | 35 (1.00) | 44 (1.25) | 49 (1.38) | 56 (1.58) | 60 (1.71) | 63 (1.79) | 67 (1.90) | 72 (2.05) | 92 (2.60) | 115 (3.27) |
| 1024x5120 | before (`matmul_mxfp4_gemv`) | 13 (1.00) | 15 (1.13) | 16 (1.21) | 21 (1.60) | 24 (1.77) | n/a | n/a | 87 (6.50) | n/a | n/a |
| | after (`_rows`) | 13 (1.00) | 16 (1.17) | 17 (1.28) | 18 (1.38) | 20 (1.47) | 21 (1.57) | 22 (1.67) | 24 (1.78) | 30 (2.27) | 36 (2.70) |

"before" is `matmul_mxfp4_gemv` (the old kernel; M=6, 7 and M > 8 were not run for it, M=6 and 7 cost 1.8x and 3.2x of M=5 in the harness raw log
because of the spills). M=1 is the same launcher in both rows. Raw: raw/gates-g1-timing-rows_exact-quick.log.
Cross-check with the harness (3 rotating weight copies so the 64 MB MALL cannot hold a whole matrix): raw/harness-final-all-shapes.log, same
ratios. Note the first call of the harness run on a cold GPU reads high (`gemv baseline 17408x5120 M=1 214 us`); ignore it.

## Gates

| gate | result |
|---|---|
| `rows_exact` `mxfp4_rows_are_bitwise_the_one_row_gemv`, M=1..16, 12 shapes (27B q 12288x5120, k/v 1024x5120, o 5120x6144, MLP 17408x5120 and 5120x17408, 10240x5120, 6144x5120; odd N, K=32, ragged tails), W4A4 and W4A16, bf16 and f32 out, 48 cases per M, 768 rows | PASS |
| negative control (`INTARSIA_HIP_GB_CONTROL=matmul_mxfp4_gemv_rows=matmul_mxfp4_mma`: the mma kernel runs where the rows kernel does) | FAILS as required: M=1 row 0 differs from solo in 2 of 12288 elements (n=12288 k=5120) |
| harness bitwise, 12 shapes incl. N=1, 9, 33, 300 and K=32, 2080, 4096, 6176, M=1..16, f32 out, every row vs the M=1 launch | 0 mismatches |
| `rows_registry` | PASS |
| `kernels test --backend hip --quick` | PASS (64 kernels, 353 cases) |
| 27B (Qwen3.8-27B-Quark-AWQ-MXFP4) batched decode == solo step, real checkpoint, HIP, N = 2..16 sequences (`batch_cuda_real real_35b_batched_decode_is_bitwise_the_solo_step` with `INTARSIA_QWEN36_DIR` set to the 27B snapshot) | dev PASS, 694 s (raw/gates-g2-27b-batched-equals-solo.log) |
| same test on base (main c6c19896) | FAILS at plan time, see "Plan check" |
| no-op, batching off, dev vs main c6c19896: `plan show` text and json, 256 greedy ids after a short and a ~3.5k-token prompt, 8B, 35B, 27B | all byte-identical / identical (raw/gates-g3-noop-vs-main.log) |
| CUDA / Metal no-op | not run: only kernels/hip, crates/intarsia-backend-hip, docs and scripts changed |

## Plan check: what blocked the 27B on HIP (scope item: lifting the refusal)

The 27B is the dense-FFN hybrid recipe (`recipes/qwen3_5_vl.toml`: GatedDeltaNet + gated attention + dense MLP). `hip_batch_needs_hybrid`
(`model/batch.rs`) refuses only pure-dense models (the 8B) and does not fire for it. On base the plan check
(`plan::check_batch_ops`, run by `Model::enable_batch(16)`) rejects exactly one op:

```
1 op(s) of this model have no kernel on hip/gfx1201:
  - matmul_mxfp4: 64 shape(s) without a kernel, e.g. no kernel for matmul_mxfp4 on hip/gfx1201 (m=9 n=10240 k=5120 act=Some(Bf16) w=Some(U8)) [row-exact verify kernels only]
```

Every other op of the 27B (norms, bf16 matmul/logits rows, GatedDeltaNet and attention row tables, MXFP4 pair, ...) already has a row-exact
kernel. With this PR the check passes at 16 rows. Not done and not needed for the unlock: nothing else was rejected.

## 27B served, HIP, one R9700, localeval `speed` (1024-token prompts, 128 forced tokens, 3 reps), `serve --max-batch N`, max-seq 8192

Aggregate = localeval's aggregate decode tok/s across the parallel streams (it includes the streams' prefills, which serialise: 16 prompts of
1k tokens are ~16 s of prefill at ~980 tok/s). Per-stream = per-stream decode tok/s. Raw: raw/localeval-*-summary.md.

| arm | single stream decode | 4 streams (agg / per-stream) | 8 streams | 16 streams |
|---|---|---|---|---|
| main (`--max-batch 1`, the only mode main had for the 27B) | 32.8 | 26.2 / 32.8 | 26.1 / 32.8 | 26.1 / 32.8 |
| dev `--max-batch 1` (no-op check) | 32.8 | 26.2 / 32.8 | 26.1 / 32.8 | 26.1 / 32.8 |
| main `--max-batch 8` (old kernel, spills at M=8) | 32.8 | 45.5 / 15.3 | 19.2 / 2.7 | 18.6 / 2.5 |
| dev `--max-batch 8` | 32.8 | 51.8 / 18.4 | 61.5 / 11.6 | 62.9 / 8.9 |
| dev `--max-batch 16` | 32.9 | 52.0 / 18.5 | 61.7 / 11.6 | 64.9 / 6.3 |

Single-stream decode at `--max-batch 1` is identical to main (32.8 vs 32.8 tok/s, ranges overlap). With batching the 27B goes from 26 tok/s
(one sequence at a time) to 62-65 tok/s aggregate on one card; main's own `--max-batch 8` is slower than no batching at 8 streams (19 tok/s)
because of the spill. Reference: radiance serves the 27B at about 525 tok/s at 16 streams, on TP2 over two cards; this is one card and the
aggregate figure here includes serialised prefill, so the two are not like for like. The gap that remains is not the MXFP4 GEMV: at 8 rows the
MXFP4 GEMVs cost 1.8-2.0x the one-row time, but per-stream decode falls from 32.8 to 11.6 tok/s (a batched step costs ~2.8x a solo step), so
the other row-exact ops and the per-row parts of the step (GatedDeltaNet, attention, lm_head rows) now dominate; profiling that step is the
next lever (`--profile-every` / `--op-table` at `--max-batch 8`), not done here.

## Method and reproduction

* Harness: `scripts/mr/mxbench.hip` (standalone hipcc, includes kernels/hip/mxfp4.hip; row-vs-solo bitwise check and timing on rotating weight
  copies). Build with `scripts/mr/mrbuild.sh`; windows: `scripts/mr/win.sh`; gates `scripts/mr/gates.sh [S1..S4]`; A/B `scripts/mr/ab.sh`
  with `scripts/mr/serve.sh`; trees `scripts/mr/sync.sh`, builds `scripts/mr/build.sh` (cargo targets bind-mounted on /data: the root disk is full).
* fleetq: #5160 (harness windows), #5165 (cargo builds), #5193/#5194/#5195 (gates), #5196 (A/B). tatooine radiance was taken down inside those slots only and
  was restored and confirmed active (`systemctl --user is-active`, answering on :8080) before each slot was released.
* Base for no-op and A/B: main c6c19896. Dev tree = this branch.

## What is not done

* The `_rows` pair launcher at M >= 2 runs two single launches (as before); only M=1 fuses. No separate pair test beyond the shared kernel.
* CUDA and Metal MXFP4 row-exact paths were not touched.
* The remaining per-step cost at 8-16 rows (see above) is not profiled.
* tatooine leftovers: see the final report.
