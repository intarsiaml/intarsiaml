# mr3: HIP lm_head rows kernel above 8 rows, and an ALU check of the MXFP4 rows kernel (lane mr, 2026-10-25)

Branch `lane-mr/hip-logits-rows` (base `main` 5241377b). HIP only: kernels/hip/matmul.hip (one new kernel and launcher, used by
`logits_bf16_gemv_rows` only), crates/intarsia-backend-hip/tests/rows_exact.rs, scripts/mr, docs. The other row-exact bf16 kernel
(`matmul_bf16_gemv_rows`, the one lane wg's WMMA table dispatches) and every kernel at M <= 8 are untouched. The profile test-harness changes of
PR #147 (`INTARSIA_TIMING_ROWS` in `real_35b_batch_step_profile`, lm_head timing to M=16) are repeated here so the gates run from one tree.

## Result

`logits_bf16_gemv_rows` at M = 9..16 now runs `gemv_rows_wide_kernel`, which computes the same numbers per (output row, activation row) as the
one-row kernel (the lane walks idx = lane, lane + 32, ... ascending, 8 fma per idx in element order, the same warp sum) and differs only in the register
shape. Every row stays bitwise the M=1 launch.

Cause of the cliff (`-Rpass-analysis=kernel-resource-usage`, raw/lgbuild.log): `gemv_kernel<M, 2, 4>` is 233 VGPR without spills at M=8 and 256 VGPR with
16..173 spilled VGPR (68..500 B/lane scratch, occupancy 5) at M = 9..16, the same cause as the MXFP4 GEMV before #145. The new kernel is 4 weight rows per
warp (the weights of an idx are unpacked once for all M rows, each loaded activation row feeds 4 of them), one activation row at a time behind a scheduler barrier:
192 VGPR at M=16, no spill, occupancy 8.

27B head 248320 x 5120 (production launcher, `rows_exact timing_mxfp4_rows`, card 0, us per call):

| M | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| before (main) | 4242 | 4262 | 4279 | 4344 | 4434 | 4451 | 4504 | 7849 | 16701 |
| after | 4244 | 4263 | 4280 | 4345 | 4437 | 4464 | 4508 | 4620 | 4639 |

M <= 8 is the same kernel as before (identical within run noise). M=16 is 3.6x faster, 1.09x of M=1 (the 2.5 GB of weights stream once at 548 GB/s).
Raw: raw/gates-win1/timing-mxfp4.log (before: 2026-10-25-mr-hip-27b-batch-profile raw/micro.log of PR #147).

27B batched step, `real_35b_batch_step_timing` (unprofiled ms per step; before = PR #147 raw/timing.log, same tree otherwise):

| B | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| before | 30.31 | 36.77 | 42.12 | 59.47 | 82.70 | 110.87 |
| after | 30.37 | 36.79 | 42.31 | 59.95 | 79.86 | 99.85 |
| aggregate tok/s after | 32.9 | 54.4 | 94.5 | 133.4 | 150.3 | 160.2 |

B=16: -11.0 ms (-9.9 %), aggregate 144 -> 160 tok/s; B=12: -2.8 ms; B <= 8 within noise (the +0.8 % at B=8 is the same kernels). Per-op profile
(raw/step-profile-after.log): `logits` at B=16 is 4.58 ms profiled against 15.01 before; op sum 111.35 against 120.2.

Served, one R9700, localeval speed (1k prompt, 128 tokens, `--max-batch 16`, max-seq 8192), aggregate decode tok/s (per stream); base = main 5241377b:

| arm | 1 stream | 4 | 8 | 16 |
|---|---|---|---|---|
| main | 32.8 | 51.9 (18.4) | 61.6 (11.6) | 64.9 (6.3) |
| this PR | 32.8 | 52.0 (18.4) | 61.7 (11.6) | 67.2 (6.6) |

Single stream is identical. The gain shows at 16 streams only (+3.5 %): localeval's streams are prefilled one after another, so the decode batch rarely holds more than 8-12
rows at a time and the 1k-token prefills dominate the aggregate. A pure decode load with 16 live sequences gets the full 10 % of the step above.

## Gates (tatooine HIP, fleetq #5306 / #5307, radiance restored and confirmed active after each)

| gate | result |
|---|---|
| `rows_exact bf16_matmul_and_logits_rows_are_bitwise_the_one_row_gemv`, logits at M=1..16 incl. the 27B head (248320 x 5120, tiled wide-exponent pool), 24000 x 5120, 9 x 5128 and the earlier shapes, f32 out | PASS (raw/gates-win1/g-rows.log) |
| negative control `INTARSIA_HIP_GB_CONTROL=logits_bf16_gemv_rows=logits_bf16` (the WMMA GEMM where the rows kernel runs) | FAILS as required: M=1 row 0 differs in 227 of 256 elements |
| harness bitwise (scripts/mr/lgbench.hip, 248320 x 5120, M=1..9 measured in the variant window; M=12 and 16 by the rows_exact case above) | 0 mismatches |
| `rows_registry`, `kernels test --backend hip --quick` | PASS, PASS (64 kernels, 353 cases) |
| 27B (Qwen3.8-27B-Quark-AWQ-MXFP4) batched decode == solo step, N=2..16, real checkpoint | dev PASS (684 s); main PASS (692 s) |
| no-op vs main 5241377b, batching off: `plan show` text and json, 256 greedy ids, short and ~3.5k prompts; 8B, 35B, 27B | all byte-identical (raw/gates-win1) |
| CUDA / Metal no-op | not run: only kernels/hip, crates/intarsia-backend-hip, docs, scripts changed |

Variant window for the shape (raw/win3-harness-variants.log): at M=9 four candidates (1, 2 or 4 weight rows per warp with the barrier, and 4 rows without it) all time
4.55-4.56 ms, the memory-bound floor there. The harness stopped at M=10 in that window (it instantiated only M=9, 12 and 16 and exited on the first missing M), so the variants were
not compared at M=12 and 16. R=4 with the barrier was fixed on the argument that the x traffic per weight byte is M/R, and the production timing above shows it reaches the byte floor at 12 and 16.
The variant knob is not in the PR.

## Item 2: ALU check of the MXFP4 rows kernel (PR #145's `mxfp4_gemv_rows_kernel`), no bits changed

ISA of `mxfp4_gemv_rows_kernel<16, float>` (gfx1201, raw/isa.log): 1684 VALU instructions, of which 382 `v_dual_fmac_f32` (VOPD dual issue, 764 fma) and 574 single
`v_fmac_f32`, so about 57 % of the fmas already dual-issue; `v_and`/`v_lshl`/`v_lshr` (the bf16 unpack among them) total only ~330; the warp-sum epilogue is 160
`ds_bpermute_b32` + 158 `v_add_f32` + about 190 waits. At K=5120 the loop runs only 5 iterations per lane, so that epilogue is a visible share. 515 `s_delay_alu`.

Order-preserving changes tried in the harness (experiment branch `lane-mr/hip-mxfp4-alu-experiment`, not a PR; every row bitwise the M=1 launch, 0 mismatches):

* reduce-scatter warp sum: the same xor 16, 8, 4, 2, 1 pairing for all R x M accumulators of a lane, but each step exchanges only the half the partner keeps, so about 31
  permutes instead of 160 (a + b == b + a, so the bits are those of `ik_warp_sum`);
* 2 or 4 activation rows per scheduler-barrier group (more independent fma chains in flight).

us per call, MXFP4 rows kernel, f32 out (raw/win3-harness-variants.log):

| shape, M | shipped | reduce-scatter | 2 rows/group | 2 rows + reduce-scatter |
|---|---|---|---|---|
| 17408x5120, M=8 | 151 | 146 | 165 | 158 |
| 17408x5120, M=16 | 264 | 249 | 280 | 268 |
| 5120x17408, M=16 | 268 | 260 | 278 | 273 |
| 5120x6144, M=4 | 56 | 55 | 47 | 46 |
| 5120x6144, M=16 | 118 | 110 | 113 | 107 |

Reduce-scatter alone is a consistent 3-7 % at M=8..16 (more on the short-K shapes); 2 rows per group helps only M <= 4 on the small shape and loses on the big ones (more registers).
The GEMVs are about 75 % of the 16-row step (83 of 111 ms profiled), so reduce-scatter is worth roughly 2-5 ms of a 100 ms step: real but small, with a larger and more intricate
kernel. Decision (coordinator, 2026-10-25): no PR; it stays a measured option on branch `lane-mr/hip-mxfp4-alu-experiment`, to be revisited if the MXFP4 GEMV becomes the bottleneck again. Packed/dot instructions (bf16 dot2 accumulates in a different order) are not an option without changing bits against main. The remaining
levers are the compiler's VOPD pairing (574 singles) and a cheaper fp4 decode, neither controllable from HIP source with certainty.

## Method and reproduction

* scripts/mr/lgbench.hip + lgbuild.sh (standalone harness and VGPR report), gates.sh (`GATE_BUILD=1 GATE_TEST=bf16_matmul_and_logits_rows GATE_CTRL=logits_bf16_gemv_rows=logits_bf16 gates.sh S1 S3 S4 S2`),
  profile.sh, ab.sh/serve.sh, isa.sh, sync.sh.
* fleetq: #5273 (ISA, cpu), #5276 (harness builds), #5280 (variant window), #5306 (gates with builds), #5307 (A/B and step profile). One stale runner of a released
  job (#5288) still executed its command when its turn came; its output was discarded and the logs here are from #5306/#5307 only.
* fleetq plans radiance as 31 GB of the box, so cargo builds only fit inside a window that takes radiance down; the gates window therefore includes the builds.

## Not done

* lm_head rows below 9 rows (4.5 ms at M=8 against 4.24 at M=1: 6 % over the byte floor; not worth a new kernel).
* `matmul_bf16_gemv_rows` above 8 rows (same old kernel and the same spill; lane wg's file family, not touched). Left for now by decision (coordinator, 2026-10-25): wg's WMMA recommendation may move bf16 rows to WMMA and make the fix moot.
* A decode-only 16-sequence served measurement (localeval's streams are staggered by prefill).
