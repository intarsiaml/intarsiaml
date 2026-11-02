# MNARROW: Metal small-N GEMVs at 9-16 rows (2026-10-30)

Lane `mnarrow`, branch `lane-mnarrow/main`, base `main` `97f8a17e3`. endor (Apple M5 Max 40-core), Metal, Qwen3.6-35B-A3B NVFP4, `forward_batch` real-traffic harness (MPROF's `batch_metal_real::real_35b_batch_step_real_traffic`, `MM_STEPS=32`). Raw: [raw/](2026-10-30-mnarrow-metal-small-n-rows/raw/) (scripts, logs, per-op tables, one ce1 trace per arm).

**Result: docs-only. The env A/B the brief proposed as step 1 shows no gain (the 8-row kernel is 1.7 ms/step slower at 16 rows), so the rule `m >= 9 || n >= 512` was not implemented. No code is in this PR.** One code variant (no chunking for the narrow nvfp4 matrices) was built and measured before the coordinator's "env A/B no gain => docs-only" rule reached this lane; its numbers and patch are recorded below as a lead, not shipped.

## 1. The brief's mechanism, corrected

`nv_rows_mb` (`nvfp4.rs`) returns 1 for `n <= 4096`, and `launch_nvfp4` then dispatches the one-row shader over a grid of `(n/32, m)` threadgroups: all rows of one call are **one** launch, not a per-row loop. The 160 + 80 launches at 16 rows are `row_chunks` (`lib.rs`) splitting 9..16 rows into two calls, times 40 layers. The B=8 -> B=9 cliff in MPROF (shared expert 0.85 -> 1.33 ms, router pair 0.30 -> 0.43, in_a/in_b 0.18 -> 0.27) is that chunking.

## 2. Step 1: env A/B `INTARSIA_METAL_NV_ROWS_SPLIT=0` (8-row kernel everywhere), same binary, 3 interleaved rounds (fleetq #7317)

Plain wall, ms/step, median of 3 (`raw/envab/plain.md`):

| arm | B=8 | B=9 | B=12 | B=16 |
|---|---:|---:|---:|---:|
| default (one-row shader, chunked) | 19.77 | 25.62 | 30.22 | 35.74 |
| `SPLIT=0` (8-row shader) | 20.57 | 26.64 | 31.57 | 37.45 |

Every round of every B agrees (e.g. B=16 default 35.81/35.42/35.74, `SPLIT=0` 37.50/37.10/37.45). Per op (ce1, `raw/envab/groups-*.md`), shared expert: 0.85/1.33/1.50/1.75 ms (B=8/9/12/16) vs 1.53/2.50/3.01/3.77; gate/up 0.925 -> 2.694 ms and down 0.723 -> 0.975 ms at B=16. Router pair and in_a/in_b do not use this path (unchanged, 0.59 / 0.33 ms). The 8-row shader gets only 32 / 128 threadgroups on the 512 / 2048-row matrices, so it is starved; the existing rule (one-row shader up to N=4096) is right at 9-16 rows too. `m >= 9 || n >= 512` would send both shared-expert matrices to it and lose.

## 3. Lead, not shipped: one launch for the narrow nvfp4 matrices (built before the docs-only rule arrived)

Variant (`raw/a1/variant-not-shipped.patch`): `row_chunks` skips `matmul_nvfp4_gemv_rows` when `nv_rows_mb(n) == 1`, and `launch_nvfp4` accepts up to 16 rows in that case (grid height m). No shader change; lm_head (n=248320, 8-row shader) stays chunked. The patch also holds a knob that runs a 16-row bf16 pair as one `_mr16x1` launch; that part was **never built or measured**.

* Exactness: `rows_exact` (all 7 gates, incl. `nvfp4_rows_are_bitwise_the_one_row_gemv` at M=2..16 for 512x2048 and 2048x512, 360 cases / 3240 rows) passed on the variant (fleetq #7327, `raw/a1/rows_exact.log`). Not done: served greedy-token identity at 1/8/16 streams, the NEGCTL stage (there is no nvfp4 NEGCTL; the existing four are fp8/moe), served A/B.
* Microbench (`timing_rows`, 200 enqueues, median of 3, us per call, `raw/a1/`): 512x2048 M=9/12/16: chunked 9.8/12.5/14.8 -> one launch 8.0/10.4/11.7; 2048x512: 9.9/11.8/14.2 -> 8.4/10.2/12.7. M=8 unchanged. It is a launch-count saving (about 2-3 us per call), not a cost-scales-with-rows regression.
* Harness A/B, main vs variant, 5 interleaved rounds (fleetq #7329, `raw/a1ab/`), plain ms/step median: B=9 25.72 -> 25.42, B=12 30.23 -> 29.98, B=16 35.79 -> 35.62 (B=1/4/8 unchanged: 8.00/13.80/19.79 vs 7.99/13.87/19.74). Per op (ce1), shared expert 0.80/1.33/1.50/1.74 -> 0.77/0.89/1.07/1.29 ms at B=8/9/12/16 (gate/up launches 160 -> 80, down 80 -> 40). The op-level gain is real (about 0.45 ms at B=16, 0.44 at B=9) but the wall gain (0.17-0.30 ms) is inside the 0.3-0.4 ms harness round spread, and served runs vary 10-20 %, so a served A/B cannot resolve it. The router pair (0.59 ms) and in_a/in_b (0.34 ms) were not touched.

If the coordinator wants the variant, the remaining work is: re-apply the patch, served greedy identity at 1/8/16 streams, the gates script, and a PR; the expected upside is about 1 % of a 16-row step.

## 4. Not done

* bf16 pair grid-over-rows (router pair, in_a/in_b): not implemented. The whole 8->16 growth of those two ops is 0.46 ms and 9..15 rows would need new shader work; given the MFP8 and MMOE negatives and the coordinator rule, stopped.
* Served A/B, identity runs, MB=2/4 nvfp4 shader variants: not run.

## 5. Provenance

Builds #7315 (main), #7324 (variant); env A/B #7317; microbench + gate #7327; harness A/B #7329 (all endor, `--quiet gpu,cpu` for timing). Build used a PATH `xcrun` shim to the mounted 27.1 Metal toolchain (`raw/scripts/xcrun-shim`, from `lane-mfp8/rows16`); nothing downloaded. Traces kept: ce1 round 1 of each arm (others dropped for size; logs and per-op tables of all rounds kept).
