# MFP8: one 16-row launch for the Metal fp8 multi-row GEMV (2026-10-30) - NEGATIVE RESULT

Lane `mfp8`, branch `lane-mfp8/rows16` (from `main` `3acb65809`). endor (M5 Max 40c), Qwen3.6-35B-A3B NVFP4, Metal.
Goal (MPROF bottleneck #1): at 9-16 rows `row_chunks` runs `matmul_fp8_gemv_rows` as two 8-row launches (fp8 group 10.0 ms at B=9..16 vs 5.5 ms at B=8).

**Result: a single 16-row launch of the existing simdgroup-matrix kernel (`fp8_rows_mmt`, `R = 2`) is SLOWER than the two 8-row launches
(fp8 group +1.4 ms, step +1.5 ms at B=9..16), and two attempts at making it faster failed. Stopped per the lane rule. The 16-row form is
kept in the tree but OFF by default (`INTARSIA_METAL_FP8_ROWS16=1` turns it on); no served A/B was run (the harness already shows a regression).**

## What was built

* `kernels/metal/fp8.metal`: `fp8_rows_mmt_impl` got a second template parameter `NT` (tiles of 8 outputs per threadgroup, default 1 = the old kernel) and `R = 2`
  entries `fp8_rows_mmt_{bf16,f32}{,_neg}_r2` (q(x) padded to 16 rows, 9..=16 rows in one launch). Each row block `r` is the same code on its own
  rows, so every row keeps the one-row accumulation order.
* `fp8.rs`: `mmt_padded(m)`, `rows16_on()`, scratch for 16 padded rows (`rows_scratch`, `rows_scratch_q`); `lib.rs` `row_chunks` skips fp8 when `rows16_on()` and the call has its scratch.
* Only fp8 was touched (coordinator scope change: nvfp4 small-N, `gemv_multirow.metal`, `moe.*` belong to other lanes).

## Correctness (holds, both ways)

`tests/rows_exact.rs` (all 7 tests, run at default and with `INTARSIA_METAL_FP8_ROWS16=1`, fleetq #7305): `fp8_rows_are_bitwise_the_one_row_gemv` covers M = 2..=16
(so 9, 12, 13, 16 included) x 8 shapes (N,K up to 8192x2048 and 2048x4096, odd N/K, NaN-code weights) x 3 input scales x bf16/f32 out x two weight/x offsets;
every row bitwise equal to the one-row `matmul_fp8_gemv`. The test's scratch size was updated to the padded-16 size. With
`INTARSIA_METAL_NEGCTL=fp8_mmt` the same test FAILS (the gate is not vacuous). Also passed with `NT = 2` and `NT = 4` (#7297) before they were removed.
Served greedy batch-16 identity was not run (nothing to ship).

## Harness, before/after (fleetq #7289, #7298; `raw/w1ab`, `raw/w1ab2`)

`batch_metal_real::real_35b_batch_step_real_traffic`, `MM_STEPS=32`, same binary, arms interleaved per round, B order flipped on even rounds.
`base` = two 8-row launches (main behaviour); `r16` = one 16-row launch. Plain ms/step, median of rounds:

| arm (NT) | B=1 | B=4 | B=8 | B=9 | B=12 | B=16 | rounds |
|---|---|---|---|---|---|---|---|
| base | 8.03 | 13.79 | 20.05 | 25.82 | 30.64 | 35.83 | 3 |
| r16, NT=1 (this tree, opt-in) | 8.03 | 14.09 | 20.13 | 28.06 | 32.23 | 38.23 | 3 |
| base | 7.98 | 13.92 | 19.93 | 26.27 | 30.59 | 37.25 | 2 |
| r16, NT=2 | 8.01 | 13.88 | 19.90 | 41.28 | 45.43 | 50.97 | 2 |
| r16, NT=4 | 8.02 | 13.91 | 19.90 | 88.53 | 92.74 | 98.13 | 2 |

Per-op (`ce1` in-stream, mean of rounds; `raw/ab_ce1_w1ab.md`, `raw/ab_ce1_w1ab2.md`, analyser `raw/scripts/mk_ab.py`), dense fp8 group ms/step, NT=1:

| | B=8 | B=9 | B=12 | B=16 |
|---|---|---|---|---|
| base | 5.47 | 10.00 | 10.02 | 10.08 |
| r16 | 5.49 | 11.30 | 11.39 | 11.53 |
| 8192x2048 base / r16 | 2.30 / 2.30 | 4.22 / 5.44 | 4.45 / 5.47 | 4.46 / 5.50 |
| 4096x2048 | 1.03 / 1.02 | 1.91 / 2.18 | 1.92 / 2.19 | 1.93 / 2.19 |
| 2048x4096 | 1.89 / 1.92 | 3.31 / 3.36 | 3.17 / 3.41 | 3.20 / 3.51 |
| 512x2048 | 0.25 / 0.25 | 0.56 / 0.32 | 0.48 / 0.32 | 0.48 / 0.32 |

Only the small 512x2048 shape wins (-0.16 ms/step); everything else loses. B <= 8 is unchanged (same kernel, NT=1), so there is no single-stream effect.

## Why (diagnosis, not verified by a counter)

The 8-row launch of 8192x2048 costs 57 us (16.8 MB, 293 GB/s); the 16-row launch costs 137 us, i.e. 2.4x for 2x the rows. The cost scales with the row
count, not the weight bytes, so the kernel at R=2 is not limited by the weight stream. The kernel loads the block-transposed q(x) operand from device
memory per simdgroup per 16-term block (`xb`, two 4-byte loads per row block) with no reuse across the 8 outputs' worth of work: per threadgroup of 8 outputs
it reads K x MP x 2 bytes of q(x) (64 KB at R=2) against 16 KB of weights, so the activation reads (cache traffic) dominate and double with R. The two
8-row launches read the same total activation bytes, but each at a lower, better-overlapped rate. Making the activation operand reused across more
weight tiles (`NT = 2, 4`: one activation load feeding 2 / 4 weight-tile MMAs) was expected to cut that traffic, but was 2.5x / 7x slower. That scaling
(superlinear in NT, in the per-lane `acc[NT][R][4]` + `wa/wb[NT][4]` arrays) looks like register spilling or lost occupancy; Metal gives no spill report
here and I did not verify it (the Metal toolchain constraint below also made quick compiler-stat experiments awkward).

The coordinator's estimate (single launch saves ~4.7 ms, not 7.9) assumed the 16-row launch costs the same per launch as an 8-row launch; it costs 2.4x.

## What would be next (not tried; the lane rule stops at two failed attempts)

* A different kernel shape for 9-16 rows rather than `R = 2` of the matrix kernel: the scalar `_xd` path (`fp8_gemv_impl`, MB 4, reads q(x) once per row from device memory) at MB 8..16 with per-thread accumulators fixed at 8 rows, or the matrix form with the q(x) block staged in threadgroup memory per simdgroup (K-block-local, so it fits) to take the activation loads off the device path.
* Check spills first (xcode `MTLComputePipelineState` `threadExecutionWidth`/`staticThreadgroupMemoryLength` and an occupancy probe via `maxTotalThreadsPerThreadgroup` for `NT = 1, 2, 4`; `examples/pso_info.rs` exists).
* The 512x2048 win (0.16 ms/step) could be taken with a shape gate (N <= 512), +0.4 %; not done.

## Environment notes

* **Metal toolchain on endor.** Xcode was updated to 27.2 (27B52) at 00:54 today and `xcrun metal` now fails ("missing Metal Toolchain", `xcodebuild -showComponent metalToolchain` = uninstalled, needs 27B47). Builds here route `xcrun -sdk macosx metal|metallib` to the already-mounted 27.1 toolchain (`/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.1.266.1.BNM1Up/Metal.xctoolchain`) through a PATH shim (`raw/scripts/xcrun-shim`, installed as `.lane/bin/xcrun`; `build.sh` prepends it). Nothing was downloaded or installed. Other lanes building on endor will hit the same error. A touch of the `.metal` file is needed after changing the shim because `build.rs` does not watch `PATH`.
* "Before" is the same binary with the 16-row form off, not a separate main build. The runs in `raw/w1ab*` used a build where the 16-row form was default-on and `base` was `INTARSIA_METAL_FP8_ROWS16=0`; `w1ab.sh` was then updated to the final semantics (default = base, `=1` = r16).

## fleetq

endor: #7283/#7285/#7287/#7296/#7304 builds (`--load cpu`), #7288/#7297/#7305 rows_exact, #7289 harness A/B NT=1 (3 rounds, 26 GB, `--quiet gpu,cpu`), #7298 harness A/B base/NT=2/NT=4 (2 rounds). Reproduce: `raw/scripts/build.sh`, then `fleetq run --host endor --quiet gpu,cpu --mem 26 --expect 15 -- raw/scripts/w1ab.sh OUT 3`.
