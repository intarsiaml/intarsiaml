# MMOE: expert-grouped MoE down at 9-16 rows on Metal (2026-10-30): negative result

Lane `mmoe`, branch `lane-mmoe/main`, base `main` `3acb65809`. endor (Apple M5 Max 40-core), Qwen3.6-35B-A3B NVFP4
(`~/models/Qwen3.6-35B-A3B-NVFP4`). Raw logs, scripts and traces: [raw/](2026-10-30-mmoe-metal-moe-grouped-down/raw/).

## 0. Result in one paragraph

Giving each distinct expert one threadgroup row that streams its down-projection block once and serves its pairs (the MPROF
idea, `moe_lead` ballot scheme) is **slower than the per-pair kernel `moe_dec_dn_p` at 9-16 rows in every one of 13
(rows-per-simdgroup x pairs-per-leader) variants tried**, bitwise-exact in all of them. On the real 35B at B = 16 the down
projection goes 5.17 ms -> 6.89 ms (best variant) / 7.98 ms (the design as proposed), and the plain decode step 35.8 ms ->
37.5 / 38.5 ms. A control (same row tile, no grouping) isolates the cause: grouping itself costs time, the weight re-read it
removes was not the bottleneck. Per the lane rules (two failed attempts: the variant sweep, then its refinement) I stopped
before gate-up and before any served A/B: gate-up grouping is the same mechanism, and a served A/B of a slower kernel
would only measure how much slower. The code stays in the tree **off by default** (`INTARSIA_METAL_MOE_DN_GROUP`, 0 = never)
so the experiment reproduces; nothing changes for any default run.

## 1. What was built

`kernels/metal/moe.metal`: `moe_dec_dn_g_<DR>x<DG>` (template `moe_dec_dn_g_impl<NEG, DR, DG>`) with `moe_lead_n<XR>` (the
existing `moe_lead` generalised to the pairs-per-leader count; `moe_lead` itself is the `XR = MOE_XR` wrapper, `_x`
kernels unchanged). Grid `(ceil(H / (4 * DR)), pairs)` like the per-pair kernel; a threadgroup row whose pair is the first
of its expert (or the first of the next `DG` of it, for experts with more than `DG` pairs) serves up to `DG` pairs of that
expert: per lane-strided weight block it decodes the `DR` rows' blocks once (`moe_blk_dec`) and, for each pair, loads the h
block once and runs `moe_blk_dot_f` (the same 16-term fma chain as `moe_blk_dot`), the scale fma, then `simd_sum` and
`f2bf(t * gs)`. Per (pair, row) the arithmetic is the per-pair kernel's; only the pair-to-threadgroup assignment changes.
`crates/intarsia-backend-metal/src/moe.rs`: the rows path swaps the down shader when `T >= INTARSIA_METAL_MOE_DN_GROUP`;
`INTARSIA_METAL_MOE_DN_GV` picks the variant (`DN_G` table); a one-line stderr note names the shader when the path is on.
Control `INTARSIA_METAL_NEGCTL=moe_dn_g` swaps in `moe_dec_dn_g_2x4_neg` (reversed 16-term chain) and leaves the per-pair
kernels real, so a gate failure belongs to the grouped kernel (the existing `moe_dn` / `moe_gu` controls are unchanged).

## 2. Exactness (all variants)

`moe_experts_rows_are_bitwise_the_one_token_call` already runs T = 1..16 over six routings x bf16/f32 out x two weight
offsets (so 9, 12, 16 are covered); added a seventh routing `pool120` (8 distinct experts per token from a 120-expert pool,
~79.7 distinct of 128 pairs at T = 16, like the served 77-82). With the grouped path on (`..._GROUP=9`):

| gate | result |
|---|---|
| that test, all 7 `DN_G` variants (`raw/gate2/`), plus the 6 of the first sweep (`raw/gate1/`) | pass, 0 differing bytes |
| full `rows_exact` suite, default / GROUP=9 gv 0, 1, 4 / GROUP=2 gv 1 (`raw/gates/suite-*.log`) | 7 passed, 0 failed (2 ignored timing tests) |
| `NEGCTL=moe_dn_g` GROUP=9 | **fails**: `T=12 routing=random out=f32 ... first differing byte 34120` (T >= 9 as intended) |
| `NEGCTL=moe_dn` / `moe_gu` (default path), and `moe_dn` with GROUP=9 | fail (T=4 / T=5 / T=4), unchanged |

Served greedy-token identity was not run: it is meaningful only for a kernel one would ship.

## 3. Timing

Method. (a) Micro: new ignored test `timing_moe_rows16` (`rows_exact.rs`): `moe_experts_bf16_rows` at 35B shapes (E = 256,
H = 2048, I = 512, K = 8), pool120 routing, 24 routings cycled per call (weights cold), 3 x 96 calls per T; wall per call
and `INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:` per-dispatch GPU time (MPROF's `ce1`). (b) Real model:
`batch_metal_real::real_35b_batch_step_real_traffic` (MPROF's harness, 4 warm-up + 32 steps, B = 9, 12, 16), a plain run and a
`ce1` run per arm, arms interleaved `off, gv0, gv1, off` in one quiet fleetq slot. Single binary, env toggles
(removes build-to-build variance). `raw/scripts/`.

### 3.1 Micro, down us per call, ce1 (`raw/micro2-ce1-summary.txt`; `off` run twice, 2nd in parentheses)

| variant (rows/simdgroup x pairs/leader) | T=9 | T=12 | T=16 |
|---|---:|---:|---:|
| **off: per-pair `moe_dec_dn_p` (2 rows)** | **97 (99)** | **136 (122)** | **143 (146)** |
| 2x4 (the proposal) | 135 | 178 | 208 |
| 4x4 | 120 | 146 | 163 |
| 4x2 | 103 | 140 | 164 |
| 2x2 | 125 | 164 | 193 |
| 8x2 | 164 | 205 | 282 |
| 4x1 (control: 4 rows, no grouping, same scan) | 104 | 135 | 153 |
| 8x1 | 157 | 194 | 235 |

Whole-call wall at T = 16 (gate-up + down + gather): off 327.7 / 327.2 us, 4x1 345, 4x2 350, 4x4 362, 2x2 379, 2x4 388,
8x1 422, 8x2 466 (`raw/micro2.log`). Sweep 1 (2x4, 2x8, 4x4, 4x8, 1x8, 1x16; `raw/micro1*`): 2x8 247, 4x8 201, 1x8 385, 1x16 539 us
down at T=16, all slower. T = 8 (below the threshold, same kernels in all arms) is flat across runs (96-102 us down), the noise
check. Gate-up is untouched by this change (~180 us at T = 16 in every arm, differences are run noise).

### 3.2 Real model, B = 16 (`raw/h1/`, `raw/h1-ce1-summary.txt`)

| B | arm | plain step ms | tok/s | ce1 gate-up | ce1 down | ce1 step |
|---|---|---:|---:|---:|---:|---:|
| 9 | off | 25.51 / 25.40 | 353 / 354 | 5.90 | 3.07 | 27.40 |
| 9 | grouped 2x4 | 27.13 | 332 | 5.86 | 4.71 | 28.62 |
| 9 | grouped 4x2 | 26.66 | 338 | 5.84 | 4.23 | 28.36 |
| 12 | off | 30.58 / 30.05 | 392 / 399 | 7.27 | 3.96 | 31.28 |
| 12 | grouped 2x4 | 32.16 | 373 | 7.29 | 6.01 | 33.25 |
| 12 | grouped 4x2 | 31.44 | 382 | 7.20 | 5.32 | 32.57 |
| 16 | off | 35.80 / 35.73 | 447 / 448 | 9.11 | **5.17** | 36.82 |
| 16 | grouped 2x4 | 38.49 | 416 | 9.07 | **7.98** | 39.53 |
| 16 | grouped 4x2 | 37.49 | 427 | 9.04 | **6.89** | 38.44 |

(`off` twice agree to 0.1 ms plain, 0.01 ms ce1 down; ms are mean per step, 32 steps.)

## 4. Why it loses

* The premise that the per-pair kernel re-reads an expert's block from DRAM does not hold: pairs of one expert run close in
  time on different cores and the second read comes from cache (lane mm already measured this at 8 rows,
  `2026-10-25-mm-metal-moe-grouped.md`). Down time scales with distinct experts here, sub-linearly in pairs: T 8 -> 16
  doubles pairs (64 -> 128), distinct experts x1.57, down time x1.43 (100 -> 143 us): a straight line through the two points is ~1.5 us per distinct expert plus ~24 us fixed.
  The 3.2 ms bandwidth floor of the MPROF doc counts DRAM bytes only; the kernel is not at it for latency reasons, not
  re-reads.
* What grouping changes is parallelism: the pairs of an expert, which ran concurrently on separate threadgroups, now run
  one after another inside one threadgroup (and the leaders are unevenly loaded). The control isolates this: at the same row
  tile and the same ballot scan, `4x1` 153 us vs `4x2` 164 us vs `4x4` 163 us at T=16, and 2 -> 4 -> 8 pairs per leader or
  8 rows per simdgroup keeps getting slower (register pressure: `s[DR][DG]` + decoded `wf[DR][16]` + h block).
* Decoding each weight block once instead of per pair (the other thing grouping saves) is not worth the lost parallelism either.
* This agrees with MFP8's finding for the multi-row fp8 kernel (cost follows rows, not weight bytes).

A third variation would be a compacted leader list (pre-pass removing the idle non-leader threadgroups and the per-group ballot scan).
The `4x1` control shows the scan costs at most ~10 us of the 143, far below what grouping loses, so I did not build it.

## 5. What is left

* Not run (nothing worth gating): gate-up grouping, the served 1/4/8/16-stream A/B, single-stream, token identity at 16 streams.
* Down at 16 rows is 5.17 ms vs a 3.2 ms bytes floor; the gap is not re-reads. Candidates untried here: more bytes in flight
  per lane (the kernel has one 8 B weight block per row per lane at I = 512) without serialising pairs, e.g. a different
  split of the 32 blocks over lanes, which changes the per-lane chain and so the simd_sum association: not row-exact, would
  need all rows kernels (and the solo path) changed together.
* Fleet note: while starting, I killed another lane's `fleetq run` client (S32's `build.sh cli-main`, job #7269 on spark1) by
  mistake (wrong PID); its job record was left queued with no runner. S32 must requeue it.

Commands: `raw/scripts/` (`build.sh` with the Xcode 27.2 metal shim `xcrun-shim`, `gate.sh`, `gates.sh`, `micro.sh`, `h1.sh`,
analysers). fleetq jobs (mine): #7293 base build and #7299 first new build (both produced an empty metallib: Xcode 27.2
Metal toolchain, rebuilt with the shim in #7308), #7303 first gate run (failed to start on the empty metallib), #7308
base+new builds, #7309 gate sweep 1, #7310/#7311 micro sweep 1 (#7310 released after a script permission error), #7313 new
build v2, #7314 gate sweep 2, #7316 micro sweep 2, #7319 real-model harness, #7322 rows_exact suite + negative controls.
