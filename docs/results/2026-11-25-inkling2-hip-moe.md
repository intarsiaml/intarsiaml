# Inkling-2-Mini experts on HIP (lane ink-hip-moe, package C, 2026-11-25)

Branch `ink/hip-moe`, base origin/main `d8c3c7148` + `ink/hip` `6c5ee5d67` (lead's skeleton). Ops: `MoeExperts` situ (NVFP4),
`MoeExpertsMxfp8` (72). W4A16 only.

## 0. Briefs received (kept here against context loss)

- Launch brief (coordinator, ~21:50 EST): package C; `moe_experts_situ_bf16` + `_rows`, `moe_experts_mxfp8_bf16` + `_rows`,
  registered `.situ()`, never `.act_quant()`; 128 experts, top-8, D 2304, I 1152. Order: decode path correct and fast (bench vs
  627 GB/s on the active experts' bytes), then a correct prefill path, speed second; WMMA prefill is speed package S2.
  Files: `kernels/hip/ink_moe.hip`, `kernels/hip/ik_hip_abi_inkmoe.h`, `crates/intarsia-backend-hip/src/inkmoe.rs`,
  `crates/intarsia-backend-hip/tests/ink_moe*.rs`, this doc, `HANDOFF.md`.
- Coordinator message after the skeleton landed: skeleton is `6c5ee5d67` on `ink/hip`, merge it; module shape and notes are in
  the last section of `.worktrees/hip-swarm/COMMON.md`. The router is the lead's, in its own module (`src/inkroute.rs`,
  `ink_route.hip`), so `inkmoe.rs` is this lane's alone. Write a SIMT `ink_moe_gu` / `dn` correctness form first, at any pair
  count, with `SITU_TOL (1.5e-2, 2e-3)`; the fast decode form after that.

- Coordinator message (from the ink-hip lead, after moe_route's first window): tatooine is ready for integration; when the expert
  ops pass their unit window, send the coordinator "ready to merge at <commit>". Checkpoint on tatooine (read-only):
  `/data/models/intarsia/Inkling-2-Mini` (in the dev image `/models/Inkling-2-Mini`). Window scripts to copy, not edit:
  `docs/results/2026-11-25-inkling2-hip/raw/window.sh` (stops radiance, capped stage in its own process group, restore trap, polls
  /v1/models on :8080) and `run.sh` (dev image, card 0), both on `ink/hip`. Own `INTARSIA_TARGET_VOL=intarsia-cargo-target-<lane>`
  and own container name. `ink/hip` head has the scripts plus the hip arm of the tiny tests on top of `6c5ee5d67`; no re-merge
  needed. The lead merges each lane once its ops pass their own unit window, then runs admission + tiny G1 + real-model G2 in one
  window once ops 67-74 and the experts exist; the experts block that window, so correct first, fast after. moe_route passed its
  window (#11924, ADMITTED 19/19). Window rules unchanged: fleetq `--takes-down radiance`, 20 min or less, at most 3 windows,
  none after 06:00, radiance back up before release.

- Coordinator message after the lead's window #11943: the whole model runs on one R9700 and G2 passes (`ink/hip` with
  `29b71ac67` merged). First decode 62.68 tok/s, 16.0 ms per token, 55 % of the 114 tok/s floor; the lead's counters put
  moe_experts first, so the decode speed round is priority 1. Start from `ink/hip` head (it contains this form); the result
  must be bitwise against `29b71ac67`. Target NVFP4 T=1 (100.8 us, 57 % of 627 GB/s); MXFP8 is 135.1 us (78 %). Two windows
  left, same rules (20 min or less, none after 06:00, radiance back up). The lead is profiling one decode token; its split
  will be forwarded, with a redirect if the experts are not the largest item. When a faster form passes, send "ready to merge
  at <commit>" with before / after us per call.

- Coordinator message with the lead's GPU profile (#11945 / #11947): the experts are the top item. Decode 16.0 ms per token
  against an 8.76 ms floor. NVFP4 experts 4.53 ms per token (gate_up 2.82, down 1.71) over 44 visits, floor 2.51 ms: about
  2.0 ms recoverable (62 -> about 71 tok/s). MXFP8 experts 0.58 ms against 0.42 (4 visits). Prefill at 4k (chunk 256, 8.17 s):
  NVFP4 experts 70.5 %, MXFP8 7.3 %, so the SIMT experts are 78 % of prefill; flat 524-552 tok/s across chunk 256-2048.
  Order: (1) NVFP4 decode gate_up and down toward bandwidth, bitwise against `29b71ac67`; (2) WMMA prefill experts (S2), the
  lead estimates about 3x before attention dominates; keep each op's Exactness class from the port map, admission + G2 must
  still pass, rows-exact stays bitwise where the op promises it. Windows: 2 left, none after 06:00. Report each landing as
  "ready to merge at <commit>" with before / after us per call (decode) or ms per chunk (prefill).

- Coordinator message after landing 1 (`afd58ae34` noted; the defaults being bit-checked as explicit settings is enough, no
  PRE=3 pin). Window rule change for all HIP lanes: the count cap becomes a time budget: each window keeps radiance down 5 min
  or less; this lane's total radiance-down time from now on 20 min or less; none after 06:00; radiance must answer
  `/v1/models` before release; log each window's down time in this doc. Go on: (1) decode round 2, LDS sized by the call,
  bitwise against `29b71ac67`; (2) S2 WMMA prefill tested on the GPU, admission + rows-exact where the op promises it, report
  ms per chunk at 2048 before and after. Integration: PR #516 (the correctness form) stays as it is until it merges; speed
  heads go in a follow-up branch after that, so do not merge into `ink/hip`; send each as "ready to merge at <commit>".

- Coordinator message after landing 1 (`afd58ae34` noted; the defaults being bit-checked as explicit settings is enough, no
  PRE=3 pin). Window rule change for all HIP lanes: the count cap becomes a time budget: each window keeps radiance down 5 min
  or less; this lane's total radiance-down time from now on 20 min or less; none after 06:00; radiance must answer
  `/v1/models` before release; log each window's down time in this doc. Go on: (1) decode round 2, LDS sized by the call,
  bitwise against `29b71ac67`; (2) S2 WMMA prefill tested on the GPU, admission + rows-exact where the op promises it, report
  ms per chunk at 2048 before and after. Integration: PR #516 (the correctness form) stays as it is until it merges; speed
  heads go in a follow-up branch after that, so do not merge into `ink/hip`; send each as "ready to merge at <commit>".

- Coordinator message after landing 2 (`628c78c63`): condition before it is taken for the follow-up branch: no regression at
  any row count the model can issue (MXFP8 T=64 1658 -> 2354 us is a real case: batch decode with 64 sequences or 8 x 8,
  speculative verify groups, the tail of a prefill chunk). Do: (1) select the lean form by row count, both formats, at the
  measured crossover; keep the old form where it is faster (bitwise holds as each form is bitwise to `29b71ac67`); (2) report
  a sweep table for NVFP4 and MXFP8 at T = 1, 2, 4, 8, 16, 32, 64, 128, 256, 2048, old vs new us per call, folded into a
  window already needed (round 3 or S2), not a window of its own; (3) then round 3 (in-wave grouping), then S2. Send the next
  "ready to merge" once no row count is slower than `29b71ac67` by more than noise (2 %).

- Coordinator message during window 6: PR #516 (the correctness form) is merged into main as `4fc9901d4`. When the row-count
  sweep passes, merge `origin/main` into this branch and send "ready to merge at <commit>". The attn lane's op 75 goes to
  main first in its own PR (`ink/hip-speed`); this lane's speed work follows as a second PR.

- Coordinator message after landing 3: `a154a5704` accepted; it goes into the attn lane's integration branch `ink/hip-speed`
  (one window, one PR for both). First add `docs/tuning.md` rows for every new env knob (`INTARSIA_HIP_INK_MOE_LEAN`, others),
  run `docs_env_vars` and the host tests, commit, send "rows at <commit>". Then S2 (WMMA prefill), based on origin/main.

- Coordinator message after S2 (`2ba2d4154`): tonight's rule is bitwise defaults only, so S2 ships off by default; the flip
  is the user's call in the morning, next to the attn lane's flash prefill (`INTARSIA_HIP_INK_ATTN_FLASH`, also not bitwise).
  Do: (1) `INTARSIA_HIP_INK_MOE_WMMA` default 0, the 16-pairs-per-expert rule kept as the value to flip to, one constant,
  tuning row updated; (2) merge `ink/hip-speed` (head `083e30f8f`: a154a5704 + attn op 75 + mix, md5 == main and G2 passed);
  (3) one evaluation window, 5 min or less down (13:37 left): admission with WMMA on incl. the real-shape T=2048 cases
  (`INK_MOE_BIG=1`); WMMA off: greedy md5 equal to `ink/hip-speed` at 64 tokens on a 4k prompt; G2 short and long with WMMA
  on, flash on, both on; 4k and 16k prefill tok/s for off, WMMA, flash, both; (4) full table in this doc, then "ready to
  merge at <commit>" with the table; PR after ink/hip-speed merges, defaults off.

- Coordinator message after the evaluation: accepted at `2ae9ae326`; it is in PR #519 (branch `ink/hip-s2`, stacked on #518).
  No more radiance windows tonight for this lane; stand down. Keep `~/intarsiaml-ink-hip-moe` and the docker volume
  `intarsia-cargo-target-ink-hip-moe` on tatooine until told (a morning flip check may reuse the build). Optional, no GPU:
  write the S2 speed work not done as a next-steps list at the end of this doc; commit, do not push.

## 0b. Radiance down time under the time budget (20 min from the rule change; windows 1 and 2 predate it: 52 s, 63 s)

| Window | fleetq | Down | Budget used |
|---|---|---|---|
| 3 | #11951 | 22:09:18 - 22:10:15, 57 s | 0:57 |
| 4 | #11957 | 22:14:14 - 22:15:12, 58 s | 1:55 |
| 5 | #11967 | 22:24:06 - 22:25:45, 99 s | 3:34 |
| 6 | #11973 | 22:30:58 - 22:31:53, 55 s | 4:29 |
| 7 (S2 window 1) | #11982 | 22:41:32 - 22:42:25, 53 s | 5:22 |
| 8 (S2 window 2) | #11987 | 22:49:36 - 22:50:37, 61 s | 6:23 |
| 9 (S2 evaluation) | #11990 | 22:57:14 - 23:00:38, 204 s | 9:47 |

## 1. Status

| Step | State |
|---|---|
| Correctness form (SIMT union-grouped port of `kernels/cuda/ink_moe.cu`), both ops, all four launchers | done, passes on the R9700 (window 1, fleetq #11935) |
| Rows-exact at every T (shared-memory path, pair-table path, every load window) | done, bitwise (window 1) |
| Decode bench vs 627 GB/s | NVFP4 T=1 100.8 us = 355 GB/s (57 %); MXFP8 T=1 135.1 us = 486 GB/s (78 %); section 3 |
| Decode speed round 1 (lean step, window 2) | NVFP4 T=1 100.8 -> 83.8 us (427 GB/s, 68 %), MXFP8 135.1 -> 126.1 us (521 GB/s, 83 %); bitwise |
| Decode speed round 2 (LDS by the call, load windows; windows 3 and 4) | NVFP4 T=1 81.6 us (439 GB/s, 70 %), T=16 623.5 us (587 GB/s, 94 %); MXFP8 T=1 122.6 us (536 GB/s, 85 %); bitwise |
| Decode speed round 3 (in-wave grouping; window 5) | negative (88.3 against 81.6 us at T=1), kept opt-in |
| Row-count rule, no regression against `29b71ac67` (window 6) | done: every T from 1 to 2048 is 11 % to 35 % faster, both formats; bitwise |
| Prefill, S2 (bf16 WMMA on the pair table, ordinary names; windows 7 to 9) | done, OFF by default (`INTARSIA_HIP_INK_MOE_WMMA=1` turns it on): NVFP4 T=2048 39.6 -> 6.2 ms per call; in the model 4k prefill 652 -> 1614 tok/s at chunk 2048; admission and G2 pass with it on; default md5 equal to `ink/hip-speed` |

## 2. Design (kernels/hip/ink_moe.hip)

A port of CUDA's SIMT kernels (`ink_moe_gu_kernel` / `ink_moe_dn_kernel` / group / gather), plain weight layout only, W4A16 only
(`aq != 0` returns `IK_ERR_UNSUPPORTED`). Same per-(pair, row) arithmetic as CUDA: 8 lanes per gate / up row (4 per down row),
32-element chunks, two 16-fma chains per NVFP4 chunk (one 32-fma chain per MXFP8 chunk), `fmaf(p, scale, acc)`, xor-shuffle lane
sum, global scale, bf16 rounding points of `inkling2::ops`. Differences on gfx12:

- e2m1 / e4m3 decode through `v_cvt_pk_f32_fp8` (the `kMwCvt` scheme of `moe_wmma_r7.h`: e2m1 code -> e4m3 byte of value * 2^-6,
  block scale * 64), no LDS table.
- No prefetch / PDL. Instead a load window: a lane requests the weights of PRE of its steps before computing on the first
  (`INTARSIA_HIP_INK_MOE_PRE` = 1 | 3 | 9, default 3; bitwise the same output).
- Pairs per accumulator batch 8 (CUDA 4). LDS 32.4 KB per block.
- Not ported: the decode repack (`ink.moe_layout = 1`), W4A4, the tensor-core prefill (`ink_moe_mma.cu`, 85-101 KB shared memory).

Every T takes the same kernels: up to 512 pairs grouped per block in LDS, above that a pair table in scratch (one grouping
launch) and a gather launch. So the `_rows` twins are the same launchers and the whole op is rows-exact.

## 3. Results

### Window 1 (fleetq #11935, radiance down 21:47:06 - 21:47:58, 52 s; log `2026-11-25-inkling2-hip-moe/raw/window1-11935.log`)

Build: fleetq #11928 (`--load cpu`), `cargo test --release -p intarsia-backend-hip --test ink_moe --no-run`, exit 0.

Tests: `INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_moe -- --test-threads=1
ink_moe_matches_cpu_reference ink_moe_rows_exact`: 2 passed.
- `ink_moe_matches_cpu_reference`: 71 kernel x case checks against the cpu reference with the admission's `judge` (the full
  non-quick sweep of both ops below T = 1024, pair-table shapes with 520 - 600 pairs, hot / duplicate / invalid routing), all
  pass. Worst rel_l2 1.63e-4 (MXFP8 gelu, real H / I) against a budget of 1e-3; worst NVFP4 situ 1.41e-4 against 2e-3.
- `ink_moe_rows_exact`: T = 5, 16, 70 (560 pairs: pair-table path), both formats: every checked row bitwise the one-row call,
  load windows 1 / 3 / 9 bitwise equal, `_rows` twin bitwise the ordinary kernel.
- Not run: the sweep's T = 2048 cases (`INK_MOE_BIG=1`); the lead's admission window covers them.

Bench (`INK_MOE_BENCH=1 ... ink_moe_bench`): real shape (top-8, H 2304, I 1152; NVFP4 128 experts, MXFP8 64), 12 calls with
different routings back to back, best of 5; GB/s over the payload + scale bytes of the distinct experts a call reads.
"Roofline" = 627 GB/s (the figure of the brief; not re-measured here).

| Format | T | us / call (best window) | GB/s | of 627 | gate_up us | down us | MB / call |
|---|---|---|---|---|---|---|---|
| NVFP4 situ | 1 | 100.8 (pre 3) | 355 | 57 % | 62.3 | 34.3 | 35.8 |
| NVFP4 situ | 2 | 180.6 (pre 3) | 389 | 62 % | 110.8 | 61.9 | 70.2 |
| NVFP4 situ | 4 | 317.1 (pre 3) | 411 | 66 % | 192.8 | 110.4 | 130.3 |
| NVFP4 situ | 8 | 535.0 (pre 3) | 428 | 68 % | 331.9 | 187.5 | 229.2 |
| NVFP4 situ | 16 | 866.3 (pre 3) | 423 | 67 % | 539.9 | 308.8 | 366.2 |
| NVFP4 situ | 64 | 2039.7 (pre 3) | 277 | | 1457.8 | 530.0 | 565.8 |
| NVFP4 situ | 256 (pair table) | 6780.1 (pre 3) | 85 | | 4707.6 | 2001.9 | 573.3 |
| MXFP8 situ | 1 | 135.1 (pre 1) | 486 | 78 % | 78.7 | 55.8 | 65.7 |
| MXFP8 situ | 2 | 241.2 (pre 1) | 513 | 82 % | 138.6 | 101.3 | 123.9 |
| MXFP8 situ | 4 | 403.8 (pre 1) | 515 | 82 % | 226.6 | 172.9 | 208.0 |
| MXFP8 situ | 8 | 641.0 (pre 1) | 528 | 84 % | 363.8 | 270.2 | 338.7 |
| MXFP8 situ | 16 | 881.1 (pre 1) | 520 | 83 % | 498.2 | 378.3 | 458.5 |
| MXFP8 situ | 64 | 1658.4 (pre 1) | 317 | | 1081.2 | 645.5 | 525.5 |
| MXFP8 situ | 256 (pair table) | 6124.9 (pre 1) | 86 | | 3799.6 | 2464.6 | 525.5 |

Load window: 3 beats 1 for NVFP4 (100.8 against 108.4 us at T = 1) and for both formats' down; 1 beats 3 for MXFP8 gate_up
(78.7 against 91.9); 9 loses everywhere (143 / 211 us at T = 1).

Reading (from the T = 1, 2, 4, 8 slopes): a call costs about 17 us that do not scale with bytes (gate_up ~12, down ~6: two
launches, the in-block grouping, the last round trip) plus bytes at a marginal 640 GB/s (MXFP8 gate_up, at the roofline),
~500 (MXFP8 down), ~470 (NVFP4 gate_up) and ~420 GB/s (NVFP4 down). The window of 9 being slowest says register count
(waves resident per SIMD) matters: next round measures VGPRs and narrows the inner loop.

### Window 2 (fleetq #11948, radiance down 22:01:46 - 22:02:49; log `raw/window2-11948.log`): the lean step

VGPR counts from the device asm (`raw/asm.sh`, fleetq #11946, no GPU): the window-1 form compiled to 256 VGPRs and spilled
(gate_up: 140 - 780 bytes of stack per lane; down 228 - 235 VGPRs), i.e. 6 waves per SIMD. The lean step (a chunk as 4 units
of 8 weights, decoded and used at once; the token row's address hoisted; 4 pairs per batch on the shared-memory path; block
scales by one fp8 conversion) is the same chains in the same order: NVFP4 gate_up 130 - 170 VGPRs, down 97 - 127; MXFP8
gate_up 153 - 241, down 109 - 159; no spills.

Tests (same command as window 1): 2 passed. New in `ink_moe_rows_exact`: the lean form at windows 1 / 2 / 3 / 4 / 6 and 2 / 4
pairs per batch is bitwise the window-1 form (T 5, 16, 70, both formats), and so is the default.

Bench, T = 1, us per call (GB/s of 627):

| Form | NVFP4 both | gate_up | down | MXFP8 both | gate_up | down |
|---|---|---|---|---|---|---|
| window-1 form, window 3 | 99.6 (360) | 61.8 | 34.4 | 144.7 (454) | 89.6 | 49.8 |
| lean, 4 pairs, window 3 | 84.0 (426) | 49.3 | 34.8 | 125.0 (526) | 75.8 | 49.3 |
| lean, 4 pairs, window 6 | 83.8 (427) | 50.5 | 34.9 | 126.1 (521) | 78.4 | 48.5 |
| lean, 2 pairs, window 6 | 82.6 (434) | 47.9 | 35.0 | 125.3 (524) | 76.6 | 48.4 |

NVFP4 at T = 2 / 4 / 16: 175.2 / 308.4 / 847.6 us (window-1 form) -> 150.9 / 268.6 / 764.1 (lean, 4 pairs, window 6). MXFP8 at
T = 16: 919.0 -> 851.1. Two pairs per batch loses at T = 16 (MXFP8 1108.8). Defaults after this window: lean, 4 pairs per
batch, windows NVFP4 6 / 6, MXFP8 gate_up 1, down 6 (`INTARSIA_HIP_INK_MOE_PRE`, `INTARSIA_HIP_INK_MOE_LEAN=0` = the first form).
T = 64 (still the shared-memory path, 512 pairs): 2137 us NVFP4 (window-1 form 2040), T = 256: 5533 us (6780).

Reading: gate_up gained 20 %, down nothing (342 GB/s). The down kernel holds 32.8 KB of LDS per block (group tables 16.4 KB
+ terms 16 KB), gate_up 16.4 KB: with 64 KB of LDS per workgroup processor that is one down block (8 waves) and three gate_up
blocks resident, whatever the registers allow. Next: LDS sized by the call (decode needs under 3 KB).

### Windows 3 and 4 (fleetq #11951, #11957; logs `raw/window3-11951.log`, `raw/window4-11957.log`): LDS by the call, windows, fixed cost

Tests (same command): 2 passed in each window. `ink_moe_rows_exact` now also checks the lean form in every LDS size class
(0 / 1 / 2) at windows 1 / 3 / 6 / 9 against the window-1 form, bitwise.

LDS sized by the call (class 0: 2.3 KB per down block instead of 32.8 KB): a small gain, not the explanation I gave after
window 2. NVFP4 down at T = 1: 34.9 -> 31.8 us (window 3), gate_up unchanged.

What did move down: all nine lane steps requested up front (window 9). Down, NVFP4, us: T=2 58.0 -> 49.0, T=4 106.5 -> 84.2,
T=16 298.8 -> 223.2 (546 GB/s). The same window costs gate_up at T = 1 (202 VGPRs: 57.5 against 48.9 us) and wins from T = 16
(408 against 456), and MXFP8 gate_up spills at 9 (110 against 75 us). Defaults since window 4: NVFP4 gate_up 6 below 8 rows,
9 from 8 rows, down 9; MXFP8 gate_up 1, down 9.

Window 4, the defaults (us per call; GB/s of 627):

| T | NVFP4 window-1 form | NVFP4 now | | MXFP8 window-1 form | MXFP8 now | |
|---|---|---|---|---|---|---|
| 1 | 100.1 | 81.6 (gate_up 48.9, down 31.0) | 439 GB/s, 70 % | 144.8 | 122.6 (74.7, 47.8) | 536 GB/s, 85 % |
| 2 | 175.4 | 139.5 | 503, 80 % | 253.9 | 216.8 | 571, 91 % |
| 4 | 308.7 | 244.2 | 533, 85 % | 417.7 | 353.0 | 589, 94 % |
| 16 | 844.5 | 623.5 | 587, 94 % | 918.7 | 810.0 | 571, 91 % |
| 64 (512 pairs, LDS class 2) | 2040 (window 1) | 1617 | | 1658 (window 1) | 2354 | |
| 256 (pair table) | 6780 (window 1) | 5347 | | 6125 (window 1) | 5590 | |

MXFP8 at T = 64 is slower than the window-1 form (2354 against 1658 us): 4 pairs per batch against 8 at 4 pairs per expert.
Not a decode size; prefill goes to S2.

Fixed cost of a call (window 4 probes, T = 1): both launches with every thread returning at once 5.9 us; returning after
the in-block grouping 11.8 us. So of the 81.6 us at T = 1: 57.1 us is the bandwidth line, 5.9 the two launches, 5.9 the
grouping (six barriers per block), about 12.7 everything else (the last round of blocks, the term barrier and token sum).

### Windows 5 and 6 (fleetq #11967, #11973; logs `raw/window5-11967.log`, `raw/window6-11973.log`): in-wave grouping, row-count rule

Tests: 2 passed in each window. `ink_moe_rows_exact` (T = 1, 3, 4, 5, 10, 16, 70; both formats) checks every size class
(0, 1, 2 and the in-wave class 4) x 4 / 8 pairs per batch x windows 1 / 3 / 6 / 9, and the default, bitwise against the
window-1 form.

Round 3, grouping inside the wave (no LDS tables, no barriers, up to 32 pairs): negative. The grouping probe at T = 1 went
11.8 -> 10.9 us, but the kernels need more registers (down at window 9: 198 VGPRs against 142) and a call got slower:
NVFP4 T = 1 88.3 us against 81.6, T = 4 266 against 244. Kept as an opt-in size class (`inkmoe::set_sz(4)`), not a default.

Window 5 sweep (every lean setting at T = 1 .. 64) gave the rule now in `moe_pre`, by pairs per expert (P = T x K over E):
- P <= 2 E (decode; T <= 32 at top-8 of 128): 4 pairs per batch, down window 9, NVFP4 gate_up window 6 (9 from 8 rows),
  MXFP8 gate_up window 1.
- P > 2 E: 8 pairs per batch, NVFP4 gate_up window 3.
- P >= 4 E, below the pair table: down takes the window-1 form (T = 64: 533 us against 649 lean).
- More than 512 pairs: the pair table, lean, 8 pairs per batch.

Window 6, the committed defaults against the window-1 form (`INTARSIA_HIP_INK_MOE_LEAN=0`, the code of `29b71ac67`), both
measured in the same run; us per call, 12 calls with different routings (3 at T = 2048), best of 7 (3 from T = 64).
NVFP4 128 experts, MXFP8 64 experts in the bench (so MXFP8 is twice as dense per row as the model's 128):

| Format | T | 29b71ac67 form | now | change | gate_up | down | GB/s |
|---|---|---|---|---|---|---|---|
| NVFP4 | 1 | 100.2 | 82.3 | -18 % | 49.6 | 31.0 | 435 |
| NVFP4 | 2 | 177.8 | 140.8 | -21 % | 91.1 | 50.6 | 498 |
| NVFP4 | 4 | 310.8 | 246.5 | -21 % | 160.7 | 86.0 | 529 |
| NVFP4 | 8 | 526.4 | 401.7 | -24 % | 269.1 | 136.3 | 571 |
| NVFP4 | 16 | 849.3 | 635.6 | -25 % | 422.4 | 217.6 | 576 |
| NVFP4 | 32 | 1183.5 | 969.2 | -18 % | 634.9 | 367.1 | 512 |
| NVFP4 | 64 | 1978.5 | 1421.2 | -28 % | 879.2 | 561.5 | 397 |
| NVFP4 | 128 | 3558.5 | 2837.2 | -20 % | 1836.5 | 1210.5 | 202 |
| NVFP4 | 256 | 6826.9 | 5303.6 | -22 % | 3317.4 | 2208.8 | 108 |
| NVFP4 | 2048 | 51197.6 | 38193.7 | -25 % | 24171.0 | 15595.2 | 15 |
| MXFP8 | 1 | 144.5 | 121.9 | -16 % | 74.8 | 47.2 | 539 |
| MXFP8 | 2 | 254.1 | 216.3 | -15 % | 134.5 | 82.0 | 573 |
| MXFP8 | 4 | 416.8 | 353.3 | -15 % | 222.8 | 130.4 | 589 |
| MXFP8 | 8 | 663.0 | 558.0 | -16 % | 361.5 | 196.1 | 607 |
| MXFP8 | 16 | 906.3 | 803.7 | -11 % | 526.0 | 282.2 | 570 |
| MXFP8 | 32 | 1107.7 | 951.2 | -14 % | 596.8 | 356.1 | 547 |
| MXFP8 | 64 | 2157.2 | 1408.3 | -35 % | 872.0 | 535.9 | 373 |
| MXFP8 | 128 | 3969.5 | 3218.0 | -19 % | 1879.7 | 1429.2 | 163 |
| MXFP8 | 256 | 7662.5 | 5597.2 | -27 % | 3381.0 | 2429.0 | 94 |
| MXFP8 | 2048 | 59689.6 | 41729.9 | -30 % | 25574.2 | 16810.3 | 13 |

No row count is slower. Prefill stays SIMT: 38.2 ms per NVFP4 call at T = 2048 (S2 is the fix).
Limits: one run per cell; MXFP8 measured at 64 experts, not 128; the rule's thresholds come from top-8 only.

### S2: tensor-core prefill (windows 7 and 8, fleetq #11982, #11987; logs `raw/window7-11982.log`, `raw/window8-11987.log`)

`ink_moe_wmma_kernel` in `ink_moe.hip`: the scheme of `moe_wmma.h` (`moe_wmma_kernel`: bf16 WMMA, weights dequantised into
the A fragments in registers, activation tile through LDS, 3 weight blocks in flight per lane) with this op's epilogues
(situ / silu / gelu, the routing weight), both formats (NVFP4 and MXFP8 through this file's fp8-conversion decode) and this
file's pair table. Ordinary names only, on the pair-table path (more than 512 pairs), H and I multiples of 64, and by default
from 16 pairs per expert on average; `INTARSIA_HIP_INK_MOE_WMMA` = 0 | 32 | 64. The `_rows` twins never take it.
Numerics: other f32 accumulation order than the decode kernels, so not their bits; within the registered tolerance.

Tests (`ink_moe_matches_cpu_reference ink_moe_rows_exact ink_moe_wmma_prefill`): 3 passed in both windows.
`ink_moe_wmma_prefill`: 3 shapes x (NVFP4 situ, MXFP8 situ / silu / gelu) x tiles 32 and 64: all pass against the cpu
reference with the admission's `judge` (worst rel_l2 1.3e-4, budget 1e-3 / 2e-3); the same bits run to run; tiles 32 and 64
give the same bits; the rows of a call reversed give the output reversed, bitwise.

Window 8, us per call (same bench as above; NVFP4 128 experts, MXFP8 64 experts):

| Format | T | 29b71ac67 form | decode kernels (lean) | WMMA tile 32 | WMMA tile 64 | default |
|---|---|---|---|---|---|---|
| NVFP4 | 128 | 3505 | 2882 | 5697 | 5080 | 2842 (decode kernels: 8 pairs per expert) |
| NVFP4 | 256 | 6860 | 5537 | 5810 | 5208 | 5209 |
| NVFP4 | 512 | 13079 | 10370 | 6192 | 5368 | 5373 |
| NVFP4 | 2048 | 50980 | 39597 | 6259 | 6179 | 6238 |
| MXFP8 | 128 | 3849 | 3226 | 2780 | 2487 | 2489 |
| MXFP8 | 256 | 7641 | 5800 | 2901 | 2582 | 2585 |
| MXFP8 | 512 | 14850 | 10863 | 3051 | 2732 | 2747 |
| MXFP8 | 2048 | 58958 | 42451 | 5596 | 4922 | 4909 |

NVFP4 at T = 2048: 51.0 ms (`29b71ac67`) -> 39.6 (decode round) -> 6.2 ms (8.2x and 6.3x). The default is never slower
than the decode kernels in this table.

Open in S2:
- A WMMA call costs about 5 ms for NVFP4 whatever the pair count (5.2 ms at 16 pairs per expert, 6.2 at 128): it is bound
  by reading and dequantising the weights (573 MB at about 110 GB/s), not by the tensor cores. The 64-pair kernels take
  217 - 240 VGPRs and 34 KB of LDS per gate_up block (one block per 64 KB). Not ported from the 35B kernels: the
  occupancy variant (up accumulators aliased onto the activation tile, 8 waves per SIMD), empty-fragment skip, the `r7`
  variants. Why the 32-pair tile (143 VGPRs, 17 KB) is slower than the 64-pair one at the same block count is not explained.
- The threshold (16 pairs per expert) comes from one NVFP4 point each side (8: slower, 16: 5 % faster) at 128 experts.
- MXFP8 weights with an E8M0 scale of 0 or 255, or products outside bf16 range, are not handled specially (admission's
  scales are 116 .. 119).
- Not run: admission (`intarsia kernels test --backend hip`), the sweep's T = 2048 real-shape cases against the cpu
  reference (`INK_MOE_BIG=1`), G2 and an end-to-end prefill measurement in the model.

### S2 evaluation in the model (window 9, fleetq #11990; log `raw/window9-11990-eval.log`, stage `raw/stage-eval.sh`)

Tree: `3ab1983fc` = this lane on `ink/hip-speed` `083e30f8f`; the tensor-core prefill is off by default (`kWmDefault = 0` in
`ink_moe.hip`; setting it to 1 is the flip, and `INTARSIA_HIP_INK_MOE_WMMA=1` is the same thing per run: 64-pair tiles from
16 pairs per expert). Inkling-2-Mini on one R9700 per stream, both cards used (card 0: 4k runs, admission; card 1: G2, 16k
runs), `intarsia generate --spec off --ignore-eos`, one run per cell.

Gates:

| Gate | Result |
|---|---|
| Default (WMMA off) greedy text md5, 4k prompt, 64 tokens | `a7c6a2acedecfac21c77486444810c38` = `ink/hip-speed` and main (their value is the one recorded in #11986, not rerun here) |
| Admission `kernels test --backend hip --op moe_experts`, WMMA on | `moe_experts_situ_bf16` 10 cases ADMITTED (max_abs 3.47e-3, rel_l2 2.08e-4), `_rows` 7 ADMITTED; the 35B kernels unchanged, ADMITTED |
| Admission `--op moe_experts_mxfp8`, WMMA on | `moe_experts_mxfp8_bf16` 19 ADMITTED (3.38e-3, 2.43e-4), `_rows` 14 ADMITTED |
| `ink_moe_matches_cpu_reference` with `INK_MOE_BIG=1`, WMMA on | passed; real shape T=2048: NVFP4 situ rel_l2 2.08e-4, MXFP8 situ 2.43e-4 |
| G2 short (n = 569), 21-way KL, need <= 3.0e-2 | WMMA 2.566e-3 PASS; flash 4.491e-3 PASS; both 4.491e-3 PASS (off, from #11986: 2.566e-3) |
| G2 long (n = 1569) | WMMA 3.286e-3 PASS; flash 4.521e-3 PASS; both 4.157e-3 PASS (off, from #11986: 3.251e-3) |

G2 short is the same number with and without WMMA: no call of that run reaches the rule (it needs chunks of 256 rows).

Prefill, tok/s (prefill ms):

| Prompt, chunk | off | WMMA | flash | both |
|---|---|---|---|---|
| 4k (4228 tokens), default chunk 256 | 613.7 (6889) | 783.6 (5396) | 722.9 (5848) | 956.9 (4418) |
| 4k, chunk 2048 | 651.9 (6485) | 1614.3 (2619) | 779.9 (5421) | 2687.4 (1573) |
| 16k (16915 tokens), chunk 2048 | 578.7 (29229) | 1207.3 (14010) | 786.9 (21496) | 2793.0 (6056) |

Decode is unchanged by either knob (69.7 - 70.5 tok/s after the 4k prompt).

What the table does not say:
- With WMMA on, the greedy text after the 4k prompt differs from the default within 64 tokens at chunk 256 (md5 `aa0a0ce6`,
  both: `2252566a`) and within 8 tokens at chunk 2048; flash alone gives the default's text in these runs, and all four
  settings give the same 8 tokens after the 16k prompt. Expected of a non-bitwise prefill, but it is a visible change.
- 16k at the default chunk was not measured (window time). The 16k prompt is the 4k prompt four times over.
- One run per cell; the 16k runs and G2 ran on card 1 while card 0 ran admission (its cpu reference loads the host).
- At chunk 256 a call is exactly at the rule's threshold (16 pairs per expert), where the per-call gain was 5 %; the
  end-to-end gain there is larger (614 -> 784 tok/s), which this lane has not explained.

### Knobs and integration notes (HANDOFF.md left the tree with the `ink/hip` merge)

The first two change no bit of the output:
- `INTARSIA_HIP_INK_MOE_PRE` = `v` or `gu:dn`, each 1 | 3 | 6 | 9: load window; default by format and pair density (`moe_pre`).
- `INTARSIA_HIP_INK_MOE_LEAN=0`: the first form (the code of `29b71ac67`).
- `INTARSIA_HIP_INK_MOE_WMMA` = 0 | 32 | 64: the tensor-core prefill (this one changes bits: other f32 order).
- Test / bench setters in `inkmoe.rs`: `set_probe`, `set_pre`, `set_lean`, `set_sz`, `set_nb`.
`docs/tuning.md` has rows for all three env knobs (`docs_env_vars` passes). Whether WMMA belongs in a fingerprint is the lead's call.

## 4. Next steps

1. User: the default flip (`kWmDefault = 1`), next to the attn lane's flash flip, from the table above.
2. S2 speed: the occupancy variant and fragment skip of `moe_wmma.h`, then the `r7` variants; the call is weight-read bound.
3. Not planned tonight: one launch for gate_up + down at decode (about 3 us of the 82 at T = 1).

## 5. S2 speed work not done (written after the evaluation; nothing here was built or measured)

Where S2 stands: a tensor-core call costs about 5 ms for NVFP4 whatever the pair count (5.2 ms at 16 pairs per expert, 6.2 at
128), i.e. 573 MB of weights at about 110 GB/s against 627 GB/s of bandwidth. It is bound by reading and dequantising the
weights, not by the tensor cores. Ordered by expected value:

1. Occupancy variant of `moe_wmma.h` (`kMwOcc`): the up accumulators reuse the activation tile's LDS, and the kernel is
   compiled for 8 waves per SIMD. Today the 64-pair gate_up block holds 34 KB of LDS (one block per 64 KB) and 217 - 240
   VGPRs; down holds 18 KB.
2. Fragment skip (`kMwSkip`): a tile's 16-pair fragments that hold no pair skip their LDS reads and WMMAs. Matters most
   near the threshold (16 pairs per expert in a 64-pair tile: three of four fragments are empty) and for the last tile.
3. The `r7` variants of `moe_wmma_r7.h` (`kMwTwo`, `kMwDeep`), which this lane has not read; the 35B defaults use them.
4. Explain why the 32-pair tile (143 VGPRs, 17 KB) is slower than the 64-pair one at the same block count (NVFP4 T=128:
   5.70 against 5.08 ms). Until that is understood, register and LDS counts are not a reliable guide here.
5. The threshold: 16 pairs per expert rests on two NVFP4 points at 128 experts (8: slower; 16: 5 % faster per call).
   Measure 10, 12, 24 and MXFP8 at 128 experts; items 1 and 2 will move it down.
6. End to end: at chunk 256 the model gained 28 % although the per-call gain at that density was 5 %; an op table with
   WMMA on would say where the rest comes from, and how much of prefill the experts still are at chunk 2048.
7. Launch size: the brief's note that launches above about 1 ms hit a dispatch stall on this card was not examined; the
   S2 launches are 2 - 4 ms each.
8. Edge values: MXFP8 E8M0 scales of 0 or 255 and products outside bf16 range are not handled specially.
9. Decode, unrelated to S2: one launch for gate_up + down (about 3 us of 82 at T = 1).
