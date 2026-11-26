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

## 1. Status

| Step | State |
|---|---|
| Correctness form (SIMT union-grouped port of `kernels/cuda/ink_moe.cu`), both ops, all four launchers | done, passes on the R9700 (window 1, fleetq #11935) |
| Rows-exact at every T (shared-memory path, pair-table path, every load window) | done, bitwise (window 1) |
| Decode bench vs 627 GB/s | NVFP4 T=1 100.8 us = 355 GB/s (57 %); MXFP8 T=1 135.1 us = 486 GB/s (78 %); section 3 |
| Decode speed round 1 (lean step, window 2) | NVFP4 T=1 100.8 -> 83.8 us (427 GB/s, 68 %), MXFP8 135.1 -> 126.1 us (521 GB/s, 83 %); bitwise |
| Decode speed round 2 (LDS sized by the call) | open |
| Prefill | correct (pair-table SIMT path); slow: T=256 6.8 ms / call; WMMA is speed package S2, not started |

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

## 4. Next steps

1. Decode speed round: VGPR count per kernel (asm), word-at-a-time decode + fma (smaller live set), per-kernel load window,
   the grouping moved out of the blocks (one table launch) if it pays; one window to compare, bitwise against this form.
2. S2 (WMMA prefill with a SiTU epilogue, I = 1152) is not started.
