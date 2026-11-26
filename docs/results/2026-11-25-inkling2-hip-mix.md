# Inkling-2-Mini HIP, package B (mix + projection), lane ink-hip-mix, 2026-11-25

Branch `ink/hip-mix`, merged `ink/hip` (skeleton `6c5ee5d67`, on origin/main `d8c3c7148`).
Ops: SconvRes 69, AttnResMix 67, LoopBlend 68, ConvMix 74, InkInProj 73 (+ `_rows` twin).

## Briefs received (kept against context loss)

- Shared brief: `/opt/code/intarsiaml/.worktrees/hip-swarm/COMMON.md` (ownership, correctness bar, tatooine rules: windows <= 20 min,
  <= 3 per lane, none after 06:00 EST, radiance up by 07:00, report once at pass or 06:30 EST).
- Lead update 22:00 EST: `git merge ink/hip` now; module in the exact shape `TABLE` / `OPS` / `all()` /
  `#[cfg(ik_hip)] pub(crate) mod launch { extern "C" {..} pub(crate) unsafe fn launch(kernel, &InkArgs) -> Option<c_int> }`;
  per-file `extern "C"`; do not touch `UNIMPLEMENTED` or shared tests; **skip MoeRoute** (lead wrote it); correctness forms first
  (`ink_row_kernel` / blend, `in_proj_gemv_kernel` bf16 arm); no `.row_table()`, no `.act_quant()`.

- Coordinator update ~22:15 EST (from the ink-hip lead): tatooine ready for integration; when the ops pass their unit window, send "ready to merge at <commit>"
  (forwarded to the lead). Checkpoint (read-only): `/data/models/intarsia/Inkling-2-Mini` (in the dev image `/models/Inkling-2-Mini`). Window scripts copied from
  `ink/hip` `docs/results/2026-11-25-inkling2-hip/raw/{window,run}.sh` into this lane's `raw/` (own repo dir, own container `ink-hip-mix-run`, own target volume
  `intarsia-cargo-target-ink-hip-mix`). moe_route is the lead's and passed (#11924); keep skipping it. The lead merges each lane after its own unit window, then runs
  admission + tiny G1 + real-model G2 in one window. Window rules unchanged (fleetq `--takes-down radiance`, <= 20 min, <= 3 windows, none after 06:00).

- **Speed-round brief (coordinator, after the first landing; start from `ink/hip` head, which has `444ead518` and all three lanes; the whole model passes G2 there).**
  Lead's GPU profile #11945 / #11947: one decode token 16.0 ms against an 8.76 ms floor; idle time between the 712 dispatches is 2.56 ms (about 3.6 us each), so
  fewer dispatches pay directly. Three items in order:
  (a) fused `InkRoute` (op 76): router GEMV + top-k in one kernel (today `moe_route` 48 calls x 8 us = 0.38 ms; fusing removes 48 dispatches, about 0.4 ms). Op 76 needs
  one line from the lead in `model/inkling2.rs` (hip only): do NOT edit that file; tell the coordinator when op 76 is admitted.
  (b) blend: ctl GEMV 0.31 + gate 0.22 + norm 0.18 = 0.71 ms over 144 dispatches against a 0.18 ms floor: fewer launches (about -96 dispatches, about 0.5 ms).
  (c) ConvMix / AttnResMix row kernel takes 11.2 us for a 2304-wide row (124 calls = 1.39 ms, almost no data): check block size, occupancy, serial reductions; target
  about 0.7 ms saved.
  Rules: keep each op's Exactness class; admission vs cpu-ref must pass; rows-exact stays bitwise where promised; bitwise against the current forms preferred (say so
  if bits change; G2 must still pass). 2 windows left, <= 20 min each, none after 06:00, radiance back up before release; own `INTARSIA_TARGET_VOL` and container name;
  tatooine `/` is 88% full: clean the own build volume between windows if tight. Report each landing as "ready to merge at <commit>" with us before / after per kernel.

- **Window rule change (coordinator, during the speed round):** the count cap is replaced by a time budget. Each window keeps radiance down 5 min or less; this lane's
  total radiance-down time from then on is 20 min or less; none after 06:00; radiance must answer `/v1/models` before release. Log each window's down time here.
  PR #516 (`ink/hip` -> main, the correctness form) stays as it is until it merges: do not merge into `ink/hip`; send speed heads as "ready to merge at <commit>" (the
  lead integrates them in a follow-up branch).
  Down-time log (speed round): 22:13:00-22:14:13, 22:19:51-22:21:03, 22:23:51-22:24:06, 22:29:11-22:30:23, 22:32:39-22:33:50, 22:36:04-22:37:16 (each radiance answered within 15 s of the restart); about 7 min total, each window under 2 min.

## Status (2026-11-25 21:45 EST)

All five ops ported and ADMITTED on the R9700 in the first unit window; rows-exact (bitwise) tests pass; bench below. Branch `ink/hip-mix`.

Files: `kernels/hip/ink_mix.hip` (row kernel + blend: port of `ink_row_kernel`, `ink_ring_kernel`, `ink_blend_gate_kernel`,
`ink_blend_norm_kernel`; the controller h projection is a new plain wave GEMV `ink_ctl_gemv_kernel`, not the CUDA mma split-K),
`kernels/hip/ink_proj.hip` (`in_proj_gemv_kernel<M>` for T <= 16; T > 16 = three `ik_matmul_bf16` calls), `kernels/hip/ik_hip_abi_inkmix.h`,
`crates/intarsia-backend-hip/src/inkmix.rs`, `crates/intarsia-backend-hip/tests/ink_mix.rs`, `HANDOFF.md`.
Launchers: `ink_in_proj_bf16_gemv`, `_gemv_rows` (twin), `ink_in_proj_bf16_matmul` (T > 16), `sconv_res_bf16`, `attn_res_mix_bf16`,
`conv_mix_bf16`, `loop_blend_bf16`. Not built: e4m3 / P12 InkInProj, row-table forms (as briefed).

## Tests (window #11927, radiance down 21:42:36 - 21:43:36, up again after 15 s; log `raw/window1-unit-bench.log`)

`cargo test --release -p intarsia-backend-hip --test ink_mix -- --nocapture --test-threads=1` (dev image, card 0, `INTARSIA_REQUIRE_GPU=1`): 4 passed.
- `ink_mix_ops_match_the_reference` (admission vs `ink_ref`, the CUDA tolerances): ink_in_proj_bf16_gemv 24 cases ADMITTED (worst max_abs 3.8e-3,
  rel_l2 7.3e-5), `_gemv_rows` 24 ADMITTED, `ink_in_proj_bf16_matmul` 5 ADMITTED (registered `Exactness::Fast`) (3.1e-3 / 4.7e-5), conv_mix_bf16 532 ADMITTED (4.5e-3 / 1.2e-3),
  attn_res_mix_bf16 93 ADMITTED (2.0e-3 / 4.9e-5; queries Q in {1, 3, 4}: the Q = 48 case is not in the quick sweep that ran, and is refused by the registration `.when` (n <= 4) as in CUDA; see HANDOFF), sconv_res_bf16 64 ADMITTED (1.9e-3 / 1.2e-5),
  loop_blend_bf16 27 ADMITTED (3.2e-3 / 2.6e-4). 0 failing cases.
- `ink_in_proj_rows_exact`, `ink_conv_mix_rows_exact`, `ink_attn_res_mix_and_blend_rows_exact`: BITWISE rows-exact (T-row call = T one-row calls, ring
  carried, final ring equal) at D 2304 and 64, all epilogue / mix / blend variants, T up to 300 for the mixes.
- Host: `INTARSIA_NO_HIP=1 cargo test -p intarsia-backend-hip --lib` 31 passed (ABI header test, names unique, op coverage).

## Bench (same window; `cargo test ... ink_mix_bench -- --ignored --nocapture`; launches back to back, host launch overhead included)

| kernel / shape | us | vs roofline |
|---|---|---|
| in_proj gemv S1 T1 (39.9 MB weights) | 66.9 | 597 GB/s = 95% of 627 |
| in_proj gemv S3 T1 | 65.7 | 608 GB/s = 97% |
| in_proj gemv S1 T4 | 70.8 | 564 GB/s = 90% |
| in_proj gemv S1 T16 / S3 T16 | 131 / 139 | 304 / 286 GB/s (SIMT FMA bound, 4.9 TFLOP/s) |
| in_proj matmul (WMMA) T64 / T512 | 89.8 / 295.7 | 445 GB/s; 28.5 / 69.1 TFLOP/s |
| sconv_res T1 / T16 / T256 / T1024 | 6.3 / 7.0 / 21.5 / 66.0 | activation traffic 274-357 GB/s at T >= 256 |
| attn_res_mix N6 Q1 T1 / T16 / T256 / T1024 | 11.7 / 12.6 / 46.0 / 167 | 180-198 GB/s at T >= 256 |
| loop_blend S3 G64 T1 / T4 / T16 / T256 / T1024 | 29.2 / 41.8 / 65.9 / 217 / 868 | the slow one (below) |

Notes: decode InkInProj is on the byte floor. LoopBlend is the speed candidate: 4 launches per call; its prefill cost is 0.85 us per row
(868 us per 1024 rows; the h wave-GEMV re-reads the 3.5 MB controller weights per group of 8 rows, the WMMA form named in the port map would fix that);
T16 verify rows pay 66 us. Not tuned tonight beyond the correctness form.

## Not verified / open

- After the window I changed one registration (`ink_in_proj_bf16_matmul` to `Exactness::Fast`, as the port map says for the T > 16 form); host tests (31) and the check-only
  build pass, the admission was not re-run on the GPU (the lead's integration window will).
- Not benched: ConvMix (with / without blend), the in-proj bench reuses one 39.9 MB weight buffer (may be partly cache-warm; rotate copies before quoting the
  byte floor), the rows-exact test covers in-proj T in {1, 2, 3, 5, 8, 9, 16} (not every `in_proj_gemv_kernel<M>` instantiation).
- tatooine: target volume `intarsia-cargo-target-ink-hip-mix` (about 1 GB+) and `~/intarsiaml-ink-hip-mix` (1.9 GB) are this lane's; `/` is 88% full: remove both when done.

- Not run through the model (G1 / G2: the lead's window). `-ffp-contract` is left at hipcc's default (fast); rows-exactness holds because every
  row runs the same compiled code, and the admission tolerances pass; no kernel is on `NO_FP_CONTRACT_SOURCES`.
- `rsqrtf` in the conv scale is the hardware approximation (CUDA's too); admitted within tolerance.

## Speed round (2026-11-25 22:10-22:40 EST), logs `raw/window2..6-*.log`

Same-window A/B of every form in the bench (`ink_mix_bench`); the first forms stay in the code (mask 0) and every test checks every form against them to the bit.

Shipped (default mask 3), all BITWISE equal to the first forms (`ink_*_rows_exact` tests run each call at mask 0, 15 and the default and compare; they passed in window 6):

| item | change | before -> after (us per launch, 1-2 decode rows) |
|---|---|---|
| (c) ConvMix / AttnResMix / SconvRes row kernel | `ink_row_kernel<NE, true>`: every load of the row (conv taps and ring rows, res / addend, sources, queries, norm weights) issued first and unconditionally (clamped addresses, value selected after), `block_fin` unrolled; used for launches of <= 16 rows | attn_res_mix 13.1 -> 10.2; conv_mix (no blend) 14.8 -> 10.3; sconv_res 6.7 -> 5.6 |
| (b) LoopBlend / ConvMix blend | fused blend launches for chunks of <= 2 rows: row kernel (np = S * G / 8 blocks per row: pair in LDS + controller GEMV) -> gate kernel whose last block per (stream, row group) does the norm (device-scope ticket): 2 launches instead of 4 | loop_blend S3 G64 T=1 28.1 -> 20.7, T=2 31.4 -> 25.4; conv_mix + blend T=1 37.9 -> 25.4, T=2 40.7 -> 35.0 |

Why (c) was 11 us: a `wall_clock64` probe of block 0 (`INK_PROBE` build, `tests/ink_mix_probe.rs`) showed the conv phase alone at 3.3 us and entry at 0.8 us of the 10-11 us: the loads sat under lane conditions (`row ? load : 0`, `n < N ? load : 0`), which the compiler serialises (wait before the merge), a memory latency (~170 ns x several, plus the ring modulo) per tap. With unconditional loads the conv phase is 0.12 us (loads overlap the entry latency). The rest (sums 0.9, fin 0.3, mix 1.4, stores 0.4-0.6 us) is the reduction chain.
Over about 124 ConvMix / AttnResMix calls per decode token this is 124 x 3.5-4.5 us = 0.43-0.56 us ms (lead's target 0.7 ms; not reached), and the fused blend saves ~7-12 us x 48-96 calls = about 0.4-0.6 ms (their estimate 0.5 ms), if the model's blend calls are 1-2 rows (decode; T = 4 verify rows are not fused: no gain measured).

Measured and NOT shipped:
- The fast row kernel above 16 rows: 160-240 VGPRs cut occupancy, attn_res_mix T=256 47 -> 57 us, T=1024 155 -> 193 us. Gated off (mask bit 8 only in tests).
- Fusing the blend at 4 and 16 rows: no gain (39.0 vs 38.7, 58.5 vs 58.2): the controller GEMV is then weight-bandwidth bound.
- Gate kernel with early weight loads (mask bit 4): no gain, off by default.
- (a) fused `InkRoute` (op 76, `kernels/hip/ink_route_fused.hip`, `ink_route_bf16`): implemented and ADMITTED, BITWISE equal to `logits_bf16_gemv` + `moe_route` (T 1..4, 3 shapes, 4 score functions, with / without logits, row = one-row call; passes), but NOT FASTER: T=1 15.9 us fused against 15.7 us for the pair (4.8 + 10.9, each measured back to back), T=4 19.4 against 15.9, T=16 59 against 26.5. The last-block tail (ticket, then `moe_route`'s serial top-k on 1 wave per row) costs as much as the dispatch it saves; the pair's `moe_route` is itself only 10 us because the GEMV is already separate. Registered for T <= 4 so the model can select it with `INTARSIA_INK_ROUTE_FUSED=1` if the lead wants to A/B it end to end, where the saved dispatch gap (~3.6 us on the 712-dispatch profile) may show: I could not see it in a microbench. Do not default it on from this data.

Unchanged ceilings: InkInProj decode 65.7-66.9 us (95-97% of 627 GB/s); LoopBlend prefill rows (0.8 us / row, 824 us per 1024 rows).

ConvMix + blend at 4 / 16 rows and LoopBlend at 4+ rows: unchanged (39 / 58 us). The remaining LoopBlend decode cost: 3 launches (row kernel + gate; ~20 us) against a 3.5 MB controller-weight floor of 5.6 us.
