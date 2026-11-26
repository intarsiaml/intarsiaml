# Inkling-2-Mini on HIP: cheaper verify rows (lane ink-hip-vrow, 2026-11-26)

Branch `ink/hip-vrow` (worktree `.worktrees/ink-hip-vrow`), base main `12c6c78ac` (#531). Follow-up of lane ink-hip-verify
(`docs/results/2026-11-26-inkling2-hip-verify.md`, PR #534): its per-op trace of verify passes by row count put the row-dependent
excess with a flat byte floor in the loop blend (+0.85 ms a pass at 4 rows: `kernels/hip/ink_mix.hip` `kFuseRowsMax = 2`, so 3+ row
chunks leave the fused launches) and the verify attention (+1.20 ms at 4 rows in op 75). Shared rules: `.worktrees/hip-swarm/COMMON-1126.md`.
Tatooine side: the lane folder and cargo volume of ink-hip-verify (`~/intarsiaml-ink-hip-verify`, this tree synced to
`.scratch/vrow-src`).

## Briefs received

1. 2026-11-26 ~15:45 EST (coordinator, brief 2 part 2 to lane ink-hip-verify), summarised: lane ink-hip-batch is closed, its files
   are this lane's. Wait for #531, branch from main as `ink/hip-vrow`. Lever 1, verify attention: the 2..k+1 rows of one sequence as one
   call instead of one decode call per row, target the +0.37 ms / row (#529's row-table attention is bitwise the solo decode call per
   row). Lever 2, loop blend: keep 3..5-row passes in the fused launch (raise `kFuseRowsMax` or a fused form up to 5 rows), target the
   +0.85 ms and 96 extra dispatches. Gates: bitwise against today's per-row path at rows 2..5 (verify probe worst diff 0, no flips);
   greedy md5 = `--spec off` and = main at short / 4k, 64 / 256; rows-exact tests and tests/ink_rows.rs pass; G2 unchanged; 4 + 4 ABBA
   of spec-default decode vs main. Default-on if bitwise, no length slower and >= +2 % at >= 1 length with CI lower bound > 1.0; else
   opt-in with the numbers. Radiance for part 2: 15 min in total. Own PR, do not merge.
2. 2026-11-26 ~15:55 EST (coordinator): #531 merged (`12c6c78ac`), #532 (`8f7356a9d`).

## 0. What lane ink-hip-verify's trace says (1021-id context, kernel time per pass, ms)

| class | 1 row | 2 | 3 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| loop blend (`ink_blend_gate`; from 3 rows + `ink_ctl_gemv_kernel` + `ink_blend_norm`) | 0.36 | 0.46 | 1.04 | 1.21 | 1.56 | 1.90 |
| verify attention (`ink_attn_dec_k` + prologue / s1 / comb) | 0.96 | 1.40 | 1.81 | 2.17 | 2.96 | 3.60 |

Op 75 is one dispatch per visit for all verify rows (grid z = rows); each row's blocks redo that row's prologue jobs and read the split's
K / V. The 1k and 4k probes cost about the same per row (4 rows: 22.36 ms at 1k, 21.57 at 4k), so the K / V bytes are not obviously
the per-row cost; window 1 below traces the 4k probe to check.

## 1. Lever 2: the blend's fused launches at more rows

Two limits behind main's `kFuseRowsMax = 2`, now separate runtime values (`INTARSIA_HIP_INK_BLEND_FUSE=r:n`, test setter
`inkmix::set_blend_fuse`): `r` rows for the fused row-kernel launch (pair and controller rows in the row kernel, np = S G / 8 = 24 blocks
per row, each recomputing the row; no `ink_ctl_gemv` launch) and `n` rows for the gate kernel's fused norm (its last block per stream
forms the norm; no `ink_blend_norm` launch). The default stays 2:2 until measured. Bitwise by construction (the same chains, tested):
`tests/ink_mix.rs` `run_r` now also runs every LoopBlend / ConvMix / AttnResMix call at 256:256, 256:0, 0:256 and 0:0 against the
first forms (the rows-exact tests cover LoopBlend at R = 2, 5, 16, 300 and ConvMix with the blend at T = 2, 3, 7, 16, 20).

Lever 1 has a bitwise switch of the same kind: op 75 already has two paths with the pair's bits, the one-dispatch fused kernel
(`ink_attn_dec_k`: one block per (kv head, split, row), every block forms its row's q / r and its own rows' conv'd k / v, so a row's
prologue work is repeated in every split) and the split path (`ink_prologue_k` once per (kv head, row), then stage 1 and the combine).
Main picks by keys only (`INTARSIA_INK_DEC_FUSED_KEYS`, fused up to 2048 keys). New: a rows rule `INTARSIA_HIP_INK_DEC_FUSED_ROWS`
(fused only for calls of at most that many rows; setter `inkattn::set_dec_fused_rows`); `tests/ink_attn.rs` `rows_exact_real_shapes`
now runs op 75 at rule 0 (split path always) and 16 (fused always, main) as well as the default, R-row and 1-row calls, host and device
position, all against the unfused pair.

## 2. Window 7: bench and probe (both rules swept)

fleetq #12192, radiance down 16:05:24-16:08:23 EST (**2:59**), both cards (card 1 ran PR #534's k5 gates in the same window, logged in
that lane's doc). Binary `vrow1` (CPU job #12190, md5 `739d1f77…`) = main `12c6c78ac` + the runtime limits at main's defaults (2:2, op
75 rows rule absent in this build: it came after; `INTARSIA_INK_DEC_FUSED_KEYS=0` forces the split path instead). Logs
`raw/window7-12192.log`, `raw/w7/`.

**Bench** (`ink_blend_fuse_bench`, us per call, best of 3 x 300; `r:n` = fused row kernel up to r rows, fused norm up to n rows):

| T | LoopBlend S3 G64: 2:2 (main) | 0:0 | 2:16 | 16:0 | 16:16 | ConvMix + blend S3: 2:2 (main) | 16:0 | 16:16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.10 | 27.18 | 17.90 | 19.39 | 18.14 | 21.15 | 24.73 | 21.60 |
| 2 | 22.04 | 30.64 | 19.79 | 20.50 | 20.73 | 28.89 | 29.68 | 28.94 |
| 3 | 36.13 | 34.08 | 35.62 | **22.12** | 23.90 | 39.17 | **34.94** | 35.82 |
| 4 | 39.28 | 37.96 | 39.87 | **24.44** | 26.82 | 41.66 | **36.25** | 39.69 |
| 5 | 41.79 | 41.68 | 46.18 | **24.73** | 30.19 | 46.36 | **42.34** | 47.23 |
| 6 | 45.15 | 45.59 | 51.93 | **27.32** | 34.07 | 49.52 | 49.04 | 55.81 |
| 8 | 53.30 | 53.27 | 62.68 | **30.00** | 40.43 | 57.64 | 57.20 | 66.73 |
| 16 | 56.88 | 56.95 | 113.94 | 44.61 | 101.82 | 61.29 | 96.81 | 153.23 |

The fused row kernel wins from 3 rows to 8 (LoopBlend -14 to -23 us, ConvMix + blend -4 to -5 us at 3-5 rows, even at 6-8) and loses
ConvMix at 16 rows; the fused norm helps at 1-2 rows only (its last block's serial norm grows with rows). Rule: **8:2**.

**Model** (verify cost probe at the real 1021-id prompt, greedy continuation as the rows, graphs on, median of 12, ms):

| rules | decode | 2 rows | 3 | 4 | 5 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| main (2:2, op 75 fused) | 13.36 | 16.58 | 20.01 | 22.31 | 24.68 | 26.98 | 31.55 |
| main again (end of the card's run) | 13.38 | 16.63 | 20.07 | 22.40 | 24.76 | 27.05 | 31.70 |
| blend 16:16 | 13.37 | 16.59 | 19.71 | 21.94 | 24.45 | 26.81 | 31.36 |
| blend 2:16 | 13.34 | 16.58 | 20.10 | 22.48 | 24.93 | 27.27 | 31.90 |
| op 75 split path always (`FUSED_KEYS=0`), blend 2:2 | 13.61 | 16.41 | 19.82 | 21.94 | 24.04 | 26.22 | 30.57 |
| op 75 split path always, blend 16:16 | 13.64 | 16.47 | 19.54 | 21.58 | 23.79 | 26.04 | 30.44 |

The split path costs +0.25 ms at one row and saves 0.2 / 0.2 / 0.4 / 0.7 / 0.8 / 1.1 ms at 2 / 3 / 4 / 5 / 6 / 8 rows: op 75's rows
rule is **1** (decode fused, verify split). Together with the blend at 16:16 (not yet 8:2, which the bench prefers) a 4-row pass is
0.77 ms (3.4 %) cheaper, a 5-row pass 0.93 ms (3.8 %). Bitwise: the probe's bitwise sections at blend 16:16 (rows 2..8 at 17 / 1024
/ 4000 + 16 greedy windows) 233 rows, worst diff 0, no flips; `ink_mix` rows-exact tests (every limit against the first forms) pass.

**Lever 1, the 4k trace** (card 1, `verify2` = main's kernels, the timing probe at 4108 ids under rocprofv3, `raw/w7/rowtrace4k.txt`):
attention 1.15 / 1.65 / 2.11 / 2.53 / 3.07 / 3.44 / 4.19 ms at 1 / 2 / 3 / 4 / 5 / 6 / 8 rows, +0.46 ms a row against +0.40 at 1k: the
per-row cost hardly grows with the context (4x the keys), so it is the per-row work (prologue jobs per split block, the bias, the
merge, the grid), not the K / V bytes. A multi-row op 75 form that shares the K / V reads would not remove it; the split path removes
the per-split prologue repeats, which is what the rows rule takes. No new op 75 form was written (the advisor's stop rule).
