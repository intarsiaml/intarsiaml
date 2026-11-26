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

(pending: bench, probe)
