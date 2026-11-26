# Inkling-2-Mini on HIP: cheaper verify rows (lane ink-hip-vrow, 2026-11-26)

Branch `ink/hip-vrow` (worktree `.worktrees/ink-hip-vrow`), base main `12c6c78ac` (#531). Follow-up of lane ink-hip-verify
(`docs/results/2026-11-26-inkling2-hip-verify.md`, PR #534): its per-op trace of verify passes by row count put the row-dependent
excess with a flat byte floor in the loop blend (+0.85 ms a pass at 4 rows: `kernels/hip/ink_mix.hip` `kFuseRowsMax = 2`, so 3+ row
chunks leave the fused launches) and the verify attention (+1.20 ms at 4 rows in op 75). Shared rules: `.worktrees/hip-swarm/COMMON-1126.md`.
Tatooine side: the lane folder and cargo volume of ink-hip-verify (`~/intarsiaml-ink-hip-verify`, this tree synced to
`.scratch/vrow-src`).

## Result

| Inkling-2-Mini, one R9700, spec-default decode (main's recipe since #534, k5; 4 + 4 processes ABBA, window 10, merged binary `vrow3`) | main rules | this branch | ratio, 95 % CI |
|---|---:|---:|---|
| short (12 golden prompts) | 104.46 tok/s | 104.68 | 1.003 [1.002-1.004] |
| 1k | 91.12 | 93.00 | **1.022** [1.016-1.028] |
| 4k | 124.47 | 127.12 | **1.023** [1.019-1.027] |
| 16k | 96.51 | 98.56 | **1.023** [1.019-1.026] |
| at k3 (window 9, `vrow2`, base `12c6c78ac`) | 103.66 / 90.05 / 114.94 / 91.75 | 104.00 / 92.16 / 117.81 / 93.78 | 1.003 / **1.023** / **1.025** / **1.022** |
| verify pass at 1k, 4 / 5 rows (probe) | 22.38 / 24.77 ms | 21.49 / 23.63 ms | -4.0 % / -4.6 % |

- **Change, bitwise:**
  - The loop blend keeps the fused row-kernel launch up to 8 rows (was 2); the fused norm stays at 2 rows.
  - Op 75 takes its fused kernel for 1-row calls only; verify passes take its split path.
  - Both are existing forms with the same bits, now chosen by rows. Env knobs `INTARSIA_HIP_INK_BLEND_FUSE`, `INTARSIA_HIP_INK_DEC_FUSED_ROWS`; revert `=2` / `=16`.
- **Gates** (section 3, on `vrow2`): verify probe rows 2..8 worst diff 0; md5 = main x 8; hip tests incl. `ink_rows` pass; admission 0 FAIL;
  G2 identical; tiny suites as on main. On the merged binary `vrow3` (window 10): md5 = main at short / 4k, 64 tokens, spec default (k5)
  and `--spec off` (4 runs).
- **Lever 1 as asked** (a multi-row op 75 form sharing the K / V reads) was not built: the 4k trace shows the per-row cost is per-row work, not K / V
  bytes (section 2).

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
the per-split prologue repeats, which is what the rows rule takes. No new op 75 form was written: it would need per-row split boundaries with each row's own reduction order, beyond this budget.

## 3. The change and window 8: its gates

Defaults (commit after window 7): `INTARSIA_HIP_INK_BLEND_FUSE` 8:2 (`kFuseRowsMax` 2 -> 8; `kFnormRowsMax` 2) and
`INTARSIA_HIP_INK_DEC_FUSED_ROWS` 1 (`kDecFusedRowsDefault`; main's rule = 16). Revert: `INTARSIA_HIP_INK_BLEND_FUSE=2
INTARSIA_HIP_INK_DEC_FUSED_ROWS=16`. The 1-row decode step runs the same launches as main (both rules keep 1-row calls where they
were); so do prefill chunks above 16 rows.

fleetq #12197, radiance down 16:13:27-16:16:34 EST (**3:07**), both cards. Binary `vrow2` (CPU job #12196, md5 `96aee014…`) and its
test targets. Logs `raw/window8-12197.log`, `raw/w8/`.

1. **Verify probe**, bitwise sections at the defaults, rows 2..8 at 17 / 1024 / 4000 ids + 16 greedy windows: 233 rows, 0 windows
   with a diff, 0 argmax flips, **worst |logit diff| 0** (graphs on).
2. **Greedy md5**, `--spec off` and the spec default, `short-0.txt` / `4k-0.txt` x 64 / 256: `04cdb28a6849…`, `9a27c21b6fdc…`,
   `2252566a5c54…`, `9675bbb95ada…`, **all eight equal to main's** (main2 = `80719483a`, lane ink-hip-verify window 4; `12c6c78ac`
   changes no solo kernel) and `--spec off` = the spec default.
3. **hip tests**: `ink_mix` (rows-exact LoopBlend / ConvMix / AttnResMix at every blend limit + the reference) 3 / 3;
   `ink_attn` `rows_exact_real_shapes` 63 cases bitwise (op 75 at rules 0, 16 and the default); `ink_rows` 2 / 2. **Admission**
   (`kernels test --backend hip`): `loop_blend` 27 / 0 FAIL, `conv_mix` 532 / 0, `attn_res_mix` 93 / 0, `ink_attn_decode` 70 / 0, all
   ADMITTED. hip tiny suites: `inkling2_dspark` 4 / 4, `inkling2_dspark_cut` 1 / 1, `inkling2_verify_softcap` 1 / 1, `inkling2` 14 / 17
   with main's three failures (lane ink-hip-verify window 5).
4. **G2**: short 21-way KL 4.491e-3 PASS, long 4.083e-3 PASS (1 top-1 miss), identical to main.
5. Verify cost probe at 1k, same binary, main's rules / the defaults (ms): decode 13.37 / 13.39; 2 rows 16.62 / 16.46; 3: 20.10 /
   19.49; 4: 22.38 / **21.49 (-4.0 %)**; 5: 24.77 / **23.63 (-4.6 %)**; 6: 27.02 / 25.82; 8: 31.77 / 30.27.

## 4. Window 9: speed (4 + 4 processes, same binary)

fleetq #12198, radiance down 16:17:04-16:21:00 EST (**3:56**), both cards. Binary `vrow2`, arm **new** = the defaults, arm **main** =
`INTARSIA_HIP_INK_BLEND_FUSE=2 INTARSIA_HIP_INK_DEC_FUSED_ROWS=16` (main `12c6c78ac`'s launch sequence; the same binary, so only the
rules differ). Spec default, no `--spec` and no TAU / CUT env (main's recipe: DSpark k3, cut 0.5). speed.json = a warm-up + the 12 short
golden prompts + 1021 / 4108 / 16455 ids, 128 tokens, greedy, EOS honoured. Order per card ABBA (card 0 main new new main, card 1 new
main main new). Ratio per (prompt, process) to the main process of the same card and rep, geometric mean per class, 95 % CI: t over the
4 processes [bootstrap over processes x prompts] (`raw/w9/speed-k3.txt`). Then one process per arm per card at k5 (`--recipe` copy with
`mtp_default_k = 5`, as PR #534 ships; `raw/w9/speed-k5.txt`).

| spec-default decode | short (12) | 1k | 4k | 16k |
|---|---:|---:|---:|---:|
| main rules, k3 (tok/s, median) | 103.66 | 90.05 | 114.94 | 91.75 |
| new rules, k3 | 104.00 | 92.16 | 117.81 | 93.78 |
| **new / main, k3 (4 + 4)** | **1.003** [1.002-1.005] | **1.023** [1.022-1.025] | **1.025** [1.024-1.026] | **1.022** [1.020-1.024] |
| new / main, k5 (2 + 2) | 1.003 [0.991-1.016] | 1.021 [1.020-1.022] | 1.023 [0.998-1.048] | 1.023 [0.995-1.052] |

The main-rules arm reproduces main: lane ink-hip-verify's window 6 ran the real main binary (`80719483a`, k3, the same protocol) at
103.67 / 90.09 / 115.00 / 91.82 tok/s, within 0.1 % of this arm's 103.66 / 90.05 / 114.94 / 91.75. Ids identical in all 16 processes;
the counters are equal arm for arm (the rules change no output). The bar for default-on (brief 1:
bitwise, no length slower, >= +2 % at >= 1 length with a CI lower bound above 1.0) is met at 1k, 4k and 16k, short is +0.3 % (lower
bound 1.002). Plain decode (`--spec off`) runs main's launches (1-row calls keep both fused forms): probe decode step 13.37 / 13.39 ms,
the md5 runs' single-run plain decode at 256 tokens 76.84 (short) / 74.02 (4k) tok/s against main2's 76.86 / 74.04 (lane
ink-hip-verify window 4); no plain ABBA was run.

**Batched decode** (`serve --max-batch`, lane ink-hip-batch's row-table ConvMix, `conv_mix_bf16_rows`): `run_blend` is shared, so
3..8 sequences now take the fused row kernel's RT form as well. Bitwise: `tests/ink_rows.rs` `conv_mix_rows_bitwise_solo` runs the
blend (S 1 and 3) at 1 / 2 / 3 / 5 / 16 sequences against the solo T = 1 calls and passed on `vrow2` (window 8). Its speed was not
measured. The row-table attention (`ink_attn_decode_bf16_rows`) has its own launcher and keeps the fused kernel.

## 4b. Window 10: the merged binary at main's k5 (4 + 4 processes, same binary)

fleetq #12217, radiance down 16:46:50-16:49:50 EST (**3:00**), both cards. Binary `vrow3` = this branch merged with main `b7450223e`
(#534: built-in HIP `mtp_default_k = 5`; every process logs "up to 5 drafts"). Same protocol as window 9: arm **new** = the defaults,
arm **main** = the revert env, spec default with no `--spec` / `--recipe` / TAU / CUT env, speed.json, 128 tokens, ABBA per card
(`raw/w10/speed-k5.txt`, `raw/w10/c0.log`, `raw/w10/c1.log`).

| spec-default decode, k5 | short (12) | 1k | 4k | 16k |
|---|---:|---:|---:|---:|
| main rules (tok/s, median of 4) | 104.46 | 91.12 | 124.47 | 96.51 |
| new rules | 104.68 | 93.00 | 127.12 | 98.56 |
| **new / main (4 + 4)** | **1.003** [1.002-1.004] | **1.022** [1.016-1.028] | **1.023** [1.019-1.027] | **1.023** [1.019-1.026] |

Ids identical in all 8 processes, spec counters equal arm for arm (694 passes, 2177 proposed, 1161 accepted, 53.3 %). The brief's bar
holds at k5 too: no length slower, >= +2 % with the CI lower bound above 1.0 at 1k, 4k and 16k. Gate-1 md5 on `vrow3`, 64 tokens,
greedy: short `04cdb28a6849728e8aa5be39bf65c07e` and 4k `2252566a5c54cf52cb564bd214c9af2b`, spec default and `--spec off` alike,
equal to main's (`verify2` = #534's code, window 7) and to `vrow2`'s (window 8).

## 5. Windows, jobs, radiance

| window | fleetq | what | radiance down (EST) | down | part 2 total |
|---|---|---|---|---:|---:|
| 7 | #12192 | bench, probe sweeps, 4k trace (card 1 also PR #534's k5 gates) | 16:05:24-16:08:23 | 2:59 | 2:59 |
| 8 | #12197 | gates of vrow2 | 16:13:27-16:16:34 | 3:07 | 6:06 |
| 9 | #12198 | speed 4 + 4 (+ k5 2 + 2) | 16:17:04-16:21:00 | 3:56 | 10:02 |
| 10 | #12217 | merged `vrow3`: k5 speed 4 + 4, gate-1 md5 short / 4k | 16:46:50-16:49:50 | 3:00 | **13:02** of 15:00 |

Radiance answered 15 s after each restart (`/v1/models` checked). CPU jobs on tatooine (`--load cpu`): #12190 (vrow1, saving
verify2's test executables), #12196 (vrow2 + test targets).

**Merge with main `b7450223e` (#534, HIP k5).** Merged origin/main into the branch (merge commit, no force push); the two conflicts
(`docs/tuning.md`, `docs/decisions/LOG.md`) keep both sides' rows. The `kernels/` and `crates/` changes against main are the same
as before the merge, apart from `verify_rows_real.rs`, whose timing knobs both branches carried and which is now main's copy. Checks
on the merged tree, no GPU: tatooine #12205 (`--load cpu`) builds `vrow3` (md5 `9d2ba4dc44263671de42380d9a9bd768`) and the hip test
targets `ink_mix` / `ink_attn` / `ink_rows` plus `verify_rows_real`, and `docs_env_vars` passes (2/2); endor #12206 (prepush:
`docs_env_vars`, `recipes_doc`) and #12212 (`intarsia-models --lib dspark_default`, 1/1) pass. The merge adds only #534 (the recipe k change
and its tests). Window 10 measured `vrow3` at k5 (section 4b); the probe, hip tests, admission, tiny suites and G2 were run on `vrow2`
(window 8) and not rerun, since the kernel and crate code is the same.
