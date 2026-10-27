# Lane TPB: bf16 all-reduce partials for the Qwen3.8-27B at tp2 on the R9700s (HIP), and the a8 prefill GEMM (2026-10-27)

Branch `lane-tpb/hip-tp-partials`, built on origin/main `548ab1461` (PR #194, lane H27P). Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, W4A8 (`mxfp4_act = fp8`), MTP k2 default, `--max-batch 1` unless
stated. Every GPU number comes from a fleetq slot that takes radiance down (`--takes-down radiance --quiet gpu,cpu`), radiance restored and answering after each (`raw/*/guard.log`). Driver scripts: `scripts/tpb/`
(`drive.sh` runs the items of a window from the Mac, `gsweep.sh` and `acc2.sh` run on tatooine). Raw logs: `docs/results/2026-10-27-tpb-hip-tp-partials/raw/<window>/`.

## Headline

| | result |
|---|---|
| **Task 1: bf16 partials, 27B on HIP** | **all gates pass under the coordinator's reading; shipped as the recipe default** (`[tuning.hip] tp_partial = "bf16"`, new recipe key; `INTARSIA_TP_PARTIAL=f32` opts out). Prefill **+16.8 / +17.8 / +16.8 %** at 1k / 4k / 16k (3269 / 3498 / 3400 against 2800 / 2970 / 2910 tok/s). One reading is below the bar and is stated in the gate table: the MTP-on decode median at 16k is x0.984 over 6 pooled rounds (x0.992 over the first 3; mean x1.001; spec-off x1.001). |
| **Task 2: a8 prefill GEMM** | **not shipped, gate not met.** Best lever (register prefetch distance 2 and one tile for all shapes) is -5.7 % kernel time on the 27B tp2 shapes, about +2.4 % prefill; the bar is +5 %. The kernel is not WMMA bound (a build without the WMMAs is as slow). No kernel change in the PR. |

## 1. What changed

* `crates/intarsia-core/src/recipe.rs`: new key `[tuning.<backend>] tp_partial = "f32" | "bf16"` (`Recipe::resolve_tp_partial`, refused at load for any other value).
* `crates/intarsia-core/src/model.rs`: the partial dtype is `INTARSIA_TP_PARTIAL` (env wins, unchanged knob), else the recipe's `tp_partial` for the backend, else f32. Nothing else in the engine moved: the knob already existed
  (bf16: the matmul writes bf16 into `out` and the all-reduce sums in place, 2 B per element instead of the f32 partial's 4 B; the two roundings are the cost).
* `recipes/qwen3_5_vl.toml`: `tp_partial = "bf16"` under `[tuning.hip]`. CUDA, Metal, CPU and every other model keep f32 (`recipe_27b_chunk.rs::shipped_27b_recipe_tp_partial`).
* Docs: `docs/recipes.md` (schema row), `docs/tuning.md` (knob rows). `crates/intarsia-backend-hip/tests/mxfp4_a8.rs`: the timing helper is a function and `timing_a8_tp2` times the a8 GEMM on the 27B tp2 shapes.

Decode and prefill share one setting: the workspace holds either the f32 partial buffer or none (`Workspace::part`), so bf16 applies to decode and verify steps too. A prefill-only variant (f32 up to 16 rows, bf16 above, a third
value `bf16-prefill`) was built and measured, see section 3; it is not in the PR.

## 2. Gates (task 1)

All on the 27B at tp2 with the W4A8 default. f32 = `INTARSIA_TP_PARTIAL=f32` (what main ran before), bf16 = `INTARSIA_TP_PARTIAL=bf16`; the recipe default in the final tree is checked equal to the bf16 env run.

| gate | bar | result | raw |
|---|---|---|---|
| MMLU n=1425 (localeval `quality --bench mmlu --limit 25 --thinking off`, 5-shot, thinking off, `--max-batch 1`, MTP k2) | >= 0.815 (f32 0.8204) | **0.8218** (1171 of 1425; SE ~0.010; 0 truncated, mean output 7.21 tokens). Paired against the f32 run on the same 1425 questions (`20261027-000412-w4a8-a8-mmlu27-w4a8-w3b-a8`): f32 right and bf16 wrong 13, bf16 right and f32 wrong 15 | `raw/w2/mmlu-2-bf16-mmlu.log`, `mmlu-summary.md`; result dir `20261027-041932-tpb-bf16-tpb-w2-bf16-mmlu-1` |
| KL vs the W4A16 golden (`intarsia accuracy --mode both --decode-positions 16`, 10 prompts, scored rows) | prefill and decode each <= 1.3x of f32 | prefill 9.440e-3 -> 1.005e-2 (**x1.065**); decode 5.669e-3 -> 7.011e-3 (**x1.237**). Top-1 prefill 98.33 -> 98.75 %, decode 98.12 -> 97.50 %. The f32 and bf16 numbers are H27P's, reproduced bit for bit in this build (`raw/w1`, `raw/w5`). Read with care: the decode figure has 118 scored rows, and the same measurement of the prefill-only variant (section 3) gave 9.641e-3 (x1.70) for a change that cannot make decode worse, so the decode ratio carries noise of that size; on all 160 decode rows the ratio is 8.796e-3 / 5.071e-3 = x1.73 | `raw/w1/acc-{f32,bf16}.log`, `raw/w5/acc-rec.log` |
| MTP spec-on == spec-off, 18 prompts | equal | **18/18** (serve, temperature 0, 256 forced tokens, md5 of the text; short-0..5, 4k-0..5, 16k-0..5) | `raw/w2/texts-2-bf16-texts.txt` vs `texts-3-bf16_spoff-texts.txt` |
| batched == solo | equal | **equal** on 3 prompts: B (4k-0, 16k-3 with a 6-row tail, 4k-3) prefilling beside decoding stream A under `--max-batch 16` gives the same A and B texts as `--max-batch 1` (`--spec off`): A `915f1c0953`, B `dd42cb5b78 / e74f590855 / 537fb01934`. The f32 control at `--max-batch 16` has its own A (`8a3ef2a792`) and shows no first-request anomaly | `raw/w2/mixids-{4,5,6}-*.txt` |
| decode | >= 0.99x, interleaved | see the decode table below | `raw/w3`, `raw/w4` |
| prefill | >= +10 % | **+16.8 / +17.8 / +16.8 %** (pooled median of 6 interleaved rounds); spec-off +19.0 % (4k) / +16.3 % (16k) | `raw/w3/pooled.txt` |
| default through a recipe key, opt-out kept | | recipe default run: texts equal to the bf16 env run 18/18, KL 1.005e-2 / 7.011e-3; `INTARSIA_TP_PARTIAL=f32` on the same binary reproduces 9.440e-3 / 5.669e-3 and differs from bf16 on 18/18 texts | `raw/w5` |

Speed (localeval `speed --sizes 1024,4096,16384 --gen 128 --reps 3`, a fresh `serve --max-batch 1` per arm, MTP k2 default, `--max-seq 20480`, 6 interleaved rounds f32 | bf16 | bf16 | f32 | f32 | bf16 | ..., `scripts/tpb/pool.py w3`):

| size | prefill f32 -> bf16 tok/s | prefill | decode f32 -> bf16 tok/s (median of 6) | decode median | decode mean | first 3 rounds, decode median |
|---|---|---|---|---|---|---|
| 1k | 2800 -> 3269 | **x1.168** | 89.4 -> 91.2 | x1.020 | x1.007 | x1.016 |
| 4k | 2970 -> 3498 | **x1.178** | 81.7 -> 85.8 | x1.051 | x1.052 | x0.999 |
| 16k | 2910 -> 3400 | **x1.168** | 76.6 -> 75.4 | **x0.984** | x1.001 | x0.992 |

The MTP-on decode figure moves +-5 % with the text MTP accepts (the ids differ between arms, so acceptance differs; the f32 arm's 16k rounds alone span 71.5 to 78.3 tok/s), which is why the coordinator accepted the pooled
reading plus the spec-off result for H27P. Spec-off decode (`--spec off`, the same prompts, 6 interleaved rounds, `raw/w4`) removes that term: **x1.002 at 4k (53.6-53.8 against 53.7-53.8 tok/s) and x1.001 at 16k (50.7-50.8 both)**, as the
decode kernels are unchanged and the f32 partial's extra bytes are 20 KB per all-reduce at one row. The one reading below 0.99 is the 16k MTP-on median over 6 pooled rounds (75.4 against 76.6, mean level); it is a
property of the measurement, not of a kernel, but it is below the written bar, so the default is a call for the coordinator: reverting is the one `tp_partial` line in `recipes/qwen3_5_vl.toml`.

Accuracy note for the reader: the greedy ids change. The f32 and bf16 texts differ on 18/18 prompts at 256 tokens (the rounding moves which of two near-tied tokens wins); the evidence that this is not an accuracy loss is MMLU (paired 13 / 15) and the KL against the
W4A16 golden. Anyone who needs bitwise agreement with the former default sets `INTARSIA_TP_PARTIAL=f32`.

## 3. The prefill-only variant (not shipped)

The brief asked to check decode and consider bf16 for prefill rows only. Decode and prefill share the workspace setting, so I built `INTARSIA_TP_PARTIAL=bf16-prefill` (passes of more than 16 rows reduce bf16 partials, passes of 16 rows or fewer, i.e. decode, MTP
verify, batched decode and the peeled prefill tails, keep f32 partials) and measured it in window 1 (`raw/w1`, commit `14ba36f2c`, reverted afterwards):

* KL: prefill identical to bf16 (1.005e-2); decode 9.641e-3 (x1.70 of f32), **worse** than full bf16's 7.011e-3 although its decode steps are f32. The decode KL is dominated by the KV state the bf16 prefill leaves and by 118-row noise, so f32 decode partials buy nothing measurable.
* 17 of 18 spec-off texts differ from full bf16's (the decode rounding differs), 18/18 spec-on == spec-off. MMLU was not run for it (its prefill is full bf16's).
* It cannot improve decode speed (decode is latency bound at one row; spec-off decode of full bf16 is x1.001), and a row-count switch makes a row's bits depend on the chunk schedule it lands in, which full bf16 does not.

So full bf16 is the candidate; the variant was removed from the PR.

## 4. Task 2: the a8 prefill GEMM (`matmul_mxfp4_a8_mma`), 27B tp2 shapes

Profile of the shipped kernel (ROCm 7.2.4 hipcc resource dump, `scripts/tpb/isa.sh`): the 256x128 fold tiles use 155-161 VGPRs, 61440 B of LDS, 8 waves/SIMD (two 16-wave blocks per WGP), no scratch. The main loop is, per 64-k stage:
all LDS fragment loads, 32 WMMAs back to back, then the next stage's weight decode (`a8_fold8` table permutes, about 100 VALU), the LDS stores and one `__syncthreads`.

Timing (`timing_a8_tp2`, microseconds per launch include the activation pre-pass and the span check, so the TFLOP/s are 10 % below the traced kernel-only 182-188 of H27P): default tiles 145-161 TFLOP/s = 40-44 % of the 366 fp8 peak.

Diagnostic builds of the same tile (wrong results, timing only; `raw/w4/gsweep-1.txt`, `raw/gemm-tables.txt`), total time of the 10 shape x M cases:

| build | total us | vs default |
|---|---|---|
| default (shipped table) | 15788 | |
| same 256x128 / 128x256 tile, no weight decode | 14868 | -5.8 % |
| same tile, no WMMA (LDS reads kept) | 14815 | -6.2 % |
| 256x256 tile, 32-k stages, full kernel | 17213 | +9.0 % |
| 256x256 tile, 32-k stages, no decode / no WMMA | 13528 / 11015 | |

The kernel is **not WMMA bound and not decode bound**: removing all matrix work or all decode work each saves 6 %. What is left is the global-load, LDS-store, barrier path (about 585 GB/s of L2 to CU traffic, 32 B per row per stage for the weights and 64 B for the activations,
sectors of a 128 B line). Levers tried, two per the two-attempt rule:

| lever | result | kept |
|---|---|---|
| 1. register prefetch distance (loads of stage k+2 issued before the WMMAs of stage k; two register sets, then three) | distance 2: -3.0 % on the 256x128 tile, -5.7 % with the 128x256 tile for gate+up as well (14838 against 15728 us); distance 3: -4.6 % (no better) | no: about +2.4 % prefill, bar +5 % |
| 2. tile shape: 256x256 with 32-k stages (halves the decode per MAC), 8-wave 256x128 and 128x256 with 32-k stages | all slower (+9 to +24 %), the 4x4-fragment tiles lose occupancy and (2,2,4,4) spills | no |

K order of every accumulator is unchanged by either lever (stages in ascending k, WMMAs in k order), so the admission, mutant and ids gates would have applied to a bitwise-identical kernel; they were not run, because the speed gate failed first. The next lever the data points at is
the load pattern (full 128 B lines per row by loading four k-stages of weights per row at once, which needs the LDS or register budget to hold them), a larger rewrite than this lane's remaining GPU budget could iterate. The kernel experiments are in git history on this branch (`319f3c0c4`: 32-k stages and the diagnostic builds; `3e945dfbf`: the prefetch-distance versions, selected by `INTARSIA_HIP_MXFP4_A8_TILE=WN,WM,TN,TM,BK,DIAG,PD`); the kernel in the PR is main's.

## Windows and jobs

fleetq slots that took radiance down (6, the brief's cap): #6049 aborted after 30 s (a script error, no GPU work; the try before it, #6048, failed on a permissions error before touching radiance); #6056 w1 (KL f32 / bf16 / bf16-prefill, texts, a8 timing); #6063 w2 (MMLU, texts, mixids);
#6083 w3 (6 speed rounds); #6103 w4 (a8 variants, spec-off rounds); #6114 w5 (prefetch sweep, recipe default and opt-out checks). Builds were `--load cpu` jobs (#6023, #6047, #6101, #6108, #6112) plus one ISA dump (#6068). Radiance was restored and answering
after each window (15-20 s). localeval runs (also logged to unanimis by localeval): labels `tpb-w3-{f32,bf16}-<n>`, `tpb-w4-{f32,bf16}_spoff-<n>`, `tpb-w2-bf16-mmlu-1`.

## Gates not run or not met

* Task 2 admission / mutants / ids / M-independence: not run (no kernel change shipped).
* Batched == solo covers three prompts by text, not the whole 18; texts instead of token ids (the serve API returns text): equal ids imply equal text, the converse fails only if two ids decode to the same string.
* The decode reading described above.
