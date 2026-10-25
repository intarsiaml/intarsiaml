# Lane mx: the MXFP4 fold as the default W4A4 prefill GEMM on HIP (option B), 2026-10-25

Branch `lane-mx/fold-default`, draft PR #153. The user decided on option B at 13:14: the fold becomes the default, and
the 27B W4A4 baseline and envelope are re-recorded once, only if every G1 gate passes. The reasons and the plan are in
`docs/results/2026-10-25-mx-fold-bitwise-note.md`; the fold itself is `docs/results/2026-10-25-mx-hip-27b-prefill.md`
section 5.

## Change

- `crates/intarsia-backend-hip/src/quantmoe.rs`: `mxfp4_fold_from` is now on unless `INTARSIA_HIP_MXFP4_FOLD=0`, so
  `matmul_mxfp4_mma_fold` (priority 5) outranks v2 for W4A4 MXFP4 at M >= 9. The GEMVs keep M <= 8.
- W4A16 calls (`--mxfp4-act none`) run v2 inside the fold launcher, bitwise.
- `INTARSIA_HIP_MXFP4_FOLD=0` is the kill switch: back to v2, main's bits.
- Unit tests: `mxfp4_fold_is_default_with_a_kill_switch`; `mxfp4_selection_is_pinned` checks both states.
- `crates/intarsia-backend-hip/tests/mxfp4_fold.rs` `fold_rows_do_not_depend_on_m`, widened:
  - every 27B projection shape (8), with M in {9, 16, 17, 63, 64, 65, 255, 256, 257, 2047, 2048, 2049, 4096}, bitwise
    against the same rows of an M = 4100 call;
  - a wide-span regime whose f32 sums round, on three shapes. The control there is that v2 and the fold disagree;
  - negative controls: one flipped weight code moves only its column; one changed activation row moves only itself.

## G1 (fleetq #5361, 13:36-14:03, one R9700, build a3ffa1b = main 0c142a3 + the change; raw/fd1)

| gate | result | evidence |
|---|---|---|
| widened `fold_rows` (HARD STOP) | PASS, 8 shapes x 13 M; both negative controls fire | raw/fd1 test-mx.txt |
| 27B S4 batched decode == solo step at max-batch 16 (HARD STOP) | PASS (683 s) | raw/fd1 mtest-s4.txt |
| tails sweep under the fold (HARD STOP): chunk 4096 + peel 16 vs chunk 256, 34 prompts | **0 of 85014 rows differ** (bitwise); the dumps' resolved kernels read `matmul_mxfp4_gemv+matmul_mxfp4_mma_fold` | raw/fd1 chunkcmp-tails2-fold-c256-vs-c4096-peel16.txt |
| kernels test --quick, 27B / 35B | 66 kernels, 365 cases, 0 fail | raw/fd1 kq-*.out |
| 27B W4A4 golden (HF Quark, P1's settings) | prefill KL 0.1653 PASS (bound 0.197), decode-mode 0.2606, the same as #144's fold arm (v2 / main: 0.1799 / 0.3397) | raw/fd1 accg-dev.out |
| 35B plan show + 256 ids short / 4k, branch vs main | identical | raw/fd1 steps.log |
| 27B `--spec mtp --spec-k 2` 256 ids == `--spec off`, short / 4k | identical | raw/fd1 steps.log |
| batch_recurrent `--include-ignored` | 11 passed | raw/fd1 mtest-recurrent.txt |
| W4A16 golden gate + dumps vs main; W4A4 gate vs #150's baseline | **not obtained in G1**: the binaries were unstamped (fd-sync wrote the stamp without `head=`, so the accuracy provenance refused `--baseline`). Rerun in G2 before the re-record | raw/fd1 a16-gate.err |

Two findings from G1:

1. **The widened test's first form was weaker than it looked.** On `matmul_mxfp4_case`'s synthetic data, v2 and the
   fold agree on every output, because those f32 sums are exact. A test on that data cannot see a summation order that
   depends on M. The wide-span regime was therefore added after G1 (743145b), with the control that v2 and the fold
   disagree there. G2 runs it first.
2. **Earlier mx windows also ran unstamped.** They passed `--allow-unstamped`, which is why this was not caught before.

## G2 (fleetq #5379, 14:55-15:01, build 092c961 = the change + the wide-span test + main 7198932, stamped `head=092c961 code-tree=4ded5072da5f dirty=no`; raw/fd2)

The engine code is the same as G1's a3ffa1b plus main's #148, which changes only the MTP draft vocabulary of the 27B
recipe on the decode side. The diff a3ffa1b..092c961 touches `tests/mxfp4_fold.rs` and no engine file of this change.
G2 first ran G1's golden steps, each a condition of the re-record, then the hard-stop test, and only then the re-record.

| step | result | evidence |
|---|---|---|
| W4A16 golden `--mxfp4-act none`, gated on #150's baseline + envelope (`--pre-change bbc0cde --revlist`) | PASS: floor PASS (prefill KL 1.484e-3, decode 9.84e-4) and baseline PASS | raw/fd2 a16-gate.out |
| W4A16 self-golden dumps, branch vs main 7198932 | **bitwise equal** in every tensor, prefill and decode rows (`scripts/mx/dumpeq.py`) | raw/fd2 a16-dumpeq.txt |
| W4A4 golden vs HF | prefill PASS: KL 0.1653, 0.830x the ref noise | raw/fd2 a4-gate-old-baseline.out |
| W4A4 decode vs #150's baseline + envelope | moved as expected: per-prompt gate FAIL on 6 of 10 prompts. Decode KL vs HF is 0.2606 (main's baseline 0.3397), closer to the reference | raw/fd2 a4-gate-old-baseline.out |
| `fold_rows` with the wide-span regime (HARD STOP) | PASS. On the rounding data v2 and the fold disagree on 5218544 of 5222400, 1535451 of 1536000 and 306977 of 307200 outputs; the fold's rows are still identical at every M | raw/fd2 test-fold2.txt |
| **re-record** of the 27B W4A4 baseline + envelope (user-approved, after every check above) | recorded from 092c961 (`--write-baseline`; envelope `--noise-perturb chunk`); its own gate PASS (floor + baseline, k = 2) | raw/fd2 a4-record, a4-envelope, a4-gate-new; the files are now in golden/baselines/ |
| 27B acc-short (`qualify --quick-only`, new baseline) | **PASS** (prefill top-1 91.95 % KL 1.72e-1; decode top-1 88.12 % KL 2.11e-1); doctor, recipe, plan, kernels (66 / 1465 cases), gate and bench PASS | raw/fd2 qual-27.err |

qualify exits 1 only because of `acc-oracle: MISSING-BASELINE`: there is no CPU-oracle golden for the 27B, on main as
here. Every other stage passes.

`crates/intarsia-cli` `committed_27b_baselines_pair_with_their_envelopes_and_goldens` passes with the new files.
The W4A16 baseline and envelope are unchanged, because their bits did not move.

## L: served speed (fleetq #5381, 15:16-15:29, localeval, one R9700, radiance down; raw/fdloc)

Method: `localeval speed --sizes 1024,4096,16384 --gen 128 --reps 2`, through `scripts/mx/loc-window.sh`.
- Arms: main 7198932 (`base`) vs this branch 092c961 (`dev`), both at the recipe's default chunk 256, no flags.
- Order: interleaved per round, with the order reversed in round 2. Each cell is the median of 3 rounds, each round
  the median of 2 reps.
- Per-round values are in raw/fdloc/served-medians.txt; the localeval results are in /opt/code/deepeval/results
  (labels `mx-fdloc-*`).

| model, prompt | main (v2) prefill tok/s | **branch (fold default)** | change | decode main / branch |
|---|---:|---:|---:|---:|
| 27B, 990 tok | 1120 | **1822** | +63 % | 33.0 / 33.0 |
| 27B, 4088 tok | 1153 | **1892** | +64 % | 32.4 / 32.4 |
| 27B, 16654 tok | 1073 | **1697** | +58 % | 30.8 / 30.8 |
| 35B, 990 tok | 6451 | 6511 | +0.9 % (spread 1.5 %) | 164.5 / 164.3 |
| 35B, 4089 tok | 9939 | 10032 | +0.9 % (spread 1.5-7.9 %) | 160.7 / 160.5 |
| 35B, 16655 tok | 8820 | 8827 | +0.1 % (spread 1.3-2.8 %) | 145.9 / 146.2 |

- The 27B main column reproduces #144's served v2 numbers (1120 / 1154 / 1073), and the branch reproduces #144's fold
  arm at chunk 256 (loc1: 1812 / 1895 / 1695).
- With the opt-in `--prefill-chunk 4096` + `INTARSIA_PREFILL_PEEL=16`, the same kernel served 2043 / 2416 / 2186 in
  #144's loc3. That arm was not rerun here: the default recipe keeps chunk 256 (serve `--max-batch 16` context, #144
  section 4).
- The 35B runs no MXFP4 and is the same within its spread. One 35B 4k round of the branch was low (9239); the other two
  read 10035 and 10032.

## Merge state

The gates ran on 092c961. The branch then merged main 7cb96938 (#152, lane wg's fp8 WMMA small-row GEMVs; #155): that
merge adds no line that mentions MXFP4 to crates/ or kernels/, and it shares only `quantmoe.rs` with this change, in
regions far apart. The HIP crate's unit tests (25, including `mxfp4_fold_is_default_with_a_kill_switch` and
`mxfp4_selection_is_pinned` in both switch states) and the hip-cfg check pass on the merged tree. CUDA and Metal
no-ops are not needed: only `intarsia-backend-hip` and docs / scripts / golden baselines change.

Leftovers on tatooine until the PR merges: `~/intarsiaml-mx-{dev,base}`, the volumes
`intarsia-cargo-target-mx-{dev,dev-hiptest,dev-mtest,base}`, and `~/mx` (the tails golden, G1_OK).
