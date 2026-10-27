# Lane GLR: the W4A8 prefill GEMM (`matmul_mxfp4_a8_mma`), Qwen3.8-27B on the R9700s (HIP) (2026-10-27)

Branch `lane-glr/hip-a8-gemm`, built on origin/main `43642235c` (after PR #204). Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, W4A8 (`mxfp4_act = fp8`), MTP k2 default, `prefill_chunk = 2048` (recipe), localeval against `intarsia serve`. Raw data: `docs/results/2026-10-27-glr-hip-a8-gemm/raw/`, scripts `scripts/glr/`.

## Headline

| | |
|---|---|
| Asked | Rewrite the a8 GEMM's global loads so each weight/activation row reads whole 128 B lines (coalesced), together with prefetch distance 2; fix the slow tile below M = 2048. |
| Line-load rewrite | **Built, bitwise correct, and slower** (+18% call time over the five 27B shapes at M = 2048, +9% on gate+up to +36% on down, [section](#the-128-b-line-load-kernel-lost)). Not shipped. The diagnostic builds say why: with every global load forced to hit the cache the kernel is only 5.8% faster, so there is little left in the loads for any load pattern to win. |
| Shipped | Prefetch distance 2 (two register sets, the loads of stage k+2 in flight during stage k) on the existing stage-load kernel, and a new tile table: the 128 x 256 tile for every M with at least 64 blocks (it also wins below M = 2048 and for gate+up), the 128 x 128 tile for smaller grids. Output **bitwise identical** to main for every shape tried. |
| Prefill, localeval, 6 interleaved rounds per arm | **1k +7.2%, 4k +8.2%, 16k +3.7%** (3295 / 3538 / 3427 -> 3533 / 3828 / 3554 tok/s). Decode unchanged (MTP off: x1.000 at 1k / 4k / 16k; MTP on x1.007 / x0.964 / x0.981, inside the 5% run spread). |
| Gap to Radiance at 4k (6133 tok/s) | 1.73x -> **1.60x**. |
| Knob | `INTARSIA_HIP_MXFP4_A8_PD=1` restores the previous kernel launch (prefetch distance 1, the previous tile table). |
| Gates | All met. One caveat on the profile request: the rocprof memory-traffic counters read 0 on this ROCm, so "traffic and occupancy explain the gain" is shown for occupancy (unchanged) and busy cycles, not for traffic; the gain is latency hiding. |
| Windows | 6 of 6 used (W1 to W6), every one restored Radiance and confirmed `/v1/models` before release. |

## What the kernel looked like, and what the diagnostics say

Main's GEMM: tile 128 x 128 (4 x 2 waves) below M = 2048, 256 x 128 for N >= 8192 above it, else 128 x 256 when there are 64 blocks. 64 k per stage, LDS double buffer, one register set of global loads (prefetch distance 1). The TPB lane measured prefetch distance 2 at -5.7% kernel time on the 27B shapes and named full-line loads as the next lever. Its variants were never merged (commits `319f3c0c4`, `3e945dfbf`).

Diagnostic builds of the stage-load kernel (tile 128 x 256, distance 1, M = 2048, sum of the five 27B shapes 5101 us, raw `raw/w1/t-diag*.txt`; timing only, wrong results):

| build | sum of 5 shapes (us) | vs 5101 (128 x 256 tile, distance 1) | reading |
|---|---|---|---|
| DIAG 3: every stage re-reads stage 0, loads always hit | 4806 | -5.8% | **upper bound for any global-load rewrite**: an infinitely good load pattern gains 5.8% |
| distance 2 (what ships, tile 128 x 256) | 4868 | -4.6% | already takes four fifths of that bound |
| DIAG 4: no block barrier | 4642 | -9.0% | the barrier costs about as much as the loads |
| DIAG 6: no stores to LDS (no weight decode either) | 3824 | -25% | stage decode and LDS stores are a quarter of the time |
| DIAG 2: no WMMA | 5054 | -0.9% | the matrix unit is not the bound |
| DIAG 5: no fragment reads from LDS | 2932 | **-43%** | the per-wave LDS fragment reads are the biggest single cost |
| DIAG 1: no weight decode | 4940 | -3.2% | decode is cheap (TPB found 6%) |

So the premise "next lever is coalesced loads" is mostly closed by distance 2; the next lever is LDS traffic (see Next).

## The 128 B line-load kernel lost

`a8_line_kernel` (in `raw/isa/mxfp4_a8_gemm.experiments.hip`, not shipped): same tile, LDS layout, WMMA loop and k order as the fold path, but 8 lanes read one row's 128 B in one load instruction (weights: 256 k = four stages = 8 scale groups per row per load set; activations: 128 k). Each thread owns one weight chunk and one or two activation chunks per stage, picked from the register set with lane-dependent selects. Bitwise equal to the stage-load kernel on 48 hashed shapes (every K tail, N tail, M on both sides of the tile cutovers) and on the reference, rows-vs-solo and position tests.

| M = 2048, us per call | gate+up | down | q+gate | z | o | sum |
|---|---|---|---|---|---|---|
| main (default table) | 2448 | 1129 | 800 | 410 | 401 | 5189 |
| line-load, same tiles | 2669 | 1538 | 961 | 482 | 479 | 6129 (+18%) |
| distance 2 stage-load | 2210 | 1093 | 780 | 394 | 392 | 4868 (-6.2%) |

It was slower on every shape and M (`raw/w1/t-line*.txt`). The compiler spent 180 to 210 VGPRs on it (155 to 161 on main's kernel; 4 x 4 tile variants spilled until the register sets were rewritten as scalars), the 4-way chunk pick adds VALU in the store path, and the cross-lane layout makes each lane's 16 B pieces land in LDS rows that are 128 B apart. Given DIAG 3, no version of this could have beaten distance 2 by more than 1 to 2%.

Other schedules tried on top of distance 2 and dropped (all bitwise equal, `raw/w3`, `raw/w4`): LDS-only barrier (`s_wait_dscnt` + `s_barrier` instead of the full fence), fragments read before the next stage's decode and store, a software-pipelined variant with `s_setprio`, half-group (8 B) weight chunks for balanced decode lanes, 2 x 4 and 4 x 2 waves with 4 x 4 fragments (VGPR 255, spills), distance 3 with the early-store schedule. None beat plain distance 2 by more than noise; the half-group decode was 3% slower.

## What ships

`kernels/hip/mxfp4_a8_gemm.hip` (diff against main is the kernel only, plus `INTARSIA_HIP_MXFP4_A8_PD` in `crates/intarsia-backend-hip/src/kernels.rs` `ENV_KNOBS` and `docs/tuning.md`):

* `a8_gemm_kernel<..., PD>`: `Stage st[PD]` register sets, prologue loads stages 1 .. PD-1, the main loop is unrolled by PD with `step(kt, nx, ld)`. The k order of every accumulator is unchanged (stages ascending, a stage's WMMAs in k order), so the output cannot change bits. 175 VGPR (161 before) at the 128 x 256 tile, still 8 waves/SIMD, no scratch (`raw/isa/resources-shipped-kernel.txt`).
* Table (fold launch; the per-group fallback launch stays distance 1 and is unchanged): M <= 64: 64 x 16 tile (unchanged); at least 64 blocks of 128 weights x 256 tokens: 128 x 256, distance 2 (this now also covers gate+up and M < 2048); else 128 x 128, distance 2.
* `INTARSIA_HIP_MXFP4_A8_PD=1`: previous table, distance 1. `INTARSIA_HIP_MXFP4_A8_TILE=WN,WM,TN,TM[,PD]` accepts an optional distance.

## Kernel time and TFLOP/s, before and after

Kernel-only (rocprofv3 kernel trace, median of 20 launches, M = 2048; old = main in W4 `raw/w4/kt-base.csv.gz`, new = the shipped binary in W6 `raw/w6/gates-1/kt-base.csv.gz`; cross-window, so about +-2% clock drift):

| shape (N x K) | before us | before TFLOP/s | after us | after TFLOP/s | change |
|---|---|---|---|---|---|
| gate+up 17408 x 5120 | 2229.6 | 163.7 (44.7% of 366) | 1950.0 | 187.2 (51.2%) | -12.5% |
| down 5120 x 8704 | 985.2 | 185.3 (50.6%) | 939.1 | 194.4 (53.1%) | -4.7% |
| q+gate 6144 x 5120 | 694.3 | 185.6 (50.7%) | 671.9 | 191.8 (52.4%) | -3.2% |
| z 3072 x 5120 | 350.4 | 183.9 (50.2%) | 337.9 | 190.6 (52.1%) | -3.6% |
| o 5120 x 3072 | 353.4 | 182.3 (49.8%) | 339.8 | 189.6 (51.8%) | -3.9% |
| sum | 4612.9 | | 4238.7 | | -8.1% |

Call level (kernel plus the activation quantisation and the weight scale pre-pass; `timing_a8_tp2`, old = `INTARSIA_HIP_MXFP4_A8_PD=1`, runs ordered old, new, old, new-reversed, old so clock drift cancels; `raw/w6/gates-1/t-*.txt`), sum of the five shapes, ratio new / old:

| M | 256 | 512 | 1024 | 1536 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| new / old | 0.953 | 0.909 | 0.797 | 0.884 | 0.924 | 0.937 |

M = 1024 (the 1k prompt) is where the old table's 128 x 128 tile hurts most (gate+up 1447 -> 1080 us, 126 -> 169 TFLOP/s).

## Speed, localeval, base vs branch

`localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3`, MTP on, max-batch 1, `prefill_chunk` 2048, one server per arm start, 14 runs in the order A B B A A B B A A B B A B A (6 per arm), `raw/w5/steps.log`, table by `scripts/glr/speedtab.py`. Base = `glr-base-cli` built from origin/main `43642235c`, branch = `glr-dev-cli`.

| prompt | prefill base | prefill branch | ratio | decode base | decode branch | ratio |
|---|---|---|---|---|---|---|
| 1k | 3295 | 3533 | **1.072** | 88.3 | 88.9 | 1.007 |
| 4k | 3538 | 3828 | **1.082** | 82.8 | 79.9 | 0.964 |
| 16k | 3427 | 3554 | **1.037** | 75.4 | 74.0 | 0.981 |

All six branch runs beat all six base runs at 4k (branch min 3703, base max 3606). Decode with MTP on has a 5 to 10% run spread (4k branch runs 78.6 to 90.8). The decode kernels are unchanged (the verify steps run 3 rows through `matmul_mxfp4_a8_gemv_rows`, not this GEMM), so the MTP-off reading is the one that counts (`--spec off`, 2 runs per arm, `raw/w6/steps.log`):

| prompt | prefill base | prefill branch | ratio | decode base | decode branch | ratio |
|---|---|---|---|---|---|---|
| 1k | 3301 | 3584 | 1.086 | 54.4 | 54.5 | 1.001 |
| 4k | 3511 | 3825 | 1.089 | 53.7 | 53.7 | 1.000 |
| 16k | 3406 | 3582 | 1.052 | 50.8 | 50.8 | 1.000 |

Gap to Radiance (6133 tok/s at 4k, from the TPO doc): 1.73x -> 1.60x. The isolated kernel gain at 2048 rows (-8.1%) times the GEMM's 51.5% share predicts +4.5% prefill; the measurement is +8.2%. I did not find the other half (the share may be larger than the TPO profile's 51.5% once the quantisation pre-pass and the 2036-row tail piece are counted); the measurement is the localeval figure and is what the gate uses.

## Profile: what explains the gain

* Occupancy and waves: unchanged. `kernel-resource-usage`: 175 VGPR vs 161, 8 waves/SIMD both, LDS 61440 B both, no scratch (`raw/isa`). rocprofv3 `SQ_WAVES` identical (q+gate 6144, gate+up 17408 per launch, old and new) (`raw/w5/kprof-16`).
* Busy time: `GRBM_GUI_ACTIVE` per launch at M = 2048: q+gate 1,824,965 -> 1,738,883 (-4.7%), gate+up 6,296,908 -> 5,517,399 (-12.4%), the same shape as the kernel times.
* Memory traffic: **not measured**. Every TCP / GL2C / SQ_INSTS / LDSBankConflict counter reads 0 on every dispatch (including rocclr's own copy kernels) in 24 counters over 3 windows, also when collected one counter per pass (`raw/w5/kprof-16`, `raw/w2/ctab.txt`). Only `GRBM_GUI_ACTIVE`, `SQ_WAVES` and `SQ_BUSY_CYCLES` return values. That is the collection (rocprofv3 on gfx1201 with ROCm 7.2.4), not the kernel. Bytes moved are unchanged by construction (same loads, same addresses, issued one stage earlier).
* So the gain is latency hiding, not less traffic and not more occupancy: the loads of a stage now have two stages of WMMA/decode/barrier time to land instead of one, which the DIAG 3 numbers bear out (distance 2 closes most of the gap to a kernel whose loads always hit), plus the 128 x 256 tile below M = 2048 where main used 128 x 128 (fewer LDS fragment reads per WMMA).

## Gates

| gate | result |
|---|---|
| prefill at 4k >= +5%, median of >= 6 interleaved rounds | **met**: +8.2% (6 vs 6) |
| prefill at 1k and 16k not slower | **met**: +7.2%, +3.7% |
| decode not slower | **met**: MTP off x1.000 at 1k / 4k / 16k (decode kernels unchanged); MTP on x1.007 / 0.964 / 0.981 inside the run spread |
| greedy ids identical to main | **met**: `texts.py`, 18 prompts x 256 forced tokens, md5 per prompt identical between base (spec on), branch (spec on) and branch (spec off) (`raw/w5/texts-*`); GEMM output hash identical to the previous kernel on 48 shape/dtype cases (`raw/w6/gates-1/hash-*.txt`). Bitwise, so KL and MMLU were not rerun. |
| M-independence and row-position test | **met**: `a8_mma_rows_do_not_depend_on_m` (existing) and the new `a8_mma_rows_do_not_depend_on_position` (a probe row at 16 positions of calls with M in {17, 65, 255, 257, 1000, 2047, 2048, 2049, 4100}, four shapes, bf16 and f32 out) pass on the default, `PD=1`, `PERGROUP=1` and both (`raw/w6/gates-1/corr-*`) |
| `kernels test` admits the kernel, >= 3 planted mutants caught | **met**: `kernels test --backend hip --quick`: `matmul_mxfp4_a8_mma` ADMITTED, 84 kernels / 414 cases, none failed (10 kernels with no quick case, as on main). 6 of 6 mutants fail the `a8_` tests while the clean build passes: m1 register sets swapped, m2 prologue loads stage 2, m3 grid drops the last M tile, m4 fold shift off by one octave, m5 row 1 scaled by 1.0000002 (caught only by the bitwise tests: M-independence and position), m6 loop runs one extra stage (`raw/w6/gates-1/mut-*.txt`, summary `raw/summary/w6-sum.txt`) |
| batched == solo, MTP spec-on == spec-off on the serve probe set | **met**: `mixids.py` (a 48-token stream B prefilled beside a running stream A) max-batch 16 vs 1: A `915f1c0953`, B `dd42cb5b78` / `e74f590855` / `537fb01934` in both (and equal to TPO's main-binary values); spec on vs off texts equal (above) |
| `plan show` byte-identical, 35B and 8B on HIP | **met**: 8B 1346 B, 35B 3184 B, base CLI vs branch CLI (`raw/w6/gates-1/plan-*`) |
| Mac `cargo test --workspace` | **met**: 195 `test result: ok`, exit 0 (`raw/mac-cargo-test.log`) |
| CI green | checked on the PR (see the PR thread / merge note) |
| `INTARSIA_*` knob restores the old kernel | **met**: `INTARSIA_HIP_MXFP4_A8_PD=1` (hash and timing arms above run it) |

## Windows (fleetq, tatooine, all took Radiance down and restored it)

| window | job | what |
|---|---|---|
| W1 | #6198 | line-load kernel: correctness, hashes, timing, DIAG builds, first counters |
| W2 | #6204 | schedule variants, tiles, counters |
| W3 | #6208 | barrier / schedule variants, drift-balanced rounds |
| W4 | #6215 | half-group decode, kernel trace |
| W5 | #6244 | localeval base vs branch rounds, ids texts, single-counter profile |
| W6 | #6261 | correctness, hashes, mutants, admission, `plan show`, call timing, mixids, MTP-off speed |

CPU jobs (builds and ISA dumps, among others): #6168 #6181 #6182 #6193 #6197 #6203 #6207 #6214 #6238 #6239 #6240 (the last built the mutants).

## Leftovers and next

* The profile's memory-counter half is open (counters read 0). A different collection route (`rocprofv3 --att` or `rocprof-compute`) was not tried.
* **Next lever: LDS fragment traffic.** DIAG 5 (no fragment reads) is -43%, DIAG 2 (no WMMA) is -1%: the kernel is bound by the LDS reads of the A/B fragments (each wave re-reads its 2 x 4 fragments per 32 k) and the barrier, not the matrix unit and, after this change, not the global loads. 4 x 4 fragments per wave halve that traffic but need 255 VGPRs and spill; a split of the k loop across a wave pair, or reading the activation fragments once per two weight tiles through `ds_load_tr`, are the candidates. Peak is 366 TFLOP/s; the shipped kernel is at 52%.
* A fleetq cpu job built the CLI with the stale-mtime trap (`rsync -a` keeps Mac mtimes, cargo then reused the other tree's kernel library once): `scripts/glr/build.sh` and `tbuild.sh` now touch `build.rs` and the kernels first. The first dev CLI of this lane was affected and rebuilt before any speed number was taken.
* The experiments (line-load kernel, BAR/ORD/WB schedules, DIAG builds, 8-wave tiles) live in `raw/isa/mxfp4_a8_gemm.experiments.hip`, not in the shipped kernel, to keep the template count (and the CI build) at main's.
