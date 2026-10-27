# Lane LDF: LDS fragment traffic of the W4A8 prefill GEMM (`matmul_mxfp4_a8_mma`), Qwen3.8-27B on the R9700s (HIP) (2026-10-27)

Branch `lane-ldf/hip-a8-lds`, built on origin/main `4bcd080f3` (PR #206 merged). Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, W4A8 (`mxfp4_act = fp8`), MTP k2 default, `prefill_chunk = 2048`, localeval against `intarsia serve`. Raw data: `docs/results/2026-10-27-ldf-hip-a8-lds/raw/`, scripts `scripts/ldf/` (the sweep-only scripts and the micro-benchmark source are in `raw/scripts/`).

## Headline

| | |
|---|---|
| Asked | Cut the LDS fragment-read cost of the a8 GEMM: wider `ds_read`, an XOR swizzle or padding, register blocking, `ds_read_tr`. Measure each in a micro-benchmark on one R9700, keep the best, show the ISA. Merge if prefill at 4k is >= +5%. |
| Result | **Prefill gate not met, so this is a draft PR.** The only candidate that helped the real kernel is the XOR-swizzled 64 B tile rows: **-2.2% call time** at M = 2048 over the five 27B shapes (-2.6% at 1k, -2.0% at 4k), bitwise equal output. End to end it is **not measurable**: prefill 1k / 4k / 16k **x1.006 / x0.992 / x1.007** (6 interleaved runs per arm, spread +-2.5%), decode with MTP off x1.000. |
| Why the premise did not pay | The LDS fragment reads are not bandwidth bound: the same read loop runs at 13.6 TB/s read-only and 5.7 TB/s inside the real WMMA loop, and the card is power bound (295 W of a 300 W cap, 2.45 GHz) in the real kernel. Padded 80 B rows were already free of read bank conflicts. `ds_read_tr` does not exist on gfx1201. The wider-read, split-barrier, late-shift, branch-free-load and table-in-LDS variants all lost or tied in the real kernel (sweep below). |
| Gap to Radiance at 4k (6133 tok/s) | 1.59x (main, 3855 tok/s) -> 1.60x (branch, 3825). Unchanged. |
| Knob | `INTARSIA_HIP_MXFP4_A8_LDS=0` restores the padded 80 B rows (the PR #206 kernel). `INTARSIA_HIP_MXFP4_A8_PD=1` still restores the older launch. |
| Gates | All met except the speed one (table below). |
| Windows | 6 of 6 used, every one restored Radiance and confirmed `/v1/models` before release. Two were partly wasted by my own script errors (table below). |

## What was measured first: the LDS read loop on its own

`raw/scripts/ldsbench.hip` (one R9700, `raw/w3/lds.txt`): the kernel's inner loop (4 x 4 waves, 2 x 4 fragments per wave, 32 k per sub-step, 12 `ds_read_b128` and 32 WMMAs per wave and stage) with no global loads and no weight decode. Peak is 366 TFLOP/s.

| row | LDS B | TFLOP/s | LDS read TB/s | reading |
|---|---|---|---|---|
| READ only, padded 80 B rows, b128 | 61440 | | 13.6 | the read rate of the LDS |
| READ only, padded, two b64 | 61440 | | 13.7 | width is not the limit |
| READ only, swizzled 64 B rows | 49152 | | 13.6 | the swizzle reads as fast as padding |
| READ only, 64 B rows, no pad | 49152 | | 3.7 | bank conflicts (3.7x slower) |
| READ only, 72 B rows | 55296 | | 1.9 | bank conflicts (7.4x slower) |
| padded, barrier + stage stores (the kernel's shape) | 61440 | 241.5 | 5.7 | the loop reads at 42% of the LDS rate |
| padded, no barrier, no stores | 61440 | 336.7 | 7.9 | -28% from barrier + stores |
| padded, barrier, no stores | 61440 | 329.5 | 7.7 | |
| padded, no barrier, stores | 61440 | 317.3 | 7.4 | the stores cost more than the barrier |
| swizzled, barrier + stores | 49152 | 264.1 | 6.2 | +9.4% over padded |
| swizzled, no barrier, no stores | 49152 | 348.7 | 8.2 | |
| padded, two b64 reads, barrier + stores | 61440 | 269.9 | 6.3 | +11.8% |
| swizzled + two b64, barrier + stores | 49152 | 268.3 | 6.3 | +11.1% |
| split barrier (signal after the first 32 k, wait at the end), padded | 61440 | 247.7 | 5.8 | +2.6% |
| split barrier, swizzled | 49152 | 261.3 | 6.1 | +8.2% |
| split barrier, padded, two b64 | 61440 | 261.9 | 6.1 | +8.4% |
| split barrier, swizzled + two b64 | 49152 | 272.8 | 6.4 | +13.0% |
| 2 x 4 waves, 4 x 4 fragments, barrier + stores | 61440 | 268.7 | 4.2 | -33% LDS bytes per WMMA, +11.3% |
| 2 x 4 waves, 4 x 4 fragments, split barrier, padded | 61440 | 298.6 | 4.7 | +23.6% |
| 2 x 4 waves, 4 x 4 fragments, split barrier, swizzled | 49152 | 285.9 | 4.5 | +18.4% |
| 4 x 2 waves, 2 x 4 fragments (128 x 128), barrier + stores | 40960 | 248.5 | 5.8 | |
| 4 x 4 waves, 2 x 2 fragments (128 x 128), barrier + stores | 40960 | 261.1 | 8.2 | |
| weights from registers (no LDS reads for them), barrier + stores | 61440 | 300.1 | 7.0 | +24% (a bound, not a design: weights cannot live in registers across stages) |

So reading the fragments needs 42% of the LDS rate when the loop runs: the loop is not LDS-bandwidth bound. What costs time is the barrier together with the stage stores (-28% against the pure loop). In the micro-benchmark the swizzle, two b64 reads and the split barrier each win 3 to 12% and stack to +13%, but in the real kernel only the swizzle won (sweep below): the micro-benchmark has no global loads, no weight decode and no power cap, so it over-states every candidate except the one that removes real conflicts.

Bank conflicts could not be counted: every rocprofv3 LDS/TCP counter reads 0 on this ROCm (as GLR found). The micro-benchmark shows the padded 80 B layout is already conflict-free for the 16 B fragment reads (13.6 TB/s read-only, same as the swizzle; 64 B and 72 B rows without the swizzle are 3.7x and 7.4x slower). Where the swizzle gains in the loop (the stage's 16 B stores, or the interplay of stores with reads) could not be separated without counters; I did not hand-derive it.

## Candidates, and what the real kernel did with each

Variant bits in `raw/isa/mxfp4_a8_gemm.sweep.hip` (`INTARSIA_HIP_MXFP4_A8_LDV`, a sweep-only knob, not shipped), sum of the five 27B shapes at M = 2048, `timing_a8_tp2`, one pass each (`raw/w3/t-v*.txt`). The card drifted +5.6% over the sweep (V0 first 4810 us, last 5078 us), so single passes carry about +-2%; the top rows were re-run in 4 interleaved rounds.

| candidate | V | sum (us) | vs V0 | 4 interleaved rounds (M = 1024 / 2048 / 4096) |
|---|---|---|---|---|
| PR #206 kernel (padded rows) | 0 | 4810 | | |
| **XOR swizzle, 64 B rows** | 1 | 4719 | -1.9% | **-4.0% / -2.4% / -1.7%** |
| fold table read from LDS, not `__constant__` | 2 | 4829 | +0.4% | |
| LDS-only barrier (no wait for loads in flight, no cache invalidate) | 4 | 4843 | +0.7% | |
| fold shift computed at the stage store | 8 | 4971 | +3.3% | |
| global loads without per-load branches | 16 | 4888 | +1.6% | |
| two `ds_read_b64` per fragment | 32 | 4928 | +2.5% | |
| split barrier (stores before the signal, wait after the last WMMAs) | 64 | 4901 | +1.9% | |
| swizzle + table in LDS | 3 | 4725 | -1.8% | -4.2% / -2.5% / -2.0% |
| swizzle + two b64 | 33 | 4822 | +0.3% | |
| swizzle + split barrier | 65 | 4818 | +0.2% | |
| swizzle + branch-free loads | 17 | 4794 | -0.3% | -3.2% / -2.1% / -1.4% |
| swizzle + table + branch-free loads | 19 | 4791 | -0.4% | -3.6% / -2.1% / -1.7% |
| all six | 127 | 4877 | +1.4% | |

Not buildable or not useful:

* `ds_read_tr` / transpose loads: the assembler rejects `ds_read_b64_tr_b8`, `ds_read_b64_tr_b16`, `ds_load_tr8_b64` and `ds_load_tr16_b128` for gfx1201 (`llvm-mc -mcpu=gfx1201`: "instruction not supported on this GPU"). They are gfx950 / gfx1250.
* Register blocking, 4 x 4 fragments per wave: the 128 x 256 tile as 2 x 4 waves compiles to 256 VGPR with 104 to 188 B of scratch per lane in every variant, 4 waves per SIMD (`raw/isa/resources-sweep.txt`). Its inner loop is +11% to +24% in the micro-benchmark (table above), but with scratch spills and half the occupancy it was not run end to end here and is not shipped (GLR's tile table has no such tile either).
* 8 waves per SIMD is kept: the swizzled kernel is 180 VGPR (175 before), no scratch.

Clock and power under the real kernel (`raw/w3/clk-*.txt`, both cards sampled every 0.1 s, M = 4096, 600 launches): V0 2451 MHz mean at 295 W, swizzle 2426 MHz at 293 W, power cap 300 W. The R9700 is at its power cap while this GEMM runs, which fits the picture that energy per WMMA (LDS replays, VALU) rather than a latency fix decides the speed here.

## ISA: what changed

`raw/isa/loopshape.txt` (`scripts/ldf/loopshape.py` on the `hipcc -S` dump): the steady-state loop of the 128 x 256 distance-2 fold kernel, two stages unrolled. PR #206 kernel, padded kernel of this branch (`LDS=0`) and swizzled kernel have the **same instruction mix**: 64 `v_wmma`, 24 `ds_load_b128`, 4 `ds_store_b128`, 2 x (`s_barrier_signal`, `s_barrier_wait`, `global_inv`), 6 `global_load_b128`, 2 `global_load_u8`, 1 `global_load_b64` (the fold table). The `s_wait_dscnt` pattern before the first WMMAs is 9, 5, 3, 1 (padded) against 9, 8, 7, 6 (swizzled): the compiler issues the 12 fragment reads in another order. What differs is the LDS address of every `ds_load`/`ds_store`, not the count, so the gain is in how the lanes of an instruction land on the banks, which is why the instruction counts and the waits do not show it. Resources (`raw/isa/resources-final.txt`): swizzled 128 x 256 kernel 180 VGPR, 49152 B LDS, occupancy 8, no scratch (padded: 175 VGPR, 61440 B); swizzled 128 x 128 kernel 180 VGPR, 32768 B LDS, occupancy 8 (padded: 175 VGPR, 40960 B, occupancy 6).

The loop keeps, in all three builds, an `s_wait_loadcnt 0x1` / `0x0` pair per stage (the scale byte and the prefetch loads are waited for inside the loop) and a `global_inv` after each barrier. Variants that remove or move them (V = 2, 4, 8, 16 in the sweep: table in LDS, LDS-only barrier, late fold shift, branch-free loads) did not help on their own.

## Kernel time and TFLOP/s, before and after

Call level (kernel + activation quantisation + weight-scale pre-pass), `timing_a8_tp2`, one window (W5), before = `INTARSIA_HIP_MXFP4_A8_LDS=0` (the PR #206 kernel; its output hashes equal main's binary), after = default; `raw/w5/gates-1/t-lds0.txt`, `t-dev.txt`. Kernel-only rocprof traces were not taken (see leftovers); GLR's kernel-only figures for the PR #206 kernel at M = 2048 were 187 to 194 TFLOP/s.

M = 2048:

| shape (N x K) | before us | before TFLOP/s | after us | after TFLOP/s | change |
|---|---|---|---|---|---|
| gate+up 17408 x 5120 | 2209.3 | 165.2 | 2161.2 | 168.9 | -2.2% |
| down 5120 x 8704 | 1084.5 | 168.3 | 1059.7 | 172.3 | -2.3% |
| q+gate 6144 x 5120 | 781.1 | 165.0 | 765.0 | 168.4 | -2.1% |
| z 3072 x 5120 | 394.9 | 163.1 | 385.2 | 167.3 | -2.5% |
| o 5120 x 3072 | 390.0 | 165.2 | 383.3 | 168.1 | -1.7% |
| sum | 4859.8 | | 4754.4 | | -2.2% |

Sum of the five shapes at the other M (before / after, us): M = 256 841 / 826 (-1.8%), 512 1353 / 1328 (-1.8%), 1024 2384 / 2322 (-2.6%), 1536 3823 / 3735 (-2.3%), 4096 9971 / 9770 (-2.0%). The older launch (`PD=1`) is 5304 us at M = 2048, so the PR #206 and this change together are -10%.

The 128 x 128 tile now also has 8 waves per SIMD (LDS 32 KB). Forced for every M (`INTARSIA_HIP_MXFP4_A8_TILE=4,2,2,4,2`) it ties the 128 x 256 tile at M = 256 and 512 (827 / 1349 us against 826 / 1328) and loses from M = 1024 (+10%), so the table is unchanged.

## Speed, localeval, base vs branch

`localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3`, MTP on, max-batch 1, `prefill_chunk` 2048, one server per arm start, 12 runs in the order A B B A A B B A A B B A (6 per arm; `raw/w6/steps.log`, the result files are in `raw/w6/results/`, table by `scripts/ldf/speedtab.py`). Base = `ldf-base-cli` built from origin/main `4bcd080f3`, branch = `ldf-dev-cli`.

| prompt | prefill base | prefill branch | ratio | decode base | decode branch | ratio |
|---|---|---|---|---|---|---|
| 1k | 3545 | 3566 | 1.006 | 91.7 | 89.6 | 0.978 |
| 4k | 3855 | 3825 | **0.992** | 78.5 | 84.2 | 1.073 |
| 16k | 3563 | 3587 | 1.007 | 74.7 | 75.4 | 1.009 |

Per-run spread at 4k: base 3812 to 3880, branch 3725 to 3913. The 2.2% call-time gain times the GEMM's share of prefill predicts about +1%, below that spread. MTP-on decode has a 5 to 10% run spread; the decode kernels are unchanged, so the MTP-off reading counts (`--spec off`, 2 runs per arm, order A B B A):

| prompt | prefill base | prefill branch | ratio | decode base | decode branch | ratio |
|---|---|---|---|---|---|---|
| 1k | 3564 | 3561 | 0.999 | 54.4 | 54.5 | 1.000 |
| 4k | 3815 | 3761 | 0.986 | 53.7 | 53.7 | 1.000 |
| 16k | 3571 | 3591 | 1.006 | 50.7 | 50.8 | 1.001 |

Gap to Radiance at 4k (6133 tok/s, from the TPO doc): main 1.59x, branch 1.60x.

## Gates

| gate | result |
|---|---|
| prefill at 4k >= +5%, median of >= 6 interleaved rounds | **NOT met**: x0.992 (6 vs 6) |
| prefill at 1k and 16k not slower | met: x1.006, x1.007 |
| decode unchanged (MTP off) | met: x1.000 at 1k / 4k / 16k; MTP on x0.978 / 1.073 / 1.009 inside the run spread |
| output bitwise equal to main | met: 18 prompts x 256 tokens, per-prompt md5 identical for base (MTP on), branch (MTP on) and branch (MTP off) (`raw/w6/texts-*`); GEMM output hash identical to **main's binary** on 48 shape/dtype cases for the default, `LDS=0` and `PD=1` (`raw/w5/gates-1/hash-*.txt`) |
| row-position and M-independence tests | met: `a8_mma_rows_do_not_depend_on_m` and `a8_mma_rows_do_not_depend_on_position` pass (all 6 `a8_` tests) on the default, `LDS=0`, `PD=1`, `PERGROUP=1` and `LDS=0` + `PERGROUP=1` (`raw/w5/gates-1/corr-*`) |
| `kernels test --backend hip --quick` admits the kernel, >= 3 planted mutants caught | met: `matmul_mxfp4_a8_mma` ADMITTED; 84 kernels, 414 cases in 3.4 s, 74 admitted rows, 10 rows without a quick case (`raw/w6/gates-1/adm.log`). 9 of 9 mutants fail the `a8_` tests while the clean build passes: m1 register sets swapped, m2 prologue loads stage 2, m3 grid drops the last M tile, m4 fold shift off by one octave, m5 row 1 scaled by 1.0000002 (only the bitwise tests catch it), m6 one extra stage, and three new ones for the swizzle: m7 second weight chunk stored over the first, m8 weight fragments read the neighbouring chunk, m9 activation chunks stored one chunk off (`raw/w6/gates-1/mut-*.txt`, built by `scripts/ldf/mutants.sh`) |
| batched == solo, spec-on == spec-off | met: `mixids.py` (a 48-token stream B prefilled beside a running stream A) max-batch 16 and 1, MTP off: A `915f1c0953`, B `dd42cb5b78` / `e74f590855` / `537fb01934` in both, equal to the values GLR recorded for main's binary; spec on vs off texts equal (above) |
| `plan show` byte-identical, 8B and 35B on HIP | met: 1346 B and 3184 B, base CLI vs branch CLI (`raw/w6/gates-1/plan-*`) |
| Mac `cargo test --workspace --no-fail-fast` | met: 196 `test result: ok`, exit 0 (`raw/mac-cargo-test.log`); `docs_env_vars` needed the new knob's row in `docs/tuning.md` |
| CI green | see the PR |
| `INTARSIA_*` knob restores the old kernel | met: `INTARSIA_HIP_MXFP4_A8_LDS=0` (hash, correctness and timing arms above run it) |

## Windows (fleetq, tatooine, all took Radiance down and restored it)

| window | job | what |
|---|---|---|
| W-a | #6291 | aborted after 10 s: `w1.sh` died on an unbound `$LDSF` (my edit); nothing measured |
| W1 | #6294 | LDS micro-benchmark (`raw/w1/lds.txt`), main's kernel timing |
| W2 | #6299 | sweep LDV 0..31 (timing, hash, correctness of the best three) |
| W3 | #6304 | micro-benchmark with split-barrier rows, sweep LDV 0..127, 4 interleaved rounds of the best five, clock and power sampling |
| W5 | #6350 | correctness, hashes and call timing of the final kernel (the useful part); then stopped by me: my `ITEMS` list was mangled by zsh (`$B:speed` is a history modifier), no localeval item ran |
| W6 | #6361 | mutants, admission, `plan show`, 12 localeval runs, texts, MTP off, mixids |

CPU jobs (builds, ISA dumps, mutants): #6285 #6286 #6287 #6288 #6289 #6290 #6296 #6297 #6298 #6302 #6303 #6336 #6339 #6342.

## Leftovers and next

* **The prefill gain is not there; the GEMM is not LDS-bound.** The next step is not another LDS layout. The loop with barrier and stores reaches 241 TFLOP/s in isolation and the kernel 166 (call level, same card state): the missing 30% is the global loads, the weight decode (VALU, which shares the matrix unit's pipe on this chip), and the 295 W cap. A design that takes the activation fragments or the weight decode out of the per-stage path (weights as fp8 in memory cost 2x the bytes and a scratch size that changes `plan show`; activation fragments straight from L0 would need 64 KB of L0 per block and stage) is the open question.
* Kernel-only rocprof traces (per-shape kernel us) of main and branch in one window were not taken; the call-level table above stands in.
* The profile counters (LDS bank conflicts, LDS busy) still read 0; `rocprofv3 --att` or `rocprof-compute` was not tried. Without it the swizzle's mechanism rests on the micro-benchmark alone.
* The sweep kernel (128 variants) and the micro-benchmark source are archived in `raw/isa/mxfp4_a8_gemm.sweep.hip` and `raw/scripts/ldsbench.hip` and are not in the shipped kernel (the shipped one adds 4 kernel instantiations: the swizzled 128 x 256 and 128 x 128 tiles for bf16 and f32 output).
* The hash test (`a8_mma_hashes`) and the correctness tests run both layouts through `INTARSIA_HIP_MXFP4_A8_LDS`, one process per value (the kernel reads it once per process).
* Prior art in the decision log: lane l8 (2026-10-25 21:30) found the same swizzled 64 B rows + LDS-only barrier worth +13 to 22% on the dense fp8 GEMM in isolation but under 3% end to end, and lane lb (2026-10-26 00:52) did not ship the LDS-only barrier. This lane's result is the same shape: a real kernel-level gain that does not reach the prefill number.
