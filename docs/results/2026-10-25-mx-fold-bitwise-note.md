# Lane mx: why the MXFP4 fold changes numerics, and the bitwise options (note, 2026-10-25)

Scope: analysis only, no code and no GPU time. Every number below is a measurement from PR #144
(`docs/results/2026-10-25-mx-hip-27b-prefill.md` on branch `lane-mx/27b-prefill`, raw under its `raw/`) or from #150
(`docs/results/2026-10-25-ac-27b-accuracy-gate.md`), unless it is marked as an estimate.

## 1. What the fold changes

The default path (`matmul_mxfp4_mma` v1 / v2) dequantizes each weight to bf16 (e2m1 x 2^(s-127), exact in bf16) and
feeds the bf16 WMMA with the MXFP4-QDQ'd activations (also exact in bf16). Every output is one chain: k steps of 16 in
order from k = 0, one f32 accumulator starting at zero, `v_wmma_f32_16x16x16_bf16`.

`INTARSIA_HIP_MXFP4_FOLD=1` (`matmul_mxfp4_mma_fold`, `launch_fold`) re-expresses both operands as e4m3 against the row's
largest group scale and runs `v_wmma_f32_16x16x16_fp8_fp8` over the whole K, then multiplies by one power of two. The
per-element products are the same exact values as the default's. Two things differ:

1. **The matrix core's accumulation (the main cause).** Both paths sum exact products into f32, but the fp8 WMMA rounds
   its "16 products + C" differently from the bf16 WMMA. This was measured directly, not inferred (#144 section 2,
   `scripts/mx/wmma_eq.hip`, 2.56M outputs per regime). With identical exact products and the same k order, the two
   instructions agree only where the f32 sums are exact. On model-like data 164 of the 534 outputs that round differ; on
   wide-span data, 830k of 2.0M. The power-of-two rescale is exact and is not the cause.
2. **Tiny activation groups (secondary).** A group more than 8 octaves below its row's largest scale (t < -8) is rounded
   to nearest even in e4m3 instead of kept exact. No 27B weight row needs this (`scripts/mx/scales.py`: none spans more
   than 14 octaves). Only activation rows with a wide span do.

The effect on the 27B is the size of the reference noise. Fold vs main: mean row KL 0.174 on prefill and 0.198 on decode,
with 115 of 1441 argmax flips. Against the HF W4A4 golden the fold is closer than main: prefill KL 0.165 vs 0.180,
decode-mode 0.261 vs 0.340 (#144 section 5).

## 2. Can the fold match the current order? No, not on the fp8 WMMA

A fold that is bitwise the default would have to reproduce the bf16 WMMA's rounding of each 16-product step. The ISA does
not let us choose an instruction's internal summation, and #144 section 2 tested the two workarounds:

- **A carried group scale on the fp8 WMMA** (`matmul_mxfp4_mma_f8`) failed its bitwise test: 117547 of 131072 outputs
  differ. It was also slower: 0.54-0.91x of main, with 256 VGPRs and spills.
- **A per-group fma control** (per-step partial sums added in f32) differs more still: 205 and 1.06M differing outputs in
  the two regimes, because it adds a second rounding.

The only bitwise way to fold's speed is therefore a faster **bf16** kernel with the same per-output WMMA chain (option A
below). It cannot be a variant of the fold itself.

## 3. Options

| | A. bitwise bf16 pipeline (default bits kept) | B. default adopts the fold's order (main's bits change once) | C. status quo (#144) |
|---|---|---|---|
| what | new `matmul_mxfp4_mma` pipeline: BK 64 / 128, both operands packed 4-bit in LDS, exact bf16 fragments rebuilt in registers, same k order per output (#144 section 8) | fold becomes the default for W4A4 MXFP4 prefill on HIP (M >= 9; the row GEMVs keep M <= 8), with v2 as the kill switch | v2 default, fold opt-in |
| bits | equal to main | 27B W4A4 prefill bits change; W4A16 path, decode GEMVs, 35B, CUDA and Metal unchanged | equal to main |
| GEMM rate at M = 4096 | v2 now 83 TFLOP/s. Ceilings: bf16 WMMA issue peak 190.7, hipBLASLt bf16 143-175 at these shapes | fold 156.6 TFLOP/s measured. Ceiling: fp8 peak 366, hipBLASLt fp8 259-321 | v2 83 |
| 27B prefill, chunk 4096 + peel 16 (served, 1k / 4k / 16k) | **estimate**, 4k only: 1790 / 2020 / 2230 tok/s if the GEMM reaches 110 / 130 / 150 TFLOP/s (74 / 83 / 92 % of the fold) | **measured** 2043 / 2416 / 2186 (loc3) | 1323 / 1437 / 1351 (opt-in) |
| 27B prefill, default chunk 256 (served) | between v2's 1154 and the fold's 1895 at 4k; the M = 256 tiles cap it, not estimated further | **measured** 1812 / 1895 / 1695 (loc1) | 1120 / 1154 / 1073 |
| user decision needed | no | yes: main's bits change, and the 27B W4A4 baseline is re-recorded | no |

How the A estimate was made: at 4k with chunk 4096 the v2 engine spends 2261 ms in the GEMM and about 585 ms in
everything else (wall 2846 ms, #144 raw/w4). The estimate scales only the GEMM time by 83 / R. The estimate is
unmeasured. The bench-only `nodq` arm (97 TFLOP/s) shows the dequant is not the limit; the tile pipeline is.

**What B costs beyond the speed:**

- **The W4A16 golden does not see the fold.** With `--mxfp4-act none` the fold runs v2 inside (tested bitwise), so #150's
  correctness gate stays bitwise and passes whatever the fold does. B's quality evidence must therefore come from the
  W4A4 golden:
  - Prefill: 0.165, PASS at 0.83x the scored noise.
  - Decode: the engine-baseline regression gate is tight by design (#150 section 6: "any change in what the engine
    computes on the 27B moves it"), so the fold will move it. The decision then rests on the fold's KL against the HF
    W4A4 golden (0.261 vs main's 0.340), after which the baseline and envelope are re-recorded from the fold build.
    That is the step the user approves.
- **B rules out A as a bitwise follow-up.** Once the fold is the default, no bf16 kernel can match it. Later speed work
  must keep the fp8 WMMA chain (the same k order on every tile). That leaves 2.3x of headroom (156.6 against the 366 fp8
  peak), against A's 2.3x on bf16 (83 against 190.7).
- **Row invariance.** It is tested at M = 9, 128, 300 and 2100. B needs it at every pass size 9..4096 across the fold's
  tile table, so that chunk 4096 + peel stays bitwise chunk 256 and batched stays equal to solo.

## 4. Hours and windows

| option | agent hours | tatooine windows (radiance down) | wall clock incl. queue (today's queue: 1-5 h per slot) |
|---|---|---|---|
| A | 15-25 (kernel, tests with negative controls, `mxbench` sweeps, gates, results) | 4-6 microbench windows of about 20 min, 1 gates window of about 35 min (no-op, tails sweep, kernels quick, batch tests), 1 localeval window of about 40 min | 2-4 days; the risk is the reached rate (110 vs 150 TFLOP/s) |
| B | 4-6 (dispatch default plus kill switch, wider row-invariance test, LOG row, results) | 1 gates window of about 35 min: W4A4 golden prefill and decode vs HF and vs the current baseline, W4A16 `--mxfp4-act none` bitwise, acc-short 27B, 27B batched == solo, tails sweep under the fold, kernels quick, 35B no-op. The re-record (#5329 took 3 min) fits in the same window once the user has approved it. 1 localeval window of about 40 min | about 1 day, gated on the user decision |
| C | 0 | 0 | done (#144) |

Recommendation: decide B first, because it is the user's call and it fixes which unit future work targets. If the user
keeps main's bits, A is the route to most of the fold's speed. It is unproven: the 110-150 TFLOP/s range above is a
target, not a measurement.
