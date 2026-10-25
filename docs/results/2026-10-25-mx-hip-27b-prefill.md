# Lane mx: Qwen3.8-27B MXFP4 prefill on one R9700 (HIP, gfx1201), 2026-10-25

Lane mx, branch `lane-mx/27b-prefill`, from origin/main 2c2d9941. Every GPU number comes from a fleetq slot on tatooine with
radiance down (`--takes-down radiance --quiet gpu,cpu`), card 0 only, restored and confirmed `active` after each window
(`raw/downtime.log`). Raw output: `2026-10-25-mx-hip-27b-prefill/raw/<window>/`. Scripts: `scripts/mx/`.

<!-- HEADLINE -->

## 1. Profile: where a 27B 4k prefill goes on one card (main)

`generate --max-tokens 1` under rocprofv3 `--kernel-trace` (raw/w1, job #5008; raw/w2 #5036), classified by
`scripts/mx/roofline27.py` (FLOPs = 2 per MAC; byte floor = every MXFP4 weight read once per chunk). Model facts: 64 layers
= 48 GatedDeltaNet + 16 gated full attention (24 q / 4 kv heads of 256), hidden 5120, MLP 17408, every projection MXFP4 W4A4
(Quark AWQ, activations QDQ'd to MXFP4 per 32-group), 48.7 GFLOP of GEMM per token.

Engine prefill-only time on main (raw/w1, w2):

| prompt | chunk 256 (default) | chunk 4096 |
|---|---:|---:|
| 1k | 1006 ms (1018 tok/s) | 973 ms (1053) |
| 4k | 3984 ms (1028) | 3669 ms (1116) |
| 16k | 16911 ms (969) | 15241 ms (1075) |

Kernel time by class, 4096-token prompt (kernel time 3952 ms at chunk 256, 3553 ms at chunk 4096; launch gaps 3 %):

| class | chunk 256 ms | % | TFLOP/s | chunk 4096 ms | % | TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| GEMM mlp gate / up (N 17408, K 5120) | 1443 | 36.5 | 64.8 | 1387 | 39.0 | 67.4 |
| GEMM mlp down (N 5120, K 17408) | 818 | 20.7 | 57.2 | 799 | 22.5 | 58.5 |
| GEMM gdn in_proj_qkv (N 10240) | 326 | 8.3 | 63.2 | 313 | 8.8 | 65.8 |
| GEMM gdn out_proj + attn o_proj (K 6144) | 301 | 7.6 | 54.7 | 248 | 7.0 | 66.4 |
| GEMM gdn in_proj_a / b (N 48) | **227** | 5.7 | **0.9** | 16 | 0.5 | 12.0 |
| GEMM gdn in_proj_z (N 6144) | 187 | 4.7 | 66.0 | 185 | 5.2 | 66.7 |
| GDN delta rule (sequential scan) | 157 | 4.0 | | 158 | 4.4 | |
| GEMM attn q + gate (N 12288) | 123 | 3.1 | 67.0 | 124 | 3.5 | 66.3 |
| flash attention (bf16 WMMA) | 82 | 2.1 | 40.4 | 60 | 1.7 | 54.6 |
| GEMM attn k / v (N 1024) | 77 | 2.0 | 17.8 | 22 | 0.6 | 63.6 |
| MXFP4 activation QDQ pre-pass | 68 | 1.7 | | 87 | 2.5 | |
| everything else (norms, conv, silu, rope) | 147 | 3.7 | | 154 | 4.3 | |
| **all MXFP4 GEMMs** | **3502** | **88.6** | **57.0** | **3095** | **87.1** | **64.5** |

* **The GEMMs are the prefill**: 87-89 % of kernel time, at 57-65 TFLOP/s. That is 30-34 % of the measured bf16 WMMA issue
  peak (190.7 TFLOP/s) and 15-17 % of the fp8 peak (366). Floors for the 4k prompt: 529 ms at the fp8 peak, 1046 ms at the
  bf16 peak, 20 ms of weight bytes per chunk (12.9 GB read once at 640 GB/s; far from binding at prefill M).
* **Measured ceilings at the same shapes** (`scripts/mx/ceiling27.hip`, raw/w2/ceiling.txt, M = 4096): hipBLASLt bf16
  143-175 TFLOP/s, hipBLASLt fp8 259-321, our own W8A8 `f8_gemm_kernel` (ik_f8_gemm.h) 188-218. The MXFP4 kernel
  (`dq_gemm_kernel<Mxfp4Policy>`, bf16-dequant WMMA, one 128 x 128 tile for every M > 64) is the outlier.
* At chunk 256 two shapes are grid-starved, not compute-bound: in_proj_a / b (N = 48: 2 blocks per launch) at 0.9 TFLOP/s
  and attn k / v (N = 1024: 16 blocks on 64 CUs) at 17.8. Chunk 4096 removes both.
* Attention and the GDN scan are 6 % together at 4k (flash 82 ms, scan 157 ms); at 16k flash attention grows to 1210 ms
  (7.3 %, 43.6 TFLOP/s causal) and the scan to 626 ms (3.8 %). Neither is the lever at 1k-16k; the GEMM is.
* The 35B on the same build does 4k in 410 ms (9981 tok/s): the 27B is a dense model with 13x the active GEMM FLOPs.

### How much of the gap to radiance is the kernel

Radiance 1.12.0 TP2 on this box: 5210 / 6133 / 5852 tok/s at 1k / 4k / 16k (2026-10-25, user). At 48.7 GFLOP/token,
6133 tok/s is 299 TFLOP/s of GEMM across two cards, at least 150 TFLOP/s per card even if radiance spent no time outside the
GEMM. Single card, our GEMMs run at 64.5 (chunk 4096). So of the 5.5x (6133 / 1116) gap at 4k: about 2x is the second card
(TP2), about 2.3x is the GEMM kernel (150 / 64.5), and the rest (~1.2x) is everything outside the GEMM that TP2 also halves
and radiance's fused attention/scan. A tp1 radiance run would make this direct; it was not run (not tried in this lane's windows), so this
split is arithmetic, not a measurement.

## 2. Negative result: the fp8 WMMA cannot reproduce the bf16 chain (carried group scale)

The first idea for a bitwise speed-up: keep today's W4A4 numbers but feed the gfx12 fp8 x fp8 WMMA (2x the bf16 MAC rate)
with the bare e2m1 codes as e4m3 bytes, and carry the accumulator in units of the current 32-group's scale (a power-of-two
rescale at each group boundary, exact). If the matrix core's "sum of 16 products + C" were the same rounding function on
both paths, the outputs would be bitwise today's. It is not:

* `scripts/mx/wmma_eq.hip` on the device (raw/w2/bin-wmma-eq.txt, 2.56M outputs per regime): where the bf16 chain's f32
  sums are exact (cancellation regime) the paths agree; wherever they round, the fp8 WMMA rounds differently. Model-like
  data: 534 outputs round, 164 differ; wide-span data: 2.0M round, 830k differ. The per-group fma control differs more
  (205 / 1.06M), so the carried scale is right and the matrix core's internal accumulation is what differs.
* The kernel built on it (`matmul_mxfp4_mma_f8`, commit 5ffb5d40) failed its own adversarial bitwise test (117547 of
  131072 outputs differ; raw/w2/test-mxfp4-f8.txt) and was also slower: 0.54-0.91x of main in isolation (256 VGPRs, 97
  spills, 296 B/lane scratch from the per-element exponent state) and 5916 vs 3669 ms in the engine. Backed out (c50453f1..).
  The 256-greedy-id matches it had (raw/w2) are not evidence of bitwise equality and are not used as such.

## 3. Bitwise default: `matmul_mxfp4_mma` v2

`kernels/hip/mxfp4.hip` (`Mxfp4PolicyV2`, `mx_gemm_run`). Every output keeps exactly v1's chain: the same bf16 WMMA, k steps
of 16 in order from k = 0, accumulator from zero, the same bf16 operands. Two things change:

1. **Dequant without the LDS table and the f32 round trip.** For a scale byte s in [3, 250], e2m1 * 2^(s-127) is a normal
   number with a 2-bit significand, so its bf16 is exact and is built from bits: magnitude code m >= 2 -> (m << 6) +
   ((s-1) << 7), m = 1 -> (s-1) << 7, m = 0 -> 0, plus the sign. Other scale bytes (2^-127 subnormals, 255 = NaN) take
   v1's path. (v1: 16 LDS table reads, 16 f32 multiplies, 16 bf16 roundings per 16 weights.)
2. **A tile table for prefill M** (v1 used one 128 x 128 tile, 8 waves, for every M > 64): 128 x 256 with 16 waves from 512
   tokens, 128 x 128 with 16 waves below, and a 16-wide weight tile for N <= 64 (gdn in_proj_a / b) so that launch is not
   2 blocks. Small M (verify / tail rows <= 64) keeps v1's tiles.

v1 stays registered as `matmul_mxfp4_mma_v1` (priority -10, never selected): the bitwise reference of the tests, and
`INTARSIA_HIP_MXFP4_V1=1` makes `matmul_mxfp4_mma` run it (kill switch). `INTARSIA_HIP_MXFP4_TILE=WN,WM,TN,TM` forces one
tile (sweeps).

Bitwise evidence:

* `crates/intarsia-backend-hip/tests/mxfp4_v2.rs` (raw/w3, raw/w4 test-mxfp4-v2.txt): v2 == v1 bit for bit on the 27B
  shapes at prefill M through every table tile, partial tiles, N % 8 != 0, both activation modes and output dtypes; on every
  weight scale byte 0..255 with outlier / zero activation groups; rows independent of M (128, 300 vs 2100: batched = solo).
  Negative controls in the test: on its adversarial data the f32 sums do round (f64 exact != output on many elements), and
  one flipped weight code changes the output.
* `scripts/mx/mxbench.hip`: every tile of the sweep is checked against v1 on the 27B shapes (no element differs).
* Engine: self-golden prefill dumps of 31876 rows (P1's 10 golden prompts plus synthetic 260 / 264 / 4089 / 4100 / 5686 /
  16036-token prompts) are bitwise main at chunk 256 and at chunk 4096 (raw/w3 cmp-v2-vs-main-tails.txt), the golden-prompt
  dumps (prefill and teacher-forced decode) are bitwise main, and 256 greedy ids match main (27B short / 4k, 35B short / 4k).

<!-- V2SPEED -->

## 4. Prefill chunk 4096 for the 27B: not bitwise, user decision

Same tails method as lane ck (raw/w3 chunkcmp-base-256-vs-4096.txt, main): chunk 4096 vs 256 is bitwise on every prompt
except those whose length leaves a tail of at most 8 rows at chunk 256 (synth_260: 4 tail rows differ, synth_264: 8), where
the tail runs the M <= 8 GEMV at chunk 256 and the MMA at 4096. Those rows move a lot (max row KL 0.39, max |d logprob|
2.26, 4 argmax flips over the 12 rows), which shows how hard the W4A4 model amplifies last-bit differences through the
activation QDQ (see section 5). Prompts with longer tails (4089, 5686, 16036) and 4100 (a 4-row tail on both schedules) are
bitwise. Speed: +9 % at 4k, +11 % at 16k (section 1). Like CUDA's `[tuning.cuda] prefill_chunk = 4096` (LOG 2026-10-22
13:08, user-approved), this changes numerics on some prompts, so it is left as a user decision; no recipe change here.
`--prefill-chunk 4096` works today as a flag.

## 5. Opt-in: `matmul_mxfp4_mma_fold` (INTARSIA_HIP_MXFP4_FOLD=1), W4A4 folded onto the fp8 WMMA

`kernels/hip/mxfp4.hip` (`launch_fold`). Both operands are folded into e4m3 against their row's largest group scale: a code of
magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} in a group with scale byte s becomes e4m3(mag * 2^t), t = s - ref + 6, exact for t in
[-8, 6] (a row spanning up to 14 octaves; the 27B's weight rows span at most 10, `scripts/mx/scales.py`: 5 of 3.9M rows above
8, none above 14), rounded to nearest even below that (tiny groups of activation rows with a wider span). Then one fp8 GEMM
over the whole K (ik_f8_gemm.h's structure and tile table) and out = acc * 2^(Wn + Am - 266). The products are the same
exact values as v1's; only the accumulation (fp8 WMMA order and rounding) and those tiny groups differ. This is closer to the
checkpoint's W4A4 semantics than radiance's W4A8 (per-token fp8 activations without the MXFP4 activation QDQ). W4A16 calls
and K % 64 != 0 run v2 inside (bitwise, tested). Rows do not depend on M (tested): batched = solo with the knob on.

Accuracy against P1's 27B golden (HF transformers + Quark W4A4, top-20, with the ref-noise sidecar; `accuracy --mode both`,
raw/w3 accg-*.out):

| | prefill top-1 | prefill KL21 (scored) | prefill gate (<= ref noise 0.197) | decode top-1 | decode KL21 (scored) | decode gate |
|---|---:|---:|---|---:|---:|---|
| main | 91.33 % | 0.1799 | PASS | 88.75 % | 0.3397 | FAIL |
| v2 (default) | 91.33 % | 0.1799 | PASS (bitwise main) | 88.75 % | 0.3397 | FAIL (bitwise main) |
| fold (opt-in) | 91.95 % | 0.1653 | PASS | 88.12 % | 0.2606 | FAIL |

* The fold is as close to the reference as main or closer on every row of this table, but it **does not pass the decode-mode
  gate, and neither does main** (0.3397 vs the 0.197 bound; it is main's state, not this lane's). No threshold was changed.
* Fold vs main directly (self-golden, raw/w3 cmp-golden-*): mean row KL 0.174 prefill, 0.198 decode, 115 of 1441 argmax
  flips. That is the size of the reference noise itself (HF W4A4 bf16 vs fp32: 0.197), i.e. this W4A4 model turns last-bit
  sum differences into ref-noise-sized logit changes (the chunk-tail rows of section 4 show the same on main).

<!-- FOLDSPEED -->

<!-- REST -->
