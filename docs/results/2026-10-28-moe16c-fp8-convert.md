# moe16c: mixed-precision fma (FHFMA) in the CUDA fp8 W8A8 and nvfp4 row kernels: no f16 / bf16 -> f32 unpack, bitwise the same

Lane moe16c, 2026-10-28, branch `lane-moe16c/fp8-convert` from `origin/main` 5f095f411 (#229 merged, side-stream knobs on by
default). spark1 (spark-4e77, GB10 sm_121), Qwen3.6-35B-A3B NVFP4, `serve --backend cuda --max-batch 16 --max-seq 8192 --mem-budget 40`.

**Status: draft PR #230. Not merged: the speed gates are not met. The coordinator decides.**

Speed, serve A/B, 3 interleaved rounds (median of the per-round ratios):

| streams | change | gate |
|---|---:|---|
| 16 | +1.1 % | +3 %: **fail** |
| 8 | +0.6 % | not slower: pass |
| 4 | -0.9 % | none set; the fallback needs positive everywhere: **fail**, though within noise |
| 1 | +0.8 % | none set |

Every exactness gate passes:
- Served greedy tokens identical to main: 21/21 cells, 9408 ids.
- `batch_cuda_real` on the 35B: 4290 pairs bitwise, plus the engine test.
- `rows_exact`: 9 tests, every form, M = 1..16.
- `plan show` is byte-identical for 8B, 35B and Gemma-4.
- An exhaustive FHFMA-vs-fmaf bit check: 0 of 18.8 M results differ.
- Mac `cargo test --workspace --release` passes.
- CI is green.

Kernel effect (section 3): at 16 rows, fp8 calls are 4.6 to 12.7 % faster and the lm_head is 19 % faster.

GPU windows used: **1 of this lane's 2 here** (the other went to Part A, the #229 defaults check): #6677. Raw:
`2026-10-28-moe16c-fp8-convert/raw/`. Scripts: `scripts/moe16c/`. `le-window.sh`, `ident.sh`, `batch-real.sh`, `plan.sh` and
`agg.py` are copies of `scripts/moe16b/` retargeted to this lane's folders.

## 1. What changed

MOE16b (#229 section 5) found that the fp8 rows GEMV at M = 16 is instruction-bound, not memory-bound:
- per 16 weight bytes per lane, 512 FFMA against 288 HADD2 and 144 F2FP;
- the HADD2 and F2FP are the e4m3 -> f16 -> f32 unpack of the 16 staged activation rows, redone for every 2-output tile.

**The instruction.** On sm_100 and later, PTX 8.6 has `fma.rn.f32.f16` and `fma.rn.f32.bf16`. They multiply two f16 (or bf16)
operands and add an f32, with one rounding. On sm_121, ptxas emits a single `FHFMA` for this, and the low or high half of a packed
register can be selected (`R0.H1`; `raw/probe/p.out`).

**Why it is exact.** An f16 or bf16 value is exact in f32. So `fma.rn.f32.f16(a, b, c)` is bitwise `fmaf(float(a), float(b), c)`.
The widening to f32 (HADD2.F32 for f16, a shift or PRMT per bf16 element) folds into the multiply. The accumulation order, the
f32 products and every rounding stay the same.

**Checked exhaustively on the GPU** (`raw/probe/q.cu`, `raw/probe/q.out`, fleetq #6668). FHFMA was compared against fmaf over:
- every e4m3 x e4m3 pair as f16, both singly and as a two-step chain;
- every bf16 x signed e2m1 pair as bf16;
- each with 16 addends: ±0, subnormals, ±1, the largest finite values, infinities, and values that cancel.

Result: **0 of 18,841,728 results differ.**

**The kernels** (`kernels/cuda/ik_quant.cuh`: `ik_fma2_f16`, `ik_fma2_bf16`, `ik_fp8x16_to_h2`, `ik_fp4x8_to_bf2`):
- **fp8 W8A8 rows kernel** (`rows_q_tiles`, XQ: `fp8_gemv_rows_q_kernel`). The staged e4m3 codes and the e4m3 weights are
  converted to f16 pairs only, two per F2FP. Each pair of the 16-long chain is two FHFMA, in element order.
  - The W8A16 (bf16-staged) branch is unchanged. No 35B projection takes it.
- **nvfp4 rows kernels** (`nvfp4_tile_body`, so also the row-group split kernel, and `nvfp4_gemv_full_kernel`, the lm_head path).
  - The bf16 activations go into `fma.rn.f32.bf16` as they are, two per 32-bit word: no unpack.
  - The fp4 weights go straight to bf16 pairs through two `prmt` look-ups (high and low bytes of the 8 magnitudes), plus the
    sign and two byte interleaves. Before, they went through e4m3, f16 and then f32.
  - The block-scale fma (`acc = fmaf(s, sc, acc)`) is unchanged.
- **Unchanged:** the one-row (M = 1) kernels, which `rows_exact` compares every row kernel against, and the bf16-staged fp8
  kernels (`fp8_gemv_rows_full_kernel`, the chunked kernel).
  - At K = 2048, the fp8 launcher takes the bf16-staged full kernel below 12 rows, except at small N (k / v, 512 x 2048, from 4
    rows). So at 1 to 8 streams the fp8 GDN qkv / z, attention q and o_proj calls are unchanged. The change reaches them only from
    12 rows.
  - The nvfp4 split / tile path (shared expert) changes from 2 rows, and the lm_head full kernel from 5 rows.
- **Fallback:** on any arch below sm_100 the helpers run the old f32 fmaf on the same images.

**Not done:**
- The routed-expert decode kernels (`moe_decode.cu`) have the same unpack pattern. #226 measured them at 84 to 88 % of
  230 GB/s, so they are bandwidth-bound and have little to win.
- Staging the fp8 activations as f16 (no F2FP for x at all) was not tried. It would double the staged bytes (M x K x 2) and halve
  the blocks per SM at K = 2048. It is the next step for the remaining fp8 instructions (section 2).

## 2. SASS, static per-function counts (`cuobjdump -sass`, `scripts/moe16c/sass.sh`, `raw/micro/sass-b2.log`, `raw/micro/sass-final.log`, `raw/build-final.log`)

The instantiations the 35B takes at 16 rows:

| kernel (35B use) | total base -> dev | FFMA | FHFMA | HADD2 | F2FP | SHF | PRMT |
|---|---|---|---|---|---|---|---|
| fp8 `rows_q<16, 2x8, 2 blocks/SM, bf16>` (GDN qkv / z, attn q, o_proj, GDN out) | **8080 -> 7192 (-11 %)** | 2396 -> 860 | 0 -> 1536 | 960 -> 96 | 590 -> 590 | 429 -> 430 | 210 -> 217 |
| fp8 `rows_q<16, 1x16, 1 block/SM, bf16>` (k / v, small N) | 5448 -> 4616 (-15 %) | 1244 -> 476 | 0 -> 768 | 864 -> 48 | 473 -> 473 | 425 -> 332 | 200 -> 200 |
| nvfp4 `full<16, 4x8, f32>` (lm_head at 12+ rows, the launcher's default) | **8992 -> 7768 (-14 %)** | 3264 -> 192 | 0 -> 3072 | 204 -> 12 | 108 -> 12 | 971 -> 201 | 432 -> 192 |
| nvfp4 `full<16, 2x16, f32>` (lm_head, forced `2x16`) | 5368 -> 4096 (-24 %) | 1632 -> 96 | 0 -> 1536 | 102 -> 6 | 54 -> 6 | 859 -> 107 | 408 -> 96 |
| nvfp4 `tile<16, 4 outputs, bf16>` | **6576 -> 5328 (-19 %)** | 3264 -> 192 | 0 -> 3072 | 204 -> 12 | 172 -> 76 | 970 -> 202 | 432 -> 192 |
| nvfp4 `split<8 rows, 1 output, bf16>` (shared expert at 16 rows) | **12632 -> 7960 (-37 %)** | 3060 -> 180 | 0 -> 2880 | 680 -> 40 | 396 -> 76 | 1896 -> 665 | 1604 -> 641 |

**Inner loop of the fp8 kernel.** Per 16 weight bytes per lane (one `u` step, M = 16, 2 outputs):
- base: 512 FFMA + 288 HADD2 + 144 F2FP + 16 LDS;
- dev: 512 FHFMA + 144 F2FP + 16 LDS, with no HADD2.

That is about 30 % fewer instructions in the loop. The whole-function total drops only 11 %, for two reasons:
- The function also holds the W8A8 prologue (re-quantising x with `__fdiv_rn`), which is the remaining FFMA, plus the unrolled
  copies and the epilogue.
- The 144 F2FP (e4m3 -> f16, two per instruction) are still there. Only the f16 -> f32 widening folded into the FHFMA.

**nvfp4.** In nvfp4 the gain is larger: the bf16 activation unpack (one SHF / PRMT per element and row) disappears entirely, and
the weight decode is shorter.

## 3. Micro-timing, evicted (`timing_rows_evict`, `scripts/moe16c/micro.sh`)

Setup: each launch reads a different copy of the weights, from a pool of about 192 MB, far past the L2. Both binaries are the
`rows_exact` test binary: base = main after #229, dev = this branch. The rows run interleaved. The "unset" form is the launcher's
default policy.

### fp8 (fleetq #6673, 3 runs per binary: base / dev / dev12 / dev12 / dev / base / base / dev / dev12; medians; `raw/micro/m2-evict-dev12.log`)

The M < 12 rows of the three large-N shapes run the unchanged bf16-staged kernel in both binaries (section 1). They show the
noise: ±2 to 5 %.

| shape (35B use) | M=2 | M=4 | M=8 | M=12 | M=16 |
|---|---:|---:|---:|---:|---:|
| 8192x2048 (GDN qkv, attn q), us base -> dev | 76.05 -> 74.68 (-1.8 %, unchanged kernel) | 77.17 -> 79.18 (+2.6 %, unchanged) | 81.33 -> 78.33 (-3.7 %, unchanged) | 84.34 -> 83.73 (-0.7 %) | **87.24 -> 83.21 (-4.6 %)** |
| 4096x2048 (GDN z) | 39.63 -> 37.91 (unchanged) | 41.79 -> 41.74 (unchanged) | 42.43 -> 41.31 (unchanged) | 45.02 -> 44.59 (-1.0 %) | **51.76 -> 49.20 (-4.9 %)** |
| 2048x4096 (GDN out, o_proj) | 40.11 -> 38.63 (unchanged) | 41.39 -> 40.33 (unchanged) | 45.74 -> 48.17 (unchanged) | 52.15 -> 49.01 (-6.0 %) | **57.49 -> 51.09 (-11.1 %)** |
| 512x2048 (k, v) | 8.33 -> 8.71 (unchanged) | 9.93 -> 8.91 (-10.3 %) | 10.67 -> 10.26 (-3.8 %) | 13.29 -> 12.29 (-7.5 %) | **16.55 -> 14.44 (-12.7 %)** |

`dev12` was a temporary variant that used FHFMA only from 12 rows. It was no better than dev where the kernels differ (M = 4 / 8
at 512x2048: +0.5 / +0.4 % against base, where dev gained -10.3 / -3.8 %), so the branch keeps FHFMA at every M.

### nvfp4 (fleetq #6671, 2 runs per binary: base / dev / dev / base; means; `raw/micro/m1-exact-evict.log`)

| shape (35B use) | M=2 | M=4 | M=8 | M=12 | M=16 |
|---|---:|---:|---:|---:|---:|
| 248320x2048 (lm_head, f32 logits), us | 1202 -> 1197 (-0.4 %) | 1222 -> 1198 (-1.9 %) | 1292 -> 1290 (-0.2 %) | 1512 -> 1335 (**-11.7 %**) | 1928 -> 1559 (**-19.2 %**) |
| 512x2048 (shared expert gate, up) | 9.04 -> 8.08 | 7.33 -> 8.16 | 8.82 -> 6.37 | 8.82 -> 8.89 | **10.93 -> 9.07 (-17.0 %)** |
| 2048x512 (shared expert down) | 6.09 -> 6.09 | 7.55 -> 6.12 (-19.0 %) | 9.51 -> 8.17 (-14.1 %) | 10.25 -> 9.14 (-10.9 %) | **13.44 -> 12.29 (-8.5 %)** |

**Estimated per-step saving at B = 16.** Per-call deltas times the call counts of the #224 / #229 profile:
- fp8: about 0.55 ms;
- lm_head: about 0.37 ms;
- shared expert: about 0.1 ms, partly hidden on the side stream.

That is about 1 ms of a 46 ms step (about 2 %). After the roughly 11 % prefill share of the burst, it projects to about +1.5 to
+2 % aggregate at 16 streams.

## 4. Serve A/B (`scripts/moe16c/w1.sh`, fleetq #6677, `raw/w1/`)

Setup:
- localeval `--sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16 --thinking off`, `ignore_eos`;
- 3 interleaved rounds, with even rounds reversed, and a fresh server per arm;
- base: `origin/main` 5f095f411 (after #229, side streams on); dev: this branch. Both arms run `--max-batch 16`.

Aggregate tok/s:

| streams | base (r1, r2, r3) | dev (r1, r2, r3) | dev vs base, median of per-round ratios (rounds) | ratio of medians |
|---|---|---|---|---:|
| 1 | 79.7, 79.5, 78.4 (79.5) | 79.9, 80.1, 79.5 (79.9) | **+0.8 %** (+0.3 / +0.8 / +1.4) | +0.5 % |
| 4 | 171.8, 168.9, 172.3 (171.8) | 170.2, 169.9, 170.4 (170.2) | **-0.9 %** (-0.9 / +0.6 / -1.1) | -0.9 % |
| 8 | 223.7, 219.6, 222.0 (222.0) | 219.6, 224.9, 223.3 (223.3) | **+0.6 %** (-1.8 / +2.4 / +0.6) | +0.6 % |
| 16 | 270.6, 275.8, 273.6 (273.6) | 277.4, 277.1, 276.7 (277.1) | **+1.1 %** (+2.5 / +0.5 / +1.1) | +1.3 % |

Other figures:
- Single stream, 1k prompt: decode 84.5 -> 84.8 tok/s (x1.005), prefill 5271 -> 5233 tok/s (x0.993; dev's rounds 5233 / 5386 / 4991
  spread far more than base's).
- 16-stream TTFT: 1.67 s in both arms.

**Reading:**
- *16 streams.* +1.1 to +1.3 % is at the low end of the +1.5 to +2 % projected from the micro (section 3).
- *4 and 8 streams.* At 4 / 8 rows, the fp8 GDN / attention projections take the unchanged bf16-staged kernel (section 1). Only
  k / v, the shared expert and (at 8) the lm_head change, worth well under 0.5 % of a step. The per-round spread here is ±1 to
  2 %, so -0.9 % at 4 and +0.6 % at 8 are noise around zero, not a measured regression.
- *1 stream.* The 1-row calls take the unchanged one-row kernels, so +0.8 % is noise as well.
- *vLLM at 16 streams* (340.1 tok/s, vr16 window 10-27): 0.80x on base, 0.81x on dev.

## 5. Gates

| gate | result |
|---|---|
| +3 % at 16 streams and 8 not slower; or smaller but exact and positive everywhere (3 rounds) | **fail**: 16 +1.1 %; 8 +0.6 % (not slower); 4 -0.9 % (within noise, but not positive) |
| served greedy tokens identical to main (21-cell set) | **pass**: 21/21 cells, 9408 ids (`raw/ident/compare.log`) |
| `batch_cuda_real` (batched equals solo) on the 35B | **pass**: 4290 (row, step) pairs bitwise, plus the engine test (`raw/batch-real.txt`) |
| `rows_exact` | **pass**: 9 tests (`raw/rows-exact.txt`; also #6671 in `raw/micro/m1-exact-evict.log`) |
| `plan show` byte-identical, 8B / 35B / Gemma-4 | **pass**: json and txt (`raw/build-final.log`) |
| FHFMA = fmaf, exhaustive over the operands the kernels feed | **pass**: 0 of 18,841,728 differ (`raw/probe/q.out`) |
| Mac `cargo test --workspace --release` | **pass**: exit 0, 0 failed (`raw/mac-cargo-test.log`) |
| CI | **pass** (PR #230) |

**Options for the coordinator.** The change is bitwise exact and makes the 16-row kernels faster. In serve it gives +1.1 to
+1.3 % at 16 streams, and it is neutral within noise at 1 / 4 / 8 streams, where most of the changed kernels do not run.
- Merging it is the #229 decision with a smaller margin.
- Holding it costs nothing, because it carries no knob.
- The next exact step on the same kernels is f16 staging of the fp8 activations (section 1, Not done), which removes the
  remaining 144 F2FP per step of the loop.

## 6. Process

GPU windows: **1** for Part B (#6677, `--quiet gpu,cpu --mem 50`, 22:51 to 23:08, 17 min 31 s). It ran `rows_exact`, the identity
check, `batch_cuda_real` and the 3-round A/B. Part A's defaults check (#6660, 5 min 20 s) was the other window of the lane's cap of 2.

Micro jobs, each under 3 min (not windows):
- #6659 (`--load cpu`): the nvcc probe of `fma.rn.f32.f16`.
- #6668 (10 s): the exhaustive FHFMA bit check.
- #6671 (1 min 40 s): `rows_exact` on dev, plus the evicted timing base / dev / dev / base.
- #6673 (31 s): fp8 evicted timing, base / dev / dev12 x 3.

`--load cpu` builds and SASS jobs: #6669, #6670, #6672, #6674 (CLI base + dev, `rows_exact`, `batch_cuda_real`, SASS,
`plan show`) and the final SASS job.

Notes:
- spark1-vllm stayed up throughout, with no `--takes-down`.
- spark2 was not touched.
- The spark1 work folder `~/intarsiaml-moe16c` was deleted at the end.
- #6669 built a stale dev binary, because rsync kept the Mac's older mtimes. Its SASS showed dev = base. `build.sh` now touches the
  sources after syncing, and #6670 has the real counts.
