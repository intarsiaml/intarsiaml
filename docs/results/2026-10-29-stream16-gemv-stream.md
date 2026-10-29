# stream16: weight streaming of the 16-row fp8 and lm_head GEMVs: +7 to +18 % kernel GB/s, -0.16 to -0.27 ms per B = 16 step, bitwise

Lane stream16, 2026-10-29, branch `lane-stream16/fp8-stream` (base main 20f4b3f42 for every A/B below; branch merged up to
492a8b35e, which adds only Metal changes). spark1 (spark-4e77, GB10 sm_121), `nvidia/Qwen3.6-35B-A3B-NVFP4`.
**Status: draft PR. Not merged.** The kernels are faster, but the step gain misses the lane's ready-PR bar (>= 0.4 ms at B = 16
or >= 2 % single-stream), and single-stream decode cannot move: it runs M = 1 kernels this lane does not touch.

## Result

Two streaming variants, both default on, both bitwise main (the same per-(m, n) u order, fma chains, warp tree and scale):

- `fp8_gemv_rows_qp_kernel` / `_qp_pair_kernel` (`kernels/cuda/fp8.cu`; `INTARSIA_CUDA_FP8_ROWS_QPIPE=0` = main's kernel).
  This is the e4m3-staged rows kernel at M >= 12, N > 1024, K = 2048 / 4096. K is a template, so the warp's next chunk of u
  blocks is loaded before the current chunk's math, across tile boundaries too. **Before the block quantises x, every warp
  bulk-prefetches its first tile into L2** (`cp.async.bulk.prefetch.L2`), so DRAM streams during the prologue. Shapes:
  K = 2048 `2x16x1` (outputs per warp x warps x blocks/SM), K = 4096 `1x16x1` with 2-block chunks and an L2 prefetch 4
  strides ahead.
- `nvfp4_gemv_pipe_kernel` (`kernels/cuda/nvfp4.cu`; `INTARSIA_CUDA_NVFP4_ROWS_PIPE=0` = main's). This is the lm_head
  persistent kernel at M >= 12, K = 2048, with shape 2x8. Each warp double-buffers its whole next tile (2 rows x 4 u blocks)
  in registers.

### Per-kernel table (nsys, served 16-stream burst, decode window; `raw/base-b16`, `raw/dev-b16`)

| kernel (calls / step) | MB / call | main us | main GB/s | dev us | dev GB/s | main ms recoverable at 230 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| GDN qkv + z pair, 12288 x 2048 (30) | 25.2 | 132.3 | 190 | 127.9 | 197 | 0.69 |
| GDN out + o_proj 2048 x 4096 (40) and attention q 8192 x 2048 (10), one bucket | 503 / step | 3.30 ms | 152 | 3.05 ms | 165 | 1.11 |
| lm_head NVFP4 248320 x 2048 (1) | 286.1 | 1530.5 | 187 | 1298.5 | **220** | 0.29 |
| k + v pair 2 x 512 x 2048 (10, side stream, unchanged) | 2.1 | 28.4 | 74 | 32.3 | 65 | (0.19, hidden) |
| sum of the three changed rows (ms / step) | | 6.80 | | **6.19** | | |

The profiles come from two separate windows (W1 13:22, W3 14:05). Kernels the lane did not touch moved between them too
(routed MoE gate_up 305.6 -> 323.3 us, down 158.9 -> 166.1). The decode-window mean was 41.46 ms (main, W1) and 42.33 ms
(dev, W3). That is cross-window drift, not this change. The interleaved step A/B below is the measurement.

B = 1 (`raw/base-b1/rank.txt`): the M = 1 kernels already stream at about 230 to 245 GB/s (`fp8_gemv_pair_kernel`, the lm_head
`nvfp4_gemv_kernel` at 1165 us, 245 GB/s in micro). No GEMV lever exists at B = 1, so the single-stream bar is out of reach for
this lane.

### Micro (evicted pool, 16 rows; `raw/micro-evict-{base,dev}.txt`, `raw/probe-w{1,2}.txt`)

| shape | main us (GB/s) | dev us (GB/s) | read-only ceiling (probe) |
|---|---:|---:|---:|
| fp8 8192 x 2048 | 82.4 (204) | 79.8 (210) | 72.5 (231) |
| fp8 4096 x 2048 | 46.5 (180) | 42.1 (199) | 36.6 (229) |
| fp8 12288 x 2048 (probe) | 115.6 (218) | 109.5 (230) | 103.0 (244) |
| fp8 2048 x 4096 | 51.1 (164) | 46.1 (182) | 37.0 (227) |
| lm_head 248320 x 2048 | 1592 (180) | 1414 (202) | |
| lm_head at 12 rows (probe) | 1381 (207) | 1337 (214) | |

**What the probe showed** (`scripts/stream16/probe.cu`: copies of main's kernels plus variants, memcmp-checked):

- The fp8 gap to the read-only ceiling is a roughly fixed 9 to 15 us per launch, not a rate. It comes from the prologue: every
  block quantises 16 x K activations with `__fdiv_rn`, and no weight loads are in flight meanwhile. A diagnostic variant that
  reads pre-quantised codes (`noq1`) runs 2048 x 4096 in 41.2 us against main's 51.6.
- One prefetched tile per warp before the prologue recovers most of that: 43.8 us. Two or four tiles, or register-preloading
  the first chunk across the barrier (`er1`), are slower.
- Deeper register chunks or steady-state L2 prefetch at K = 2048 do not help (ch2 / ch4 / pf2..8 are 2 to 20 % slower).
- The lm_head at 16 rows is issue-bound (1165 us at M = 1, 1281 at 8, 1580 at 16). A whole-tile register double buffer at
  2x8 is the only shape that helps; 1x8, 1x16, 3x8, 2x12 and 4x12 are slower.
- More threads per block (1x32) and contiguous per-block tile ranges are slower.

## Step and serve A/B (main 20f4b3f42 vs dev)

`timing_35b_batched_step`, 7 interleaved runs (base, dev, base, dev, dev-off, dev, base), min / median ms (`raw/steptime-w2.txt`;
the W2 dev binary had EP = 2, the final default is EP = 1, which the probe puts 0 to 1.5 us per call faster):

| B | main (3 runs) min | dev (3 runs) min | delta min | main median | dev median | delta median | dev, knobs off |
|---|---|---|---:|---|---|---:|---|
| 1 | 11.65 / 11.57 / 11.58 | 11.58 / 11.58 / 11.57 | -0.02 (no kernel changed) | 11.69 / 11.56 / 11.59 | 11.55 / 11.57 / 11.56 | -0.05 | 11.54 |
| 4 | 18.74 / 18.71 / 18.73 | 18.67 / 18.75 / 18.71 | -0.02 (no kernel changed) | 19.26 / 19.20 / 19.21 | 19.18 / 19.33 / 19.21 | 0.0 | 19.20 |
| 8 | 25.94 / 25.94 / 26.00 | 25.86 / 25.92 / 25.95 | -0.05 (no kernel changed) | 27.08 / 27.24 / 27.23 | 27.02 / 27.13 / 27.10 | -0.10 | 27.14 |
| 16 | 41.22 / 41.15 / 41.39 | 40.98 / 41.14 / 41.14 | **-0.16** | 42.80 / 42.59 / 42.84 | 42.50 / 42.40 / 42.50 | **-0.27** | 41.36 / 42.72 |

Identical-arm noise at B = 16 is 0.24 ms (min) and 0.25 ms (median). The dev gain is just above noise and below the 0.4 ms bar.
The kernels save about 0.6 ms of main-stream GEMV time, and the step shows about half of that. Part of it was already
overlapped by side-stream work.

Serve (localeval, 3 interleaved rounds, 1k prompt, 256 tokens, aggregate decode tok/s; `raw/w3-conc`, `raw/w3-single`):

| cell | main r1 / r2 / r3 | dev r1 / r2 / r3 | change |
|---|---|---|---:|
| 1 stream | 80.7 / 80.7 / 81.0 | 80.5 / 80.4 / 80.4 | -0.5 % |
| 4 streams | 178.5 / 177.6 / 178.3 | 175.1 / 178.5 / 176.3 | -0.8 % |
| 8 streams | 235.7 / 237.8 / 234.5 | 234.8 / 232.9 / 233.2 | -1.0 % |
| 16 streams | 304.0 / 298.8 / 309.5 | 309.0 / 304.2 / 311.9 | +1.4 % |
| single stream 1k | 85.7 / 84.9 / 86.6 | 84.9 / 85.3 / 85.7 | -0.5 % |
| single stream 4k | 82.6 / 82.6 / 83.9 | 81.2 / 83.7 / 82.7 | -0.6 % |
| single stream 16k | 74.8 / 74.7 / 75.2 | 74.2 / 75.5 / 74.9 | 0.0 % |

At 1 to 8 streams and single-stream, dev runs exactly main's kernels: every new path needs M >= 12, and B = 8 has at most
8 rows. The -0.5 to -1.0 % there is run-to-run spread; main's own spread is 0.4 to 2.1 %. Read literally, though, the
4- and 8-stream cells fail the "no cell slower than -0.5 %" bar. 16 streams is +1.4 % with 3.5 % main spread.

## Gates

- rows_exact: the new `fp8_rows_qpipe_are_bitwise_the_one_row_gemv` covers every compiled shape and PF / EP mix,
  W8A8 / W8A16, single and pair, and ragged N. The new `nvfp4_rows_pipe_are_bitwise_the_one_row_gemv` covers the lm_head and
  a ragged N, `auto` / `all` and every shape. The rest of rows_exact also passes (W2, `raw/gates-*.txt`), and so do the
  final-default re-run (W3, `raw/gates-w3-rows.txt`) and row_tables.
- `batch_cuda_real` 35B: 4290 (row, step) pairs are bitwise identical (`raw/batch-real.txt`).
- Served `equal.py` main vs dev: 21/21 cells identical, 9408 token ids (`raw/ident/compare.log`).
- `plan show` is byte-identical for the 8B, 35B and Gemma-4, json and text (`raw/plan.txt`).
- Mac `cargo test --workspace --release` passes; `docs_env_vars` passes with the new tuning rows.

## Process

fleetq jobs on spark1:
- builds (`--load cpu`): #7085, #7100 (failed: missing `<cstdio>`), #7103 (its error log), #7104, #7107, #7110;
- probe builds #7089 and #7099;
- plan show #7111;
- GPU windows (`--quiet gpu,cpu`): W1 #7094 (4 min: probe, main micro, nsys B = 16 / B = 1), W2 #7108 (14 min: probe round 2,
  gates, batch_cuda_real, dev micro, step A/B) and W3 #7112 (22 min: rows re-run, nsys dev, served identity, serve A/B).

Scripts are in `scripts/stream16/`. `~/intarsiaml-stream16` on spark1 was removed afterwards.

## Next

The remaining fp8 gap is the per-block re-quantisation of x: about 10 us per K = 4096 call, 40 calls per step. Quantising once,
in the producing kernel or one small kernel per activation, with the GEMV reading the codes, would recover about 0.3 to 0.4 ms
per B = 16 step, bitwise. It needs a new op or a fused producer, so it changes `plan show`; this lane did not do it.
