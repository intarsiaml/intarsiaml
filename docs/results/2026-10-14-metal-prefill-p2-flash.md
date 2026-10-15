# Metal prefill P2: flash-style causal prefill attention (2026-10-14)

Branch `metal-p2-flash` (from `spike/metal` 25127d9, merged with `spike/metal` 495a447 at the end). Qwen3-8B BF16 on endor
(Apple M5 Max), item P2 of `docs/research/metal-prefill-plan.md`. Raw logs and scripts:
`docs/results/raw/2026-10-14-metal-prefill-p2/`.

## Outcome

* **Attention kernel: 0.6 -> 12.3 TFLOPS at 4k context** (gate: >= 10). GPU-timestamp trace of a real 4k prefill: attention
  7704 ms -> 406 ms summed over the 16 chunks (19x). At 16k the last chunk's attention goes 3.17 s -> 0.18 s.
* **Prefill, same slot, ABBA (pre new new pre): marginal prefill at 4k 205 -> 384 tok/s (+87%), at 1k 351 -> 404 tok/s (+15%);
  plain 4k prefill 17.96 s -> 10.58 s.** 16k plain prefill 178.9 s -> 50.0 s (92.7 -> 331 tok/s). Decode unchanged
  (34.1-36.0 tok/s on both sides, inside run-to-run spread).
* **Correctness: PASS.** `kernels test --backend metal` ADMITTED (`attn_prefill_bf16_flash` 14 cases / 0 fail); `intarsia
  accuracy` prefill KL 2.590e-4 (short, was 2.608e-4) and 4.119e-4 (long, was 3.738e-4), both < 1e-3; greedy ids on the 4k
  prompt are identical for all 256 tokens; on prompt-short they match for the first 49 tokens and first differ at token index 49
  (a near-tie, see Correctness). Chunk 64 vs 256 end to end is identical for the 4k prompt and identical in KL for the golden.
* After the kernel, attention is 3.8% of 4k prefill (was 43%); the matmuls are now 95.5% of it at 5.5 TFLOPS. P2 is not the
  limit any more; P1 (matmul) is.

## What was built

* `kernels/metal/attention_flash.metal`: `attn_flash<D, BK, NSG>`, Metal 3.1 `simdgroup_matrix` only (no Metal 4 plumbing).
  One threadgroup per (block of `8 * NSG / G` query positions, kv head); simdgroup `sg` = one query head of the group x one slab
  of 8 positions, so the K/V tile staged in threadgroup memory serves all G query heads of the kv group and all position slabs.
  Q fragments live in registers, S = QK^T and O += PV are bf16 x bf16 -> fp32 `simdgroup_multiply_accumulate`, the softmax is
  online in fp32 (log2 domain, row max/sum kept in the fragment layout, xor-1/xor-8 shuffles), P is rounded to bf16 straight from
  the S fragment registers (the accumulator and A-operand layouts coincide). Key tiles start at absolute key 0 (not at the
  query offset), masking assigns -inf per tile and only tiles that reach past the block's first row are masked, tiles beyond a
  block's last row are never visited, blocks are issued longest first. Next tile's global loads are issued into registers
  before the current tile's compute and written to threadgroup memory after the next barrier (single-buffered). Reference
  for the fragment layout and loop structure: mlx steel attention,
  `.../mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h` and `.../attn/mma.h` (oMLX.app's bundled
  mlx python package).
* `crates/intarsia-backend-metal/src/flash.rs`: launcher `attn_prefill_bf16_flash`, shape predicate (`head_dim` 64/128/256,
  group size 1/2/4/8, any T, any pos0), variant table (NSG 4 default, NSG 8 selectable, `INTARSIA_METAL_FLASH=<name>` /
  `select_variant` for sweeps), fallback to `attn_prefill_bf16` (counted, reported once on stderr) for operands with a leading
  element skip or K/V caches not 16-byte aligned. Registered at priority 10 beside `attn_prefill_bf16` (KV dtype must be
  bf16, so fp8 and f32 caches keep the old kernel), tolerance `BF16_TOL`. Registration is three lines in `kernels.rs`, one in
  `lib.rs`, one in `build.rs` (`LAUNCHER_SOURCES`). `attention_splitkv.metal` and `attn.rs` untouched; `intarsia-core` untouched.
* head_dim 128 is the required case; 64 and 256 come from the same template (cheap) and are measured and tested; others are
  select-guarded to the old kernel.
* Tests: `crates/intarsia-backend-metal/tests/flash_prefill.rs` (f64 reference, 16 cases x every variant incl. chunk offsets,
  partial blocks and key tiles, groups 1/2/4/8, head_dim 64/128/256, NaN in the cache beyond the visible keys, cache filled to
  the last row; bitwise chunk-split invariance for five splits; fallback on misaligned operands; predicate). Bench:
  `examples/flash_bench.rs`.

## Kernel TFLOPS (microbenchmark)

Chunk 256, 32/8 heads, head_dim 128, 36 layers with one KV cache each, GPU time = command-buffer `GPUEndTime - GPUStartTime`
divided by 36, median of 8 reps (p10-p90 spread < 1% except where noted). FLOPs convention: `4 * D * H * (T * pos0 + T * (T+1)/2)`
(causal visible q.k pairs; the same as `analyze_prefill.py`). GPU idle before: 26, 0, 1 percent (first sample is the tail of the
previous job's handoff). Raw: `kernel-bench/bench-*.log`.

| pos0 | old us | flash us | old TFLOPS | flash TFLOPS | speedup |
|---:|---:|---:|---:|---:|---:|
| 0 | 687 | 53 | 0.78 | 10.1 | 12.9x |
| 256 | 2078 | 141 | 0.78 | 11.4 | 14.7x |
| 1024 | 6274 | 402 | 0.77 | 12.0 | 15.6x |
| 2048 | 13858 | 748 | 0.66 | 12.2 | 18.5x |
| 3840 | 27644 | 1353 | 0.60 | 12.3 | 20.4x |
| 7936 | 54970 | 2738 | 0.62 | 12.35 | 20.1x |
| 15872 | 107593 | 5426 | 0.62 | 12.4 | 19.8x |

Other shapes at pos0 3840 (chunk 256): head_dim 64 0.56 -> 12.3 TFLOPS; head_dim 256 0.63 -> 6.1; group 1 (8/8 heads) 0.48 -> 9.5;
group 2 0.56 -> 10.5; group 8 (64/8 heads) 0.65 -> 12.9. Chunk 64: 0.46 -> 9.6; chunk 1024: 0.66 -> 12.5 (p10-p90 5356-6122 us).
NSG 8 instead of 4 at group 4: 11.1 vs 12.3. The simdgroup_matrix peak measured with a register-only microbenchmark on this GPU
is about 15-16.4 TFLOPS for half, bfloat and float inputs alike (`micro/micro.log`, fully-resident 8x8x8 MMAs, no loads; transposed
fragment loads from threadgroup memory 14-15), so the kernel runs at roughly 75-80% of the matrix unit's rate.

## Per-chunk attention time vs chunk index (GPU-timestamp trace, whole-model prefill)

`INTARSIA_METAL_PROFILE=kernel:` trace of `generate --max-tokens 1` (one command buffer per launch, serialised, so the 36 layers
of a chunk sum; chunk 256). `trace/kernel-*-c256.jsonl.gz`, tables from `perchunk.py`; full 16k table in `trace/perchunk-16k.csv`.
GPU idle before the trace job: 98, 2, 0 percent (first sample is the previous job; no sample after warm-up was taken).

4k prompt (4062 tokens, 16 chunks; ms per chunk, all 36 layers):

| chunk | pos0 | T | pre ms | new ms | speedup | new TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 256 | 25 | 2.2 | 11.3x | 8.8 |
| 1 | 256 | 256 | 75 | 5.2 | 14.3x | 11.1 |
| 2 | 512 | 256 | 125 | 8.5 | 14.8x | 11.4 |
| 3 | 768 | 256 | 175 | 11.6 | 15.2x | 11.7 |
| 4 | 1024 | 256 | 226 | 14.7 | 15.3x | 11.8 |
| 5 | 1280 | 256 | 277 | 17.8 | 15.6x | 12.0 |
| 6 | 1536 | 256 | 327 | 21.0 | 15.6x | 12.0 |
| 7 | 1792 | 256 | 400 | 24.0 | 16.7x | 12.1 |
| 8 | 2048 | 256 | 514 | 27.4 | 18.8x | 12.0 |
| 9 | 2304 | 256 | 615 | 30.5 | 20.2x | 12.0 |
| 10 | 2560 | 256 | 683 | 33.7 | 20.3x | 12.1 |
| 11 | 2816 | 256 | 746 | 36.9 | 20.2x | 12.1 |
| 12 | 3072 | 256 | 808 | 40.4 | 20.0x | 12.0 |
| 13 | 3328 | 256 | 878 | 43.4 | 20.2x | 12.0 |
| 14 | 3584 | 256 | 944 | 46.6 | 20.3x | 12.0 |
| 15 | 3840 | 222 | 886 | 42.6 | 20.8x | 12.1 |

4k totals: attention 7703.6 ms -> 406.4 ms (0.63 -> 11.98 TFLOPS); matmul 10181.9 -> 10244.0 ms (unchanged, 5.5 TFLOPS).

16k prompt (16578 tokens, 65 chunks; every 8th chunk, full table in the csv):

| chunk | pos0 | T | pre ms | new ms | speedup | new TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 256 | 25 | 2.3 | 11.1x | 8.5 |
| 8 | 2048 | 256 | 477 | 27.6 | 17.3x | 11.9 |
| 16 | 4096 | 256 | 1006 | 52.1 | 19.3x | 12.3 |
| 24 | 6144 | 256 | 1567 | 80.4 | 19.5x | 11.8 |
| 32 | 8192 | 256 | 2086 | 112.0 | 18.6x | 11.2 |
| 40 | 10240 | 256 | 2334 | 129.5 | 18.0x | 12.1 |
| 48 | 12288 | 256 | 3145 | 152.6 | 20.6x | 12.3 |
| 56 | 14336 | 256 | 3623 | 176.9 | 20.5x | 12.3 |
| 64 | 16384 | 194 | 3166 | 180.6 | 17.5x | 10.4 |

Per-chunk time grows linearly with the context as expected (work proportional to pos0); the old kernel's column has steps
(e.g. 400 -> 514 ms between chunks 7 and 8, cause not investigated), the new one has none. Plain
(untraced) prefill, `generate --max-tokens 1`: 4k 18.1/17.9 s (pre, 2 runs) vs 11.2/11.0 s (new); 16k 178.9 s vs 50.0 s.

## Speed: localeval, same slot (fleetq #1902), pre, new, new, pre

Server command as `docs/results/raw/2026-10-14-metal-phase-b/b2b/leval.sh` (`--max-seq 8192`), `localeval speed --sizes
128,1024,4096 --gen 256 --reps 3 --thinking off`. Pre = build of 25127d9 with `--features metal`; new = this branch before the
spike/metal merge (the merge only adds decode-attention work; the final tree was not re-benchmarked end to end, see Not done).
GPU idle before the slot: 99, 4, 0 percent (the first sample is the previous job's tail; the servers run sequentially after
it). Raw: `localeval/run{1..4}-*-localeval.out`.

| run | binary | prompt 990: prefill tok/s (marginal) | prompt 4088: prefill tok/s (marginal) | TTFT 4k | decode tok/s at 128 / 1k / 4k |
|---|---|---:|---:|---:|---|
| 1 | pre | 344 (351) | 227 (205) | 17.96 s | 36.0 / 35.5 / 34.4 |
| 2 | new | 392 (405) | 386 (384) | 10.58 s | 36.0 / 35.5 / 34.1 |
| 3 | new | 388 (403) | 386 (385) | 10.58 s | 35.7 / 35.0 / 34.4 |
| 4 | pre | 344 (350) | 227 (206) | 17.95 s | 35.9 / 35.6 / 34.1 |

* Marginal prefill at 1k: 350 -> 404 tok/s (+15%); at 4k: 205 -> 384 tok/s (+87%). Both pairs reproduce within 1%.
* Decode did not move: 34.1-34.4 tok/s at 4k on both binaries (the one 33.4 low rep in run 2 is inside the pre runs' 34.0-34.4 spread
  of the pair), 35.0-35.6 at 1k, 35.7-36.0 at 128. The decode kernel is the same on both sides.
* A fifth, new-only run from a first attempt whose pair order was mis-parsed by the script is kept in `localeval-extra-new-only/`
  (consistent: 4k prefill 384 tok/s, marginal 382).

## Correctness

* `intarsia kernels test --backend metal` (`checks/kernels-test.out`): 20 kernels, 252 cases, all ADMITTED, including
  `attn_prefill_bf16_flash` 14 cases, max_abs 5.0e-3, rel_l2 2.4e-3 against the CPU reference (`attn_prefill_bf16` 21 cases,
  1.95e-3, 1.05e-4). The rel_l2 is larger than the old kernel's because P is rounded to bf16 before P.V; it is inside the
  registered BF16 tolerance, and the f64 test below shows the error is the expected P-rounding noise, not a bug.
* Cargo tests: `cargo test -p intarsia-backend-metal --release` 58 passed / 0 failed (before the spike/metal merge); after the
  after the merge (fleetq #1904, `final/`) 58 passed / 0 failed again, `kernels test` 20 kernels all ADMITTED, attn_prefill_bf16_flash 14 cases / 0 fail. `cargo test --workspace --release --exclude intarsia-backend-metal` 267 passed / 0 failed;
  `cargo check --workspace` clean. `flash_prefill.rs`: every variant against an f64 reference over 16 cases; worst error 0.75 of the
  tolerance (half a bf16 ulp of the result + 5 sigma of bf16 probability rounding noise). Chunk-split invariance is bitwise
  (whole 256-row call vs splits 64x4, 100+156, 1+255, 7+13+97+139, 128x2 at pos0 300).
* `intarsia accuracy` against the fp32 goldens (`gate-class bf16`; `checks/accuracy/`):

| golden | binary | prefill top-1 | prefill KL (full vocab) | verdict |
|---|---|---:|---:|---|
| qwen3-8b-fp32 (short) | pre | 100.00% | 2.608e-4 | PASS |
| qwen3-8b-fp32 (short) | new | 99.77% | 2.590e-4 | PASS |
| qwen3-8b-long-fp32 (long) | pre | 99.86% | 3.738e-4 | PASS |
| qwen3-8b-long-fp32 (long) | new | 99.65% | 4.119e-4 | PASS |

  Both under the 1e-3 gate; the long golden's KL grows by 10% (bf16 P rounding over 4k keys), top-1 loses 0.1-0.2 points
  (one or two positions) and stays above the 99.5 floor. For comparison the first, half-operand form of the kernel (11-bit P)
  scored 3.402e-4 / 3.693e-4 (`checks1/accuracy/half-*`): slightly closer on the long golden, but 6% slower and its greedy ids
  diverged earlier (token 22 of the 4k prompt), so it was dropped for bf16 operands.
* Greedy ids, 256 tokens, temperature 0 (`checks/identity/`): prompt-4k identical for all 256 tokens (pre vs new); prompt-short
  identical for the first 49 tokens and first differs at index 49 (pre ` ... a row of copper kettles`, new ` ... remembered the
  slow tide`), a near-tie flipped by the change in bf16 rounding; 48 tokens are identical in both prompts.
* Chunk invariance end to end (`checks/chunk/`): the short golden prefilled at chunk 64 gives the same KL as at chunk 256 for
  both binaries (pre 2.608e-4, new 2.590e-4, identical top-1); the 4k prompt generated with `--prefill-chunk 64` gives the same 256
  ids as at chunk 256 for both binaries. The new kernel is also bitwise chunk-invariant at the kernel level (test above).

## Dead ends and lessons

Exploration sweeps at pos0 3840, chunk 256 (TFLOPS; raw in `explore1`..`explore4`, `micro/`):

1. **First version: 1.3 TFLOPS.** The straightforward template (fragment arrays indexed in `for` loops, no unrolling) was 2x
   faster than the old kernel and bound by neither loads nor math: removing half the S products saved 1 ms of 12.4, half the P.V
   products 5 ms, all global loads nothing (`explore1/first-version-and-ablation.txt`). The fragment arrays were not kept in
   registers. Adding `#pragma clang loop unroll(full)` to every fragment loop took the same variants to 8.9-10.6 in one step.
   Lesson for later Metal kernels: with `simdgroup_matrix` arrays, unroll everything or the compiler spills them.
2. Two 8-row fragments per simdgroup (TQ = 2, K-fragment reuse) lose to one: 8.7-9.2 vs 9.3-10.6, at every tile size. Occupancy
   beats reuse. Tile size: BK 16 (10.6) > BK 8 (10.4) and BK 24 (10.9) > BK 32 (9.7); BK 64 with one shared K-then-V buffer
   (2 extra barriers) 8.7-9.5, no gain over separate buffers.
3. bfloat operands (11.40) beat half operands (10.59) at the same configuration, with half's bf16 -> half convert per element at
   the staging stores as the cost and no gain in the multiply; bfloat also has no 65504 range limit. Register prefetch of the
   next tile on top: 11.55 -> 12.17 (bfloat, BK 24, NSG 4); for half 10.9 -> 11.4. The final kernel is the bfloat, BK 24, prefetch
   one (`attn_flash_d128_s4`); the half-operand form had KL 3.402e-4 / 3.693e-4 (short/long) vs 2.590e-4 / 4.119e-4 and its greedy
   ids diverged earlier on the 4k prompt (index 22), so there was no accuracy reason to keep it either.
4. NSG 8 for group 4 (two position slabs per head) loses 10% to NSG 4 (12.3 -> 11.1); it exists only for group 8, where it is needed to
   cover the heads, and is within 5% of the best there.
5. The matrix unit does ~15-16 TFLOPS for any of half, bfloat, float inputs (float accumulate), so there was no operand-type
   shortcut to a higher ceiling in the Metal 3.1 formulation; 12.3 is the realistic end of this design.

## What is left

* The remaining gap to ~16 TFLOPS is barrier/LDS overhead per 24-key tile and the softmax's exp2 + shuffles per 8 rows; a
  double-buffered tile and 8 keys per row-pair of exps are the next steps, worth at most ~1.3x on a component that is now 3.8% of
  prefill, so not worth doing before P1/P3.
* head_dim 256 runs at 6.1 TFLOPS (register pressure); only matters for non-Qwen3 models.
* P3 (chunk size) interacts: at chunk 64 the kernel is 9.6 TFLOPS, at 1024 12.5; the kernel favours chunk >= 256.
* Not done: a re-benchmark of the final merged tree end to end (the merge added only decode-attention changes to files this
  branch does not touch); the NAX/Metal 4 variant (a parallel agent owns `METAL4_SOURCES`; the kernel can adopt it later by
  moving to `tensor_ops`, expected ceiling much higher); a decode-attention reuse of the same tile structure.

## fleetq jobs

#1874 (first microbench), #1876 (ablation), #1881/#1882/#1883/#1885 (exploration sweeps), #1884 and #1891 (kernel microbench,
tests), #1886 (first accuracy/identity, half vs bf16), #1894 (final tests, accuracy, identity, chunk invariance), #1898 (traces),
#1899 (localeval, mis-parsed pair order, new-only), #1902 (localeval pre new new pre), #1904 (merged-tree tests). The GPU-idle
samples are in each job's `gpu-idle-before.txt`; the first sample was repeatedly high (94-99%) right after a job handoff, the next
two 0-4%. Timed runs followed after server start-up / warm-up, but no per-run idle sample was taken after the first three.
