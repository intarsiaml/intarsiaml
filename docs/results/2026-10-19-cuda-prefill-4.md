# CUDA prefill, 4: the GatedDeltaNet recurrence 1.6x faster with bitwise-identical results; e4m3 GEMM accuracy alone; decode at 16k (2026-10-19)

## 1. Headline

- **Ships, default on: `gdn_delta_rule_bf16_split`.** It does the one-thread-per-row GatedDeltaNet recurrence's fp32
  arithmetic in the same order, with the element-wise work split over 2 lanes per state row.
  - **Bitwise identical.** Outputs and state equal `gdn_delta_rule_bf16`'s bit for bit for dk 64 / 128, for every T and
    for lanes per row 2, 4 and 8 (tests). The 35B gives the same KL digits, the same greedy ids and the same
    `chat_reuse_4k` results as `origin/main`.
  - **Speed.** The kernel is 1.6x faster at T = 2048 (2834 -> 1697 us per layer call) and 2.2x faster at T = 1
    (14.4 -> 6.6 us).
  - **End-to-end** (interleaved A/B, 3 rounds, same window): prefill **+7.8 / +13.4 / +9.5 %** at 1k / 4k / 16k,
    decode **+1.9 / +2.4 / +2.4 %** (section 6).
- **localeval, dev vs vLLM** (section 2): prefill 4166 / 5699 / 5281 vs vLLM 5281 / 6374 / 6328 tok/s; the gap to vLLM is now
  21 / 11 / 17 % at 1k / 4k / 16k (round 3: 24 / 19 / 24 %). Decode 82.0 / 79.5 / 72.2 vs vLLM 77.2 / 76.6 / 74.2:
  ahead at 1k and 4k, 2.7 % behind at 16k (round 3: 4.9 %).
- **Why the opt-in lanes kernel failed the bars (Part 2).** The cause is not a bug. The lanes kernel sums the two dot
  products of the recurrence (`S^T k`, `S^T q`) in a different order. The 35B's bars are this sensitive to a change of
  fp32 rounding:
  - the 35b-short KL21 and the greedy ids;
  - the `chat_reuse_4k` `continue` row, which compares the warm run against the cold run inside the arm.
  The fix keeps the order and splits the work instead.
- **`matmul_fp8_gemm_f8` alone (Part 1) stays opt-in.** It now has its own gate, `INTARSIA_CUDA_FP8_GEMM_F8=1`.
  - It fails three bars: 35b-short KL21 (7.498e-3 at chunk 2048 against the base range 5.430e-3 .. 6.271e-3), 35b-long
    at chunk 256 (4.656e-2 against 4.273e-2 .. 4.576e-2), and greedy ids (they differ at token 66).
  - It passes 35b-long at chunk 2048 (4.315e-2) and `chat_reuse_4k` (32/32 x 3).
- **Decode at 16k (Part 3).** Profiled: the split-KV attention decode kernel accounts for the whole step-time growth from
  1k to 16k context (section 8).

## 2. Results vs base and vLLM

localeval speed (`uv run localeval speed`, 3 reps, gen 128, thinking off). dev = `intarsia serve` of this branch's
default, in window B (vLLM down, `raw/dev/08-windowB.out`); vLLM = `spark-guidance` in its own fleetq slot (#2623, vLLM
up, 03:05, `raw/dev/09-localeval-vllm.out`). In-engine = the interleaved A/B of section 6 (medians of 3 rounds).

| | 1k | 4k | 16k |
|---|---|---|---|
| prefill, in-engine base (origin/main) | 4122 | 5191 | 4910 |
| prefill, in-engine dev | 4445 (+7.8 %) | 5889 (+13.4 %) | 5375 (+9.5 %) |
| prefill, localeval dev | 4166 | 5699 | 5281 |
| prefill, localeval vLLM | 5281 | 6374 | 6328 |
| dev vs vLLM (localeval) | -21.1 % | -10.6 % | -16.5 % |
| round 3 dev vs vLLM (localeval) | -24.2 % (3866 / 5102) | -18.7 % (5164 / 6351) | -23.5 % (4848 / 6340) |
| decode, in-engine base | 80.8 | 78.0 | 70.7 |
| decode, in-engine dev | 82.3 (+1.9 %) | 79.9 (+2.4 %) | 72.4 (+2.4 %) |
| decode, localeval dev | 82.0 | 79.5 | 72.2 |
| decode, localeval vLLM | 77.2 | 76.6 | 74.2 |

- localeval's 1k prefill figure is marked `*` (under 1024 tokens, TTFT-dominated) for both servers.
- dev and vLLM cannot run in the same window (one GPU); vLLM's figures match round 3's (5102 / 6351 / 6340) within 4 %.

## 3. What runs where (35B, this branch's default)

| op | intarsia base (origin/main) | intarsia dev | vLLM |
|---|---|---|---|
| GDN recurrence | `gdn_delta_rule_bf16` (one thread per row) | `gdn_delta_rule_bf16_split` (same bits) | Triton / FLA chunked kernels |
| fp8 dense projections | gemv 1..4, mma 5..32, gemm 33..255, gemm_reg >= 256 | unchanged (`matmul_fp8_gemm_f8` opt-in) | `CutlassFP8ScaledMMLinearKernel` (e4m3 W8A8) |
| NVFP4 / MoE experts | unchanged | unchanged | `MarlinNvFp4LinearKernel` |
| attention decode | split-KV, bf16 KV cache | unchanged | (FP8 KV cache per the checkpoint; not verified this round) |

## 4. What was built

### 4.1 `gdn_delta_rule_bf16_split` (kernels/cuda/gdn.cu)

**The order of operations.** It is read from the SASS of the shipped `gdn_delta_rows_kernel<128>` (nvcc 13.0, `-O3`,
fmad on; `cuobjdump -sass` of the round-3 build). Per token and state row:

- the decay is an FMUL per element;
- each float4 group's dot product `s.x*k.x + s.y*k.y + s.z*k.z + s.w*k.w` is one FMUL (`y`), then FFMA `x`, `z`, `w`;
- `kv = ((0 + g0) + g1) + ... + g31` is a left-to-right FADD chain over the 32 group sums;
- `delta = (v - kv) * beta`;
- the update is `s = fma(k, delta, s)`;
- the output dot product is formed the same way on the updated state.

The new kernel writes exactly these operations with `__fmul_rn` / `__fmaf_rn` / `__fadd_rn` / `__fsub_rn`, so nvcc
cannot contract them differently. The q/k l2 norm, beta and decay code is the rows kernel's code, unchanged.

**The split.**

- **Group ownership.** Lane `l` of a row owns the contiguous float4 groups `[l*NG, (l+1)*NG)`. The left-to-right sum is a
  carry handed lane to lane: LPR stages of NG adds, then one shuffle each. The FADD chain is still 32 long, but the
  decay, the update and the group sums (3/4 of the rows kernel's instructions) are divided by LPR.
- **q/k staging.** q/k are staged in shared memory with one pad float4 after every NG groups, so the lanes of a row hit
  distinct banks. The staging is double-buffered: the next token's l2 norm (two warp reductions) overlaps the current
  token's update instead of preceding it.
- **State load and store.** The state is loaded and stored coalesced through a padded shared-memory tile. The warp's rows
  are contiguous in memory.
  - The first version loaded each lane's contiguous groups straight from global memory. It was no faster than the rows
    kernel at T = 1 (14.4 us, `raw/dev/01-split-first.out`).
  - The tile made T = 1 2.2x faster (6.5 us, `raw/dev/02-split-coalesced.out`).
- **Lanes per row.** 2 lanes per row (16 rows per warp, 209 registers, no spill) is the default. `INTARSIA_GDN_SPLIT_LPR=4|8`
  is for A/B only; the bits do not depend on it.
  - The rows kernel itself uses 255 registers and spills 232 bytes of stack (`raw/dev/02-split-coalesced.out`,
    `cuobjdump --dump-resource-usage`).

**Selection.** It is registered for dk 64 / 128 at priority 5. That is above the rows kernel (0) and below the opt-in
lanes kernel (10 with `INTARSIA_CUDA_GDN_LANES=1`). It serves prefill and decode. `INTARSIA_CUDA_GDN_SPLIT=0` switches it
off. dk 32 and 256 stay on `gdn_delta_rule_bf16`. `kernels::tests::gdn_delta_rule_selection` pins all of this.

**Kernel timing.** These are `tests/gdn_timing.rs` at the 35B shape (32 value heads x 128 x 128), in us per launch, two
rounds. vLLM was up (small slot), so read them as relative numbers (`raw/dev/04-split-lpr2-tests.out`):

| T | rows (`gdn_delta_rule_bf16`, origin/main) | lanes (opt-in, not bitwise) | split (new default) |
|---|---|---|---|
| 1 | 14.4, 14.4 | 6.2, 6.2 | 6.7, 6.6 |
| 16 | 32.5, 31.6 | 16.5, 16.5 | 16.5, 16.5 |
| 128 | 159.7, 162.4 | 98.2, 98.9 | 98.3, 103.9 |
| 512 | 588.2, 598.5 | 376.0, 377.2 | 377.4, 377.3 |
| 2048 | 2833.6, 2796.5 | 1833.1, 1836.6 | 1696.5, 1717.9 |

Lanes per row, two rounds (`raw/dev/02-split-coalesced.out`):

| T | LPR 2 | LPR 4 | LPR 8 |
|---|---|---|---|
| 1 | 6.5, 7.0 us | 8.2, 7.8 us | 8.2, 8.2 us |
| 2048 | 1699, 1866 us | 1865, 1864 us | 4064, 4047 us |

The default is therefore LPR 2.

**Tests** (`crates/intarsia-backend-cuda/tests/gdn_split.rs`, run under LPR 2, 4 and 8, all pass;
`raw/dev/04-split-lpr2-tests.out`):

- `gdn_split_is_bitwise_the_rows_kernel` compares output and final state byte for byte against `gdn_delta_rule_bf16`:
  - at the 35B shape for T in 1, 2, 3, 7, 16, 17, 130, 513 and 2048;
  - at dk 64 (dv 40, 20), at dk 128 with dv 36 (not a multiple of the rows per warp), and with Hv / Hk = 2;
  - with normal gates and with gates scaled x12, which takes softplus past its `x > 20` branch and pushes decay / beta
    to 0 and 1.
- `gdn_split_state_carries_across_calls` checks that splitting a sequence (tails of 1 included) gives the bits of one call.

### 4.2 `INTARSIA_CUDA_FP8_GEMM_F8=1` (crates/intarsia-backend-cuda/src/kernels.rs, quantmoe.rs)

`matmul_fp8_gemm_f8` was reachable only through `INTARSIA_CUDA_FAST_PREFILL=1`, together with the lanes GDN kernel. The
new gate `kernels::fp8_gemm_f8()` turns it on alone; `FAST_PREFILL` still turns it on as before.

**Token range.** It serves every M >= 5 (the gemv keeps 1..4), as under `FAST_PREFILL`. The kernel's registration says
why: one rounding for every chunk length, so a 5..32-token tail chunk gives the bits a full chunk gives its rows. This
range was a deliberate choice, not a narrower "prefill only" one. `kernels::tests::fp8_gemm_selection_by_batch_size`
pins it under the gate: M 4 -> gemv; M 5, 32, 33, 255, 256, 2048 and 4096 -> `matmul_fp8_gemm_f8`. The unused
`opt_in_priority` is removed.

**Numerics.** The kernel does not change the activation scaling. Like the default path, it uses the checkpoint's static
per-tensor `input_scale` (`fp8_actq_kernel`), as vLLM / ModelOpt do. The default path already multiplies e4m3-valued
activations by e4m3 weights (the bf16 holds them exactly). So the only numerical difference is the tensor core's
accumulation (e4m3 `m16n8k32` against bf16 `m16n8k16`, f32 accumulators) and its order.

## 5. Accuracy

Setup:

- Goldens: vLLM, sha b6b51e1b1699c66c, `vllm-top20` (35b-short) and `vllm-long-top20` (35b-long).
- One fleetq slot (#2540) with vLLM up, one 35B process at a time (`--mem-budget 40`). The kernels are deterministic, so
  the numbers do not depend on what else runs on the GPU; base and the round-3 values agree to every digit.
- Driver: `scripts/cuda-prefill4/acc.sh`; log `raw/dev/05-acc-driver.out`; per-run logs `raw/p4A/`.
- Arms:
  - base = tree `p4-base`, i.e. `origin/main` 8638db03 as is;
  - split = `p4-dev` default;
  - ks = `p4-dev` with `INTARSIA_CUDA_GDN_SPLIT=0`;
  - f8 = `p4-dev` with `INTARSIA_CUDA_GDN_SPLIT=0 INTARSIA_CUDA_FP8_GEMM_F8=1`.

| bar | base (range over chunk 2048 / 1024 / 256) | split (default) | f8 (opt-in) |
|---|---|---|---|
| 35b-short KL21, chunk 2048 | 5.430e-3 (top-1 97.92 %) | **5.430e-3** (97.92 %) | 7.498e-3 (97.99 %) **outside** |
| 35b-short KL21, chunk 256 | 6.271e-3 (97.99 %); range 5.430e-3 .. 6.271e-3 | **6.271e-3** (97.99 %) | 6.522e-3 (98.13 %) **outside** |
| 35b-long KL21, chunk 2048 | 4.576e-2 (top-1 94.17 %) | **4.576e-2** (94.17 %) | 4.315e-2 (94.27 %) inside |
| 35b-long KL21, chunk 256 | 4.273e-2 (94.32 %); 1024: 4.421e-2; range 4.273e-2 .. 4.576e-2 | not run (bitwise) | 4.656e-2 (94.28 %) **outside** |
| greedy ids, 1497-token prompt, 256 tokens | reference | identical (graph and eager) | differ at token 66 |
| `chat_reuse_4k` (continue / restore / aligned) | 32/32, 32/32, 32/32 | 32/32, 32/32, 32/32 | 32/32, 32/32, 32/32 |
| prefix-reuse gate (also `long_docs_12k`, `long_code_12k`) | PASS | PASS | PASS |
| 8B fp32 short, dev default | | prefill KL 2.897e-4, decode 3.238e-4, PASS | |
| 8B fp32 long, dev default | | prefill KL 4.004e-4, decode 3.332e-4, PASS | |

- **ks.** The kill switch gives 5.430e-3 on 35b-short at chunk 2048, and its greedy ids are identical to base.
- **35b-long exits 1 in every arm, base included.** The 21-way KL floor of the class (3.0e-2) fails there for the
  `origin/main` kernels too, as in rounds 2 and 3. The bar is the same-window base range, not that floor.
- **f8 vs the golden.** f8 moves toward vLLM on 35b-long at chunk 2048 (-5.7 %) and away from it on 35b-short (+38 % at
  chunk 2048, +4 % at chunk 256) and on 35b-long at chunk 256 (+9 %). Its top-1 is higher in every case. The golden uses
  fp8 activations, but this kernel's activations already were e4m3-valued, so it does not move "toward fp8"; it changes
  only the accumulation.
- **The pattern is the same as the lanes kernel's in round 3.** Lanes went 7.571e-3 at chunk 2048 and 6.628e-3 at 256,
  with a higher top-1. That fits a KL21 dominated by a few near-tie positions: in round 3, `long_prose_700`'s worst
  position had KL 1.1, worth 7.6e-4 of the 1441-position mean on its own. Any reordering of fp32 sums lands somewhere in
  a ~1e-3 spread.

## 6. Speed A/B

Window B (vLLM down, fleetq #2608, `scripts/cuda-decode/window.sh` -> `scripts/cuda-prefill4/wBC.sh` -> `wB-ab.sh`; log
`raw/dev/08-windowB.out`, per-run JSON / logs `raw/p4B/ab/`). `intarsia bench speed` against a fresh `serve` per arm
(`scripts/cuda-decode/w-ab.sh`, unmodified), sizes 1024 / 4096 / 16384, gen 128, 2 reps per size, 3 rounds interleaved
(order base, split, f8 / f8, split, base / base, split, f8). base = tree `p4-base` (origin/main 8638db03 as built; main
456e6e7d differs from it only in HIP and docs), split = `p4-dev` default, f8 = `p4-dev` with
`INTARSIA_CUDA_GDN_SPLIT=0 INTARSIA_CUDA_FP8_GEMM_F8=1`.

Prefill tok/s (r1, r2, r3; change vs base paired per round):

| prompt | base | split (default) | change, median (paired) | f8 (opt-in, fails accuracy) | change, median (paired) |
|---|---|---|---|---|---|
| 1024 | 4112, 4164, 4122 | 4445, 4455, 4398 | **+7.8 %** (+8.1, +7.0, +6.7) | 4226, 4196, 4232 | +2.5 % (+2.8, +0.8, +2.7) |
| 4096 | 5286, 5168, 5191 | 5681, 5889, 5896 | **+13.4 %** (+7.5, +14.0, +13.6) | 5577, 5579, 5427 | +7.4 % (+5.5, +8.0, +4.5) |
| 16384 | 4896, 4943, 4910 | 5375, 5380, 5368 | **+9.5 %** (+9.8, +8.8, +9.3) | 5149, 5127, 5143 | +4.7 % (+5.2, +3.7, +4.7) |

Decode tok/s:

| prompt | base | split | change, median (paired) | f8 | change |
|---|---|---|---|---|---|
| 1024 | 80.7, 80.8, 81.5 | 82.7, 82.2, 82.3 | **+1.9 %** (+2.5, +1.7, +1.0) | 80.7, 81.1, 81.1 | +0.4 % |
| 4096 | 77.9, 78.0, 78.7 | 80.5, 79.7, 79.9 | **+2.4 %** (+3.3, +2.2, +1.5) | 77.9, 78.5, 78.2 | +0.3 % |
| 16384 | 70.5, 70.7, 71.3 | 72.6, 72.4, 72.3 | **+2.4 %** (+3.0, +2.4, +1.4) | 70.5, 71.1, 70.6 | -0.1 % |

- No regression in either phase: the split kernel serves prefill and decode, and both improve in every round.
- The base arm reproduces round 3's dev figures (4113 / 5294 / 4912 prefill, 80.8 / 78.0 / 70.4 decode) within the
  run-to-run spread, as it must (same code).
- Prompt lengths differ by a few tokens per arm (the bench's prompt; 993 / 998 / 991 etc., in the JSON), as in round 3.
- f8 alone gives about half the prefill gain of split (its GEMM is 21 % faster per chunk, round 3) and nothing in decode
  (decode uses the gemv); it is not shipped (section 5).
- The A/B took 9.8 of the window's 14.6 minutes (01:48:23 .. 01:58:12); localeval of dev followed in the same window (section 2).

## 7. Part 2 diagnosis: why the lanes kernel diverges

1. **No indexing, state or chunk bug.** Code review of `gdn_delta_lanes_kernel` against `gdn_delta_rows_kernel` shows the
   same state layout and load / store, the same q/k normalisation code, the same decay, beta and update, and the same
   output index. Its round-3 tests already showed it chunk-invariant bit for bit (`gdn_lanes_state_carries_across_calls`)
   and closer to the f64 reference than the rows kernel (output rel_l2 1.89e-5 against 2.92e-5 at T = 513).
2. **What differs is the summation tree of the two dot products** `kv = S^T k` and `out = S^T q`. The lanes kernel uses
   four partial sums per lane, then `(p0 + p1) + (p2 + p3)` and a butterfly over the lanes of the row. The rows kernel
   uses the left-to-right chain of section 4.1. Everything else is element-wise and identical.
3. **The `chat_reuse_4k` 22/32 was the `continue` row** (`raw/p3A/reuse-lanes.log` of round 3). There, turn 1's 15
   reply tokens go through decode and the cold run prefills them. The `restore` and `aligned` rows, which restore a saved
   state at a chunk boundary, were 32/32. The gate itself marked the row `ok`: the divergence came past its first 16
   tokens and at a near-tie.
   - The warm and cold runs differ in their GEMM paths: decode gemv against prefill GEMM for those 15 tokens. They
     therefore differ in rounding in every arm.
   - Base happened to keep the argmax for all 32 tokens; with lanes it flipped at token 22.
   - A save / restore or chunk-boundary bug would break `restore` / `aligned`, and the split test. It does neither.
4. **Precision is not the issue.** The lanes kernel is not less accurate, and the split kernel shows that the rows
   kernel's order costs nothing in speed. So the fix is to keep the order. Making the sums more accurate (Kahan, f64)
   would still change bits, and would face the same coin flip against the bars.

## 7b. Tried and dropped

- **Split kernel v1, state read per lane from global memory** (no coalesced tile): T = 1 was 14.4 us, no faster than
  the rows kernel (`raw/dev/01-split-first.out`). The padded shared-memory tile made it 6.5 us.
- **4 and 8 lanes per row:** slower than 2 at T = 1 (8.2 vs 6.5 us) and, for 8, at T = 2048 (4064 vs 1699 us)
  (`raw/dev/02-split-coalesced.out`). Kept only as an A/B switch.
- **`matmul_fp8_gemm_f8` as default:** dropped, fails 35b-short KL21, 35b-long chunk 256 and greedy ids (section 5).
- **Repairing the lanes kernel in place:** not done. Any change to its summation tree changes bits, so it cannot meet
  the bitwise bars; it remains opt-in (`INTARSIA_CUDA_GDN_LANES=1`) and still fails them. The split kernel supersedes it:
  as fast for T >= 16, 7 % faster at T = 2048, 0.4 us slower at T = 1 (section 4.1).
- **More accurate sums (Kahan / f64) in the recurrence:** not tried, for the reason in section 7 point 4.

## 8. Part 3: decode at 16k

nsys decode-step profile, graph decode, 64 greedy tokens, exclusive slot with vLLM up and idle. Script
`scripts/cuda-prefill4/prof-decode.sh`, log `raw/dev/07-prof-decode.out`, per-step tables `raw/p4prof-dec-*/steps.txt`.

| context | step span | split-KV attention decode (10 layers) | everything else |
|---|---|---|---|
| 1034 tokens | 11890.7 us | 146.4 us (14.6 us per layer) | 11744 us |
| 16026 tokens | 13487.1 us | 1816.5 us (181.6 us per layer) | 11671 us |
| growth | +1596 us | +1670 us | -73 us |

- **Attention is the whole gap.** No other kernel grows with context (the GDN state is fixed-size).
- **The cost is memory traffic.** The checkpoint's config (`raw/dev/12-config.out`): 40 layers, full attention every 4th
  (10 layers), 16 query / 2 KV heads, head_dim 256. bf16 K + V = 2 x 2 x 256 x 2 B = 2 KiB per position per layer, so
  one layer at 16026 positions reads 32.8 MB in 181.6 us = **181 GB/s**; at 1034 positions 2.1 MB in 14.6 us = 145 GB/s.
  Against GB10's published LPDDR5X peak of about 273 GB/s (spec figure, not measured here) that is about 66 % at 16k
  (arithmetic from byte counts, not a counter measurement).
- **vLLM's side, NOT verified.** The checkpoint declares an FP8 KV cache (`kv_cache_quant_algo FP8`; `plan show`
  notes that we deviate from it and use bf16). If vLLM honours it, its attention reads half the bytes, which would cover
  most of the remaining 16k decode gap (72.2 vs 74.2 tok/s localeval = 13.9 vs 13.5 ms per token, 0.4 ms; our 16k
  attention is 1.8 ms). This is an inference from byte counts; vLLM's KV dtype and kernel times were not inspected.
- **Levers, not done here:**
  - FP8 KV cache: a numerics change, needing its own accuracy round;
  - a faster split-KV kernel at long context: at most about 1.5x headroom if the 273 GB/s spec peak holds.
- **The split GDN kernel's effect on decode** (section 6): 70.7 -> 72.4 tok/s at 16k (+2.4 %), from the T = 1 kernel
  (14.4 -> 6.6 us x 30 layers). The remaining 16k gap to vLLM is the attention share above.

## 9. Mutation check

`scripts/cuda-prefill4/mutcheck.sh` (fleetq #2641, vLLM up; `raw/dev/10-mutcheck.out`): for each mutant of
`scripts/cuda-prefill4/mutate.py`, patch `kernels/cuda/gdn.cu` in a scratch tree, rebuild, run `tests/gdn_split.rs`.
Unmutated: pass. **26 mutants: 23 non-equivalent, all killed; 3 equivalent, survived as expected.**

- **22 killed by the bitwise tests:** both group-dot contraction orders, both carry orders, a tree instead of the chain,
  delta sign, decay of z / w dropped, update with q, output from the wrong lane, the four tile / staging pad offsets
  (`tile-pad`, `tile-mine`, `stage-pad`, `stage-first`), prefetch of t + 1, next token into the current buffer, the
  prologue prefetch, the store guard and `nrows` (partial last block), `__expf`, q with k's norm, v = 0, beta = 1.
- **1 killed by racecheck, not by the tests: `no-loop-sync`** (the `__syncwarp` at the top of each token removed). The
  lanes of a converged warp never show the race, so no output test can catch it. `scripts/cuda-prefill4/racecheck.sh`
  (fleetq #2643, `raw/dev/11-racecheck.out`) runs `gdn_split_state_carries_across_calls` under
  `compute-sanitizer --tool racecheck`: unmutated **0 hazards**; `no-loop-sync` **12 RAW hazards** on the staged q/k.
  This check is a script, not part of `cargo test`.
- **3 equivalent (survived):** `EQ-add-commute` (IEEE addition commutes), `EQ-last-shuffle` (lane LPR-1 shuffles to
  itself), `EQ-lpr-default` (LPR 4 default; the bits do not depend on it).
- Registry: `kernels::tests::gdn_delta_rule_selection` and `fp8_gemm_selection_by_batch_size` pin the selection under
  each gate (default, `INTARSIA_CUDA_GDN_SPLIT=0`, `INTARSIA_CUDA_FP8_GEMM_F8=1`, `INTARSIA_CUDA_GDN_LANES=1`; all pass
  on spark1, `raw/dev/04-split-lpr2-tests.out`, and on the Mac for the default).

## 10. Admission

`scripts/cuda-prefill4/admission.sh`, slot #2572, vLLM up; log `raw/dev/06-admission.out`.

- **Shared sweep, `intarsia kernels test --op gdn_delta_rule`.** `gdn_delta_rule_bf16_split` is ADMITTED: 12 cases,
  0 fail, max_abs 1.221e-4, rel_l2 3.773e-5. That is the rows kernel's figure to every digit, as expected of a bitwise
  kernel. Rows: 20 / 0 ADMITTED; lanes: 12 / 0 ADMITTED.
- **Shared sweep, `--op matmul_fp8`.** `matmul_fp8_gemm_f8` is ADMITTED: 27 / 0, max_abs 4.329e-3, rel_l2 1.849e-4. gemv,
  mma, gemm and gemm_reg are all ADMITTED with 0 fails.
- **Plan-time admission on the 35B plan's shapes** (`plan show --admit --no-admission-cache`, chunk 2048), nothing
  unadmitted:
  - base (`INTARSIA_CUDA_GDN_SPLIT=0`): 36 kernels, 505 cases, `gdn_delta_rule_bf16`;
  - dev: 36 kernels, 505 cases, `gdn_delta_rule_bf16_split`;
  - f8 (`INTARSIA_CUDA_FP8_GEMM_F8=1`): 34 kernels, 497 cases, `matmul_fp8_gemm_f8` replaces mma / gemm / gemm_reg.

## 10b. Mac checks

At the final code, rebased on origin/main 10eef81c (`raw/mac/`): `cargo test --workspace` 475 passed, 0 failed;
`cargo check --workspace` exit 0; `cargo check -p intarsia-cli --features hip` exit 0.

## 11. Memory

**0 bytes.** `plan show --json` of the 35B (chunk 2048) gives `scratch_bytes` 83963392 for base, dev and f8 alike
(`raw/p4adm/plan-mem.out`); the split kernel declares no tensors and no scratch. Its 11,904 bytes of static shared memory
per block are on-chip.

## 12. vLLM downtime and queue time

- **One window, 14.6 min** (`downtime.log`): fleetq #2608, vLLM stopped 01:48:12, serving again 02:02:53, through
  `scripts/cuda-decode/window.sh` (cap 15 min). `/v1/models` answered afterwards (also checked by hand).
- An earlier window (#2578) was queued 23:05 .. 23:39 behind another owner's exclusive network test and withdrawn
  before it got its lease; vLLM did not stop then.
- **Queue wait about 2.9 h** in total, all behind other owners' exclusive spark1 / spark2 jobs (#2545, #2574, #2601,
  #2610): fetches and syncs that gave up (about 40 min), window #2578 (34 min), window #2608 (34 min), localeval of
  vLLM #2623 (about 60 min).
- **Work time on spark1 about 1.4 h:** builds and kernel tests (about 20 min), accuracy slot #2540 (24 min), admission
  (2 min), decode profile (3 min), window (14.6 min), localeval of vLLM (4 min), mutation check (3 min), racecheck
  (15 min).

## 13. PR handoff

- **Default change:** `gdn_delta_rule_bf16_split` replaces `gdn_delta_rule_bf16` for dk 64 / 128 (prefill and decode).
  Bitwise identical outputs and state (tests, 35B KL digits, greedy ids, `chat_reuse_4k`).
- **Env vars:**
  - `INTARSIA_CUDA_GDN_SPLIT=0`: kill switch of the new default;
  - `INTARSIA_GDN_SPLIT_LPR=4|8`: lanes per row, A/B only (default 2; bits unchanged);
  - `INTARSIA_CUDA_FP8_GEMM_F8=1`: new opt-in gate for the e4m3 tensor-core GEMM alone (fails the KL / greedy bars);
  - unchanged: `INTARSIA_CUDA_GDN_LANES=1`, `INTARSIA_CUDA_FAST_PREFILL=1`, `INTARSIA_CUDA_FP8_GEMM_REG=0`.
- **Files outside `crates/intarsia-backend-cuda`, `kernels/cuda`, docs and scripts: none.** The new launcher is declared in
  the CUDA-only `kernels/cuda/ik_cuda_abi.h`, not the shared `kernels/abi/ik_abi.h`.
- **Caveats:**
  - The bitwise identity rests on the rows kernel's contraction as nvcc 13.0 emits it today (read from SASS); a
    compiler change could alter the rows kernel's bits (the new kernel's are fixed by `__f*_rn`).
    `gdn_split_is_bitwise_the_rows_kernel` catches either.
  - The missing-`__syncwarp` mutant is caught only by `scripts/cuda-prefill4/racecheck.sh`, not by `cargo test`.
  - The in-engine A/B ran with the base arm from `p4-base` (8638db03) and dev from `p4-dev`; main moved to 10eef81c
    with HIP-only changes, so the CUDA code compared is the same.
  - The 16k decode attribution to vLLM's FP8 KV cache is an inference, not measured.
- Diff from the merge base (origin/main 10eef81c, three-dot): 25 code / script / doc files plus 115 raw log files
  under `docs/results/2026-10-19-cuda-prefill-4/raw/`.
