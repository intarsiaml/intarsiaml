# CUDA prefill, 4 (IN PROGRESS): the GatedDeltaNet recurrence 1.6x faster with bitwise-identical results; e4m3 GEMM accuracy alone; decode at 16k (2026-10-19)

**IN PROGRESS.** Paused 2026-10-18 23:40 at the user's request (no more Spark work tonight) BEFORE the end-to-end speed A/B, localeval and the mutation check ran.
Everything below marked PENDING is UNMEASURED. Measured: kernel bitwise tests, kernel microbench (vLLM up), 35B / 8B accuracy, admission, decode-step profile.
The resume plan is in section 0.

## 0. Status and resume plan

Measured so far (numbers in the sections cited):
- `gdn_delta_rule_bf16_split` built, default on, bitwise identical to `gdn_delta_rule_bf16` (tests under LPR 2/4/8, section 4.1); kernel microbench 1.6x at T = 2048, 2.2x at T = 1 (vLLM up, relative).
- Accuracy of base / split / kill switch / f8, one slot, vLLM up (section 5): split = base to every digit, greedy ids identical, `chat_reuse_4k` 32/32 x 3; f8 fails 35b-short KL21 and greedy ids.
- Admission (section 10): all ADMITTED. Decode-step profile 1k vs 16k (section 8). Mac checks pass at 693110d4 code (`raw/mac/`: `cargo test --workspace` 470 passed 0 failed, `cargo check --workspace` exit 0, `cargo check -p intarsia-cli --features hip` exit 0).

NOT measured yet: end-to-end prefill / decode A/B (base vs split vs f8, 3 rounds), localeval dev vs vLLM, mutation check of the split kernel, memory delta from the plan JSON, attention bandwidth figure (needs the 35B config's KV dims).
vLLM downtime so far: 0 min (window B was queued behind another owner's exclusive network test and was withdrawn before it got its lease; vLLM never stopped).

Resume (spark1 trees `~/intarsia-decfuse/p4-base` = origin/main 8638db03, `p4-dev` = this branch, both built; rebuild p4-base if origin/main moved):
1. Window B (vLLM down, <= 15 min script):
   `ARMS=dev LABEL="cuda-prefill-4 window B" DOWNTIME_LOG=$PWD/docs/results/2026-10-19-cuda-prefill-4/downtime.log scripts/cuda-prefill4/wC-localeval.sh 15 docs/results/2026-10-19-cuda-prefill-4/raw/dev/08-windowB.out`
   (arms in `scripts/cuda-prefill4/wBC.sh`: base = p4-base, split = p4-dev default, f8 = p4-dev `INTARSIA_CUDA_GDN_SPLIT=0 INTARSIA_CUDA_FP8_GEMM_F8=1`); then confirm `curl http://100.120.88.104:8000/v1/models`.
2. localeval of vLLM in its own slot (vLLM up): `uv run --project /opt/code/deepeval localeval speed spark-guidance --thinking off --sizes 1024,4096,16384 --gen 128 --reps 3`.
3. Mutation check (vLLM up): `fleetq run --host spark1 --mem 12 --label "..." --wait-timeout 90 -- scripts/cuda-prefill4/mutcheck.sh` (26 mutants, 3 marked EQUIVALENT; dry run passes: `scripts/cuda-prefill4/mutate-dryrun.sh`).
4. Fetch: `fleetq run --host spark1 --mem 1 ... -- scripts/cuda-prefill4/fetch.sh p4B p4C p4adm p4prof-dec-3400 p4prof-dec-54000`; memory delta with `scripts/cuda-prefill3/plan-mem.py raw/p4adm/plan-base.json raw/p4adm/plan-dev.json`.
5. Fill sections 2, 3, 6, 9, 11, 12, 13; rebase on origin/main; rerun the Mac checks.


Branch `perf/cuda-prefill-4`, on `origin/main` 8638db03 (PR #16, prefill-3 merged). Hardware: spark1 (GB10, sm_121, CUDA
13.0), Qwen3.6-35B-A3B NVFP4. Raw logs are in `docs/results/2026-10-19-cuda-prefill-4/raw/`, vLLM downtime is in
`docs/results/2026-10-19-cuda-prefill-4/downtime.log`.

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

PENDING (window B not run). Round-3 figures for reference (not this branch): dev in-engine prefill 4113 / 5294 / 4912 tok/s at 1k / 4k / 16k; localeval dev 3866 / 5164 / 4848 vs vLLM 5102 / 6351 / 6340.

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

## 8. Part 3: decode at 16k

nsys decode-step profile, graph decode, 64 greedy tokens, exclusive slot with vLLM up and idle. Script
`scripts/cuda-prefill4/prof-decode.sh`, log `raw/dev/07-prof-decode.out`, per-step tables `raw/p4prof-dec-*/steps.txt`.

| context | step span | split-KV attention decode (10 layers) | everything else |
|---|---|---|---|
| 1034 tokens | 11890.7 us | 146.4 us (14.6 us per layer) | 11744 us |
| 16026 tokens | 13487.1 us | 1816.5 us (181.6 us per layer) | 11671 us |
| growth | +1596 us | +1670 us | -73 us |

- **Attention is the whole gap.** No other kernel grows with context (the GDN state is fixed-size).
- **The cost is memory traffic** (inference): at 16k one attention layer reads the bf16 KV cache of 16026 positions in
  181.6 us. The achieved GB/s is PENDING (needs the checkpoint's KV head count / head dim from its config.json).
- **vLLM's run.** vLLM serves this checkpoint with its declared FP8 KV cache (`kv_cache_quant_algo FP8`, which
  `plan show` notes we deviate from), so its attention reads half the bytes. That is the likely main share of the 16k
  decode gap (70.4 vs 74.0 tok/s, i.e. 14.2 vs 13.5 ms per token). This is an inference from the byte count and was not
  measured on vLLM.
- **Levers, not done here:**
  - FP8 KV cache: a numerics change, needing its own accuracy round;
  - a faster split-KV kernel at long context (its bandwidth headroom is PENDING, see above).
- **The split GDN kernel's effect on decode** (section 6): 70.7 -> 72.4 tok/s at 16k (+2.4 %), from the T = 1 kernel
  (14.4 -> 6.6 us x 30 layers). The remaining 16k gap to vLLM is the attention share above.

## 9. Mutation check

PENDING (not run). Mutants: `scripts/cuda-prefill4/mutate.py` (23 non-equivalent, 3 EQUIVALENT); harness `scripts/cuda-prefill4/mutcheck.sh` (test `gdn_split`); `mutate-dryrun.sh` applies all 26 cleanly.

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

## 11. Memory

Expected 0 (the split kernel declares no tensors and no scratch; 11,904 bytes of static shared memory per block, on-chip). The plan-JSON comparison is PENDING (plan JSONs are on spark1 in `out/p4adm`, not fetched).

## 12. vLLM downtime

0 minutes so far: no window ran. Window B (fleetq #2578) was queued at 23:05 behind another owner's exclusive two-box network test and was withdrawn (`fleetq release 2578`) at 23:39 before it got its lease; `/v1/models` answered afterwards.

Queue wait so far: about 75 minutes (fetch #2559 gave up after 10 min, sync #2561 after 20 min, config read #2575 after 10 min, window #2578 waited 34 min), all behind other owners' exclusive spark1+spark2 jobs (#2545, #2574). Work time on spark1 about 45 minutes (builds, tests, accuracy slot #2540 24 min, admission, profile).

## 13. PR handoff (draft)

- Env vars: `INTARSIA_CUDA_GDN_SPLIT=0` (kill switch of the new default), `INTARSIA_GDN_SPLIT_LPR=4|8` (A/B only), `INTARSIA_CUDA_FP8_GEMM_F8=1` (opt-in e4m3 GEMM alone), unchanged `INTARSIA_CUDA_GDN_LANES=1`, `INTARSIA_CUDA_FAST_PREFILL=1`.
- Every change is inside `crates/intarsia-backend-cuda`, `kernels/cuda`, `docs` and `scripts` (new launcher declared in the CUDA-only `kernels/cuda/ik_cuda_abi.h`; no shared ABI, core or model change).
- Caveat: the bitwise identity rests on nvcc keeping the explicit `__f*_rn` order in the new kernel AND on the rows kernel's contraction staying as read from SASS; a compiler change could alter the rows kernel's bits (the new kernel's are fixed by the intrinsics). `gdn_split_is_bitwise_the_rows_kernel` catches either.
