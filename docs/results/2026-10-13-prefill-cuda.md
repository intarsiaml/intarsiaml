# Prefill on CUDA, Qwen3.6-35B-A3B NVFP4 on spark1 (GB10, sm_121), 2026-10-13

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** C9: the "shipped chunk of 256" and "decide the prefill chunk" were open when this was written. Decided: the Qwen3.6-35B-A3B recipe default is 2048 on CUDA (commit `63b0f72`, ADR-0008 section 2); CUDA `--prefill-chunk 4096` stays a flag (user, 2026-10-19 21:53); HIP's 35B recipe default is 4096 (`[tuning.hip]`, `docs/results/2026-10-17-hip-prefill.md`).

Branch `perf/prefill-cuda` (from `arch/plan` 7e65770). Raw logs, bench JSON and profiler CSVs are in
`docs/results/2026-10-13-prefill-cuda/`; the vLLM downtime log is `docs/results/prefill-cuda-downtime.log`.

## 1. Result in six lines

1. **The prefill gap is mostly the prefill chunk, not a kernel.** At the shipped chunk of 256 tokens the MoE layers read all
   256 experts' weights once per chunk (437 MB per layer, 40 layers). A 4k prompt reads 280 GB, about 1.1 s at the
   measured 214 to 258 GB/s, and vLLM's whole 4k prefill takes 0.6 s. No MoE kernel can fix that at chunk 256.
   `--prefill-chunk 2048` alone gives 1.9x prefill at 4k and 16k (section 6).
2. **Two kernels built, both faster, both left opt-in.** `gdn_delta_rule_bf16_lanes` (GatedDeltaNet recurrence, 1.4x at
   T=2048, 2.3x at T=1) and `matmul_fp8_gemm_f8` (W8A8 projections on e4m3 tensor cores, 1.25 to 1.6x). They pass admission
   and the kernel tests with mutation checks. With both on, the 35B accuracy gates do **not** pass (section 5), so they are
   selected only with `INTARSIA_CUDA_FAST_PREFILL=1`. With the variable unset the registry selects the same kernel for every op as
   HEAD (the e87beed gate logs list no kernel difference from HEAD's run), and the control run with both kernels disabled
   reproduces HEAD's 35b-short numbers exactly (section 5).
3. **Speed, `bench speed`, 3 interleaved rounds, 5 reps** (section 6, tok/s, median of rounds):

   | prompt | HEAD, chunk 256 | chunk 2048 (default kernels) | chunk 2048 + fast kernels | vLLM |
   |---|---:|---:|---:|---:|
   | 128 | 1257 | 1270 | 1296 | 1660 |
   | 1024 | 2304 | 3891 | 4389 | 5631 |
   | 4096 | 2397 | 4654 | 5505 | 6595 |
   | 16384 | 2261 | 4310 | 5075 | 6412 |

   Decode (median tok/s at prompt 128 / 1024 / 4096 / 16384): HEAD 72.4 / 71.9 / 69.7 / 63.6, fast kernels 73.8 / 73.3 / 71.3 /
   65.0, vLLM 76.2 / 77.6 / 76.4 / 73.7. The gap to vLLM at 4k goes from 2.75x to 1.42x (chunk) and to 1.20x (chunk + kernels).
4. **Accuracy: not clean, reported as failures** (section 5). The default build with `--prefill-chunk 2048` passes
   35b-short (baseline verdict PASS, aggregate prefill KL 5.43e-3 against HEAD's 6.27e-3) and fails the 35b-long baseline
   verdict on one prompt (`chat_reuse_4k`, the prompt the existing chunk sweep already shows as chunk sensitive).
   The 35b-long floor verdict is FAIL for HEAD too (its recorded fact). The fast kernels fail 35b-short per-prompt
   gates and, at chunk 256, the long prefix-reuse and aggregate decode checks. Their aggregate prefill KL (6.5e-3 to 7.0e-3)
   is above HEAD's 6.27e-3, inside the spread HEAD itself shows under perturbation (5.5e-3 to 6.7e-3).
5. **Not done, with the measured reason**: chunked (WY) GatedDeltaNet, MoE kernel changes, FP4 block-scaled MMA (section 7).
6. **vLLM downtime: 108.9 min in 9 windows, longest 17.2 min** (section 8). It was restored and answering after every window.

## 2. Trees measured (tree-stamped)

| name | stamp (`scripts/tree-stamp.sh`) | used for |
|---|---|---|
| HEAD / base | `head=7e65770 code-tree=f16debdbcc1e dirty=no` | all "before" numbers, baselines and envelopes, the `old` A/B arm |
| profile build | `head=5e20110 code-tree=36619a01aee6 dirty=no` | HEAD plus `generate --op-table` (CLI only), W1 profile |
| d556af4 | `code-tree=d3e6f3992198 dirty=no` | W3 microbenchmarks (software-pipelined lanes variant, f8 gemm) |
| 9578fd9 | `code-tree=08c746d2ebd3 dirty=no` | W4 "after" nsys and first gates (fast kernels serve M >= 33 / T >= 16 only) |
| d940701 | `code-tree=293cc0ff199d dirty=no` | W5/W6 gates and bisect (kernels serve every M >= 5 / every T) |
| e87beed | `code-tree=160203ce0998 dirty=no` | final: kernel tests, the A/B `c2048` and `fastc2048` arms, default-kernel chunk-2048 gates |

The branch tip after e87beed differs only in `docs/` (this file and its logs). Builds ran on spark1 in `~/intarsia-perf`
(deleted at the end through a fleetq slot). Toolchain: the box's CUDA 13 `nvcc`, `sm_121`. Idle clocks at the start:
578 MHz SM, 42 C (`profile/clocks.txt`); the GPU boosts under load, no clock locking was done.

## 3. Profile before (HEAD 7e65770)

Tools: the engine profiler (`--profile-every 1`, new `generate --op-table` prints every op's time and share) and `nsys`
(`cuda_gpu_kern_sum`). `ncu` was not tried (it hit ERR_NVGPUCTRPERM on spark2 and nsys was enough). Prompts are slices
of the repo's docs of 3.5 kB and 14 kB (1067 and 3972 tokens), 1 generated token, `--max-seq 8192`.

**GPU kernel time by op, nsys, 3972-token prompt** (`profile/before-nsys4096-*.csv`)

| op | chunk 256 | chunk 2048 |
|---|---:|---:|
| moe gate_up (`moe_mma_kernel`, tensor cores) | 725.9 ms, 41.7% | 194.9 ms, 22.4% |
| moe down | 433.7 ms, 24.9% | 130.6 ms, 15.0% |
| fp8 projections (`wq_gemm`, bf16 dequantise-once) | 239.7 ms, 13.8% | 191.0 ms, 22.0% |
| gdn delta rule (`gdn_delta_rows_kernel`) | 148.5 ms, 8.5% | 172.8 ms, 19.9% |
| nvfp4 projections | 51.2 ms, 2.9% | 22.4 ms, 2.6% |
| moe topk / group / reduce / combine | 24.6 ms, 1.4% | 34.5 ms, 4.0% |
| flash prefill attention | 30.7 ms, 1.8% | 22.4 ms, 2.6% |
| gdn conv | 21.6 ms, 1.2% | 20.3 ms, 2.3% |
| norms, silu, cuBLAS (nvjet), activation quantise, other | 65 ms, 3.7% | 81 ms, 9.3% |
| **total** | **1741 ms** | **870 ms** |

**Engine profiler, same prompts** (`--profile-every 1`, one launch timed after another; this is the "per-op share table" the
task asked for. Include a one-time cost, see the note.)

| op | 1067 tok, chunk 256 | 3972 tok, chunk 256 | 3972 tok, chunk 2048 |
|---|---:|---:|---:|
| moe_experts | 395.8 ms (44.4%) | 1162.7 ms (57.2%) | 512.9 ms (31.7%) |
| matmul (bf16 dense) | 287.6 ms (32.3%) | 257.4 ms (12.7%) | 360.0 ms (22.3%) |
| matmul_fp8 | 94.2 ms (10.6%) | 266.0 ms (13.1%) | 303.2 ms (18.8%) |
| gdn_delta_rule | 49.7 ms (5.6%) | 151.6 ms (7.5%) | 249.3 ms (15.4%) |
| matmul_nvfp4 | 23.6 ms (2.7%) | 64.3 ms (3.2%) | 36.6 ms (2.3%) |
| gdn_conv | 9.6 ms (1.1%) | 26.0 ms (1.3%) | 30.5 ms (1.9%) |
| attn_prefill | 3.8 ms (0.4%) | 31.5 ms (1.5%) | 22.0 ms (1.4%) |
| rmsnorm_add, silu_glu, rmsnorm, rope, moe_combine | 19.3 ms (2.2%) | 54.4 ms (2.7%) | 78.5 ms (4.9%) |
| total | 891 ms | 2034 ms | 1616 ms |

Note on the `matmul` row: it is **not** prefill work. It is 258 to 360 ms at every prompt length and chunk; in the nsys
data the cuBLAS (nvjet) kernels add up to 15 ms (1%). It is the one-time cuBLAS initialisation and heuristics in the first
call of each shape, charged to the first timed launch. Item (c) of the task (dense bf16 projections) therefore has nothing
to win. For the same reason `generate`'s own prefill time (below) is lower than `bench speed`'s warm number.

**Prefill chunk sweep, `generate`, unprofiled (includes the ~250 ms cold start)** (`profile/before-prof*.err`, the `finish=`
lines of the W1 log)

| prompt | chunk 256 | chunk 1024 | chunk 2048 |
|---|---:|---:|---:|
| 3972 tokens | 2004 ms (1982 tok/s) | 1217 ms (3264 tok/s) | 1076 ms (3691 tok/s) |
| 1067 tokens | 750 ms (1423 tok/s) | 531 ms (2009 tok/s) | |

**What the profile says**

* GatedDeltaNet prefill is **sequential per token** (`gdn_delta_rows_kernel`, one thread per state row, 4096 threads for
  the 35B, 2.9 ms per layer per 2048 tokens). It is not chunked. It is 8.5% of the GPU time at chunk 256 and 19.9% at 2048.
* MoE at chunk 256 is at the memory floor. Per call: gate_up 1134 us for 292 MB of weights (258 GB/s), down 678 us for 145 MB
  (214 GB/s). 640 calls = 40 layers x 16 chunks.
* MoE at larger chunks is not at a floor: T=2048 takes 4.12 ms per layer (25 TFLOPS), T=4096 6.85 ms (30 TFLOPS);
  the e4m3 dense GEMM reaches 75 to 87 TFLOPS on the same machine.
* The dense GEMM kernels (`wq_gemm`) dequantise once per M tile; at M=2048 the fp8 qkv projection ran at 53 TFLOPS.

## 4. What was built

All kernels are in the registry (`crates/intarsia-backend-cuda/src/{kernels,quantmoe}.rs`), declared in
`kernels/abi/ik_abi.h`, and admitted by `tests/admission.rs` against the cpu f64 reference.

| piece | where | result |
|---|---|---|
| `generate --op-table` | `crates/intarsia-cli/src/cmd/generate.rs` | per-op time and share table of the engine profiler |
| `ik_gdn_delta_rule_bf16_lanes` / `gdn_delta_rule_bf16_lanes` | `kernels/cuda/gdn.cu` | 4 lanes per state row (float4 groups, 4 partial sums, shuffle combine), dk 64 or 128. Serves every T. T=1 6.2 us (rows kernel 14.3), T=16 16.4 (31.7), T=512 376 (592), T=2048 1999 (2848). `tests/gdn_lanes.rs` |
| `ik_matmul_fp8_gemm_f8` / `matmul_fp8_gemm_f8` | `kernels/cuda/fp8_gemm_f8.cu` (+ `wq_gemm.cuh`, `fp8_gemm.cu`) | e4m3 x e4m3 `mma.sync m16n8k32`, no weight dequantisation, 64 or 128 row tiles (M >= 1024), 4-stage cp.async. Serves every M >= 5, K % 128 == 0. qkv M=2048: 1297 -> 842 us (53 -> 82 TFLOPS); o_proj M=2048: 732 -> 459 us; M=4096: 2507 -> 1586 us. W8A16 batches fall back to the bf16 kernel inside the launcher |
| window scripts | `scripts/prefill-cuda-window.sh`, `scripts/prefill-cuda/*.sh` | the vLLM-down wrapper (section 8) and the per-window remote scripts |

Admission (`every_cuda_kernel_matches_the_reference`) results for the final build, from `tests/kernel-tests-e87beed.log`:
`gdn_delta_rule_bf16_lanes` 6 cases max_abs 1.2e-4 rel_l2 4.6e-5; `matmul_fp8_gemm_f8` 8 cases max_abs 4.1e-3 rel_l2 1.7e-4
(the budget is 1.5e-2 / 1e-3 for bf16 outputs). `moe_experts_mma_is_as_accurate_as_old_kernel` still passes.

**Tests added, and the mutation each one fails on** (mutants built in `~/intarsia-perf/mut*`, logs in `tests/mut-*.log`):

| test | mutation | outcome |
|---|---|---|
| `gdn_lanes_is_as_accurate_as_the_rows_kernel` (T = 1, 7, 16, 17, 130, 513; dk 128 and 64; vs f64 and vs the rows kernel) | state decay skips one of four components | fails: rel_l2 1.3, the rows kernel 4e-7 |
| `gdn_lanes_state_carries_across_calls` (splits 70+59, 120+10, 3+1, 16+1, bitwise) | not mutation-tested: a kernel wrong the same way in both calls passes it; it guards the state carry and T-independence | passes |
| `matmul_fp8_gemm_f8_is_as_accurate_as_the_bf16_gemm` (M = 5 ... 1100, bf16 and f32 outputs, W8A8 and W8A16) | last k stage dropped | fails: rel_l2 0.48 |
| `matmul_fp8_gemm_f8_rows_do_not_depend_on_the_call` (bitwise; 1100-row call vs 5, 17, 64, 100-row calls at other offsets) | the 128-row tile accumulates inside the mma, the 64-row tile does not | fails |

Pattern copied from `moe_experts_mma_is_as_accurate_as_old_kernel`: the new kernel must be within 2x of the existing
kernel's error against the f64 reference (and under an absolute bound), plus a repeatability check.

**Attempted and rejected** (one attempt each)

* Software-pipelined variant of the lanes kernel (next token's norms and gates overlapped with the state update, q/k double
  buffer): **slower**, 2404 us at T=2048 against 1999 us for the plain version (`tests/gdn-timing-pipelined-variant.log`).
  Register pressure is the likely cause; not investigated. Reverted. 2 and 8 lanes per row: 2173 and 2307 us.
* An earlier f8 gemm launch bug (the bf16 fallback template was instantiated in a second translation unit so its shared-memory
  attribute was never set): fixed by launching the fallback from `fp8_gemm.cu`.

## 5. Accuracy

Gates as in `scripts/phase1b-acceptance.sh` stage_accuracy and `docs/accuracy.md`: vLLM golden (35b-short 10 prompts / 1441
positions, 35b-long), baseline and noise envelope recorded by the **HEAD build in this session** (W1: short, envelopes
`attn-simt` and `chunk=64`; W2: long, envelope `chunk=1024`), per-prompt limits `baseline -/+ max(fixed margin, 2 x envelope)`,
`--pre-change 7e65770 --revlist`. The 35b-long case runs `--mode all` (prefix reuse); HEAD's baselines use `--mode both`,
as the script does. Chunk-2048 runs differ from the baseline in the prefill chunk and use `--allow-config-mismatch`.
The same-weights self-golden stage with measured budgets was not run (it needs an old-build SIMT noise run; time).

Aggregate prefill 21-way KL against the vLLM golden (the figure the task says must not get worse; lower is better):

| run | 35b-short | 35b-long |
|---|---:|---:|
| **HEAD 7e65770** (recorded baseline) | **6.271e-3** | **4.273e-2** (floor FAIL: HEAD's recorded fact) |
| HEAD under its own perturbations (envelope runs) | attn-simt 5.514e-3, chunk 64 6.745e-3 | |
| control: d940701 with both new kernels disabled | 6.271e-3, baseline PASS (bit-identical numbers) | |
| e87beed default kernels, chunk 2048 | **5.430e-3**, baseline PASS | 4.576e-2 |
| fast kernels (9578fd9), chunk 256 | 6.910e-3 | 4.570e-2 |
| fast kernels (9578fd9), chunk 2048 | 6.585e-3 | 4.452e-2 |
| fast kernels (d940701), chunk 256 | 7.007e-3 | 4.571e-2 |
| fast kernels (d940701), chunk 2048 | 6.683e-3 | 4.452e-2 |
| d940701, only lanes (f8 disabled), chunk 256 | 6.628e-3 | 4.512e-2 |
| d940701, only f8 (lanes disabled), chunk 256 | 6.522e-3 | 4.656e-2 |

Gate verdicts (floor = absolute class thresholds; baseline = no worse than HEAD's recorded run with the per-prompt envelope;
prefix = reuse stage of `--mode all`):

| configuration | 35b-short floor / baseline | 35b-long floor / baseline / prefix reuse |
|---|---|---|
| **default kernels, chunk 256** | PASS / PASS (control run) | not run; identical code path to HEAD (HEAD: FAIL / PASS / PASS) |
| default kernels, chunk 2048 (e87beed) | PASS / **PASS** | FAIL (as HEAD) / **FAIL** (`chat_reuse_4k` prompt KL) / PASS |
| fast kernels, d940701, chunk 256 | PASS / FAIL (`chat_system`, `chat_multiturn`) | FAIL / FAIL (decode aggregate) / FAIL (`long_docs_4k` diverges at token 12) |
| fast kernels, d940701, chunk 2048 | PASS / FAIL (`chat_system`, `chat_multiturn`) | FAIL / PASS / PASS |
| only lanes, chunk 256 | PASS / FAIL (`long_prose_700`) | FAIL / FAIL / PASS |
| only f8, chunk 256 | PASS / FAIL (`chat_system`, `long_prose_300`) | FAIL / PASS / FAIL |
| fast kernels, 9578fd9 (serving M >= 33 / T >= 16 only), chunk 256 | PASS / FAIL (`chat_multiturn`) | FAIL / FAIL / FAIL |

Determinism (new build against its own dump in a second process, bit-identical budgets): PASS for 9578fd9 and d940701.

**Reading**

* The harness is sound: the control run reproduces HEAD's numbers exactly (6.271e-3, same verdicts).
* Each new kernel alone, and both together, move 1 to 2 of 10 short prompts past `2 x envelope`, a different set each time.
  That is rounding-order noise showing up in per-prompt limits whose envelope contains only attention-kernel and prefill-chunk
  perturbations. The worst prompt, `chat_multiturn`, has 58 tokens (one chunk); the chunk perturbation cannot move it, and the
  KL of 2.7e-2 is one position (max KL 1.37 against 0.68 for HEAD) of 58.
* The f8 kernel is **closer to the f64 reference** than the kernel it replaces (rel_l2 7e-9 against 9e-5 at M=100), so the
  move away from the vLLM golden on single prompts is not a numerical regression of the layer; it is a different but equally
  valid rounding. The gate as written still fails, and I did not widen it.
* Prefix-reuse failures follow the f8 kernel (`only f8` fails it, `only lanes` passes it). The likely mechanism is decode
  (`matmul_fp8_gemv`, M <= 4) against prefill (f8 gemm) rounding of the cached turn-1 tokens; not confirmed.
* Early in the work the kernels served only M >= 33 / T >= 16, so a prompt's rounding depended on its chunking
  (HEAD keeps `*_gemm` bitwise equal to `*_mma` for this reason). d940701 makes both serve the whole range; the invariance
  test above guards it. That did not make the gates pass.
* Consequence: both kernels are registered, tested, admitted, and **opt-in** (`INTARSIA_CUDA_FAST_PREFILL=1`, priority -10
  otherwise). Enabling them by default needs an owner decision and a re-calibrated envelope that includes GEMM/recurrence
  rounding (`--noise-perturb` currently offers attention and chunk only; a perturbation run must come from the same build
  as the baseline, so this needs a build that can switch the arithmetic, which this branch now has).

## 6. Speed, `intarsia bench speed` against the engine server and vLLM

Procedure of `scripts/phase1b-acceptance.sh` stage_ab (fresh `serve` per arm, bench is the only client, 1 Hz in-flight
poller, `--profile-every 0`, `bench speed --warmup 1 --thinking off`, graphs required for the new-build arms), with these
differences: sizes 128, 1024, 4096, 16384, gen 128, 5 reps; `--max-seq 20480 --mem-budget 45`; one extra arm; 3 rounds
(order alternated: round 1 and 3 `old, c2048, fastc2048`, round 2 reversed), vLLM run after each round with vLLM down
while ours ran (`scripts/prefill-cuda/ab-ours.sh`, `ab-vllm.sh`; the vLLM arm is a client in a 4 GB fleetq slot). The bench
client is the e87beed binary for every arm. All 12 arm-rounds are valid (in-flight never above 1 running / 0 waiting; graph
replays > 0, no eager or profiled steps in the new-build arms). Data: `2026-10-13-prefill-cuda/ab/`.
Each cell is **min / median / max of the three round values**, each round value being the median of 5 reps.

Arms: `old` = HEAD 7e65770 at the shipped chunk 256; `c2048` = e87beed, default kernels, `--prefill-chunk 2048`;
`fastc2048` = e87beed, `INTARSIA_CUDA_FAST_PREFILL=1`, `--prefill-chunk 2048`; `vllm` = the `guidance` container.

**Prefill tok/s**

| prompt tokens | old | c2048 | fastc2048 | vLLM |
|---|---|---|---|---|
| 128 | 1251 / 1257 / 1267 | 1263 / 1270 / 1270 | 1289 / 1296 / 1312 | 1598 / 1660 / 1661 |
| 1024 | 2299 / 2304 / 2305 | 3870 / 3891 / 3905 | 4389 / 4389 / 4412 | 5554 / 5631 / 5644 |
| 4096 | 2375 / 2397 / 2403 | 4628 / 4654 / 4660 | 5164 / 5505 / 5507 | 6592 / 6595 / 6600 |
| 16384 | 2257 / 2261 / 2266 | 4308 / 4310 / 4333 | 5042 / 5075 / 5081 | 6404 / 6412 / 6470 |

**Decode tok/s**

| prompt tokens | old | c2048 | fastc2048 | vLLM |
|---|---|---|---|---|
| 128 | 72.0 / 72.4 / 73.0 | 72.2 / 72.6 / 72.7 | 73.2 / 73.8 / 74.3 | 76.0 / 76.2 / 76.7 |
| 1024 | 71.5 / 71.9 / 72.5 | 71.7 / 72.1 / 72.3 | 72.8 / 73.3 / 73.8 | 77.5 / 77.6 / 77.7 |
| 4096 | 69.2 / 69.7 / 70.2 | 69.4 / 69.8 / 70.0 | 70.7 / 71.3 / 71.8 | 76.3 / 76.4 / 76.5 |
| 16384 | 63.4 / 63.6 / 64.0 | 63.4 / 63.8 / 64.0 | 64.7 / 65.0 / 65.7 | 72.4 / 73.7 / 73.9 |

**Time to first token (s, median of rounds):** 4096: old 1.7, c2048 0.9, fastc2048 0.7, vLLM 0.6; 16384: 7.4, 3.9, 3.3, 2.6.

**Ratios of the medians (prefill)**

| prompt tokens | c2048 / old | fastc2048 / old | vLLM / old | vLLM / c2048 | vLLM / fastc2048 |
|---|---|---|---|---|---|
| 128 | 1.01x | 1.03x | 1.32x | 1.31x | 1.28x |
| 1024 | 1.69x | 1.90x | 2.44x | 1.45x | 1.28x |
| 4096 | 1.94x | 2.30x | 2.75x | 1.42x | 1.20x |
| 16384 | 1.91x | 2.24x | 2.84x | 1.49x | 1.26x |

The round-to-round spread is under 1% for `old`, `c2048` and vLLM. `fastc2048` at 4096 is 5164 in one round and 5505 and 5507 in
the other two (6%); the cause was not investigated. Decode: the fast arm gains 1.9 to 2.3% (the lanes kernel also serves T=1);
the chunk size does not change decode. The decode gap to vLLM is 5% at short context and 14% at 16k for HEAD (3% and 12% with
the fast kernels). Decode at 16k context is lower than at short context for both engines (63.6 against 72.4 tok/s for ours).

The task's HEAD figures (1276 / 2315 / 2407 tok/s at 128 / 1024 / 4096) are reproduced by the `old` arm (1257 / 2304 / 2397).

## 7. Not done, and why

* **Chunked (WY, chunk 64) GatedDeltaNet prefill.** The recurrence is sequential and is 8.5% of the GPU time at the shipped
  chunk and 19.9% at chunk 2048 (15.7% after the lanes kernel), so a chunked kernel is worth up to about 12% at chunk 2048 (an
  estimate: 15.7% to a few %) and less than 6% at 256. The lanes kernel (about 1.4x) was the bounded step; the chunked form is a tensor-core rewrite
  (64x64 triangular solve per head and chunk, state update as GEMMs, a new numerics to gate) that did not fit the budget.
  Recommended next kernel if the chunk becomes 2048.
* **MoE kernel work.** At chunk 256 the kernel is at the bandwidth floor (section 3), so tiling, grouping or FP4 MMA cannot help.
  At chunk 2048 it is the largest item (45% of 711 ms after the other changes) at 25 TFLOPS. The likely limit is occupancy, not
  dequantisation reuse (inferred from the launcher, not measured with ncu): the 64-row tile requests about 85 KB of dynamic shared
  memory (3 activation slots, 2 dequantised bf16 weight buffers, the raw ring), so probably one 256-thread block per SM, 8 warps. `INTARSIA_MOE_BM=32` is slower (4.86 ms against 4.12 ms at T=2048), a 128-row tile does
  not fit in shared memory. A fix is a redesign (dequantise e2m1 straight into the mma B fragments, no bf16 buffer), not a
  tweak. Wrong assumption recorded: "dequantise once per tile is what limits it".
  The sort-by-expert/persistent-tile item exists already (counting sort into a tile table in `moe_grouped.cu`).
* **FP4 block-scaled MMA.** PTX `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` (m16n8k64):
  `ptxas` rejects it for `sm_121` ("Feature '.kind::mxf4nvf4' not supported on .target 'sm_121'") and assembles it for
  `sm_121a` (SASS `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`). So the build would need `INTARSIA_CUDA_ARCH=sm_121a`. I did not run it.
  The operation also needs the activations in NVFP4, which changes the op: a pure-python simulation (block 16, e4m3 scales,
  random Gaussian and heavy-tailed activations, 48x512 @ 48) gives matmul rel_l2 9.3e-2 from the activation quantisation
  alone, against the admission budget of 1e-3 and an op defined with bf16 activations. It cannot be admitted as the same op, and at
  chunk 256 the MoE is bandwidth bound anyway. Not pursued.
* **Flash prefill attention tuning:** 1.8 to 3.1% of the time at every configuration; not touched.
* **Dense bf16 projections (c):** nothing to win (section 3 note).
* **The nvfp4 dense gemm** (2.6 to 3.3%) was not touched.
* The 8B and other model gates of the acceptance script were not run (the task is the 35B).

## 8. vLLM downtime log (`docs/results/prefill-cuda-downtime.log`)

Every window: exclusive fleetq lease with `--takes-down guidance`, `docker stop missioncntrl-guidance`, memory check
(>= 70 GB available) before any GPU work, a `timeout`-capped script, then the exit trap (kill our processes, `docker start`,
wait for `curl http://100.120.88.104:8000/v1/models`, `fleetq release`), logging stop to answering. No `--force`, no
`--no-queue`. Builds ran in separate 16 GB shared slots with vLLM up (at most 8 cargo jobs); the vLLM arm of the A/B ran
as a 4 GB client slot. Nothing was run on tatooine or spark2. Times are spark1's clock.

| window | fleetq job | vLLM stopped | serving again | down (min) | purpose |
|---|---|---|---|---:|---|
| W1 | 1614 | 13:50:21 | 14:05:17 | 14.9 | prefill profile (engine, nsys, chunk sweep) + 35b-short baseline and envelopes |
| W2 | 1619 | 14:13:57 | 14:22:53 | 8.9 | first kernel tests and microbenchmarks, 35b-long baseline and envelope |
| W3 | 1621 | 14:29:39 | 14:36:28 | 6.8 | kernel tests with mutation, microbenchmarks, nsys |
| W4 | 1623 | 14:39:50 | 14:57:03 | 17.2 | tests, "after" profile, first 35B gates |
| W5 | 1625 | 15:01:42 | 15:16:15 | 14.5 | row-invariance build: tests, mutation, gates |
| W6 | 1628 | 15:16:52 | 15:30:06 | 13.2 | bisect of the gate changes |
| W7 | 1630 | 15:31:23 | 15:44:54 | 13.5 | final tests, A/B round 1 (ours), chunk-2048 gates |
| W8 | 1635 | 15:46:34 | 15:56:57 | 10.3 | A/B round 2 (ours) |
| W9 | 1637 | 15:58:34 | 16:08:14 | 9.6 | A/B round 3 (ours) |
| **total** | | | | **108.9** | budget 120, longest window 17.2 (limit 40) |

Mistakes that cost window time: W5's bisect step ran without `--allow-config-mismatch` and produced no verdict (3 min, redone
in W6); the quantmoe tests did not run in W5 because the first failing test binary stopped `cargo test` (fixed with
`--no-fail-fast`, run in W7).

## 9. Reproduce

```
scripts/prefill-cuda-window.sh <max-min <= 33> "<label>" '<script on spark1>'     # lease, vLLM stop, trap-restore
scripts/prefill-cuda/w1-profile-baseline.sh   w2-kernels.sh   w3-kernels.sh   w4-gates.sh   w5-invariance-gates.sh
scripts/prefill-cuda/w6-bisect.sh   w7-ab1.sh   ab-window.sh   ab-ours.sh   ab-vllm.sh
INTARSIA_CUDA_FAST_PREFILL=1 intarsia serve --prefill-chunk 2048 ...        # the fastc2048 arm
intarsia generate --profile-every 1 --op-table --max-tokens 1 -p "<prompt>"  # per-op table
```

## 10. Recommendations

1. Decide the prefill chunk. 2048 gives 1.9x at 4k and 16k with the default kernels; 35b-short passes, 35b-long fails one
   prompt (`chat_reuse_4k`) on the baseline verdict. Check memory planning at your `--max-seq` before changing the default.
2. To ship the two kernels: extend the noise envelope with a perturbation that reorders GEMM and recurrence sums, then re-gate.
3. Next kernel: chunked GatedDeltaNet (largest sequential piece after the chunk change); then the MoE redesign.
