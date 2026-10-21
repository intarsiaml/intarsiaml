# CUDA prefill, round 2: chunked (WY) GatedDeltaNet, Qwen3.6-35B-A3B NVFP4 on spark1 (GB10, sm_121), 2026-10-17

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** C22: the chunked GatedDeltaNet kernel is NOT on main (the status line is right); section 1 item 4 ("ships opt-in, off by default", `INTARSIA_CUDA_GDN_CHUNKED=1`) describes the branch `perf/cuda-prefill-2` only, and no such variable exists in the code on main. The ptxas probe in section 8 must use `sm_121a`: plain `sm_121` is rejected (see `2026-10-17-cuda-moe-nvfp4-mma.md`, `crates/intarsia-backend-cuda/build.rs`). The vLLM figures quoted here (5338 / 6334 / ~6400 tok/s) were not re-measured and are not comparable with the measured ones in `2026-10-18-cuda-moe-w4a16.md` (5599 / ~6570 / ~6450).

> **Status (2026-10-17): kernel not merged; kept on branch `perf/cuda-prefill-2` at 287c371.** Only this page, its raw data and `scripts/cuda-prefill2/` are on main. Reason: `gdn_delta_rule_bf16_chunked` takes 1.3 to 1.5x as long as the existing opt-in lanes kernel at every measured T (2633 vs 1998 us at T=2048, 4975 vs 3838 us at 4096); with `INTARSIA_CUDA_FAST_PREFILL=1` and `INTARSIA_CUDA_GDN_CHUNKED=1` both set, chunked wins dispatch over lanes, so enabling both is slower; and its prefill KL is outside base's spread in 2 of 4 cells. The scripts under `scripts/cuda-prefill2/` need that branch's `kernels/cuda/gdn_chunked.cu` and `tests/gdn_chunked.rs` to reproduce section 6. The top lever is still section 8 item 1, MoE experts on NVFP4 block-scaled MMA: a probe on 2026-10-17 (fleetq #2400) showed CUDA 13.0.88 compiles `kind::mxf4nvf4.block_scale` to SASS `OMMA.SF.16864` for `-arch=sm_121a` (not plain `sm_121`, the build default).


Branch `perf/cuda-prefill-2`, from `perf/cuda-norm-gemv` (85823f7, origin/main df18c6a at the time; rebased onto the then-current
origin/main at the end). Raw logs: `docs/results/2026-10-17-cuda-prefill-2/raw/{w1,w2,w3}`; vLLM downtime log
`docs/results/2026-10-17-cuda-prefill-2/downtime.log`; window scripts `scripts/cuda-prefill2/`.

## 1. Result in five lines

1. Built `gdn_delta_rule_bf16_chunked` (fp32 SIMT, WY chunk form, 64 tokens per chunk) for the GatedDeltaNet prefill. Registered with
   admission (ADMITTED, rel_l2 5.6e-5), the sequential kernels stay as fallback.
2. Speed A/B against a `perf/cuda-norm-gemv` build, 3 interleaved rounds: prefill **+3.1% / +3.1% / +3.2%** at 1024 / 4096 / 16384
   tokens, decode unchanged (80.9 / 78.1 / 70.9 tok/s both arms). Positive in every one of the 3 rounds at every size.
3. That closes about 3 points of a 25 to 30% gap to vLLM. It is not the lever that closes it (section 3).
4. **Ships opt-in, off by default** (`INTARSIA_CUDA_GDN_CHUNKED=1`). The default-on accuracy bar is not met (section 5) and the
   other default-on gates were not run.
5. The largest lever is the MoE expert GEMMs (39.7% of prefill kernel time at about 25 TFLOPS). That is candidate (a), not built here;
   section 7 is the plan.

## 2. Profile (step 1)

`nsys profile -t cuda` of one 4160-token prefill (prompt = first 14000 bytes of docs/accuracy.md, 1 generated token), base tree
85823f7, default config, chunk 2048 (chunks of 2048 + 2048 + 64 tokens), alone on the GPU with vLLM stopped.
Raw: `raw/w1/kern_cuda_gpu_kern_sum.csv`. Kernel total 926.0 ms (the run's wall prefill was 1347 ms under nsys: 3088 tok/s).

| op | kernels | ms | share | calls | avg per call | achieved |
|---|---|---|---|---|---|---|
| MoE experts | `moe_mma_kernel` 4 variants | 367.5 | 39.7% | 240 | 1.6 to 2.4 ms (2048-token chunks) | about 25 TFLOPS bf16 mma; 107 GB/s weights (39% of 273) |
| NVFP4/FP8 projections | `wq_gemm_kernel` | 221.6 | 23.9% | 750 | 0.50 ms (large) / 0.07 ms | not derived: no shapes in the CSV |
| GatedDeltaNet recurrence | `gdn_delta_rows_kernel<128>` | 176.2 | 19.0% | 90 | 1.96 ms per 2048 tokens | about 20 GB/s (7% of 273); about 1 us per token, latency bound |
| MoE reduce + combine | `moe_reduce`, `moe_combine` | 32.9 | 3.6% | 240 | | |
| norms | `rmsnorm`, `gated_rmsnorm` | 30.1 | 3.3% | 330 | | |
| attention | `flash_kernel` | 23.9 | 2.6% | 30 | 0.80 ms | |
| GDN conv | `gdn_conv_kernel` | 20.8 | 2.2% | 90 | 0.23 ms | |
| activation quant | `fp8_actq_kernel` | 19.6 | 2.1% | 390 | | |
| other (gate split, rope, ...) | | 33.4 | 3.6% | | | |

How the achieved numbers were derived. MoE FLOPs: 2048 hidden, 512 expert intermediate, top-8 routed, 40 layers (3 matrices per
expert, 2 flops per MAC): 8 x 3 x 2 x 2048 x 512 = 50.3 MFLOP per token per layer, x 4160 x 40 = 8.4 TFLOP, over 367.5 ms = 22.8
TFLOPS (about 25 over just the two 2048-token chunks, which also use the big tiles). The intermediate size and top-8 are from the
public model config and agree with the 437 MB per layer of expert weights in `2026-10-13-prefill-cuda.md`; I did not re-read them from
the checkpoint. Weight traffic is 437 MB per layer per chunk over 4.08 ms (gate/up 2.44 ms + down 1.64 ms) = 107 GB/s. GDN traffic:
qkv 2048 x 8192 x 2 B = 34 MB plus the f32 state in and out (2 x 2.1 MB) about 40 MB per call, over 1.96 ms = 20 GB/s.

Peaks and their source. NVIDIA's DGX Spark page (https://www.nvidia.com/en-us/products/workstations/dgx-spark/) lists 273 GB/s memory
bandwidth and "up to 1 PFLOP FP4"; it does not say whether the FP4 figure is with sparsity and gives no BF16 or FP8 figure. Under the
usual 2:4-sparsity-and-2x-per-precision assumption (my assumption, not on that page) that is 500 TFLOPS dense FP4, 250 FP8, 125 BF16,
so 25 TFLOPS is about 20% of the bf16 mma peak and 5% of the dense FP4 one. Treat the TFLOPS percentages as an estimate.

## 3. The pick, and why it was the wrong size of lever

I picked (b), chunked GatedDeltaNet. Reasoning at the time: 19% of the profile, one self-contained kernel, testable against the cpu
reference, and no activation-quantisation question (which (a) has). I wrote the kernel after seeing the top rows of the W1 profile and
before measuring anything about the kernel itself. Candidate (a), FP4 block-scaled MMA for the experts, is the bigger lever (39.7%)
but needs a new activation quantiser, a new kernel and a KL validation, which does not fit the 40 minutes of vLLM downtime together
with a first-of-its-kind accuracy question. Candidate (c), attention, is 2.6% and was ruled out by the profile.

What I did not establish: what CUDA 13.0.88 exposes for `kind::mxf4nvf4.block_scale` on sm_121, and which MoE backend vLLM picks
on GB10. My read-only survey of the running vLLM container's log and of the cutlass/flashinfer trees on spark1 did not produce an
answer I can cite, so section 7 lists both as open.

The wrong assumption, recorded. I assumed that turning the 64-step token-serial chain into one chunk step would be a large win
because the sequential kernels are latency bound (20 GB/s, 1 us per token). It is not: each chunk step has its own dependent latency
(global tile loads, about 14 barriers, a 64-step forward substitution) and the state kernel runs only `dv/32 x Hv` = 128 blocks of
128 threads on 48 SMs. Measured per call at the 35B shape (W2 `gdn_timing`, `raw/w2/t-gdn_timing.log`):

| T | rows (default) | lanes (opt-in fast) | chunked (this) |
|---|---|---|---|
| 512 | 598 us | 377 us | 577 us |
| 2048 | 2802 us | 1998 us | 2633 us |
| 4096 | 5590 us | 3838 us | 4975 us |

So the chunked kernel is 3 to 11% faster than the default rows kernel and takes 1.3 to 1.5x as long as the existing opt-in lanes kernel.
The nsys split of the microbench (`raw/w3/gt_cuda_gpu_kern_sum.csv`, mixed T) puts 36% of its time in the prep kernel (Gram products
and the triangular solve, 275 us average) and 64% in the state kernel (491 us average). ptxas: 79 / 96 registers, no spills.

The end-to-end gain (+3.1%) is larger than the microbench predicts (about 1.2%: 6% of the 19% GDN share). The sign and size repeat
across 3 rounds x 3 sizes, so I report it, but I do not have an explanation for the difference (candidates: the rows kernel is slower
in situ than in isolation; nothing else changed). Not investigated.

One attempt at the idea, no second attempt: a second variant would be a guess without a per-kernel split in situ, and the plan in
section 7 says what to change.

## 4. What was built

* `kernels/cuda/gdn_chunked.cu`: `ik_gdn_delta_rule_bf16_chunked`, same contract as `ik_gdn_delta_rule_bf16` plus plan scratch of
  `ceil(T/64) x Hv x 8512` floats. dk in {64, 128}, dv % 32 == 0. fp32 SIMT. Deterministic (fixed summation order, no atomics).
  Not bitwise equal to the sequential kernels, and not bitwise independent of where a sequence is split into calls (the chunk
  boundaries move with the split).
* `crates/intarsia-backend-cuda/src/{kernels,device,ffi}.rs`: table entry, shape gate (`64 <= T <= 8192`, dk 64/128, dv % 32 == 0),
  scratch size, dispatch. Priority 15 when `INTARSIA_CUDA_GDN_CHUNKED=1`, else -10 (the sequential kernels win). With
  `INTARSIA_CUDA_FAST_PREFILL=1` also set, the chunked kernel wins over `lanes` for T >= 64. Calls below 64 tokens and decode keep
  the sequential kernels.
* `kernels/abi/ik_abi.h`: one declaration (shared header; see section 9).
* `crates/intarsia-backend-cuda/tests/gdn_chunked.rs` (3 tests) and the chunked kernel added to `tests/gdn_timing.rs`.
* A numpy float64 check of the WY algebra against the sequential recurrence (chunk 64, tail chunks, decays 0 to -6): max relative
  difference 1e-15, in the session scratchpad (not committed). It validated the maths before the CUDA was written.

## 5. Accuracy and the default-on bar (step 5)

The bar: aggregate prefill KL21 against the vLLM golden (sha b6b51e1b1699c66c, `golden/qwen36-35b-a3b-nvfp4-vllm-*top20`) on
35b-short and 35b-long no worse than base's, beyond base's measured chunk-perturbation spread; plus the 8B fp32 goldens, the
prefix-reuse gate and graph == eager ids.

How measured. `intarsia accuracy --mode prefill --backend cuda --prefill-chunk N` on both goldens, deterministic, vLLM stopped.
Base (85823f7) at chunks 256 / 1024 / 2048 gives base's spread; dev with `INTARSIA_CUDA_GDN_CHUNKED=1` at 256 / 2048. The
quantity is the "21-way KL" of the OVERALL row (second KL column).

| golden | base c256 | base c1024 | base c2048 | base range | dev c256 | dev c2048 |
|---|---|---|---|---|---|---|
| 35b-short | 6.271e-3 | 5.430e-3 | 5.430e-3 | 5.430e-3 .. 6.271e-3 (0.84e-3) | 6.015e-3 | 6.306e-3 |
| 35b-long | 4.273e-2 | 4.421e-2 | 4.576e-2 | 4.273e-2 .. 4.576e-2 (0.30e-2) | 4.495e-2 | 4.636e-2 |

Verdict: **not met, by a hair, so opt-in.** Dev is inside base's spread in 2 of 4 cells; the two outside are 35b-short c2048
(6.306e-3 vs base max 6.271e-3, +0.6%) and 35b-long c2048 (4.636e-2 vs 4.576e-2, +1.3%). Both are far smaller than the spread, but
"no worse than base beyond the spread" is not demonstrable with 2 dev points per golden and a miss at the edge. Short top-1 is 98.33 /
98.40% (base 97.92 / 97.99%). The 35b-long golden fails the 3.0e-2 KL21 gate on base too (4.4e-2 to 4.6e-2, pre-existing; the base
logs `raw/w1/acc-long-*.log` show it), so that failure is not caused by this change.

Not run (so nothing here is claimed): the 8B fp32 goldens (the dense 8B has no GatedDeltaNet, so it cannot change), the prefix-reuse
gate, graph == eager ids. The reuse gate is the one that matters for anyone enabling this: the chunk boundaries move with where a
prompt is split, so a warm run is not bit-equal to a cold one by construction. Prefill is not captured in a graph and the kernel
requires T >= 64, so graph vs eager decode ids cannot depend on it, but that is reasoning, not a measurement. Window time was
better spent not bringing vLLM down again for gates of a kernel that is not going default-on.

## 6. Kernel tests and mutations (step 4)

Window W2, `raw/w2/`. `tests/gdn_chunked.rs`, 3 tests, pass:

* `gdn_chunked_is_as_accurate_as_the_rows_kernel`: 4 shapes (including 16 key / 32 value heads of 128) x token counts around the
  64-boundaries x two decay regimes, against the cpu f64 reference and against the rows kernel; repeatability bit for bit.
* `gdn_chunked_handles_correlated_keys`: keys nearly parallel, beta near 1, almost no decay (worst conditioning of I + L).
* `gdn_chunked_state_carries_across_calls`: 5 splits that cut a chunk and that do not, output and final state against the f64
  reference of the whole sequence.

Worst chunked rel_l2 against the f64 reference: 1.4e-4 on the bf16 output (rows kernel 5e-5; the reference rounds to bf16, so any
fp32 reassociation shows up as one-ulp flips) and about 1e-6 on the f32 state. The bound is 2x the rows kernel's error with a floor of
2.5e-4 on the output and 5e-5 on the state (the 5e-5 floor of the first draft was raised in the second commit, before any GPU run, since an fp32 reassociation on
a bf16-rounded reference cannot be expected to stay at the rows kernel's level; the
floor was fixed before the first GPU run, from the rows kernel's 5e-5 plus the extra fp32 sums of the WY form, and the first GPU run
passed it with the worst case at 1.4e-4; it was not tuned to the mutants).
Admission (`every_cuda_kernel_matches_the_reference`): ADMITTED, max_abs 6.1e-5, rel_l2 5.6e-5.

Mutations (`scripts/cuda-prefill2/mutate.py`, pattern of `scripts/cuda-decode/mutate.py`; a mutant is built from the committed source
by `mutbuild.sh`, run by `mutcheck.sh`): 9 of 9 killed, each fails all 3 tests; the unmutated binary passes.

| mutation | what it breaks | smallest / largest failing rel_l2 |
|---|---|---|
| no-interdecay | `e_last * S0` dropped in the state update | 1.4e-1 .. 1.8 |
| no-kd | `(e_last / e_r)` factor dropped in the state update | 5.8e-2 .. 6.8 |
| mask-strict | M strictly lower: the diagonal term lost | 8.3e-1 .. 9.9e-1 |
| no-solve | T = I (no triangular-solve correction) | 3.0e-2 .. 4.4 |
| no-carry | no state carry between chunks | 7.4e-4 .. 6.4e-1 |
| tail-decay | padded tokens of a tail chunk decay the state | 5.5e-2 .. 1.0 |
| no-intra-decay | no `e_r / e_p` in L | 2.1e-2 .. 2.5e-1 |
| no-qe | output without the `e_r` factor on the S0 term | 9.1e-2 .. 1.7 |
| no-qscale | q without 1/sqrt(dk) | 10.3 |

`no-carry` has the smallest minimum (7.4e-4: the carry test with a long memory where one chunk matters little); it is still
above the 5e-4 limit (2x the output floor) and killed.

## 7. Speed A/B (step 6) and the remaining gap

W3, `raw/w3/w3-window.out`, `raw/w3/ab/`. `bench speed`, a fresh `serve` per arm, recipe default chunk 2048, `--require-graphs`,
gen 128, **2 reps per size** (not 5: the 8-minute script budget; the 3 interleaved rounds were kept: base,dev / dev,base / base,dev),
warmup 1. Prefill tok/s, per round r1, r2, r3, then mean:

| prompt tokens | base (norm-gemv) | dev, `GDN_CHUNKED=1` | delta |
|---|---|---|---|
| 1024 | 3925, 3891, 3896 = **3904** | 4054, 3987, 4038 = **4026** | +3.1% |
| 4096 | 4561, 4708, 4712 = **4660** | 4705, 4865, 4844 = **4805** | +3.1% |
| 16384 | 4374, 4363, 4383 = **4373** | 4514, 4511, 4514 = **4513** | +3.2% |

Decode (128 tokens): 80.9 / 78.1 / 70.9 tok/s base, 80.9 / 78.2 / 70.8 dev (noise).

Against vLLM (5338 / 6334 / ~6400 tok/s from the task statement; not re-measured): gap 27% / 26% / 32% on base, 25% / 24% / 29%
with this kernel on.

## 8. Plan for the rest (what to do next)

1. **MoE experts, 39.7% at about 25 TFLOPS (largest lever).** Weights are NVFP4 already; the open question is the activation side.
   Before writing a kernel: (i) check `ptxas -arch=sm_121` accepts `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X`
   (one `nvcc` compile of a 10-line probe; CUDA 13.0.88 is on spark1); (ii) read which MoE backend the running vLLM selects on GB10
   (`docker logs missioncntrl-guidance | grep -i moe` at startup, or its `VLLM_*` env), my survey did not surface it. If (i) holds, the
   kernel is: quantise the activation tile to fp4 with a per-16 e4m3 scale in the load stage (as vLLM does), block-scaled MMA, same
   grouped-GEMM scheduling as `moe_grouped.cu`. Accuracy bar is the same KL21 one in section 5, measured with the base spread as here.
   Note `acc/moe-mma-small-t` (shelved, cost 7% prefill) touches the same kernel; coordinate.
2. **GatedDeltaNet, if wanted.** The lanes kernel (opt-in fast) already takes 29% off this op. To beat it the chunked kernel needs
   (a) more blocks: split dv into 16-column slices (256 blocks) or two chunks per block pair, (b) tile loads double-buffered (`cp.async`)
   instead of 8 global tile loads per chunk, (c) the 64-step forward substitution replaced by a recursive block inverse on bf16 mma
   (accuracy to be re-measured: it would no longer be fp32). The nsys split says the state kernel (64%) first.
3. Explain the +3.1% vs the 1.2% the microbench predicts, with an in-situ nsys of dev at chunk 2048 (not run; it needs one more
   window of about 2 minutes of script).

## 9. PR handoff (as written for the kernel branch; not applicable on main, where `kernels/abi/ik_abi.h` is unchanged)

Changed files outside `crates/intarsia-backend-cuda` and `kernels/cuda`:

* `kernels/abi/ik_abi.h` **shared code**: one new declaration (`ik_gdn_delta_rule_bf16_chunked`), used only by the CUDA backend. The
  HIP and Metal backends do not implement it; the `every_launcher_is_declared_in_the_abi_header` test is CUDA-side. HIP was not run
  (`cargo check -p intarsia-cli --features hip` type-checks on the Mac only).
* `scripts/cuda-prefill2/` (new: window, build, push, pull, mutation scripts).
* `docs/results/2026-10-17-cuda-prefill-2.md` and `docs/results/2026-10-17-cuda-prefill-2/` (this page, raw logs, downtime log).

No change to core, quant, `model.rs`, fused-op or registry definitions, or recipe keys. The only user-facing switch is the
environment variable `INTARSIA_CUDA_GDN_CHUNKED` (no `[tuning.cuda]` key added). Metal and HIP were not run.

Housekeeping: `~/intarsia-decfuse/out/w3` on spark1 already held files from an earlier branch's window; my script added `gt.*` and
`ab/*` there. I removed the older files from the copy in `raw/w3`; nothing on spark1 was deleted or overwritten except same-named
`ab/35b-{base,dev}-r{1,2,3}` directories.

## 10. vLLM downtime

Stop to serving again, from `downtime.log`: W1 7.9 min (14:46:21 to 14:54:15), W2 7.9 min (15:07:04 to 15:15:00), W3 10.1 min
(15:18:15 to 15:28:25) = **25.9 min of the 40 min cap**, no window over 12 min. Each window used one exclusive fleetq lease with
`--takes-down guidance` through `scripts/cuda-decode/window.sh`, released at the end, no `--force`. After W3, `/v1/models` at
http://100.120.88.104:8000 answered again right after W3 (15:28:25, the window script's own wait) and in a final check at 15:35:05 (HTTP 200, model `guidance`).
W1's log line was written by hand: the window script could not append because the log directory did not exist yet.
