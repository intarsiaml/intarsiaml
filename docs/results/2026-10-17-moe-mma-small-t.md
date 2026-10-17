# Grouped MoE tensor-core kernel: accuracy at small T (2026-10-17)

Branch `acc/moe-mma-small-t` (from `origin/main` c00af21). Follow-up to
`docs/results/2026-10-14-35b-accuracy.md` section 3.4: `moe_experts_mma_bf16` (kernels/cuda/moe_grouped.cu, selected for
T >= 32) was about 2x further from the exact result than the warp-per-row kernel and broke the bounds of
`moe_experts_mma_is_as_accurate_as_old_kernel` in 40 of the 192 cases of `moe_experts_mma_small_t_sweep`.

## 1. Result

| | origin/main | this branch |
|---|---|---|
| small-T sweep (16 seeds x 12 T in 32..64), cases inside the per-case bounds | 152 / 192 | **192 / 192** |
| elements off the cpu reference over the sweep (17.6 M outputs); old warp-per-row kernel: 14 511 | 25 593 (1.76x old) | **3 860 (0.27x old)** |
| `moe_experts_mma_is_as_accurate_as_old_kernel` (T 128 and 512) and the other CUDA kernel tests | pass | pass |
| `intarsia kernels test --backend cuda --model <35B>` | - | 41 kernels, 569 cases, all ADMITTED |
| 35b-short vs vLLM (recorded baseline + envelope) | FAIL (doc 5) | PASS (see 4.2) |
| 35b-long vs vLLM, aggregate prefill KL21 (floor metric) | 4.273e-2 | **4.394e-2 (+2.8 %, not better; see 4.3)** |
| 35b-long prefix-reuse gate | PASS | **FAIL on two prompts (see 4.3)** |
| 8B fp32 goldens | PASS | PASS, identical numbers |
| 35B `bench speed` prefill tok/s at 1024 / 4096 / 16384 | 3929 / 4640 / 4365 | **3619 / 4309 / 4078 (-7.9 / -7.1 / -6.6 %)** |
| 35B decode tok/s at the same prompts | 80.4 / 77.6 / 70.6 | 79.8 / 77.2 / 70.0 (-0.6 %) |

The accuracy fix is complete; **it is not free at large T**: the grouped kernel takes 19 % longer at T = 4096 (8.5 vs 7.1 ms
per call at the real layer shape) and 35B prefill is 7 to 8 % slower end to end. At T <= 256 the kernel time is unchanged.
No cheaper variant that passes the sweep was found (section 3). Whether that price is acceptable is a decision for the
owner; the alternative offered by the task (raising the selection threshold) does not help, because the error ratio does
not depend on T (section 2).

## 2. Mechanism

The error is the **fp32 accumulation of the k16 results over the K reduction**, kernels/cuda/moe_grouped.cu on origin/main
line 281 (`acc[i][j][v] += t[v]`): gate/up add 128 k16 results (K = H = 2048) into one running fp32 sum, down adds 32
(K = I = 512). Every add rounds to fp32 relative to a sum that has grown to the size of the result. The warp-per-row kernel
(moe.cu:222-226, 233) sums 16 products per block, 4 blocks per lane and then 32 lanes in a butterfly: about 9 adds deep for
gate/up. A gate/up pre-activation that is a few fp32 ulps off flips the bf16 rounding of `g`, `u` or the SiLU product `a`;
one flipped element of the 512-wide intermediate of a pair moves 50 or so of its 2048 outputs by a bf16 step.

Not the cause, read from the code and confirmed by experiment:

* h/act rounding points: moe_grouped.cu:335-340 round `g`, `u`, `a` and `a*u` where reference.rs `expert_forward` and moe.cu
  do.
* global scales: applied after the dot in both kernels (moe_grouped.cu:335 and 353, moe.cu:233 and 293), same as the reference
  folding them into the f64 weights up to one fp32 rounding.
* NVFP4 block scales: the dequantised bf16 tile is exact (e2m1 x e4m3, at most 6 significant bits).
* combine weights and tile-edge padding: the reduce kernel is the same code in both kernels; `moe_experts_mma_is_independent_of_tile_height`
  shows rows do not depend on padding or tile height.
* the tensor core's own k16 sum: splitting every mma into two m16n8k8 (experiment mode 3 of the first window) changed nothing
  (1.88x old against 1.76x), and compensating the adds at k8 granularity (mode 4) was no better than at k16.

"Small T" is a statistics effect, not a mechanism. A row's arithmetic does not depend on T, so the 1.7x to 1.8x ratio holds
at every T; at T = 32..64 a case has only 30 to 60 wrong elements, so the per-case bounds of the old test fire on luck (an
old kernel that happens to have 6 wrong elements against 61 for the grouped one when a single intermediate flips). The
sweep pooled over its 192 cases is the stable number:

| T | old kernel | mma on origin/main | mma fixed |
|---|---|---|---|
| 32 | 1048 | 1684 | 321 |
| 33 | 724 | 1543 | 191 |
| 34 | 1045 | 1326 | 224 |
| 35 | 880 | 1949 | 410 |
| 40 | 713 | 2236 | 238 |
| 41 | 1132 | 2095 | 408 |
| 42 | 995 | 1656 | 212 |
| 48 | 1140 | 1763 | 412 |
| 49 | 1173 | 1862 | 232 |
| 57 | 1834 | 3915 | 219 |
| 63 | 2027 | 2773 | 452 |
| 64 | 1800 | 2791 | 541 |
| all | 14 511 | 25 593 | 3 860 |

(elements off the reference, summed over 16 seeds; the fixed kernel is below the old one at every T). The grouped kernel
is therefore less accurate than the old one at every T it serves, and raising `MMA_MIN_TOKENS` would not make it as accurate;
that fallback was not taken.

## 3. Fix and what was measured

Fix (kernels/cuda/moe_grouped.cu): every k16 result is still formed from zero by the mma (chaining the accumulator through the
mma was the earlier 5x to 10x error, phase1b 8). The four k16 results of a 64-wide step are summed in fp32 (`sa`, line 309) and
the step sum is added into a running pair `(acc, lo)` with a TwoSum (`two_sum_add`, line 40, called at line 310), which keeps
the rounding error of the add. The epilogue scales the exact pair with one fp32 rounding (`scaled_pair_to_bf16`, line 49,
used at lines 335, 336, 353, 354) and then rounds to bf16, which is the reference's own double rounding (reference.rs `round`:
f64 to f32 to bf16). Two other epilogues were tried with the same TwoSum-per-k16 accumulation: `acc + lo` rounded to fp32 and then
scaled (two fp32 roundings): 0.39x old (window W1); the pair scaled and rounded to bf16 *directly*, with ties broken by the
exact value: 0.77x (W1b), worse, because the reference itself rounds through fp32 (f64 to f32 to bf16), so a "more exact" bf16
rounding disagrees with it at the bf16 ties that the fp32 rounding creates; one fp32 rounding of the exact scaled pair (the
shipped form): 0.07x (W1c, mode 1 of the table).

Experiment (windows W1 to W1c, 192 sweep cases, every mode on the same reference and old-kernel outputs; the experiment kernel
is kept in `raw/moe_grouped.experiment.cu.txt`, the harness tests were removed; the first window, `raw/w1`, used a plain
fp32 epilogue and modes 0 to 4 of that day's first version, so its numbers are not comparable to the table below). Kernel time
is `moe_experts_mma_bf16` at 256 experts, top-8, real layer shape, us per call, GB10.

| mode | accumulation | off-reference (old = 14 511) | cases in bounds | us T=256 | us T=1024 | us T=4096 |
|---|---|---|---|---|---|---|
| 0 | origin/main (fp32 add per k16) | 25 593 (1.76x) | 152 | 2489 | 3128 | 7171 |
| 1 | TwoSum per k16 | 1 013 (0.07x) | 191 | 2702 | 4276 | 10680 (+49 %) |
| 2 | **sum of the 4 k16 of a step, TwoSum per step (shipped)** | **3 860 (0.27x)** | **192** | 2474 | 3384 | 8504 (+19 %) |
| 3 | sum of 2 k16, TwoSum per pair | 1 726 (0.12x) | 192 | 2481 | 3599 | 9038 (+26 %) |
| 4 | Fast2Sum (3 flops, no ordering guarantee) per k16 | 1 934 (0.13x) | 191 | 2510 | 3348 | 8429 (+18 %) |
| 5 | 4 k16 chained through the mma accumulator, TwoSum per step | 4 532 (0.31x) | 186 | 2499 | 3243 | 8161 (+14 %) |
| 6 | 2 k16 chained through the mma accumulator, TwoSum per pair | 2 045 (0.14x) | 190 | 2514 | 3544 | 8780 (+22 %) |

Mode 2 is the cheapest of the modes that put all 192 cases inside the per-case bounds; modes 4 and 5 are cheaper by 1 to 5
points of kernel time but leave 1 and 6 cases outside. The window budget did not allow trying local sums of 8 k16 or a
Fast2Sum on the 4-sum (which would cut the extra FP32 work further); that is the obvious next experiment if the 7 % prefill
cost matters. `lo` and `sa` add 2 x 16 x MT fp32 registers per thread (64 at BM = 64); register use and spills were not inspected.

## 4. Gates (spark1, GB10, vLLM stopped for each window)

Tree `moe-mma-dev` at the committed code (`git rev-parse` stamp `dirty=no` in each log), baseline tree `moe-mma-base` =
origin/main. Raw logs in `raw/`.

### 4.1 Tests

* `moe_experts_mma_small_t_sweep` is now an asserting test (it was an `#[ignore]`d diagnostic): every case must be inside the
  bounds of `moe_experts_mma_is_as_accurate_as_old_kernel` and the pooled elements-off count of the grouped kernel must not
  exceed the old kernel's. Final run (`raw/w4/sweep.log`): 192 cases, 0 break the bounds, 3 860 against 14 511.
* **Mutation check**: the same test file on the origin/main kernel (`raw/w2/mutation.log`, tree `moe-mma-base`): FAILS, exit
  101, "40 of 192 cases break the bounds ... off-reference elements old 14511 mma 25593", exactly the 40 cases and the
  median of the 35b-accuracy doc. So the test detects the defect and reproduces its numbers.
* All CUDA kernel tests of `intarsia-backend-cuda` pass (`raw/w2/tests.log`, `--test-threads=1`), including the existing
  large-T test. `intarsia kernels test --backend cuda --model <35B>`: 41 kernels, 569 cases, all ADMITTED;
  `moe_experts_mma_bf16` worst rel_l2 1.459e-4 (`moe_experts_bf16` 3.160e-4).
* Mac: `INTARSIA_CPU_FREE_MEM_GB=8 cargo test --workspace` (82 test binaries, 0 failed) and `cargo check -p intarsia-cli --features hip`
  pass. The change is confined to a CUDA kernel and its test; no shared header changed.

### 4.2 35b-short, 8B (`raw/w2`)

Against section 5 of the 35b-accuracy doc ("after", `208ed00`):

| | doc, before this change | this branch |
|---|---|---|
| 35b-short golden gate | FAIL (decode mean KL 1.506e-2 > 1.043e-2, tail KL 1.556e-2 > 1.102e-2, max 1.329 > 0.602, `chat_multiturn`) | **PASS**: decode mean KL 7.167e-3, tail-bucket 7.676e-3, max KL 0.153; prefill mean KL 6.970e-3 (recorded baseline 7.161e-3) |
| `qwen3-8b-fp32` | prefill 2.897e-4, decode 3.238e-4 | 2.897e-4, 3.238e-4 (identical: the 8B has no MoE) |
| `qwen3-8b-long-fp32` | 4.004e-4, 3.332e-4 | 4.004e-4, 3.332e-4 (identical) |

The 35b-short change is mostly the one fragile position of the earlier doc (`chat_multiturn@55`, 8.3e-3 of the 1.556e-2 mean):
the earlier doc showed that position flips with tiny numeric changes in either direction, so this is not evidence that the
kernel accuracy buys a better match to vLLM (4.3 shows the opposite on the other gate). It is reported because the gate
verdict changed.

### 4.3 35b-long: not better, and the prefix-reuse gate flips (`raw/w2/35b-long.log`, reference run `raw/w5base`)

The origin/main build on the same day and box reproduces the doc exactly (`raw/w5base/35b-long.log`: prefill KL21 4.273e-2,
prefix-reuse PASS), so the harness is deterministic and the differences below are the kernel's.

| | origin/main | this branch |
|---|---|---|
| prefill floor KL21 (limit 3.0e-2, FAILs on both) | 4.273e-2 | **4.394e-2** |
| prefill mean KL (supp) | 4.025e-2 | 4.110e-2 |
| decode floor KL21 | 9.138e-3 | 9.258e-3 |
| baseline verdict (recorded run + envelope) | PASS | PASS |
| prefix-reuse gate | PASS | FAIL: `long_docs_4k` and `chat_reuse_4k` "continue" |

Per prompt prefill KL21 (origin/main to branch): long_docs_4k 1.688e-2 to 1.740e-2, long_code_4k 7.271e-3 to 7.117e-3,
chat_reuse_4k 0.2417 to 0.2514, long_docs_12k 2.103e-2 to 2.146e-2, long_code_12k 8.686e-3 to 8.156e-3. The whole +1.2e-3
shift of the aggregate is `chat_reuse_4k` (+0.0097, which the earlier doc showed moves by 0.13 to 0.14 under a chunk change
or a different backend); two prompts improve, two get worse. The old-numerics build measured 4.49e-2 there (doc section 5),
so 4.394e-2 is inside the spread between implementations, but it **is above the 4.273e-2 of origin/main, so the criterion
"aggregate prefill KL against vLLM must not get worse" is not met** and is reported as such.

The prefix-reuse gate compares a warm run (turn 2 reusing turn 1, so its tail is prefilled in other chunks) with a cold
run, and judges a divergence of the two greedy streams against the difference that a re-chunked cold run shows (the
"rounding scale"). Here the streams diverge at token 12 (`long_docs_4k`; warm margin 0.074 nats, cold 0.345) and token 5
(`chat_reuse_4k`; warm 0.333, cold 0.013), and on `long_docs_4k` the largest warm/cold difference before the divergence
(3.072 nats) exceeds that run's control scale (2.450); origin/main had 1.828 against 4.028 and agreed for 27 of 32 tokens,
`chat_reuse_4k` agreed for all 32. These are near-tie divergences on prompts that amplify small numeric changes (the doc's
`chat_reuse_4k` finding); this reads as the same sensitivity falling differently, but it was not separated from a real
loss (that needs the same gate with the old MoE kernel for every T, and no downtime was left for it).

## 5. Windows and downtime

vLLM (`missioncntrl-guidance`) stopped through `scripts/cuda-decode/window.sh` (fleetq exclusive lease with
`--takes-down guidance`, trap restore, `/v1/models` wait at http://100.120.88.104:8000), never co-tenant; builds and file
pulls ran as small fleetq jobs with vLLM up; every ssh carried `FLEETQ_JOB`. `downtime.log`:

| window | what | down |
|---|---|---|
| W1, W1b, W1c | accumulation-mode experiment (W1b was the direct-bf16 epilogue mistake of section 3) | 4.5 + 4.4 + 4.5 min |
| W2 | CUDA tests, mutation, kernels test, 35b-short, 8B x2, 35b-long | 14.3 min |
| W3 | 35B `bench speed` A/B | 11.2 min |
| W4 | sweep of the final kernel; the origin/main 35b-long attempt was refused (dirty base tree) | 4.7 min |
| W5 | origin/main 35b-long | 8.2 min |
| total | | **51.8 min**, each window 4 to 14 min; vLLM answered `/v1/models` after every window and at the end |

## 6. Speed A/B (W3; `raw/moe-mma-ab`)

`scripts/cuda-decode/w-ab.sh` through `scripts/moe-mma/w3.sh`: a fresh `intarsia serve` per arm, `bench speed --warmup 1
--thinking off --require-graphs`, gen 128, 3 reps per round, 3 rounds interleaved (base, dev / dev, base / base, dev),
`--max-seq 20480`. Means over the 3 rounds (per-round values in the json files; the rounds agree within 1 to 4 %):

| prompt | prefill tok/s base | dev | dev/base | TTFT s base | dev | decode tok/s base | dev |
|---|---|---|---|---|---|---|---|
| 1024 | 3929 | 3619 | 0.921 | 0.251 | 0.271 | 80.4 | 79.8 |
| 4096 | 4640 | 4309 | 0.929 | 0.884 | 0.948 | 77.6 | 77.2 |
| 16384 | 4365 | 4078 | 0.934 | 3.829 | 4.077 | 70.6 | 70.0 |

(the prompt token counts differ by up to 1 % between arms and rounds because the bench builds its prompt from random text;
tok/s is per actual prompt token). Decode does not use the grouped kernel; its 0.6 % deficit is the same sign in every
round and size and is not explained (box noise or arm order are the candidates; not investigated).

## 7. Not done / uncertain

* Prefill speed is lost (section 6); a cheaper accumulation (local sums of 8 k16, Fast2Sum on the step sum) was not tried
  for lack of downtime (51.8 of 60 minutes used, and a changed kernel would need the gates of section 4 again).
* The aggregate prefill KL against vLLM on 35b-long is 2.8 % worse and the prefix-reuse gate fails on two prompts (4.3);
  both are within what the earlier doc calls noise between implementations, neither was shown to be, and no gate was changed.
* The experiment's `INTARSIA_MOE_ACC` modes and their test harness are not in the branch (the experiment kernel is
  archived under `raw/`).
* `raw/w1` is the first experiment window (plain fp32 epilogue, modes of that version); `raw/windows/*-window.log` hold the
  console summaries of W1b and W1c, whose per-case logs were not pulled.
