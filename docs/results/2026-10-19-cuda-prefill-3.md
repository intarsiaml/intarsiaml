# CUDA prefill, round 3: fp8 dense projections and the lanes GDN kernel, Qwen3.6-35B-A3B NVFP4 on spark1 (2026-10-19)

Branch `perf/cuda-prefill-3` (from `origin/main` ce5209c7, PR #13 included). Hardware: spark1 (GB10, sm_121, CUDA 13.0).
Raw logs cited below are under `docs/results/2026-10-19-cuda-prefill-3/raw/`; vLLM downtime in `downtime.log` next to them.

## 1. Result

**Part B ships: `matmul_fp8_gemm_reg`**, the fp8 W8A8 prefill GEMM with the weights dequantised in registers and 128 x 128
tiles. Default on from 256 tokens, **bitwise the old kernel's output** (tests, and every end-to-end gate equal to base to the
printed digit). 35B prefill **+3.9 / +4.8 / +5.0 %** at 1024 / 4096 / 16384 tokens (interleaved A/B, 3 rounds, same-window
base), decode unchanged, memory delta 0. In situ the fp8 projections drop from 197.9 to 157.5 ms of GPU time for a
4160-token prompt (-20 %).

**Part A does not ship as default.** The lanes GatedDeltaNet kernel alone is faster (+5.3 / +4.1 / +7.7 % prefill, +2.4 to
+2.8 % decode) but fails three accuracy bars: 35b-short KL21 outside the base range, greedy ids differ, `chat_reuse_4k`
changed. It now has its own opt-in switch, `INTARSIA_CUDA_GDN_LANES=1`, and stays off.

| prompt | base (A/B median) | dev (A/B median) | change (paired per round) | dev, localeval | vLLM, localeval same day | dev / vLLM |
|---|---|---|---|---|---|---|
| 1024 | 3957 | 4113 | +3.9 % (+4.6, +3.9, +4.2) | 3866 * | 5102 * | 76 % |
| 4096 | 5053 | 5294 | +4.8 % (+4.5, +9.2, +2.3) | 5164 | 6351 | 81 % |
| 16384 | 4679 | 4912 | +5.0 % (+5.0, +4.8, +5.0) | 4848 | 6340 | 76 % |
| decode 1k / 4k / 16k | 80.5 / 77.8 / 70.4 | 80.8 / 78.0 / 70.4 | +0.4 / +0.3 / +0.0 % | 80.4 / 77.6 / 70.5 | 78.1 / 76.6 / 74.0 | |

Prefill and decode in tok/s. dev = this branch's default (reg on, lanes off). The A/B columns are `intarsia bench speed`
(window B, `raw/p3B/ab/`, prompts 1007 / 4107 / 16760 tokens); the localeval columns are the localeval harness (prompts
993 / 4091 / 16678 for intarsia in window C, `raw/final/windowBC.out`, 992 / 4088 / 16657 for vLLM with vLLM up,
`raw/final/localeval-vllm.out`); compare dev with vLLM only within the localeval columns. \* under 1024 uncached prompt
tokens localeval flags the figure as per-request overhead. Marginal prefill (localeval, 4k -> 16k row): dev 4753, vLLM 6338.
The task's carried-over numbers (base 3920 / 4958 / 4703, vLLM 5599 / 6627 / 6428) were not used; everything above was
measured in this round.

## 2. Part A: the lanes GatedDeltaNet kernel alone

**Gate.** `gdn_delta_rule_bf16_lanes` (kernels/cuda/gdn.cu, already registered) was selected only by
`INTARSIA_CUDA_FAST_PREFILL=1`, which also selects the e4m3 GEMM `matmul_fp8_gemm_f8`: the two were coupled. Commit 071584f8
gives the lanes kernel its own switch, `INTARSIA_CUDA_GDN_LANES=1` (`kernels::gdn_lanes()`, priority via
`gdn_lanes_priority`); `INTARSIA_CUDA_FAST_PREFILL=1` still turns on both as before. Registry test
`kernels::tests::gdn_delta_rule_selection` pins the pick at the 35B shape for T = 1 and T = 2048 (rows kernel by default,
lanes kernel with either switch).

**Kernel alone** (spark1, vLLM up, `raw/dev/01-peak-gdn-wq-bisect.out`): `tests/gdn_lanes.rs` passes (rel_l2 against the f64
reference on par with the rows kernel, e.g. T = 513 output 1.89e-5 vs 2.92e-5, state 8.4e-8 both); `tests/gdn_timing.rs` at
the 35B shapes:

| T | rows kernel (default) | lanes kernel |
|---|---|---|
| 1 (decode) | 14.1 us | 6.2 us |
| 16 | 31.4 us | 16.5 us |
| 512 | 592 us | 377 us |
| 2048 | 2948 us | 1896 us |

At a 2048-token chunk that is 30 layers x 1.05 ms = 32 ms per chunk (inference from the microbench).
Admission sweep (`raw/final/admission.out`): `gdn_delta_rule_bf16_lanes` 12 cases, 0 fail, ADMITTED.

**End to end, accuracy** (window A, vLLM down, `raw/p3A/`): see section 5. It **fails** three bars:

* 35b-short KL21 7.571e-3 at chunk 2048 (6.628e-3 at chunk 256), outside the same-window base range 5.430e-3 .. 6.271e-3
  (top-1 is higher, 98.40 % vs 97.92 %, but the bar is the KL range);
* greedy ids differ from base at generated token 72 of 256 (prompt 1497 tokens), graph and eager alike;
* `chat_reuse_4k` continue: 22/32 tokens instead of base's 32/32 (`ok` by the gate's rounding rule, but changed).

35b-long KL21 4.481e-2 is inside its range (4.273e-2 .. 4.576e-2).

**Speed** (window B): see section 6.

**Verdict: not proposed for default-on.** It stays opt-in (`INTARSIA_CUDA_GDN_LANES=1`). Likely cause (inference, not
measured): the lanes kernel sums each state row's fp32 dot products in another order; the f64 test shows that is no less
accurate per call, but across 30 layers and thousands of tokens it moves the logits by more than the attention / chunk noise
the short gate's range is built from. Also inference: the old FAST_PREFILL failure was not only the e4m3 GEMM, since the
lanes kernel alone already leaves the short range.

## 3. Part B: the fp8 dense projections (wq_gemm)

### 3.1 What runs today, and what vLLM runs

* **intarsia, 2048-token chunk:** the registry picks `matmul_fp8_gemm` (crates/intarsia-backend-cuda/src/quantmoe.rs,
  `m >= 33 && k % 128 == 0`, priority 10) for every W8A8 projection (GDN in_proj_qkv / in_proj_z / out_proj, attention
  q+gate / k / v / o). Its launcher (kernels/cuda/fp8_gemm.cu) first runs `fp8_actq_kernel` (activations to e4m3 values held
  as bf16, scratch `M x K x 2` bytes), then `wq::gemm_by_m<FMT_FP8>` of `kernels/cuda/wq_gemm.cuh`: **a custom kernel that
  stages a bf16 tile through shared memory**. Raw e4m3 weight rows come in by cp.async (64-byte runs, 3-slot ring), are
  dequantised once per tile and k step into a bf16 shared-memory tile, then `ldmatrix` + bf16 `mma.sync m16n8k16` (f32
  accumulate), 64 x 128 block tiles. Not native fp8 mma (that is `matmul_fp8_gemm_f8`, opt-in), not cuBLASLt (cuBLAS is used
  only by the bf16 `matmul_bf16` path in kernels/cuda/matmul.cu).
* **vLLM, same checkpoint** (container startup log, `raw/probe/quant-config-and-vllm-log.out`, fleetq #2490):
  `CutlassFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod` for the fp8 projections (native e4m3 tensor cores, CUTLASS
  scaled_mm; the container sets `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`, so CUTLASS is the pick),
  `MarlinNvFp4LinearKernel for NVFP4 GEMM`, and `Triton/FLA GDN prefill kernel` for the recurrence.

### 3.2 Where the time goes (before writing code)

Microbench `bench_wq_prefill_shapes` (tests/quantmoe.rs, new, ignored): the six 35B fp8 projection shapes at M tokens, one
call each, summed over one 2048-token chunk (30 GDN layers x 3 + 10 attention layers x 4 calls). Tensor-core probe
`kernels/cuda_checks/mma_peak.cu` (`raw/dev/01-peak-gdn-wq-bisect.out`): bf16 m16n8k16 122 TFLOPS, e4m3 m16n8k32 244
TFLOPS, 48 SMs, 100 KB shared memory per SM. One chunk of fp8 projections is 5.24 TFLOP, so the bf16 floor is 43 ms and the
e4m3 floor 21 ms.

Variant bisect of the old kernel at M = 2048 (instrumentation patch `scripts/cuda-prefill3/exp-wq-variants.patch` on
`kernels/cuda/wq_gemm.cuh` of ce5209c7, not shipped; vLLM up, so relative only):

| variant | fp8 ms / chunk |
|---|---|
| `matmul_fp8_gemm` as is | 109.0 |
| no dequant to the bf16 tile | 88.4 |
| no raw weight cp.async | 89.9 |
| no A cp.async | 95.4 |
| no loads at all | 85.4 |
| no loads, no dequant | 69.5 |
| no mma | 76.1 |
| `matmul_fp8_gemm_f8` (e4m3 mma, reference) | 65.0 |

So the shared-memory dequant (write the bf16 tile, barrier, ldmatrix it back) costs about a fifth, and the kernel as a
whole runs at 48 TFLOPS (39 % of the bf16 peak).

### 3.3 Approach

The W4A16 recipe (register dequant, scales through the cp.async ring, 2 blocks per SM) applies in part: there are no block
scales here (one per-tensor weight scale and one static input scale, applied in the epilogue), and with 64 x 128 tiles the
old kernel (`MT = 2, WM = 2` for M > 32, so 64-row tiles) reads every weight byte M / 64 times from L2. What applies: dequantise e4m3 to bf16 **in registers**
straight from the raw byte tile (no bf16 tile, no extra barrier), and make the block tile **128 x 128** so the activations'
`ldmatrix` and the weight bytes are each read half as often per mma. Two blocks per SM do not fit at 128 x 128 (86 KB of
shared memory per block); one block of 8 warps per SM does.

No weight re-layout and no extra memory: the kernel reads the same e4m3 `[N, K]` weights, the same scales and the same
`M x K x 2` scratch.

### 3.4 What was built: `matmul_fp8_gemm_reg`

`kernels/cuda/fp8_gemm.cu` (namespace `reg`), launcher `ik_matmul_fp8_gemm_reg` (declared in the CUDA-only
`kernels/cuda/ik_cuda_abi.h`), registered in `crates/intarsia-backend-cuda/src/quantmoe.rs`, dispatch in
`device_quantmoe.rs`.

* Block 256 threads = 8 warps (2 along M x 4 along N), warp tile 64 x 32 (4 x 4 m16n8 fragments), block tile 128 x 128,
  k step 64. A tile (128 x 64 bf16, rows padded to 144 B) and the raw weight tile (128 rows x 64 B, padded to 80 B) in one
  3-slot cp.async ring, 86 KB, one barrier per k step. 128 registers (bf16 output; 135 for f32 output), no spills
  (`raw/dev/05-reg-final-tests-bench.out`, "resources").
* B fragments are built in registers: each lane reads the 32-bit words holding its k = 2t, 2t+1 and 2t+8, 2t+9 bytes of row
  `lane / 4`, one `prmt` places two e4m3 bytes at the bf16 positions, a mask/shift moves sign, exponent and mantissa to bf16
  bits 15 and 10..4, and one `bf16x2` multiply by 2^120 restores the exponent bias (exact; e4m3 subnormals land on bf16
  subnormals first and the multiply keeps them).
* Small grids (fewer than 24 tiles of 128 x 128: the k/v projections below 768 tokens) fall back to the 64-row
  `wq_gemm.cuh` kernel, which gives the same bits.
* Selection: priority 15 (above `matmul_fp8_gemm`'s 10) for `m >= FP8_REG_MIN_TOKENS = 256 && k % 128 == 0`; below 256
  tokens `matmul_fp8_gemm` stays (the reg kernel was not measured faster there). Kill switch `INTARSIA_CUDA_FP8_GEMM_REG=0`.
  Registry test `kernels::tests::fp8_gemm_selection_by_batch_size` pins gemv 1..4, mma 5..32, the reg kernel at 256, 2048 and
  4096 tokens for K = 2048 and 4096, `matmul_fp8_gemm` at 33 and 255, the threshold as a literal (256), and `matmul_fp8_gemm` with
  the switch off.

**Numerics: bitwise `matmul_fp8_gemm`.** Same A fragments (the same ldmatrix loads of the same bf16 / e4m3-valued bf16
activations), each B register holds the same two bf16 values that ldmatrix of the old dequantised tile gives, every output
element goes through the same chain of m16n8k16 mma in the same k order with its position inside the m16n8 fragment
unchanged (tile origins are multiples of 16 rows and 128 columns), and the epilogue is the same scale and rounding. The one
difference: the NaN codes 0x7F / 0xFF become +-480 instead of NaN. ModelOpt fp8 export saturates at +-448, and on the 35B
every end-to-end output matched base; the tests use every finite code.

Microbench (exclusive slot, vLLM up, `raw/dev/05-reg-final-tests-bench.out`, 2 rounds, round 1 shown, round 2 within 2 %):

| M | `matmul_fp8_gemm` ms / chunk | `matmul_fp8_gemm_reg` | change | `matmul_fp8_gemm_f8` (not bitwise, opt-in) |
|---|---|---|---|---|
| 256 | 14.68 | 11.92 | -18.8 % | 11.12 |
| 512 | 25.32 | 21.93 | -13.4 % | 19.07 |
| 1024 | 50.68 | 41.34 | -18.4 % | 31.89 |
| 2048 | 101.92 | 83.01 | -18.6 % | 65.63 |
| 4096 | 208.44 | 161.97 | -22.3 % | 129.38 |

At 2048 tokens that is 19 ms per chunk, 63 TFLOPS (52 % of the bf16 peak).

## 4. Tried and dropped

* **Register-dequant bisect** of the new kernel (`scripts/cuda-prefill3/exp-reg-variants.patch`, applies to b0abf6df, the
  kernel before it was templated; `raw/dev/03-reg-bisect-noisy.out`,
  vLLM up and not exclusive, relative only): no dequant -9 %, no cp.async loads -11 %, no mma -34 %, no mma and no loads
  -54 %.
  The mma stream (with its A ldmatrix) is now the ceiling: further dequant tricks cannot buy more than about 9 %.
* **Tile shapes** (`raw/dev/04-reg-tile-variants.out`, all bitwise, tests passed for each): 128 x 256 with 2 stages
  (83.4 ms) ties 128 x 128 with 3 stages (83.6 ms); 128 x 128 with 2 stages is slower (90.6 ms). Kept 128 x 128 x 3 (fewer
  registers, smaller tail waste).
* **Tile order** (consecutive blocks along M vs along N, same noisy run): 100.8 / 96.5 ms against 97.6 ms for the old
  kernel's rule, within that run's noise; kept the rule.
* **e4m3 tensor cores (`matmul_fp8_gemm_f8`)**: 21 % faster than the new kernel per chunk, but it rounds differently (fp32
  sums of e4m3 products in another order); it was the half of FAST_PREFILL that was never tested alone. Not attempted here
  because the bar prefers bitwise; it is the next lever (section 12).

## 5. Accuracy

Window A (vLLM down, one binary for every arm, golden sha b6b51e1b1699c66c = `vllm-top20` goldens, `raw/p3A/`,
driver log `raw/final/windowA.out`):

| gate | base (same window) | reg (Part B, default) | lanes (Part A, opt-in) | bar |
|---|---|---|---|---|
| 35b-short KL21, chunk 2048 | 5.430e-3, top-1 97.92 % | 5.430e-3, 97.92 % | **7.571e-3**, 98.40 % | 5.430e-3 .. 6.271e-3 |
| 35b-short KL21, chunk 1024 / 256 | 5.430e-3 / 6.271e-3 | | - / 6.628e-3 | (base range) |
| 35b-long KL21, chunk 2048 | 4.576e-2, top-1 94.17 % | 4.576e-2, 94.17 % | 4.481e-2, 94.33 % | 4.273e-2 .. 4.576e-2 |
| 35b-long KL21, chunk 1024 / 256 | 4.421e-2 / 4.273e-2 | | | (base range) |
| greedy ids, 256 tokens, prompt 1497 tokens | | identical to base | **differ at token 72** (graph == eager) | identical |
| `chat_reuse_4k` (continue / restore / aligned) | 32/32, 32/32, 32/32 ok | 32/32, 32/32, 32/32 ok | **22/32**, 32/32, 32/32 ok | unchanged |
| `long_docs_12k` reuse | 32/32 x 3 ok | 32/32 x 3 ok | 32/32 x 3 ok | unchanged |
| 8B fp32 short (lanes + reg on) | | prefill KL 2.897e-4, decode 3.238e-4, PASS | | pass |
| 8B fp32 long (dev default, vLLM up) | | prefill KL 4.004e-4, decode 3.332e-4, PASS | | pass |

The base range is the same window's base at chunks 2048, 1024 and 256 (the method of docs/results/2026-10-18-cuda-moe-w4a16.md).
The long gate "FAIL" against the 3.0e-2 floor in the logs is the base's, unchanged. reg equals base to every printed digit,
as the bitwise tests predict; that is equality, not a range pass. The 8B model has no fp8 projections and no GatedDeltaNet,
so its goldens are a sanity check of the shared path only (8B long ran in `raw/final/admission.out`, after window A's
budget ran out for it).

Kernel tests (spark1, `raw/dev/05-reg-final-tests-bench.out`): `tests/fp8_gemm_reg.rs`, 3 tests, all pass:
`fp8_gemm_reg_is_bitwise_the_gemm` (M 33..1025 incl. M and N tails, N % 8 != 0 scalar epilogue, K 128..1024, bf16 and f32
outputs, W8A8 and weight-only, both sides of the small-grid fallback; plus rel_l2 < 1e-3 against the cpu reference),
`fp8_gemm_reg_every_e4m3_code` (all 254 finite codes in every position class), `fp8_gemm_reg_35b_shapes` (the four 35B
projection shapes at 2048 tokens and a 300-token tail chunk). The existing `tests/quantmoe.rs` passes unchanged (10 passed).

## 6. Speed A/B

Window B (vLLM down, `raw/final/windowBC.out`, `raw/p3B/ab/`, summary by `scripts/cuda-prefill3/ab-summary.py`):
`intarsia bench speed` against a fresh `serve` per arm (`scripts/cuda-decode/w-ab.sh`, unmodified), sizes 1024 / 4096 /
16384, gen 128, 2 reps per size, 3 rounds interleaved (round 1 base, lanes, reg; round 2 reversed; round 3 as round 1). One
binary for all arms, only the environment differs: base = `INTARSIA_CUDA_FP8_GEMM_REG=0` (origin/main selection), lanes =
base + `INTARSIA_CUDA_GDN_LANES=1`, reg = default. All 9 bench runs exit 0; every `serve` admitted its plan (reg arm 36
kernels / 505 cases, the others 35 / 501).

Prefill tok/s (r1, r2, r3):

| prompt | base | reg | change, median (paired) | lanes | change, median (paired) |
|---|---|---|---|---|---|
| 1024 | 3933, 3957, 3960 | 4113, 4111, 4127 | +3.9 % (+4.6, +3.9, +4.2) | 4190, 4165, 4149 | +5.3 % (+6.5, +5.3, +4.8) |
| 4096 | 5067, 4850, 5053 | 5294, 5296, 5168 | +4.8 % (+4.5, +9.2, +2.3) | 5262, 5146, 5304 | +4.1 % (+3.8, +6.1, +5.0) |
| 16384 | 4679, 4678, 4680 | 4912, 4902, 4916 | +5.0 % (+5.0, +4.8, +5.0) | 5045, 5037, 5012 | +7.7 % (+7.8, +7.7, +7.1) |

Decode tok/s (r1, r2, r3):

| prompt | base | reg | change | lanes | change |
|---|---|---|---|---|---|
| 1024 | 81.1, 80.5, 80.5 | 81.0, 80.4, 80.8 | +0.4 % | 82.4, 82.3, 83.2 | +2.4 % |
| 4096 | 78.4, 77.6, 77.8 | 78.1, 77.5, 78.0 | +0.3 % | 80.0, 79.8, 80.9 | +2.8 % |
| 16384 | 70.5, 70.3, 70.4 | 70.6, 70.3, 70.4 | +0.0 % | 72.4, 72.4, 73.1 | +2.8 % |

No decode regression for reg (decode never runs the new kernel: it needs 256+ tokens). The 4096 base cell varies
4850..5067 between rounds, so its reg pairs spread +2.3..+9.2 %; the 16384 pairs are tight. The reg gain matches the
microbench: 19 ms per 2048-token chunk of fp8 projections out of roughly 400 to 440 ms per chunk (+4 to 5 %).

In-situ kernel profile (nsys, 4160-token prompt, chunk 2048, exclusive slot with vLLM up and idle,
`scripts/cuda-prefill3/prof.sh`, `raw/final/nsys-profile.out`, GPU time per kernel group):

| kernel group | base | dev |
|---|---|---|
| MoE experts (all) | 344.4 ms (39.8 %) | 350.0 ms (41.6 %) |
| fp8 projections | `wq_gemm` FMT_FP8 197.9 ms (22.9 %, 390 calls) | `gemm_reg_kernel` 146.9 ms (260 calls) + `wq_gemm` FMT_FP8 10.6 ms (130 calls: the 64-token tail chunk) |
| GDN recurrence (rows kernel) | 174.6 ms (20.2 %) | 179.5 ms (21.3 %) |
| `fp8_actq` (activation quantiser) | 18.4 ms | 19.7 ms |

This confirms in situ which kernels run (section 3.1) and that the tail chunk below 256 tokens stays on the old kernel.

## 7. Mutation check

`scripts/cuda-prefill3/mutate.py` (25 mutations of the new kernel), `mutcheck.sh`, spark1 with vLLM up
(`raw/final/mutcheck.out`, fleetq #2506): unmutated tree passes; **23 of 23 non-equivalent mutants killed**, none failed to
apply or build. Covered: sign, shift, mantissa mask and bias restore of the e4m3 -> bf16 conversion (5), byte order, pair
and word selection, second-register offset, b0/b1 swap, B row stride, A row, warp column (8), ring wait depth, slot and
prefetch distance (3), weight copy chunk, weight row tail, W8A8 activation source, output scale, block -> tile map,
vectorised and scalar epilogue columns (7). The three ring mutants were killed by failing tests (prefetch distance by 2 of
the 3 tests, the others by all 3).

**Equivalent mutants (2, survive by design):** `a-tail` (activation rows past M copied from row 0 instead of zero-filled:
they feed only output rows past M, which are never stored) and `fallback-off` (the small-grid fallback to the 64-row
kernel never taken: same bits, other schedule).

## 8. Admission

`scripts/cuda-prefill3/admission.sh` on spark1 with vLLM up (`raw/final/admission.out`, fleetq #2510):

* Shared sweep `intarsia kernels test --backend cuda --model <35B> --op matmul_fp8 -v`: the sweep's cases stopped at M = 40,
  so the first run (`raw/final/admission-1-before-sweep-case.out`) gave the new kernel **0 cases and failed** ("kernel
  admission failed"). Commit e3cce5ce adds one prefill-shaped case to the shared sweep (M = 300, N = 3080, K = 256, three
  dtype / activation modes, appended so the existing cases' data is unchanged). Rerun: **`matmul_fp8_gemm_reg` 3 cases, 0
  fail, ADMITTED** (max_abs 3.03e-3, rel_l2 4.18e-5); gemv 24, mma 63, gemm 11, gemm_f8 27 cases, all ADMITTED; exit 0.
* `--op gdn_delta_rule`: rows kernel 20 cases, lanes kernel 12 cases, ADMITTED.
* Plan-time admission on the 35B plan's exact shapes (`plan show --admit --no-admission-cache`, chunk 2048, the check
  `serve` runs at load): base 35 kernels / 501 cases, dev 36 kernels / 505 cases (`matmul_fp8_gemm_reg` at the four
  projection shapes, rows clamped to 32), nothing unadmitted.

## 9. Memory

**Delta 0.** `matmul_fp8_gemm_reg` reads the same e4m3 weight and scale tensors and declares the same scratch as
`matmul_fp8_gemm` (`M x K x 2` bytes for the quantised activations); the planned scratch arena is 83,963,392 bytes for
both base and dev (`plan show --json`, `raw/p3adm/plan-base.json`, `raw/p3adm/plan-dev.json`, compared by
`scripts/cuda-prefill3/plan-mem.py`). Its 86 KB of dynamic shared memory per block is on-chip, not resident. The lanes
kernel declares no scratch and uses the same state tensors. No weight re-layout.

## 10. vLLM downtime

Two exclusive windows through the unmodified `scripts/cuda-decode/window.sh` (only `DOWNTIME_LOG` set; `--takes-down
guidance`, no `--force`, no `--no-queue`), logged in `downtime.log` in this directory; driver logs `raw/final/windowA.out`
and `raw/final/windowBC.out`:

| window | job | stop -> serving again | down |
|---|---|---|---|
| A accuracy and gates (lanes, reg, base) | #2500 | 20:31:38 -> 20:47:52 | 16.2 min |
| B+C speed A/B (3 arms x 3 rounds) + localeval of dev | #2513 | 20:56:33 -> 21:11:13 | 14.6 min |

**30.8 min total, no window over 20.** Each window's script ran 12.7 and 11.1 min; the rest is the vLLM stop and restart.
`/v1/models` answered after each window (the script waits for it; checked again with curl). Windows B and C shared one
downtime. Everything else ran with vLLM up: development, tests, microbench, mutation check, admission, 8B long golden,
the nsys profile (exclusive slot, no `--takes-down`) and the vLLM localeval run (its own slot). Queue wait: none to speak
of; the profile job (#2519) waited for one in-flight vLLM request before it started.

## 11. PR handoff

**Changed-file list** (`git diff --name-only origin/main`, origin/main = 56da3172; the 94 raw-log paths under this
doc's folder are collapsed into one line):

```
crates/intarsia-backend-cuda/src/device_quantmoe.rs
crates/intarsia-backend-cuda/src/kernels.rs
crates/intarsia-backend-cuda/src/quantmoe.rs
crates/intarsia-backend-cuda/tests/fp8_gemm_reg.rs
crates/intarsia-backend-cuda/tests/quantmoe.rs
crates/intarsia-core/src/quant/cases.rs
docs/results/2026-10-19-cuda-prefill-3.md
kernels/cuda/fp8_gemm.cu
kernels/cuda/ik_cuda_abi.h
kernels/cuda_checks/mma_peak.cu
scripts/cuda-prefill3/ab-summary.py
scripts/cuda-prefill3/admission.sh
scripts/cuda-prefill3/dev.sh
scripts/cuda-prefill3/exp-reg-variants.patch
scripts/cuda-prefill3/exp-wq-variants.patch
scripts/cuda-prefill3/fetch.sh
scripts/cuda-prefill3/mutate-dryrun.sh
scripts/cuda-prefill3/mutate.py
scripts/cuda-prefill3/mutcheck.sh
scripts/cuda-prefill3/plan-mem.py
scripts/cuda-prefill3/probe-config.sh
scripts/cuda-prefill3/prof.sh
scripts/cuda-prefill3/wA-accuracy.sh
scripts/cuda-prefill3/wB-ab.sh
scripts/cuda-prefill3/wBC.sh
scripts/cuda-prefill3/wC-localeval.sh
scripts/cuda-prefill3/wC-serve.sh
+ 94 files under docs/results/2026-10-19-cuda-prefill-3/ (raw logs, downtime.log)
```

Outside `crates/intarsia-backend-cuda`, `kernels/cuda`, docs and scripts:

* **`crates/intarsia-core/src/quant/cases.rs` (shared code, needs Metal and HIP checks by others).** One prefill-shaped
  `matmul_fp8` admission sweep case (M = 300, N = 3080, K = 256, three dtype / activation modes), appended after the existing
  cases so their random data is unchanged. Without it the sweep gives the new kernel 0 cases and `intarsia kernels test
  --op matmul_fp8` fails. The case runs on every backend's fp8 kernels whose predicates accept M = 300, so the Metal and
  HIP sweeps must be rerun there.
* `kernels/cuda_checks/mma_peak.cu`: new standalone tensor-core probe, not built by cargo.
* No `build.rs`, build target, ABI header (`kernels/abi`), model, recipe or registry-core change. The new launcher is
  declared in the CUDA-only `kernels/cuda/ik_cuda_abi.h`; `fp8_gemm.cu` was already built.

Mac checks after the rebase onto 56da3172 (`raw/mac/`): `INTARSIA_CPU_FREE_MEM_GB=8 cargo test --workspace` exit 0 (470 passed, 0 failed),
`cargo check --workspace` exit 0, `cargo check -p intarsia-cli --features hip` exit 0. Metal and HIP were not run here.

**Changed inside the CUDA crate:** `src/kernels.rs` (`gdn_lanes()`, `gdn_lanes_priority()`, `fp8_gemm_reg()`,
`fp8_gemm_reg_priority()`, two registry tests), `src/quantmoe.rs` (table entry, `FP8_REG_MIN_TOKENS`), `src/device_quantmoe.rs`
(extern + dispatch), `tests/fp8_gemm_reg.rs` (new), `tests/quantmoe.rs` (ignored microbench `bench_wq_prefill_shapes`);
`kernels/cuda/fp8_gemm.cu` (new kernel; the old launcher's checks moved into a shared `prologue`, unchanged),
`kernels/cuda/ik_cuda_abi.h`.

**Environment variables**

* `INTARSIA_CUDA_FP8_GEMM_REG=0`: kill switch for the new fp8 GEMM (default on).
* `INTARSIA_CUDA_GDN_LANES=1`: opt-in lanes GDN kernel alone (default off; fails the accuracy bars, section 2).
* `INTARSIA_CUDA_FAST_PREFILL=1`: unchanged, still selects lanes + `matmul_fp8_gemm_f8`.
* Bench only: `INTARSIA_BENCH_M`, `INTARSIA_BENCH_WQ`, `INTARSIA_BENCH_REPS` (`bench_wq_prefill_shapes`).

**Caveats**

* The new kernel serves batches of 256+ tokens; shorter batches (prompt tails below 256, decode) keep `matmul_fp8_gemm` /
  mma / gemv. Below 256 it was not measured.
* Window A ran the binary built before commit 59418f43 (the sweep case). That commit changes admission test data only; no
  kernel, model or selection code changed between the windows. All measurements ran on the tree before the final rebase;
  the rebase brought only HIP and docs changes (no diff in core, the CUDA crate, models or CLI).
* The A/B ran 2 reps per size per round (the existing window script), not 5.
* "base" in every window is this branch's binary with the kill switch (`INTARSIA_CUDA_FP8_GEMM_REG=0`), not an origin/main
  build. Its gates match the origin/main base of docs/results/2026-10-18-cuda-moe-w4a16.md to every digit (35b-short KL21
  5.430e-3 at 2048, 6.271e-3 at 256; 35b-long 4.576e-2 at 2048, 4.273e-2 at 256), which is
  good evidence the kill-switch base is equivalent.
* The NaN e4m3 codes (0x7F, 0xFF) give +-480 in the new kernel instead of NaN; no checkpoint has NaN weights.
* 8B fp32 short golden ran with lanes + reg on, 8B long with the dev default; the 8B has neither kernel's op, so these are
  sanity checks only.
* intarsia prefill is still 19 to 24 % below vLLM (localeval). Decode is level or ahead up to 4k and 5 % behind at 16k.

## 12. Next levers (not done here)

* **`matmul_fp8_gemm_f8` alone** (e4m3 tensor cores, vLLM's approach): 21 % faster than the new kernel per chunk in the
  microbench (65.6 vs 83.0 ms at 2048 tokens), worth about 17 ms per chunk, roughly +4 % prefill (inference). It cannot be
  bitwise, and its accuracy alone has never been measured (it only ran bundled with lanes under FAST_PREFILL). Next step: a
  window-A run of it alone against the same-window base range.
* **GDN recurrence** (20 % of kernel time in situ): the lanes kernel shows the speed is there (+4 to 8 %), but not its
  summation order. A lanes kernel that keeps the rows kernel's order would be needed to ship it under these bars; not
  investigated.
* **MoE experts** remain the largest share (40 %); see docs/proposals/2026-10-19-cuda-moe-weight-relayout.md.

## 13. Reproduce

```
fleetq run --host spark1 --mem 12 --label "..." -- scripts/cuda-prefill3/dev.sh                       # build, mma peak, gdn tests, microbench
TREE=p3-dev STEPS=res,wqt,wq WQ_TESTS="--test fp8_gemm_reg --test quantmoe" \
  WQ_CFGS='INTARSIA_BENCH_WQ=matmul_fp8_gemm,matmul_fp8_gemm_reg,matmul_fp8_gemm_f8 INTARSIA_BENCH_M=256,512,1024,2048,4096' ROUNDS=2 \
  fleetq run --host spark1 --exclusive --mem 12 --label "..." -- scripts/cuda-prefill3/dev.sh
fleetq run --host spark1 --mem 12 --label "..." -- scripts/cuda-prefill3/mutcheck.sh
fleetq run --host spark1 --mem 30 --label "..." -- scripts/cuda-prefill3/admission.sh
DOWNTIME_LOG=$PWD/docs/results/2026-10-19-cuda-prefill-3/downtime.log scripts/cuda-decode/window.sh 14 "..." /home/nvidia/intarsia-decfuse/p3-dev/scripts/cuda-prefill3/wA-accuracy.sh
ARMS=dev DOWNTIME_LOG=... scripts/cuda-prefill3/wC-localeval.sh 15 <log>                              # window B+C (wBC.sh)
uv run --project /opt/code/deepeval localeval speed spark-guidance --thinking off --sizes 1024,4096,16384 --gen 128 --reps 3
fleetq run --host spark1 --exclusive --mem 34 --label "..." -- scripts/cuda-prefill3/prof.sh base:p3-dev:INTARSIA_CUDA_FP8_GEMM_REG=0 dev:p3-dev
```
