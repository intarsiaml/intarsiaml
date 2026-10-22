# S1: CUDA 35B prefill on DGX Spark: three bitwise prefill kernels, an opt-in cuBLAS projection path (2026-10-22)

Target: DGX Spark GB10 (sm_121, CUDA 13.0), Qwen3.6-35B-A3B-NVFP4, tp=1, default (numerics-preserving) path. Lane S1 of the
2026-10-21 evening round. Branch `lane-s1/spark-prefill`, base origin/main `f9087322`. Raw logs: `2026-10-22-s1-spark-prefill/raw`.
The FP8-E4M3 activation path (`INTARSIA_CUDA_FP8_GEMM_LT=1`, lane F2) is untouched; the new kernels step aside when it is on.

## 1. Headline

- **Default path (dev), bitwise: prefill +2.1 / +2.3 / +2.1 % at 1k / 4k / 16k** (`intarsia bench speed --at-most`, spark2
  with vLLM down, 3 interleaved rounds; every dev round above every base round), **decode +0.0..+0.4 %**. Three kernels:
  the fp8 projection *pair* at prefill (one activation quantiser, one grid), and register-dequant NVFP4 GEMM / fused
  GLU for the shared expert. Greedy ids (8B and 35B, short and 4k), the 35B KL21 digits (short 5.430e-3 / long 4.576e-2,
  top-1 97.92 / 94.17 %) and chat_reuse_4k are identical to base; the 8B plan is byte-identical.
- **Opt-in `INTARSIA_CUDA_FP8_CUBLAS=1`, also bitwise: +3.2 / +5.1 / +5.0 %.** cuBLAS's bf16 GEMM on exactly expanded e4m3
  operands reproduces `matmul_fp8_gemm_reg`'s bytes at every 35B shape on sm_121 / CUDA 13.0, so the projections can run
  on nvjet. It keeps a bf16 copy of each fp8 weight: **+2.6 GB outside the memory plan**, which is why it is not default.
  Making it default (or planning that memory) is a decision for the user.
- **Against vLLM** (localeval speed): the clean vLLM reference is round 7's localeval of spark1's vLLM in its own slot
  (2026-10-21 10:56, `2026-10-21-cuda-round7/raw/08-localeval-vllm.out`): 4k 6353 / marginal 6862, 16k 6275. Q1's
  5278 / 5967 ran concurrently with the intarsia speed run on the same box (start 18:32:25 vs 18:32:41) and is not a
  usable reference. Tonight, localeval on spark2: **4k dev 5739 (-9.7 % vs vLLM), cub 5836 (-8.1 %)**, base 5002;
  **16k dev 5432 (-13.4 %), cub 5549 (-11.6 %)**; marginal 4k dev 6376 / cub 6612 vs 6862. The gap closes by about a
  third at 4k (base was -21 % on this harness), not fully. vLLM was not re-measured tonight (both vLLMs were down for
  #3828 by the time the window closed); a same-night same-box vLLM run is the open check.
- **The biggest sink is untouched:** routed experts are 40 % of prefill GPU time and stream all ~18 GB of expert weights
  per 2048-token chunk (131 ms per chunk vs a 66 ms bandwidth floor; the 93-token tail chunk alone costs 51 ms). See 6.

## 2. Where a 4k prefill goes (step 1)

One nsys profile per arm of `intarsia generate --max-tokens 1` on a 4189-token prompt (chunks 2048 + 2048 + 93), spark2,
exclusive slot #3618 with vLLM up and idle (`raw/gates-spark2/prof-*-14000/rank.txt`; single runs, so rows move by a few
% between arms on their own: compare the rows a change touches). GPU kernel time per arm, ms:

| kernel | base f9087322 | dev (default) | cub (opt-in) | what it is |
|---|---|---|---|---|
| `moe_w4a16_kernel` (2048 chunks) | 270.9 | 262.9 | 280.0 | routed experts, 2 launches per layer and chunk |
| `moe_mma_kernel` (93-token tail) | 61.3 | 51.4 | 53.6 | routed experts of the tail chunk: still streams almost every expert |
| fp8 projections | 150.0 `gemm_reg` | 82.9 pair + 59.2 single = **142.1** | 126.3 cuBLAS nvjet | dense W8A8 GEMMs |
| `fp8_actq_kernel` | 19.7 (390 calls) | **13.9** (310) | 22.2 `actq_alpha` (260) | activation quantiser |
| shared expert (NVFP4) | 41.0 `wq_gemm` (490 calls) + 0.7 `silu_glu` | **16.6** `nvfp4_reg` + 13.5 `wq_gemm` (250, tail chunk + other) | 17.3 + 13.6 | gate / up / down |
| `gdn_delta_split_kernel` | 111.1 | 108.9 | 116.2 | GDN recurrence, 30 layers |
| `moe_reduce` / `moe_combine` | 28.8 / 6.3 | 26.8 / 5.7 | 29.7 / 6.0 | |
| `flash_kernel` | 24.6 | 23.4 | 24.4 | full attention, 10 layers |
| norms (`rmsnorm`, `gated_rmsnorm`), `gdn_conv`, rope, gate_split, sigmoid_mul | 70.2 | 68.7 | 73.2 | |
| `e4m3_to_bf16_kernel` | | | 18.9 | cub only: the one-time bf16 weight copies (first prompt) |
| **kernel total** | **802.2** | **748.7** (-6.7 %) | 798.0 (779 without the one-time copies) | |
| launch gaps < 20 us | 3105 gaps, 6.9 ms | 2882, 6.2 ms | | not a lever: 0.8 % |

The window also shows one 330-370 ms host gap before the first cuBLAS call of the process (cuBLAS / cuBLASLt lazy init)
in every arm; `serve` pays it during warmup, so the bench below does not see it.

Reading: the routed experts are 40 % of the GPU time and the fp8 projections 18 %; the 93-token tail chunk costs 51-61 ms
of experts because 744 (token, expert) pairs still touch nearly all 256 experts, i.e. the expert kernels are bound by
weight streaming, not by tokens.

Roofline per 2048-token chunk (GB10: 122 TFLOPS bf16 m16n8k16, 244 e4m3, ~273 GB/s LPDDR5x; numbers per chunk, 40 layers):

| sink | work per chunk | floor | measured (kernel, chunk 2048) | gap |
|---|---|---|---|---|
| MoE experts (256 x top-8, NVFP4) | 4.1 TFLOP; every expert hit, so all ~18 GB of expert weights + block scales stream once per chunk | 66 ms (bandwidth) / 34 ms (bf16 mma) | 262.9 ms / 2 chunks = 131 ms per chunk (2.0 x the bandwidth floor); the 93-token tail chunk alone 51 ms | bandwidth-bound: re-streamed per chunk |
| fp8 dense projections (W8A8, e4m3-valued bf16 mma) | 5.24 TFLOP | 43 ms (bf16 mma) | 83.6 ms `matmul_fp8_gemm_reg` (52 % of peak); 64.7 ms on cuBLAS (opt-in, 67 %) | tile schedule; cuBLAS shows 19 ms are reachable bitwise |
| GDN delta rule (30 layers) | sequential in tokens per head | latency-bound | 108.9 ms / 4189 tokens | | out of scope tonight |
| attention (10 layers, flash, hd 256) | ~2 x 2048 x 4k-ctx x 16 heads x 256 x 2 | small | 23.4 ms | not a lever | |
| norms, rope, actq, combine, small GEMMs | memory | a few ms | ~110 ms (norms, conv, reduce/combine, actq, rope) | fusion candidates, each < 3 % | |
| launch gaps | | 0 | 6.2 ms in 2882 gaps | not a lever | |

## 3. What was built

All three default-on kernels give the bytes of the calls they replace (GPU tests on spark1, below); none changes an
accumulation order, a rounding or the KV / logits path. CUDA backend and `kernels/cuda` only: no shared core change
(`crates/intarsia-core` untouched), so Metal and HIP see no difference.

1. **`matmul_fp8_pair_gemm_reg`** (`kernels/cuda/fp8_gemm.cu`, Op::MatmulFp8Pair at M >= 256). The model already asks for
   the GDN `in_proj_qkv` + `in_proj_z` pair and the attention k + v pair; before, the CUDA registry served the pair only at
   one row, so prefill ran two fp8 GEMMs with two activation quantisers. The new launcher quantises once (when the two
   input scales are equal, checked on the device; true for all 40 layers of the checkpoint) and runs both GEMMs in one grid
   (k + v at N = 512 alone is 1.3 waves of 48 SMs, 2.7 together). The per-tile body is the existing `gemm_reg` tile
   (`gemm_reg_tile`), so every tile runs the same instructions. Microbench M = 2048 (vLLM up): GDN pair 1542 -> 1471 us,
   k + v 234 -> 185 us: **2.6 ms per chunk**.
2. **`matmul_nvfp4_glu_gemm`** (`kernels/cuda/nvfp4_gemm_reg.cu`, Op::MatmulNvfp4Glu at M >= 256): the shared expert's
   gate + up + silu_glu in one launch, weights dequantised in registers (the `moe_w4a16` schedule on dense rows), epilogue
   = the two GEMMs' bf16 stores followed by `silu_glu_kernel`'s arithmetic. 186 -> 135 us: **2.0 ms per chunk**.
3. **`matmul_nvfp4_gemm_reg`** (same file, Op::MatmulNvfp4 at M >= 256, N % 128 == 0): the shared expert's down
   projection, 96 -> 68 us: **1.1 ms per chunk**.
4. **Opt-in `INTARSIA_CUDA_FP8_CUBLAS=1`: `matmul_fp8_gemm_cublas`** (`kernels/cuda/fp8_gemm.cu`). Probe first: cuBLAS
   `GemmEx` (bf16, compute 32F) on e4m3-valued operands gives *exactly* `matmul_fp8_gemm_reg`'s bytes at every 35B
   projection shape and M in {256, 300, 512, 768, 1024, 1500, 2048, 4096}, on e4m3-grid and on full-mantissa activations
   (`raw/dev-spark1/bench/cublas-bits*.log`: 0 differing outputs of 2.6e8): on sm_121 / CUDA 13.0 cuBLAS runs the same
   m16n8k16 chain in k order, no split-K. So the projections can run on cuBLAS's tuned kernels with the bf16-expanded e4m3
   weights. Expanding per call costs more than it saves; the opt-in keeps one bf16 copy per fp8 weight for the
   allocation's lifetime (**+2.6 GB device memory outside the engine's memory plan**), and runs actq + cuBLAS with alpha
   read on the device. Per chunk 83.6 -> **64.7 ms (-18.9 ms)**. Falls back to `gemm_reg` (same bytes) under graph capture,
   for weights outside an `ik_rt_malloc` allocation, or when the copy cannot be allocated. Off by default because of the
   memory outside the plan, not numerics: a decision for the user.
5. **Tried, not shipped: twelve `gemm_reg` schedules** (128 x 256 tiles, k 32 steps, 2 blocks / SM, a fragment-pipelined
   main loop; all bitwise, `INTARSIA_CUDA_FP8_REG_CFG`). Best per-chunk total 80.5 ms vs 84.4 (128 x 256), but shape by
   shape it loses as often as it wins (in_proj_z +7..34 %, k / v +61 % at M = 1024) and the repeat moved several rows by
   5-10 % on the shared box. Cfg0 stays; Cfg1 / Cfg4 / Cfg7 stay selectable for re-measurement. cuBLAS's 64.7 ms says the
   remaining 20 ms are in the tile schedule (it uses TMA + wgmma-style nvjet kernels), not in the dequantisation.

## 4. Speed

`raw/ab-spark2/summary.txt` (job #3822, spark2, vLLM down, `scripts/s1/wALL.sh` -> `scripts/cuda-prefill4/wB-ab.sh`,
fresh `serve` per arm, gen 128, 2 reps, rounds base,dev,cub / cub,dev,base / base,dev,cub). Median of the 3 rounds
[min-max], tok/s:

| size | metric | base f9087322 | dev (default) | cub (opt-in) | dev/base | cub/base |
|---|---|---|---|---|---|---|
| 1k | prefill | 4564 [4551-4576] | 4660 [4610-4678] | 4710 [4673-4781] | **+2.1 %** | +3.2 % |
| 1k | decode | 83.7 [83.6-83.8] | 84.0 [83.1-84.1] | 83.8 [83.5-84.1] | +0.4 % | +0.1 % |
| 4k | prefill | 5855 [5851-5864] | 5989 [5957-5996] | 6155 [6071-6158] | **+2.3 %** | +5.1 % |
| 4k | decode | 81.2 [81.1-81.2] | 81.5 [80.5-81.5] | 81.3 [81.0-81.6] | +0.3 % | +0.1 % |
| 16k | prefill | 5545 [5541-5549] | 5663 [5655-5678] | 5820 [5790-5843] | **+2.1 %** | +5.0 % |
| 16k | decode | 74.0 [73.8-74.0] | 74.0 [73.3-74.2] | 74.2 [74.0-74.6] | +0.0 % | +0.2 % |

localeval speed in the same window (`raw/w-window-2.out`; one `serve` per arm, 3 reps; results under
`/opt/code/deepeval/results/20261022-02*-adhoc-intarsia-s1-*`):

| prompt | base prefill / marginal | dev | cub | vLLM, spark1, round 7 (2026-10-21 10:56) |
|---|---|---|---|---|
| ~4.1k | 5002 / 6100 | 5739 / 6376 | 5836 / 6612 | 6353 / 6862 |
| ~16.7k | 5339 / 5460 | 5432 / 5339 | 5549 / 5461 | 6275 / 6252 |
| decode at 4k / 16k | 82.6 / 74.5 | 81.3 / 73.5 | 80.8 / 73.3 | 76.6 / 73.6 |

localeval's base 1k row (3202 tok/s, TTFT 0.31 s) is the first request after `serve` came up and is not used; its 4k
decode spread (82.6 vs 81.3) is within the per-arm noise of single localeval runs, and the interleaved bench above
(decode +0.0..+0.4 %) is the decode gate.


## 5. Gates

| gate | result | where |
|---|---|---|
| new-kernel GPU tests (bitwise vs the replaced calls), spark1 | `fp8_pair_gemm_reg_is_bitwise_two_calls` (equal / different / weight-only / mixed input scales, bf16 + f32, the 35B shapes, N tails, small grids), `nvfp4_gemm_reg_is_bitwise_the_gemm`, `nvfp4_glu_gemm_is_bitwise_the_chain`, `fp8_gemm_cublas_is_bitwise_gemm_reg` (56 cases incl. all e4m3 codes), every `gemm_reg` schedule bitwise: all PASS | `raw/dev-spark1/tests/` |
| lib pins (selection by M, kernel list, ABI header) | 16 passed (also with `INTARSIA_CUDA_FP8_CUBLAS=1` / `_FP8_GEMM_LT=1`) | `raw/dev-spark1/tests/test-lib.log` |
| `intarsia kernels test --backend cuda` | 58 kernels, 1069 cases, **0 FAIL**. The 4 new kernels ran 0 cases: the sweep's matmul cases stop below M = 256 (changing the shared case list would change the Metal / HIP sweeps); their coverage is the bitwise tests above against admitted kernels | `raw/gates-spark2/ktest/dev.log` |
| greedy ids 35B short (1497 + 256) and 4k (4201 + 64), 8B short and 4k | dev and cub **identical** to base, all 8 | `raw/gates-spark2/ids/` |
| 8B plan (`plan show --json`) | byte-identical (8630 B) | same |
| 35B plan | dev selects `matmul_fp8_pair_gemm_reg`, `matmul_nvfp4_gemm_reg`, `matmul_nvfp4_glu_gemm`; cub `matmul_fp8_gemm_cublas`; scratch unchanged (84.0 MB) | `ids/plan35-*.txt` |
| 35B accuracy KL21, short golden | base / dev / cub 5.430e-3 all rows (4.666e-3 scored), top-1 97.92 %: identical lines, PASS | `raw/gates-spark2/acc/` |
| 35B accuracy KL21, long golden | 4.576e-2 all rows (1.486e-2 scored), top-1 94.17 %: identical lines, PASS | same |
| prefix reuse incl. chat_reuse_4k | continue / restore / aligned 32/32 ok, logits bit-identical, gate PASS, all arms | `acc/reuse-*.log` |
| Metal / HIP | no kernel or engine change: the diff is `crates/intarsia-backend-cuda`, `kernels/cuda`, scripts, docs, and **one shared-core file**, `crates/intarsia-core/src/quant/cases.rs`: main's new `quick_coverage` test needs a quick-sweep case each new kernel accepts, so one prefill-shaped case (M = 260 quick, 300 full) is appended to the `MatmulNvfp4`, `MatmulFp8Pair` and `MatmulNvfp4Glu` sweeps, after the existing cases (their random data is unchanged), as main's fp8 prefill case is. Metal / HIP kernels that accept those shapes now also see them. CPU sweep re-run: 0 FAIL; HIP and CUDA `quick_coverage` pass | `git diff --stat origin/main` |

Accuracy, ids and profiles ran in exclusive slot #3618 on spark2 with vLLM up (idle).

## 6. Next lever: MoE weight streaming

Per 2048-token chunk the routed experts read every expert's NVFP4 weights and block scales once (~18 GB over 40 layers,
66 ms at ~273 GB/s), whatever the token count: the 93-token tail chunk costs 51 ms of experts. Measured 131 ms per full
chunk is 2.0 x that floor, and it is the bulk of the remaining distance to vLLM, which batches up to 8192 tokens per step.
Two directions, both bigger than one night:
1. **Fewer weight passes per token**: a larger prefill chunk (opt-in `--prefill-chunk 4096` already gives +20 % in round
   7, but changes bits against chunk 2048, so making it default is a user decision), or a schedule that keeps more tokens
   per expert per pass.
2. **Closer to the floor per pass**: `moe_w4a16` reaches about half the bandwidth floor; its rounds (5, 6) document the
   per-block prologue / reduce overheads that remain.
Smaller, bitwise: making the cuBLAS projection path default with its 2.6 GB accounted in the memory plan (+3 % more over
dev at 4k), and adding `matmul_fp8_pair_gemm_reg`'s shared quantiser to the cuBLAS path.

## 7. Fleet log

| job | host | what | vLLM |
|---|---|---|---|
| #3502 | spark1, 12 GB | dev build, kernel GPU tests, schedule sweep, cuBLAS probes | up |
| #3618 | spark2 exclusive | build base / dev; gates (ids, plans, profiles, accuracy, reuse, kernels test). Its window step refused itself (the agent-harness check matched this job's own label); fixed | up |
| #3822 | spark2 exclusive, takes down retro | A/B + localeval window | **down 02:08:49-02:26:22, 17.5 min** (`raw/downtime.log`); agent-harness checked first (none) |

Unqueued: Mac `cargo check` / `cargo test --lib` of the CUDA crate (no GPU), as allowed for this lane.
