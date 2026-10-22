# S1: CUDA 35B prefill on DGX Spark: three bitwise prefill kernels, an opt-in cuBLAS projection path (2026-10-22)

Target: DGX Spark GB10 (sm_121, CUDA 13.0), Qwen3.6-35B-A3B-NVFP4, tp=1, default (numerics-preserving) path. Lane S1 of the
2026-10-21 evening round. Branch `lane-s1/spark-prefill`, base origin/main `f9087322`. Raw logs: `2026-10-22-s1-spark-prefill/raw`.
The FP8-E4M3 activation path (`INTARSIA_CUDA_FP8_GEMM_LT=1`, lane F2) is untouched; the new kernels step aside when it is on.

## 1. Headline

<!-- HEADLINE -->

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

<!-- SPEED -->

## 5. Gates

<!-- GATES -->

## 6. Next lever: MoE weight streaming

<!-- NEXT -->

## 7. Fleet log

<!-- FLEET -->
