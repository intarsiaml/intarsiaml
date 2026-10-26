# sk: single-stream CUDA prefill of the 35B on a Spark, against vLLM (2026-10-25)

Lane `sk`, branch `lane-sk/spark-prefill`, base origin/main `cf84efa4` (PR #149 merged). DGX Spark spark2 (GB10, sm_121), `Qwen3.6-35B-A3B-NVFP4`,
default recipe (chunk 2048, PF on, fp8 lt on), default env. Raw logs: `2026-10-25-sk-spark-prefill/raw/`, scripts: `scripts/sk/`.
Every GPU use was a fleetq job on spark2 (ids in section 7). **Status: scope done; build in progress (section 3).**

## 0. Premise check: the gap the brief describes is gone at 4k

The brief starts from 10-22 (-9 % at 4k, -15 % at 16k vs vLLM). Since then PF (#131), the cuBLASLt fp8 projections (#117 lt, default for W8A8
since 10-23) and the GEMM-pair kernels landed. Fresh numbers, spark2, `localeval speed` sizes 1024/4096/16384, gen 128, 3 reps, thinking off,
one fresh `intarsia serve` per round (default flags, `--max-seq 20480 --mem-budget 45`), vLLM = spark2's own `retro`-role server (serving `guidance`
and `capcom`; `--max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.32
--enable-prefix-caching --enable-chunked-prefill`), 3 interleaved rounds (round 2 reversed), one fleetq slot (#5438, `--quiet gpu,cpu`):

| prefill tok/s, median of 3 round medians (min-max) | 1k (993 tokens) | 4k (4089) | 16k (16655) |
|---|---|---|---|
| intarsia main `cf84efa4` | 5255 (5172-5308) | 6547 (6474-6573) | 6015 (6000-6025) |
| vLLM (spark2) | 5470 (5372-5475) | 6382 (6362-6409) | 6299 (6288-6306) |
| intarsia vs vLLM, per request | -3.9 % (localeval: overhead-dominated at 1k) | **+2.6 %** | **-4.5 %** |
| marginal prefill (delta tokens / delta TTFT) | | 1k-4k: 7117 vs 6758 (+5.3 %) | 4k-16k: 5866 vs 6271 (-6.5 %) |
| TTFT s | 0.189 / 0.182 | 0.624 / 0.640 | 2.769 / 2.643 |

(`raw/base3/`, aggregated by `scripts/sk/agg.py`; decode: main 85.3 / 82.8 / 75.2 tok/s at 1k / 4k / 16k, vLLM 80.0 / 78.8 / 76.2.) At 4k main is ahead;
the remaining gap is at 16k (-4.5 % per request, -6.5 % marginal), where main's throughput falls from 6547 to 6015 tok/s while vLLM's stays flat.
The vLLM reference is spark2's server, not spark1's as the brief names: same checkpoint and flags, same box as the intarsia run, so no cross-box
difference; it reproduces the 10-22 figure of the same server (6389 / 6140 then, 6382 / 6299 now).

The benchmark prompts matter for the levers: the 4k prompt is 4089 tokens = 2048 + 2040 + 1 (the hybrid schedule runs the last token alone), so there
is no short tail; the 16k prompt is 16655 = 8 x 2048 + 271 (+1): a 271-row tail pass that streams all expert weights again.

## 1. Where a prefill goes (nsys `-t cuda`, `generate --max-tokens 1`, spark2, main; `raw/prof-main/`)

Prompts as lane sp2: 14000 / 55000 chars of `tools/prompts/corpus_docs.txt` = 4189 / 16355 tokens (4189 = 2048 + 2048 + 93: the 4k profile carries a
93-row tail pass that the localeval 4k prompt does not have). Kernel time summed by family, % of the kernel total (674.7 ms / 2643.0 ms):

| op family | 4k ms | 4k % | 16k ms | 16k % | note |
|---|---|---|---|---|---|
| routed experts, `moe_w4a16_kernel` gate/up + down (PF on at <= 2048 rows) | 262.2 | 38.9 | 1047.0 | 39.6 | 3.28 ms per layer and 2048-row chunk |
| routed experts, 93-row tail (`moe_mma_kernel`) | 51.7 | 7.7 | 0 | 0 | one full weight pass for 93 rows |
| GDN delta rule (`gdn_delta_split_kernel`) | 111.5 | 16.5 | 439.8 | 16.6 | strict left-to-right chain per head: latency bound, bitwise-pinned |
| dense matmuls (fp8 lt `nvjet`/`xmma`, `lt_actq8`, bf16 router / in_a / in_b via cuBLAS) | 97.3 | 14.4 | 342.3 | 13.0 | vendor GEMMs |
| norms (`rmsnorm`, `gated_rmsnorm`) | 36.3 | 5.4 | 142.2 | 5.4 | at the byte floor |
| `moe_reduce` + `moe_combine` | 32.4 | 4.8 | 128.6 | 4.9 | 234 us per call, ~320 GB/s: at the floor |
| attention `flash_kernel` (10 layers) | 24.1 | 3.6 | 309.6 | 11.7 | grows with context (16k: 66 TFLOP/s) |
| GDN conv | 21.7 | 3.2 | 84.9 | 3.2 | |
| shared expert (`nvfp4_reg`) | 16.5 | 2.4 | 68.0 | 2.6 | |
| rest | 21.1 | 3.1 | 80.6 | 3.1 | |

Idle gaps under 20 us total 6.9 / 17.1 ms (1 % / 0.6 %): not a lever. The engine's own `--op-table` agrees on the order (experts 41 %, matmul 22 %,
GDN delta 13 %, fp8 matmul 10 % of a synchronised, timed pass; `raw/prof-main/*/optable.err`).

The routed experts are 51 % of a 4189-token prefill and 40 % of a 16k one. Their time is a serial sum of weight DRAM streaming, A-tile L2 traffic and a
fixed skeleton (lane sp2 ablations, `docs/results/2026-10-22-sp2-cuda-prefill.md` section 2), and every 2048-row chunk streams all ~18 GB of expert
weights again. A call with more rows reads each weight once for all of them (lane mp microbench, `docs/results/2026-10-23-mp-cuda-moe-prefill.md`,
launcher incl. grouping and reduce, uniform routing: 2048 rows 3506 us with PF, 4096 rows 5504 us, i.e. 1508 us less than two 2048-row calls per layer,
x 40 layers = 60 ms = 9.7 % of the 4089-token TTFT).

## 2. Scope: levers, bitwise status, expected gain

All gains are against main's TTFT at the benchmark prompts (0.624 s at 4k, 2.769 s at 16k).

| # | lever | bitwise? | expected | notes |
|---|---|---|---|---|
| L1 | **macro prefill pass**: run several consecutive 2048-row chunks of one prompt layer by layer in one pass (the existing packed-prefill machinery with one state, each chunk a segment); the row-wise kernels (experts, shared expert, norms, quantisers) see all rows and read their weights once, attention, KV append and the GDN recurrence run per chunk; the vendor GEMMs (cuBLAS / cuBLASLt pick their tiling by row count) run per chunk | yes by construction if every batched kernel is row-wise bitwise, the vendor GEMMs stay chunk-sized | 4k: -60 ms (+10 %); 16k: the 271-row tail pass (~55 ms) goes away and the experts run at 8192 rows per pass (microbench: 8192 rows 9840 us launcher vs 4 x 3506 us): about +8 to +12 % | opt-in (`INTARSIA_PREFILL_MACRO=<rows>`), because a long pass delays a request that arrives meanwhile; the chunk stays 2048, so PR #138 is untouched |
| L2 | extend the PF (L2 prefetch) schedule of the gate/up kernel past 64 pairs per expert (> 2048 rows per call) | yes (same sums) | +1 to +2 % on top of L1 at 4k (PF is worth ~1.4 % at 4k at 2048 rows) | only useful once L1 makes calls larger |
| L3 | `moe_reduce` fused into the down GEMM (last arriver per token and column block) | yes | <= 3.6 % at 4k, needs a counter array in shared scratch | lane sp2 section 4; not attempted |
| L4 | `moe_reduce` + `moe_combine` + next `RmsNormAdd` fused | yes | <= 1.5 % | |
| L5 | flash attention schedule at long context | no (summation order of the softmax) | up to 3 to 5 % at 16k | opt-in only; 16k only |
| L6 | GDN delta (16.5 %) | bitwise-pinned strict chain | 0 | latency bound; nothing bitwise |

Sum of the bitwise levers at 4k: L1 alone is ~10 %, well above the 3 % bar, so the lane goes on to step 3 with L1 (below).
