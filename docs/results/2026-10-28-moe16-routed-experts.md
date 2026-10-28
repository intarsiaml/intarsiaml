# moe16: routed-MoE experts at 8 and 16 streams, Qwen3.6-35B-A3B NVFP4, one GB10: roofline, and what vLLM does differently

Lane moe16, 2026-10-28, branch `lane-moe16/routed-experts` from `origin/main` 318d508fb (#224). spark1 (spark-4e77, GB10 sm_121).
Status: **draft PR, gates not attempted**. Step 1, the roofline check with the distinct experts **measured** in serve: the routed
experts run at **84 to 88 % of 230 GB/s** (72 to 75 % of the 273 GB/s peak) at 8 and 16 streams, at the task's 85 % ceiling line.
Full recovery to 230 GB/s would be 3.1 ms per 16-stream step, about +5.5 % aggregate, only at perfect efficiency, and the only
lever on the remaining cost (expert-major processing of the hot experts) is the row-grouping the brief marks dead. So no kernel
was built and no serve A/B was spent. vLLM reads the same routed-expert and GDN bytes (section 4); its step lead is mostly elsewhere
(section 5).

**New measured fact that changes #222's premise**: a 16-row decode step touches **61 to 66 distinct experts per layer**, not about
100 (2.0 pairs per expert, not 1.3), and distinct prompts give the same D as identical-body prompts: the sharing comes from expert
popularity skew, not from prompt similarity (section 3).

GPU windows used: **1 of 2** (#6592, the route-stats serve; section 6). Raw: `2026-10-28-moe16-routed-experts/raw/`. Scripts: `scripts/moe16/`.

## 1. Bytes per routed expert, from the model

Qwen3.6-35B-A3B (`config.json` text_config): 40 layers, 256 routed experts, top-8, `moe_intermediate_size` 512, hidden 2048
(shared expert 512 wide, separate). `hf_quant_config.json`: routed and shared experts `W4A16_NVFP4`, group 16 (e2m1 codes, one
e4m3 scale per 16 weights, one f32 global scale per tensor).

| tensor | weights | e2m1 bytes | e4m3 scale bytes | total |
|---|---:|---:|---:|---:|
| gate_up `[2*512, 2048]` | 2,097,152 | 1,048,576 | 131,072 | 1,179,648 |
| down `[2048, 512]` | 1,048,576 | 524,288 | 65,536 | 589,824 |
| **one expert, one layer** | | | | **1,769,472 B (1.77 MB)** |
| one distinct expert in all 40 layers | | | | 70.8 MB |

(#222 wrote 1.72 MB: the same bytes in MiB.) Activations, ids and the `h` scratch are under 0.1 MB per layer and are ignored.

The byte floor of the routed experts per decode step is `D x 1.77 MB x 40`, where D is the number of **distinct** experts per layer
in that step. If the B rows routed independently and uniformly, D = 256 (1 - (248/256)^B): 102.0 at B=16, 57.4 at B=8.

## 2. Roofline: byte floor against measured

Measured routed-expert time per step (`moe_dec_gate_up` + `moe_dec_down_v2`, nsys of a burst, #224's W1 base / W2 dev profiles,
`2026-10-28-msd2-dense-gemv.md` section 1-2): **21.96 / 22.16 ms at B=16, 13.24 / 13.84 ms at B=8**.

| | B=16 | B=8 |
|---|---:|---:|
| independent routing: D | 102.0 | 57.4 |
| independent routing: bytes per step | 7.22 GB | 4.06 GB |
| independent routing: floor at 230 GB/s (at 273 GB/s peak) | 31.4 ms (26.4) | 17.7 ms (14.9) |
| measured | 21.96 ms | 13.24 ms |
| bandwidth the measured time would need with independent routing | 329 GB/s | 307 GB/s |

The measured times are **below** the independent-routing floor at the 273 GB/s peak, so the serve rows do not route independently:
the burst's streams (same prompt body after a per-stream prefix) share experts. The floor therefore has to come from the D the
workload actually has. Two ways to get it:

**(a) D measured in serve** (window 1, `scripts/moe16/route.sh`, `INTARSIA_MOE_ROUTE_STATS=1`, section 3), on the same burst load
the profiles timed (`burst.py`, same prompt body after a per-stream prefix; routing is bitwise the same eager or graph-replayed, so
this D belongs to the profiled times): **D = 61.45 at B=16, 37.91 at B=8**.

| | B=16 | B=8 |
|---|---:|---:|
| measured D (burst load) | 61.45 | 37.91 |
| pairs per distinct expert | 2.08 | 1.69 |
| bytes per step | 4.35 GB | 2.68 GB |
| floor at 230 GB/s / at the kernel's 243 GB/s (b, below) | 18.9 / 17.9 ms | 11.7 / 11.0 ms |
| measured, #224 base / dev profile | 21.96 / 22.16 ms | 13.24 / 13.84 ms |
| **achieved, base / dev** | **198 / 196 GB/s = 86 / 85 % of 230** (73 / 72 % of peak) | **203 / 194 GB/s = 88 / 84 % of 230** (74 / 71 % of peak) |
| headroom to 230 GB/s / to 243 GB/s | 3.1 / 4.1 ms | 1.5 / 2.2 ms |

(#224 called its base-to-dev MoE difference, +0.2 / +0.6 ms, run-to-run noise; dev is the same MoE code as base.)

**(b) Kernel efficiency at a known D.** #222's evicted micro-timing (`timing_moe_rows`, 35B layer, weights cold, independent random
top-8 per row, so D is the formula above) gives the bandwidth of the same per-pair kernels on the distinct experts they read:

| T (rows) | 2 | 4 | 5 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| us per layer (per-pair kernels, #222 section 4) | 123.9 | 224.3 | 272.4 | 416.1 | 586.9 | 744.7 |
| D (independent) | 15.8 | 30.5 | 37.6 | 57.4 | 81.1 | 102.0 |
| distinct-expert bytes | 27.9 MB | 54.0 MB | 66.5 MB | 101.6 MB | 143.5 MB | 180.4 MB |
| **achieved on distinct bytes** | 225 GB/s | 241 GB/s | 244 GB/s | 244 GB/s | 245 GB/s | 242 GB/s |

So from 4 rows up the per-pair kernels stream the distinct experts at about 243 GB/s: 106 % of the 230 GB/s assumed achievable, 89 %
of the 273 GB/s peak. Pairs that share an expert are deduplicated in L2 (the gate_up grid is pair-fastest, the down grid
token-fastest, so blocks that read the same expert rows run next to each other). The same table's shared-routing rows ("same 8
experts", "two groups") are flat at about 365 us (T=16) and 187 us (T=8): a per-pair floor (dequantise and FMA once per pair, out of
L2) that only matters when D is small. `time ~ max(per-pair floor, D x 1.77 MB / 243 GB/s)` fits all three rows of that table.

That max model fits the micro rows but not serve: at the measured D it predicts 17.9 ms at B=16 against 22 ms measured. Serve
lands between max and sum of the two costs. With skewed routing a few hot experts carry many pairs (D = 61 for 128 pairs, against
102 if uniform: the uniform formula overstates D by 35 to 40 %), and their per-pair work out of L2 no longer hides under the
DRAM stream of the cold ones. That per-pair work on shared experts is the about 3 to 4 ms over the floor at B=16.

**Verdict**: 84 to 88 % of 230 GB/s, at the 85 % line. Recovering all of it to 230 GB/s is 3.1 ms x 255 steps = 0.79 s of a 15.27 s
16-stream burst, about +5.5 % aggregate (to 243 GB/s about +7 %); half of that misses the +5 % gate. The cost sits in per-pair work
on hot experts, and the remedy for that is expert-major processing (read an expert once, do all its pairs), which is the row-grouping
#222 built and the brief marks as a dead end; per-pair ALU tuning already had about 10 iterations in #222. So no kernel was built.

## 3. Measured distinct experts (window 1)

`INTARSIA_MOE_ROUTE_STATS=1` (`kernels/cuda/moe_decode.cu`, section 7) copies the ids of every row-exact MoE call back and prints
cumulative means per row count T every 400 calls; `INTARSIA_BATCH_GRAPHS=0` makes the batched steps eager (bitwise the graphs,
tuning.md) so the hook runs. One server (`serve --max-batch 16`, the dev CLI = main plus the hook), four loads in a row; per-load means
are differences of the cumulative snapshots, `(calls2 x mean2 - calls1 x mean1) / (calls2 - calls1)` (`raw/route/snap-p*.txt`). The
print granularity leaves at most 399 calls per T outside a snapshot, under 5 % of each load's calls.

| load | T=16: calls | T=16: mean D (min..max) | T=8: calls | T=8: mean D |
|---|---:|---:|---:|---:|
| p1 `burst.py` 16 streams (the msd/msd2 profile load) | 9600 | **61.45** (32..103) | | |
| p2 `burst.py` 8 streams | | | 10000 | **37.91** (14..61) |
| p3 localeval `speed --sizes 1024 --gen 256 --reps 1 --concurrency 8,16` (the gated benchmark's prompts) | 9600 | **66.11** | 10000 | **41.57** |
| p4 `burst.py` 16, `DISTINCT=1` (a different 3.7k-char document slice per stream) | 9600 | **66.30** | | |
| uniform independent top-8 (formula) | | 102.0 | | 57.4 |

Smaller row counts during the ramps (p4, cumulative): T=2 13.0, 3 19.2, 4 24.2, 5 27.2, 6 31.2, 7 35.5 distinct of 8T pairs.
p3's localeval figures (205.5 / 256.6 tok/s) are of the eager, syncing server, not a benchmark.

Reading: distinct documents per stream (p4) touch the same number of experts as one shared body (p1, p3 within 8 %), so the
sharing is not prompt similarity but how concentrated this model's expert popularity is. #222 assumed about 100 distinct experts
(1.3 pairs per expert) from the independent-uniform formula; the measured 2.0 pairs per expert is what an expert-major kernel would
amortise over. #222's serve A/B of its grouped kernel was nevertheless null on this same load (section 4 of its doc), so the
measured D does not by itself say a grouped kernel wins.

## 4. What vLLM does differently (source of the running image, read-only)

vLLM image `missioncntrl/vllm-openai:65b7662d` (0.26.1rc1.dev602+g65b7662d3), source read inside the running `missioncntrl-guidance`
container on spark1 (`/usr/local/lib/python3.12/dist-packages/vllm`), launched for vr16 with `--quantization modelopt_fp4
--max-num-seqs 16 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill`,
`VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`. No speculative decoding (no MTP). Paths are
relative to that directory.

| part | vLLM | ours | bytes |
|---|---|---|---|
| routed experts | Marlin W4A16 MoE (`fused_moe/experts/marlin_moe.py`): `moe_align_block_size` sorts the pairs by expert with `block_size_m` = 8 at 16 x 8 pairs (`marlin_moe.py:333-345`), so each distinct expert is read once; gate_up GEMM, separate silu_and_mul, down GEMM with the router weight folded in, `torch.sum` over top-k: 4 to 5 kernels. Selected because the checkpoint is `W4A16_NVFP4` (`quantization/modelopt.py:1392-1397`: `activation_key=None` rules out every W4A4 backend; the log names MARLIN) | 2 per-pair kernels, L2 dedups shared experts | **same** (distinct experts once) |
| GDN state | **f32**: `Qwen3_5MoeForConditionalGeneration` maps to `Qwen3_5ForConditionalGenerationConfig` (`models/config.py:935`), whose hook copies the HF `mamba_ssm_dtype: float32` into `mamba_ssm_cache_dtype` when it is `auto` (`config.py:783-785`); `mamba/mamba_utils.py:96-108`. Conv state bf16. Decode: `causal_conv1d_update` plus the Triton `fused_recurrent_gated_delta_rule_packed_decode`, state updated in place, one read and one write | f32, one read and one write | **same** |
| fp8 projections | CUTLASS `cutlass_scaled_mm` (W8A8, static per-tensor activation scale from the checkpoint, separate quant op fused by inductor); GDN `in_proj_qkvz` one merged GEMM, `in_proj_ba` one merged GEMM; attention QKV merged | row-exact GEMVs per projection | same weight bytes; vLLM reads each once per step with a tensor-core GEMM |
| lm_head | NVFP4, `MarlinNvFp4LinearKernel` (W4A16) | nvfp4 rows GEMV | same |
| shared expert | separate MLP (merged gate_up) on an **aux CUDA stream, overlapped with the routed experts** (`fused_moe/runner/shared_experts.py:99-112`) | in the main stream | same bytes, hidden latency |
| KV cache | **fp8** (`--kv-cache-dtype fp8`), FlashInfer decode | bf16 | vLLM reads half the KV bytes (about 0.38 GB against 0.19 GB per step at 16 streams, 1.15k mean context: about 0.8 ms) |
| graphs | `FULL_AND_PIECEWISE`: a pure-decode batch replays one FULL graph incl. GDN and attention; capture sizes 1, 2, 4, 8, 16, ... with `--max-num-seqs 16` (derived from `config/vllm.py:1884-1934`; the live container runs `--max-num-seqs 2`, so this was not observed) | one graph per row count | same |
| prefill of a burst | chunked prefill, `--max-num-batched-tokens 8192`: up to 8 prompts of 1k in one step, mixed with decode | prompts prefilled one after another | not bytes: scheduling |

Not verified: the Marlin and CUTLASS kernel internals (compiled `.so`), and the CUDA graph sizes at seqs 16 (derived from config code).

**So vLLM reads the same routed-expert bytes and the same f32 GDN state as we do.** Our GDN is at its byte floor; our routed experts
read the same bytes at 85 % of 230 GB/s (80 % of the kernel's own 243). vLLM's Marlin is expert-major (`moe_align_block_size`), so it
does each hot expert's pairs from one read: if it streams at the kernel-level 243 GB/s, that is worth up to about 4 ms of its step
lead at 16 streams. Unverified: no vLLM profile was taken.

## 5. Where the 16-stream gap is (estimate from the localeval numbers, not a vLLM profile)

localeval's aggregate is total completion tokens over the burst's wall time (`localeval/speed.py:129-133`), so it includes the
prefill of all streams. From the vr16 and #224 W2 tables (`--sizes 1024 --gen 256`, 4096 tokens at 16 streams, 2048 at 8):

| | ours (#224) | vLLM seqs 16 | difference |
|---|---:|---:|---:|
| 16 streams: wall time of the burst | 15.27 s (268.2 tok/s) | 12.04 s (340.1) | 3.23 s |
| 16: slowest first token (TTFT max, vr16) | 3.38 s | 2.19 s | 1.19 s |
| 16: decode step (ours nsys dev profile; vLLM from its per-stream decode 24.6 tok/s) | 46.9 ms | 40.7 ms (38.6 if the wall time is TTFT max + 255 steps) | 6 to 8 ms, x 255 steps = 1.6 to 2.1 s |
| 8 streams: wall time | 9.44 s (216.9) | 8.21 s (249.6) | 1.23 s |
| 8: TTFT max | 1.60 s | 1.11 s | 0.49 s |
| 8: decode step | 32.9 ms | 27.8 ms (36.0 tok/s per stream) | about 5 ms |

About a third of the 16-stream gap is the burst's prefill (ours serial, vLLM chunked and batched), two thirds a 6 to 8 ms shorter decode
step. GDN (8.9 ms) is at its floor on the same bytes as vLLM; the routed experts may carry up to about 3 to 4 ms of the step gap
(section 2: 22 ms against a 17.9 to 18.9 ms floor; vLLM's expert-major Marlin may get closer to it; unverified). The rest has to sit
in the remaining buckets, which are ours at about 15 ms against a byte floor of about 9 ms at B=16:

| bucket at B=16 (#224 dev profile) | ours | byte floor at 230 GB/s | vLLM's way |
|---|---:|---:|---|
| fp8 projections (1.28 GB) | 8.1 ms | 5.6 ms | merged CUTLASS GEMMs |
| lm_head nvfp4 (286 MB) | 1.85 ms | 1.24 ms | Marlin |
| shared expert nvfp4 (71 MB) | about 1.4 ms | 0.31 ms | merged gate_up, aux stream overlapped with routed experts |
| bf16 small GEMVs (router, GDN b/a, shared gate) | 0.69 ms | about 0.2 ms | merged b/a |
| attention (KV 0.38 GB bf16) | 2.4 ms | 1.6 ms (0.8 with fp8 KV) | fp8 KV, FlashInfer |
| norms, top-k, glue | 0.7 ms | small | inductor-fused |
| **sum** | **about 15 ms** | **about 9 ms** | |

That is about 6 ms of headroom at B=16; with the experts' 3 to 4 ms, more than the 6 to 8 ms step gap, so vLLM need not be at the
floor everywhere. These are estimates: no vLLM profile was taken (the task allowed one confirm window; it went to measuring D).

## 6. Process

- GPU windows: **1 of 2**. #6592 (`scripts/moe16/route.sh`, `--quiet gpu,cpu --mem 50`, about 3 min): the route-stats server and
  the four loads of section 3. Window 2 not used: no kernel to A/B (section 2), and the ceiling branch allows one confirm window.
- No micro-benchmark jobs.
- `--load cpu` builds (not windows): #6589 (dev CLI with the route-stats diagnostic).
- spark1-vllm (role guidance/capcom/flight) stayed up; no `--takes-down`.

## 7. What changed in the tree

- `kernels/cuda/moe_decode.cu`: `route_stats()`, called by the row-exact launchers (`max_t > 0`) only when
  `INTARSIA_MOE_ROUTE_STATS=1`; skipped while a stream is capturing. With the variable unset it returns at the first branch: no change
  to launches, results or `plan show`.
- `docs/tuning.md`: its row; the cited lines of `INTARSIA_MOE_DOWN_R` / `INTARSIA_MOE_GU_UNROLL` updated.
- `scripts/moe16/`: `build.sh`, `route.sh`, `burst.py` (copy of `scripts/msd/burst.py` with the corpus path of this lane).

## 8. Next steps, by size at 16 streams

1. Burst prefill scheduling (about 1.2 s of a 15.3 s burst, equal to about 4.7 ms per step): batch the waiting prompts of a burst
   into one prefill (vLLM: up to 8192 tokens per step), or interleave chunked prefill with decode.
2. fp8 projections, 8.1 against 5.6 ms: the K=2048 row GEMVs and the qkv/z (and b/a) projections as one merged read.
3. Shared expert on a second stream overlapped with the routed experts, and the lm_head / bf16 small GEMVs (together about 2 ms
   over their floors).
4. fp8 KV cache for decode attention (about 0.8 ms at 16 streams; a numerics change, opt-in).
5. Expert-major routed experts (read each distinct expert once, then all its pairs; what vLLM's Marlin does), up to about 3 to
   4 ms at 16 streams now that D is measured at 2.0 pairs per expert. Caveat: the brief marks row-grouping a dead end, and #222's
   grouped kernel was null in serve on this load (4 to 9 % slower on uniform routing in its micro); a new attempt would need a
   micro on the measured skew first (`INTARSIA_MOE_ROUTE_STATS`, or a dump of real ids) and is the user's call.
