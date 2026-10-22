# Spark (GB10, CUDA) standings 2026-10-22 (lane fx3): intarsiaml against vLLM, SGLang and llama.cpp

Answers finding B.1 of `docs/reviews/2026-10-22-fable-adversarial.md` ("Spark: #1 of 2 on decode at 1k/4k, #2 of 2 on prefill": n = 2
engines, no noise band, no quality number) the way lane fx6 did for R9700 and Metal
(`docs/results/2026-10-22-fx6-standings.md`): same box, interleaved visits, medians of visit medians,
a stated noise band, withdrawn headlines in `docs/decisions/LOG.md`. Measurement only: no engine code changed. intarsia is the
`origin/main` build 45dd97fd (PR #67 merged), default flags. Raw data, per-visit localeval output and result JSON:
`docs/results/2026-10-22-fx3-spark-standings/raw/`; scripts `scripts/standings-fx3/`.

## 0. Read this first

All speed numbers are **single stream** (concurrency 1) unless a section says otherwise. Ranks are by median of the visit medians; two engines
are **tied** where the min-max bands of their visit medians overlap, and in the prose also where the difference is under the measured noise (about 5 % in decode, 8 % in prefill; see section 1). "Top 3" is a test only where the field has 3 or more engines, so every
table states N. intarsia is always the default build; the `--prefill-chunk 4096` rows are opt-in and are never counted in a rank.

| model on the Spark | N engines | intarsia decode | intarsia prefill | top 3? |
|---|---:|---|---|---|
| Qwen3.6-35B-A3B NVFP4 (same ModelOpt checkpoint in all four; see section 1 for what each engine does with it) | **4** (intarsia, vLLM, SGLang, llama.cpp) | **#1 of 4 at 128 and 1k** (+11.9 % and +9.2 % over vLLM, bands disjoint; SGLang is 7-9 % behind intarsia at 128-4k inside the one vLLM-down window); **#1 at 4k** over vLLM (+7.2 %, bands disjoint); **tied with vLLM at 16k** (+1.1 %) | **tied with vLLM at 1k-4k** (vLLM +6 % marginal, +9 % per-request, bands overlap), both behind SGLang (+19 % at 4k); **#3 of 4 at 16k** (vLLM +18-21 %, SGLang +20 %); ahead of llama.cpp 2.1-2.4x | yes on decode; on prefill #3 of 4 |
| Qwen3-8B BF16 (same checkpoint in all four) | **4** (intarsia, vLLM, SGLang, llama.cpp) | **#1 of 4, tied with llama.cpp** (15.5 against 15.6 tok/s); 9-11 % ahead of vLLM and SGLang at every length | **#4 of 4**: llama.cpp is 1.1-1.2x faster, SGLang 1.7-1.8x and vLLM 1.7-1.9x faster than intarsia's 2.1-2.4k tok/s; the opt-in `--prefill-chunk 4096` lifts intarsia to 3.6-4.2k (section 3.1) | decode yes; **prefill no** |

What this changes against the standings of 2026-10-21 (LOG rows added):

1. **"Spark #1 of 2 on decode at 1k / 4k" is restated, not withdrawn.** With six interleaved visits per engine on the 35B the lead over vLLM is
   +9.2 % at 1k and +7.2 % at 4k with disjoint bands; the 10-19/10-21 figure (+5.7 % / +2.1 %) came from one pair of runs and sat inside
   the ~5 % noise that review B.1 pointed at. Lane og's independent O1 run on spark1 (+9.4 / +7.3 / +5.5 / -2.0 % at 128 / 1k / 4k / 16k) agrees. At 16k the two are tied.
2. **The prefill deficit is the robust finding and is now the headline.** On the 35B intarsia is tied with vLLM at 1k-4k, behind it at 16k (4k-16k marginal prefill
   6.3k against 5.2k tok/s) and behind SGLang from 4k up. On the **8B BF16 it is last of four**.
3. **"Top 3" on Spark is met on the 35B only because the field is 4.** It is not met on 8B prefill.
4. **Concurrency: intarsia on CUDA serves one sequence at a time.** Aggregate decode stays at the single-stream figure at 4 and 16 streams (35B 74 tok/s,
   8B 15 tok/s) while vLLM and SGLang scale (section 4). No batching was implemented or attempted here.
5. **Quality: a tie.** MMLU (n = 1425) 0.840 / 0.840 / 0.833 / 0.836 for intarsia / vLLM / SGLang / llama.cpp on the 35B (standard error about 0.01); 8B MMLU 0.676 / 0.676 / 0.674 / 0.676.

## 1. Method

**Box and engines.** Qwen3.6-35B-A3B on **spark1** (spark-4e77, continuity with the earlier Spark standings and with lane og); Qwen3-8B on
**spark2** (spark-fa00): both GB10, 121 GB unified memory, SM clocks 2398-2411 MHz in every sampled 35B visit and 2398-2483 MHz on the 8B (two samples above 2411) (`raw/*/<arm>-<pass>-gpu.txt`). Every speed pass of a table ran on the same box
under one exclusive fleetq slot with the production vLLM resident (quality passes in non-exclusive slots) and idle (poll of its `/metrics` every 2 s: no foreign request during any
35B pass; `raw/35b/prod-poll-*.log`; spark2's retro vLLM likewise 0).

| engine | build | 35B NVFP4 | 8B BF16 |
|---|---|---|---|
| intarsia | `origin/main` 45dd97fd, `--features cuda`, default flags, `serve --max-seq 20480 --mem-budget 30` (35B) / `32` (8B), `--profile-every 0` | recipe's NVFP4 experts + fp8 projections from the checkpoint | BF16 |
| vLLM | 35B: the **live production** server on spark1 (`vllm 0.26.1rc1.dev602`, image `vllm/vllm-openai:nightly-aarch64`, `--max-model-len 32768 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.50 --enable-prefix-caching --enable-chunked-prefill`, `modelopt_fp4`), not restarted. 8B: own container `fx3-vllm` from the same image and env, `--max-model-len 20480 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.20`, bf16 KV | modelopt_fp4, fp8 KV | BF16 |
| SGLang | `lmsysorg/sglang:qwen38-27b-dflash2` (sglang `0.0.0.dev0+qwen38.27b.g561c8f3`, torch 2.13+cu130), `--context-length 20480 --max-running-requests 16`; 35B `--quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.50`; 8B `--mem-fraction-static 0.36` | modelopt_fp4, fp8 KV | BF16 |
| llama.cpp | `d88040c7` (2026-10-22), `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`, `-ngl 999 -fa on -c 20480 -np 1 --jinja --cache-ram 0` (concurrency arm: `-np 16 -c 81920`) | GGUF made with the repo's `convert_hf_to_gguf.py --outtype auto` from the **same checkpoint**: 241 NVFP4 tensors (18.5 GB: the experts) and 202 **BF16** tensors (5.3 GB: the projections the checkpoint stores in fp8 are dequantized), f16 KV. So not byte-for-byte the same numbers as the other three | BF16 GGUF (`--outtype bf16`) |

**Why SGLang 35B is a separate window.** SGLang needs about 48 GB for this model (`--mem-fraction-static` below 0.4 failed at load with
"not enough memory for the mamba state cache"); that does not fit beside the 64 GB production vLLM (the box has hung before at 64 + 38 GB). Its 35B visits
therefore ran in **two vLLM-down windows** (section 6; speed in the first, quality in the second), together with one intarsia visit (r22) as an in-window reference. Everything else ran with vLLM up.

**Harness.** `localeval` 0.3.0 (`/opt/code/deepeval`), `speed --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off`, nonce-first prompts (no
prefix-cache hit), 1 discarded warmup per size. One **visit** = one server start + the sweep; a cell is the **median of the visit medians**, the
band is **min-max over visits**, n = number of visits. The 1k prompt has 990 tokens, below localeval's 1024 threshold, so its per-request prefill includes request
overhead (this is why vLLM's 1k band is 25 % wide); the **marginal** prefill column (delta tokens / delta TTFT) is the cleanest figure and is what the prefill ranks
use. Concurrency is localeval's `--concurrency 1,4,16`: one burst of N identical-length (990-token) streams, 128 forced output tokens each, one burst per visit.
Passes were rotated (arm order per pass in `raw/*/window-<pass>.log`). Failed starts (SGLang `--mem-fraction-static` too low on p1-p4 of the 8B, p1-p3 of the 35B;
vLLM argument form in p1-p2; llama.cpp context 2560 per slot in the first concurrency attempt) are kept in `raw/*/*-startfail.txt` and are not in any table.

**Measured run-to-run noise (spread of the visit medians, (max - min) / median).** Decode: 0.2-2.7 % for every engine and length except one cell (vLLM 35B at 4k, 5.0 %, one fast visit);
intarsia 2.1-2.4 % on the 35B and 2.1-2.6 % on the 8B. Per-request prefill at 16k: 1-5 %. Per-request prefill at 1k and 4k can reach 13-25 % when one visit is slow (vLLM 35B p2, intarsia4k 8B p31), which is why prefill ranks use the marginal
column. A difference smaller than about 5 % in decode or 8 % in prefill is called a tie in the prose even where the bands do not overlap. Across conditions (vLLM resident or stopped) one intarsia visit moved about 4 % (section 2.1).

**Visit kinds.** `p*` = vLLM resident (the main tables use only these). `r*` = vLLM-down window (own table). `p9*` = quality.

## 2. Qwen3.6-35B-A3B NVFP4 on spark1: single stream

Visits (vLLM up): intarsia 6 (p1-p3, p31-p33), vLLM 6, llama.cpp 3, opt-in intarsia `--prefill-chunk 4096` 3; SGLang only in the window (n = 3, below).

### Single stream (concurrency 1): medians of visit medians, (min-max over visits)

| engine | n visits | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---|---|---|---|
| intarsia | 6 | 84.5 (84.0-85.8) | 84.0 (83.7-85.6) | 81.5 (81.0-82.9) | 73.7 (73.3-75.1) |
| intarsia4k | 3 | 84.7 (83.6-84.8) | 84.0 (83.1-84.3) | 81.7 (80.3-81.8) | 72.0 (71.6-73.7) |
| vllm | 6 | 75.5 (75.1-76.4) | 76.9 (75.9-78.0) | 76.0 (75.5-79.3) | 72.9 (72.6-73.8) |
| llamacpp | 3 | 55.4 (54.8-56.3) | 55.1 (54.5-55.6) | 54.0 (53.5-55.5) | 51.4 (50.7-52.0) |

| engine | prefill 1024 (per-request) | prefill 4096 (per-request) | prefill 16384 (per-request) | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| intarsia | 4324 (4301-4361) | 5673 (5604-5706) | 5309 (5123-5340) | 6295 (6210-6335) / 5199 (4982-5232) | 0.23 (0.23-0.23) / 0.72 (0.72-0.73) / 3.14 (3.12-3.26) |
| intarsia4k | 4331 (4273-4351) | 5977 (5957-5989) | 5513 (5354-5626) | 6811 (6807-6824) / 5377 (5180-5516) | 0.23 (0.23-0.23) / 0.68 (0.68-0.69) / 3.02 (2.96-3.11) |
| vllm | 5034 (3908-5180) | 6210 (5563-6350) | 6252 (6179-6292) | 6702 (6432-6874) / 6273 (6217-6410) | 0.20 (0.19-0.25) / 0.66 (0.64-0.73) / 2.67 (2.65-2.70) |
| llamacpp | 2079 (2076-2117) | 2385 (2383-2409) | 2377 (2361-2388) | 2505 (2498-2521) / 2374 (2354-2382) | 0.48 (0.47-0.48) / 1.71 (1.70-1.72) / 7.01 (6.97-7.06) |


### Ratio to intarsia (median of visit medians; tie = bands overlap)

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia4k | 1.00x tie | 1.00x tie | 1.00x tie | 0.98x tie | 1.00x tie | 1.05x ahead | 1.04x ahead |
| vllm | 0.89x behind | 0.92x behind | 0.93x behind | 0.99x tie | 1.16x tie | 1.09x tie | 1.18x ahead |
| llamacpp | 0.66x behind | 0.66x behind | 0.66x behind | 0.70x behind | 0.48x behind | 0.42x behind | 0.45x behind |


### Run-to-run noise (visit medians): (max-min)/median in %

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia | 2.1 | 2.2 | 2.3 | 2.4 | 1.4 | 1.8 | 4.1 |
| intarsia4k | 1.4 | 1.5 | 1.8 | 2.9 | 1.8 | 0.5 | 4.9 |
| vllm | 1.8 | 2.7 | 5.0 | 1.6 | 25.3 | 12.7 | 1.8 |
| llamacpp | 2.7 | 1.9 | 3.6 | 2.5 | 2.0 | 1.1 | 1.1 |


### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)

| engine | n | 1: agg tok/s / per-stream / TTFT s / errors | 4: agg tok/s / per-stream / TTFT s / errors | 16: agg tok/s / per-stream / TTFT s / errors |
|---|---:|---|---|---|
| intarsia | 3 | 74 (73-74) / 84.2 / 0.23 / 0 | 74 (74-75) / 84.1 / 2.81 / 0 | 74 (74-75) / 84.1 / 13.15 / 0 |
| vllm | 3 | 70 (70-70) / 77.7 / 0.20 / 0 | 105 (104-106) / 59.0 / 1.52 / 0 | 105 (105-106) / 58.7 / 8.82 / 0 |

### 2.1 The vLLM-down window (SGLang, with one intarsia reference visit)

### Single stream (concurrency 1): medians of visit medians, (min-max over visits)

| engine | n visits | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---|---|---|---|
| intarsia | 1 | 81.7 (81.7-81.7) | 81.3 (81.3-81.3) | 78.9 (78.9-78.9) | 71.6 (71.6-71.6) |
| sglang | 3 | 75.0 (74.9-75.0) | 74.5 (73.7-74.7) | 73.6 (72.3-73.8) | 70.8 (70.5-71.4) |

| engine | prefill 1024 (per-request) | prefill 4096 (per-request) | prefill 16384 (per-request) | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| intarsia | 4299 (4299-4299) | 5698 (5698-5698) | 5304 (5304-5304) | 6346 (6346-6346) / 5189 (5189-5189) | 0.23 (0.23-0.23) / 0.72 (0.72-0.72) / 3.14 (3.14-3.14) |
| sglang | 4948 (4774-5018) | 6772 (6732-6828) | 6364 (6332-6384) | 7728 (7668-7767) / 6251 (6202-6251) | 0.20 (0.20-0.21) / 0.60 (0.60-0.61) / 2.62 (2.61-2.63) |


### Ratio to intarsia (median of visit medians; tie = bands overlap)

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| sglang | 0.92x behind | 0.92x behind | 0.93x behind | 0.99x behind | 1.15x ahead | 1.19x ahead | 1.20x ahead |


### Run-to-run noise (visit medians): (max-min)/median in %

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia | - | - | - | - | - | - | - |
| sglang | 0.2 | 1.3 | 2.0 | 1.2 | 4.9 | 1.4 | 0.8 |


### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)

| engine | n | 1: agg tok/s / per-stream / TTFT s / errors | 4: agg tok/s / per-stream / TTFT s / errors | 16: agg tok/s / per-stream / TTFT s / errors |
|---|---:|---|---|---|
| intarsia | 1 | 71 (71-71) / 81.2 / 0.23 / 0 | 72 (72-72) / 81.1 / 2.90 / 0 | 72 (72-72) / 81.0 / 13.60 / 0 |
| sglang | 3 | 67 (67-68) / 74.6 / 0.20 / 0 | 155 (149-158) / 47.4 / 0.62 / 0 | 217 (217-220) / 23.0 / 2.02 / 0 |

How to read it. The in-window intarsia visit (r22, n = 1) decoded **about 4 % slower than every vLLM-up intarsia visit** (81.7 against 84.0-85.8 at 128 tokens) at identical clocks;
the cause is not known. Intarsia therefore moved about 4 % between the two conditions, so the **SGLang-against-vLLM decode comparison (75.0 against 75.5, 1 % apart) is a cross-condition
tie, not a rank**. SGLang against intarsia is the within-window ratio: intarsia ahead by 9 % (128), 9 % (1k), 7 % (4k), 1 % (16k), with the caveat that the reference is one visit.
Prefill did not drift between the conditions (intarsia 4k: 5698 in the window, 5604-5706 with vLLM up), so SGLang's prefill lead over intarsia (+15 % at 1k per-request, +19 % at 4k, +20 % at 16k) and over vLLM
(about +9 % at 4k, +2 % at 16k, the latter inside noise) is reported as measured.

## 3. Qwen3-8B BF16 on spark2: single stream

Visits (vLLM up): see the n column. 8B BF16 decode is memory-bound: the 16.4 GB of weights over 273 GB/s is a ceiling of 16.6 tok/s; intarsia and llama.cpp reach 15.5 (94 %), vLLM and SGLang 14.0 (84 %).

### Single stream (concurrency 1): medians of visit medians, (min-max over visits)

| engine | n visits | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---|---|---|---|
| intarsia | 10 | 15.4 (15.3-15.6) | 15.3 (15.1-15.5) | 14.8 (14.7-15.1) | 13.3 (13.2-13.5) |
| intarsia4k | 6 | 15.3 (15.2-15.3) | 15.1 (15.0-15.2) | 14.7 (14.6-14.8) | 13.2 (13.2-13.3) |
| vllm | 8 | 13.9 (13.9-14.0) | 13.8 (13.8-13.9) | 13.4 (13.4-13.5) | 12.2 (12.0-12.2) |
| sglang | 3 | 13.9 (13.9-13.9) | 13.8 (13.7-13.8) | 13.5 (13.4-13.5) | 12.2 (12.2-12.2) |
| llamacpp | 5 | 15.6 (15.4-15.6) | 15.3 (15.3-15.5) | 14.9 (14.9-15.0) | 13.3 (13.3-13.4) |

| engine | prefill 1024 (per-request) | prefill 4096 (per-request) | prefill 16384 (per-request) | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| intarsia | 2334 (2248-2381) | 2397 (2304-2441) | 2133 (2042-2155) | 2418 (2323-2461) / 2052 (1969-2076) | 0.42 (0.41-0.44) / 1.70 (1.67-1.77) / 7.81 (7.74-8.15) |
| intarsia4k | 4430 (2806-4493) | 4170 (3489-4265) | 3583 (3404-3608) | 4092 (3785-4212) / 3423 (3377-3470) | 0.22 (0.22-0.35) / 0.98 (0.96-1.17) / 4.65 (4.62-4.89) |
| vllm | 4053 (3638-4199) | 4603 (4542-4763) | 3837 (3717-3913) | 4806 (4752-5191) / 3646 (3501-3699) | 0.24 (0.24-0.27) / 0.89 (0.86-0.90) / 4.35 (4.26-4.49) |
| sglang | 3855 (3855-3880) | 4305 (4246-4317) | 3830 (3764-3851) | 4462 (4392-4488) / 3694 (3617-3738) | 0.26 (0.26-0.26) / 0.95 (0.95-0.96) / 4.35 (4.33-4.43) |
| llamacpp | 2833 (2734-2910) | 2865 (2794-2987) | 2417 (2369-2449) | 2914 (2783-3014) / 2289 (2242-2315) | 0.35 (0.34-0.36) / 1.43 (1.37-1.46) / 6.89 (6.80-7.03) |


### Ratio to intarsia (median of visit medians; tie = bands overlap)

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia4k | 0.99x tie | 0.99x tie | 0.99x tie | 1.00x tie | 1.90x ahead | 1.74x ahead | 1.68x ahead |
| vllm | 0.90x behind | 0.90x behind | 0.91x behind | 0.92x behind | 1.74x ahead | 1.92x ahead | 1.80x ahead |
| sglang | 0.91x behind | 0.91x behind | 0.91x behind | 0.92x behind | 1.65x ahead | 1.80x ahead | 1.80x ahead |
| llamacpp | 1.01x tie | 1.00x tie | 1.00x tie | 1.01x tie | 1.21x ahead | 1.20x ahead | 1.13x ahead |


### Run-to-run noise (visit medians): (max-min)/median in %

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia | 2.1 | 2.1 | 2.6 | 2.1 | 5.7 | 5.7 | 5.3 |
| intarsia4k | 0.6 | 1.0 | 1.0 | 0.5 | 38.1 | 18.6 | 5.7 |
| vllm | 1.2 | 1.0 | 0.9 | 1.8 | 13.8 | 4.8 | 5.1 |
| sglang | 0.6 | 0.7 | 0.3 | 0.6 | 0.6 | 1.7 | 2.3 |
| llamacpp | 1.2 | 1.2 | 1.1 | 0.8 | 6.2 | 6.7 | 3.3 |


### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)

| engine | n | 1: agg tok/s / per-stream / TTFT s / errors | 4: agg tok/s / per-stream / TTFT s / errors | 16: agg tok/s / per-stream / TTFT s / errors |
|---|---:|---|---|---|
| intarsia | 7 | 15 (14-15) / 15.3 / 0.43 / 0 | 15 (15-15) / 15.3 / 13.47 / 0 | 15 (15-15) / 15.3 / 65.61 / 0 |
| vllm | 5 | 14 (14-14) / 13.8 / 0.24 / 0 | 52 (52-52) / 14.2 / 0.89 / 0 | 156 (156-157) / 11.6 / 2.04 / 0 |
| sglang | 3 | 14 (14-14) / 13.8 / 0.26 / 0 | 52 (52-52) / 14.2 / 0.90 / 0 | 91 (91-154) / 11.9 / 2.43 / 0 |
| llamacpp16 | 6 | 15 (15-15) / 15.4 / 0.34 / 0 | 48 (46-48) / 13.4 / 1.22 / 0 | 114 (109-114) / 9.2 / 3.92 / 0 |

Why intarsia's 8B prefill is last: the default prefill chunk is 256 tokens (intarsia's own serve log, `prefill chunk 256 (default)`; the lines are the head of the log that the failed-start dumps `raw/8b/vllm-p1-startfail.txt` and `raw/8b/sglang-p1-startfail.txt` captured from the intarsia server that was still running). The opt-in arm `intarsia4k` (`--prefill-chunk 4096`) is in the table above and
close to vLLM and SGLang (section 3.1). At the default chunk the 8B BF16 prefill works out at about 38 TFLOP/s (2.4k tok/s x 16 GFLOP) against about 75 TFLOP/s for vLLM; a per-recipe chunk for the 8B on CUDA is the engine owner's call (defaults stay as they are here).

### 3.1 The prefill gap on the 8B is mostly the default chunk size (opt-in arm `intarsia4k`)

`intarsia4k` is `serve --prefill-chunk 4096`, **opt-in, not a default, not counted in any rank**. On the 8B it lifts prefill from 2.4k to 4.2k tok/s at 4k (1.7-1.9x the default build) and puts
intarsia within 7-10 % of vLLM and SGLang (4k: 4170 against vLLM 4603 and SGLang 4305, bands disjoint from vLLM's; 16k: 3583 against 3837 and 3830); decode is unchanged. On the 35B the same flag gives +5 % at 4k and 16k (section 2).
Chunk size changes the numerics and the prefix-reuse behaviour (round 7 reported the opt-in 4096 build failing `chat_reuse_4k`; LOG 2026-10-21 11:23 kept it opt-in), so this lane only reports it and changes no default. The `--mem-budget 32` used for the
other intarsia 8B visits leaves a 11800-token context with a 4096 chunk, so `intarsia4k` visits p34-p36 ran with `--mem-budget 40` to cover 16k; p31-p33 have no 16k row.
The first intarsia visit (p1) also ran with `--mem-budget 24` (context 11800, no 16k row); its 128-4k rows are in the table.

## 4. Concurrency

localeval at 1 / 4 / 16 streams, 990-token prompts, 128 output tokens, one burst per visit; the aggregate figure is generated tokens over the burst's wall time (so the 1-stream
figure includes prefill and sits below the decode rate). The tables are in sections 2 and 3. Summary (median over visits):

| | streams | intarsia | vLLM | SGLang | llama.cpp |
|---|---:|---|---|---|---|
| 35B aggregate tok/s | 1 / 4 / 16 | 73 / 74 / 74 | 70 / 105 / 105 | 67 / 155 / 217 | not measured |
| 35B TTFT of the median stream (s) | 1 / 4 / 16 | 0.23 / 2.8 / 13.2 | 0.20 / 1.5 / 8.8 | 0.20 / 0.62 / 2.0 | not measured |
| 8B aggregate tok/s | 1 / 4 / 16 | 15 / 15 / 15 | 14 / 52 / 156 | 14 / 52 / 91 | 15 / 48 / 114 |

- **intarsia on CUDA serves one sequence at a time.** Aggregate decode does not move from 1 to 16 streams and requests queue behind each other: the median stream's TTFT is 13 s on the 35B and 65 s on the 8B at 16 streams, and the last one waits about twice that;
  no request failed. This is the known state (continuous batching is Metal-only; `--max-batch` is not wired for CUDA). Nothing was implemented here.
- **The 35B vLLM number is the production setting, not the engine's ceiling.** The production server runs `--max-num-seqs 2`, so it plateaus at about 105 tok/s from 4 streams on. The 8B vLLM arm (`--max-num-seqs 16`) shows what the engine does when allowed: see the 8B row.
- SGLang's 35B figure comes from the vLLM-down window and used `--max-running-requests 16`. SGLang's 8B figure at 16 streams varies from 91 to 154 tok/s across its three visits (median 91), so it is not a rank against vLLM's 156.
- llama.cpp 35B ran with `-np 1`; its concurrency was not measured. The 8B llama.cpp arm used `-np 16`.

## 5. Quality

localeval `quality --bench ... --structured none --thinking off`, the same bench, limit and structured mode for every engine (plain text prompts, so no JSON-schema handling differs between engines).
MMLU is the only figure ranked on: n = 1425 (57 subjects x 25), standard error about 0.01. GSM8K at n = 25 has a standard error of about 0.09; it is shown, not ranked on.

**Qwen3.6-35B-A3B NVFP4 (spark1)**

| engine | visits | benchmark | score | n predictions |
|---|---:|---|---|---:|
| intarsia | 1 | mmlu | 0.840 | 1425 |
| intarsia | 1 | gsm8k | 0.680 | 25 |
| vllm | 1 | mmlu | 0.840 | 1425 |
| vllm | 1 | gsm8k | 0.560 | 25 |
| sglang | 1 | mmlu | 0.833 | 1425 |
| sglang | 1 | gsm8k | 0.600 | 25 |
| llamacpp | 1 | mmlu | 0.836 | 1425 |
| llamacpp | 1 | gsm8k | 0.680 | 25 |

**Qwen3-8B BF16 (spark2)**

| engine | visits | benchmark | score | n predictions |
|---|---:|---|---|---:|
| intarsia | 1 | mmlu | 0.676 | 1425 |
| vllm | 1 | mmlu | 0.676 | 1425 |
| sglang | 1 | mmlu | 0.674 | 1425 |
| llamacpp | 1 | mmlu | 0.676 | 1425 |

## 6. Downtime and fleet

vLLM `guidance` (role flight / guidance / capcom, spark1) was down twice, both inside the 00:00-07:00 free window, no agent-harness job on the Sparks (`fleetq status` before each; the `--takes-down` request itself waits for users of the role), each `fleetq run --takes-down flight,guidance,capcom` with an EXIT trap that
restarts the container (`docker start missioncntrl-guidance`) and waits for `/v1/models`:

| fleetq job | stop | back up | down | what |
|---|---|---|---:|---|
| #3977 | 05:38:48 | 05:59:25 | 20.6 min | SGLang 35B speed + concurrency x 3, intarsia x 1 (r21-r24) |
| #3990 | 06:24:18 | 06:37:04 | 12.7 min | SGLang 35B quality (MMLU + GSM8K) |

Total 33.3 min against the 3 h cap; windows under 30 min; 25 min of serving between them. spark2's retro vLLM was never touched.
fleetq jobs of this lane (owner `claude-f31ccf91-fx3`). Builds and conversions, vLLM up, non-exclusive: spark2 #3925 (intarsia), #3926 and #3929 (llama.cpp, 8B GGUF), #3932 (35B GGUF, not used); spark1 #3941 (intarsia), #3942 (llama.cpp), #3958 (35B GGUF).
Read-only looks: #3921-#3924, #3927, #3931, #3940, #4018. Speed passes (exclusive, vLLM up): 8B on spark2 #3943 (p1), #3944 (p2), #3945 (p3), #3961 (p4), #3962 (p5), #3963 (p6), #4004 (p7), opt-in #3994-#3996 (p31-p33) and #4124-#4126 (p34-p36);
35B on spark1 #3959 (p1), #3969 (p2), #3970 (p3), opt-in #3991-#3993 (p31-p33). Quality (non-exclusive): #3979 (35B), #3980 (8B). vLLM-down windows: #3977, #3990. Cancelled or killed before they produced anything:
#3928, #3930, #3934-#3936 (smoke tests replaced by the first real passes), #3933 (smoke), #3971 (a fourth 35B pass that was not needed). The `smoke*` raw files are debug runs and are in no table. Every slot is released.

## 7. What is not covered

- **No tp=2 arm.** intarsia on two Sparks against vLLM on one (review B.3) is not part of this table; tp=2 numbers are in `2026-10-21-tp2-cuda-graphs.md`.
- The llama.cpp 35B model is not the same weights format as the other three (section 1). Its decode rank is therefore a statement about that GGUF.
- Prompt lengths above 16k, speculative decoding and structured output were not run.
- The SGLang 35B visits are cross-condition against vLLM (section 2.1).
- Quality was run once per engine (n = 1 visit); GSM8K is too small to rank on.
- spark2's vLLM is configured with `--max-model-len 131072`, spark1's with 32768; the 35B was measured on spark1 and the 8B (own container) on spark2, so no cross-box claim is made.
- vLLM's per-request 1k prefill band is wide (25 %): one slow visit (p2) at 1k and 4k; the marginal column does not show it.

## 8. Files

- `scripts/standings-fx3/`: `window.sh` (one pass = one visit per arm, runs inside a fleetq slot), `window-opt.sh` (same plus the opt-in `intarsia4k` arm), `takedown-window.sh` (the vLLM-down window with watchdog and restore trap), `remote/start-sglang.sh`,
  `build-llamacpp.sh`, `convert35.sh`, `aggregate.py`, `make-doc.py` (fills this document from the raw data), `doc.tmpl.md`, `smoke*.sh` (smoke tests, not results).
- `docs/results/2026-10-22-fx3-spark-standings/raw/{35b,8b}/`: per visit `<arm>-<pass>-{speed,conc,quality}.{out,result.json,summary.md}`, GPU samples, production-vLLM poll, `window-<pass>.log`, start failures; `raw/downtime.log`, `raw/aggregate-*.md`, build and conversion logs.
