# Spark (GB10, CUDA) standings 2026-10-22 (lane fx3): intarsiaml against vLLM, SGLang and llama.cpp

Answers finding B.1 of `docs/reviews/2026-10-22-fable-adversarial.md` ("Spark: #1 of 2 on decode at 1k/4k, #2 of 2 on prefill": n = 2
engines, no noise band, no quality number) the way lane fx6 did for R9700 and Metal
(`docs/results/2026-10-22-fx3-spark-standings/../2026-10-22-fx6-standings.md`): same box, interleaved visits, medians of visit medians,
a stated noise band, withdrawn headlines in `docs/decisions/LOG.md`. Measurement only: no engine code changed. intarsia is the
`origin/main` build 45dd97fd (PR #67 merged), default flags. Raw data, per-visit localeval output and result JSON:
`docs/results/2026-10-22-fx3-spark-standings/raw/`; scripts `scripts/standings-fx3/`.

## 0. Read this first

All speed numbers are **single stream** (concurrency 1) unless a section says otherwise. Ranks are by median of the visit medians; two engines
are **tied** where the min-max bands of their visit medians overlap. "Top 3" is a test only where the field has 3 or more engines, so every
table states N. intarsia is always the default build; the `--prefill-chunk 4096` rows are opt-in and are never counted in a rank.

| model on the Spark | N engines | intarsia decode | intarsia prefill | top 3? |
|---|---:|---|---|---|
| Qwen3.6-35B-A3B NVFP4 (same ModelOpt checkpoint in all four; see section 1 for what each engine does with it) | **4** (intarsia, vLLM, SGLang, llama.cpp) | **#1 of 4 at 128 and 1k** (+11.8 % and +9.2 % over vLLM, bands disjoint; SGLang is 8-9 % behind intarsia inside the one vLLM-down window); **#1 at 4k** over vLLM (+7.2 %, bands disjoint); **tied with vLLM at 16k** (+1.1 %) | **#3 of 4** at 4k and 16k: behind SGLang by 19-20 % and vLLM by 7-20 % (marginal figure), ahead of llama.cpp 2.1-2.4x | yes on decode; on prefill #3 of 4 |
| Qwen3-8B BF16 (same checkpoint in all four) | **4** (intarsia, vLLM, SGLang, llama.cpp) | **#1 of 4, tied with llama.cpp** (15.5 against 15.6 tok/s); 10 % ahead of vLLM and SGLang at every length | **#4 of 4**: llama.cpp is 1.2x faster, vLLM and SGLang 1.7-2.0x faster than intarsia's 2.3-2.4k tok/s | decode yes; **prefill no** |

What this changes against the standings of 2026-10-21 (LOG rows added):

1. **"Spark #1 of 2 on decode at 1k / 4k" is restated, not withdrawn.** With six interleaved visits per engine on the 35B the lead over vLLM is
   +9.2 % at 1k and +7.2 % at 4k with disjoint bands; the 10-19/10-21 figure (+5.7 % / +2.1 %) came from one pair of runs and sat inside
   the ~5 % noise that review B.1 pointed at. Lane og's independent O1 run on spark1 (+9.4 / +7.3 / +5.5 / -2.0 % at 128 / 1k / 4k / 16k) agrees. At 16k the two are tied.
2. **The prefill deficit is the robust finding and is now the headline.** On the 35B intarsia is behind vLLM at 4k-16k (marginal prefill
   6.2-6.3k against 5.0-5.2k tok/s) and behind SGLang. On the **8B BF16 it is last of four**.
3. **"Top 3" on Spark is met on the 35B only because the field is 4.** It is not met on 8B prefill.
4. **Concurrency: intarsia on CUDA serves one sequence at a time.** Aggregate decode stays at the single-stream figure at 4 and 16 streams (35B 74 tok/s,
   8B 15 tok/s) while vLLM and SGLang scale (section 4). No batching was implemented or attempted here.
5. **Quality: a tie.** MMLU (n = 1425) 0.840 / 0.840 / 0.833 / 0.836 for intarsia / vLLM / SGLang / llama.cpp on the 35B (standard error about 0.01).

## 1. Method

**Box and engines.** Qwen3.6-35B-A3B on **spark1** (spark-4e77, continuity with the earlier Spark standings and with lane og); Qwen3-8B on
**spark2** (spark-fa00): both GB10, 121 GB unified memory, clocks 2398-2411 MHz in every sampled visit (`raw/*/<arm>-<pass>-gpu.txt`). Every pass of a table ran on the same box
under one exclusive fleetq slot with the production vLLM resident and idle (poll of its `/metrics` every 2 s: no foreign request during any
35B pass; `raw/35b/prod-poll-*.log`; spark2's retro vLLM likewise 0).

| engine | build | 35B NVFP4 | 8B BF16 |
|---|---|---|---|
| intarsia | `origin/main` 45dd97fd, `--features cuda`, default flags, `serve --max-seq 20480 --mem-budget 30` (35B) / `32` (8B), `--profile-every 0` | recipe's NVFP4 experts + fp8 projections from the checkpoint | BF16 |
| vLLM | 35B: the **live production** server on spark1 (`vllm 0.26.1rc1.dev602`, image `vllm/vllm-openai:nightly-aarch64`, `--max-model-len 32768 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.50 --enable-prefix-caching --enable-chunked-prefill`, `modelopt_fp4`), not restarted. 8B: own container `fx3-vllm` from the same image and env, `--max-model-len 20480 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.20`, bf16 KV | modelopt_fp4, fp8 KV | BF16 |
| SGLang | `lmsysorg/sglang:qwen38-27b-dflash2` (sglang `0.0.0.dev0+qwen38.27b.g561c8f3`, torch 2.13+cu130), `--context-length 20480 --max-running-requests 16`; 35B `--quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.50`; 8B `--mem-fraction-static 0.36` | modelopt_fp4, fp8 KV | BF16 |
| llama.cpp | `d88040c7` (2026-10-22), `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`, `-ngl 999 -fa on -c 20480 -np 1 --jinja --cache-ram 0` (concurrency arm: `-np 16 -c 81920`) | GGUF made with the repo's `convert_hf_to_gguf.py --outtype auto` from the **same checkpoint**: 241 NVFP4 tensors (18.5 GB: the experts) and 202 **BF16** tensors (5.3 GB: the projections the checkpoint stores in fp8 are dequantized), f16 KV. So not byte-for-byte the same numbers as the other three | BF16 GGUF (`--outtype bf16`) |

**Why SGLang 35B is a separate window.** SGLang needs about 48 GB for this model (`--mem-fraction-static` below 0.4 failed at load with
"not enough memory for the mamba state cache"); that does not fit beside the 64 GB production vLLM (the box has hung before at 64 + 38 GB). Its 35B visits
therefore ran in **two vLLM-down windows** (section 6), together with one intarsia visit (r22) as an in-window reference. Everything else ran with vLLM up.

**Harness.** `localeval` 0.3.0 (`/opt/code/deepeval`), `speed --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off`, nonce-first prompts (no
prefix-cache hit), 1 discarded warmup per size. One **visit** = one server start + the sweep; a cell is the **median of the visit medians**, the
band is **min-max over visits**, n = number of visits. The 1k prompt has 990 tokens, below localeval's 1024 threshold, so its per-request prefill includes request
overhead (this is why vLLM's 1k band is 25 % wide); the **marginal** prefill column (delta tokens / delta TTFT) is the cleanest figure and is what the prefill ranks
use. Concurrency is localeval's `--concurrency 1,4,16`: one burst of N identical-length (990-token) streams, 128 forced output tokens each, one burst per visit.
Passes were rotated (arm order per pass in `raw/*/window-<pass>.log`). Failed starts (SGLang `--mem-fraction-static` too low on p1-p4 of the 8B, p2 and p3 of the 35B;
vLLM argument form in p1-p2; llama.cpp context 2560 per slot in the first concurrency attempt) are kept in `raw/*/*-startfail.txt` and are not in any table.

**Visit kinds.** `p*` = vLLM resident (the main tables use only these). `r*` = vLLM-down window (own table). `p9*` = quality.

## 2. Qwen3.6-35B-A3B NVFP4 on spark1: single stream

Visits (vLLM up): intarsia 6 (p1-p3, p31-p33), vLLM 6, llama.cpp 3, opt-in intarsia `--prefill-chunk 4096` 3; SGLang only in the window (n = 3, below).

@@T35@@

### 2.1 The vLLM-down window (SGLang, with one intarsia reference visit)

@@T35R@@

How to read it. The in-window intarsia visit (r22, n = 1) decoded **about 4 % slower than every vLLM-up intarsia visit** (81.7 against 84.0-85.8 at 128 tokens) at identical clocks;
the cause is not known. Intarsia therefore moved about 4 % between the two conditions, so the **SGLang-against-vLLM decode comparison (75.0 against 75.5, 1 % apart) is a cross-condition
tie, not a rank**. SGLang against intarsia is the within-window ratio: intarsia ahead by 8 % (128), 8 % (1k), 7 % (4k), 1 % (16k), with the caveat that the reference is one visit.
Prefill did not drift between the conditions (intarsia 4k: 5698 in the window, 5604-5706 with vLLM up), so SGLang's prefill lead over intarsia (+15 % at 1k per-request, +19 % at 4k, +20 % at 16k) and over vLLM
(about +9 % at 4k, +2 % at 16k, the latter inside noise) is reported as measured.

## 3. Qwen3-8B BF16 on spark2: single stream

Visits (vLLM up): see the n column. 8B BF16 decode is memory-bound: the 16.4 GB of weights over 273 GB/s is a ceiling of 16.6 tok/s; intarsia and llama.cpp reach 15.5 (94 %), vLLM and SGLang 14.0 (84 %).

@@T8@@

Why intarsia's 8B prefill is last: its default prefill chunk is 256 tokens (server log, `raw/8b/vllm-p1-startfail.txt`); the opt-in `--prefill-chunk 4096` arm (`intarsia4k`) is in the table above. It does not close the gap by itself
(section 3.1); the BF16 GEMM throughput (about 38 TFLOP/s at 2.3k tok/s against about 66 TFLOP/s for vLLM at 4.0k) is the difference. Not investigated further here.

@@T8NOTE@@

## 4. Concurrency

localeval at 1 / 4 / 16 streams, 990-token prompts, 128 output tokens, one burst per visit; the aggregate figure is generated tokens over the burst's wall time (so the 1-stream
figure includes prefill and sits below the decode rate). The tables are in sections 2 and 3. Summary (median over visits):

| | streams | intarsia | vLLM | SGLang | llama.cpp |
|---|---:|---|---|---|---|
| 35B aggregate tok/s | 1 / 4 / 16 | 73 / 74 / 74 | 70 / 105 / 105 | 67 / 155 / 217 | not measured |
| 35B TTFT of the median stream (s) | 1 / 4 / 16 | 0.23 / 2.8 / 13.2 | 0.20 / 1.5 / 8.8 | 0.20 / 0.62 / 2.0 | not measured |
| 8B aggregate tok/s | 1 / 4 / 16 | @@C8I@@ | @@C8V@@ | @@C8S@@ | @@C8L@@ |

- **intarsia on CUDA serves one sequence at a time.** Aggregate decode does not move from 1 to 16 streams and the 16th stream waits for the 15 before it (13 s TTFT on the 35B, 65 s on the 8B);
  no request failed. This is the known state (continuous batching is Metal-only; `--max-batch` is not wired for CUDA). Nothing was implemented here.
- **The 35B vLLM number is the production setting, not the engine's ceiling.** The production server runs `--max-num-seqs 2`, so it plateaus at about 105 tok/s from 4 streams on. The 8B vLLM arm (`--max-num-seqs 16`) shows what the engine does when allowed: see the 8B row.
- SGLang's 35B figure comes from the vLLM-down window and used `--max-running-requests 16`.
- llama.cpp 35B ran with `-np 1`; its concurrency was not measured. The 8B llama.cpp arm used `-np 16`.

## 5. Quality

localeval `quality --bench ... --structured none --thinking off`, the same bench, limit and structured mode for every engine (plain text prompts, so no JSON-schema handling differs between engines).
MMLU is the only figure ranked on: n = 1425 (57 subjects x 25), standard error about 0.01. GSM8K at n = 25 has a standard error of about 0.09; it is shown, not ranked on.

@@Q@@

## 6. Downtime and fleet

vLLM `guidance` (role flight / guidance / capcom, spark1) was down twice, both inside the 00:00-07:00 free window, no agent-harness job on the Sparks (`fleetq status` before each; the `--takes-down` request itself waits for users of the role), each `fleetq run --takes-down flight,guidance,capcom` with an EXIT trap that
restarts the container (`docker start missioncntrl-guidance`) and waits for `/v1/models`:

| fleetq job | stop | back up | down | what |
|---|---|---|---:|---|
| #3977 | 05:38:48 | 05:59:25 | 20.6 min | SGLang 35B speed + concurrency x 3, intarsia x 1 (r21-r24) |
| #3990 | 06:24:18 | 06:37:04 | 12.7 min | SGLang 35B quality (MMLU + GSM8K) |

Total 33.3 min against the 3 h cap; windows under 30 min; 25 min of serving between them. spark2's retro vLLM was never touched.
Other fx3 slots: builds #3925 (intarsia, spark2), #3926 and #3929 (llama.cpp, GGUF, spark2), #3941 and #3942 (spark1), #3958 and the 35B GGUF (spark1, spark2 #3932),
read-only looks #3921-#3924, #3927, #3931, #3940, #4018; speed passes: 8B #3943 #3944 #3945 #3961 #3962 #3963 and the late ones (below), 35B #3959 #3969 #3970 and the opt-in passes #3991-#3993;
quality #3979 (35B) and #3980 (8B); the opt-in 8B passes #3994-#3996 and p7 #4004. Every slot is released.

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
  `build-llamacpp.sh`, `convert35.sh`, `aggregate.py`, `smoke*.sh` (smoke tests, not results).
- `docs/results/2026-10-22-fx3-spark-standings/raw/{35b,8b}/`: per visit `<arm>-<pass>-{speed,conc,quality}.{out,result.json,summary.md}`, GPU samples, production-vLLM poll, `window-<pass>.log`, start failures; `raw/downtime.log`, `raw/aggregate-*.md`, build and conversion logs.
