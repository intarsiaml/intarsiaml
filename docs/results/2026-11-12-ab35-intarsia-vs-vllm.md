# ab35: intarsiaml vs production vLLM, Qwen3.6-35B-A3B NVFP4, one DGX Spark (spark2)

## User go-ahead

2026-11-12 ~16:35 EST: the user picked option "1" from a list that said it needs vLLM down on one Spark for about 45 minutes. This lane (ab35) runs under that go-ahead: spark2's vLLM (`missioncntrl-guidance`, role `retro`) is stopped for the measurement window and restored afterwards; hard cap 60 min down.

Lane `ab35`, branch `ab35/intarsia-vs-vllm`, fleetq owner `claude-ab35`. Tree under test: origin/main `8beddc5e2` plus this doc (build stamp `ab35-d3a72bfca76`).

## Question

Could intarsiaml replace the production vLLM for the Qwen3.6-35B-A3B NVFP4 on one DGX Spark? Same harness (localeval 0.4.0, `/opt/code/deepeval`, git `d96f899`), same suite, same box (spark2, GB10), same settings, one engine at a time. This is a measurement; it changes no default and recommends no cutover beyond what the numbers show.

## Setup

Both arms ran on spark2 and were measured by localeval from endor over the LAN (`http://spark-fa00:<port>/v1`, model name `guidance`). Settings were identical in every call:

- thinking off, `--structured prompt`, temperature 0 (localeval sends it), `max_tokens` 1024 for quality;
- speed: `--sizes 128,1024,4096,16384 --gen 256 --reps 3 --concurrency 2` (nonce-first prompts, `ignore_eos` + `min_tokens` forcing exactly 256 tokens, one discarded warmup per size, median of 3 measured reps, 2 streams at 1k-token prompts);
- quality: `gsm8k --limit 100 --pin-gsm8k --nonce` and `ifeval --limit 25 --nonce`;
- each arm's localeval ran inside one `--quiet gpu,cpu` fleetq slot on spark2 (`--wait-idle 600`).

### Arm A: production vLLM (container `missioncntrl-guidance`, role `retro`), left up and untouched

`docker inspect` of the running container (image `missioncntrl/vllm-openai:65b7662d`, vLLM `0.26.1rc1.dev602+g65b7662d3`, `--network host`, `--shm-size 16g`, restart `unless-stopped`):

```
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom
  --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32
  --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
env: VLLM_USE_FLASHINFER_MOE_FP4=0  VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
```

There is **no speculative config**: production vLLM decodes plainly.

### Arm B: intarsiaml `serve`, shipped CUDA defaults, built on spark2 in `~/intarsia-ab35`

```
~/intarsia-ab35/bin/intarsia serve --backend cuda --model <HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a>
  --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 20480 --mem-budget 40 --max-batch 2 --profile-every 0
```

Built with `cargo build --release --locked --features cuda -p intarsia-cli` (`MAX_JOBS=4`, 5 min 45 s, fleetq #9158, `--load cpu`, vLLM up). What the recipe `[tuning.cuda]` ships and the server log confirms (`raw/serve-B-default.log`): `--spec mtp --spec-k 2` default (via `batched_mtp`: a sequence drafts while at most 1 sequence decodes), bf16 KV, prefill chunk 2048 (macro prefill 8192 for a prompt that prefills alone), GatedDeltaNet state bf16, `--max-batch 2`. MTP was running: 7563 of 10932 proposed draft tokens accepted (69 %), 5469 verify steps, 0 cost-gate fall-backs (`raw/serve-B-default.metrics`).

Two B arms: **B-default** (as shipped, MTP k2) and **B-specoff** (`--spec off`, speed only). B-specoff exists because A has no speculation: B-specoff vs A is the engine-to-engine comparison, B-default vs A is the as-shipped comparison. Quality was run on B-default only (greedy MTP verify is bitwise the plain decode ids per the lane docs).

Port 8191 (not 8000) so no production client could reach the experimental engine while vLLM was down.

### Timeline

| step | fleetq | time (EST, endor clock) |
|---|---|---|
| build on spark2 (`--load cpu`) | #9158 | 18:24 to 18:29 |
| arm A suite (vLLM up) | #9159 | 18:29:49 to 18:33:15 |
| smoke: B beside vLLM on :8191 (min MemAvailable 51 GB) | #9160 | 18:33 to 18:34 |
| two exploratory A GSM8K variants (not in tables, see quality) | #9161, #9162 | 18:34 to 18:36 |
| window 1 (ABORTED by me: localeval choked on the `--recipe` string, 'File name too long'; no B data) | #9163 | vLLM down 18:36:19, answering again 18:40:53 (274 s) |
| window 2 (this data): `--takes-down retro --quiet gpu,cpu --expect 60 --mem 48` | #9164 | vLLM down 18:41:22, answering `/v1/models` 200 at 18:49:21 (479 s) |
| arm A repeat after the restart | #9165 | 18:49:39 to 18:50:37 |

vLLM was down 274 s + 479 s in total (about 12.5 min, far under the 60 min cap); each restore waited for `/v1/models` 200. The window script restores on any exit (trap) and the first window's abort exercised that path. Lowest MemAvailable during B alone: 87.6 GB (B-default), 89.1 GB (B-specoff); earlyoom active on spark2. The aborted window's raw files are kept in `raw/aborted-1/`.

## Results

localeval stores median, min and max of the 3 measured reps (not each rep), so cells read median (min-max). All four arms: `cached_tokens_max` 0, i.e. no prefix-cache hit in any speed call, on vLLM either (nonce-first prompts defeat its prefix cache). Prompt sizes are the real token counts the server reported.

### Decode tok/s, single stream, 256 forced tokens: median (min-max of 3 reps)

| arm | ~128 | ~1k | ~4k | ~16k |
|---|---|---|---|---|
| A (vLLM, before) | 76.8 (76.2-77.3) | 77.6 (77.5-77.7) | 76.5 (76.2-77.7) | 73.6 (73.5-73.7) |
| A-post (vLLM, after restart) | 75.9 (75.8-77.4) | 77.5 (75.6-77.5) | 76.5 (76.3-76.6) | 73.5 (73.3-73.6) |
| B-default (intarsiaml MTP k2) | 107.4 (103.5-110.9) | 105.1 (104.2-130.0) | 107.6 (105.9-107.9) | 88.2 (83.2-95.6) |
| B-specoff (intarsiaml --spec off) | 86.6 (86.5-86.6) | 86.0 (86.0-86.0) | 83.3 (81.5-83.4) | 75.4 (75.3-75.4) |

### Peak decode tok/s (fastest 32-token stretch): median (min-max)

| arm | ~128 | ~1k | ~4k | ~16k |
|---|---|---|---|---|
| A (vLLM, before) | 93.1 (90.5-94.2) | 93.5 (93.2-97.0) | 90.9 (90.5-91.8) | 89.7 (87.5-137.8) |
| A-post (vLLM, after restart) | 91.7 (90.8-93.5) | 92.8 (92.4-93.7) | 92.0 (91.2-92.2) | 87.7 (87.3-88.8) |
| B-default (intarsiaml MTP k2) | 163.2 (151.7-238.8) | 210.5 (155.4-210.9) | 156.9 (155.9-173.2) | 112.6 (103.9-124.4) |
| B-specoff (intarsiaml --spec off) | 108.1 (107.8-135.2) | 107.3 (107.0-111.5) | 102.8 (102.1-252.1) | 92.4 (90.1-201.5) |

### TTFT s: median (min-max)

| arm | ~128 | ~1k | ~4k | ~16k |
|---|---|---|---|---|
| A (vLLM, before) | 0.09 (0.08-0.14) | 0.18 (0.18-0.19) | 0.65 (0.64-0.69) | 2.66 (2.66-2.67) |
| A-post (vLLM, after restart) | 0.08 (0.08-0.08) | 0.18 (0.18-0.18) | 0.63 (0.63-0.63) | 2.61 (2.60-2.62) |
| B-default (intarsiaml MTP k2) | 0.10 (0.10-0.15) | 0.19 (0.19-0.19) | 0.59 (0.58-0.59) | 2.51 (2.51-2.51) |
| B-specoff (intarsiaml --spec off) | 0.10 (0.10-0.17) | 0.19 (0.19-0.19) | 0.59 (0.59-0.59) | 2.51 (2.51-2.58) |

### Prefill tok/s (uncached prompt_tokens / TTFT): median (min-max)

| arm | ~128 | ~1k | ~4k | ~16k |
|---|---|---|---|---|
| A (vLLM, before) | 1225 (789-1401) | 5404 (5142-5407) | 6288 (5911-6374) | 6254 (6241-6257) |
| A-post (vLLM, after restart) | 1351 (1330-1435) | 5484 (5447-5488) | 6463 (6454-6510) | 6393 (6354-6402) |
| B-default (intarsiaml MTP k2) | 1319 (886-1348) | 5241 (5166-5282) | 6988 (6935-7045) | 6632 (6631-6636) |
| B-specoff (intarsiaml --spec off) | 1146 (695-1149) | 5274 (5168-5327) | 6933 (6918-6951) | 6637 (6473-6647) |

### Actual prompt tokens, cached_tokens_max, marginal prefill (tok/s)

| arm | prompt tokens | cached_tokens_max | marginal prefill 1k/4k/16k |
|---|---|---|---|
| A (vLLM, before) | 112/991/4090/16654 | 0 | 9554/6636/6241 |
| A-post (vLLM, after restart) | 112/990/4090/16678 | 0 | 8869/6858/6370 |
| B-default (intarsiaml MTP k2) | 135/989/4091/16654 | 0 | 9930/7814/6523 |
| B-specoff (intarsiaml --spec off) | 114/992/4089/16679 | 0 | 9865/7704/6547 |

### 2 streams (1k-token prompts, 256 forced tokens each)

| arm | aggregate decode tok/s | per-stream decode median (min-max) | TTFT s median (min-max) |
|---|---|---|---|
| A (vLLM, before) | 113.1 | 60.5 (59.8-61.1) | 0.30 (0.25-0.35) |
| A-post (vLLM, after restart) | 112.1 | 60.1 (59.6-60.6) | 0.31 (0.27-0.35) |
| B-default (intarsiaml MTP k2) | 122.1 | 65.5 (63.9-67.1) | 0.28 (0.19-0.39) |
| B-specoff (intarsiaml --spec off) | 125.6 | 67.4 (65.7-69.1) | 0.28 (0.19-0.38) |

run ids: A (vLLM, before): `20261112-182950-adhoc-ab35-A-speed`, A-post (vLLM, after restart): `20261112-184939-adhoc-ab35-A-post-speed`, B-default (intarsiaml MTP k2): `20261112-184207-adhoc-ab35-B-default-speed`, B-specoff (intarsiaml --spec off): `20261112-184521-adhoc-ab35-B-specoff-speed`


### Quality (thinking off, temperature 0, `--nonce`)

| benchmark | A vLLM | B-default intarsiaml | note |
|---|---|---|---|
| GSM8K flexible, n=100, pinned prompt | 0.590 (run `20261112-183050-adhoc-ab35-A-gsm8k`) | 0.660 (`20261112-184255-adhoc-ab35-B-default-gsm8k`) | SE about 0.05: a 0.07 gap is inside 2 SE |
| IFEval, n=25 | 0.920 (`20261112-183155-adhoc-ab35-A-ifeval`) | 0.920 (`20261112-184348-adhoc-ab35-B-default-ifeval`) | identical |

GSM8K detail: with thinking off the pinned template makes the model answer with a bare number in 76 (A) and 69 (B) of 100 cases, and those bare answers are right 35 times on both arms; the replies that show their working are right 24/24 (A) and 31/31 (B). So the 0.59 vs 0.66 gap is only the mix of reply styles (24 vs 31 worked answers). Correctness agrees on 85 of 100 questions and the reply text is identical on 57, so the engines do not decode identically (bf16 vs fp8 KV and different kernels), but this test cannot tell them apart in quality. Two extra A-only variants (`--structured prompt` unpinned 0.550 with 12-token answers, `--structured none` 0.580) show the thinking-off GSM8K number is a reply-style artefact of the harness, not a capability measure. Thinking-on quality, long-context quality and sampled (temperature above 0) quality were not run.

## Fairness notes

- **KV cache dtype**: vLLM fp8 e4m3 (8 GiB pool); intarsiaml bf16 (the checkpoint declares FP8 KV scales; intarsiaml does not use them and says so in its log). bf16 KV reads twice the bytes per token, which costs intarsiaml at long context and is part of why its 16k decode gain shrinks; fp8 KV is an opt-in for intarsiaml (`--kv-dtype fp8`, `docs/results/2026-10-29-fp8kv-opt-in.md`) and was not tried here.
- **Speculation**: A none; B-default MTP k2, which drafts only while one sequence decodes (at 2 streams B-default is plain batched decode, hence 122 vs 126 tok/s, B-default 3 % under B-specoff). Judge decode by B-specoff vs A for the engine itself and by B-default vs A for the shipped product.
- **Prefill scheduling**: vLLM chunked prefill with `--max-num-batched-tokens 8192`; intarsiaml 2048-row chunks (macro pass up to 8192 rows when alone).
- **Context and memory**: vLLM serves 131072 tokens in 38 GB; B was run with `--max-seq 20480 --mem-budget 40` and was alone on the GPU (vLLM stopped). B at 131072 context, and B beside the other roles, were not measured.
- **Time order**: A before, B in the window, A again after: A drifted at most 2 % (decode 76.8/77.6/76.5/73.6 vs 75.9/77.5/76.5/73.5), so the box was stable. The first A run followed 6 days of vLLM uptime, A-post a fresh container.
- **Noise**: decode about 5 %, long-prompt prefill about 3 %, mid-length prefill up to 39 % (unanimis 6a7efe40). B-default decode had wide spreads at 1k (104-130) and 16k (83-96); medians are used, no rerun was done. Prefill and TTFT under about 1k uncached tokens are request overhead.
- **Not the same as production traffic**: the fleetq quiet slot stops other fleetq jobs, not missioncntrl requests; every localeval activity check on A read idle.
- **Different frontends**: parsers differ (production vLLM: `qwen3_xml` tool parser and `qwen3` reasoning parser; intarsiaml recipe: `hermes` and think-tags). No tool-calling or reasoning-parser equivalence test was run.

## Verdict per metric (this data only)

| metric | where it lands | size |
|---|---|---|
| Decode, single stream, as shipped (MTP k2) | **intarsiaml wins** at every length | 1.40x / 1.35x / 1.41x / 1.20x at 128 / 1k / 4k / 16k (107 / 105 / 108 / 88 vs 77 / 78 / 77 / 74 tok/s) |
| Decode, single stream, no speculation (engine only) | intarsiaml ahead, shrinking with length | 1.13x / 1.11x / 1.09x at 128 / 1k / 4k (about 2x the 5 % noise), 1.02x at 16k = parity |
| Prefill | parity to a small intarsiaml lead | 4k +8 to +11 %, 16k +4 to +6 % (near the 3 % noise), 1k -3 to -4 % (inside noise), 128 parity |
| TTFT | split | intarsiaml 6-9 % lower at 4k (0.59 vs 0.63-0.65 s) and 4-6 % at 16k (2.51 vs 2.61-2.66 s); vLLM 0.01-0.02 s lower at 128 and 1k (request overhead) |
| 2-stream aggregate decode (1k prompts) | intarsiaml wins | 122.1 (B-default) / 125.6 (B-specoff) vs 113.1 / 112.1 tok/s: +8 to +12 %; per-stream 65.5-67.4 vs 60.1-60.5 |
| Quality (GSM8K n=100, IFEval n=25, thinking off) | parity within noise | GSM8K 0.66 vs 0.59 (inside 2 SE, reply-style mix), IFEval 0.92 vs 0.92 |

Reading: on one Spark, for this model and these short single-turn probes, intarsiaml as shipped is not behind vLLM on any measured metric and is ahead on decode (mostly from MTP) and 2-stream throughput. The 2.4-2.8x prefill gap of 2026-10-13 is gone (prefill is level with vLLM here). Whether that is enough to replace vLLM is **not** settled by this data. Not measured: contexts beyond 16k (vLLM serves 131072; B ran at 20480), more than 2 streams, prefix-cache-heavy agent traffic (vLLM has prefix caching on; intarsiaml's prefix store is off by default and excluded with MTP), thinking-on quality, tool calling and the reasoning/tool parsers, sampled requests (MTP applies to greedy requests only), long soak and stability, and intarsiaml with `--kv-dtype fp8` or the other roles resident.

## Raw results

- Run directories (summary.md, result.json): `docs/results/2026-11-12-ab35-intarsia-vs-vllm/raw/<run id>/`; the localeval originals are under `/opt/code/deepeval/results/<run id>/` and each run was logged to unanimis.
- A speed `20261112-182950-adhoc-ab35-A-speed` (unanimis 35803f53), A gsm8k `20261112-183050-adhoc-ab35-A-gsm8k`, A ifeval `20261112-183155-adhoc-ab35-A-ifeval` (94a87015), A-post speed `20261112-184939-adhoc-ab35-A-post-speed`.
- B-default speed `20261112-184207-adhoc-ab35-B-default-speed` (e3eeb9a6), gsm8k `20261112-184255-adhoc-ab35-B-default-gsm8k`, ifeval `20261112-184348-adhoc-ab35-B-default-ifeval`; B-specoff speed `20261112-184521-adhoc-ab35-B-specoff-speed`.
- Logs: `raw/window.log`, `raw/serve-B-default.{log,metrics}`, `raw/serve-B-specoff.{log,metrics}`, `raw/*.out` (localeval stdout), `raw/aborted-1/` (window 1). Scripts: `.ab35/` in the worktree (not committed; `window.sh`, `le.sh`, `build.sh`, `smoke.sh`, `memwatch.sh`).

## Decisions

None to flip. The user's go-ahead (top of this file) is the only decision; one row in `docs/decisions/LOG.md` records it.
