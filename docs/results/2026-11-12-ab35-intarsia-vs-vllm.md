# ab35: intarsiaml vs production vLLM, Qwen3.6-35B-A3B NVFP4, one DGX Spark (spark2)

## User go-ahead

2026-11-12 ~16:35 EST: the user picked option "1" from a list that said it needs vLLM down on one Spark for about 45 minutes. This lane (ab35) runs under that go-ahead: spark2's vLLM (`missioncntrl-guidance`, role `retro`) is stopped for the measurement window and restored afterwards; hard cap 60 min down.

Lane `ab35`, branch `ab35/intarsia-vs-vllm`, fleetq owner `claude-ab35`. Tree under test: origin/main `8beddc5e2` plus this doc (build stamp `ab35-d3a72bfca76`).

## Question

Could intarsiaml replace the production vLLM for the Qwen3.6-35B-A3B NVFP4 on one DGX Spark? Same harness (localeval 0.4.0, `/opt/code/deepeval`, git `d96f899`), same suite, same box (spark2, GB10), same settings, one engine at a time. This is a measurement; it changes no default and recommends no cutover beyond what the numbers show.

## Setup

Both arms ran on spark2 and were measured by localeval from endor over the LAN (`http://spark-fa00:<port>/v1`, model name `guidance`). Settings were identical in every call:

- thinking off, `--structured prompt`, temperature 0 (localeval sends it), `max_tokens` 1024 for quality;
- speed: `--sizes 128,1024,4096,16384 --gen 256 --reps 3 --concurrency 2` (nonce-first prompts, `ignore_eos` + `min_tokens` forcing exactly 256 tokens, two discarded 1-token calls up front, median of 3 measured reps, 2 streams at 1k-token prompts);
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

Two B arms: **B-default** (as shipped, MTP k2) and **B-specoff** (`--spec off`, speed only). B-specoff exists because A has no speculation: B-specoff vs A is the engine-to-engine comparison, B-default vs A is the as-shipped comparison. Quality was run on B-default only; output equality between B-default and `--spec off` was not checked in this run.

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

Each cell is median (min-max) over the 3 measured reps; with n=3 that is all three reps (the minimum, the median and the maximum), the order of the reps is not kept. The 2-stream cell is one aggregate measurement per arm. All four arms: `cached_tokens_max` 0, i.e. no prefix-cache hit in any speed call, on vLLM either (nonce-first prompts defeat its prefix cache). Prompt sizes are the real token counts the server reported.

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

| benchmark | A vLLM | B-default intarsiaml | compare |
|---|---|---|---|
| GSM8K flexible, n=100, pinned prompt | 0.590 (`20261112-183050-adhoc-ab35-A-gsm8k`) | 0.660 (`20261112-184255-adhoc-ab35-B-default-gsm8k`) | +0.070, 2 SE = 0.137: noise |
| IFEval, n=25 | 0.920 (`20261112-183155-adhoc-ab35-A-ifeval`) | 0.920 (`20261112-184348-adhoc-ab35-B-default-ifeval`) | +0.000, 2 SE = 0.153: noise |

What this does and does not show:

- `--nonce` puts a fresh uuid4 at the front of every prompt (`/opt/code/deepeval/src/localeval/client.py:206-211`), so the two arms never saw the same prompts. Question-by-question agreement (85 of 100 same correctness, 57 of 100 identical reply text) compares different prompts and says nothing about how the engines' numerics differ; no reason for the differences is claimed.
- With thinking off the pinned GSM8K template mostly yields a bare number, so n=100 here mostly measures reply style: bare answers were right 35 of 76 (A) and 35 of 69 (B), worked answers 24 of 24 (A) and 31 of 31 (B). The 0.59 vs 0.66 gap is the count of worked answers (24 vs 31), not an accuracy difference.
- IFEval at n=25 has 2 SE of +-0.153, so only a large gap would show.
- Neither benchmark shows that the two engines are equivalent; they show no gap large enough to see at these n, in this reply-style regime.
- **MTP output equality with `--spec off` was not checked in this run** (B-specoff ran speed only, with no id comparison). That MTP actually ran is shown by the server counters: 5469 verify steps, 7563 of 10932 draft tokens accepted, 0 cost-gate fall-backs (`raw/serve-B-default.metrics`). The lane docs report greedy-id equality with `--spec off`; this run did not re-test it.
- Two exploratory A-only GSM8K runs (not on B, not in the tables): `--structured prompt` without the pinned template 0.550 with 12-token answers (`20261112-183420-adhoc-ab35-A-gsm8k-cot`) and `--structured none` 0.580 (`20261112-183457-adhoc-ab35-A-gsm8k-none`). Both were `--no-unim`; their summaries are in `raw/`.
- Short-prompt TTFT in the quality runs (about 500-token GSM8K prompts, about 125-token IFEval prompts, median of the calls): intarsiaml is **15-20 % slower**, 0.156 vs 0.136 s (GSM8K) and 0.106 vs 0.088 s (IFEval). The speed sweep's 128/1k cells show the same sign at 0.01-0.02 s. This is the one metric where vLLM is ahead outside the 10 % line, and it is a per-request overhead of 20-30 ms.
- Thinking-on, long-context and sampled (temperature above 0) quality were not run.

## Fairness notes

- **KV cache dtype**: vLLM fp8 e4m3 (8 GiB pool); intarsiaml bf16 (the checkpoint declares FP8 KV scales; intarsiaml does not use them and says so in its log). bf16 KV reads twice the bytes per token and may be part of why its 16k decode gain shrinks (a hypothesis, not tested: only one layer in four of this hybrid model has a KV cache); fp8 KV is an opt-in for intarsiaml (`--kv-dtype fp8`, `docs/results/2026-10-29-fp8kv-opt-in.md`) and was not tried here.
- **Speculation**: A none; B-default MTP k2, which drafts only while one sequence decodes. At 2 streams B-default is plain batched decode, so the 2-stream figure says nothing about MTP.
- **Prefill scheduling**: vLLM chunked prefill with `--max-num-batched-tokens 8192`; intarsiaml 2048-row chunks (macro pass up to 8192 rows when alone).
- **Context and memory**: vLLM serves 131072 tokens in 38 GB; B was run with `--max-seq 20480 --mem-budget 40` and was alone on the GPU (vLLM stopped). B at 131072 context, and B beside the other roles, were not measured.
- **Prefix caching (vLLM's main production advantage for multi-turn traffic)**: the harness's nonce-first prompts defeat vLLM's prefix cache, and `cached_tokens_max` is 0 in every speed run, so this A/B measures the uncached case only. vLLM runs with `--enable-prefix-caching`; intarsiaml's prefix store is off by default, and per the recipe (`recipes/qwen3_5_moe.toml`, `[tuning.cuda]`) a configured prefix store keeps plain decode, i.e. it disables the MTP default. Multi-turn traffic on vLLM would see much lower TTFT than the table; what intarsiaml would give with the store, and at what decode speed, is unmeasured.
- **Time order**: A before, B in the window, A again after: A drifted at most 2 % (decode 76.8/77.6/76.5/73.6 vs 75.9/77.5/76.5/73.5), so the box was stable. The first A run followed 6 days of vLLM uptime, A-post a fresh container.
- **Noise**: decode about 5 %, long-prompt prefill about 3 %, mid-length prefill up to 39 % (unanimis 6a7efe40). **Noisy B-default cells that localeval flagged and that were not rerun**: decode at 1k, reps 104.2 / 105.1 / 130.0 (24.5 % spread), and at 16k, 83.2 / 88.2 / 95.6 (14.1 % spread). The medians are quoted; the 1k median is not far from its low rep but the 130 outlier is unexplained. Prefill and TTFT under about 1k uncached tokens are request overhead.
- **Warmup**: the speed sweep makes two discarded 1-token calls up front (warmup and token calibration, `/opt/code/deepeval/src/localeval/speed.py:76-78`); it does not warm each prompt size separately beyond that.
- **Not the same as production traffic**: the fleetq quiet slot stops other fleetq jobs, not missioncntrl requests; every localeval activity check on A read idle.
- **Sampled requests**: this run sent temperature 0. intarsiaml's MTP also runs for sampled requests by default on CUDA (`INTARSIA_SPEC_SAMPLED` unset = `match`, the server log says so), and a request that omits temperature gets the checkpoint's `generation_config` defaults (temperature 1, top_p 0.95, top_k 20) on both engines. Draft acceptance and the MTP gain at those settings were not measured.
- **Different frontends**: parsers differ (production vLLM: `qwen3_xml` tool parser and `qwen3` reasoning parser; intarsiaml recipe: `hermes` and think-tags). No tool-calling or reasoning-parser equivalence test was run.

## Verdict per metric

Verdict words follow `localeval compare` (`raw/compare-*.txt`): a speed delta under 10 % is noise, a quality delta inside 2 SE is noise. compare does not read the older 2-stream cell or TTFT, so those are judged by the same 10 % rule by hand.

**Headline: decode is faster in the measured slice; cutover is not settled.** The slice is greedy (temperature 0), no prefix cache hits, single stream, prompts up to 16k, short 256-token outputs.

| metric | where it lands | size and limits |
|---|---|---|
| Decode, single stream, as shipped (MTP k2) | intarsiaml faster (compare: faster against A and A-post), in the slice above | +40 % / +36 % / +41 % / +20 % at 128 / 1k / 4k / 16k (107 / 105 / 108 / 88 vs 77 / 78 / 77 / 74 tok/s). The MTP part of it shrinks with context: B-default over B-specoff is 1.24x at 128 and 1.17x at 16k, and the recipe cites about 1.13x at 16k and 1.05x at 34k, while production serves up to 131072. The 1k and 16k B-default cells are noisy (see Noise) |
| Decode, single stream, no speculation (engine only) | ahead at short prompts, parity beyond | +13 % at 128, +11 % at 1k (faster), +9 % at 4k and +2 % at 16k (noise): the lead shrinks with context |
| Prefill | 4k ahead, rest parity | 4k marginal prefill +12 to +17 % (faster against both A runs), total +11 % vs A but +7 to +8 % vs A-post (noise); 16k +4 to +6 % noise; 1k and 128 overhead-dominated, no difference. Uncached prompts only |
| TTFT | parity on the sweep; intarsiaml slower on short quality prompts | sweep 4k 0.59 vs 0.63-0.65 s and 16k 2.51 vs 2.61-2.66 s (4-9 % lower, under the 10 % line); vLLM 0.01-0.02 s lower at 128 and 1k; quality-run prompts 15-20 % slower on intarsiaml |
| 2-stream aggregate decode (1k prompts) | not firm | B-default +8 to +9 % (noise), B-specoff +11 to +12 % (just over the line); MTP is off at 2 streams; one aggregate measurement per arm, no reps |
| Quality (GSM8K n=100, IFEval n=25, thinking off) | no gap seen, no equivalence shown | see Quality above |

Reading: in this slice intarsiaml as shipped is not behind vLLM on decode, prefill or the 2-stream figure, and the 2.4-2.8x prefill gap of 2026-10-13 is gone. That does not settle a cutover.

### Unmeasured, by cutover risk

1. Multi-turn prefix-cached traffic (vLLM's prefix cache on; intarsiaml's prefix store off by default and disabling MTP when set).
2. Tool-calling and reasoning parsers (`qwen3_xml` / `qwen3` on vLLM, `hermes` / think-tags on intarsiaml).
3. Context beyond 16k, and memory at 131072 (B ran at 20480; bf16 KV vs fp8).
4. The MTP gain for sampled requests (temperature above 0).
5. Soak and stability.
6. More than 2 streams.

Also not run: thinking-on quality, `--kv-dtype fp8` on intarsiaml, MTP output equality with `--spec off` in this run.

## Raw results

- Run directories (summary.md, result.json): `docs/results/2026-11-12-ab35-intarsia-vs-vllm/raw/<run id>/`; the localeval originals are under `/opt/code/deepeval/results/<run id>/` and each run was logged to unanimis.
- A speed `20261112-182950-adhoc-ab35-A-speed` (unanimis 35803f53), A gsm8k `20261112-183050-adhoc-ab35-A-gsm8k` (c15562c7), A ifeval `20261112-183155-adhoc-ab35-A-ifeval` (94a87015), A-post speed `20261112-184939-adhoc-ab35-A-post-speed` (63ea5ec6).
- B-default speed `20261112-184207-adhoc-ab35-B-default-speed` (e3eeb9a6), gsm8k `20261112-184255-adhoc-ab35-B-default-gsm8k` (76c6a920), ifeval `20261112-184348-adhoc-ab35-B-default-ifeval` (a08c8612); B-specoff speed `20261112-184521-adhoc-ab35-B-specoff-speed` (b1fa51fb).
- `localeval compare` outputs: `raw/compare-A-vs-Bdefault.txt`, `compare-A-vs-Bspecoff.txt`, `compare-Apost-vs-Bdefault.txt`, `compare-Apost-vs-Bspecoff.txt`, `compare-A-vs-Apost.txt` (A drift), `compare-gsm8k.txt`, `compare-ifeval.txt`.
- Logs: `raw/window.log`, `raw/serve-B-default.{log,metrics}`, `raw/serve-B-specoff.{log,metrics}`, `raw/*.out` (localeval stdout), `raw/aborted-1/` (window 1). Driver scripts as run: `scripts/ab35/` (`build.sh`, `smoke.sh`, `window.sh`, `le.sh`, `memwatch.sh`; they were run from `.ab35/` in the worktree, paths rewritten here; `memwatch.sh` was copied by hand to `~/intarsia-ab35/memwatch.sh` on spark2 before the smoke and the windows).

## Decisions

None to flip. The user's go-ahead (top of this file) is the only decision; one row in `docs/decisions/LOG.md` records it.
