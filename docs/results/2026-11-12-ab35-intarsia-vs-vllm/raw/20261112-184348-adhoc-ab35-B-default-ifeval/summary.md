# localeval: adhoc — 2026-11-12T18:43

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: quality; label: ab35-B-default-ifeval
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe ~/intarsia-ab35/bin/intarsia serve --backend cuda --model <HF snapshot 1355db6a> --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 20480 --mem-budget 40 --max-batch 2 --profile-every 0 (shipped CUDA defaults: MTP k2 via batched_mtp, bf16 KV, prefill chunk 2048, macro 8192) --bench ifeval --limit 25 --max-tokens 1024 --nonce --label ab35-B-default-ifeval`
- result file: `results/20261112-184348-adhoc-ab35-B-default-ifeval/result.json`

## Launch recipe

from inline command (secrets redacted):

````
~/intarsia-ab35/bin/intarsia serve --backend cuda --model <HF snapshot 1355db6a> --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 20480 --mem-budget 40 --max-batch 2 --profile-every 0 (shipped CUDA defaults: MTP k2 via batched_mtp, bf16 KV, prefill chunk 2048, macro 8192)
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ifeval | 0.920 | 25 | 0.11 | - | 114.8 | 30 | 0 | 53.8 |
- ifeval: 25 replies kept; 0 truncated (score 0); thinking verified off; mean output 201.08 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
