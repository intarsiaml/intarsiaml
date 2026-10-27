# localeval: adhoc — 2026-10-27T05:57

- model: `qwen3.5-4b` on ? (?), url `http://127.0.0.1:8123/v1`
- thinking: off; run kind: quality; label: nmd-qwen35-4b-metal
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://127.0.0.1:8123/v1 --model qwen3.5-4b --label nmd-qwen35-4b-metal --thinking off --bench mmlu --limit 10 --judge none --recipe intarsia serve --model Qwen3.5-4B --backend metal --port 8123 --served-model-name qwen3.5-4b --max-seq 20480 --profile-every 0 (build lane-nmd/new-model-drill, a_log/norm bf16 cast)`
- result file: `results/20261027-055731-adhoc-nmd-qwen35-4b-metal/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --model Qwen3.5-4B --backend metal --port 8123 --served-model-name qwen3.5-4b --max-seq 20480 --profile-every 0 (build lane-nmd/new-model-drill, a_log/norm bf16 cast)
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.751 | 570 | 0.11 | 4525 | 60.5 | 597 | 0 | 2050.6 |
- mmlu: 570 replies kept; 0 truncated (score 0); thinking verified off; mean output 153.12 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
