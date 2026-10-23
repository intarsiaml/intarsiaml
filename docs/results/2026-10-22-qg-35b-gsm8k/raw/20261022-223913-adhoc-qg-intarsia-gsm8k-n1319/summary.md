# localeval: adhoc — 2026-10-22T22:39

- model: `intarsia` on ? (?), url `http://spark-fa00:8190/v1`
- thinking: off; run kind: quality; label: qg-intarsia-gsm8k-n1319
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 98707d5
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8190/v1 --model intarsia --thinking off --structured none --bench gsm8k --limit 1319 --label qg-intarsia-gsm8k-n1319 --recipe intarsia serve --backend cuda --model nvidia/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 8190 --served-model-name intarsia --max-seq 20480 --mem-budget 30 --profile-every 0 --no-supervisor --no-ui (binary ~/intarsia-decfuse/fx3-main = origin/main 45dd97fd, --features cuda, bf16 KV; spark2, driver 580.178.04) --wait-idle 600`
- result file: `results/20261022-223913-adhoc-qg-intarsia-gsm8k-n1319/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --backend cuda --model nvidia/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 8190 --served-model-name intarsia --max-seq 20480 --mem-budget 30 --profile-every 0 --no-supervisor --no-ui (binary ~/intarsia-decfuse/fx3-main = origin/main 45dd97fd, --features cuda, bf16 KV; spark2, driver 580.178.04)
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.683 | 1319 | 0.30 | - | 48.0 | 1319 | 0 | 2571.2 |

- gsm8k scores: `gsm8k_flexible` 0.683 (primary); `gsm8k_strict` 0.240

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
