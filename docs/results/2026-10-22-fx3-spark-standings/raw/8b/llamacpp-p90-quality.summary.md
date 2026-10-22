# localeval: adhoc — 2026-10-22T07:57

- model: `llamacpp` on ? (?), url `http://spark-fa00:8400/v1`
- thinking: off; run kind: quality; label: fx3-8b-llamacpp-p90-quality
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8400/v1 --model llamacpp --thinking off --label fx3-8b-llamacpp-p90-quality --structured none --bench mmlu --limit 25`
- result file: `results/20261022-075727-adhoc-fx3-8b-llamacpp-p90-quality/result.json`

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.676 | 1425 | 0.12 | 2921 | 15.1 | 1425 | 0 | 1684.1 |

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
