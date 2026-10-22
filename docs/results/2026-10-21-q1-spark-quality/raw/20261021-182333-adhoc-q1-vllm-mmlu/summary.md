# localeval: adhoc — 2026-10-21T18:23

- model: `guidance` on ? (?), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: quality; label: q1-vllm-mmlu
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-4e77:8000/v1 --model guidance --thinking off --structured none --bench mmlu --limit 30 --label q1-vllm-mmlu`
- result file: `results/20261021-182333-adhoc-q1-vllm-mmlu/result.json`

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.835 | 1710 | 0.25 | 4681 | - | 1710 | 0 | 513.0 |

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
