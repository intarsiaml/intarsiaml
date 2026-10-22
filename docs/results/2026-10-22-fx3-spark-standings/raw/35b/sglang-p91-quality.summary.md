# localeval: adhoc — 2026-10-22T06:28

- model: `sglang` on ? (?), url `http://spark-4e77:8200/v1`
- thinking: off; run kind: quality; label: fx3-35b-sglang-p91-quality
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-4e77:8200/v1 --model sglang --thinking off --label fx3-35b-sglang-p91-quality --structured none --bench mmlu,gsm8k --limit 25`
- result file: `results/20261022-062855-adhoc-fx3-35b-sglang-p91-quality/result.json`

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.833 | 1425 | 0.12 | 21867 | 75.2 | 1425 | 0 | 262.0 |
| gsm8k | 0.600 | 25 | 0.10 | - | 75.8 | 25 | 0 | 16.3 |

- gsm8k scores: `gsm8k_flexible` 0.600 (primary); `gsm8k_strict` 0.320

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
