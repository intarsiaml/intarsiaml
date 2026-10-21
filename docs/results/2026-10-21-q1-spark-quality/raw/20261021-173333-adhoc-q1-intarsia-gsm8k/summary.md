# localeval: adhoc — 2026-10-21T17:33

- model: `intarsia-35b` on ? (?), url `http://spark-4e77:8091/v1`
- thinking: off; run kind: quality; label: q1-intarsia-gsm8k
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-4e77:8091/v1 --model intarsia-35b --thinking off --structured none --bench gsm8k --limit 1319 --label q1-intarsia-gsm8k`
- result file: `results/20261021-173333-adhoc-q1-intarsia-gsm8k/result.json`

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.240 | 1319 | 0.17 | - | 83.4 | 1319 | 0 | 1474.3 |

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
