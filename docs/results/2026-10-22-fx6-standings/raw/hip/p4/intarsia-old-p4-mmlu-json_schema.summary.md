# localeval: adhoc — 2026-10-22T01:44

- model: `qwen36-35b-intarsia-hip` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: quality; label: fx6-intarsia-old-p4-mmlu-json_schema
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://tatooine:8090/v1 --model qwen36-35b-intarsia-hip --bench mmlu --limit 4 --tasks high_school_physics,virology --thinking off --structured json_schema --judge none --no-unim --label fx6-intarsia-old-p4-mmlu-json_schema`
- result file: `results/20261022-014439-adhoc-fx6-intarsia-old-p4-mmlu-json_schema/result.json`

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.750 | 8 | 0.12 | - | 159.9 | 16 | 0 | 31.7 |

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
