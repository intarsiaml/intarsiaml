# localeval: adhoc — 2026-10-27T09:54

- model: `lfm2.5-1.2b-instruct` on ? (?), url `http://127.0.0.1:8123/v1`
- thinking: off; run kind: quality; label: nab2-lfm2-metal-mmlu10
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://127.0.0.1:8123/v1 --model lfm2.5-1.2b-instruct --bench mmlu --limit 10 --thinking off --label nab2-lfm2-metal-mmlu10 --recipe intarsia serve --model LiquidAI/LFM2.5-1.2B-Instruct --backend metal --max-seq 6144 (nab2 branch lane-nab2/lfm2-shortconv)`
- result file: `results/20261027-095430-adhoc-nab2-lfm2-metal-mmlu10/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --model LiquidAI/LFM2.5-1.2B-Instruct --backend metal --max-seq 6144 (nab2 branch lane-nab2/lfm2-shortconv)
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.314 | 570 | 0.04 | - | - | 595 | 0 | 120.7 |
- mmlu: 570 replies kept; 1 truncated (score 0); thinking verified off; mean output 16.69 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Warnings

- server rejected ['response_format', 'stream_options']; fell back without them

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
