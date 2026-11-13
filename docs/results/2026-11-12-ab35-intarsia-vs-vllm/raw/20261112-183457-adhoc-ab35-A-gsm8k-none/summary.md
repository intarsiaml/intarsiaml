# localeval: adhoc — 2026-11-12T18:34

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: off; run kind: quality; label: ab35-A-gsm8k-none
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/src/localeval/cli.py quality --url http://spark-fa00:8000/v1 --model guidance --thinking off --structured none --wait-idle 600 --recipe production vLLM (see ab35 doc) --bench gsm8k --limit 100 --max-tokens 1024 --nonce --label ab35-A-gsm8k-none --no-unim`
- result file: `results/20261112-183457-adhoc-ab35-A-gsm8k-none/result.json`

## Launch recipe

from inline command (secrets redacted):

````
production vLLM (see ab35 doc)
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.580 | 100 | 0.14 | - | 78.5 | 100 | 0 | 59.2 |

- gsm8k scores: `gsm8k_flexible` 0.580 (primary); `gsm8k_bare_number` 0.350
- gsm8k: 100 replies kept; 0 truncated (score 0); thinking verified off; mean output 35.0 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
