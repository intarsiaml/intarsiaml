# localeval: adhoc — 2026-11-20T15:30

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: off; run kind: quality; label: style7-vllm
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8000/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/recipes/vllm.txt --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k --label style7-vllm`
- result file: `results/20261120-153022-adhoc-style7-vllm/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/recipes/vllm.txt` (secrets redacted):

````
production vLLM (retro, spark2), request traffic only, one request at a time
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.600 | 200 | 0.14 | - | 79.2 | 200 | 0 | 177.8 |

- gsm8k scores: `gsm8k_flexible` 0.600 (primary); `gsm8k_bare_number` 0.350
- gsm8k: 200 replies kept; 1 truncated (score 0); thinking verified off; mean output 46.45 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
