# localeval: adhoc — 2026-11-20T16:14

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: quality; label: style7-nostore-c128-fp8kv
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/recipes/nostore-c128-fp8kv.txt --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k --label style7-nostore-c128-fp8kv`
- result file: `results/20261120-161440-adhoc-style7-nostore-c128-fp8kv/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/recipes/nostore-c128-fp8kv.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --no-supervisor --max-batch 2 --prefill-chunk 128 --kv-dtype fp8_e4m3
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.710 | 200 | 0.17 | - | 159.1 | 200 | 0 | 121.3 |

- gsm8k scores: `gsm8k_flexible` 0.710 (primary); `gsm8k_bare_number` 0.350
- gsm8k: 200 replies kept; 0 truncated (score 0); thinking verified off; mean output 62.02 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
