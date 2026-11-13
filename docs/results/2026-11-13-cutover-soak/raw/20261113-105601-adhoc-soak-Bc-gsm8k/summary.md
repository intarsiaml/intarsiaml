# localeval: adhoc — 2026-11-13T10:56

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: on; run kind: quality; label: soak-Bc-gsm8k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8191/v1 --model guidance --thinking on --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/soak/.soak-recipes/Bc.txt --bench gsm8k --limit 85 --max-tokens 8192 --pin-gsm8k --label soak-Bc-gsm8k`
- result file: `results/20261113-105601-adhoc-soak-Bc-gsm8k/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/soak/.soak-recipes/Bc.txt` (secrets redacted):

````
intarsia serve --backend cuda (HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a) --max-seq 131072 --mem-budget 40 --max-batch 2 --prefix-store on --profile-every 0; bf16 KV, no MTP, thinking on
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.988 | 85 | 0.20 | - | 85.3 | 85 | 0 | 2154.7 |

- gsm8k scores: `gsm8k_flexible` 0.988 (primary); `gsm8k_bare_number` 0.718
- gsm8k: 85 replies kept; 0 truncated (score 0); mean output 2133.13 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
