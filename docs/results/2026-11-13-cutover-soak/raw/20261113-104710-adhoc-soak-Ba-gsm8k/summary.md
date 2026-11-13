# localeval: adhoc — 2026-11-13T10:47

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: quality; label: soak-Ba-gsm8k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/soak/.soak-recipes/Ba.txt --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k --label soak-Ba-gsm8k`
- result file: `results/20261113-104710-adhoc-soak-Ba-gsm8k/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/soak/.soak-recipes/Ba.txt` (secrets redacted):

````
intarsia serve --backend cuda (HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a) --max-seq 131072 --mem-budget 40 --max-batch 2 --prefix-store on --profile-every 0; bf16 KV, no MTP (the store refuses drafting), prefill chunk 2048, tool parser hermes
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.715 | 200 | 0.20 | - | 86.6 | 200 | 0 | 211.3 |

- gsm8k scores: `gsm8k_flexible` 0.715 (primary); `gsm8k_bare_number` 0.340
- gsm8k: 200 replies kept; 2 truncated (score 0); thinking verified off; mean output 72.22 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
