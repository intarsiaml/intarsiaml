# localeval: adhoc — 2026-11-20T00:11

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storettft-w2-v10-off
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storettft/docs/results/2026-11-20-storettft/raw/recipes/w2-v10-off.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storettft-w2-v10-off`
- result file: `results/20261120-001134-adhoc-storettft-w2-v10-off/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storettft/docs/results/2026-11-20-storettft/raw/recipes/w2-v10-off.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2   (env: none)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4090 | 0.59 | 6887 (6522–6997) | - | 121.3 (119.3–121.4) | 217.2 (198.0–229.7) | - | - | 256 |
| 16655 | 2.56 | 6502 (6434–6645) | 6385 | 112.6 (100.2–113.9) | 181.9 (151.1–188.8) | - | - | 256 |
| 33902 | 5.69 | 5958 (5951–5959) | 5514 | 82.0 (80.1–82.0) | 106.7 (104.6–109.0) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 12.1% at 16384 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
