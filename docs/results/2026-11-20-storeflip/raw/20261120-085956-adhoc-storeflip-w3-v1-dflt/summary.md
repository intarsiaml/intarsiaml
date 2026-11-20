# localeval: adhoc — 2026-11-20T08:59

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeflip-w3-v1-dflt
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w3-v1-dflt.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeflip-w3-v1-dflt`
- result file: `results/20261120-085956-adhoc-storeflip-w3-v1-dflt/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w3-v1-dflt.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on   (env: none)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4089 | 0.67 | 6098 (6047–6158) | - | 119.6 (117.8–123.3) | 170.8 (157.8–218.6) | - | - | 256 |
| 16654 | 2.61 | 6378 (6373–6390) | 6477 | 105.5 (101.5–106.1) | 149.6 (143.9–175.8) | - | - | 256 |
| 33881 | 5.81 | 5832 (5818–5839) | 5387 | 81.9 (81.8–93.4) | 107.9 (107.0–127.3) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 14.2% at 32768 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
