# localeval: adhoc — 2026-11-20T10:18

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeflip-w7-v2-mac4k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w7-v2-mac4k.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeflip-w7-v2-mac4k`
- result file: `results/20261120-101813-adhoc-storeflip-w7-v2-mac4k/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w7-v2-mac4k.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on   (env: INTARSIA_PREFILL_MACRO=4096)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4090 | 0.67 | 6105 (6080–6137) | - | 118.1 (117.0–129.0) | 183.1 (171.2–202.5) | - | - | 256 |
| 16655 | 2.69 | 6191 (6166–6194) | 6220 | 99.4 (98.2–101.3) | 136.2 (123.8–139.7) | - | - | 256 |
| 33880 | 5.99 | 5657 (5569–5665) | 5221 | 81.0 (81.0–82.5) | 113.7 (107.9–117.4) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 10.2% at 4096 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
