# localeval: adhoc — 2026-11-20T10:15

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeflip-w7-v1-m0
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w7-v1-m0.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeflip-w7-v1-m0`
- result file: `results/20261120-101556-adhoc-storeflip-w7-v1-m0/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w7-v1-m0.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on   (env: INTARSIA_PREFIX_STORE_MACRO=0)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4089 | 0.67 | 6097 (6091–6119) | - | 119.7 (111.5–120.6) | 176.3 (161.6–184.9) | - | - | 256 |
| 16657 | 2.83 | 5881 (5855–5883) | 5813 | 98.0 (95.2–105.4) | 144.2 (132.1–145.8) | - | - | 256 |
| 33903 | 6.38 | 5312 (5283–5331) | 4858 | 87.5 (86.1–101.7) | 120.0 (119.7–139.3) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 10.4% at 16384 tokens: noisy; rerun before comparing
- decode spread 17.8% at 32768 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
