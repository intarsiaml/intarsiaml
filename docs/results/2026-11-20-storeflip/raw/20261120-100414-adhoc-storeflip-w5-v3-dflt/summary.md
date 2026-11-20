# localeval: adhoc — 2026-11-20T10:04

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeflip-w5-v3-dflt
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w5-v3-dflt.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeflip-w5-v3-dflt`
- result file: `results/20261120-100414-adhoc-storeflip-w5-v3-dflt/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w5-v3-dflt.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on   (env: none)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4090 | 0.67 | 6113 (6112–6164) | - | 121.6 (118.1–125.1) | 172.8 (171.2–234.9) | - | - | 256 |
| 16654 | 2.61 | 6372 (6367–6393) | 6463 | 98.1 (97.7–101.1) | 143.8 (128.8–154.9) | - | - | 256 |
| 33902 | 5.83 | 5813 (5813–5838) | 5358 | 80.5 (80.4–82.1) | 115.9 (112.2–126.4) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
