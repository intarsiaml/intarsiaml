# localeval: adhoc — 2026-11-20T10:20

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeflip-w7-v3-mac4k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w7-v3-mac4k.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeflip-w7-v3-mac4k`
- result file: `results/20261120-102039-adhoc-storeflip-w7-v3-mac4k/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeflip/docs/results/2026-11-20-storeflip/raw/recipes/w7-v3-mac4k.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on   (env: INTARSIA_PREFILL_MACRO=4096)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4092 | 0.67 | 6095 (6055–6130) | - | 120.7 (115.7–125.1) | 242.5 (185.1–250.4) | - | - | 256 |
| 16655 | 2.70 | 6159 (6155–6177) | 6180 | 105.9 (101.9–109.2) | 154.9 (153.7–185.4) | - | - | 256 |
| 33903 | 6.03 | 5625 (5618–5650) | 5190 | 80.2 (79.0–83.3) | 112.7 (108.6–124.9) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
