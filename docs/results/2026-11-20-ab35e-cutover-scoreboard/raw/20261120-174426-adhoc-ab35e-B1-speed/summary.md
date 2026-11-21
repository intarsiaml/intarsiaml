# localeval: adhoc — 2026-11-20T17:44

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: ab35e-B1-speed
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/ab35e/docs/results/2026-11-20-ab35e-cutover-scoreboard/raw/recipes/B.txt --sizes 4096,16384,32768,65536 --gen 256 --reps 3 --label ab35e-B1-speed`
- result file: `results/20261120-174426-adhoc-ab35e-B1-speed/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/ab35e/docs/results/2026-11-20-ab35e-cutover-scoreboard/raw/recipes/B.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4089 | 0.69 | 5940 (5716–6099) | - | 121.3 (119.9–124.7) | 176.2 (175.0–225.8) | - | - | 256 |
| 16653 | 2.84 | 5858 (5772–5903) | 5830 | 116.7 (111.7–116.8) | 174.7 (163.0–195.9) | - | - | 256 |
| 33880 | 6.32 | 5360 (5304–5363) | 4953 | 107.7 (99.9–112.0) | 145.3 (143.4–173.3) | - | - | 256 |
| 68907 | 15.81 | 4360 (4353–4363) | 3693 | 93.0 (88.1–94.5) | 124.9 (122.6–124.9) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 11.3% at 32768 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
