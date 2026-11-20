# localeval: adhoc — 2026-11-20T04:31

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeprefault-w4-v1-off
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w4-v1-off.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeprefault-w4-v1-off`
- result file: `results/20261120-043153-adhoc-storeprefault-w4-v1-off/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w4-v1-off.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2   (env: none)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4087 | 0.59 | 6936 (6893–6961) | - | 117.9 (117.4–122.5) | 163.7 (161.6–182.7) | - | - | 256 |
| 16656 | 2.51 | 6640 (6520–6665) | 6546 | 99.7 (96.9–102.2) | 142.6 (141.5–142.7) | - | - | 256 |
| 33881 | 5.70 | 5949 (5932–5953) | 5406 | 79.9 (79.7–85.1) | 106.9 (100.5–114.9) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
