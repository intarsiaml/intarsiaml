# localeval: adhoc — 2026-11-20T04:07

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeprefault-w1-v1-off
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w1-v1-off.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeprefault-w1-v1-off`
- result file: `results/20261120-040724-adhoc-storeprefault-w1-v1-off/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w1-v1-off.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2   (env: none)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4087 | 0.59 | 6944 (6920–6948) | - | 117.4 (116.8–120.1) | 193.1 (191.9–195.5) | - | - | 256 |
| 16679 | 2.52 | 6610 (6603–6610) | 6508 | 102.2 (96.1–104.9) | 143.8 (128.7–159.5) | - | - | 256 |
| 33927 | 5.75 | 5902 (5901–5904) | 5348 | 79.0 (76.9–79.9) | 99.4 (95.9–104.9) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
