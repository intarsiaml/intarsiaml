# localeval: adhoc — 2026-11-19T22:01

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: ab35d-B2-speed
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/ab35d/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw/recipes/B.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label ab35d-B2-speed`
- result file: `results/20261119-220115-adhoc-ab35d-B2-speed/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/ab35d/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw/recipes/B.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4090 | 0.74 | 5512 (5298–5661) | - | 120.4 (119.5–123.9) | 164.8 (156.4–176.9) | - | - | 256 |
| 16656 | 3.27 | 5098 (5068–5745) | 4977 | 100.5 (98.3–100.6) | 141.6 (135.6–146.3) | - | - | 256 |
| 33882 | 6.42 | 5279 (5070–5335) | 5467 | 82.6 (79.5–87.4) | 123.2 (102.7–149.5) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
