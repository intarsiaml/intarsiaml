# localeval: adhoc — 2026-11-19T22:49

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: ab35d-ctl-N1-speed
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/ab35d/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw/recipes/N-w5.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label ab35d-ctl-N1-speed`
- result file: `results/20261119-224942-adhoc-ab35d-ctl-N1-speed/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/ab35d/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw/recipes/N-w5.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4090 | 0.59 | 6909 (6284–6966) | - | 120.1 (118.1–121.0) | 200.8 (199.7–206.3) | - | - | 256 |
| 16657 | 2.52 | 6603 (6425–6605) | 6511 | 105.2 (99.1–106.0) | 180.8 (173.5–181.9) | - | - | 256 |
| 33879 | 5.77 | 5874 (5871–5942) | 5307 | 81.7 (80.4–82.3) | 114.9 (110.6–116.8) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
