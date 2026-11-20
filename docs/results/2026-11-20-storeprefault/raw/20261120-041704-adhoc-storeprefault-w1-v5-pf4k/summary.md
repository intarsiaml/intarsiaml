# localeval: adhoc — 2026-11-20T04:17

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeprefault-w1-v5-pf4k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w1-v5-pf4k.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeprefault-w1-v5-pf4k`
- result file: `results/20261120-041704-adhoc-storeprefault-w1-v5-pf4k/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w1-v5-pf4k.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on   (env: INTARSIA_PREFIX_STORE_MACRO=1 INTARSIA_PREFIX_STORE_PREFAULT=4k)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4091 | 0.67 | 6111 (6110–6119) | - | 121.6 (116.0–125.2) | 202.9 (194.4–234.3) | - | - | 256 |
| 16656 | 2.68 | 6224 (6186–6361) | 6261 | 97.3 (96.5–103.6) | 135.2 (134.4–148.8) | - | - | 256 |
| 33880 | 5.91 | 5733 (5694–5734) | 5326 | 85.1 (80.1–94.7) | 111.2 (106.7–143.9) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 17.2% at 32768 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
