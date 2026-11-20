# localeval: adhoc — 2026-11-20T05:17

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storeprefault-w6-v4-smacro
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w6-v4-smacro.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storeprefault-w6-v4-smacro`
- result file: `results/20261120-051746-adhoc-storeprefault-w6-v4-smacro/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storeprefault/docs/results/2026-11-20-storeprefault/raw/recipes/w6-v4-smacro.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on,host_mb=4096   (env: INTARSIA_PREFIX_STORE_MACRO=1)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4086 | 0.68 | 6025 (5839–6087) | - | 122.1 (117.7–129.1) | 182.1 (179.5–185.2) | - | - | 256 |
| 16656 | 2.63 | 6338 (6324–6353) | 6446 | 101.2 (96.4–107.7) | 140.0 (138.5–143.5) | - | - | 256 |
| 33878 | 5.87 | 5770 (5748–5832) | 5310 | 87.6 (79.7–93.2) | 114.7 (111.3–154.0) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 11.2% at 16384 tokens: noisy; rerun before comparing
- decode spread 15.5% at 32768 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
