# localeval: adhoc — 2026-11-20T00:41

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: storettft-w4-v1-off
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/storettft/docs/results/2026-11-20-storettft/raw/recipes/w4-v1-off.txt --sizes 4096,16384,32768 --gen 256 --reps 3 --label storettft-w4-v1-off`
- result file: `results/20261120-004152-adhoc-storettft-w4-v1-off/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/storettft/docs/results/2026-11-20-storettft/raw/recipes/w4-v1-off.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2   (env: none)
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4088 | 0.59 | 6954 (6820–7016) | - | 119.1 (116.6–126.0) | 186.3 (156.6–191.0) | - | - | 256 |
| 16632 | 2.52 | 6596 (6504–6607) | 6489 | 99.4 (95.1–101.1) | 134.0 (128.5–134.6) | - | - | 256 |
| 33856 | 5.68 | 5958 (5948–5962) | 5449 | 80.4 (79.4–87.0) | 111.7 (100.1–141.5) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
