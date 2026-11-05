# localeval: adhoc — 2026-11-05T16:18

- model: `dsv41-intarsia` on ? (?), url `http://100.120.88.104:8899/v1`
- thinking: off; run kind: speed; label: dsv41validate-speed-skip0_1b
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a43b846
- command: `/opt/code/deepeval/.venv/bin/localeval speed --sizes 1024,63700 --gen 8 --reps 2 --label dsv41validate-speed-skip0_1b --url http://100.120.88.104:8899/v1 --model dsv41-intarsia --thinking off --recipe intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB`
- result file: `results/20261105-161841-adhoc-dsv41validate-speed-skip0_1b/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB
````

## Speed sweep

served `dsv41-intarsia`, fingerprint `None`, ctx 66000, 2 reps, 8 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1030 | 0.74 | 1389 (1353–1426) | - | 43.2 (37.3–49.1) | - | - | - | 8 |
| 65691 | 39.85 | 1648 (1643–1654) | 1653 | 44.8 (37.0–52.6) | - | - | - | 8 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 27.4% at 1024 tokens: noisy; rerun before comparing
- decode spread 35.0% at 63700 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
