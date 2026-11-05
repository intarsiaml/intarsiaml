# localeval: adhoc — 2026-11-05T16:01

- model: `dsv41-intarsia` on ? (?), url `http://100.120.88.104:8899/v1`
- thinking: off; run kind: speed; label: dsv41validate-speed-def2
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a43b846
- command: `/opt/code/deepeval/.venv/bin/localeval speed --sizes 1024,65536 --gen 8 --reps 2 --label dsv41validate-speed-def2 --url http://100.120.88.104:8899/v1 --model dsv41-intarsia --thinking off --recipe intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB`
- result file: `results/20261105-160102-adhoc-dsv41validate-speed-def2/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB
````

## Speed sweep

served `dsv41-intarsia`, fingerprint `None`, ctx 66000, 2 reps, 8 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1027 | 0.71 | 1439 (1436–1443) | - | 37.4 (22.2–52.5) | - | - | - | 8 |
| None | - | - * | - | - | - | - | - | - |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 81.0% at 1024 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
