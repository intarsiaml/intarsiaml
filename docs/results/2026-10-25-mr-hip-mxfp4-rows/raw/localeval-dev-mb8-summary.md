# localeval: adhoc — 2026-10-25T06:30

- model: `intarsia` on ? (?), url `http://tatooine:8294/v1`
- thinking: default; run kind: speed; label: mr-dev-mb8
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a21ef33
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8294/v1 --model intarsia --sizes 1024 --gen 128 --reps 3 --concurrency 4,8,16 --label mr-dev-mb8 --no-unim --recipe intarsia serve --backend hip --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 8192 --max-batch 8 tree=dev`
- result file: `results/20261025-063028-adhoc-mr-dev-mb8/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --backend hip --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 8192 --max-batch 8 tree=dev
````

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 8192, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 987 | 1.01 | 973 (956–974) * | - | 32.8 (32.8–32.9) | 33.2 (33.0–34.2) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 4 | 51.8 | 18.4 (15.1–23.6) | 2.74 |
| 8 | 61.5 | 11.6 (8.8–17.6) | 5.11 |
| 16 | 62.9 | 8.9 (8.7–17.6) | 13.00 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
