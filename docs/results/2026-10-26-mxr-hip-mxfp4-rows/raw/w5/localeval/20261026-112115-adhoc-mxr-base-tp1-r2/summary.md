# localeval: adhoc — 2026-10-26T11:21

- model: `intarsia` on ? (?), url `http://tatooine:8294/v1`
- thinking: off; run kind: speed; label: mxr-base-tp1-r2
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8294/v1 --model intarsia --thinking off --sizes 1024 --gen 512 --reps 1 --concurrency 4,8,16 --label mxr-base-tp1-r2 --no-unim --recipe intarsia serve --backend hip --tp 1 --max-seq 8192 --max-batch 16 (tree base: lane-mxr/hip-mxfp4-rows fc0291746 or origin/main)`
- result file: `results/20261026-112115-adhoc-mxr-base-tp1-r2/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --backend hip --tp 1 --max-seq 8192 --max-batch 16 (tree base: lane-mxr/hip-mxfp4-rows fc0291746 or origin/main)
````

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 4930, 1 reps, 512 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 992 | 0.54 | 1839 (1839–1839) * | - | 32.9 (32.9–32.9) | 33.1 (33.1–33.1) | - | - | 512 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 4 | 84.8 | 22.8 (22.1–23.7) | 1.55 |
| 8 | 114.0 | 15.8 (15.0–16.8) | 2.96 |
| 16 | 133.0 | 9.4 (8.9–10.4) | 5.96 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
