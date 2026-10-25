# localeval: adhoc — 2026-10-25T06:26

- model: `intarsia` on ? (?), url `http://tatooine:8294/v1`
- thinking: default; run kind: speed; label: mr-base-mb8
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a21ef33
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8294/v1 --model intarsia --sizes 1024 --gen 128 --reps 3 --concurrency 4,8,16 --label mr-base-mb8 --no-unim --recipe intarsia serve --backend hip --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 8192 --max-batch 8 tree=base`
- result file: `results/20261025-062627-adhoc-mr-base-mb8/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --backend hip --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 8192 --max-batch 8 tree=base
````

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 8192, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 986 | 1.03 | 959 (918–974) * | - | 32.8 (32.8–33.4) | 33.1 (33.0–35.5) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 4 | 45.5 | 15.3 (13.0–18.8) | 2.73 |
| 8 | 19.2 | 2.7 (2.6–3.0) | 5.14 |
| 16 | 18.6 | 2.5 (2.3–3.0) | 31.73 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
