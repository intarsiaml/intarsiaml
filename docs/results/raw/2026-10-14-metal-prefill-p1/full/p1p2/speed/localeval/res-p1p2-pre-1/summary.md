# localeval: adhoc — 2026-10-14T22:02

- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8131/v1`
- thinking: off; run kind: speed; label: p1p2-pre-1
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8131/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label p1p2-pre-1`
- result file: `results/20261014-220259-adhoc-p1p2-pre-1/result.json`

## Speed sweep

served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 136 | 0.42 | 309 (302–310) * | - | 36.4 (36.4–36.5) | 36.6 (36.5–36.6) | - | - | 256 |
| 991 | 2.65 | 372 (372–388) * | 384 | 35.9 (35.8–36.0) | 36.2 (36.1–36.3) | - | - | 256 |
| 4087 | 11.17 | 366 (364–368) | 363 | 35.1 (35.0–35.2) | 35.2 (35.1–35.3) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
