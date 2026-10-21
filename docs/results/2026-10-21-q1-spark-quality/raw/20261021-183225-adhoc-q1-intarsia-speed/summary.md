# localeval: adhoc — 2026-10-21T18:32

- model: `intarsia-35b` on ? (?), url `http://spark-4e77:8091/v1`
- thinking: off; run kind: speed; label: q1-intarsia-speed
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8091/v1 --model intarsia-35b --thinking off --sizes 1024,4096 --gen 128 --reps 3 --label q1-intarsia-speed`
- result file: `results/20261021-183225-adhoc-q1-intarsia-speed/result.json`

## Speed sweep

served `intarsia-35b`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 992 | 0.25 | 3928 (2836–4158) * | - | 75.7 (74.8–84.1) | 131.4 (100.8–215.9) | - | - | 128 |
| 4089 | 0.84 | 4870 (4455–5477) | 5267 | 81.9 (79.0–83.7) | 102.4 (101.9–108.2) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 12.3% at 1024 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
