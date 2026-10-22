# localeval: adhoc — 2026-10-22T09:26

- model: `intarsia` on ? (?), url `http://spark-fa00:8190/v1`
- thinking: off; run kind: speed; label: fx3-8b-intarsia4k-p34-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8190/v1 --model intarsia --thinking off --label fx3-8b-intarsia4k-p34-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-092639-adhoc-fx3-8b-intarsia4k-p34-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.21 | 523 (468–650) * | - | 15.2 (15.2–15.3) | 15.7 (15.3–16.1) | - | - | 128 |
| 989 | 0.28 | 3559 (2794–3846) * | 12696 | 15.0 (9.3–15.1) | 15.7 (15.6–743.9) | - | - | 128 |
| 4086 | 1.05 | 3892 (3857–4078) | 4006 | 14.6 (14.3–14.8) | 16.8 (16.8–18.4) | - | - | 128 |
| 16677 | 4.68 | 3564 (3549–3590) | 3470 | 13.2 (13.1–13.2) | 16.2 (13.2–16.3) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 38.6% at 1024 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
