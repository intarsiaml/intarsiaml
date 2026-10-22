# localeval: adhoc — 2026-10-22T01:54

- model: `qwen36-35b-intarsia-hip` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: fx6-intarsia-p6
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen36-35b-intarsia-hip --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-intarsia-p6`
- result file: `results/20261022-015445-adhoc-fx6-intarsia-p6/result.json`

## Speed sweep

served `qwen36-35b-intarsia-hip`, fingerprint `None`, ctx 17408, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.08 | 1375 (1339–1393) * | - | 159.7 (159.3–160.0) | 161.7 (161.3–168.0) | - | - | 128 |
| 992 | 0.17 | 5692 (5625–5797) * | 9756 | 159.3 (159.2–159.5) | 160.9 (160.7–161.0) | - | - | 128 |
| 4089 | 0.45 | 9120 (9044–9121) | 11303 | 155.4 (155.4–155.6) | 158.8 (157.1–160.4) | - | - | 128 |
| 16655 | 2.16 | 7725 (7721–7739) | 7357 | 141.6 (141.4–141.7) | 143.3 (143.2–151.8) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
