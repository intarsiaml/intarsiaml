# localeval: adhoc — 2026-10-22T06:39

- model: `intarsia` on ? (?), url `http://spark-4e77:8190/v1`
- thinking: off; run kind: speed; label: fx3-35b-intarsia-p31-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8190/v1 --model intarsia --thinking off --label fx3-35b-intarsia-p31-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-063937-adhoc-fx3-35b-intarsia-p31-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.11 | 1027 (1022–1034) * | - | 84.5 (84.4–84.5) | 85.0 (84.9–85.9) | - | - | 128 |
| 992 | 0.23 | 4313 (4256–4352) * | 7441 | 83.9 (83.7–84.0) | 84.6 (84.5–85.0) | - | - | 128 |
| 4088 | 0.72 | 5686 (5663–5686) | 6331 | 81.0 (80.9–81.2) | 81.8 (81.6–81.9) | - | - | 128 |
| 16681 | 3.14 | 5316 (5274–5325) | 5208 | 73.4 (73.4–73.5) | 74.0 (74.0–75.2) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
