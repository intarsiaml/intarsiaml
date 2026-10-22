# localeval: adhoc — 2026-10-22T04:39

- model: `llamacpp` on ? (?), url `http://spark-fa00:8400/v1`
- thinking: off; run kind: speed; label: fx3-8b-llamacpp-p1-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8400/v1 --model llamacpp --thinking off --label fx3-8b-llamacpp-p1-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-043951-adhoc-fx3-8b-llamacpp-p1-speed/result.json`

## Speed sweep

served `llamacpp`, fingerprint `b1-d88040c`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 135 | 0.12 | 1109 (1093–1113) * | - | 15.6 (15.6–15.6) | 15.7 (15.7–15.7) | 1230 | 15.6 | 128 |
| 989 | 0.36 | 2734 (2720–2746) * | 3514 | 15.5 (15.5–15.5) | 15.5 (15.5–15.6) | 2853 | 15.5 | 128 |
| 4086 | 1.44 | 2842 (2840–2858) | 2878 | 15.0 (15.0–15.0) | 15.1 (15.1–15.1) | 2898 | 15.0 | 128 |
| 16631 | 6.99 | 2380 (2378–2380) | 2260 | 13.4 (13.4–13.4) | 13.5 (13.5–13.5) | 2399 | 13.4 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
