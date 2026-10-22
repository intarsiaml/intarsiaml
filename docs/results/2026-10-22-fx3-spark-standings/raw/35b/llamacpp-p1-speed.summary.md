# localeval: adhoc — 2026-10-22T05:13

- model: `llamacpp` on ? (?), url `http://spark-4e77:8400/v1`
- thinking: off; run kind: speed; label: fx3-35b-llamacpp-p1-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8400/v1 --model llamacpp --thinking off --label fx3-35b-llamacpp-p1-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-051334-adhoc-fx3-35b-llamacpp-p1-speed/result.json`

## Speed sweep

served `llamacpp`, fingerprint `b1-7170cad`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.17 | 655 (630–659) * | - | 55.4 (55.0–55.5) | 56.0 (55.8–56.1) | 715 | 55.4 | 128 |
| 992 | 0.48 | 2076 (2076–2122) * | 2872 | 55.1 (54.8–55.6) | 55.7 (55.6–56.7) | 2147 | 55.1 | 128 |
| 4088 | 1.71 | 2385 (2384–2401) | 2505 | 53.5 (53.3–53.6) | 54.1 (53.9–54.3) | 2419 | 53.5 | 128 |
| 16656 | 7.01 | 2377 (2370–2377) | 2374 | 50.7 (50.5–52.1) | 51.3 (51.1–52.5) | 2392 | 50.7 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
