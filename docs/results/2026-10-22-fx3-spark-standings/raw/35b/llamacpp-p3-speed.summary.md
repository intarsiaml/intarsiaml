# localeval: adhoc — 2026-10-22T05:32

- model: `llamacpp` on ? (?), url `http://spark-4e77:8400/v1`
- thinking: off; run kind: speed; label: fx3-35b-llamacpp-p3-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8400/v1 --model llamacpp --thinking off --label fx3-35b-llamacpp-p3-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-053201-adhoc-fx3-35b-llamacpp-p3-speed/result.json`

## Speed sweep

served `llamacpp`, fingerprint `b1-7170cad`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.18 | 656 (623–666) * | - | 54.8 (54.8–55.6) | 55.5 (55.4–56.6) | 704 | 54.8 | 128 |
| 992 | 0.48 | 2079 (2063–2104) * | 2946 | 54.5 (54.5–54.8) | 55.3 (55.1–55.7) | 2153 | 54.5 | 128 |
| 4090 | 1.72 | 2383 (2370–2387) | 2498 | 54.0 (53.8–54.6) | 55.3 (55.3–55.5) | 2418 | 54.0 | 128 |
| 16678 | 7.06 | 2361 (2359–2367) | 2354 | 51.4 (51.4–51.5) | 52.2 (51.9–52.2) | 2380 | 51.4 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
