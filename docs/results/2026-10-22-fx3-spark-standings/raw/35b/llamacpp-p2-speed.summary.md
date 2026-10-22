# localeval: adhoc — 2026-10-22T05:19

- model: `llamacpp` on ? (?), url `http://spark-4e77:8400/v1`
- thinking: off; run kind: speed; label: fx3-35b-llamacpp-p2-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8400/v1 --model llamacpp --thinking off --label fx3-35b-llamacpp-p2-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-051926-adhoc-fx3-35b-llamacpp-p2-speed/result.json`

## Speed sweep

served `llamacpp`, fingerprint `b1-7170cad`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 137 | 0.17 | 782 (779–825) * | - | 56.3 (55.5–56.5) | 57.6 (56.0–58.3) | 854 | 56.3 | 128 |
| 991 | 0.47 | 2117 (2103–2147) * | 2915 | 55.6 (55.3–55.7) | 56.5 (56.2–57.0) | 2184 | 55.6 | 128 |
| 4089 | 1.70 | 2409 (2406–2420) | 2521 | 55.5 (54.5–55.7) | 56.2 (55.3–56.4) | 2446 | 55.5 | 128 |
| 16656 | 6.97 | 2388 (2387–2390) | 2382 | 52.0 (52.0–52.1) | 53.0 (52.7–53.8) | 2403 | 52.0 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
