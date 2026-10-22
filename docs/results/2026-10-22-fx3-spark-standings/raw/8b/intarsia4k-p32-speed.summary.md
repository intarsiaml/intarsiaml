# localeval: adhoc — 2026-10-22T08:45

- model: `intarsia` on ? (?), url `http://spark-fa00:8190/v1`
- thinking: off; run kind: speed; label: fx3-8b-intarsia4k-p32-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8190/v1 --model intarsia --thinking off --label fx3-8b-intarsia4k-p32-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-084532-adhoc-fx3-8b-intarsia4k-p32-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 135 | 0.10 | 1237 (1219–1241) * | - | 15.3 (15.3–15.3) | 15.4 (15.3–15.4) | - | - | 128 |
| 990 | 0.22 | 4463 (4362–4495) * | 7371 | 15.2 (15.2–15.2) | 15.2 (15.2–15.3) | - | - | 128 |
| 4090 | 0.96 | 4265 (4252–4278) | 4212 | 14.8 (14.8–14.8) | 14.8 (14.8–14.8) | - | - | 128 |
| 16652 | 4.63 | 3599 (3594–3599) | 3423 | 13.3 (13.3–13.3) | 13.3 (13.3–13.3) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
