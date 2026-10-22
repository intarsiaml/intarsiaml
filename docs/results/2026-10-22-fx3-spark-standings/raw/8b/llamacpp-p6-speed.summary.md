# localeval: adhoc — 2026-10-22T06:01

- model: `llamacpp` on ? (?), url `http://spark-fa00:8400/v1`
- thinking: off; run kind: speed; label: fx3-8b-llamacpp-p6-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8400/v1 --model llamacpp --thinking off --label fx3-8b-llamacpp-p6-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-060127-adhoc-fx3-8b-llamacpp-p6-speed/result.json`

## Speed sweep

served `llamacpp`, fingerprint `b1-d88040c`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 134 | 0.11 | 1141 (1116–1148) * | - | 15.4 (15.4–15.4) | 15.5 (15.5–15.5) | 1285 | 15.4 | 128 |
| 992 | 0.34 | 2910 (2877–2911) * | 3796 | 15.3 (15.3–15.3) | 15.3 (15.3–15.3) | 3046 | 15.3 | 128 |
| 4087 | 1.37 | 2987 (2956–2988) | 3014 | 14.9 (14.8–14.9) | 14.9 (14.9–14.9) | 3045 | 14.8 | 128 |
| 16654 | 6.80 | 2449 (2442–2450) | 2314 | 13.3 (13.3–13.3) | 13.3 (13.3–13.4) | 2464 | 13.3 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
