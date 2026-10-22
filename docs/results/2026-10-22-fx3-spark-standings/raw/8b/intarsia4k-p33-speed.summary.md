# localeval: adhoc — 2026-10-22T09:02

- model: `intarsia` on ? (?), url `http://spark-fa00:8190/v1`
- thinking: off; run kind: speed; label: fx3-8b-intarsia4k-p33-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8190/v1 --model intarsia --thinking off --label fx3-8b-intarsia4k-p33-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-090229-adhoc-fx3-8b-intarsia4k-p33-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.10 | 1057 (1033–1057) * | - | 15.3 (15.3–15.3) | 15.3 (15.3–15.3) | - | - | 128 |
| 990 | 0.22 | 4434 (4428–4456) * | 7308 | 15.2 (15.2–15.2) | 15.2 (15.2–15.3) | - | - | 128 |
| 4088 | 0.97 | 4203 (4200–4478) | 4136 | 14.7 (14.7–14.7) | 14.8 (14.8–14.8) | - | - | 128 |
| 16655 | 4.62 | 3608 (3600–3619) | 3449 | 13.2 (13.2–13.3) | 13.3 (13.2–13.3) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
