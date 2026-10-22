# localeval: adhoc — 2026-10-22T06:42

- model: `intarsia` on ? (?), url `http://spark-4e77:8190/v1`
- thinking: off; run kind: speed; label: fx3-35b-intarsia4k-p32-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8190/v1 --model intarsia --thinking off --label fx3-35b-intarsia4k-p32-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-064247-adhoc-fx3-35b-intarsia4k-p32-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 115 | 0.11 | 1027 (1020–1034) * | - | 83.6 (77.4–83.6) | 85.4 (83.8–92.3) | - | - | 128 |
| 992 | 0.23 | 4273 (4269–4309) * | 7248 | 83.1 (82.8–83.1) | 85.9 (83.8–94.9) | - | - | 128 |
| 4090 | 0.69 | 5957 (5644–5976) | 6824 | 80.3 (80.3–82.3) | 84.7 (81.0–87.8) | - | - | 128 |
| 16656 | 3.11 | 5354 (5352–5550) | 5180 | 71.6 (71.4–73.8) | 72.1 (72.1–81.0) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
