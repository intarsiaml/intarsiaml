# localeval: adhoc — 2026-10-18T18:06

- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
- thinking: off; run kind: speed; label: ctl-main-1
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label ctl-main-1`
- result file: `results/20261018-180648-adhoc-ctl-main-1/result.json`

## Speed sweep

served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.05 | 2115 (2095–2122) * | - | 36.0 (36.0–36.1) | 36.2 (36.2–36.5) | - | - | 256 |
| 990 | 0.25 | 3929 (3867–3965) * | 4402 | 35.7 (35.0–35.8) | 36.3 (35.8–36.4) | - | - | 256 |
| 4090 | 1.21 | 3392 (2464–3511) | 3250 | 34.3 (33.3–34.4) | 35.0 (34.3–35.1) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
