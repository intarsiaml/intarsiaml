# localeval: adhoc — 2026-10-15T12:29

- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
- thinking: off; run kind: speed; label: p3-on
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label p3-on`
- result file: `results/20261015-122933-adhoc-p3-on/result.json`

## Speed sweep

served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.05 | 2100 (2075–2143) * | - | 37.6 (36.7–37.8) | 39.3 (39.0–42.8) | - | - | 256 |
| 990 | 0.25 | 3905 (3881–3906) * | 4346 | 38.6 (38.2–40.1) | 49.1 (44.4–54.0) | - | - | 256 |
| 4086 | 1.11 | 3665 (3145–3738) | 3596 | 39.2 (39.1–40.3) | 48.1 (46.5–54.9) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
