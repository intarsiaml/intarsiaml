# localeval: adhoc — 2026-10-15T08:03

- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
- thinking: off; run kind: speed; label: p3noise-p4-cand-c2048
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096,4108 --gen 256 --reps 3 --thinking off --label p3noise-p4-cand-c2048`
- result file: `results/20261015-080311-adhoc-p3noise-p4-cand-c2048/result.json`

## Speed sweep

served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 136 | 0.05 | 2419 (2418–2424) * | - | 36.7 (36.6–36.9) | 37.0 (36.8–37.0) | - | - | 256 |
| 991 | 0.25 | 3912 (3887–3914) * | 4296 | 36.6 (36.3–36.6) | 36.8 (36.7–36.8) | - | - | 256 |
| 4088 | 1.09 | 3756 (3725–3815) | 3713 | 35.2 (35.1–35.3) | 35.4 (35.3–35.6) | - | - | 256 |
| 4110 | 1.28 | 3215 (3203–3228) | 116 | 35.5 (35.2–35.6) | 35.6 (35.4–35.6) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
