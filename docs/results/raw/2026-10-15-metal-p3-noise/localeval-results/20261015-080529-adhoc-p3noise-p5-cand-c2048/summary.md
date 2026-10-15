# localeval: adhoc — 2026-10-15T08:05

- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
- thinking: off; run kind: speed; label: p3noise-p5-cand-c2048
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096,4108 --gen 256 --reps 3 --thinking off --label p3noise-p5-cand-c2048`
- result file: `results/20261015-080529-adhoc-p3noise-p5-cand-c2048/result.json`

## Speed sweep

served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.05 | 2113 (2043–2134) * | - | 36.7 (36.3–36.8) | 37.0 (36.8–37.0) | - | - | 256 |
| 992 | 0.25 | 3879 (3603–3897) * | 4330 | 33.6 (32.0–35.5) | 36.0 (35.5–36.5) | - | - | 256 |
| 4088 | 1.10 | 3708 (2953–3753) | 3655 | 34.3 (34.3–34.9) | 35.2 (35.1–35.5) | - | - | 256 |
| 4110 | 1.26 | 3245 (3224–3256) | 134 | 35.1 (34.9–35.3) | 35.4 (35.4–35.4) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 10.4% at 1024 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
