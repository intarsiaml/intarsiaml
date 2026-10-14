# localeval: adhoc — 2026-10-14T13:52

- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8479/v1`
- thinking: off; run kind: speed; label: intarsia-metal-bf16-b2fuse
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8479/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label intarsia-metal-bf16-b2fuse`
- result file: `results/20261014-135214-adhoc-intarsia-metal-bf16-b2fuse/result.json`

## Speed sweep

served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.38 | 282 (216–297) * | - | 33.8 (33.2–33.9) | 34.6 (34.5–34.7) | - | - | 256 |
| 992 | 3.48 | 283 (270–300) * | 284 | 33.5 (33.2–33.8) | 34.1 (34.0–34.1) | - | - | 256 |
| 4089 | 19.26 | 212 (211–213) | 196 | 30.8 (30.7–30.8) | 31.4 (31.4–31.5) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
