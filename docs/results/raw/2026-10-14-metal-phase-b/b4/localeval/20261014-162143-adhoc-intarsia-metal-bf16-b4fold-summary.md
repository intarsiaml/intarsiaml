# localeval: adhoc — 2026-10-14T16:21

- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:18311/v1`
- thinking: off; run kind: speed; label: intarsia-metal-bf16-b4fold
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:18311/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label intarsia-metal-bf16-b4fold`
- result file: `results/20261014-162143-adhoc-intarsia-metal-bf16-b4fold/result.json`

## Speed sweep

served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.37 | 301 (299–301) * | - | 35.7 (35.6–35.7) | 35.8 (35.7–35.8) | - | - | 256 |
| 990 | 2.87 | 343 (343–343) * | 350 | 35.3 (35.3–35.3) | 35.5 (35.5–35.5) | - | - | 256 |
| 4086 | 17.84 | 229 (229–229) | 207 | 34.3 (34.3–34.3) | 34.4 (34.4–34.4) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
