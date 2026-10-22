# localeval: adhoc — 2026-10-22T01:57

- model: `qwen3.6-35b-a3b.mq4r` on ? (?), url `http://tatooine:11435/v1`
- thinking: off; run kind: speed; label: fx6-hipfire-p7
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:11435/v1 --model qwen3.6-35b-a3b.mq4r --sizes 128,1024 --gen 128 --reps 3 --thinking off --ignore-busy --label fx6-hipfire-p7`
- result file: `results/20261022-015734-adhoc-fx6-hipfire-p7/result.json`

## Speed sweep

served `qwen3.6-35b-a3b.mq4r`, fingerprint `None`, ctx ?, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.09 | 1335 (1321–1405) * | - | 145.5 (144.8–146.9) | 147.5 (146.5–152.5) | - | - | 128 |
| 991 | 0.36 | 2764 (2762–2785) * | 3212 | 145.1 (145.0–145.2) | 147.1 (147.0–149.6) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
