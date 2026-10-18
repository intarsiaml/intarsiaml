# localeval: adhoc — 2026-10-18T17:49

- model: `/Users/josef/models/Qwen3-8B` on ? (?), url `http://127.0.0.1:8092/v1`
- thinking: off; run kind: speed; label: retry-mlx-lm
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8092/v1 --model /Users/josef/models/Qwen3-8B --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label retry-mlx-lm`
- result file: `results/20261018-174918-adhoc-retry-mlx-lm/result.json`

## Speed sweep

served `/Users/josef/models/Qwen3-8B`, fingerprint `0.31.3-0.32.1-macOS-27.0-arm64-arm-64bit-applegpu_g17s`, ctx ?, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.22 | 486 (482–507) * | - | 33.7 (33.6–33.9) | 39.1 (38.5–39.2) | - | - | 256 |
| 990 | 0.43 | 2293 (2282–2339) * | 4237 | 33.2 (33.1–33.4) | 39.7 (37.6–40.9) | - | - | 256 |
| 4088 | 1.54 | 2647 (2431–2866) | 2781 | 31.4 (31.2–31.7) | 36.7 (35.8–38.7) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
