# localeval: adhoc — 2026-10-15T02:46

- model: `/Users/josef/models/Qwen3-8B` on ? (?), url `http://127.0.0.1:8092/v1`
- thinking: off; run kind: speed; label: final-mlx-lm
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8092/v1 --model /Users/josef/models/Qwen3-8B --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label final-mlx-lm`
- result file: `results/20261015-024608-adhoc-final-mlx-lm/result.json`

## Speed sweep

served `/Users/josef/models/Qwen3-8B`, fingerprint `0.31.3-0.32.1-macOS-27.0-arm64-arm-64bit-applegpu_g17s`, ctx ?, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.22 | 493 (424–506) * | - | 34.5 (34.4–34.5) | 41.1 (40.2–41.3) | - | - | 256 |
| 991 | 0.43 | 2319 (2238–2319) * | 4309 | 34.0 (33.9–34.0) | 40.9 (38.3–41.1) | - | - | 256 |
| 4089 | 1.38 | 2958 (2886–3062) | 3244 | 32.7 (32.7–32.7) | 38.0 (37.9–41.7) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
