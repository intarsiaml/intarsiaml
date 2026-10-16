# localeval: adhoc — 2026-10-14T17:07

- model: `/Users/josef/models/Qwen3-8B` on ? (?), url `http://127.0.0.1:8092/v1`
- thinking: off; run kind: speed; label: sameslot-mlx-lm
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8092/v1 --model /Users/josef/models/Qwen3-8B --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label sameslot-mlx-lm`
- result file: `results/20261014-170715-adhoc-sameslot-mlx-lm/result.json`

## Speed sweep

served `/Users/josef/models/Qwen3-8B`, fingerprint `0.31.3-0.32.1-macOS-27.0-arm64-arm-64bit-applegpu_g17s`, ctx ?, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 133 | 0.23 | 566 (324–597) * | - | 34.3 (33.6–34.3) | 43.2 (39.3–43.3) | - | - | 256 |
| 991 | 0.42 | 2337 (2252–2340) * | 4423 | 33.6 (32.9–33.6) | 39.6 (39.0–41.1) | - | - | 256 |
| 4087 | 1.41 | 2896 (2287–2933) | 3134 | 32.6 (32.5–32.6) | 37.9 (37.7–41.8) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
