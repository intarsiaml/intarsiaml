# localeval: adhoc — 2026-10-15T04:54

- model: `/Users/josef/models/Qwen3-8B` on ? (?), url `http://127.0.0.1:8092/v1`
- thinking: off; run kind: speed; label: clk-mlx-lm-r1
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8092/v1 --model /Users/josef/models/Qwen3-8B --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-mlx-lm-r1`
- result file: `results/20261015-045418-adhoc-clk-mlx-lm-r1/result.json`

## Speed sweep

served `/Users/josef/models/Qwen3-8B`, fingerprint `0.31.3-0.32.1-macOS-27.0-arm64-arm-64bit-applegpu_g17s`, ctx ?, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 111 | 0.22 | 487 (440–514) * | - | 34.5 (34.4–34.5) | 41.4 (41.3–48.4) | - | - | 256 |
| 991 | 0.43 | 2315 (2310–2366) * | 4272 | 33.8 (33.7–33.9) | 40.4 (39.1–40.6) | - | - | 256 |
| 4090 | 1.34 | 3058 (3053–3093) | 3406 | 32.6 (32.6–32.7) | 39.7 (38.3–40.1) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
