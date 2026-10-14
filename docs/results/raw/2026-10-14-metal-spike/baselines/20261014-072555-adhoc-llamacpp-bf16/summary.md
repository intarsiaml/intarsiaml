# localeval: adhoc — 2026-10-14T07:25

- model: `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf` on ? (?), url `http://127.0.0.1:8091/v1`
- thinking: off; run kind: speed; label: llamacpp-bf16
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8091/v1 --model /Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf --thinking off --sizes 128,1024,4096 --gen 256 --reps 3 --label llamacpp-bf16`
- result file: `results/20261014-072555-adhoc-llamacpp-bf16/result.json`

## Speed sweep

served `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf`, fingerprint `b11146-7fe450e19`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 135 | 0.11 | 1152 (1151–1214) * | - | 34.8 (34.6–35.1) | 35.3 (35.2–35.5) | 1187 | 34.8 | 256 |
| 991 | 0.41 | 2427 (2374–2494) * | 2922 | 33.4 (32.6–34.4) | 34.3 (33.6–35.1) | 2453 | 33.4 | 256 |
| 4089 | 2.36 | 1732 (1442–2035) | 1585 | 31.7 (31.6–32.4) | 33.3 (33.1–33.6) | 1737 | 31.7 | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
