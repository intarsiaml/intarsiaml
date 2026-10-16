# localeval: adhoc — 2026-10-15T02:47

- model: `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf` on ? (?), url `http://127.0.0.1:8091/v1`
- thinking: off; run kind: speed; label: final-llamacpp
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8091/v1 --model /Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label final-llamacpp`
- result file: `results/20261015-024739-adhoc-final-llamacpp/result.json`

## Speed sweep

served `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf`, fingerprint `b11429-d81235049`, ctx 8192, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 110 | 0.09 | 1240 (1088–1240) * | - | 35.4 (35.3–35.4) | 35.5 (35.5–35.6) | 1281 | 35.4 | 256 |
| 990 | 0.39 | 2503 (2382–2530) * | 2885 | 35.1 (35.1–35.1) | 35.2 (35.2–35.3) | 2526 | 35.1 | 256 |
| 4091 | 2.03 | 2013 (1930–2115) | 1897 | 34.0 (33.9–34.0) | 34.3 (34.1–34.3) | 2019 | 34.0 | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
