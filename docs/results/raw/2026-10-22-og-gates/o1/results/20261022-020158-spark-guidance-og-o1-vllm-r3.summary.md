# localeval: spark-guidance — 2026-10-22T02:01

- model: `guidance` on spark-4e77 (vLLM), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: speed; label: og-o1-vllm-r3
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed spark-guidance --thinking off --sizes 128,1024,4096,16384 --gen 128 --reps 3 --label og-o1-vllm-r3 --wait-idle 300`
- result file: `results/20261022-020158-spark-guidance-og-o1-vllm-r3/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 116 | 0.14 | 856 (836–1226) * | - | 76.3 (76.3–76.4) | 80.0 (78.8–80.1) | - | - | 128 |
| 991 | 0.25 | 3971 (3871–4135) * | 7675 | 77.3 (75.9–78.1) | 79.3 (78.7–80.4) | - | - | 128 |
| 4090 | 0.73 | 5576 (5319–6278) | 6403 | 76.0 (74.7–76.2) | 78.1 (77.9–80.3) | - | - | 128 |
| 16654 | 2.64 | 6310 (6306–6334) | 6595 | 73.3 (73.1–73.4) | 94.6 (75.7–97.7) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
