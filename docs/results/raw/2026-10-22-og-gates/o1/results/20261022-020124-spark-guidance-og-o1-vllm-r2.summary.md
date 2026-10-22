# localeval: spark-guidance — 2026-10-22T02:01

- model: `guidance` on spark-4e77 (vLLM), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: speed; label: og-o1-vllm-r2
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed spark-guidance --thinking off --sizes 128,1024,4096,16384 --gen 128 --reps 3 --label og-o1-vllm-r2 --wait-idle 300`
- result file: `results/20261022-020124-spark-guidance-og-o1-vllm-r2/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 135 | 0.10 | 1382 (1220–1403) * | - | 75.8 (75.6–75.9) | 77.8 (77.2–78.0) | - | - | 128 |
| 993 | 0.21 | 4735 (4519–4978) * | 7661 | 77.5 (77.3–77.6) | 79.6 (79.3–79.7) | - | - | 128 |
| 4089 | 0.66 | 6231 (6170–6246) | 6926 | 76.0 (75.8–76.1) | 78.0 (77.6–78.3) | - | - | 128 |
| 16656 | 2.64 | 6315 (6301–6363) | 6347 | 73.1 (73.0–73.6) | 75.9 (73.6–76.0) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
