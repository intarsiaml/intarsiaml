# localeval: spark-guidance — 2026-10-22T01:59

- model: `guidance` on spark-4e77 (vLLM), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: speed; label: og-o1-vllm-r1
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed spark-guidance --thinking off --sizes 128,1024,4096,16384 --gen 128 --reps 3 --label og-o1-vllm-r1 --wait-idle 300`
- result file: `results/20261022-015931-spark-guidance-og-o1-vllm-r1/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.11 | 1001 (330–1104) * | - | 76.8 (76.1–77.7) | 78.9 (77.8–79.6) | - | - | 128 |
| 990 | 0.26 | 3856 (3579–4516) * | 6000 | 77.4 (76.9–77.7) | 101.0 (80.0–109.1) | - | - | 128 |
| 4090 | 0.70 | 5845 (4919–5968) | 6998 | 78.6 (77.5–82.7) | 105.1 (86.9–106.1) | - | - | 128 |
| 16677 | 2.67 | 6253 (5793–6320) | 6399 | 74.0 (73.6–74.9) | 84.9 (77.3–96.8) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
