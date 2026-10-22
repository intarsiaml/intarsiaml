# localeval: adhoc — 2026-10-22T05:09

- model: `guidance` on ? (?), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: speed; label: fx3-35b-vllm-p1-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8000/v1 --model guidance --thinking off --label fx3-35b-vllm-p1-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-050915-adhoc-fx3-35b-vllm-p1-speed/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 134 | 0.11 | 1254 (1126–1303) * | - | 76.1 (75.9–76.1) | 78.0 (76.8–78.0) | - | - | 128 |
| 990 | 0.20 | 4861 (4789–5105) * | 8825 | 77.5 (76.3–77.9) | 79.2 (77.5–79.6) | - | - | 128 |
| 4090 | 0.67 | 6080 (5901–6090) | 6624 | 77.1 (76.8–77.9) | 81.9 (79.6–86.5) | - | - | 128 |
| 16656 | 2.67 | 6237 (6192–6258) | 6289 | 73.8 (73.7–73.8) | 76.6 (75.9–100.4) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
