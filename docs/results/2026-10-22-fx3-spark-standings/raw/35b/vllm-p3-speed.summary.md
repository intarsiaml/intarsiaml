# localeval: adhoc — 2026-10-22T05:25

- model: `guidance` on ? (?), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: speed; label: fx3-35b-vllm-p3-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8000/v1 --model guidance --thinking off --label fx3-35b-vllm-p3-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-052525-adhoc-fx3-35b-vllm-p3-speed/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 115 | 0.09 | 1259 (1183–1265) * | - | 75.9 (75.8–75.9) | 78.1 (77.8–78.4) | - | - | 128 |
| 991 | 0.20 | 4937 (4817–5173) * | 8037 | 77.0 (76.8–77.9) | 79.2 (77.8–79.4) | - | - | 128 |
| 4090 | 0.67 | 6136 (6119–6225) | 6664 | 76.4 (76.1–76.8) | 77.1 (76.8–78.3) | - | - | 128 |
| 16681 | 2.65 | 6292 (6269–6309) | 6343 | 72.8 (72.8–73.1) | 74.7 (73.9–75.7) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
