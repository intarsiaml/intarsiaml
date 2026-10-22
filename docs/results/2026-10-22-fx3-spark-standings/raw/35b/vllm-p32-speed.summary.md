# localeval: adhoc — 2026-10-22T06:41

- model: `guidance` on ? (?), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: speed; label: fx3-35b-vllm-p32-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8000/v1 --model guidance --thinking off --label fx3-35b-vllm-p32-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-064125-adhoc-fx3-35b-vllm-p32-speed/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 134 | 0.10 | 1405 (1144–1461) * | - | 75.2 (75.1–76.4) | 75.8 (75.6–77.5) | - | - | 128 |
| 988 | 0.19 | 5132 (5119–5146) * | 8714 | 76.8 (76.7–76.9) | 77.4 (77.3–77.7) | - | - | 128 |
| 4088 | 0.64 | 6350 (6320–6373) | 6874 | 75.7 (75.5–75.7) | 76.2 (76.2–76.5) | - | - | 128 |
| 16654 | 2.66 | 6265 (6263–6268) | 6239 | 72.8 (72.8–73.0) | 73.8 (73.5–74.2) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
