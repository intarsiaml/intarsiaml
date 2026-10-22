# localeval: adhoc — 2026-10-22T08:58

- model: `vllm` on ? (?), url `http://spark-fa00:8300/v1`
- thinking: off; run kind: speed; label: fx3-8b-vllm-p33-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8300/v1 --model vllm --thinking off --label fx3-8b-vllm-p33-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-085853-adhoc-fx3-8b-vllm-p33-speed/result.json`

## Speed sweep

served `vllm`, fingerprint `vllm-0.31.1rc1.dev296+gfe4df3df0-0eb2ed73`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.10 | 1062 (662–1063) * | - | 13.9 (13.9–13.9) | 13.9 (13.9–13.9) | - | - | 128 |
| 990 | 0.24 | 4055 (4047–4056) * | 6316 | 13.8 (13.8–13.8) | 13.8 (13.8–13.8) | - | - | 128 |
| 4090 | 0.89 | 4602 (4589–4607) | 4806 | 13.4 (13.4–13.4) | 13.5 (13.5–13.5) | - | - | 128 |
| 16656 | 4.42 | 3767 (3747–3775) | 3558 | 12.2 (12.1–12.2) | 12.2 (12.2–12.6) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
