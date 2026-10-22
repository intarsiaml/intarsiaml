# localeval: adhoc — 2026-10-22T06:14

- model: `vllm` on ? (?), url `http://spark-fa00:8300/v1`
- thinking: off; run kind: speed; label: fx3-8b-vllm-p6-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8300/v1 --model vllm --thinking off --label fx3-8b-vllm-p6-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-061419-adhoc-fx3-8b-vllm-p6-speed/result.json`

## Speed sweep

served `vllm`, fingerprint `vllm-0.31.1rc1.dev296+gfe4df3df0-0eb2ed73`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.11 | 1035 (624–1071) * | - | 13.9 (13.9–13.9) | 13.9 (13.9–14.0) | - | - | 128 |
| 992 | 0.24 | 4090 (3996–4094) * | 6652 | 13.8 (13.8–13.8) | 13.8 (13.8–13.9) | - | - | 128 |
| 4090 | 0.90 | 4572 (4557–4577) | 4752 | 13.5 (13.4–13.5) | 13.5 (13.5–13.5) | - | - | 128 |
| 16656 | 4.36 | 3818 (3809–3826) | 3623 | 12.2 (12.2–12.2) | 12.2 (12.2–12.2) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
