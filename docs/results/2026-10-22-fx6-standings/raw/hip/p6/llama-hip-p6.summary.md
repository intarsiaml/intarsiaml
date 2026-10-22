# localeval: adhoc — 2026-10-22T01:53

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-hip-p6
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-hip-p6`
- result file: `results/20261022-015341-adhoc-fx6-llama-hip-p6/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 115 | 0.11 | 1080 (1042–1080) * | - | 83.4 (82.3–83.9) | 84.6 (83.6–85.0) | 1201 | 83.4 | 128 |
| 991 | 0.33 | 2983 (2981–2984) * | 3876 | 83.3 (83.3–83.3) | 84.3 (84.3–87.4) | 3097 | 83.3 | 128 |
| 4089 | 1.21 | 3373 (3371–3381) | 3520 | 81.7 (81.7–81.8) | 84.8 (83.1–85.5) | 3426 | 81.7 | 128 |
| 16655 | 5.14 | 3238 (3234–3246) | 3196 | 77.7 (77.7–77.8) | 79.4 (79.0–79.4) | 3261 | 77.8 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
