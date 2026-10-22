# localeval: adhoc — 2026-10-22T01:32

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-hip-p1
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-hip-p1`
- result file: `results/20261022-013258-adhoc-fx6-llama-hip-p1/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 136 | 0.11 | 1239 (1158–1252) * | - | 83.0 (81.7–83.4) | 84.7 (83.6–85.4) | 1379 | 83.0 | 128 |
| 991 | 0.33 | 2983 (2957–3022) * | 3834 | 83.0 (83.0–83.1) | 84.3 (84.3–84.3) | 3105 | 83.1 | 128 |
| 4090 | 1.21 | 3382 (3374–3403) | 3534 | 81.5 (81.4–81.5) | 83.2 (83.2–83.2) | 3435 | 81.5 | 128 |
| 16657 | 5.15 | 3234 (3190–3245) | 3188 | 77.7 (77.6–77.9) | 81.5 (80.8–82.3) | 3264 | 77.6 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
