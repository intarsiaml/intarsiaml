# localeval: adhoc — 2026-10-22T01:35

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-hip-p2
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-hip-p2`
- result file: `results/20261022-013500-adhoc-fx6-llama-hip-p2/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.10 | 1093 (959–1109) * | - | 83.3 (82.6–83.9) | 84.4 (83.9–85.2) | 1242 | 83.4 | 128 |
| 990 | 0.33 | 3027 (3004–3077) * | 3915 | 83.2 (83.2–83.2) | 84.6 (84.6–86.9) | 3136 | 83.2 | 128 |
| 4087 | 1.21 | 3381 (3377–3391) | 3515 | 81.6 (81.6–81.7) | 82.9 (82.9–83.0) | 3436 | 81.6 | 128 |
| 16656 | 5.16 | 3227 (3222–3241) | 3180 | 77.7 (77.7–77.8) | 78.8 (78.7–78.8) | 3254 | 77.7 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
