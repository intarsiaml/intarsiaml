# localeval: adhoc — 2026-10-22T01:46

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-vk-p4
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-vk-p4`
- result file: `results/20261022-014619-adhoc-fx6-llama-vk-p4/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.14 | 831 (831–844) * | - | 115.5 (113.5–115.9) | 117.2 (116.6–118.0) | 913 | 115.4 | 128 |
| 991 | 0.35 | 2814 (2807–2815) * | 4079 | 115.3 (115.2–115.4) | 117.9 (117.9–118.0) | 2928 | 115.3 | 128 |
| 4089 | 1.28 | 3198 (3186–3199) | 3342 | 112.2 (111.8–112.3) | 113.7 (113.5–114.2) | 3242 | 112.2 | 128 |
| 16679 | 5.63 | 2963 (2957–2968) | 2894 | 106.7 (106.6–106.7) | 107.8 (107.7–108.8) | 2983 | 106.7 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
