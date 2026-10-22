# localeval: adhoc — 2026-10-22T01:30

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-vk-p1
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-vk-p1`
- result file: `results/20261022-013049-adhoc-fx6-llama-vk-p1/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.14 | 830 (806–831) * | - | 115.6 (114.7–116.1) | 117.2 (116.6–117.4) | 899 | 115.6 | 128 |
| 991 | 0.35 | 2817 (2816–2832) * | 4156 | 115.5 (115.4–115.6) | 117.4 (116.6–117.7) | 2930 | 115.5 | 128 |
| 4089 | 1.28 | 3200 (3199–3210) | 3338 | 112.5 (112.2–112.9) | 114.2 (113.8–117.3) | 3250 | 112.4 | 128 |
| 16656 | 5.61 | 2969 (2966–2976) | 2902 | 106.8 (106.8–107.0) | 108.5 (107.8–109.4) | 2989 | 106.8 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
