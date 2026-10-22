# localeval: adhoc — 2026-10-22T01:55

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-vk-p6
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-vk-p6`
- result file: `results/20261022-015519-adhoc-fx6-llama-vk-p6/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.14 | 815 (778–821) * | - | 115.2 (113.7–115.5) | 117.2 (116.8–117.8) | 889 | 115.0 | 128 |
| 990 | 0.35 | 2795 (2792–2795) * | 4094 | 114.8 (114.7–114.8) | 116.3 (116.0–116.5) | 2899 | 114.7 | 128 |
| 4089 | 1.28 | 3183 (3182–3198) | 3336 | 111.6 (111.5–112.2) | 113.2 (113.2–113.4) | 3238 | 111.7 | 128 |
| 16653 | 5.62 | 2963 (2959–2970) | 2898 | 106.2 (106.2–106.2) | 107.5 (107.5–108.1) | 2985 | 106.2 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
