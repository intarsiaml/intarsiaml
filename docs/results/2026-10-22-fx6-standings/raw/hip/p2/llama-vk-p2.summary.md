# localeval: adhoc — 2026-10-22T01:37

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-vk-p2
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-vk-p2`
- result file: `results/20261022-013706-adhoc-fx6-llama-vk-p2/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 113 | 0.14 | 833 (788–838) * | - | 115.5 (113.8–115.6) | 116.9 (116.1–117.0) | 909 | 115.5 | 128 |
| 992 | 0.35 | 2796 (2779–2812) * | 4069 | 115.3 (115.2–115.4) | 117.5 (117.2–123.4) | 2895 | 115.3 | 128 |
| 4091 | 1.28 | 3184 (3171–3189) | 3332 | 112.0 (112.0–112.7) | 113.4 (113.3–114.0) | 3232 | 112.1 | 128 |
| 16654 | 5.61 | 2967 (2960–2970) | 2903 | 106.7 (106.5–106.7) | 107.8 (107.6–108.0) | 2986 | 106.6 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
