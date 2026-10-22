# localeval: adhoc — 2026-10-22T01:39

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-vk-p3
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-vk-p3`
- result file: `results/20261022-013905-adhoc-fx6-llama-vk-p3/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b1-7fe450e`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 136 | 0.14 | 969 (953–969) * | - | 114.7 (113.9–115.1) | 116.4 (115.3–122.0) | 1055 | 114.6 | 128 |
| 991 | 0.36 | 2784 (2781–2809) * | 3977 | 114.7 (114.6–114.8) | 115.9 (115.9–118.0) | 2888 | 114.7 | 128 |
| 4090 | 1.28 | 3199 (3183–3199) | 3358 | 111.7 (111.5–112.0) | 113.5 (113.1–117.9) | 3245 | 111.7 | 128 |
| 16632 | 5.60 | 2972 (2965–2976) | 2905 | 106.0 (106.0–106.1) | 109.9 (108.7–110.5) | 2990 | 106.0 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
