# localeval: adhoc — 2026-10-22T01:51

- model: `MXFP4_MOE` on ? (?), url `http://tatooine:8091/v1`
- thinking: off; run kind: speed; label: fx6-llama-vk4-p6
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8091/v1 --model MXFP4_MOE --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-llama-vk4-p6`
- result file: `results/20261022-015132-adhoc-fx6-llama-vk4-p6/result.json`

## Speed sweep

served `MXFP4_MOE`, fingerprint `b10864-5d806aa25`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.15 | 742 (734–745) * | - | 113.1 (111.4–113.4) | 115.1 (114.2–116.5) | 800 | 113.1 | 128 |
| 990 | 0.39 | 2536 (2526–2539) * | 3696 | 112.6 (112.5–112.8) | 114.1 (114.0–120.0) | 2623 | 112.6 | 128 |
| 4090 | 1.43 | 2861 (2844–2862) | 2986 | 110.4 (110.1–110.5) | 111.6 (111.4–112.1) | 2906 | 110.2 | 128 |
| 16657 | 6.22 | 2680 (2671–2687) | 2626 | 105.1 (105.1–105.1) | 106.5 (105.8–110.4) | 2695 | 105.1 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
