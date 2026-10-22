# localeval: adhoc — 2026-10-22T05:29

- model: `llamacpp` on ? (?), url `http://spark-fa00:8400/v1`
- thinking: off; run kind: speed; label: fx3-8b-llamacpp16-p4-conc
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8400/v1 --model llamacpp --thinking off --label fx3-8b-llamacpp16-p4-conc --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,16`
- result file: `results/20261022-052942-adhoc-fx3-8b-llamacpp16-p4-conc/result.json`

## Speed sweep

served `llamacpp`, fingerprint `b1-d88040c`, ctx 5120, 1 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 991 | 0.34 | 2880 (2880–2880) * | - | 15.4 (15.4–15.4) | 16.0 (16.0–16.0) | 3000 | 15.4 | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 1 | 14.9 | 15.4 (15.4–15.4) | 0.35 |
| 4 | 48.0 | 13.5 (12.5–14.0) | 1.22 |
| 16 | 114.2 | 9.2 (7.6–11.4) | 3.91 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
