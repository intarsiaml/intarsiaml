# localeval: adhoc — 2026-10-22T01:48

- model: `qwen36-35b-intarsia-hip` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: fx6-intarsia-old-p5
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen36-35b-intarsia-hip --sizes 128 --gen 128 --reps 3 --thinking off --label fx6-intarsia-old-p5`
- result file: `results/20261022-014850-adhoc-fx6-intarsia-old-p5/result.json`

## Speed sweep

served `qwen36-35b-intarsia-hip`, fingerprint `None`, ctx 17408, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.09 | 1349 (1292–1392) * | - | 159.8 (159.5–160.0) | 162.2 (161.9–176.9) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
