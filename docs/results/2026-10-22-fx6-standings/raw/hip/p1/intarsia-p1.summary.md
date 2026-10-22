# localeval: adhoc — 2026-10-22T01:30

- model: `qwen36-35b-intarsia-hip` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: fx6-intarsia-p1
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen36-35b-intarsia-hip --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-intarsia-p1`
- result file: `results/20261022-013016-adhoc-fx6-intarsia-p1/result.json`

## Speed sweep

served `qwen36-35b-intarsia-hip`, fingerprint `None`, ctx 17408, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 116 | 0.09 | 1355 (1341–1424) * | - | 159.2 (158.9–159.7) | 161.6 (160.8–162.9) | - | - | 128 |
| 991 | 0.17 | 5824 (5407–5829) * | 10417 | 159.0 (158.8–160.9) | 164.4 (160.6–197.2) | - | - | 128 |
| 4091 | 0.45 | 9089 (9082–9117) | 11071 | 155.4 (155.3–155.4) | 157.3 (157.1–158.0) | - | - | 128 |
| 16679 | 2.14 | 7783 (7775–7788) | 7435 | 141.6 (141.6–141.7) | 143.4 (143.2–144.1) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
