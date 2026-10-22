# localeval: adhoc — 2026-10-22T01:38

- model: `qwen36-35b-intarsia-hip` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: fx6-intarsia-p2
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1b02605
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen36-35b-intarsia-hip --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label fx6-intarsia-p2`
- result file: `results/20261022-013809-adhoc-fx6-intarsia-p2/result.json`

## Speed sweep

served `qwen36-35b-intarsia-hip`, fingerprint `None`, ctx 17408, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.08 | 1361 (1341–1396) * | - | 159.7 (159.6–159.9) | 162.6 (162.1–163.1) | - | - | 128 |
| 992 | 0.17 | 5768 (5723–5775) * | 9778 | 159.5 (159.4–159.5) | 165.8 (161.2–174.5) | - | - | 128 |
| 4087 | 0.45 | 9122 (9034–9146) | 11214 | 155.9 (155.8–156.3) | 168.7 (163.9–178.0) | - | - | 128 |
| 16653 | 2.15 | 7742 (7740–7758) | 7379 | 141.6 (141.2–141.6) | 145.1 (143.1–147.1) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
