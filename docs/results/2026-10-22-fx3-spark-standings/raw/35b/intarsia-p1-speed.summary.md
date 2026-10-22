# localeval: adhoc — 2026-10-22T05:07

- model: `intarsia` on ? (?), url `http://spark-4e77:8190/v1`
- thinking: off; run kind: speed; label: fx3-35b-intarsia-p1-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8190/v1 --model intarsia --thinking off --label fx3-35b-intarsia-p1-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-050755-adhoc-fx3-35b-intarsia-p1-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 115 | 0.11 | 1037 (1027–1045) * | - | 85.8 (82.1–86.1) | 87.1 (86.4–87.2) | - | - | 128 |
| 992 | 0.23 | 4335 (4311–4380) * | 7370 | 85.6 (85.2–85.6) | 86.5 (86.3–86.6) | - | - | 128 |
| 4088 | 0.72 | 5661 (5604–5676) | 6267 | 82.9 (82.9–82.9) | 83.6 (83.4–83.7) | - | - | 128 |
| 16677 | 3.14 | 5312 (5306–5323) | 5208 | 75.1 (75.1–75.3) | 76.0 (75.7–79.5) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
