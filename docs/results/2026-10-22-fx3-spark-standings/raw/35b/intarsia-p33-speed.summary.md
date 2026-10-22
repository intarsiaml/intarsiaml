# localeval: adhoc — 2026-10-22T06:46

- model: `intarsia` on ? (?), url `http://spark-4e77:8190/v1`
- thinking: off; run kind: speed; label: fx3-35b-intarsia-p33-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8190/v1 --model intarsia --thinking off --label fx3-35b-intarsia-p33-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-064613-adhoc-fx3-35b-intarsia-p33-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.11 | 993 (969–1013) * | - | 84.5 (84.4–85.2) | 85.3 (85.3–105.7) | - | - | 128 |
| 991 | 0.23 | 4309 (4302–4322) * | 7513 | 84.0 (83.9–84.0) | 85.2 (84.6–86.7) | - | - | 128 |
| 4090 | 0.73 | 5604 (5540–5617) | 6210 | 81.5 (81.3–81.6) | 82.5 (82.4–83.1) | - | - | 128 |
| 16657 | 3.18 | 5235 (5223–5244) | 5123 | 73.7 (73.7–73.8) | 74.3 (74.3–74.4) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
