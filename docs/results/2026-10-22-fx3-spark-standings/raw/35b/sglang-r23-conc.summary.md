# localeval: adhoc — 2026-10-22T05:50

- model: `sglang` on ? (?), url `http://spark-4e77:8200/v1`
- thinking: off; run kind: speed; label: fx3-35b-sglang-r23-conc
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8200/v1 --model sglang --thinking off --label fx3-35b-sglang-r23-conc --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,16`
- result file: `results/20261022-055053-adhoc-fx3-35b-sglang-r23-conc/result.json`

## Speed sweep

served `sglang`, fingerprint `None`, ctx 20480, 1 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 988 | 0.20 | 4953 (4953–4953) * | - | 74.7 (74.7–74.7) | 75.3 (75.3–75.3) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 1 | 67.4 | 74.6 (74.6–74.6) | 0.20 |
| 4 | 148.9 | 44.2 (40.2–46.8) | 0.57 |
| 16 | 219.9 | 23.0 (17.7–74.6) | 1.88 |

## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
