# localeval: adhoc — 2026-10-23T14:14

- model: `intarsia` on ? (?), url `http://tatooine:8296/v1`
- thinking: default; run kind: speed; label: hb2-on4
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 3622c22
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8296/v1 --model intarsia --label hb2-on4 --sizes 1024 --gen 128 --reps 3 --concurrency 4,16`
- result file: `results/20261023-141405-adhoc-hb2-on4/result.json`

## Launch recipe

none recorded (pass `--recipe <launch script or command>`)

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 8192, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 989 | 0.16 | 6070 (6012–6118) * | - | 158.0 (157.9–158.4) | 160.1 (160.1–160.7) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 4 | 197.4 | 58.3 (53.2–65.2) | 0.39 |
| 16 | 199.9 | 56.1 (52.6–65.2) | 4.24 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
