# localeval: adhoc — 2026-10-22T05:50

- model: `sglang` on ? (?), url `http://spark-4e77:8200/v1`
- thinking: off; run kind: speed; label: fx3-35b-sglang-r23-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8200/v1 --model sglang --thinking off --label fx3-35b-sglang-r23-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-055015-adhoc-fx3-35b-sglang-r23-speed/result.json`

## Speed sweep

served `sglang`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 136 | 0.10 | 1408 (746–1415) * | - | 75.0 (74.8–76.7) | 76.1 (75.3–81.4) | - | - | 128 |
| 991 | 0.21 | 4774 (4568–5038) * | 7703 | 73.7 (73.7–73.8) | 75.4 (75.3–76.6) | - | - | 128 |
| 4090 | 0.61 | 6732 (6712–6831) | 7767 | 72.3 (72.2–72.8) | 74.4 (74.3–74.6) | - | - | 128 |
| 16655 | 2.62 | 6364 (6271–6373) | 6251 | 70.5 (70.2–70.6) | 71.1 (70.8–71.3) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
