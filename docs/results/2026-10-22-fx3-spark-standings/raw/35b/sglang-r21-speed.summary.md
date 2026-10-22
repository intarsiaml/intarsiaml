# localeval: adhoc — 2026-10-22T05:43

- model: `sglang` on ? (?), url `http://spark-4e77:8200/v1`
- thinking: off; run kind: speed; label: fx3-35b-sglang-r21-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8200/v1 --model sglang --thinking off --label fx3-35b-sglang-r21-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-054309-adhoc-fx3-35b-sglang-r21-speed/result.json`

## Speed sweep

served `sglang`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 116 | 0.10 | 1195 (1072–1233) * | - | 75.0 (75.0–75.9) | 75.8 (75.7–76.7) | - | - | 128 |
| 990 | 0.20 | 4948 (4935–4954) * | 8324 | 74.7 (74.7–74.9) | 76.0 (75.5–77.2) | - | - | 128 |
| 4088 | 0.60 | 6772 (6735–6774) | 7668 | 73.8 (73.8–73.8) | 74.5 (74.3–74.9) | - | - | 128 |
| 16678 | 2.63 | 6332 (6270–6382) | 6202 | 71.4 (70.5–71.5) | 72.2 (71.0–72.4) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
