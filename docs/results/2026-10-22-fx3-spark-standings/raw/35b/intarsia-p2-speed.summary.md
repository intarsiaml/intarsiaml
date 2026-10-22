# localeval: adhoc — 2026-10-22T05:23

- model: `intarsia` on ? (?), url `http://spark-4e77:8190/v1`
- thinking: off; run kind: speed; label: fx3-35b-intarsia-p2-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8190/v1 --model intarsia --thinking off --label fx3-35b-intarsia-p2-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-052328-adhoc-fx3-35b-intarsia-p2-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 135 | 0.12 | 1166 (1128–1185) * | - | 84.0 (83.9–84.2) | 84.7 (84.6–88.6) | - | - | 128 |
| 991 | 0.23 | 4335 (4280–4354) * | 7575 | 83.7 (83.5–83.8) | 84.5 (84.4–85.3) | - | - | 128 |
| 4089 | 0.72 | 5695 (5639–5699) | 6335 | 81.0 (80.9–81.1) | 81.8 (81.8–81.9) | - | - | 128 |
| 16656 | 3.14 | 5306 (5300–5321) | 5191 | 73.3 (73.2–73.3) | 74.0 (73.8–74.1) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
