# localeval: adhoc — 2026-10-22T05:45

- model: `intarsia` on ? (?), url `http://spark-4e77:8190/v1`
- thinking: off; run kind: speed; label: fx3-35b-intarsia-r22-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8190/v1 --model intarsia --thinking off --label fx3-35b-intarsia-r22-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-054504-adhoc-fx3-35b-intarsia-r22-speed/result.json`

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.11 | 1011 (1008–1017) * | - | 81.7 (81.6–81.9) | 82.6 (82.5–83.5) | - | - | 128 |
| 990 | 0.23 | 4299 (4255–4344) * | 7317 | 81.3 (81.3–81.3) | 82.1 (82.1–83.2) | - | - | 128 |
| 4087 | 0.72 | 5698 (5660–5703) | 6346 | 78.9 (78.9–79.0) | 83.9 (81.3–87.6) | - | - | 128 |
| 16655 | 3.14 | 5304 (5303–5318) | 5189 | 71.6 (71.6–71.6) | 78.1 (72.3–78.7) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
