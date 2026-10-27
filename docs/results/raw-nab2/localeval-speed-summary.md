# localeval: adhoc — 2026-10-27T09:54

- model: `lfm2.5-1.2b-instruct` on ? (?), url `http://127.0.0.1:8123/v1`
- thinking: off; run kind: speed; label: nab2-lfm2-metal
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8123/v1 --model lfm2.5-1.2b-instruct --sizes 128,1024,4096 --gen 128 --reps 3 --thinking off --label nab2-lfm2-metal --recipe intarsia serve --model LiquidAI/LFM2.5-1.2B-Instruct --backend metal --max-seq 6144 (nab2 branch lane-nab2/lfm2-shortconv)`
- result file: `results/20261027-095421-adhoc-nab2-lfm2-metal/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --model LiquidAI/LFM2.5-1.2B-Instruct --backend metal --max-seq 6144 (nab2 branch lane-nab2/lfm2-shortconv)
````

## Speed sweep

served `lfm2.5-1.2b-instruct`, fingerprint `None`, ctx 6144, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 126 | 0.01 | 8418 (8180–8478) * | - | 196.2 (191.0–199.6) | 200.6 (199.0–200.8) | - | - | 128 |
| 1026 | 0.06 | 16327 (15357–17531) | 18750 | 195.5 (195.4–196.8) | 199.5 (197.8–199.9) | - | - | 128 |
| 4091 | 0.25 | 16579 (16257–16851) | 16658 | 188.8 (184.4–194.6) | 193.0 (191.7–196.8) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
