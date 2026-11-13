# localeval: adhoc — 2026-11-12T18:45

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: off; run kind: speed; label: ab35-B-specoff-speed
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8191/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe ~/intarsia-ab35/bin/intarsia serve --spec off --backend cuda --model <HF snapshot 1355db6a> --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 20480 --mem-budget 40 --max-batch 2 --profile-every 0 --sizes 128,1024,4096,16384 --gen 256 --reps 3 --concurrency 2 --label ab35-B-specoff-speed`
- result file: `results/20261112-184521-adhoc-ab35-B-specoff-speed/result.json`

## Launch recipe

from inline command (secrets redacted):

````
~/intarsia-ab35/bin/intarsia serve --spec off --backend cuda --model <HF snapshot 1355db6a> --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 20480 --mem-budget 40 --max-batch 2 --profile-every 0
````

## Speed sweep

served `guidance`, fingerprint `None`, ctx 20480, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.10 | 1146 (695–1149) * | - | 86.6 (86.5–86.6) | 108.1 (107.8–135.2) | - | - | 256 |
| 992 | 0.19 | 5274 (5168–5327) * | 9865 | 86.0 (86.0–86.0) | 107.3 (107.0–111.5) | - | - | 256 |
| 4089 | 0.59 | 6933 (6918–6951) | 7704 | 83.3 (81.5–83.4) | 102.8 (102.1–252.1) | - | - | 256 |
| 16679 | 2.51 | 6637 (6473–6647) | 6547 | 75.4 (75.3–75.4) | 92.4 (90.1–201.5) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 2 | 125.6 | 67.4 (65.7–69.1) | 0.28 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
