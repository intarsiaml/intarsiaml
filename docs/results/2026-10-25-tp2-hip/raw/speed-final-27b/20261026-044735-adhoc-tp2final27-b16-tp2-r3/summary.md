# localeval: adhoc — 2026-10-26T04:47

- model: `qwen38-27b-tp2` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: tp2final27-b16-tp2-r3
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen38-27b-tp2 --label tp2final27-b16-tp2-r3 --sizes 1024 --gen 512 --reps 1 --concurrency 4,16 --thinking off --wait-idle 60 --recipe HIP_VISIBLE_DEVICES=0,1 tp2-intarsia serve --backend hip --model /models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0 --served-model-name qwen38-27b-tp2 --max-batch 16 --tp 2 --devices 0,1 --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree lock, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)`
- result file: `results/20261026-044735-adhoc-tp2final27-b16-tp2-r3/result.json`

## Launch recipe

from inline command (secrets redacted):

````
HIP_VISIBLE_DEVICES=0,1 tp2-intarsia serve --backend hip --model /models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0 --served-model-name qwen38-27b-tp2 --max-batch 16 --tp 2 --devices 0,1 --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree lock, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)
````

## Speed sweep

served `qwen38-27b-tp2`, fingerprint `None`, ctx 18432, 1 reps, 512 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 992 | 0.45 | 2215 (2215–2215) * | - | 53.3 (53.3–53.3) | 53.6 (53.6–53.6) | - | - | 512 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 4 | 130.7 | 35.7 (34.2–37.6) | 1.24 |
| 16 | 214.2 | 15.6 (14.4–17.7) | 4.60 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
