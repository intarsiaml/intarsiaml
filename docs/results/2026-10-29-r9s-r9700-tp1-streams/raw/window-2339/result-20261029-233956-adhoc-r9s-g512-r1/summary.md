# localeval: adhoc — 2026-10-29T23:39

- model: `qwen36-35b` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: r9s-g512-r1
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen36-35b --label r9s-g512-r1 --sizes 1024 --gen 512 --reps 1 --concurrency 4,8,16 --thinking off --wait-idle 60 --recipe HIP_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --served-model-name qwen36-35b --max-batch 16 --spec off --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree r9s, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)`
- result file: `results/20261029-233956-adhoc-r9s-g512-r1/result.json`

## Launch recipe

from inline command (secrets redacted):

````
HIP_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --served-model-name qwen36-35b --max-batch 16 --spec off --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree r9s, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)
````

## Speed sweep

served `qwen36-35b`, fingerprint `None`, ctx 15410, 1 reps, 512 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 992 | 0.13 | 7423 (7423–7423) * | - | 164.2 (164.2–164.2) | 258.6 (258.6–258.6) | - | - | 512 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 4 | 294.8 | 77.4 (75.2–79.8) | 0.33 |
| 8 | 356.0 | 47.1 (45.5–49.2) | 0.61 |
| 16 | 372.2 | 24.7 (23.7–26.0) | 1.18 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
