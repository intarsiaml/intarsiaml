# localeval: adhoc — 2026-10-26T04:19

- model: `qwen36-35b-tp1` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: tp2final35-b1-tp1-r2
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen36-35b-tp1 --label tp2final35-b1-tp1-r2 --sizes 1024,4096,16384 --gen 512 --reps 1 --thinking off --wait-idle 60 --recipe HIP_VISIBLE_DEVICES=0 tp2-intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --served-model-name qwen36-35b-tp1  --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree lock, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)`
- result file: `results/20261026-041909-adhoc-tp2final35-b1-tp1-r2/result.json`

## Launch recipe

from inline command (secrets redacted):

````
HIP_VISIBLE_DEVICES=0 tp2-intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --served-model-name qwen36-35b-tp1  --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree lock, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)
````

## Speed sweep

served `qwen36-35b-tp1`, fingerprint `None`, ctx 18432, 1 reps, 512 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 991 | 0.14 | 7199 (7199–7199) * | - | 164.1 (164.1–164.1) | 169.1 (169.1–169.1) | - | - | 512 |
| 4087 | 0.41 | 9975 (9975–9975) | 11382 | 160.0 (160.0–160.0) | 175.8 (175.8–175.8) | - | - | 512 |
| 16657 | 1.84 | 9071 (9071–9071) | 8815 | 145.5 (145.5–145.5) | 159.9 (159.9–159.9) | - | - | 512 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
