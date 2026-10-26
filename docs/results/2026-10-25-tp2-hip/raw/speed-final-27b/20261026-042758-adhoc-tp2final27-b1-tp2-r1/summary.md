# localeval: adhoc — 2026-10-26T04:27

- model: `qwen38-27b-tp2` on ? (?), url `http://tatooine:8090/v1`
- thinking: off; run kind: speed; label: tp2final27-b1-tp2-r1
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://tatooine:8090/v1 --model qwen38-27b-tp2 --label tp2final27-b1-tp2-r1 --sizes 1024,4096,16384 --gen 512 --reps 1 --thinking off --wait-idle 60 --recipe HIP_VISIBLE_DEVICES=0,1 tp2-intarsia serve --backend hip --model /models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0 --served-model-name qwen38-27b-tp2  --tp 2 --devices 0,1 --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree lock, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)`
- result file: `results/20261026-042758-adhoc-tp2final27-b1-tp2-r1/result.json`

## Launch recipe

from inline command (secrets redacted):

````
HIP_VISIBLE_DEVICES=0,1 tp2-intarsia serve --backend hip --model /models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0 --served-model-name qwen38-27b-tp2  --tp 2 --devices 0,1 --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor --host 0.0.0.0 --port 8090 (tree lock, docker intarsia-hip-dev:rocm7.2.4-rust1.90.0)
````

## Speed sweep

served `qwen38-27b-tp2`, fingerprint `None`, ctx 18432, 1 reps, 512 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 992 | 0.45 | 2190 (2190–2190) * | - | 53.3 (53.3–53.3) | 64.0 (64.0–64.0) | - | - | 512 |
| 4089 | 1.81 | 2260 (2260–2260) | 2284 | 52.6 (52.6–52.6) | 61.8 (61.8–61.8) | - | - | 512 |
| 16657 | 8.18 | 2036 (2036–2036) | 1972 | 50.0 (50.0–50.0) | 50.2 (50.2–50.2) | - | - | 512 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
