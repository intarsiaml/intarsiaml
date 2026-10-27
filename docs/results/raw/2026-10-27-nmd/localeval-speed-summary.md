# localeval: adhoc — 2026-10-27T05:51

- model: `qwen3.5-4b` on ? (?), url `http://127.0.0.1:8123/v1`
- thinking: off; run kind: speed; label: nmd-qwen35-4b-metal
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8123/v1 --model qwen3.5-4b --label nmd-qwen35-4b-metal --thinking off --sizes 128,1024,4096,16384 --recipe intarsia serve --model Qwen3.5-4B --backend metal --port 8123 --served-model-name qwen3.5-4b --max-seq 20480 --profile-every 0 (build lane-nmd/new-model-drill, a_log/norm bf16 cast)`
- result file: `results/20261027-055134-adhoc-nmd-qwen35-4b-metal/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia serve --model Qwen3.5-4B --backend metal --port 8123 --served-model-name qwen3.5-4b --max-seq 20480 --profile-every 0 (build lane-nmd/new-model-drill, a_log/norm bf16 cast)
````

## Speed sweep

served `qwen3.5-4b`, fingerprint `None`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.05 | 2258 (2215–2323) * | - | 61.5 (60.6–62.0) | 62.4 (62.1–62.5) | - | - | 128 |
| 992 | 0.20 | 4835 (4805–4859) * | 5664 | 61.5 (60.1–62.0) | 62.4 (61.5–62.5) | - | - | 128 |
| 4089 | 0.89 | 4623 (4599–4631) | 4554 | 61.0 (60.6–61.0) | 61.6 (61.3–62.1) | - | - | 128 |
| 16678 | 6.03 | 2767 (2690–3002) | 2448 | 57.6 (56.7–57.7) | 58.2 (57.8–58.8) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
