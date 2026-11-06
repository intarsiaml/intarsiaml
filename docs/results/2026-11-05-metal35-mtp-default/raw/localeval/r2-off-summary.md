# localeval: adhoc — 2026-11-05T17:36

- model: `intarsia` on ? (?), url `http://127.0.0.1:18491/v1`
- thinking: off; run kind: speed; label: flipmtp-r2-off
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a43b846
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:18491/v1 --model intarsia --sizes 1024,4096,16384 --gen 256 --reps 3 --thinking off --label flipmtp-r2-off --recipe /opt/code/intarsiaml/.worktrees/flip-metal-mtp/.scratch/intarsia.branch serve --backend metal --model /Users/josef/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18491 --spec off`
- result file: `results/20261105-173610-adhoc-flipmtp-r2-off/result.json`

## Launch recipe

from inline command (secrets redacted):

````
/opt/code/intarsiaml/.worktrees/flip-metal-mtp/.scratch/intarsia.branch serve --backend metal --model /Users/josef/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18491 --spec off
````

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20000, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 992 | 0.22 | 4564 (4559–4576) * | - | 121.9 (121.2–122.3) | 123.1 (123.1–123.1) | - | - | 256 |
| 4090 | 0.70 | 5831 (5360–5841) | 6401 | 118.9 (118.0–119.0) | 120.0 (119.5–120.0) | - | - | 256 |
| 16656 | 4.88 | 3412 (3276–3414) | 3006 | 102.7 (100.1–102.9) | 106.9 (103.9–107.6) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
