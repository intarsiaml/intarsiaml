# localeval: adhoc — 2026-11-05T17:38

- model: `intarsia` on ? (?), url `http://127.0.0.1:18491/v1`
- thinking: off; run kind: speed; label: flipmtp-r3-mtp
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a43b846
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:18491/v1 --model intarsia --sizes 1024,4096,16384 --gen 256 --reps 3 --thinking off --label flipmtp-r3-mtp --recipe /opt/code/intarsiaml/.worktrees/flip-metal-mtp/.scratch/intarsia.branch serve --backend metal --model /Users/josef/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18491`
- result file: `results/20261105-173808-adhoc-flipmtp-r3-mtp/result.json`

## Launch recipe

from inline command (secrets redacted):

````
/opt/code/intarsiaml/.worktrees/flip-metal-mtp/.scratch/intarsia.branch serve --backend metal --model /Users/josef/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18491
````

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20000, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 990 | 0.26 | 3862 (3450–3980) * | - | 148.9 (145.7–172.1) | 214.1 (185.5–219.5) | - | - | 256 |
| 4088 | 0.93 | 4409 (4316–4416) | 4624 | 140.5 (139.1–141.7) | 185.0 (181.9–186.6) | - | - | 256 |
| 16655 | 4.59 | 3626 (3363–3673) | 3428 | 119.6 (109.4–122.6) | 155.1 (128.9–159.1) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Warnings

- decode spread 17.7% at 1024 tokens: noisy; rerun before comparing
- decode spread 11.1% at 16384 tokens: noisy; rerun before comparing

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
