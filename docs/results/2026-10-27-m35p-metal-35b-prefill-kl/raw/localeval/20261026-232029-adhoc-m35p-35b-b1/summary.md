# localeval: adhoc — 2026-10-26T23:20

- model: `intarsia` on ? (?), url `http://127.0.0.1:18435/v1`
- thinking: off; run kind: speed; label: m35p-35b-b1
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:18435/v1 --model intarsia --sizes 1024,4096,16384 --gen 256 --reps 3 --thinking off --label m35p-35b-b1 --recipe /opt/code/intarsiaml/.worktrees/m35p/.m35p-out/intarsia-base serve --backend metal --model /Users/josef/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18435 `
- result file: `results/20261026-232029-adhoc-m35p-35b-b1/result.json`

## Launch recipe

from inline command (secrets redacted):

````
/opt/code/intarsiaml/.worktrees/m35p/.m35p-out/intarsia-base serve --backend metal --model /Users/josef/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18435
````

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20000, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 988 | 0.60 | 1636 (1631–1638) * | - | 121.4 (118.8–121.8) | 122.8 (121.9–123.0) | - | - | 256 |
| 4088 | 2.31 | 1769 (1748–1770) | 1818 | 118.4 (116.7–118.5) | 119.5 (119.0–119.9) | - | - | 256 |
| 16680 | 12.35 | 1351 (1351–1388) | 1254 | 105.0 (104.1–106.1) | 106.3 (106.2–106.8) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
