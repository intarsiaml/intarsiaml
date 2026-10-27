# localeval: adhoc — 2026-10-26T23:26

- model: `intarsia` on ? (?), url `http://127.0.0.1:18435/v1`
- thinking: off; run kind: speed; label: m35p-8b
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:18435/v1 --model intarsia --sizes 1024,4096,16384 --gen 256 --reps 3 --thinking off --label m35p-8b --recipe /opt/code/intarsiaml/.worktrees/m35p/.m35p-out/intarsia-base serve --backend metal --model /Users/josef/models/Qwen3-8B --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18435 `
- result file: `results/20261026-232637-adhoc-m35p-8b/result.json`

## Launch recipe

from inline command (secrets redacted):

````
/opt/code/intarsiaml/.worktrees/m35p/.m35p-out/intarsia-base serve --backend metal --model /Users/josef/models/Qwen3-8B --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18435
````

## Speed sweep

served `intarsia`, fingerprint `None`, ctx 20000, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 991 | 0.25 | 3997 (3777–4008) * | - | 35.9 (35.8–36.1) | 36.7 (36.6–36.8) | - | - | 256 |
| 4087 | 1.04 | 3914 (3893–3927) | 3889 | 35.1 (34.8–35.4) | 35.9 (35.8–36.0) | - | - | 256 |
| 16653 | 6.33 | 2630 (2269–2740) | 2376 | 31.6 (31.5–31.7) | 32.1 (32.0–32.1) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
