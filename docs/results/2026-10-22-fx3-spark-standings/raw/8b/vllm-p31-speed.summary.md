# localeval: adhoc — 2026-10-22T08:32

- model: `vllm` on ? (?), url `http://spark-fa00:8300/v1`
- thinking: off; run kind: speed; label: fx3-8b-vllm-p31-speed
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8300/v1 --model vllm --thinking off --label fx3-8b-vllm-p31-speed --sizes 128,1024,4096,16384 --gen 128 --reps 3`
- result file: `results/20261022-083200-adhoc-fx3-8b-vllm-p31-speed/result.json`

## Speed sweep

served `vllm`, fingerprint `vllm-0.31.1rc1.dev296+gfe4df3df0-0eb2ed73`, ctx 20480, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.11 | 1012 (612–1016) * | - | 13.9 (13.8–13.9) | 13.9 (13.8–13.9) | - | - | 128 |
| 990 | 0.24 | 4075 (4068–4101) * | 6652 | 13.8 (13.8–13.8) | 13.8 (13.8–13.8) | - | - | 128 |
| 4088 | 0.89 | 4604 (4588–4611) | 4803 | 13.4 (13.4–13.4) | 13.4 (13.4–13.4) | - | - | 128 |
| 16677 | 4.34 | 3840 (3828–3851) | 3644 | 12.1 (12.1–12.1) | 12.2 (12.2–12.2) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
