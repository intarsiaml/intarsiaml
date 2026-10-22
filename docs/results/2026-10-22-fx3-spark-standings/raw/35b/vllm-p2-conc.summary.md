# localeval: adhoc — 2026-10-22T05:21

- model: `guidance` on ? (?), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: speed; label: fx3-35b-vllm-p2-conc
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-4e77:8000/v1 --model guidance --thinking off --label fx3-35b-vllm-p2-conc --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,16`
- result file: `results/20261022-052111-adhoc-fx3-35b-vllm-p2-conc/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 1 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 990 | 0.22 | 4513 (4513–4513) * | - | 76.3 (76.3–76.3) | 78.0 (78.0–78.0) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 1 | 69.6 | 77.9 (77.9–77.9) | 0.21 |
| 4 | 104.5 | 58.9 (57.5–61.7) | 1.57 |
| 16 | 105.1 | 56.6 (55.7–61.1) | 8.82 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
