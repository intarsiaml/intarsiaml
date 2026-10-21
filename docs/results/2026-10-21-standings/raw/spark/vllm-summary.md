# localeval: spark-guidance — 2026-10-21T10:24

- model: `guidance` on spark-4e77 (vLLM), url `http://spark-4e77:8000/v1`
- thinking: off; run kind: run; label: standings-vllm-35b-nvfp4
- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
- command: `/opt/code/deepeval/src/localeval/cli.py run spark-guidance --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --bench mmlu,gsm8k --limit 25 --judge none --wait-idle 900 --label standings-vllm-35b-nvfp4`
- result file: `results/20261021-102415-spark-guidance-standings-vllm-35b-nvfp4/result.json`

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-a11d7492`, ctx 32768, 3 reps, 128 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0.10 | 1195 (1037–1204) * | - | 76.5 (76.3–78.1) | 77.2 (77.0–79.4) | - | - | 128 |
| 990 | 0.19 | 5190 (5120–5220) * | 9125 | 78.0 (76.3–78.2) | 79.0 (76.9–79.1) | - | - | 128 |
| 4090 | 0.64 | 6361 (6205–6385) | 6858 | 76.8 (76.7–77.0) | 77.3 (77.3–79.2) | - | - | 128 |
| 16654 | 2.65 | 6284 (6254–6313) | 6260 | 73.9 (73.9–74.0) | 74.7 (74.3–75.0) | - | - | 128 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.844 | 1425 | 0.15 | 5615 | - | 1425 | 0 | 469.4 |
| gsm8k | 0.520 | 25 | 0.14 | - | - | 25 | 0 | 7.7 |

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Dataset `/opt/code/deepeval/datasets/core.jsonl` (10 rows, judge none)

| metric | mean | pass rate | n | n/a | errors |
|---|---:|---:|---:|---:|---:|
| Contains expected | 0.300 | 0.3 | 10 | 0 | 0 |
| TTFT (s) | 0.071 | 1.0 | 10 | 0 | 0 |
| Prefill (tok/s) | - | - | 0 | 10 | 0 |
| Decode (tok/s) | 78.5 | 1.0 | 9 | 1 | 0 |

Per-row speed is incidental: dataset prompts are short, so prefill here is mostly request overhead. Speed metrics pass unless --min-prefill/--min-decode/--max-ttft is set.

| row | pass | correctness | TTFT (s) | decode tok/s | output |
|---|---|---:|---:|---:|---|
| arith-units | yes | 1.0 | 0.07 | - | 155 |
| logic-sibling | no | 0.0 | 0.07 | 78.9 | Mara has one sister because she and her single sister are th… |
| code-bug | no | 0.0 | 0.07 | 76.6 | # The Problem: Mutable Default Argument ## What's Wrong The … |
| json-extract | no | 0.0 | 0.07 | 77.9 | ```json { "name": "Priya Natarajan", "role": "staff SRE", "s… |
| rag-grounded | yes | 1.0 | 0.08 | 79.7 | Based on the context provided, the EECOM chat server listens… |
| summarize | no | 0.0 | 0.08 | 78.5 | The migration of RETRO from the Spark to the Mac MoE reduced… |
| code-py-dedupe | no | 0.0 | 0.06 | 77.6 | ```python def dedupe(xs): seen = set() result = [] for x in … |
| shell-find | yes | 1.0 | 0.06 | 79.6 | ```sh find /var/log -name '*.log' -mtime -1 -type f ``` |
| instruction-format | no | 0.0 | 0.07 | 79.6 | 1. Catch bugs early in development. 2. Enable safe code refa… |
| refuse-unknown | no | 0.0 | 0.07 | 78.0 | Based on the context provided, there is no information regar… |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
