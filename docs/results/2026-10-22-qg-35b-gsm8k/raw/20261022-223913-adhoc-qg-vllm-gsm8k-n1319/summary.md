# localeval: adhoc — 2026-10-22T22:39

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: off; run kind: quality; label: qg-vllm-gsm8k-n1319
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 98707d5
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8000/v1 --model guidance --thinking off --structured none --bench gsm8k --limit 1319 --label qg-vllm-gsm8k-n1319 --recipe docker missioncntrl/vllm-openai:65b7662d: vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (spark2 retro role) --wait-idle 600`
- result file: `results/20261022-223913-adhoc-qg-vllm-gsm8k-n1319/result.json`

## Launch recipe

from inline command (secrets redacted):

````
docker missioncntrl/vllm-openai:65b7662d: vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (spark2 retro role)
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.582 | 1319 | 0.31 | - | 27.0 | 1319 | 0 | 2732.5 |

- gsm8k scores: `gsm8k_flexible` 0.582 (primary); `gsm8k_strict` 0.271

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
