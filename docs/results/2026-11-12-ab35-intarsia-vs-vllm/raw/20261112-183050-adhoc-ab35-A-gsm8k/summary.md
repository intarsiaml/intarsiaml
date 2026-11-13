# localeval: adhoc — 2026-11-12T18:30

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: off; run kind: quality; label: ab35-A-gsm8k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8000/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; no speculative config; spark2 retro role) --bench gsm8k --limit 100 --max-tokens 1024 --nonce --pin-gsm8k --label ab35-A-gsm8k`
- result file: `results/20261112-183050-adhoc-ab35-A-gsm8k/result.json`

## Launch recipe

from inline command (secrets redacted):

````
docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; no speculative config; spark2 retro role)
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.590 | 100 | 0.14 | - | 78.5 | 100 | 0 | 61.6 |

- gsm8k scores: `gsm8k_flexible` 0.590 (primary); `gsm8k_bare_number` 0.350
- gsm8k: 100 replies kept; 0 truncated (score 0); thinking verified off; mean output 36.87 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
