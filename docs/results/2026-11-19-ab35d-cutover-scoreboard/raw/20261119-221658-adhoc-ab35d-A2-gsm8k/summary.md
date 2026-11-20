# localeval: adhoc — 2026-11-19T22:16

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: off; run kind: quality; label: ab35d-A2-gsm8k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8000/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/ab35d/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw/recipes/A.txt --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k --label ab35d-A2-gsm8k`
- result file: `results/20261119-221658-adhoc-ab35d-A2-gsm8k/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/ab35d/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw/recipes/A.txt` (secrets redacted):

````
production vLLM (missioncntrl-guidance, image missioncntrl/vllm-openai:65b7662d): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; vllm 0.26.1rc1.dev602; no speculative config; spark2 retro role)
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.575 | 200 | 0.13 | - | 80.9 | 200 | 0 | 107.3 |

- gsm8k scores: `gsm8k_flexible` 0.575 (primary); `gsm8k_bare_number` 0.375
- gsm8k: 200 replies kept; 0 truncated (score 0); thinking verified off; mean output 32.27 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
