# localeval: adhoc — 2026-11-12T18:49

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: off; run kind: speed; label: ab35-A-post-speed
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8000/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; no speculative config; spark2 retro role; post-window repeat after the container restart) --sizes 128,1024,4096,16384 --gen 256 --reps 3 --concurrency 2 --label ab35-A-post-speed`
- result file: `results/20261112-184939-adhoc-ab35-A-post-speed/result.json`

## Launch recipe

from inline command (secrets redacted):

````
docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; no speculative config; spark2 retro role; post-window repeat after the container restart)
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-9e4897ff`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.08 | 1351 (1330–1435) * | - | 75.9 (75.8–77.4) | 91.7 (90.8–93.5) | - | - | 256 |
| 990 | 0.18 | 5484 (5447–5488) * | 8869 | 77.5 (75.6–77.5) | 92.8 (92.4–93.7) | - | - | 256 |
| 4090 | 0.63 | 6463 (6454–6510) | 6858 | 76.5 (76.3–76.6) | 92.0 (91.2–92.2) | - | - | 256 |
| 16678 | 2.61 | 6393 (6354–6402) | 6370 | 73.5 (73.3–73.6) | 87.7 (87.3–88.8) | - | - | 256 |

\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


| streams | aggregate decode tok/s | per-stream decode | TTFT s |
|---:|---:|---:|---:|
| 2 | 112.1 | 60.1 (59.6–60.6) | 0.31 |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
