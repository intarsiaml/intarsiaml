# localeval: adhoc — 2026-11-20T17:42

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: off; run kind: speed; label: ab35e-A1-speed
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8000/v1 --model guidance --thinking off --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/ab35e/docs/results/2026-11-20-ab35e-cutover-scoreboard/raw/recipes/A.txt --sizes 4096,16384,32768,65536 --gen 256 --reps 3 --label ab35e-A1-speed`
- result file: `results/20261120-174221-adhoc-ab35e-A1-speed/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/ab35e/docs/results/2026-11-20-ab35e-cutover-scoreboard/raw/recipes/A.txt` (secrets redacted):

````
production vLLM (missioncntrl-guidance, image missioncntrl/vllm-openai:65b7662d): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; vllm 0.26.1rc1.dev602; no speculative config; spark2 retro role)
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Speed sweep

served `guidance`, fingerprint `vllm-0.26.1rc1.dev602+g65b7662d3-9e4897ff`, ctx 131072, 3 reps, 256 forced output tokens

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4090 | 0.64 | 6433 (6377–6464) | - | 76.9 (76.8–78.7) | 94.8 (92.4–95.8) | - | - | 256 |
| 16654 | 2.61 | 6382 (6347–6392) | 6365 | 75.0 (74.8–76.0) | 90.7 (90.6–91.5) | - | - | 256 |
| 33903 | 6.09 | 5571 (5429–5583) | 4962 | 72.6 (72.2–76.0) | 93.5 (85.6–120.3) | - | - | 256 |
| 68930 | 15.86 | 4345 (4334–4347) | 3582 | 65.8 (64.6–66.2) | 89.9 (85.0–93.8) | - | - | 256 |

Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.

Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.


## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
