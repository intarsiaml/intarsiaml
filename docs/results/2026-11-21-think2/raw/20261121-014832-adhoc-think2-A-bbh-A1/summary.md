# localeval: adhoc — 2026-11-21T01:48

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: on; run kind: quality; label: think2-A-bbh-A1
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8000/v1 --model guidance --thinking on --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/think2/docs/results/2026-11-21-think2/raw/recipes/A.txt --max-tokens 16384 --bench bbh --tasks dyck_languages,multistep_arithmetic_two,logical_deduction_seven_objects,geometric_shapes,tracking_shuffled_objects_seven_objects --limit 30 --label think2-A-bbh-A1`
- result file: `results/20261121-014832-adhoc-think2-A-bbh-A1/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/think2/docs/results/2026-11-21-think2/raw/recipes/A.txt` (secrets redacted):

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
| bbh | 0.933 | 150 | 0.20 | 5245 | 80.0 | 152 | 0 | 4179.8 |
- bbh: 150 replies kept; 1 truncated (score 0); mean output 1973.06 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
