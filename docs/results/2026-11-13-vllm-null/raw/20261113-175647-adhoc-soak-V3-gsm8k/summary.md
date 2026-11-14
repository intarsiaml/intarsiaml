# localeval: adhoc — 2026-11-13T17:56

- model: `guidance` on ? (?), url `http://spark-fa00:8000/v1`
- thinking: on; run kind: quality; label: soak-V3-gsm8k
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d96f899
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8000/v1 --model guidance --thinking on --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/vllm-null/.soak-recipes/V3.txt --bench gsm8k --limit 85 --max-tokens 8192 --pin-gsm8k --label soak-V3-gsm8k`
- result file: `results/20261113-175647-adhoc-soak-V3-gsm8k/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/vllm-null/.soak-recipes/V3.txt` (secrets redacted):

````
docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill --tool-call-parser qwen3_xml --reasoning-parser qwen3 (no speculative config); rerun of V-think
````

server-reported (version):

````json
{"version":{"version":"0.26.1rc1.dev602+g65b7662d3"}}
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.988 | 85 | 0.13 | - | 77.1 | 85 | 0 | 2151.8 |

- gsm8k scores: `gsm8k_flexible` 0.988 (primary); `gsm8k_bare_number` 0.847
- gsm8k: 85 replies kept; 0 truncated (score 0); mean output 1937.48 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
