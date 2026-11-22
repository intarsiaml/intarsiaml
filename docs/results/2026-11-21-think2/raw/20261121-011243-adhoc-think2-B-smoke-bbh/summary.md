# localeval: adhoc — 2026-11-21T01:12

- model: `guidance` on ? (?), url `http://spark-fa00:8191/v1`
- thinking: on; run kind: quality; label: think2-B-smoke-bbh
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git 2c5524c
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://spark-fa00:8191/v1 --model guidance --thinking on --structured prompt --wait-idle 600 --recipe /opt/code/intarsiaml/.worktrees/think2/docs/results/2026-11-21-think2/raw/recipes/B.txt --max-tokens 16384 --bench bbh --tasks dyck_languages,multistep_arithmetic_two,logical_deduction_seven_objects,geometric_shapes,tracking_shuffled_objects_seven_objects --limit 1 --no-unim --label think2-B-smoke-bbh`
- result file: `results/20261121-011243-adhoc-think2-B-smoke-bbh/result.json`

## Launch recipe

from `/opt/code/intarsiaml/.worktrees/think2/docs/results/2026-11-21-think2/raw/recipes/B.txt` (secrets redacted):

````
intarsia serve --backend cuda --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bbh | 1.000 | 5 | 0.25 | 4607 | 149.2 | 5 | 0 | 61.6 |
- bbh: 5 replies kept; 0 truncated (score 0); mean output 1719.8 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
