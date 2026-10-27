# localeval: tpb-bf16 — 2026-10-27T04:19

- model: `qwen38-27b-tpb` on ? (?), url `http://tatooine:8095/v1`
- thinking: off; run kind: quality; label: tpb-w2-bf16-mmlu-1
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git d12b61c
- command: `/opt/code/deepeval/.venv/bin/localeval quality tpb-bf16 --bench mmlu --limit 25 --thinking off --label tpb-w2-bf16-mmlu-1 --recipe HIP_VISIBLE_DEVICES=0,1 tpb-intarsia serve --backend hip --model 27B --tp 2 --devices 0,1 --max-batch 1 --mem-budget 29 --max-seq 20480 (-e INTARSIA_TP_PARTIAL=bf16; lane tpb) --wait-idle 600`
- result file: `results/20261027-041932-tpb-bf16-tpb-w2-bf16-mmlu-1/result.json`

## Launch recipe

from inline command (secrets redacted):

````
HIP_VISIBLE_DEVICES=0,1 tpb-intarsia serve --backend hip --model 27B --tp 2 --devices 0,1 --max-batch 1 --mem-budget 29 --max-seq 20480 (-e INTARSIA_TP_PARTIAL=bf16; lane tpb)
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.822 | 1425 | 0.21 | 3440 | 53.5 | 1425 | 0 | 534.7 |
- mmlu: 1425 replies kept; 0 truncated (score 0); thinking verified off; mean output 7.21 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
