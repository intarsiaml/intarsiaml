### Single stream (concurrency 1): medians of visit medians, (min-max over visits)

| engine | n visits | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---|---|---|---|
| intarsia | 1 | 81.7 (81.7-81.7) | 81.3 (81.3-81.3) | 78.9 (78.9-78.9) | 71.6 (71.6-71.6) |
| sglang | 3 | 75.0 (74.9-75.0) | 74.5 (73.7-74.7) | 73.6 (72.3-73.8) | 70.8 (70.5-71.4) |

| engine | prefill 1024 (per-request) | prefill 4096 (per-request) | prefill 16384 (per-request) | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| intarsia | 4299 (4299-4299) | 5698 (5698-5698) | 5304 (5304-5304) | 6346 (6346-6346) / 5189 (5189-5189) | 0.23 (0.23-0.23) / 0.72 (0.72-0.72) / 3.14 (3.14-3.14) |
| sglang | 4948 (4774-5018) | 6772 (6732-6828) | 6364 (6332-6384) | 7728 (7668-7767) / 6251 (6202-6251) | 0.20 (0.20-0.21) / 0.60 (0.60-0.61) / 2.62 (2.61-2.63) |

### Ratio to intarsia (median of visit medians; tie = bands overlap)

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| sglang | 0.92x behind | 0.92x behind | 0.93x behind | 0.99x behind | 1.15x ahead | 1.19x ahead | 1.20x ahead |

### Run-to-run noise (visit medians): (max-min)/median in %

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia | - | - | - | - | - | - | - |
| sglang | 0.2 | 1.3 | 2.0 | 1.2 | 4.9 | 1.4 | 0.8 |

### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)

| engine | n | 1: agg tok/s / per-stream / TTFT s / errors | 4: agg tok/s / per-stream / TTFT s / errors | 16: agg tok/s / per-stream / TTFT s / errors |
|---|---:|---|---|---|
| intarsia | 1 | 71 (71-71) / 81.2 / 0.23 / 0 | 72 (72-72) / 81.1 / 2.90 / 0 | 72 (72-72) / 81.0 / 13.60 / 0 |
| sglang | 3 | 67 (67-68) / 74.6 / 0.20 / 0 | 155 (149-158) / 47.4 / 0.62 / 0 | 217 (217-220) / 23.0 / 2.02 / 0 |

### Quality (localeval quality tier, same bench/limit/structured for every engine)

| engine | visits | benchmark | score | n predictions |
|---|---:|---|---|---:|
| intarsia | 1 | mmlu | 0.840 | 1425 |
| intarsia | 1 | gsm8k | 0.680 | 25 |
| vllm | 1 | mmlu | 0.840 | 1425 |
| vllm | 1 | gsm8k | 0.560 | 25 |
| sglang | 1 | mmlu | 0.833 | 1425 |
| sglang | 1 | gsm8k | 0.600 | 25 |
| llamacpp | 1 | mmlu | 0.836 | 1425 |
| llamacpp | 1 | gsm8k | 0.680 | 25 |

