### Single stream (concurrency 1): medians of visit medians, (min-max over visits)

| engine | n visits | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---|---|---|---|
| intarsia | 6 | 84.5 (84.0-85.8) | 84.0 (83.7-85.6) | 81.5 (81.0-82.9) | 73.7 (73.3-75.1) |
| intarsia4k | 3 | 84.7 (83.6-84.8) | 84.0 (83.1-84.3) | 81.7 (80.3-81.8) | 72.0 (71.6-73.7) |
| vllm | 6 | 75.5 (75.1-76.4) | 76.9 (75.9-78.0) | 76.0 (75.5-79.3) | 72.9 (72.6-73.8) |
| llamacpp | 3 | 55.4 (54.8-56.3) | 55.1 (54.5-55.6) | 54.0 (53.5-55.5) | 51.4 (50.7-52.0) |

| engine | prefill 1024 (per-request) | prefill 4096 (per-request) | prefill 16384 (per-request) | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| intarsia | 4324 (4301-4361) | 5673 (5604-5706) | 5309 (5123-5340) | 6295 (6210-6335) / 5199 (4982-5232) | 0.23 (0.23-0.23) / 0.72 (0.72-0.73) / 3.14 (3.12-3.26) |
| intarsia4k | 4331 (4273-4351) | 5977 (5957-5989) | 5513 (5354-5626) | 6811 (6807-6824) / 5377 (5180-5516) | 0.23 (0.23-0.23) / 0.68 (0.68-0.69) / 3.02 (2.96-3.11) |
| vllm | 5034 (3908-5180) | 6210 (5563-6350) | 6252 (6179-6292) | 6702 (6432-6874) / 6273 (6217-6410) | 0.20 (0.19-0.25) / 0.66 (0.64-0.73) / 2.67 (2.65-2.70) |
| llamacpp | 2079 (2076-2117) | 2385 (2383-2409) | 2377 (2361-2388) | 2505 (2498-2521) / 2374 (2354-2382) | 0.48 (0.47-0.48) / 1.71 (1.70-1.72) / 7.01 (6.97-7.06) |

### Ratio to intarsia (median of visit medians; tie = bands overlap)

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia4k | 1.00x tie | 1.00x tie | 1.00x tie | 0.98x tie | 1.00x tie | 1.05x ahead | 1.04x ahead |
| vllm | 0.89x behind | 0.92x behind | 0.93x behind | 0.99x tie | 1.16x tie | 1.09x tie | 1.18x ahead |
| llamacpp | 0.66x behind | 0.66x behind | 0.66x behind | 0.70x behind | 0.48x behind | 0.42x behind | 0.45x behind |

### Run-to-run noise (visit medians): (max-min)/median in %

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia | 2.1 | 2.2 | 2.3 | 2.4 | 1.4 | 1.8 | 4.1 |
| intarsia4k | 1.4 | 1.5 | 1.8 | 2.9 | 1.8 | 0.5 | 4.9 |
| vllm | 1.8 | 2.7 | 5.0 | 1.6 | 25.3 | 12.7 | 1.8 |
| llamacpp | 2.7 | 1.9 | 3.6 | 2.5 | 2.0 | 1.1 | 1.1 |

### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)

| engine | n | 1: agg tok/s / per-stream / TTFT s / errors | 4: agg tok/s / per-stream / TTFT s / errors | 16: agg tok/s / per-stream / TTFT s / errors |
|---|---:|---|---|---|
| intarsia | 3 | 74 (73-74) / 84.2 / 0.23 / 0 | 74 (74-75) / 84.1 / 2.81 / 0 | 74 (74-75) / 84.1 / 13.15 / 0 |
| vllm | 3 | 70 (70-70) / 77.7 / 0.20 / 0 | 105 (104-106) / 59.0 / 1.52 / 0 | 105 (105-106) / 58.7 / 8.82 / 0 |

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

