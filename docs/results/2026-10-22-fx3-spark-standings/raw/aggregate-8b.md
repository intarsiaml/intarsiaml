### Single stream (concurrency 1): medians of visit medians, (min-max over visits)

| engine | n visits | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---|---|---|---|
| intarsia | 6 | 15.5 (15.4-15.6) | 15.4 (15.3-15.5) | 14.9 (14.8-15.1) | 13.4 (13.3-13.5) |
| vllm | 4 | 14.0 (13.9-14.0) | 13.9 (13.8-13.9) | 13.5 (13.5-13.5) | 12.2 (12.0-12.2) |
| sglang | 2 | 13.9 (13.9-13.9) | 13.8 (13.8-13.8) | 13.5 (13.5-13.5) | 12.2 (12.2-12.2) |
| llamacpp | 5 | 15.6 (15.4-15.6) | 15.3 (15.3-15.5) | 14.9 (14.9-15.0) | 13.3 (13.3-13.4) |

| engine | prefill 1024 (per-request) | prefill 4096 (per-request) | prefill 16384 (per-request) | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| intarsia | 2305 (2248-2338) | 2367 (2304-2401) | 2093 (2042-2109) | 2386 (2323-2421) / 2019 (1969-2029) | 0.43 (0.42-0.44) / 1.73 (1.70-1.77) / 7.97 (7.90-8.15) |
| vllm | 4017 (3638-4090) | 4638 (4542-4724) | 3854 (3818-3884) | 4876 (4752-5191) / 3656 (3623-3672) | 0.25 (0.24-0.27) / 0.88 (0.86-0.90) / 4.32 (4.29-4.36) |
| sglang | 3867 (3855-3880) | 4311 (4305-4317) | 3797 (3764-3830) | 4475 (4462-4488) / 3656 (3617-3694) | 0.26 (0.26-0.26) / 0.95 (0.95-0.95) / 4.39 (4.35-4.43) |
| llamacpp | 2833 (2734-2910) | 2865 (2794-2987) | 2417 (2369-2449) | 2914 (2783-3014) / 2289 (2242-2315) | 0.35 (0.34-0.36) / 1.43 (1.37-1.46) / 6.89 (6.80-7.03) |

### Ratio to intarsia (median of visit medians; tie = bands overlap)

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| vllm | 0.90x behind | 0.90x behind | 0.90x behind | 0.91x behind | 1.74x ahead | 1.96x ahead | 1.84x ahead |
| sglang | 0.90x behind | 0.90x behind | 0.90x behind | 0.91x behind | 1.68x ahead | 1.82x ahead | 1.81x ahead |
| llamacpp | 1.00x tie | 1.00x tie | 1.00x tie | 1.00x tie | 1.23x ahead | 1.21x ahead | 1.15x ahead |

### Run-to-run noise (visit medians): (max-min)/median in %

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia | 1.4 | 1.4 | 1.7 | 1.1 | 3.9 | 4.1 | 3.2 |
| vllm | 0.9 | 0.7 | 0.6 | 1.8 | 11.2 | 3.9 | 1.7 |
| sglang | 0.1 | 0.1 | 0.0 | 0.4 | 0.6 | 0.3 | 1.7 |
| llamacpp | 1.2 | 1.2 | 1.1 | 0.8 | 6.2 | 6.7 | 3.3 |

### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)

| engine | n | 1: agg tok/s / per-stream / TTFT s / errors | 4: agg tok/s / per-stream / TTFT s / errors | 16: agg tok/s / per-stream / TTFT s / errors |
|---|---:|---|---|---|
| intarsia | 6 | 15 (15-15) / 15.4 / 0.43 / 0 | 15 (15-15) / 15.4 / 13.43 / 0 | 15 (15-15) / 15.4 / 65.42 / 0 |
| vllm | 4 | 14 (14-14) / 13.9 / 0.24 / 0 | 52 (52-52) / 14.2 / 0.89 / 0 | 157 (156-157) / 12.0 / 2.41 / 0 |
| sglang | 2 | 14 (14-14) / 13.8 / 0.25 / 0 | 52 (52-52) / 14.2 / 0.90 / 0 | 122 (91-154) / 11.8 / 2.46 / 0 |
| llamacpp16 | 6 | 15 (15-15) / 15.4 / 0.34 / 0 | 48 (46-48) / 13.4 / 1.22 / 0 | 114 (109-114) / 9.2 / 3.92 / 0 |

### Quality (localeval quality tier, same bench/limit/structured for every engine)

| engine | visits | benchmark | score | n predictions |
|---|---:|---|---|---:|
| intarsia | 1 | mmlu | 0.676 | 1425 |
| vllm | 1 | mmlu | 0.676 | 1425 |

