### Single stream (concurrency 1): medians of visit medians, (min-max over visits)

| engine | n visits | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---|---|---|---|
| intarsia | 10 | 15.4 (15.3-15.6) | 15.3 (15.1-15.5) | 14.8 (14.7-15.1) | 13.3 (13.2-13.5) |
| intarsia4k | 6 | 15.3 (15.2-15.3) | 15.1 (15.0-15.2) | 14.7 (14.6-14.8) | 13.2 (13.2-13.3) |
| vllm | 8 | 13.9 (13.9-14.0) | 13.8 (13.8-13.9) | 13.4 (13.4-13.5) | 12.2 (12.0-12.2) |
| sglang | 3 | 13.9 (13.9-13.9) | 13.8 (13.7-13.8) | 13.5 (13.4-13.5) | 12.2 (12.2-12.2) |
| llamacpp | 5 | 15.6 (15.4-15.6) | 15.3 (15.3-15.5) | 14.9 (14.9-15.0) | 13.3 (13.3-13.4) |

| engine | prefill 1024 (per-request) | prefill 4096 (per-request) | prefill 16384 (per-request) | marginal 1k-4k / 4k-16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| intarsia | 2334 (2248-2381) | 2397 (2304-2441) | 2133 (2042-2155) | 2418 (2323-2461) / 2052 (1969-2076) | 0.42 (0.41-0.44) / 1.70 (1.67-1.77) / 7.81 (7.74-8.15) |
| intarsia4k | 4430 (2806-4493) | 4170 (3489-4265) | 3583 (3404-3608) | 4092 (3785-4212) / 3423 (3377-3470) | 0.22 (0.22-0.35) / 0.98 (0.96-1.17) / 4.65 (4.62-4.89) |
| vllm | 4053 (3638-4199) | 4603 (4542-4763) | 3837 (3717-3913) | 4806 (4752-5191) / 3646 (3501-3699) | 0.24 (0.24-0.27) / 0.89 (0.86-0.90) / 4.35 (4.26-4.49) |
| sglang | 3855 (3855-3880) | 4305 (4246-4317) | 3830 (3764-3851) | 4462 (4392-4488) / 3694 (3617-3738) | 0.26 (0.26-0.26) / 0.95 (0.95-0.96) / 4.35 (4.33-4.43) |
| llamacpp | 2833 (2734-2910) | 2865 (2794-2987) | 2417 (2369-2449) | 2914 (2783-3014) / 2289 (2242-2315) | 0.35 (0.34-0.36) / 1.43 (1.37-1.46) / 6.89 (6.80-7.03) |

### Ratio to intarsia (median of visit medians; tie = bands overlap)

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia4k | 0.99x tie | 0.99x tie | 0.99x tie | 1.00x tie | 1.90x ahead | 1.74x ahead | 1.68x ahead |
| vllm | 0.90x behind | 0.90x behind | 0.91x behind | 0.92x behind | 1.74x ahead | 1.92x ahead | 1.80x ahead |
| sglang | 0.91x behind | 0.91x behind | 0.91x behind | 0.92x behind | 1.65x ahead | 1.80x ahead | 1.80x ahead |
| llamacpp | 1.01x tie | 1.00x tie | 1.00x tie | 1.01x tie | 1.21x ahead | 1.20x ahead | 1.13x ahead |

### Run-to-run noise (visit medians): (max-min)/median in %

| engine | decode 128 | decode 1024 | decode 4096 | decode 16384 | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---|---|---|---|---|---|---|
| intarsia | 2.1 | 2.1 | 2.6 | 2.1 | 5.7 | 5.7 | 5.3 |
| intarsia4k | 0.6 | 1.0 | 1.0 | 0.5 | 38.1 | 18.6 | 5.7 |
| vllm | 1.2 | 1.0 | 0.9 | 1.8 | 13.8 | 4.8 | 5.1 |
| sglang | 0.6 | 0.7 | 0.3 | 0.6 | 0.6 | 1.7 | 2.3 |
| llamacpp | 1.2 | 1.2 | 1.1 | 0.8 | 6.2 | 6.7 | 3.3 |

### Concurrency (1k-token prompts, 128 generated tokens, one burst per visit; median over visits)

| engine | n | 1: agg tok/s / per-stream / TTFT s / errors | 4: agg tok/s / per-stream / TTFT s / errors | 16: agg tok/s / per-stream / TTFT s / errors |
|---|---:|---|---|---|
| intarsia | 7 | 15 (14-15) / 15.3 / 0.43 / 0 | 15 (15-15) / 15.3 / 13.47 / 0 | 15 (15-15) / 15.3 / 65.61 / 0 |
| vllm | 5 | 14 (14-14) / 13.8 / 0.24 / 0 | 52 (52-52) / 14.2 / 0.89 / 0 | 156 (156-157) / 11.6 / 2.04 / 0 |
| sglang | 3 | 14 (14-14) / 13.8 / 0.26 / 0 | 52 (52-52) / 14.2 / 0.90 / 0 | 91 (91-154) / 11.9 / 2.43 / 0 |
| llamacpp16 | 6 | 15 (15-15) / 15.4 / 0.34 / 0 | 48 (46-48) / 13.4 / 1.22 / 0 | 114 (109-114) / 9.2 / 3.92 / 0 |

### Quality (localeval quality tier, same bench/limit/structured for every engine)

| engine | visits | benchmark | score | n predictions |
|---|---:|---|---|---:|
| intarsia | 1 | mmlu | 0.676 | 1425 |
| vllm | 1 | mmlu | 0.676 | 1425 |
| sglang | 1 | mmlu | 0.674 | 1425 |
| llamacpp | 1 | mmlu | 0.676 | 1425 |

