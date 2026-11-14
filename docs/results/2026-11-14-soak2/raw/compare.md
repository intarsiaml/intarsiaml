| run | replays | requests | errors | TTFT sum s per replay (mean, min-max) | TTFT p50 / p90 / p99 s | decode tok/s mean / median | client ITL p99 ms median / max | agg output tok/s (mean of replays) | cached frac | wall s per replay | engine gaps >0.25 s / >0.5 s of total |
|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---|
| soak (4 GiB stress store, no MTP, greedy), spark2, vLLM down, 2026-11-13 | 10 | 480 | 0 | 179.8 (169.4-199.1) | 2.52 / 7.10 / 11.27 | 34.4 / 40.4 | 418 / 2375 | 33.3 | 0.201 | 224 | 3021 / 0 of 74355 |
| restore-acc soak (8 GiB default, no MTP, greedy), spark2, vLLM down, 2026-11-13 | 15 | 720 | 0 | 70.5 (66.6-76.9) | 1.41 / 1.77 / 2.88 | 41.6 / 40.8 | 276 / 2295 | 58.5 | 0.732 | 127 | 5 / 0 of 110650 |
| soak2 (8 GiB default, MTP on, half T=1.0), spark1, vLLM up, 2026-11-14 | 18 | 864 | 0 | 71.0 (65.9-78.8) | 1.39 / 2.04 / 2.85 | 42.8 / 42.0 | 249 / 1484 | 59.6 | 0.732 | 125 | 1 / 0 of 137156 |
