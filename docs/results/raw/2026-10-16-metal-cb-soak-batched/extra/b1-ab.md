| run | variant | requests | ITL n | ITL p50 | ITL p95 | ITL p99 | TTFT p50 | TTFT p95 | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| b1-base | base | 80 | 3214 | 28.3 | 28.9 | 29.3 | 433 | 2500 | 35.4 |
| b1-head | cand | 80 | 3214 | 28.0 | 28.8 | 29.4 | 417 | 2470 | 35.7 |
| **base pooled** | | 80 | 3214 | 28.3 | 28.9 | 29.3 | 433 | 2500 | 35.4 |
| **cand pooled** | | 80 | 3214 | 28.0 | 28.8 | 29.4 | 417 | 2470 | 35.7 |

itl_p50: cand/base = 0.990 (limit 1.1)
itl_p95: cand/base = 0.995 (limit 1.1)
RESULT: PASS
