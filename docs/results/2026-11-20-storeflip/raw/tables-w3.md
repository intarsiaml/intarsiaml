### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| dflt | w3v1, w3v4 | 0.67 / 0.68 (mean 0.675) | 2.61 / 2.62 (mean 2.615) | 5.81 / 5.82 (mean 5.815) |
| zero | w3v2, w3v3 | 0.74 / 0.80 (mean 0.770) | 3.29 / 3.33 (mean 3.310) | 7.24 / 7.30 (mean 7.270) |

### Arm mean TTFT / `zero` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| dflt | 0.877 | 0.790 | 0.800 |
| zero | 1.000 | 1.000 | 1.000 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| dflt | 6068 | 6372 | 5827 |
| zero | 5311 | 5027 | 4663 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| dflt | 121.0 | 104.2 | 82.2 |
| zero | 118.8 | 102.3 | 81.6 |
