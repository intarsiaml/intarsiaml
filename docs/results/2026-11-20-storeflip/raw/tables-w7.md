### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| m0 | w7v1, w7v4 | 0.67 / 0.67 (mean 0.670) | 2.83 / 2.84 (mean 2.835) | 6.38 / 6.35 (mean 6.365) |
| mac4k | w7v2, w7v3 | 0.67 / 0.67 (mean 0.670) | 2.69 / 2.70 (mean 2.695) | 5.99 / 6.03 (mean 6.010) |

### Arm mean TTFT / `zero` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| m0 | - | - | - |
| mac4k | - | - | - |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| m0 | 6110 | 5880 | 5328 |
| mac4k | 6100 | 6175 | 5641 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| m0 | 121.4 | 100.3 | 88.2 |
| mac4k | 119.4 | 102.7 | 80.6 |
