### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| on | w7v1, w7v4 | 0.75 / 0.78 (mean 0.765) | 3.33 / 3.34 (mean 3.335) | 7.29 / 7.31 (mean 7.300) |
| onpf | w7v2, w7v3 | 0.70 / 0.67 (mean 0.685) | 2.83 / 2.85 (mean 2.840) | 6.33 / 6.36 (mean 6.345) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| on | - | - | - |
| onpf | - | - | - |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| on | 5317 | 4995 | 4642 |
| onpf | 5980 | 5872 | 5343 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| on | 118.1 | 99.8 | 80.6 |
| onpf | 120.6 | 102.4 | 81.6 |
