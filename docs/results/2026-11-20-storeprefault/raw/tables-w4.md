### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w4v1, w4v4 | 0.59 / 0.59 (mean 0.590) | 2.51 / 2.51 (mean 2.510) | 5.70 / 5.71 (mean 5.705) |
| pf | w4v2, w4v3 | 0.67 / 0.67 (mean 0.670) | 2.62 / 2.62 (mean 2.620) | 5.84 / 5.83 (mean 5.835) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| pf | 1.136 | 1.044 | 1.023 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6952 | 6634 | 5941 |
| pf | 6104 | 6361 | 5806 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 119.6 | 97.8 | 80.6 |
| pf | 117.9 | 100.2 | 82.4 |
