### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w9v1, w9v6 | 0.59 / 0.59 (mean 0.590) | 2.51 / 2.60 (mean 2.555) | 5.72 / 5.71 (mean 5.715) |
| on | w9v2, w9v5 | 0.80 / 0.78 (mean 0.790) | 3.35 / 3.36 (mean 3.355) | 7.24 / 7.30 (mean 7.270) |
| smacro | w9v3, w9v4 | 0.75 / 0.79 (mean 0.770) | 3.04 / 3.05 (mean 3.045) | 6.80 / 6.76 (mean 6.780) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| on | 1.339 | 1.313 | 1.272 |
| smacro | 1.305 | 1.192 | 1.186 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6922 | 6529 | 5932 |
| on | 5176 | 4960 | 4662 |
| smacro | 5344 | 5468 | 5001 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 118.3 | 103.8 | 80.8 |
| on | 121.1 | 105.5 | 81.0 |
| smacro | 117.7 | 99.3 | 83.5 |
