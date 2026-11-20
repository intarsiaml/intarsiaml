### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w1v1, w1v8, w4v1, w4v4 | 0.59 / 0.60 / 0.59 / 0.59 (mean 0.593) | 2.52 / 2.50 / 2.51 / 2.51 (mean 2.510) | 5.75 / 5.70 / 5.70 / 5.71 (mean 5.715) |
| smacro | w1v2, w1v7 | 0.77 / 0.76 (mean 0.765) | 3.03 / 3.05 (mean 3.040) | 6.66 / 6.65 (mean 6.655) |
| pf | w1v3, w1v6, w4v2, w4v3 | 0.67 / 0.67 / 0.67 / 0.67 (mean 0.670) | 2.63 / 2.63 / 2.62 / 2.62 (mean 2.625) | 5.86 / 5.83 / 5.84 / 5.83 (mean 5.840) |
| pf4k | w1v4, w1v5 | 0.67 / 0.67 (mean 0.670) | 2.69 / 2.68 (mean 2.685) | 6.00 / 5.91 (mean 5.955) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| smacro | 1.291 | 1.211 | 1.164 |
| pf | 1.131 | 1.046 | 1.022 |
| pf4k | 1.131 | 1.070 | 1.042 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6925 | 6632 | 5932 |
| smacro | 5362 | 5487 | 5094 |
| pf | 6102 | 6347 | 5803 |
| pf4k | 6094 | 6210 | 5694 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 119.6 | 101.3 | 81.6 |
| smacro | 120.1 | 100.3 | 84.3 |
| pf | 120.9 | 100.2 | 81.3 |
| pf4k | 121.0 | 98.8 | 83.2 |
