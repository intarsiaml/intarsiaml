### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w1v1, w1v8 | 0.59 / 0.60 (mean 0.595) | 2.52 / 2.50 (mean 2.510) | 5.75 / 5.70 (mean 5.725) |
| smacro | w1v2, w1v7 | 0.77 / 0.76 (mean 0.765) | 3.03 / 3.05 (mean 3.040) | 6.66 / 6.65 (mean 6.655) |
| pf | w1v3, w1v6 | 0.67 / 0.67 (mean 0.670) | 2.63 / 2.63 (mean 2.630) | 5.86 / 5.83 (mean 5.845) |
| pf4k | w1v4, w1v5 | 0.67 / 0.67 (mean 0.670) | 2.69 / 2.68 (mean 2.685) | 6.00 / 5.91 (mean 5.955) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| smacro | 1.286 | 1.211 | 1.162 |
| pf | 1.126 | 1.048 | 1.021 |
| pf4k | 1.126 | 1.070 | 1.040 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6898 | 6630 | 5922 |
| smacro | 5362 | 5487 | 5094 |
| pf | 6100 | 6333 | 5800 |
| pf4k | 6094 | 6210 | 5694 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 119.6 | 104.8 | 82.6 |
| smacro | 120.1 | 100.3 | 84.3 |
| pf | 123.8 | 100.2 | 80.2 |
| pf4k | 121.0 | 98.8 | 83.2 |
