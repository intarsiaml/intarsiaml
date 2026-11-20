### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w7v1, w7v6 | 0.59 / 0.59 (mean 0.590) | 2.51 / 2.51 (mean 2.510) | 5.72 / 5.70 (mean 5.710) |
| smacro | w7v2, w7v5 | 0.77 / 0.78 (mean 0.775) | 3.08 / 2.99 (mean 3.035) | 6.72 / 6.75 (mean 6.735) |
| smacrodefer | w7v3, w7v4 | 0.77 / 0.76 (mean 0.765) | 3.05 / 3.01 (mean 3.030) | 6.70 / 6.66 (mean 6.680) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| smacro | 1.314 | 1.209 | 1.180 |
| smacrodefer | 1.297 | 1.207 | 1.170 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6951 | 6646 | 5936 |
| smacro | 5270 | 5492 | 5039 |
| smacrodefer | 5364 | 5499 | 5072 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 121.4 | 98.9 | 84.9 |
| smacro | 119.3 | 98.9 | 84.2 |
| smacrodefer | 119.8 | 100.8 | 85.8 |
