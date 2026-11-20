### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w6v1, w6v8 | 0.59 / 0.60 (mean 0.595) | 2.51 / 2.52 (mean 2.515) | 5.72 / 5.75 (mean 5.735) |
| on | w6v2, w6v7 | 0.73 / 0.75 (mean 0.740) | 3.38 / 3.32 (mean 3.350) | 7.33 / 7.22 (mean 7.275) |
| defer | w6v3, w6v6 | 0.78 / 0.82 (mean 0.800) | 3.32 / 3.35 (mean 3.335) | 7.36 / 7.32 (mean 7.340) |
| smacrodefer | w6v4, w6v5 | 0.76 / 0.71 (mean 0.735) | 2.96 / 3.05 (mean 3.005) | 6.66 / 6.82 (mean 6.740) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| on | 1.244 | 1.332 | 1.269 |
| defer | 1.345 | 1.326 | 1.280 |
| smacrodefer | 1.235 | 1.195 | 1.175 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6888 | 6620 | 5920 |
| on | 5562 | 4976 | 4660 |
| defer | 5142 | 4996 | 4616 |
| smacrodefer | 5564 | 5542 | 5027 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 119.3 | 99.3 | 81.9 |
| on | 119.7 | 103.5 | 82.2 |
| defer | 120.6 | 99.5 | 81.8 |
| smacrodefer | 122.2 | 98.8 | 84.9 |
