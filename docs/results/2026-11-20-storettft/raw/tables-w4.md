### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w4v1, w4v8 | 0.59 / 0.60 (mean 0.595) | 2.52 / 2.52 (mean 2.520) | 5.68 / 5.71 (mean 5.695) |
| on | w4v2, w4v7 | 0.67 / 0.68 (mean 0.675) | 2.82 / 2.84 (mean 2.830) | 6.29 / 6.30 (mean 6.295) |
| smacrodefer | w4v3, w4v6 | 0.67 / 0.68 (mean 0.675) | 2.62 / 2.63 (mean 2.625) | 5.82 / 5.82 (mean 5.820) |
| defer | w4v4, w4v5 | 0.68 / 0.67 (mean 0.675) | 2.89 / 2.83 (mean 2.860) | 6.26 / 6.27 (mean 6.265) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| on | 1.134 | 1.123 | 1.105 |
| smacrodefer | 1.134 | 1.042 | 1.022 |
| defer | 1.134 | 1.135 | 1.100 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6914 | 6603 | 5946 |
| on | 6066 | 5885 | 5384 |
| smacrodefer | 6060 | 6352 | 5824 |
| defer | 6046 | 5827 | 5413 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 119.3 | 103.7 | 80.9 |
| on | 120.8 | 102.3 | 90.5 |
| smacrodefer | 122.5 | 98.6 | 82.3 |
| defer | 123.5 | 100.2 | 85.1 |
