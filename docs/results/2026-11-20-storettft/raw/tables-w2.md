### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | w2v1, w2v10 | 0.59 / 0.59 (mean 0.590) | 2.51 / 2.56 (mean 2.535) | 5.71 / 5.69 (mean 5.700) |
| on | w2v2, w2v9 | 0.79 / 0.76 (mean 0.775) | 3.32 / 3.28 (mean 3.300) | 7.29 / 7.20 (mean 7.245) |
| nocap | w2v3, w2v8 | 0.67 / 0.67 (mean 0.670) | 2.80 / 2.81 (mean 2.805) | 6.19 / 6.20 (mean 6.195) |
| offm0 | w2v4, w2v7 | 0.62 / 0.62 (mean 0.620) | 2.75 / 2.75 (mean 2.750) | 6.15 / 6.16 (mean 6.155) |
| gridoff | w2v5, w2v6 | 0.76 / 0.75 (mean 0.755) | 3.18 / 3.26 (mean 3.220) | 7.11 / 7.21 (mean 7.160) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 1.000 | 1.000 | 1.000 |
| on | 1.314 | 1.302 | 1.271 |
| nocap | 1.136 | 1.107 | 1.087 |
| offm0 | 1.051 | 1.085 | 1.080 |
| gridoff | 1.280 | 1.270 | 1.256 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 6910 | 6563 | 5948 |
| on | 5258 | 5042 | 4678 |
| nocap | 6092 | 5946 | 5470 |
| offm0 | 6585 | 6058 | 5502 |
| gridoff | 5416 | 5178 | 4734 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 122.5 | 105.0 | 86.0 |
| on | 119.3 | 100.1 | 80.7 |
| nocap | 117.5 | 103.2 | 81.8 |
| offm0 | 120.3 | 100.9 | 81.1 |
| gridoff | 117.6 | 99.8 | 91.2 |
