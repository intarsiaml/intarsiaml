### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| smacro | w6v1, w6v4 | 0.67 / 0.68 (mean 0.675) | 2.63 / 2.63 (mean 2.630) | 5.83 / 5.87 (mean 5.850) |
| pf | w6v2, w6v3 | 0.67 / 0.68 (mean 0.675) | 2.62 / 2.64 (mean 2.630) | 5.81 / 5.83 (mean 5.820) |

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| smacro | - | - | - |
| pf | - | - | - |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| smacro | 6055 | 6339 | 5794 |
| pf | 6042 | 6336 | 5823 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| smacro | 123.1 | 101.8 | 84.3 |
| pf | 120.8 | 99.2 | 83.2 |
