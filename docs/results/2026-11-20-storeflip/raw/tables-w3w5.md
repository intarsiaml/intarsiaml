### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| dflt | w3v1, w3v4, w5v2, w5v3 | 0.67 / 0.68 / 0.67 / 0.67 (mean 0.673) | 2.61 / 2.62 / 2.61 / 2.61 (mean 2.612) | 5.81 / 5.82 / 5.81 / 5.83 (mean 5.817) |
| zero | w3v2, w3v3, w5v1, w5v4 | 0.74 / 0.80 / 0.78 / 0.78 (mean 0.775) | 3.29 / 3.33 / 3.32 / 3.35 (mean 3.322) | 7.24 / 7.30 / 7.32 / 7.30 (mean 7.290) |

### Arm mean TTFT / `zero` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| dflt | 0.868 | 0.786 | 0.798 |
| zero | 1.000 | 1.000 | 1.000 |

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| dflt | 6083 | 6372 | 5824 |
| zero | 5289 | 5014 | 4650 |

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| dflt | 122.2 | 101.2 | 84.7 |
| zero | 119.2 | 100.8 | 82.2 |
