### TTFT s, median of 3 reps per visit (visit number in the window in brackets)

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|

### Arm mean TTFT / `off` mean (1.00 = same; >1 slower)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|

### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|

### Decode tok/s, arm mean of the visit medians

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|

### TTFT s with an idle gap between requests (ttft_gap.py): median of the reps, mean over visits

| arm | gap s | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | 5 | 0.589 (n=2) | 2.511 (n=2) | 5.720 (n=2) |
| smacro | 5 | 0.673 (n=2) | 2.618 (n=2) | 5.815 (n=2) |
