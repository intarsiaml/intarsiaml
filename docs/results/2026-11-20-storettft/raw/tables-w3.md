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
| dbgsync | 5 | 0.674 (n=1) | 2.853 (n=1) | 6.834 (n=1) |
| off | 5 | 0.587 (n=2) | 2.510 (n=2) | 5.720 (n=2) |
| on | 5 | 0.671 (n=2) | 2.827 (n=2) | 6.810 (n=2) |
