Simulated speedup vs plain decode (cost model: plain step 1, verify pass 1 + 0.06 x drafts), replayed over the greedy streams; k = 4

| row | ungated (min_n 1) | min_n 2 only | min_n 2 + streak 6 (CHOSEN: 6,2,32,2) | min_n 1 + streak 2 (4,64) | min_n 1 + streak 3 (4,64) | min_n 3 only |
|---|---|---|---|---|---|---|
| sweep-open-256 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sweep-grounded-256 | 3.16 | 3.13 | 3.13 | 3.06 | 3.16 | 3.12 |
| sweep-golden-docs-256 | 1.00 | 1.07 | 1.08 | 1.00 | 1.00 | 1.04 |
| sweep-golden-code-256 | 1.78 | 1.68 | 1.69 | 1.43 | 1.50 | 1.61 |
| sweep-4k-256 | 3.07 | 3.07 | 3.07 | 2.93 | 3.07 | 2.60 |
| sweep-short-256 | 2.13 | 1.93 | 1.93 | 1.54 | 2.13 | 1.72 |
| g3-grounded-edit-docs-comments | 2.06 | 2.05 | 2.05 | 1.86 | 1.89 | 1.98 |
| g3-grounded-edit-error-msgs | 3.25 | 3.25 | 3.25 | 3.23 | 2.90 | 3.14 |
| g3-grounded-edit-field | 2.98 | 2.95 | 2.95 | 2.83 | 2.84 | 2.89 |
| g3-grounded-edit-refactor-model | 2.94 | 2.95 | 2.95 | 2.60 | 2.94 | 2.79 |
| g3-grounded-edit-rename-fn | 3.56 | 3.41 | 3.41 | 3.56 | 3.56 | 3.34 |
| g3-grounded-extract-fields | 2.11 | 1.89 | 1.89 | 1.67 | 1.83 | 1.69 |
| g3-grounded-extract-ops | 1.77 | 1.73 | 1.73 | 1.46 | 1.44 | 1.65 |
| g3-grounded-summ-bench | 1.20 | 1.20 | 1.20 | 1.06 | 1.05 | 1.16 |
| g3-grounded-summ-health | 1.08 | 1.10 | 1.07 | 1.06 | 1.03 | 1.09 |
| g3-grounded-summ-overview | 1.17 | 1.14 | 1.14 | 0.99 | 0.99 | 1.10 |
| g3-open-brainstorm-appnames | 1.12 | 1.04 | 1.04 | 1.05 | 1.07 | 1.02 |
| g3-open-brainstorm-lessons | 1.07 | 1.07 | 1.07 | 1.01 | 1.02 | 1.05 |
| g3-open-brainstorm-weekend | 1.12 | 1.09 | 1.09 | 1.03 | 1.13 | 1.07 |
| g3-open-explain-hashmap | 0.98 | 1.01 | 1.00 | 1.00 | 0.99 | 1.01 |
| g3-open-explain-kvcache | 1.01 | 1.06 | 1.06 | 0.99 | 1.02 | 1.03 |
| g3-open-explain-tides | 1.11 | 1.10 | 1.10 | 1.05 | 1.03 | 1.05 |
| g3-open-story-cartographer | 0.95 | 0.99 | 0.99 | 0.99 | 1.01 | 1.00 |
| g3-open-story-chef | 0.96 | 1.02 | 1.02 | 0.99 | 0.99 | 1.01 |
| g3-open-story-lighthouse | 0.97 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 |
| g3-open-story-robot | 0.95 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 |
| **g3 grounded mean** | 2.21 | 2.17 | 2.16 | 2.03 | 2.05 | 2.08 |
| **g3 open mean** | 1.024 | 1.039 | 1.038 | 1.011 | 1.023 | 1.025 |
| **g3 open min** | 0.948 | 0.992 | 0.995 | 0.993 | 0.992 | 0.998 |
| **sweep open-256 (story)** | 0.950 | 1.002 | 1.002 | 0.997 | 0.997 | 1.002 |
