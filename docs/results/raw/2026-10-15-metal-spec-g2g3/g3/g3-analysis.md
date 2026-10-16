| row | cat | prompt tok | gen tok | off tok/s (3 reps) | on tok/s (3 reps) | on/off (median) | on/off (pooled ms) | passes | proposed | accepted | acceptance | ids identical |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grounded-edit-docs-comments | grounded | 855 | 256 | 35.9 (36.4/35.9/35.6) | 82.0 (82.0/77.8/82.7) | **2.29x** | 2.25x | 59 | 236 | 148 | 63% | yes |
| grounded-edit-error-msgs | grounded | 1161 | 256 | 35.2 (35.2/33.9/35.9) | 137.6 (137.9/135.7/137.6) | **3.91x** | 3.92x | 56 | 224 | 192 | 86% | yes |
| grounded-edit-field | grounded | 847 | 256 | 36.2 (35.9/36.2/36.3) | 124.0 (121.5/124.0/127.0) | **3.42x** | 3.43x | 68 | 272 | 188 | 69% | yes |
| grounded-edit-refactor-model | grounded | 1014 | 256 | 36.2 (36.3/36.1/36.2) | 126.0 (126.0/121.8/126.7) | **3.48x** | 3.45x | 69 | 276 | 188 | 68% | yes |
| grounded-edit-rename-fn | grounded | 1038 | 256 | 36.3 (36.3/35.7/36.3) | 145.5 (146.8/145.2/145.5) | **4.01x** | 4.04x | 57 | 228 | 198 | 87% | yes |
| grounded-extract-fields | grounded | 930 | 256 | 36.0 (36.3/36.0/35.1) | 74.7 (74.7/74.0/76.0) | **2.07x** | 2.09x | 62 | 248 | 137 | 55% | yes |
| grounded-extract-ops | grounded | 2263 | 256 | 35.1 (34.9/35.5/35.1) | 66.5 (66.5/66.0/68.0) | **1.89x** | 1.90x | 72 | 288 | 125 | 43% | yes |
| grounded-summ-bench | grounded | 1388 | 256 | 36.0 (36.3/35.6/36.0) | 44.5 (44.5/43.9/45.2) | **1.23x** | 1.24x | 48 | 192 | 54 | 28% | yes |
| grounded-summ-health | grounded | 1042 | 256 | 36.0 (35.7/36.0/36.5) | 39.7 (39.6/39.7/40.2) | **1.10x** | 1.11x | 46 | 184 | 30 | 16% | yes |
| grounded-summ-overview | grounded | 586 | 256 | 36.4 (36.4/36.4/36.0) | 43.4 (43.4/43.4/43.3) | **1.19x** | 1.20x | 41 | 163 | 41 | 25% | yes |
| open-brainstorm-appnames | open | 35 | 256 | 36.2 (36.2/36.2/36.7) | 39.3 (39.3/39.2/39.8) | **1.09x** | 1.08x | 41 | 164 | 22 | 13% | yes |
| open-brainstorm-lessons | open | 27 | 256 | 35.7 (35.7/36.2/35.4) | 39.2 (39.2/39.9/39.1) | **1.10x** | 1.10x | 24 | 96 | 22 | 23% | yes |
| open-brainstorm-weekend | open | 34 | 256 | 36.8 (36.2/36.8/36.8) | 41.2 (41.3/41.1/41.2) | **1.12x** | 1.13x | 31 | 124 | 29 | 23% | yes |
| open-explain-hashmap | open | 34 | 256 | 36.8 (36.5/36.8/36.8) | 37.1 (37.0/37.1/37.1) | **1.01x** | 1.01x | 13 | 52 | 2 | 4% | yes |
| open-explain-kvcache | open | 32 | 256 | 36.9 (36.9/36.8/36.9) | 39.6 (39.6/39.4/39.6) | **1.07x** | 1.07x | 18 | 72 | 18 | 25% | yes |
| open-explain-tides | open | 35 | 256 | 36.8 (36.8/36.8/36.6) | 42.2 (42.2/42.2/42.2) | **1.15x** | 1.15x | 39 | 156 | 33 | 21% | yes |
| open-story-cartographer | open | 32 | 256 | 36.9 (36.9/36.9/36.9) | 37.2 (37.3/37.2/37.2) | **1.01x** | 1.01x | 18 | 72 | 3 | 4% | yes |
| open-story-chef | open | 32 | 256 | 36.8 (36.8/36.8/36.8) | 38.3 (38.3/38.3/38.3) | **1.04x** | 1.04x | 16 | 64 | 10 | 16% | yes |
| open-story-lighthouse | open | 31 | 256 | 36.8 (36.8/36.6/36.9) | 37.7 (37.7/37.7/37.7) | **1.02x** | 1.02x | 21 | 84 | 6 | 7% | yes |
| open-story-robot | open | 34 | 256 | 36.9 (36.9/36.5/36.9) | 37.5 (37.5/37.5/37.3) | **1.02x** | 1.02x | 20 | 79 | 5 | 6% | yes |

| category | rows | mean of ratios | geomean | median | min | max | rows >= target |
|---|---|---|---|---|---|---|---|
| grounded | 10 | 2.460x | 2.205x | 2.179x | 1.10x | 4.01x | 7/10 (target 1.3x) |
| open | 10 | 1.062x | 1.061x | 1.058x | 1.01x | 1.15x | 10/10 (target 0.97x) |

Identical ids on every row (all reps, both modes): YES
