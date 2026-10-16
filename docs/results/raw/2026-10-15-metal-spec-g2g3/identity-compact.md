| case | finish / tokens | off, on4, on6 vs base: stdout and ids | on4: passes / accepted / proposed | on6: passes / accepted / proposed | ended inside an accepted run (on4 / on6) | decode tok/s base / off / on4 / on6 |
|---|---|---|---|---|---|---|
| 4k-256 | length / 256 | identical | 67 / 188 / 268 (70.1%) | 55 / 200 / 330 (60.6%) | no / no | 33.81 / 34.97 / 116.68 / 112.02 |
| 4k-48 | length / 48 | identical | 13 / 36 / 52 (69.2%) | 11 / 38 / 66 (57.6%) | yes / yes | 33.72 / 35.29 / 111.75 / 109.83 |
| 4k-max44 | length / 44 | identical | 12 / 33 / 48 (68.8%) | 11 / 38 / 66 (57.6%) | yes / yes | 34.24 / 35.47 / 113.45 / 100.87 |
| 4k-max9 | length / 9 | identical | 4 / 6 / 16 (37.5%) | 4 / 8 / 24 (33.3%) | yes / yes | 33.6 / 35.12 / 56.82 / 53.02 |
| 4k-stop250 | stop / 85 | identical | 24 / 62 / 96 (64.6%) | 20 / 66 / 120 (55.0%) | yes / yes | 34.13 / 35.56 / 111.01 / 104.81 |
| golden-code-256 | length / 256 | identical | 67 / 121 / 268 (45.1%) | 60 / 130 / 360 (36.1%) | yes / yes | 33.58 / 33.94 / 56.89 / 56.14 |
| golden-code-48 | length / 48 | identical | 14 / 6 / 56 (10.7%) | 14 / 6 / 84 (7.1%) | no / no | 33.35 / 33.7 / 38.18 / 36.12 |
| golden-docs-256 | length / 256 | identical | 55 / 31 / 220 (14.1%) | 55 / 31 / 330 (9.4%) | no / no | 34.0 / 34.28 / 38.55 / 37.02 |
| golden-docs-48 | length / 48 | identical | 12 / 6 / 48 (12.5%) | 12 / 6 / 72 (8.3%) | yes / yes | 34.23 / 35.38 / 38.15 / 36.8 |
| grounded-256 | length / 256 | identical | 64 / 189 / 256 (73.8%) | 52 / 201 / 312 (64.4%) | no / no | 31.51 / 35.76 / 132.71 / 138.39 |
| grounded-512 | stop / 329 | identical | 82 / 242 / 328 (73.8%) | 69 / 255 / 414 (61.6%) | no / no | 32.0 / 36.06 / 127.14 / 122.54 |
| grounded-max102 | length / 102 | identical | 24 / 77 / 96 (80.2%) | 20 / 81 / 120 (67.5%) | yes / yes | 35.32 / 36.22 / 132.48 / 143.36 |
| grounded-max188 | length / 188 | identical | 47 / 140 / 188 (74.5%) | 39 / 151 / 234 (64.5%) | yes / yes | 35.34 / 35.62 / 130.67 / 136.83 |
| grounded-min320-stopfence | stop / 328 | identical | 82 / 242 / 328 (73.8%) | 69 / 255 / 414 (61.6%) | no / no | 35.23 / 36.25 / 131.12 / 138.15 |
| grounded-min331 | length / 400 | identical | 97 / 297 / 388 (76.5%) | 83 / 312 / 498 (62.7%) | no / yes | 34.01 / 36.11 / 128.64 / 133.48 |
| grounded-min335 | length / 400 | identical | 97 / 297 / 388 (76.5%) | 83 / 312 / 498 (62.7%) | no / yes | 34.96 / 36.37 / 132.77 / 132.02 |
| grounded-stop-atomic | stop / 224 | identical | 55 / 168 / 220 (76.4%) | 44 / 181 / 264 (68.6%) | yes / yes | 34.81 / 35.96 / 133.13 / 139.66 |
| grounded-stop-topp | stop / 115 | identical | 30 / 86 / 120 (71.7%) | 25 / 93 / 150 (62.0%) | yes / yes | 35.94 / 36.7 / 124.47 / 131.25 |
| grounded-stop-vecstring | stop / 186 | identical | 47 / 140 / 188 (74.5%) | 38 / 145 / 228 (63.6%) | yes / no | 35.58 / 35.9 / 130.82 / 141.88 |
| open-256 | length / 256 | identical | 27 / 7 / 107 (6.5%) | 27 / 7 / 159 (4.4%) | no / no | 31.67 / 36.44 / 37.37 / 36.89 |
| repeat-eos | stop / 16 | identical | 3 / 11 / 12 (91.7%) | 2 / 12 / 12 (100.0%) | yes / yes | 34.58 / 36.7 / 108.83 / 131.75 |
| short-256 | length / 256 | identical | 87 / 144 / 348 (41.4%) | 82 / 150 / 492 (30.5%) | no / yes | 35.62 / 36.06 / 81.05 / 72.24 |
| short-48 | length / 48 | identical | 16 / 23 / 64 (35.9%) | 14 / 25 / 84 (29.8%) | no / no | 35.8 / 35.93 / 70.09 / 73.31 |
| short-stopkettles | stop / 57 | identical | 20 / 32 / 80 (40.0%) | 17 / 32 / 102 (31.4%) | yes / yes | 35.06 / 36.82 / 72.96 / 76.59 |
