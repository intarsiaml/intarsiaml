| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-256 | base | ref | ref | length / 256 | 33.81 | - | - | - | - | - | - | - |
| 4k-256 | off | yes | yes | length / 256 | 33.71 | - | - | - | - | - | - | - |
| 4k-256 | on4 | yes | yes | length / 256 | 114.32 | 67 | 268 | 188 | 70.1% | 3.81 | no | 0 |
| 4k-256 | on6 | yes | yes | length / 256 | 99.75 | 55 | 330 | 200 | 60.6% | 4.64 | no | 0 |
| 4k-48 | base | ref | ref | length / 48 | 33.72 | - | - | - | - | - | - | - |
| 4k-48 | off | yes | yes | length / 48 | 33.95 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 115.81 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-48 | on6 | yes | yes | length / 48 | 121.43 | 11 | 66 | 38 | 57.6% | 4.45 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-max44 | base | ref | ref | length / 44 | 34.24 | - | - | - | - | - | - | - |
| 4k-max44 | off | yes | yes | length / 44 | 34.36 | - | - | - | - | - | - | - |
| 4k-max44 | on4 | yes | yes | length / 44 | 97.27 | 12 | 48 | 33 | 68.8% | 3.75 | yes (2 unemitted, KV 4107->4105) | 0 |
| 4k-max44 | on6 | yes | yes | length / 44 | 95.34 | 11 | 66 | 38 | 57.6% | 4.45 | yes (6 unemitted, KV 4111->4105) | 0 |
| 4k-max9 | base | ref | ref | length / 9 | 33.6 | - | - | - | - | - | - | - |
| 4k-max9 | off | yes | yes | length / 9 | 34.28 | - | - | - | - | - | - | - |
| 4k-max9 | on4 | yes | yes | length / 9 | 56.4 | 4 | 16 | 6 | 37.5% | 2.50 | yes (2 unemitted, KV 4072->4070) | 0 |
| 4k-max9 | on6 | yes | yes | length / 9 | 45.92 | 4 | 24 | 8 | 33.3% | 3.00 | yes (4 unemitted, KV 4074->4070) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 34.13 | - | - | - | - | - | - | - |
| 4k-stop250 | off | yes | yes | stop / 85 | 34.1 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 101.55 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | on6 | yes | yes | stop / 85 | 95.0 | 20 | 120 | 66 | 55.0% | 4.30 | yes (2 unemitted, KV 4148->4146) | 0 |
| grounded-256 | base | ref | ref | length / 256 | 31.51 | - | - | - | - | - | - | - |
| grounded-256 | off | yes | yes | length / 256 | 34.59 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 105.82 | 65 | 260 | 190 | 73.1% | 3.92 | no | 0 |
| grounded-256 | on6 | yes | yes | length / 256 | 104.09 | 53 | 318 | 202 | 63.5% | 4.81 | no | 0 |
| grounded-512 | base | ref | ref | stop / 329 | 32.0 | - | - | - | - | - | - | - |
| grounded-512 | off | yes | yes | stop / 329 | 34.0 | - | - | - | - | - | - | - |
| grounded-512 | on4 | yes | yes | stop / 329 | 100.47 | 84 | 336 | 243 | 72.3% | 3.89 | no | 0 |
| grounded-512 | on6 | yes | yes | stop / 329 | 95.2 | 71 | 426 | 256 | 60.1% | 4.61 | no | 0 |
| grounded-max102 | base | ref | ref | length / 102 | 35.32 | - | - | - | - | - | - | - |
| grounded-max102 | off | yes | yes | length / 102 | 35.97 | - | - | - | - | - | - | - |
| grounded-max102 | on4 | yes | yes | length / 102 | 122.73 | 25 | 100 | 78 | 78.0% | 4.12 | yes (2 unemitted, KV 474->472) | 0 |
| grounded-max102 | on6 | yes | yes | length / 102 | 118.35 | 21 | 126 | 82 | 65.1% | 4.90 | yes (2 unemitted, KV 474->472) | 0 |
| grounded-max188 | base | ref | ref | length / 188 | 35.34 | - | - | - | - | - | - | - |
| grounded-max188 | off | yes | yes | length / 188 | 35.26 | - | - | - | - | - | - | - |
| grounded-max188 | on4 | yes | yes | length / 188 | 114.18 | 48 | 192 | 141 | 73.4% | 3.94 | yes (2 unemitted, KV 560->558) | 0 |
| grounded-max188 | on6 | yes | yes | length / 188 | 100.09 | 40 | 240 | 152 | 63.3% | 4.80 | yes (5 unemitted, KV 563->558) | 0 |
| grounded-min320-stopfence | base | ref | ref | stop / 328 | 35.23 | - | - | - | - | - | - | - |
| grounded-min320-stopfence | off | yes | yes | stop / 328 | 34.81 | - | - | - | - | - | - | - |
| grounded-min320-stopfence | on4 | yes | yes | stop / 328 | 115.33 | 83 | 332 | 243 | 73.2% | 3.93 | no | 0 |
| grounded-min320-stopfence | on6 | yes | yes | stop / 328 | 99.09 | 70 | 420 | 256 | 61.0% | 4.66 | no | 0 |
| grounded-min331 | base | ref | ref | length / 400 | 34.01 | - | - | - | - | - | - | - |
| grounded-min331 | off | yes | yes | length / 400 | 34.85 | - | - | - | - | - | - | - |
| grounded-min331 | on4 | yes | yes | length / 400 | 118.39 | 100 | 400 | 298 | 74.5% | 3.98 | no | 1 |
| grounded-min331 | on6 | yes | yes | length / 400 | 112.38 | 85 | 510 | 314 | 61.6% | 4.69 | yes (1 unemitted, KV 771->770) | 1 |
| grounded-min335 | base | ref | ref | length / 400 | 34.96 | - | - | - | - | - | - | - |
| grounded-min335 | off | yes | yes | length / 400 | 35.51 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 113.63 | 100 | 400 | 298 | 74.5% | 3.98 | no | 1 |
| grounded-min335 | on6 | yes | yes | length / 400 | 95.32 | 85 | 510 | 314 | 61.6% | 4.69 | yes (1 unemitted, KV 771->770) | 1 |
| grounded-stop-atomic | base | ref | ref | stop / 224 | 34.81 | - | - | - | - | - | - | - |
| grounded-stop-atomic | off | yes | yes | stop / 224 | 34.41 | - | - | - | - | - | - | - |
| grounded-stop-atomic | on4 | yes | yes | stop / 224 | 116.05 | 56 | 224 | 169 | 75.4% | 4.02 | yes (2 unemitted, KV 596->594) | 0 |
| grounded-stop-atomic | on6 | yes | yes | stop / 224 | 105.15 | 45 | 270 | 182 | 67.4% | 5.04 | yes (4 unemitted, KV 598->594) | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 35.94 | - | - | - | - | - | - | - |
| grounded-stop-topp | off | yes | yes | stop / 115 | 35.08 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 108.59 | 31 | 124 | 87 | 70.2% | 3.81 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | on6 | yes | yes | stop / 115 | 97.55 | 26 | 156 | 94 | 60.3% | 4.62 | yes (6 unemitted, KV 491->485) | 0 |
| grounded-stop-vecstring | base | ref | ref | stop / 186 | 35.58 | - | - | - | - | - | - | - |
| grounded-stop-vecstring | off | yes | yes | stop / 186 | 35.03 | - | - | - | - | - | - | - |
| grounded-stop-vecstring | on4 | yes | yes | stop / 186 | 111.62 | 48 | 192 | 141 | 73.4% | 3.94 | yes (4 unemitted, KV 560->556) | 0 |
| grounded-stop-vecstring | on6 | yes | yes | stop / 186 | 108.64 | 39 | 234 | 146 | 62.4% | 4.74 | no | 0 |
| open-256 | base | ref | ref | length / 256 | 31.67 | - | - | - | - | - | - | - |
| open-256 | off | yes | yes | length / 256 | 34.79 | - | - | - | - | - | - | - |
| open-256 | on4 | yes | yes | length / 256 | 33.03 | 106 | 423 | 12 | 2.8% | 1.11 | no | 0 |
| open-256 | on6 | yes | yes | length / 256 | 30.79 | 105 | 627 | 13 | 2.1% | 1.12 | no | 0 |
| short-256 | base | ref | ref | length / 256 | 35.62 | - | - | - | - | - | - | - |
| short-256 | off | yes | yes | length / 256 | 35.82 | - | - | - | - | - | - | - |
| short-256 | on4 | yes | yes | length / 256 | 89.88 | 95 | 380 | 158 | 41.6% | 2.66 | no | 0 |
| short-256 | on6 | yes | yes | length / 256 | 84.7 | 90 | 540 | 164 | 30.4% | 2.82 | yes (1 unemitted, KV 378->377) | 0 |
| short-48 | base | ref | ref | length / 48 | 35.8 | - | - | - | - | - | - | - |
| short-48 | off | yes | yes | length / 48 | 35.54 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 81.62 | 18 | 72 | 27 | 37.5% | 2.50 | no | 0 |
| short-48 | on6 | yes | yes | length / 48 | 87.06 | 16 | 96 | 29 | 30.2% | 2.81 | no | 0 |
| short-stopkettles | base | ref | ref | stop / 57 | 35.06 | - | - | - | - | - | - | - |
| short-stopkettles | off | yes | yes | stop / 57 | 35.63 | - | - | - | - | - | - | - |
| short-stopkettles | on4 | yes | yes | stop / 57 | 79.72 | 22 | 88 | 36 | 40.9% | 2.64 | yes (4 unemitted, KV 182->178) | 0 |
| short-stopkettles | on6 | yes | yes | stop / 57 | 76.18 | 19 | 114 | 36 | 31.6% | 2.89 | yes (1 unemitted, KV 179->178) | 0 |

ALL IDENTICAL
