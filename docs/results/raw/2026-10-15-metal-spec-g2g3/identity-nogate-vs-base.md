| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-max44 | base | ref | ref | length / 44 | 34.24 | - | - | - | - | - | - | - |
| 4k-max44 | on4 | yes | yes | length / 44 | 111.37 | 12 | 48 | 33 | 68.8% | 3.75 | yes (2 unemitted, KV 4107->4105) | 0 |
| 4k-max44 | on6 | yes | yes | length / 44 | 102.09 | 11 | 66 | 38 | 57.6% | 4.45 | yes (6 unemitted, KV 4111->4105) | 0 |
| 4k-max9 | base | ref | ref | length / 9 | 33.6 | - | - | - | - | - | - | - |
| 4k-max9 | on4 | yes | yes | length / 9 | 58.2 | 4 | 16 | 6 | 37.5% | 2.50 | yes (2 unemitted, KV 4072->4070) | 0 |
| 4k-max9 | on6 | yes | yes | length / 9 | 50.52 | 4 | 24 | 8 | 33.3% | 3.00 | yes (4 unemitted, KV 4074->4070) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 34.13 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 107.09 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | on6 | yes | yes | stop / 85 | 109.81 | 20 | 120 | 66 | 55.0% | 4.30 | yes (2 unemitted, KV 4148->4146) | 0 |
| grounded-max102 | base | ref | ref | length / 102 | 35.32 | - | - | - | - | - | - | - |
| grounded-max102 | on4 | yes | yes | length / 102 | 141.0 | 25 | 100 | 78 | 78.0% | 4.12 | yes (2 unemitted, KV 474->472) | 0 |
| grounded-max102 | on6 | yes | yes | length / 102 | 143.2 | 21 | 126 | 82 | 65.1% | 4.90 | yes (2 unemitted, KV 474->472) | 0 |
| grounded-max188 | base | ref | ref | length / 188 | 35.34 | - | - | - | - | - | - | - |
| grounded-max188 | on4 | yes | yes | length / 188 | 131.84 | 48 | 192 | 141 | 73.4% | 3.94 | yes (2 unemitted, KV 560->558) | 0 |
| grounded-max188 | on6 | yes | yes | length / 188 | 116.76 | 40 | 240 | 152 | 63.3% | 4.80 | yes (5 unemitted, KV 563->558) | 0 |
| grounded-min320-stopfence | base | ref | ref | stop / 328 | 35.23 | - | - | - | - | - | - | - |
| grounded-min320-stopfence | on4 | yes | yes | stop / 328 | 131.58 | 83 | 332 | 243 | 73.2% | 3.93 | no | 0 |
| grounded-min320-stopfence | on6 | yes | yes | stop / 328 | 124.68 | 70 | 420 | 256 | 61.0% | 4.66 | no | 0 |
| grounded-min331 | base | ref | ref | length / 400 | 34.01 | - | - | - | - | - | - | - |
| grounded-min331 | on4 | yes | yes | length / 400 | 132.43 | 100 | 400 | 298 | 74.5% | 3.98 | no | 1 |
| grounded-min331 | on6 | yes | yes | length / 400 | 117.16 | 85 | 510 | 314 | 61.6% | 4.69 | yes (1 unemitted, KV 771->770) | 1 |
| grounded-min335 | base | ref | ref | length / 400 | 34.96 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 132.64 | 100 | 400 | 298 | 74.5% | 3.98 | no | 1 |
| grounded-min335 | on6 | yes | yes | length / 400 | 117.03 | 85 | 510 | 314 | 61.6% | 4.69 | yes (1 unemitted, KV 771->770) | 1 |
| grounded-stop-atomic | base | ref | ref | stop / 224 | 34.81 | - | - | - | - | - | - | - |
| grounded-stop-atomic | on4 | yes | yes | stop / 224 | 129.97 | 56 | 224 | 169 | 75.4% | 4.02 | yes (2 unemitted, KV 596->594) | 0 |
| grounded-stop-atomic | on6 | yes | yes | stop / 224 | 140.2 | 45 | 270 | 182 | 67.4% | 5.04 | yes (4 unemitted, KV 598->594) | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 35.94 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 125.47 | 31 | 124 | 87 | 70.2% | 3.81 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | on6 | yes | yes | stop / 115 | 130.99 | 26 | 156 | 94 | 60.3% | 4.62 | yes (6 unemitted, KV 491->485) | 0 |
| grounded-stop-vecstring | base | ref | ref | stop / 186 | 35.58 | - | - | - | - | - | - | - |
| grounded-stop-vecstring | on4 | yes | yes | stop / 186 | 135.79 | 48 | 192 | 141 | 73.4% | 3.94 | yes (4 unemitted, KV 560->556) | 0 |
| grounded-stop-vecstring | on6 | yes | yes | stop / 186 | 132.48 | 39 | 234 | 146 | 62.4% | 4.74 | no | 0 |
| repeat-eos | base | ref | ref | stop / 16 | 34.58 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 137.43 | 3 | 12 | 12 | 100.0% | 5.00 | yes (1 unemitted, KV 51->50) | 0 |
| repeat-eos | on6 | yes | yes | stop / 16 | 174.46 | 2 | 12 | 12 | 100.0% | 7.00 | no | 0 |
| short-stopkettles | base | ref | ref | stop / 57 | 35.06 | - | - | - | - | - | - | - |
| short-stopkettles | on4 | yes | yes | stop / 57 | 84.64 | 22 | 88 | 36 | 40.9% | 2.64 | yes (4 unemitted, KV 182->178) | 0 |
| short-stopkettles | on6 | yes | yes | stop / 57 | 89.61 | 19 | 114 | 36 | 31.6% | 2.89 | yes (1 unemitted, KV 179->178) | 0 |

ALL IDENTICAL
