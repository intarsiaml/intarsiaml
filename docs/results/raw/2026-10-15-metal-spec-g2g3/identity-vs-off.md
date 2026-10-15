| case | variant | stdout == off | ids == off | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-256 | off | ref | ref | length / 256 | 34.97 | - | - | - | - | - | - | - |
| 4k-256 | base | yes | yes | length / 256 | 33.81 | - | - | - | - | - | - | - |
| 4k-256 | on4 | yes | yes | length / 256 | 116.68 | 67 | 268 | 188 | 70.1% | 3.81 | no | 0 |
| 4k-256 | on6 | yes | yes | length / 256 | 112.02 | 55 | 330 | 200 | 60.6% | 4.64 | no | 0 |
| 4k-48 | off | ref | ref | length / 48 | 35.29 | - | - | - | - | - | - | - |
| 4k-48 | base | yes | yes | length / 48 | 33.72 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 111.75 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-48 | on6 | yes | yes | length / 48 | 109.83 | 11 | 66 | 38 | 57.6% | 4.45 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-max44 | off | ref | ref | length / 44 | 35.47 | - | - | - | - | - | - | - |
| 4k-max44 | base | yes | yes | length / 44 | 34.24 | - | - | - | - | - | - | - |
| 4k-max44 | on4 | yes | yes | length / 44 | 113.45 | 12 | 48 | 33 | 68.8% | 3.75 | yes (2 unemitted, KV 4107->4105) | 0 |
| 4k-max44 | on6 | yes | yes | length / 44 | 100.87 | 11 | 66 | 38 | 57.6% | 4.45 | yes (6 unemitted, KV 4111->4105) | 0 |
| 4k-max9 | off | ref | ref | length / 9 | 35.12 | - | - | - | - | - | - | - |
| 4k-max9 | base | yes | yes | length / 9 | 33.6 | - | - | - | - | - | - | - |
| 4k-max9 | on4 | yes | yes | length / 9 | 56.82 | 4 | 16 | 6 | 37.5% | 2.50 | yes (2 unemitted, KV 4072->4070) | 0 |
| 4k-max9 | on6 | yes | yes | length / 9 | 53.02 | 4 | 24 | 8 | 33.3% | 3.00 | yes (4 unemitted, KV 4074->4070) | 0 |
| 4k-stop250 | off | ref | ref | stop / 85 | 35.56 | - | - | - | - | - | - | - |
| 4k-stop250 | base | yes | yes | stop / 85 | 34.13 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 111.01 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | on6 | yes | yes | stop / 85 | 104.81 | 20 | 120 | 66 | 55.0% | 4.30 | yes (2 unemitted, KV 4148->4146) | 0 |
| golden-code-256 | off | ref | ref | length / 256 | 33.94 | - | - | - | - | - | - | - |
| golden-code-256 | base | yes | yes | length / 256 | 33.58 | - | - | - | - | - | - | - |
| golden-code-256 | on4 | yes | yes | length / 256 | 56.89 | 67 | 268 | 121 | 45.1% | 2.81 | yes (1 unemitted, KV 3852->3851) | 0 |
| golden-code-256 | on6 | yes | yes | length / 256 | 56.14 | 60 | 360 | 130 | 36.1% | 3.17 | yes (3 unemitted, KV 3854->3851) | 0 |
| golden-code-48 | off | ref | ref | length / 48 | 33.7 | - | - | - | - | - | - | - |
| golden-code-48 | base | yes | yes | length / 48 | 33.35 | - | - | - | - | - | - | - |
| golden-code-48 | on4 | yes | yes | length / 48 | 38.18 | 14 | 56 | 6 | 10.7% | 1.43 | no | 0 |
| golden-code-48 | on6 | yes | yes | length / 48 | 36.12 | 14 | 84 | 6 | 7.1% | 1.43 | no | 0 |
| golden-docs-256 | off | ref | ref | length / 256 | 34.28 | - | - | - | - | - | - | - |
| golden-docs-256 | base | yes | yes | length / 256 | 34.0 | - | - | - | - | - | - | - |
| golden-docs-256 | on4 | yes | yes | length / 256 | 38.55 | 55 | 220 | 31 | 14.1% | 1.56 | no | 0 |
| golden-docs-256 | on6 | yes | yes | length / 256 | 37.02 | 55 | 330 | 31 | 9.4% | 1.56 | no | 0 |
| golden-docs-48 | off | ref | ref | length / 48 | 35.38 | - | - | - | - | - | - | - |
| golden-docs-48 | base | yes | yes | length / 48 | 34.23 | - | - | - | - | - | - | - |
| golden-docs-48 | on4 | yes | yes | length / 48 | 38.15 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| golden-docs-48 | on6 | yes | yes | length / 48 | 36.8 | 12 | 72 | 6 | 8.3% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| grounded-256 | off | ref | ref | length / 256 | 35.76 | - | - | - | - | - | - | - |
| grounded-256 | base | yes | yes | length / 256 | 31.51 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 132.71 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-256 | on6 | yes | yes | length / 256 | 138.39 | 52 | 312 | 201 | 64.4% | 4.87 | no | 0 |
| grounded-512 | off | ref | ref | stop / 329 | 36.06 | - | - | - | - | - | - | - |
| grounded-512 | base | yes | yes | stop / 329 | 32.0 | - | - | - | - | - | - | - |
| grounded-512 | on4 | yes | yes | stop / 329 | 127.14 | 82 | 328 | 242 | 73.8% | 3.95 | no | 0 |
| grounded-512 | on6 | yes | yes | stop / 329 | 122.54 | 69 | 414 | 255 | 61.6% | 4.70 | no | 0 |
| grounded-max102 | off | ref | ref | length / 102 | 36.22 | - | - | - | - | - | - | - |
| grounded-max102 | base | yes | yes | length / 102 | 35.32 | - | - | - | - | - | - | - |
| grounded-max102 | on4 | yes | yes | length / 102 | 132.48 | 24 | 96 | 77 | 80.2% | 4.21 | yes (2 unemitted, KV 474->472) | 0 |
| grounded-max102 | on6 | yes | yes | length / 102 | 143.36 | 20 | 120 | 81 | 67.5% | 5.05 | yes (2 unemitted, KV 474->472) | 0 |
| grounded-max188 | off | ref | ref | length / 188 | 35.62 | - | - | - | - | - | - | - |
| grounded-max188 | base | yes | yes | length / 188 | 35.34 | - | - | - | - | - | - | - |
| grounded-max188 | on4 | yes | yes | length / 188 | 130.67 | 47 | 188 | 140 | 74.5% | 3.98 | yes (2 unemitted, KV 560->558) | 0 |
| grounded-max188 | on6 | yes | yes | length / 188 | 136.83 | 39 | 234 | 151 | 64.5% | 4.87 | yes (5 unemitted, KV 563->558) | 0 |
| grounded-min320-stopfence | off | ref | ref | stop / 328 | 36.25 | - | - | - | - | - | - | - |
| grounded-min320-stopfence | base | yes | yes | stop / 328 | 35.23 | - | - | - | - | - | - | - |
| grounded-min320-stopfence | on4 | yes | yes | stop / 328 | 131.12 | 82 | 328 | 242 | 73.8% | 3.95 | no | 0 |
| grounded-min320-stopfence | on6 | yes | yes | stop / 328 | 138.15 | 69 | 414 | 255 | 61.6% | 4.70 | no | 0 |
| grounded-min331 | off | ref | ref | length / 400 | 36.11 | - | - | - | - | - | - | - |
| grounded-min331 | base | yes | yes | length / 400 | 34.01 | - | - | - | - | - | - | - |
| grounded-min331 | on4 | yes | yes | length / 400 | 128.64 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min331 | on6 | yes | yes | length / 400 | 133.48 | 83 | 498 | 312 | 62.7% | 4.76 | yes (1 unemitted, KV 771->770) | 0 |
| grounded-min335 | off | ref | ref | length / 400 | 36.37 | - | - | - | - | - | - | - |
| grounded-min335 | base | yes | yes | length / 400 | 34.96 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 132.77 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min335 | on6 | yes | yes | length / 400 | 132.02 | 83 | 498 | 312 | 62.7% | 4.76 | yes (1 unemitted, KV 771->770) | 0 |
| grounded-stop-atomic | off | ref | ref | stop / 224 | 35.96 | - | - | - | - | - | - | - |
| grounded-stop-atomic | base | yes | yes | stop / 224 | 34.81 | - | - | - | - | - | - | - |
| grounded-stop-atomic | on4 | yes | yes | stop / 224 | 133.13 | 55 | 220 | 168 | 76.4% | 4.05 | yes (2 unemitted, KV 596->594) | 0 |
| grounded-stop-atomic | on6 | yes | yes | stop / 224 | 139.66 | 44 | 264 | 181 | 68.6% | 5.11 | yes (4 unemitted, KV 598->594) | 0 |
| grounded-stop-topp | off | ref | ref | stop / 115 | 36.7 | - | - | - | - | - | - | - |
| grounded-stop-topp | base | yes | yes | stop / 115 | 35.94 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 124.47 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | on6 | yes | yes | stop / 115 | 131.25 | 25 | 150 | 93 | 62.0% | 4.72 | yes (6 unemitted, KV 491->485) | 0 |
| grounded-stop-vecstring | off | ref | ref | stop / 186 | 35.9 | - | - | - | - | - | - | - |
| grounded-stop-vecstring | base | yes | yes | stop / 186 | 35.58 | - | - | - | - | - | - | - |
| grounded-stop-vecstring | on4 | yes | yes | stop / 186 | 130.82 | 47 | 188 | 140 | 74.5% | 3.98 | yes (4 unemitted, KV 560->556) | 0 |
| grounded-stop-vecstring | on6 | yes | yes | stop / 186 | 141.88 | 38 | 228 | 145 | 63.6% | 4.82 | no | 0 |
| open-256 | off | ref | ref | length / 256 | 36.44 | - | - | - | - | - | - | - |
| open-256 | base | yes | yes | length / 256 | 31.67 | - | - | - | - | - | - | - |
| open-256 | on4 | yes | yes | length / 256 | 37.37 | 27 | 107 | 7 | 6.5% | 1.26 | no | 0 |
| open-256 | on6 | yes | yes | length / 256 | 36.89 | 27 | 159 | 7 | 4.4% | 1.26 | no | 0 |
| repeat-eos | off | ref | ref | stop / 16 | 36.7 | - | - | - | - | - | - | - |
| repeat-eos | base | yes | yes | stop / 16 | 34.58 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 108.83 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| repeat-eos | on6 | yes | yes | stop / 16 | 131.75 | 2 | 12 | 12 | 100.0% | 7.00 | yes (1 unemitted, KV 51->50) | 0 |
| short-256 | off | ref | ref | length / 256 | 36.06 | - | - | - | - | - | - | - |
| short-256 | base | yes | yes | length / 256 | 35.62 | - | - | - | - | - | - | - |
| short-256 | on4 | yes | yes | length / 256 | 81.05 | 87 | 348 | 144 | 41.4% | 2.66 | no | 0 |
| short-256 | on6 | yes | yes | length / 256 | 72.24 | 82 | 492 | 150 | 30.5% | 2.83 | yes (1 unemitted, KV 378->377) | 0 |
| short-48 | off | ref | ref | length / 48 | 35.93 | - | - | - | - | - | - | - |
| short-48 | base | yes | yes | length / 48 | 35.8 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 70.09 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |
| short-48 | on6 | yes | yes | length / 48 | 73.31 | 14 | 84 | 25 | 29.8% | 2.79 | no | 0 |
| short-stopkettles | off | ref | ref | stop / 57 | 36.82 | - | - | - | - | - | - | - |
| short-stopkettles | base | yes | yes | stop / 57 | 35.06 | - | - | - | - | - | - | - |
| short-stopkettles | on4 | yes | yes | stop / 57 | 72.96 | 20 | 80 | 32 | 40.0% | 2.60 | yes (4 unemitted, KV 182->178) | 0 |
| short-stopkettles | on6 | yes | yes | stop / 57 | 76.59 | 17 | 102 | 32 | 31.4% | 2.88 | yes (1 unemitted, KV 179->178) | 0 |

ALL IDENTICAL
