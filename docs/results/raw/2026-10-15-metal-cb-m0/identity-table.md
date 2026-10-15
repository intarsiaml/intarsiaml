| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-256 | base | ref | ref | length / 256 | 34.61 | - | - | - | - | - | - | - |
| 4k-256 | off | yes | yes | length / 256 | 34.7 | - | - | - | - | - | - | - |
| 4k-256 | on4 | yes | yes | length / 256 | 115.71 | 67 | 268 | 188 | 70.1% | 3.81 | no | 0 |
| 4k-48 | base | ref | ref | length / 48 | 35.27 | - | - | - | - | - | - | - |
| 4k-48 | off | yes | yes | length / 48 | 35.39 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 113.69 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-max44 | base | ref | ref | length / 44 | 35.37 | - | - | - | - | - | - | - |
| 4k-max44 | off | yes | yes | length / 44 | 34.97 | - | - | - | - | - | - | - |
| 4k-max44 | on4 | yes | yes | length / 44 | 106.9 | 12 | 48 | 33 | 68.8% | 3.75 | yes (2 unemitted, KV 4107->4105) | 0 |
| 4k-max9 | base | ref | ref | length / 9 | 33.7 | - | - | - | - | - | - | - |
| 4k-max9 | off | yes | yes | length / 9 | 35.11 | - | - | - | - | - | - | - |
| 4k-max9 | on4 | yes | yes | length / 9 | 61.75 | 4 | 16 | 6 | 37.5% | 2.50 | yes (2 unemitted, KV 4072->4070) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 33.83 | - | - | - | - | - | - | - |
| 4k-stop250 | off | yes | yes | stop / 85 | 35.3 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 108.52 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| golden-code-256 | base | ref | ref | length / 256 | 35.37 | - | - | - | - | - | - | - |
| golden-code-256 | off | yes | yes | length / 256 | 34.84 | - | - | - | - | - | - | - |
| golden-code-256 | on4 | yes | yes | length / 256 | 62.36 | 67 | 268 | 121 | 45.1% | 2.81 | yes (1 unemitted, KV 3852->3851) | 0 |
| golden-code-48 | base | ref | ref | length / 48 | 35.36 | - | - | - | - | - | - | - |
| golden-code-48 | off | yes | yes | length / 48 | 34.81 | - | - | - | - | - | - | - |
| golden-code-48 | on4 | yes | yes | length / 48 | 38.93 | 14 | 56 | 6 | 10.7% | 1.43 | no | 0 |
| golden-docs-256 | base | ref | ref | length / 256 | 35.28 | - | - | - | - | - | - | - |
| golden-docs-256 | off | yes | yes | length / 256 | 34.78 | - | - | - | - | - | - | - |
| golden-docs-256 | on4 | yes | yes | length / 256 | 39.01 | 55 | 220 | 31 | 14.1% | 1.56 | no | 0 |
| golden-docs-48 | base | ref | ref | length / 48 | 35.18 | - | - | - | - | - | - | - |
| golden-docs-48 | off | yes | yes | length / 48 | 34.83 | - | - | - | - | - | - | - |
| golden-docs-48 | on4 | yes | yes | length / 48 | 38.11 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| grounded-256 | base | ref | ref | length / 256 | 35.9 | - | - | - | - | - | - | - |
| grounded-256 | off | yes | yes | length / 256 | 36.6 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 135.45 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-512 | base | ref | ref | stop / 329 | 35.7 | - | - | - | - | - | - | - |
| grounded-512 | off | yes | yes | stop / 329 | 35.84 | - | - | - | - | - | - | - |
| grounded-512 | on4 | yes | yes | stop / 329 | 134.78 | 82 | 328 | 242 | 73.8% | 3.95 | no | 0 |
| grounded-max102 | base | ref | ref | length / 102 | 34.98 | - | - | - | - | - | - | - |
| grounded-max102 | off | yes | yes | length / 102 | 35.53 | - | - | - | - | - | - | - |
| grounded-max102 | on4 | yes | yes | length / 102 | 138.2 | 24 | 96 | 77 | 80.2% | 4.21 | yes (2 unemitted, KV 474->472) | 0 |
| grounded-max188 | base | ref | ref | length / 188 | 36.23 | - | - | - | - | - | - | - |
| grounded-max188 | off | yes | yes | length / 188 | 35.77 | - | - | - | - | - | - | - |
| grounded-max188 | on4 | yes | yes | length / 188 | 133.28 | 47 | 188 | 140 | 74.5% | 3.98 | yes (2 unemitted, KV 560->558) | 0 |
| grounded-min320-stopfence | base | ref | ref | stop / 328 | 36.09 | - | - | - | - | - | - | - |
| grounded-min320-stopfence | off | yes | yes | stop / 328 | 34.85 | - | - | - | - | - | - | - |
| grounded-min320-stopfence | on4 | yes | yes | stop / 328 | 124.96 | 82 | 328 | 242 | 73.8% | 3.95 | no | 0 |
| grounded-min331 | base | ref | ref | length / 400 | 35.97 | - | - | - | - | - | - | - |
| grounded-min331 | off | yes | yes | length / 400 | 36.23 | - | - | - | - | - | - | - |
| grounded-min331 | on4 | yes | yes | length / 400 | 138.44 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min335 | base | ref | ref | length / 400 | 36.27 | - | - | - | - | - | - | - |
| grounded-min335 | off | yes | yes | length / 400 | 36.12 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 135.18 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-stop-atomic | base | ref | ref | stop / 224 | 36.45 | - | - | - | - | - | - | - |
| grounded-stop-atomic | off | yes | yes | stop / 224 | 36.47 | - | - | - | - | - | - | - |
| grounded-stop-atomic | on4 | yes | yes | stop / 224 | 138.54 | 55 | 220 | 168 | 76.4% | 4.05 | yes (2 unemitted, KV 596->594) | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 36.51 | - | - | - | - | - | - | - |
| grounded-stop-topp | off | yes | yes | stop / 115 | 36.66 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 126.98 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-vecstring | base | ref | ref | stop / 186 | 36.44 | - | - | - | - | - | - | - |
| grounded-stop-vecstring | off | yes | yes | stop / 186 | 35.88 | - | - | - | - | - | - | - |
| grounded-stop-vecstring | on4 | yes | yes | stop / 186 | 125.36 | 47 | 188 | 140 | 74.5% | 3.98 | yes (4 unemitted, KV 560->556) | 0 |
| open-256 | base | ref | ref | length / 256 | 36.57 | - | - | - | - | - | - | - |
| open-256 | off | yes | yes | length / 256 | 36.09 | - | - | - | - | - | - | - |
| open-256 | on4 | yes | yes | length / 256 | 37.59 | 27 | 107 | 7 | 6.5% | 1.26 | no | 0 |
| repeat-eos | base | ref | ref | stop / 16 | 36.44 | - | - | - | - | - | - | - |
| repeat-eos | off | yes | yes | stop / 16 | 36.48 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 107.73 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| sampled-4k-64 | base | ref | ref | length / 64 | 34.01 | - | - | - | - | - | - | - |
| sampled-4k-64 | off | yes | yes | length / 64 | 33.87 | - | - | - | - | - | - | - |
| sampled-4k-64 | on4 | yes | yes | length / 64 | 33.97 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-grounded-256 | base | ref | ref | length / 256 | 26.31 | - | - | - | - | - | - | - |
| sampled-grounded-256 | off | yes | yes | length / 256 | 26.33 | - | - | - | - | - | - | - |
| sampled-grounded-256 | on4 | yes | yes | length / 256 | 26.22 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | base | ref | ref | stop / 51 | 26.06 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | off | yes | yes | stop / 51 | 25.84 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | on4 | yes | yes | stop / 51 | 26.36 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-repeat-eos | base | ref | ref | stop / 16 | 26.28 | - | - | - | - | - | - | - |
| sampled-repeat-eos | off | yes | yes | stop / 16 | 25.16 | - | - | - | - | - | - | - |
| sampled-repeat-eos | on4 | yes | yes | stop / 16 | 26.37 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | base | ref | ref | length / 128 | 35.21 | - | - | - | - | - | - | - |
| sampled-short-128 | off | yes | yes | length / 128 | 35.17 | - | - | - | - | - | - | - |
| sampled-short-128 | on4 | yes | yes | length / 128 | 35.31 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| short-256 | base | ref | ref | length / 256 | 36.59 | - | - | - | - | - | - | - |
| short-256 | off | yes | yes | length / 256 | 35.99 | - | - | - | - | - | - | - |
| short-256 | on4 | yes | yes | length / 256 | 76.93 | 87 | 348 | 144 | 41.4% | 2.66 | no | 0 |
| short-48 | base | ref | ref | length / 48 | 36.72 | - | - | - | - | - | - | - |
| short-48 | off | yes | yes | length / 48 | 36.55 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 71.39 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |
| short-stopkettles | base | ref | ref | stop / 57 | 36.72 | - | - | - | - | - | - | - |
| short-stopkettles | off | yes | yes | stop / 57 | 36.17 | - | - | - | - | - | - | - |
| short-stopkettles | on4 | yes | yes | stop / 57 | 68.55 | 20 | 80 | 32 | 40.0% | 2.60 | yes (4 unemitted, KV 182->178) | 0 |

ALL IDENTICAL
