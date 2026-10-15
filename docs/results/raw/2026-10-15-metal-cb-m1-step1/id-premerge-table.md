| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-48 | base | ref | ref | length / 48 | 34.87 | - | - | - | - | - | - | - |
| 4k-48 | bon4 | yes | yes | length / 48 | 116.54 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-48 | off | yes | yes | length / 48 | 34.89 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 112.03 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 34.84 | - | - | - | - | - | - | - |
| 4k-stop250 | bon4 | yes | yes | stop / 85 | 105.11 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | off | yes | yes | stop / 85 | 35.11 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 107.97 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| golden-docs-48 | base | ref | ref | length / 48 | 35.33 | - | - | - | - | - | - | - |
| golden-docs-48 | bon4 | yes | yes | length / 48 | 35.46 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| golden-docs-48 | off | yes | yes | length / 48 | 33.22 | - | - | - | - | - | - | - |
| golden-docs-48 | on4 | yes | yes | length / 48 | 35.2 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| grounded-256 | base | ref | ref | length / 256 | 36.25 | - | - | - | - | - | - | - |
| grounded-256 | bon4 | yes | yes | length / 256 | 135.97 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-256 | off | yes | yes | length / 256 | 36.35 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 136.95 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-min335 | base | ref | ref | length / 400 | 35.9 | - | - | - | - | - | - | - |
| grounded-min335 | bon4 | yes | yes | length / 400 | 131.2 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min335 | off | yes | yes | length / 400 | 35.09 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 135.47 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 36.07 | - | - | - | - | - | - | - |
| grounded-stop-topp | bon4 | yes | yes | stop / 115 | 126.13 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | off | yes | yes | stop / 115 | 36.13 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 126.4 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| repeat-eos | base | ref | ref | stop / 16 | 32.47 | - | - | - | - | - | - | - |
| repeat-eos | bon4 | yes | yes | stop / 16 | 107.02 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| repeat-eos | off | yes | yes | stop / 16 | 34.02 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 90.06 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| sampled-4k-64 | base | ref | ref | length / 64 | 34.02 | - | - | - | - | - | - | - |
| sampled-4k-64 | bon4 | yes | yes | length / 64 | 33.58 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-4k-64 | off | yes | yes | length / 64 | 33.78 | - | - | - | - | - | - | - |
| sampled-4k-64 | on4 | yes | yes | length / 64 | 33.74 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | base | ref | ref | stop / 51 | 25.85 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | bon4 | yes | yes | stop / 51 | 26.42 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | off | yes | yes | stop / 51 | 26.5 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | on4 | yes | yes | stop / 51 | 26.39 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | base | ref | ref | length / 128 | 31.77 | - | - | - | - | - | - | - |
| sampled-short-128 | bon4 | yes | yes | length / 128 | 33.04 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | off | yes | yes | length / 128 | 34.65 | - | - | - | - | - | - | - |
| sampled-short-128 | on4 | yes | yes | length / 128 | 34.38 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| short-48 | base | ref | ref | length / 48 | 34.88 | - | - | - | - | - | - | - |
| short-48 | bon4 | yes | yes | length / 48 | 71.13 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |
| short-48 | off | yes | yes | length / 48 | 35.74 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 70.49 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |

ALL IDENTICAL
