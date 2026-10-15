| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-48 | base | ref | ref | length / 48 | 35.15 | - | - | - | - | - | - | - |
| 4k-48 | bon4 | yes | yes | length / 48 | 116.46 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-48 | off | yes | yes | length / 48 | 34.83 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 112.96 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 34.25 | - | - | - | - | - | - | - |
| 4k-stop250 | bon4 | yes | yes | stop / 85 | 108.5 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | off | yes | yes | stop / 85 | 34.84 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 111.51 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| golden-docs-48 | base | ref | ref | length / 48 | 33.97 | - | - | - | - | - | - | - |
| golden-docs-48 | bon4 | yes | yes | length / 48 | 38.38 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| golden-docs-48 | off | yes | yes | length / 48 | 33.83 | - | - | - | - | - | - | - |
| golden-docs-48 | on4 | yes | yes | length / 48 | 38.0 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| grounded-256 | base | ref | ref | length / 256 | 36.13 | - | - | - | - | - | - | - |
| grounded-256 | bon4 | yes | yes | length / 256 | 133.12 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-256 | off | yes | yes | length / 256 | 36.18 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 137.32 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-min335 | base | ref | ref | length / 400 | 35.62 | - | - | - | - | - | - | - |
| grounded-min335 | bon4 | yes | yes | length / 400 | 133.97 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min335 | off | yes | yes | length / 400 | 35.61 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 136.29 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 34.98 | - | - | - | - | - | - | - |
| grounded-stop-topp | bon4 | yes | yes | stop / 115 | 123.88 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | off | yes | yes | stop / 115 | 35.01 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 126.57 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| repeat-eos | base | ref | ref | stop / 16 | 36.96 | - | - | - | - | - | - | - |
| repeat-eos | bon4 | yes | yes | stop / 16 | 108.19 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| repeat-eos | off | yes | yes | stop / 16 | 36.9 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 102.18 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| sampled-4k-64 | base | ref | ref | length / 64 | 34.14 | - | - | - | - | - | - | - |
| sampled-4k-64 | bon4 | yes | yes | length / 64 | 33.24 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-4k-64 | off | yes | yes | length / 64 | 32.97 | - | - | - | - | - | - | - |
| sampled-4k-64 | on4 | yes | yes | length / 64 | 34.17 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | base | ref | ref | stop / 51 | 26.37 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | bon4 | yes | yes | stop / 51 | 25.76 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | off | yes | yes | stop / 51 | 26.62 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | on4 | yes | yes | stop / 51 | 26.43 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | base | ref | ref | length / 128 | 34.24 | - | - | - | - | - | - | - |
| sampled-short-128 | bon4 | yes | yes | length / 128 | 34.94 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | off | yes | yes | length / 128 | 35.26 | - | - | - | - | - | - | - |
| sampled-short-128 | on4 | yes | yes | length / 128 | 33.99 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| short-48 | base | ref | ref | length / 48 | 35.49 | - | - | - | - | - | - | - |
| short-48 | bon4 | yes | yes | length / 48 | 70.63 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |
| short-48 | off | yes | yes | length / 48 | 36.69 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 71.3 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |

ALL IDENTICAL
