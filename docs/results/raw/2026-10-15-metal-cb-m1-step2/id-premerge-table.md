| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-48 | base | ref | ref | length / 48 | 35.36 | - | - | - | - | - | - | - |
| 4k-48 | bon4 | yes | yes | length / 48 | 115.86 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-48 | off | yes | yes | length / 48 | 34.96 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 115.4 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 35.44 | - | - | - | - | - | - | - |
| 4k-stop250 | bon4 | yes | yes | stop / 85 | 113.8 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | off | yes | yes | stop / 85 | 35.42 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 113.28 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| golden-docs-48 | base | ref | ref | length / 48 | 35.53 | - | - | - | - | - | - | - |
| golden-docs-48 | bon4 | yes | yes | length / 48 | 38.58 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| golden-docs-48 | off | yes | yes | length / 48 | 35.35 | - | - | - | - | - | - | - |
| golden-docs-48 | on4 | yes | yes | length / 48 | 38.49 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| grounded-256 | base | ref | ref | length / 256 | 36.56 | - | - | - | - | - | - | - |
| grounded-256 | bon4 | yes | yes | length / 256 | 135.2 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-256 | off | yes | yes | length / 256 | 36.43 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 132.75 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-min335 | base | ref | ref | length / 400 | 36.45 | - | - | - | - | - | - | - |
| grounded-min335 | bon4 | yes | yes | length / 400 | 131.37 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min335 | off | yes | yes | length / 400 | 36.39 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 135.66 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 36.18 | - | - | - | - | - | - | - |
| grounded-stop-topp | bon4 | yes | yes | stop / 115 | 126.95 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | off | yes | yes | stop / 115 | 36.62 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 127.62 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| repeat-eos | base | ref | ref | stop / 16 | 37.01 | - | - | - | - | - | - | - |
| repeat-eos | bon4 | yes | yes | stop / 16 | 110.57 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| repeat-eos | off | yes | yes | stop / 16 | 36.98 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 110.26 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| sampled-4k-64 | base | ref | ref | length / 64 | 34.33 | - | - | - | - | - | - | - |
| sampled-4k-64 | bon4 | yes | yes | length / 64 | 34.31 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-4k-64 | off | yes | yes | length / 64 | 34.42 | - | - | - | - | - | - | - |
| sampled-4k-64 | on4 | yes | yes | length / 64 | 34.27 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | base | ref | ref | stop / 51 | 26.44 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | bon4 | yes | yes | stop / 51 | 26.66 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | off | yes | yes | stop / 51 | 26.52 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | on4 | yes | yes | stop / 51 | 26.4 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | base | ref | ref | length / 128 | 35.47 | - | - | - | - | - | - | - |
| sampled-short-128 | bon4 | yes | yes | length / 128 | 35.47 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | off | yes | yes | length / 128 | 35.52 | - | - | - | - | - | - | - |
| sampled-short-128 | on4 | yes | yes | length / 128 | 35.57 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| short-48 | base | ref | ref | length / 48 | 36.74 | - | - | - | - | - | - | - |
| short-48 | bon4 | yes | yes | length / 48 | 71.02 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |
| short-48 | off | yes | yes | length / 48 | 36.65 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 71.1 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |

ALL IDENTICAL
