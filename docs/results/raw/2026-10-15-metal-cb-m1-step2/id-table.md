| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-48 | base | ref | ref | length / 48 | 35.54 | - | - | - | - | - | - | - |
| 4k-48 | bon4 | yes | yes | length / 48 | 110.79 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-48 | off | yes | yes | length / 48 | 35.09 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 114.78 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 34.69 | - | - | - | - | - | - | - |
| 4k-stop250 | bon4 | yes | yes | stop / 85 | 111.42 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | off | yes | yes | stop / 85 | 34.81 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 113.32 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| golden-docs-48 | base | ref | ref | length / 48 | 35.12 | - | - | - | - | - | - | - |
| golden-docs-48 | bon4 | yes | yes | length / 48 | 34.2 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| golden-docs-48 | off | yes | yes | length / 48 | 34.4 | - | - | - | - | - | - | - |
| golden-docs-48 | on4 | yes | yes | length / 48 | 34.9 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| grounded-256 | base | ref | ref | length / 256 | 36.27 | - | - | - | - | - | - | - |
| grounded-256 | bon4 | yes | yes | length / 256 | 128.66 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-256 | off | yes | yes | length / 256 | 36.18 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 135.99 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-min335 | base | ref | ref | length / 400 | 35.9 | - | - | - | - | - | - | - |
| grounded-min335 | bon4 | yes | yes | length / 400 | 137.18 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min335 | off | yes | yes | length / 400 | 35.52 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 136.42 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 35.95 | - | - | - | - | - | - | - |
| grounded-stop-topp | bon4 | yes | yes | stop / 115 | 121.93 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | off | yes | yes | stop / 115 | 35.8 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 122.67 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| repeat-eos | base | ref | ref | stop / 16 | 35.73 | - | - | - | - | - | - | - |
| repeat-eos | bon4 | yes | yes | stop / 16 | 98.58 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| repeat-eos | off | yes | yes | stop / 16 | 35.84 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 95.41 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| sampled-4k-64 | base | ref | ref | length / 64 | 33.42 | - | - | - | - | - | - | - |
| sampled-4k-64 | bon4 | yes | yes | length / 64 | 33.92 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-4k-64 | off | yes | yes | length / 64 | 34.2 | - | - | - | - | - | - | - |
| sampled-4k-64 | on4 | yes | yes | length / 64 | 33.45 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | base | ref | ref | stop / 51 | 25.88 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | bon4 | yes | yes | stop / 51 | 26.1 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | off | yes | yes | stop / 51 | 26.12 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | on4 | yes | yes | stop / 51 | 25.87 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | base | ref | ref | length / 128 | 33.61 | - | - | - | - | - | - | - |
| sampled-short-128 | bon4 | yes | yes | length / 128 | 34.57 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | off | yes | yes | length / 128 | 35.1 | - | - | - | - | - | - | - |
| sampled-short-128 | on4 | yes | yes | length / 128 | 34.68 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| short-48 | base | ref | ref | length / 48 | 36.39 | - | - | - | - | - | - | - |
| short-48 | bon4 | yes | yes | length / 48 | 71.15 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |
| short-48 | off | yes | yes | length / 48 | 36.72 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 71.14 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |

ALL IDENTICAL
