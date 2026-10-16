| case | variant | stdout == base | ids == base | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k-48 | base | ref | ref | length / 48 | 34.9 | - | - | - | - | - | - | - |
| 4k-48 | bon4 | yes | yes | length / 48 | 113.88 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-48 | off | yes | yes | length / 48 | 35.27 | - | - | - | - | - | - | - |
| 4k-48 | on4 | yes | yes | length / 48 | 115.06 | 13 | 52 | 36 | 69.2% | 3.77 | yes (2 unemitted, KV 4111->4109) | 0 |
| 4k-stop250 | base | ref | ref | stop / 85 | 34.08 | - | - | - | - | - | - | - |
| 4k-stop250 | bon4 | yes | yes | stop / 85 | 106.0 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| 4k-stop250 | off | yes | yes | stop / 85 | 35.12 | - | - | - | - | - | - | - |
| 4k-stop250 | on4 | yes | yes | stop / 85 | 100.95 | 24 | 96 | 62 | 64.6% | 3.58 | yes (2 unemitted, KV 4148->4146) | 0 |
| golden-docs-48 | base | ref | ref | length / 48 | 35.03 | - | - | - | - | - | - | - |
| golden-docs-48 | bon4 | yes | yes | length / 48 | 34.97 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| golden-docs-48 | off | yes | yes | length / 48 | 33.34 | - | - | - | - | - | - | - |
| golden-docs-48 | on4 | yes | yes | length / 48 | 38.0 | 12 | 48 | 6 | 12.5% | 1.50 | yes (1 unemitted, KV 4374->4373) | 0 |
| grounded-256 | base | ref | ref | length / 256 | 36.41 | - | - | - | - | - | - | - |
| grounded-256 | bon4 | yes | yes | length / 256 | 127.0 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-256 | off | yes | yes | length / 256 | 35.93 | - | - | - | - | - | - | - |
| grounded-256 | on4 | yes | yes | length / 256 | 132.86 | 64 | 256 | 189 | 73.8% | 3.95 | no | 0 |
| grounded-min335 | base | ref | ref | length / 400 | 35.08 | - | - | - | - | - | - | - |
| grounded-min335 | bon4 | yes | yes | length / 400 | 105.36 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-min335 | off | yes | yes | length / 400 | 34.23 | - | - | - | - | - | - | - |
| grounded-min335 | on4 | yes | yes | length / 400 | 132.41 | 97 | 388 | 297 | 76.5% | 4.06 | no | 0 |
| grounded-stop-topp | base | ref | ref | stop / 115 | 36.15 | - | - | - | - | - | - | - |
| grounded-stop-topp | bon4 | yes | yes | stop / 115 | 126.61 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| grounded-stop-topp | off | yes | yes | stop / 115 | 34.36 | - | - | - | - | - | - | - |
| grounded-stop-topp | on4 | yes | yes | stop / 115 | 125.8 | 30 | 120 | 86 | 71.7% | 3.87 | yes (4 unemitted, KV 489->485) | 0 |
| repeat-eos | base | ref | ref | stop / 16 | 34.41 | - | - | - | - | - | - | - |
| repeat-eos | bon4 | yes | yes | stop / 16 | 97.98 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| repeat-eos | off | yes | yes | stop / 16 | 32.6 | - | - | - | - | - | - | - |
| repeat-eos | on4 | yes | yes | stop / 16 | 101.94 | 3 | 12 | 11 | 91.7% | 4.67 | yes (1 unemitted, KV 51->50) | 0 |
| sampled-4k-64 | base | ref | ref | length / 64 | 33.24 | - | - | - | - | - | - | - |
| sampled-4k-64 | bon4 | yes | yes | length / 64 | 33.48 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-4k-64 | off | yes | yes | length / 64 | 32.99 | - | - | - | - | - | - | - |
| sampled-4k-64 | on4 | yes | yes | length / 64 | 33.67 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | base | ref | ref | stop / 51 | 25.12 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | bon4 | yes | yes | stop / 51 | 26.15 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-open-min50-stop | off | yes | yes | stop / 51 | 25.86 | - | - | - | - | - | - | - |
| sampled-open-min50-stop | on4 | yes | yes | stop / 51 | 25.25 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | base | ref | ref | length / 128 | 33.45 | - | - | - | - | - | - | - |
| sampled-short-128 | bon4 | yes | yes | length / 128 | 33.78 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| sampled-short-128 | off | yes | yes | length / 128 | 34.13 | - | - | - | - | - | - | - |
| sampled-short-128 | on4 | yes | yes | length / 128 | 34.36 | 0 | 0 | 0 | 0.0% | 0.00 | no | 0 |
| short-48 | base | ref | ref | length / 48 | 36.22 | - | - | - | - | - | - | - |
| short-48 | bon4 | yes | yes | length / 48 | 69.92 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |
| short-48 | off | yes | yes | length / 48 | 36.57 | - | - | - | - | - | - | - |
| short-48 | on4 | yes | yes | length / 48 | 69.67 | 16 | 64 | 23 | 35.9% | 2.44 | no | 0 |

ALL IDENTICAL
