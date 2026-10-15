| case | prompt tok | finish / tokens | off == base | on4 == base | on6 == base | acceptance on4 / on6 | tokens per pass on4 / on6 | verified tokens dropped at the stop, on4 / on6 | min_tokens cuts on4 / on6 | decode tok/s base / off / on4 / on6 |
|---|---|---|---|---|---|---|---|---|---|---|
| short-48 | 122 | length / 48 | yes | yes | yes | 38% / 30% | 2.50 / 2.81 | 0 / 0 | 0 / 0 | 35.8 / 35.5 / 81.6 / 87.1 |
| short-256 | 122 | length / 256 | yes | yes | yes | 42% / 30% | 2.66 / 2.82 | 0 / 1 | 0 / 0 | 35.6 / 35.8 / 89.9 / 84.7 |
| 4k-48 | 4062 | length / 48 | yes | yes | yes | 69% / 58% | 3.77 / 4.45 | 2 / 2 | 0 / 0 | 33.7 / 34.0 / 115.8 / 121.4 |
| 4k-256 | 4062 | length / 256 | yes | yes | yes | 70% / 61% | 3.81 / 4.64 | 0 / 0 | 0 / 0 | 33.8 / 33.7 / 114.3 / 99.8 |
| grounded-256 | 371 | length / 256 | yes | yes | yes | 73% / 64% | 3.92 / 4.81 | 0 / 0 | 0 / 0 | 31.5 / 34.6 / 105.8 / 104.1 |
| grounded-512 | 371 | stop / 329 | yes | yes | yes | 72% / 60% | 3.89 / 4.61 | 0 / 0 | 0 / 0 | 32.0 / 34.0 / 100.5 / 95.2 |
| open-256 | 40 | length / 256 | yes | yes | yes | 3% / 2% | 1.11 / 1.12 | 0 / 0 | 0 / 0 | 31.7 / 34.8 / 33.0 / 30.8 |
| grounded-max102 | 371 | length / 102 | yes | yes | yes | 78% / 65% | 4.12 / 4.90 | 2 / 2 | 0 / 0 | 35.3 / 36.0 / 122.7 / 118.3 |
| grounded-max188 | 371 | length / 188 | yes | yes | yes | 73% / 63% | 3.94 / 4.80 | 2 / 5 | 0 / 0 | 35.3 / 35.3 / 114.2 / 100.1 |
| 4k-max9 | 4062 | length / 9 | yes | yes | yes | 38% / 33% | 2.50 / 3.00 | 2 / 4 | 0 / 0 | 33.6 / 34.3 / 56.4 / 45.9 |
| 4k-max44 | 4062 | length / 44 | yes | yes | yes | 69% / 58% | 3.75 / 4.45 | 2 / 6 | 0 / 0 | 34.2 / 34.4 / 97.3 / 95.3 |
| grounded-stop-topp | 371 | stop / 115 | yes | yes | yes | 70% / 60% | 3.81 / 4.62 | 4 / 6 | 0 / 0 | 35.9 / 35.1 / 108.6 / 97.5 |
| grounded-stop-vecstring | 371 | stop / 186 | yes | yes | yes | 73% / 62% | 3.94 / 4.74 | 4 / 0 | 0 / 0 | 35.6 / 35.0 / 111.6 / 108.6 |
| grounded-stop-atomic | 371 | stop / 224 | yes | yes | yes | 75% / 67% | 4.02 / 5.04 | 2 / 4 | 0 / 0 | 34.8 / 34.4 / 116.0 / 105.2 |
| 4k-stop250 | 4062 | stop / 85 | yes | yes | yes | 65% / 55% | 3.58 / 4.30 | 2 / 2 | 0 / 0 | 34.1 / 34.1 / 101.5 / 95.0 |
| short-stopkettles | 122 | stop / 57 | yes | yes | yes | 41% / 32% | 2.64 / 2.89 | 4 / 1 | 0 / 0 | 35.1 / 35.6 / 79.7 / 76.2 |
| grounded-min331 | 371 | length / 400 | yes | yes | yes | 74% / 62% | 3.98 / 4.69 | 0 / 1 | 1 / 1 | 34.0 / 34.9 / 118.4 / 112.4 |
| grounded-min335 | 371 | length / 400 | yes | yes | yes | 74% / 62% | 3.98 / 4.69 | 0 / 1 | 1 / 1 | 35.0 / 35.5 / 113.6 / 95.3 |
| grounded-min320-stopfence | 371 | stop / 328 | yes | yes | yes | 73% / 61% | 3.93 / 4.66 | 0 / 0 | 0 / 0 | 35.2 / 34.8 / 115.3 / 99.1 |

on4: 1001 verify passes, 4003 proposed, 2446 accepted (61.1%)
on6: 866 verify passes, 5193 proposed, 2590 accepted (49.9%)
