matmul best 63.4 TFLOPS (reference clock state)

| pos0 | matmul ms | matmul TFLOPS | attention ms | attention TFLOPS raw | attention TFLOPS at reference clock |
|---:|---:|---:|---:|---:|---:|
| 0 | 458 | 62.2 | 25 | 49.6 (clock >= 1450 proxy) | 50.5 |
| 2048 | 449 | 63.3 | 73 | 51.0 (clock >= 1450 proxy) | 51.1 |
| 4096 | 449 | 63.3 | 123 | 50.3 (clock >= 1450 proxy) | 50.3 |
| 6144 | 449 | 63.4 | 173 | 50.0 (clock >= 1450 proxy) | 50.0 |
| 8192 | 449 | 63.3 | 223 | 49.9 (clock >= 1450 proxy) | 49.9 |
| 10240 | 449 | 63.3 | 273 | 49.8 (clock >= 1450 proxy) | 49.8 |
| 12288 | 512 | 55.6 | 367 | 43.8 | 50.0 |
| 14336 | 514 | 55.3 | 429 | 43.3 | 49.6 |
| 16384 | 528 | 53.9 | 500 | 42.0 | 49.4 |
| 18432 | 535 | 53.1 | 569 | 41.3 | 49.2 |
| 20480 | 548 | 51.9 | 650 | 40.0 | 48.8 |
| 22528 | 556 | 51.2 | 720 | 39.5 | 48.9 |
| 24576 | 538 | 52.9 | 770 | 40.1 | 48.1 |
| 26624 | 596 | 47.7 | 909 | 36.8 | 48.8 |
| 28672 | 515 | 55.2 | 859 | 41.8 | 47.9 |
| 30720 | 533 | 53.3 | 942 | 40.6 | 48.3 |

Totals: attention 7604 ms raw, 6468 ms at the reference clock (48.9 TFLOPS); matmul 8079 ms raw, 7182 ms at the reference clock.
Admissible chunks (matmul rate >= 56.4 TFLOPS, the proxy for a GPU clock >= 1450 MHz): 6 of 16; attention there 50.0 TFLOPS raw (no rescaling).
