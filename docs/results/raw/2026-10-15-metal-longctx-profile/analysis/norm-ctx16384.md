matmul best 63.3 TFLOPS (reference clock state)

| pos0 | matmul ms | matmul TFLOPS | attention ms | attention TFLOPS raw | attention TFLOPS at reference clock |
|---:|---:|---:|---:|---:|---:|
| 0 | 456 | 62.4 | 26 | 48.3 (clock >= 1450 proxy) | 49.0 |
| 2048 | 450 | 63.3 | 73 | 50.9 (clock >= 1450 proxy) | 51.0 |
| 4096 | 450 | 63.2 | 123 | 50.3 (clock >= 1450 proxy) | 50.3 |
| 6144 | 449 | 63.3 | 173 | 50.0 (clock >= 1450 proxy) | 50.0 |
| 8192 | 450 | 63.2 | 223 | 49.9 (clock >= 1450 proxy) | 50.0 |
| 10240 | 450 | 63.3 | 274 | 49.7 (clock >= 1450 proxy) | 49.8 |
| 12288 | 452 | 62.9 | 326 | 49.3 (clock >= 1450 proxy) | 49.6 |
| 14336 | 516 | 55.2 | 429 | 43.3 | 49.7 |

Totals: attention 1646 ms raw, 1588 ms at the reference clock (49.9 TFLOPS); matmul 3672 ms raw, 3596 ms at the reference clock.
Admissible chunks (matmul rate >= 56.4 TFLOPS, the proxy for a GPU clock >= 1450 MHz): 7 of 8; attention there 49.8 TFLOPS raw (no rescaling).
