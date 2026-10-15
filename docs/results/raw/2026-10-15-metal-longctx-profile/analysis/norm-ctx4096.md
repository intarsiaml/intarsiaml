matmul best 63.3 TFLOPS (reference clock state)

| pos0 | matmul ms | matmul TFLOPS | attention ms | attention TFLOPS raw | attention TFLOPS at reference clock |
|---:|---:|---:|---:|---:|---:|
| 0 | 456 | 62.5 | 25 | 48.8 (clock >= 1450 proxy) | 49.5 |
| 2048 | 449 | 63.3 | 73 | 50.9 (clock >= 1450 proxy) | 50.9 |

Totals: attention 98 ms raw, 98 ms at the reference clock (50.6 TFLOPS); matmul 905 ms raw, 899 ms at the reference clock.
Admissible chunks (matmul rate >= 56.4 TFLOPS, the proxy for a GPU clock >= 1450 MHz): 2 of 2; attention there 50.4 TFLOPS raw (no rescaling).
