matmul best 62.3 TFLOPS (reference clock state)

| pos0 | matmul ms | matmul TFLOPS | attention ms | attention TFLOPS raw | attention TFLOPS at reference clock |
|---:|---:|---:|---:|---:|---:|
| 0 | 457 | 62.3 | 25 | 49.6 (clock >= 1450 proxy) | 49.6 |
| 2048 | 515 | 55.3 | 83 | 44.9 | 50.6 |
| 4096 | 577 | 49.3 | 157 | 39.5 | 49.8 |
| 6144 | 575 | 49.5 | 220 | 39.4 | 49.5 |
| 8192 | 575 | 49.5 | 285 | 39.1 | 49.2 |
| 10240 | 525 | 54.2 | 320 | 42.5 | 48.9 |
| 12288 | 530 | 53.7 | 381 | 42.2 | 48.9 |
| 14336 | 550 | 51.7 | 456 | 40.7 | 49.0 |
| 16384 | 520 | 54.7 | 491 | 42.8 | 48.8 |
| 18432 | 538 | 52.9 | 568 | 41.4 | 48.7 |
| 20480 | 596 | 47.8 | 696 | 37.3 | 48.7 |
| 22528 | 586 | 48.5 | 747 | 38.1 | 48.9 |
| 24576 | 534 | 53.3 | 757 | 40.9 | 47.7 |
| 26624 | 516 | 55.1 | 784 | 42.6 | 48.1 |
| 28672 | 542 | 52.5 | 892 | 40.2 | 47.7 |
| 30720 | 563 | 50.5 | 980 | 39.0 | 48.2 |

Totals: attention 7841 ms raw, 6528 ms at the reference clock (48.5 TFLOPS); matmul 8698 ms raw, 7310 ms at the reference clock.
Admissible chunks (matmul rate >= 56.4 TFLOPS, the proxy for a GPU clock >= 1450 MHz): 1 of 16; attention there 49.6 TFLOPS raw (no rescaling).
