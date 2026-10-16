| context | chunks | matmul ms | attention ms | norms/rope ms | other ms | kernel total ms | attention share | matmul TFLOPS | attention TFLOPS | same, at the reference clock: matmul ms | attention ms | total ms | attention share | attention TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2 | 907 | 98 | 21 | 19 | 1045 | 9% | 62.9 | 50.4 | 903 | 98 | 1041 | 9% | 50.5 |
| 16384 | 8 | 3675 | 1646 | 78 | 75 | 5474 | 30% | 62.0 | 48.1 | 3596 | 1587 | 5333 | 30% | 49.9 |
| 32764 | 16 | 8427 | 7691 | 167 | 163 | 16449 | 47% | 54.0 | 41.2 | 7319 | 6570 | 14177 | 46% | 48.2 |
| 40000 | 20 | 9700 | 11075 | 197 | 193 | 21166 | 52% | 57.3 | 42.6 | 8835 | 9749 | 18941 | 51% | 48.4 |
