| context | admissible chunks | pos0 range | matmul ms | attention ms | norms/rope ms | other ms | attention share | matmul TFLOPS | attention TFLOPS |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2 of 2 | 0-2048 | 907 | 98 | 21 | 19 | 9% | 62.9 | 50.4 |
| 16384 | 7 of 8 | 0-12288 | 3157 | 1217 | 68 | 66 | 27% | 63.1 | 49.8 |
| 32764 | 6 of 16 | 0-10240 | 2704 | 891 | 60 | 57 | 24% | 63.1 | 50.0 |
| 40000 | 10 of 20 | 0-18432 | 4711 | 2656 | 98 | 97 | 35% | 60.4 | 46.6 |
