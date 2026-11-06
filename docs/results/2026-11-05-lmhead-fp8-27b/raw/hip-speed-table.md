| config | prompt | bf16 head tok/s (A, A) | quantised | head tok/s (B, B) | B/A (medians of the two visits) |
|---|---|---|---|---|---|
| head tp2 MTP k2 | 1024 | 88.2, 89.7 | fp8 | 99.6, 96.5 | 1.103 |
| head tp2 MTP k2 | 4096 | 83.4, 84.3 | fp8 | 92.4, 96.5 | 1.126 |
| head tp2 MTP k2 | 16384 | 80.7, 80.9 | fp8 | 98.0, 86.1 | 1.139 |
| head tp2 plain | 1024 | 55.7, 55.8 | fp8 | 58.5, 58.7 | 1.052 |
| head tp2 plain | 4096 | 54.9, 54.9 | fp8 | 57.7, 57.7 | 1.051 |
| head tp2 plain | 16384 | 52.0, 52.0 | fp8 | 54.5, 54.4 | 1.048 |
| head tp1 MTP k2 | 1024 | 57.1, 54.3 | fp8 | 61.7, 63.7 | 1.126 |
| head tp1 MTP k2 | 4096 | 58.1, 59.2 | fp8 | 63.1, 59.9 | 1.049 |
| head tp1 MTP k2 | 16384 | 53.3, 52.9 | fp8 | 58.5, 57.9 | 1.097 |
| head tp1 plain | 1024 | 34.1, 33.2 | fp8 | 35.8, 35.8 | 1.064 |
| head tp1 plain | 4096 | 33.1, 33.1 | fp8 | 35.2, 35.2 | 1.063 |
| head tp1 plain | 16384 | 31.5, 31.5 | fp8 | 33.5, 33.3 | 1.062 |
| head tp2 plain | 1024 | 55.6, 55.9 | mxfp4 | 61.4, 61.6 | 1.103 |
| head tp2 plain | 4096 | 54.9, 55.0 | mxfp4 | 60.6, 60.6 | 1.103 |
| head tp2 plain | 16384 | 51.4, 51.3 | mxfp4 | 57.0, 57.2 | 1.112 |
