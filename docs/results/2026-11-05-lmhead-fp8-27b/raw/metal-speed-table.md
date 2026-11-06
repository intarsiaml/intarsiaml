| config | prompt | bf16 head tok/s (A, A) | quantised | head tok/s (B, B) | B/A (medians of the two visits) |
|---|---|---|---|---|---|
| metal MTP k2 | 1024 | 31.5, 33.3 | fp8 | 39.9, 38.6 | 1.212 |
| metal MTP k2 | 4096 | 32.3, 33.0 | fp8 | 38.9, 40.3 | 1.212 |
| metal MTP k2 | 16384 | 28.5, 31.5 | fp8 | 37.8, 36.2 | 1.232 |
| metal MTP k2 | 1024 | 31.5, 33.3 | mxfp4 | 40.2 | 1.243 |
| metal MTP k2 | 4096 | 32.3, 33.0 | mxfp4 | 42.1 | 1.289 |
| metal MTP k2 | 16384 | 28.5, 31.5 | mxfp4 | 33.6 | 1.118 |
| metal plain | 1024 | 29.2, 28.9, 28.3 | fp8 | 32.4, 31.9 | 1.114 |
| metal plain | 4096 | 28.7, 28.7, 27.6 | fp8 | 31.7, 30.8 | 1.089 |
| metal plain | 16384 | 26.0, 25.7, 25.2 | fp8 | 28.7, 28.3 | 1.108 |
| metal plain | 1024 | 29.2, 28.9, 28.3 | mxfp4 | 32.3 | 1.119 |
| metal plain | 4096 | 28.7, 28.7, 27.6 | mxfp4 | 31.1 | 1.083 |
| metal plain | 16384 | 26.0, 25.7, 25.2 | mxfp4 | 28.8 | 1.119 |
