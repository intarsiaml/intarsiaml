| chunk | prompt tokens | pass | off prefill s (tok/s) | on prefill s (tok/s) | on / off |
|---:|---:|---:|---|---|---:|
| 512 | 4074 | 1 | 4.82 (845) | 2.58 (1586) | 0.536 |
| 512 | 4097 | 2 | 4.87 (842) | 2.54 (1605) | 0.522 |
| 512 | 33259 | 1 | 40.26 (826) | 19.92 (1673) | 0.495 |
| 512 | 33302 | 2 | 40.52 (822) | 19.87 (1674) | 0.490 |
| 2048 | 4075 | 1 | 4.01 (1015) | 2.20 (1849) | 0.549 |
| 2048 | 4093 | 2 | 4.06 (1008) | 2.16 (1887) | 0.531 |
| 2048 | 33280 | 1 | 33.36 (998) | 16.79 (1983) | 0.503 |
| 2048 | 33302 | 2 | 33.43 (996) | 16.60 (2006) | 0.497 |

chunk 512, 4096 tokens: mean off 4.84 s, mean on 2.56 s, on/off 0.529 (2 off, 2 on cells)
chunk 512, 32768 tokens: mean off 40.39 s, mean on 19.89 s, on/off 0.493 (2 off, 2 on cells)
chunk 2048, 4096 tokens: mean off 4.04 s, mean on 2.18 s, on/off 0.540 (2 off, 2 on cells)
chunk 2048, 32768 tokens: mean off 33.39 s, mean on 16.69 s, on/off 0.500 (2 off, 2 on cells)

Criterion: mean on/off prefill time at most 1.05 for every chunk and size, no errors: met
