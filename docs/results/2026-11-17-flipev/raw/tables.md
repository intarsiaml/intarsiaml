## herd: TTFT sum / p50 / last / wall (s), mean over cells (range), n cells

| cell | arm | n | TTFT sum | TTFT p50 | TTFT last | wall | cached tokens (first, rest) |
|---|---|---|---|---|---|---|---|
| mix 4 shared 16K + 2 unrelated 8K | off | 4 | 51.31 (51.09 to 51.51) | 8.51 (8.48 to 8.53) | 16.74 (16.64 to 16.79) | 17.22 (17.10 to 17.28) | [0, 0, 0, 0, 0, 0] |
| mix 4 shared 16K + 2 unrelated 8K | on | 4 | 32.65 (32.07 to 33.10) | 6.99 (6.86 to 7.10) | 7.41 (7.24 to 7.52) | 8.66 (8.57 to 8.70) | [16384, 0, 16384, 0, 0, 16384] |
| shared 16448 stagger 0.5 | off | 4 | 45.85 (43.51 to 46.81) | 6.19 (5.70 to 6.39) | 6.90 (6.55 to 7.03) | 10.78 (10.48 to 10.94) | [0, 6144, 8192, 12288, 14336, 16384, 16384, 16384] |
| shared 16448 stagger 0.5 | on | 4 | 13.32 (13.29 to 13.35) | 1.67 (1.66 to 1.67) | 2.86 (2.86 to 2.87) | 6.14 (6.12 to 6.21) | [0, 16384, 16384, 16384, 16384, 16384, 16384, 16384] |
| shared 16448 stagger 2.0 | off | 4 | 6.11 (5.97 to 6.17) | 0.24 (0.23 to 0.25) | 2.85 (2.83 to 2.89) | 14.72 (14.66 to 14.82) | [0, 14336, 16384, 16384, 16384, 16384, 16384, 16384] |
| shared 16448 stagger 2.0 | on | 4 | 5.30 (5.16 to 5.46) | 0.23 (0.20 to 0.25) | 2.83 (2.82 to 2.84) | 14.72 (14.67 to 14.74) | [0, 16384, 16384, 16384, 16384, 16384, 16384, 16384] |
| shared 32832 stagger 0.0 | off | 4 | 237.38 (236.59 to 239.13) | 29.72 (29.63 to 29.96) | 52.30 (52.20 to 52.50) | 52.88 (52.74 to 53.04) | [2048, 2048, 2048, 2048, 2048, 2048, 0, 2048] |
| shared 32832 stagger 0.0 | on | 4 | 59.48 (59.27 to 59.70) | 7.43 (7.39 to 7.47) | 8.17 (8.13 to 8.22) | 10.83 (10.80 to 10.85) | [32768, 32768, 32768, 0, 32768, 32768, 32768, 32768] |
| shared 4160 stagger 0.0 | off | 6 | 20.69 (19.63 to 22.53) | 2.58 (2.44 to 2.84) | 4.36 (4.20 to 4.64) | 5.59 (5.45 to 5.87) | [2048, 2048, 2048, 2048, 2048, 2048, 2048, 0] |
| shared 4160 stagger 0.0 | on | 6 | 9.83 (9.65 to 10.07) | 1.22 (1.20 to 1.25) | 1.69 (1.66 to 1.74) | 3.11 (3.07 to 3.15) | [4096, 4096, 4096, 4096, 4096, 0, 4096, 4096] |

## per request index (mean over cells): TTFT s / end-of-reply s since the barrier

### mix 4 shared 16K + 2 unrelated 8K

| idx | send at | off TTFT | on TTFT | off end | on end |
|---|---|---|---|---|---|
| 0 | 0.00 | 11.86 | 7.32 | 15.33 | 8.66 |
| 1 | 0.00 | 2.38 | 2.01 | 7.42 | 6.77 |
| 2 | 0.00 | 14.23 | 7.12 | 15.55 | 8.63 |
| 3 | 0.00 | 10.99 | 7.15 | 13.99 | 8.63 |
| 4 | 0.00 | 1.66 | 2.01 | 5.53 | 6.28 |
| 5 | 0.00 | 10.18 | 7.02 | 13.83 | 8.61 |

### shared 16448 stagger 0.5

| idx | send at | off TTFT | on TTFT | off end | on end |
|---|---|---|---|---|---|
| 0 | 0.00 | 2.87 | 2.86 | 6.83 | 5.41 |
| 1 | 0.51 | 4.54 | 2.53 | 9.89 | 5.98 |
| 2 | 1.00 | 6.17 | 2.19 | 10.33 | 6.02 |
| 3 | 1.51 | 6.82 | 1.84 | 10.60 | 6.05 |
| 4 | 2.00 | 6.90 | 1.49 | 10.70 | 6.07 |
| 5 | 2.50 | 6.52 | 1.15 | 10.72 | 6.12 |
| 6 | 3.01 | 6.20 | 0.80 | 10.78 | 6.14 |
| 7 | 3.50 | 5.82 | 0.45 | 10.78 | 6.14 |

### shared 16448 stagger 2.0

| idx | send at | off TTFT | on TTFT | off end | on end |
|---|---|---|---|---|---|
| 0 | 0.00 | 2.85 | 2.83 | 4.82 | 4.52 |
| 1 | 2.00 | 1.41 | 1.02 | 4.05 | 3.67 |
| 2 | 4.00 | 0.77 | 0.42 | 5.23 | 4.90 |
| 3 | 6.01 | 0.23 | 0.20 | 6.73 | 6.71 |
| 4 | 8.00 | 0.22 | 0.22 | 8.63 | 8.61 |
| 5 | 10.00 | 0.22 | 0.19 | 10.67 | 10.63 |
| 6 | 12.00 | 0.19 | 0.21 | 12.73 | 12.73 |
| 7 | 14.00 | 0.21 | 0.21 | 14.72 | 14.72 |

## herd ids: every reply (concurrent and solo) against the solo reply of the same prompt set (first visit)

- arm A herd8: 144 / 144 equal
- arm A mix: 24 / 24 equal
- arm A solo: 168 / 168 equal
- arm B herd8: 144 / 144 equal
- arm B mix: 24 / 24 equal
- arm B solo: 168 / 168 equal
- errors: 0; reply lengths (tokens: count): {48: 672}

## q4: ITL of the ongoing decodes (event gaps, ms) while the 40K prompt prefills; pooled over runs

| decoders | scenario | arm | runs | window gaps n | p50 | p99 | max | base p50 | base p99 | base max | long TTFT s | short TTFTs s (s1 / s2) | dec tok/s/stream in window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | long | off | 4 | 869 | 109.6 | 235.9 | 269.8 | 0.0 | 86.7 | 102.1 | 8.82 | - | 12.3 |
| 2 | long | on | 4 | 884 | 98.1 | 225.4 | 251.6 | 0.0 | 85.1 | 96.8 | 8.87 | - | 12.4 |
| 2 | longshort | off | 4 | 869 | 103.6 | 221.2 | 268.8 | 0.0 | 78.9 | 90.0 | 8.83 | 7.56 / 5.57 | 12.3 |
| 2 | longshort | on | 4 | 772 | 121.3 | 220.3 | 249.1 | 0.0 | 83.5 | 95.8 | 9.29 | 0.69 / 0.50 | 10.4 |
| 4 | long | off | 4 | 14597 | 19.3 | 86.6 | 168.7 | 19.1 | 89.9 | 113.5 | 24.81 | - | 36.8 |
| 4 | long | on | 4 | 14611 | 19.4 | 86.7 | 161.7 | 19.2 | 83.3 | 89.2 | 24.98 | - | 36.6 |
| 4 | longshort | off | 4 | 14584 | 19.3 | 88.0 | 119.7 | 18.9 | 86.9 | 88.8 | 25.36 | 24.10 / 21.81 | 35.9 |
| 4 | longshort | on | 4 | 14590 | 19.4 | 87.4 | 134.2 | 19.0 | 82.8 | 88.4 | 25.49 | 16.92 / 14.42 | 35.8 |

per-run window max gap (ms) and p99:

- d2 long off: max 220, 270, 236, 203 | p99 211, 263, 230, 202
- d2 long on: max 225, 218, 252, 215 | p99 204, 205, 250, 209
- d2 longshort off: max 202, 230, 269, 217 | p99 200, 228, 250, 199
- d2 longshort on: max 215, 220, 201, 249 | p99 212, 205, 199, 241
- d4 long off: max 169, 152, 113, 118 | p99 89, 87, 84, 85
- d4 long on: max 124, 140, 162, 111 | p99 87, 89, 84, 85
- d4 longshort off: max 118, 117, 120, 116 | p99 87, 88, 89, 86
- d4 longshort on: max 134, 108, 122, 117 | p99 85, 85, 88, 88

## q4 ids

- arm off: 80 / 80 contended replies equal their own server's solo reply
- arm on: 80 / 80 contended replies equal their own server's solo reply
- distinct id sequences per prompt across all runs and arms (1 = all equal): d0: 1, d1: 1, d2: 1, d3: 1, long: 1, s1: 1, s2: 1
- reply lengths (prompt kind, tokens: replies): {('d', 1000): 96, ('l', 16): 32, ('s', 16): 32}
- solo references across the servers: d0: 1 distinct, d1: 1 distinct, d2: 1 distinct, d3: 1 distinct, long: 1 distinct, s1: 1 distinct, s2: 1 distinct
