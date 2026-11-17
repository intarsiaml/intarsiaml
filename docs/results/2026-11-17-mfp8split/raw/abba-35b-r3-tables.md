| cell | A agg tok/s | B agg tok/s | B / A | A TTFT p50 s | B TTFT p50 s | B / A | dec tok/s p50 A -> B | spread A / B (agg, %) | visits |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| s8 | 342.01 | 337.65 | 0.987 | 0.267 | 0.265 | 0.993 | 45.05 -> 44.30 | 18.6 / 13.3 | 4 / 4 |
| s9 | 311.61 | 323.17 | 1.037 | 0.260 | 0.261 | 1.004 | 36.05 -> 37.63 | 16.6 / 11.8 | 4 / 4 |

| cell | decode iteration ms p50 A -> B | mean Emit gap ms A -> B | prefill iteration ms A -> B (count per visit) |
|---|---|---|---|
| s8 | 20.7 -> 21.1 | 23.0 -> 23.3 | 76.6 -> 74.5 (16 / 16) |
| s9 | 26.2 -> 25.0 | 28.5 -> 27.2 | 71.1 -> 73.6 (18 / 18) |

| cell | A agg tok/s per visit | B agg tok/s per visit | median B / A | best-visit B / A | A step p50 ms per visit | B step p50 ms per visit | median step B / A |
|---|---|---|---|---|---|---|---|
| s8 | 359 298 350 361 | 320 316 361 353 | 0.950 | 0.999 | 19.7 23.5 20.3 19.2 | 22.2 22.4 19.7 20.2 | 1.057 |
| s9 | 325 274 325 322 | 326 298 331 336 | 1.016 | 1.034 | 25.0 29.7 25.0 25.1 | 24.5 27.3 24.0 24.1 | 0.969 |
