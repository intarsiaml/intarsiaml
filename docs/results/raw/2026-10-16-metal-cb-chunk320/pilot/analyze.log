Rounds per arm (valid/total): 256: 1/1, 320: 1/1

| metric | busy chunk 256 | busy chunk 320 | 320 vs 256 |
|---|---|---|---|
| 4k TTFT, ms (median of rounds) | 2040 | 2084 | +2.2% |
| 4k TTFT, ms (mean +- sd) | 2040 +- - | 2084 +- - |  |
| 4k prefill wall (first chunk to first token), ms (median) | 1969 | 2016 | +2.4% |
| prefill chunks of the 4k prompt (median) | 17 | 14 | -17.6% |
| GPU busy during the prefill (macmon, mean of rounds) | 0.883 | 0.854 | -3.3% |
| GPU MHz during the prefill (mean) | 1430 | 1383 | -3.3% |
| GPU busy over the whole scenario (mean) | 0.933 | 0.883 | -5.4% |
| GPU MHz over the scenario: mean / min | 1511 / 1313 | 1429 / 1323 |  |
| decode-row stall (client gap in a prefill-carrying iteration) p50, ms | 118.6 | 148.2 | +25.0% |
|   p95 | 127.9 | 160.2 | +25.3% |
|   max | 128.1 | 160.3 | +25.1% |
|   n gaps | 119 | 98 | -17.6% |
| decode ITL without prefill (reference) p50, ms | 36.2 | 38.5 | +6.4% |
|   p95 | 37.8 | 39.7 | +5.1% |
|   max | 39.0 | 40.5 | +3.9% |
|   n gaps | 2045 | 2110 | +3.2% |
| server iteration time with a chunk (Decode-to-Decode) p50 / p95 / max, ms | 119.0 / 157.8 / 207.0 | 149.0 / 196.0 / 234.0 |  |
| server iteration time without a chunk p50 / p95 / max, ms | 36.0 / 89.9 / 121.0 | 39.0 / 117.2 / 159.0 |  |
| decode tokens/s delivered during the 4k prefill (all 7 streams, mean) | 61.8 | 50.4 | -18.4% |
| aggregate tok/s from the 4k send to the last completion (mean +- sd) | 167.5 +- - | 160.2 +- - |  |
| time from the 4k send to the last completion, s (mean) | 13.38 | 14.16 | +5.8% |

Paired by round (320 minus 256, same round index, alternating start order):

| metric | per-round deltas | mean delta | rounds where 320 is lower |
|---|---|---|---|
| 4k TTFT ms | +44 | +44 | 0/1 |
| prefill wall ms | +47 | +47 | 0/1 |
| aggregate tok/s | -7.2 | -7.2 | 1/1 |
| time to last completion s | +0.78 | +0.78 | 0/1 |
| GPU busy in prefill | -0.029 | -0.029 | 1/1 |
| stall p95 ms | +32.3 | +32.3 | 0/1 |
| stall max ms | +32.1 | +32.1 | 0/1 |

Per round:

| arm | round | order | big prompt tok | stream prompt tok (min-max) | chunk / decoding at start | chunks | TTFT ms | wall ms | stall p50/p95/max | ITL ref p50 | agg tok/s | GPU busy | MHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 1 | 4305 | 851-1210 | 256 / 7 | 17 | 2040 | 1969 | 118.6/127.9/128.1 | 36.2 | 167.5 | 0.883 | 1430 |
| 320 | 1 | 2 | 4305 | 851-1210 | 320 / 7 | 14 | 2084 | 2016 | 148.2/160.2/160.3 | 38.5 | 160.2 | 0.854 | 1383 |
