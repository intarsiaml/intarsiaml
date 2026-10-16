# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T13:26:28-04:00  wall 1802 s  planned rate 0.33 req/s over 1800.0 s  seed 21

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 586 | 547 | 39 | 0 (0/0) | 0 | 0 |

offered 0.326 req/s, completed 0.304 req/s, output 41.4 tok/s (over the 1802 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 6.66%, goodput 93.34%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 580 | 657.2 | 401.0 | 1976.1 | 3426.5 | 4740.7 |
| ITL (ms) | 74000 | 32.6 | 28.8 | 84.5 | 102.1 | 127.7 |
| per-stream decode (tok/s) | 547 | 31.5 | 32.7 | 36.2 | 36.7 | 36.9 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (51 such prompts): n 1923, p50 96.1, p95 112.9, p99 122.5, max 127.4 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 55, p50 278.0, p95 1009.0, max 1215.1 ms; all other requests: n 525, p50 464.4, p95 2019.9 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 144 | 120 | 113 | 0 | 0 | 102.9 | 1183.4 | 78.4 (30) | 28.7 | 81.9 |
| 512 | 548 | 151 | 139 | 0 | 0 | 273.1 | 1706.3 | 163.5 (30) | 28.6 | 85.9 |
| 1024 | 1071 | 168 | 161 | 0 | 0 | 471.0 | 1736.7 | 285.4 (44) | 28.7 | 36.1 |
| 2048 | 2073 | 92 | 86 | 0 | 0 | 809.5 | 2857.2 | 565.1 (22) | 29.2 | 93.4 |
| 4096 | 4172 | 55 | 48 | 0 | 0 | 1639.0 | 3081.4 | 1096.3 (16) | 29.8 | 94.7 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 17 | 15 | 0 | 0 | 2 | 30.5 | 298 | 1354 | 28.7 | 31.4 | 96.8 | 0 | 5154 |
| 1 | 60 | 26 | 26 | 0 | 0 | 0 | 60.1 | 511 | 3098 | 29.7 | 98.1 | 111.3 | 2 | 5157 |
| 2 | 120 | 14 | 13 | 0 | 0 | 1 | 36.3 | 158 | 726 | 28.1 | 30.2 | 92.9 | 0 | 5157 |
| 3 | 180 | 24 | 24 | 0 | 0 | 0 | 54.0 | 597 | 3644 | 28.8 | 95.0 | 101.0 | 1 | 5158 |
| 4 | 240 | 25 | 24 | 0 | 0 | 1 | 44.4 | 416 | 1581 | 28.4 | 94.9 | 101.9 | 2 | 5015 |
| 5 | 300 | 21 | 20 | 0 | 0 | 1 | 41.6 | 336 | 1837 | 28.7 | 77.8 | 101.1 | 0 | 5161 |
| 6 | 360 | 22 | 20 | 0 | 0 | 2 | 31.2 | 368 | 1323 | 28.5 | 86.0 | 100.0 | 0 | 5161 |
| 7 | 420 | 16 | 15 | 0 | 0 | 1 | 37.6 | 270 | 2116 | 28.9 | 94.6 | 103.4 | 1 | 5162 |
| 8 | 480 | 28 | 27 | 0 | 0 | 1 | 69.2 | 610 | 3097 | 29.5 | 96.5 | 112.1 | 1 | 5163 |
| 9 | 540 | 13 | 10 | 0 | 0 | 3 | 26.5 | 288 | 949 | 27.7 | 29.7 | 95.8 | 1 | 5019 |
| 10 | 600 | 16 | 15 | 0 | 0 | 1 | 30.3 | 394 | 902 | 28.4 | 93.4 | 98.5 | 0 | 5156 |
| 11 | 660 | 8 | 7 | 0 | 0 | 1 | 23.0 | 415 | 750 | 28.5 | 37.5 | 96.8 | 0 | 5165 |
| 12 | 720 | 19 | 19 | 0 | 0 | 0 | 34.8 | 481 | 1971 | 28.7 | 95.5 | 99.2 | 1 | 5166 |
| 13 | 780 | 21 | 15 | 0 | 0 | 6 | 39.6 | 353 | 1904 | 28.6 | 95.9 | 103.2 | 0 | 5022 |
| 14 | 840 | 18 | 17 | 0 | 0 | 1 | 33.2 | 358 | 1063 | 28.5 | 30.4 | 99.8 | 0 | 5159 |
| 15 | 900 | 22 | 21 | 0 | 0 | 1 | 58.8 | 499 | 1596 | 29.8 | 95.0 | 104.9 | 1 | 5169 |
| 16 | 960 | 14 | 13 | 0 | 0 | 1 | 26.6 | 655 | 1186 | 28.1 | 30.1 | 97.7 | 0 | 5169 |
| 17 | 1020 | 15 | 14 | 0 | 0 | 1 | 26.2 | 284 | 1265 | 28.1 | 29.9 | 97.7 | 1 | 5025 |
| 18 | 1080 | 20 | 18 | 0 | 0 | 2 | 45.5 | 413 | 1609 | 28.9 | 31.4 | 98.3 | 1 | 5026 |
| 19 | 1140 | 19 | 19 | 0 | 0 | 0 | 38.0 | 302 | 2101 | 28.1 | 30.4 | 96.5 | 1 | 5172 |
| 20 | 1200 | 19 | 18 | 0 | 0 | 1 | 33.6 | 308 | 1040 | 28.1 | 30.3 | 95.4 | 0 | 5172 |
| 21 | 1260 | 19 | 19 | 0 | 0 | 0 | 39.9 | 271 | 731 | 28.2 | 29.8 | 98.9 | 0 | 5173 |
| 22 | 1320 | 20 | 18 | 0 | 0 | 2 | 37.9 | 617 | 1637 | 28.7 | 80.1 | 99.5 | 0 | 5166 |
| 23 | 1380 | 20 | 19 | 0 | 0 | 1 | 43.4 | 389 | 1622 | 28.7 | 31.0 | 97.5 | 2 | 5175 |
| 24 | 1440 | 18 | 18 | 0 | 0 | 0 | 40.0 | 294 | 1038 | 28.5 | 30.4 | 99.1 | 0 | 5176 |
| 25 | 1500 | 20 | 18 | 0 | 0 | 2 | 48.2 | 873 | 2124 | 30.5 | 96.4 | 104.9 | 2 | 5033 |
| 26 | 1560 | 23 | 23 | 0 | 0 | 0 | 49.0 | 324 | 1017 | 29.0 | 31.3 | 98.5 | 0 | 5178 |
| 27 | 1620 | 28 | 23 | 0 | 0 | 5 | 53.2 | 295 | 1400 | 29.0 | 89.1 | 101.5 | 0 | 5179 |
| 28 | 1680 | 17 | 17 | 0 | 0 | 0 | 48.0 | 400 | 1581 | 29.1 | 31.6 | 99.1 | 0 | 5180 |
| 29 | 1740 | 24 | 22 | 0 | 0 | 2 | 59.3 | 1067 | 4000 | 29.3 | 97.2 | 113.9 | 1 | 5182 |
| 30 | 1800 | 0 | 0 | 0 | 0 | 0 | 3.3 | - | - | 30.1 | 30.5 | 30.5 | - | - |

## Memory

source `footprint_mb`, 352 samples; start 5153 MB, end 5182 MB, max 5182 MB (end - start 29 MB).

Growth fit over t >= 360 s (286 samples, 24.0 min): Theil-Sen slope 0.85 MB/min (OLS 1.88, r2 0.05), fitted growth 20 MB, projected 51 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 20499 MB, end 20499 MB, max 20499 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

360 samples. running: max 4.0 / mean 1.48. queue depth: max 2.0 / mean 0.07. step EWMA ms: max 101.1 / mean 32.52. KV bytes used: max 1638088704.0 / mean 769805516.80 of 4608 MB. rejected (delta): 0. server TTFT mean 493.9 ms, ITL mean 32.6 ms. prefill chunks (delta) 2361.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 20073 | 49.2% |
| 2 | 11823 | 29.0% |
| 3 | 5554 | 13.6% |
| 4 | 3313 | 8.1% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |

mean rows per decode pass 1.81

## SLO

| check | limit | value | |
|---|---|---|---|
| TTFT p50 (ms) | 1500.0 | 401.013 | ok |
| TTFT p95 (ms) | 5000.0 | 1976.098 | ok |
| TTFT p99 (ms) | 8000.0 | 3426.480 | ok |
| ITL p95 (ms) | 150.0 | 84.508 | ok |
| ITL p95 (ms) of gaps overlapping a >=3000-token prompt's prefill | 150.0 | 112.876 | ok |
| error rate | 0.001 | 0.000 | ok |
| shed rate | 0.01 | 0.000 | ok |
| memory growth (fitted, MB) | 100.0 | 20.452 | ok |
| device memory growth (fitted, MB; /metrics total - free, system-wide) | 100.0 | 0.000 | ok |

**PASS**
