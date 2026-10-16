# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T10:33:50-04:00  wall 1803 s  planned rate 0.5 req/s over 1800.0 s  seed 21

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 897 | 835 | 62 | 0 (0/0) | 0 | 0 |

offered 0.498 req/s, completed 0.463 req/s, output 63.8 tok/s (over the 1803 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 6.91%, goodput 93.09%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 889 | 659.5 | 445.8 | 1957.8 | 2729.7 | 4336.8 |
| ITL (ms) | 114178 | 37.4 | 30.1 | 100.2 | 113.9 | 243.3 |
| per-stream decode (tok/s) | 835 | 27.8 | 28.3 | 34.9 | 35.8 | 36.2 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (83 such prompts): n 4019, p50 104.1, p95 121.4, p99 127.2, max 193.3 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 22, p50 379.0, p95 1064.3, max 1145.6 ms; all other requests: n 867, p50 460.5, p95 1959.9 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 144 | 197 | 184 | 0 | 0 | 109.6 | 1135.9 | 242.2 (21) | 29.9 | 99.0 |
| 512 | 543 | 231 | 212 | 0 | 0 | 287.6 | 1282.9 | 192.4 (13) | 29.8 | 100.4 |
| 1024 | 1067 | 252 | 241 | 0 | 0 | 485.2 | 1558.7 | 385.7 (30) | 30.0 | 100.0 |
| 2048 | 2084 | 129 | 119 | 0 | 0 | 891.0 | 2051.5 | 587.1 (11) | 30.5 | 101.3 |
| 4096 | 4146 | 88 | 79 | 0 | 0 | 1747.8 | 2774.4 | 1175.1 (11) | 31.0 | 102.3 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 29 | 27 | 0 | 0 | 2 | 48.3 | 414 | 1485 | 29.5 | 95.4 | 101.6 | 0 | 9767 |
| 1 | 60 | 31 | 30 | 0 | 0 | 1 | 78.4 | 598 | 3291 | 31.6 | 114.4 | 123.9 | 1 | 9768 |
| 2 | 120 | 37 | 37 | 0 | 0 | 0 | 83.6 | 614 | 2425 | 30.6 | 103.5 | 113.6 | 1 | 9770 |
| 3 | 180 | 31 | 29 | 0 | 0 | 2 | 56.0 | 494 | 2055 | 29.5 | 97.3 | 106.4 | 1 | 9771 |
| 4 | 240 | 28 | 26 | 0 | 0 | 2 | 45.3 | 401 | 1367 | 29.0 | 99.3 | 108.3 | 1 | 9771 |
| 5 | 300 | 39 | 36 | 0 | 0 | 3 | 94.3 | 489 | 2301 | 32.1 | 108.0 | 117.8 | 0 | 9773 |
| 6 | 360 | 18 | 16 | 0 | 0 | 2 | 35.0 | 288 | 845 | 28.2 | 29.6 | 95.8 | 1 | 9774 |
| 7 | 420 | 18 | 16 | 0 | 0 | 2 | 45.8 | 466 | 1091 | 28.8 | 95.8 | 100.5 | 0 | 9774 |
| 8 | 480 | 30 | 27 | 0 | 0 | 3 | 58.4 | 706 | 2227 | 30.2 | 104.6 | 111.9 | 0 | 9776 |
| 9 | 540 | 28 | 24 | 0 | 0 | 4 | 53.8 | 464 | 1952 | 29.3 | 99.9 | 108.0 | 0 | 9777 |
| 10 | 600 | 31 | 29 | 0 | 0 | 2 | 71.4 | 565 | 1375 | 30.8 | 99.9 | 105.8 | 0 | 9778 |
| 11 | 660 | 21 | 20 | 0 | 0 | 1 | 37.9 | 365 | 1548 | 28.7 | 29.8 | 98.0 | 0 | 9779 |
| 12 | 720 | 32 | 30 | 0 | 0 | 2 | 69.1 | 534 | 1825 | 30.0 | 99.2 | 109.7 | 1 | 9780 |
| 13 | 780 | 30 | 29 | 0 | 0 | 1 | 51.9 | 299 | 1195 | 28.7 | 32.4 | 97.7 | 0 | 9782 |
| 14 | 840 | 28 | 26 | 0 | 0 | 2 | 51.1 | 487 | 2073 | 29.0 | 96.2 | 103.9 | 1 | 9782 |
| 15 | 900 | 28 | 27 | 0 | 0 | 1 | 68.6 | 386 | 1403 | 29.7 | 97.3 | 108.1 | 0 | 9783 |
| 16 | 960 | 31 | 29 | 0 | 0 | 2 | 66.6 | 497 | 1618 | 30.4 | 97.8 | 107.5 | 0 | 9785 |
| 17 | 1020 | 35 | 31 | 0 | 0 | 4 | 72.0 | 412 | 1725 | 30.3 | 97.5 | 104.6 | 0 | 9785 |
| 18 | 1080 | 34 | 33 | 0 | 0 | 1 | 81.1 | 384 | 1565 | 30.8 | 99.5 | 107.5 | 0 | 9787 |
| 19 | 1140 | 32 | 30 | 0 | 0 | 2 | 81.4 | 986 | 2280 | 30.8 | 109.5 | 119.4 | 1 | 9788 |
| 20 | 1200 | 31 | 30 | 0 | 0 | 1 | 68.3 | 402 | 1987 | 31.0 | 104.3 | 118.8 | 0 | 9790 |
| 21 | 1260 | 41 | 38 | 0 | 0 | 3 | 84.3 | 314 | 1737 | 30.9 | 103.7 | 115.0 | 1 | 9791 |
| 22 | 1320 | 29 | 26 | 0 | 0 | 3 | 68.5 | 412 | 1364 | 30.1 | 97.3 | 119.9 | 0 | 9791 |
| 23 | 1380 | 39 | 37 | 0 | 0 | 2 | 86.6 | 485 | 1891 | 31.2 | 106.5 | 120.8 | 1 | 9794 |
| 24 | 1440 | 31 | 29 | 0 | 0 | 2 | 69.3 | 347 | 1574 | 30.3 | 100.5 | 113.6 | 1 | 9795 |
| 25 | 1500 | 29 | 29 | 0 | 0 | 0 | 57.5 | 480 | 1703 | 29.9 | 97.0 | 105.6 | 1 | 9650 |
| 26 | 1560 | 26 | 23 | 0 | 0 | 3 | 70.5 | 505 | 1780 | 30.9 | 98.7 | 108.1 | 0 | 9797 |
| 27 | 1620 | 25 | 23 | 0 | 0 | 2 | 50.2 | 297 | 1785 | 28.9 | 84.1 | 101.9 | 1 | 9797 |
| 28 | 1680 | 33 | 31 | 0 | 0 | 2 | 64.8 | 309 | 1337 | 30.7 | 99.0 | 105.7 | 0 | 9799 |
| 29 | 1740 | 22 | 17 | 0 | 0 | 5 | 45.9 | 400 | 1149 | 30.0 | 97.2 | 108.7 | 0 | 9654 |
| 30 | 1800 | 0 | 0 | 0 | 0 | 0 | 1.9 | - | - | 27.6 | 27.9 | 28.1 | 0 | - |

## Memory

source `footprint_mb`, 357 samples; start 9765 MB, end 9654 MB, max 9800 MB (end - start -111 MB).

Growth fit over t >= 360 s (285 samples, 23.9 min): Theil-Sen slope 1.15 MB/min (OLS 1.23, r2 0.06), fitted growth 27 MB, projected 69 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 25107 MB, end 25107 MB, max 25107 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

361 samples. running: max 8.0 / mean 2.67. queue depth: max 1.0 / mean 0.05. step EWMA ms: max 112.7 / mean 40.03. KV bytes used: max 2739142656.0 / mean 1669749672.07 of 9216 MB. rejected (delta): 0. server TTFT mean 626.3 ms, ITL mean 37.5 ms. prefill chunks (delta) 4135.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 12609 | 28.6% |
| 2 | 12561 | 28.5% |
| 3 | 8370 | 19.0% |
| 4 | 5166 | 11.7% |
| 5 | 2866 | 6.5% |
| 6 | 1606 | 3.6% |
| 7 | 629 | 1.4% |
| 8 | 294 | 0.7% |

mean rows per decode pass 2.59

## SLO

| check | limit | value | |
|---|---|---|---|
| TTFT p50 (ms) | 1500.0 | 445.804 | ok |
| TTFT p95 (ms) | 5000.0 | 1957.772 | ok |
| TTFT p99 (ms) | 8000.0 | 2729.658 | ok |
| ITL p95 (ms) | 150.0 | 100.248 | ok |
| ITL p95 (ms) of gaps overlapping a >=3000-token prompt's prefill | 150.0 | 121.378 | ok |
| error rate | 0.001 | 0.000 | ok |
| shed rate | 0.01 | 0.000 | ok |
| memory growth (fitted, MB) | 100.0 | 27.493 | ok |
| device memory growth (fitted, MB; /metrics total - free, system-wide) | 100.0 | 0.000 | ok |

**PASS**
