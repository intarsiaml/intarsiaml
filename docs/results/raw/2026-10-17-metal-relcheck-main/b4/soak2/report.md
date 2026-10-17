# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-17T16:08:27-04:00  wall 604 s  planned rate 0.33 req/s over 600.0 s  seed 21

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 206 | 194 | 12 | 0 (0/0) | 0 | 0 |

offered 0.343 req/s, completed 0.321 req/s, output 43.1 tok/s (over the 604 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 5.83%, goodput 94.17%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 205 | 757.7 | 405.6 | 2515.0 | 3678.2 | 4648.2 |
| ITL (ms) | 25830 | 33.6 | 29.0 | 94.2 | 110.6 | 162.7 |
| per-stream decode (tok/s) | 194 | 30.6 | 31.6 | 36.5 | 36.7 | 36.8 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (19 such prompts): n 889, p50 95.5, p95 117.0, p99 128.0, max 157.5 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 19, p50 253.8, p95 573.8, max 598.4 ms; all other requests: n 186, p50 473.9, p95 2795.0 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 146 | 37 | 34 | 0 | 0 | 99.9 | 665.8 | 81.9 (10) | 28.5 | 33.2 |
| 512 | 540 | 54 | 48 | 0 | 0 | 270.6 | 2375.0 | 170.1 (9) | 28.6 | 93.9 |
| 1024 | 1060 | 59 | 58 | 0 | 0 | 438.2 | 1652.9 | 300.3 (17) | 29.1 | 89.7 |
| 2048 | 2045 | 37 | 36 | 0 | 0 | 911.3 | 3199.9 | 549.2 (6) | 29.4 | 96.3 |
| 4096 | 4117 | 19 | 18 | 0 | 0 | 1785.4 | 3107.7 | 1119.3 (5) | 30.6 | 100.6 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 17 | 15 | 0 | 0 | 2 | 30.6 | 293 | 1349 | 28.6 | 31.0 | 96.9 | 2 | 5146 |
| 1 | 60 | 26 | 26 | 0 | 0 | 0 | 59.9 | 504 | 2931 | 29.4 | 97.8 | 113.4 | 2 | 5157 |
| 2 | 120 | 14 | 13 | 0 | 0 | 1 | 36.4 | 152 | 736 | 27.7 | 30.6 | 92.5 | 0 | 5157 |
| 3 | 180 | 24 | 24 | 0 | 0 | 0 | 53.6 | 648 | 3644 | 29.3 | 95.6 | 107.1 | 1 | 5158 |
| 4 | 240 | 25 | 24 | 0 | 0 | 1 | 44.6 | 559 | 1845 | 31.5 | 109.3 | 132.4 | 2 | 5159 |
| 5 | 300 | 21 | 20 | 0 | 0 | 1 | 41.6 | 365 | 2162 | 29.3 | 63.0 | 119.7 | 0 | 5152 |
| 6 | 360 | 22 | 20 | 0 | 0 | 2 | 31.4 | 355 | 1295 | 28.0 | 82.2 | 102.8 | 0 | 5161 |
| 7 | 420 | 16 | 15 | 0 | 0 | 1 | 37.6 | 278 | 2095 | 28.7 | 94.7 | 102.9 | 1 | 5162 |
| 8 | 480 | 28 | 27 | 0 | 0 | 1 | 69.1 | 622 | 2934 | 29.4 | 95.5 | 105.4 | 1 | 5163 |
| 9 | 540 | 13 | 10 | 0 | 0 | 3 | 26.5 | 265 | 926 | 27.5 | 28.8 | 95.5 | 0 | 5019 |
| 10 | 600 | 0 | 0 | 0 | 0 | 0 | 2.5 | - | - | 27.3 | 27.8 | 28.0 | 0 | 5164 |

## Memory

source `footprint_mb`, 120 samples; start 5152 MB, end 5164 MB, max 5164 MB (end - start 11 MB).

Growth fit over t >= 120 s (96 samples, 8.0 min): Theil-Sen slope 0.91 MB/min (OLS -2.40, r2 0.01), fitted growth 7 MB, projected 54 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 20499 MB, end 20499 MB, max 20499 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

121 samples. running: max 4.0 / mean 1.56. queue depth: max 2.0 / mean 0.10. step EWMA ms: max 112.5 / mean 33.45. KV bytes used: max 1572470784.0 / mean 713754065.45 of 4608 MB. rejected (delta): 0. server TTFT mean 538.2 ms, ITL mean 33.7 ms. prefill chunks (delta) 868.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 6412 | 47.3% |
| 2 | 3511 | 25.9% |
| 3 | 2209 | 16.3% |
| 4 | 1416 | 10.5% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |

mean rows per decode pass 1.90

## SLO

| check | limit | value | |
|---|---|---|---|
| TTFT p50 (ms) | 1500.0 | 405.631 | ok |
| TTFT p95 (ms) | 5000.0 | 2514.997 | ok |
| TTFT p99 (ms) | 8000.0 | 3678.223 | ok |
| ITL p95 (ms) | 150.0 | 94.185 | ok |
| ITL p95 (ms) of gaps overlapping a >=3000-token prompt's prefill | 150.0 | 117.029 | ok |
| error rate | 0.001 | 0.000 | ok |
| shed rate | 0.01 | 0.000 | ok |
| memory growth (fitted, MB) | 100.0 | 7.246 | ok |
| device memory growth (fitted, MB; /metrics total - free, system-wide) | 100.0 | 0.000 | ok |

**PASS**
