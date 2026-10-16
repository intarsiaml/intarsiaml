# cbsoak report

http://127.0.0.1:8143  model `Qwen3-8B`  started 2026-10-16T15:29:26-04:00  wall 605 s  planned rate 0.8 req/s over 600.0 s  seed 31

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 459 | 406 | 53 | 0 (0/0) | 0 | 0 |

offered 0.765 req/s, completed 0.678 req/s, output 93.2 tok/s (over the 599 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 11.55%, goodput 88.45%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 440 | 1846.2 | 1129.1 | 5880.6 | 9771.8 | 12261.9 |
| ITL (ms) | 55339 | 52.2 | 40.6 | 117.3 | 126.8 | 230.0 |
| per-stream decode (tok/s) | 406 | 20.8 | 19.1 | 33.7 | 35.9 | 36.8 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (44 such prompts): n 8711, p50 97.7, p95 126.6, p99 133.2, max 229.3 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 2, p50 658.5, p95 1092.2, max 1140.4 ms; all other requests: n 438, p50 1129.6, p95 5892.6 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 146 | 84 | 75 | 0 | 0 | 174.6 | 5409.5 | 76.3 (4) | 35.7 | 114.4 |
| 512 | 541 | 120 | 107 | 0 | 0 | 971.0 | 5727.9 | 175.6 (3) | 40.7 | 117.7 |
| 1024 | 1080 | 117 | 101 | 0 | 0 | 699.5 | 4486.1 | 318.2 (2) | 40.8 | 118.1 |
| 2048 | 2124 | 92 | 80 | 0 | 0 | 1346.8 | 6602.8 | 583.2 (2) | 41.0 | 117.5 |
| 4096 | 4137 | 46 | 43 | 0 | 0 | 2336.0 | 5201.3 | 1168.2 (2) | 41.8 | 117.7 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 33 | 30 | 0 | 0 | 3 | 67.0 | 599 | 2146 | 30.0 | 104.8 | 110.8 | 1 | 9764 |
| 1 | 60 | 34 | 30 | 0 | 0 | 4 | 54.0 | 303 | 1482 | 28.7 | 94.1 | 101.7 | 0 | 9764 |
| 2 | 120 | 46 | 40 | 0 | 0 | 6 | 91.5 | 1041 | 3047 | 40.2 | 112.5 | 122.4 | 3 | 9767 |
| 3 | 180 | 54 | 47 | 0 | 0 | 7 | 116.8 | 2560 | 10782 | 41.6 | 117.4 | 125.4 | 7 | 9768 |
| 4 | 240 | 42 | 37 | 0 | 0 | 5 | 81.3 | 935 | 3647 | 36.7 | 116.7 | 128.9 | 2 | 9769 |
| 5 | 300 | 42 | 36 | 0 | 0 | 6 | 90.8 | 880 | 2021 | 34.1 | 113.0 | 119.6 | 2 | 9770 |
| 6 | 360 | 52 | 45 | 0 | 0 | 7 | 89.7 | 880 | 3521 | 40.6 | 121.4 | 127.9 | 2 | 9772 |
| 7 | 420 | 56 | 47 | 0 | 0 | 9 | 119.1 | 2460 | 5496 | 42.4 | 121.1 | 128.8 | 7 | 9772 |
| 8 | 480 | 51 | 47 | 0 | 0 | 4 | 99.5 | 1105 | 4876 | 40.6 | 117.5 | 126.0 | 2 | 9774 |
| 9 | 540 | 49 | 47 | 0 | 0 | 2 | 116.8 | 2835 | 6309 | 42.7 | 122.3 | 133.3 | 7 | 9776 |
| 10 | 600 | 0 | 0 | 0 | 0 | 0 | 3.1 | - | - | 28.7 | 29.8 | 30.4 | 0 | 9775 |

## Memory

source `footprint_mb`, 120 samples; start 3998 MB, end 9775 MB, max 9776 MB (end - start 5776 MB).

Growth fit over t >= 120 s (96 samples, 8.0 min): Theil-Sen slope 1.28 MB/min (OLS 1.94, r2 0.05), fitted growth 10 MB, projected 77 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 25107 MB, end 25107 MB, max 25107 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

121 samples. running: max 8.0 / mean 5.30. queue depth: max 7.0 / mean 0.93. step EWMA ms: max 113.3 / mean 53.99. KV bytes used: max 2536538112.0 / mean 1614367786.31 of 9216 MB. rejected (delta): 0. server TTFT mean 795.3 ms, ITL mean 52.5 ms. prefill chunks (delta) 2346.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 1467 | 12.3% |
| 2 | 1830 | 15.4% |
| 3 | 1341 | 11.3% |
| 4 | 1407 | 11.8% |
| 5 | 1063 | 8.9% |
| 6 | 1013 | 8.5% |
| 7 | 1317 | 11.1% |
| 8 | 2479 | 20.8% |

mean rows per decode pass 4.63

## SLO

no thresholds given (informational run).
