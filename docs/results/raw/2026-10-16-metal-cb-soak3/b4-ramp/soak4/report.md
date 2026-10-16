# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T12:50:05-04:00  wall 105 s  planned rate 0.6 req/s over 100.0 s  seed 44

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 47 | 43 | 4 | 0 (0/0) | 0 | 0 |

offered 0.470 req/s, completed 0.415 req/s, output 60.5 tok/s (over the 104 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 8.51%, goodput 91.49%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 46 | 1703.6 | 754.8 | 7421.2 | 9883.2 | 11588.0 |
| ITL (ms) | 6223 | 38.8 | 31.1 | 122.6 | 139.7 | 181.7 |
| per-stream decode (tok/s) | 43 | 26.9 | 26.5 | 33.3 | 33.5 | 33.6 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (4 such prompts): n 643, p50 32.9, p95 139.6, p99 140.9, max 143.9 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 1, p50 73.7, p95 73.7, max 73.7 ms; all other requests: n 45, p50 778.2, p95 7476.1 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 142 | 5 | 5 | 0 | 0 | 115.0 | 631.7 | 73.7 (1) | 29.7 | 31.9 |
| 512 | 545 | 11 | 9 | 0 | 0 | 349.2 | 844.8 | - (0) | 30.9 | 126.8 |
| 1024 | 1107 | 19 | 18 | 0 | 0 | 698.2 | 6707.4 | 297.9 (1) | 31.3 | 122.7 |
| 2048 | 2133 | 8 | 7 | 0 | 0 | 1519.3 | 9415.8 | - (0) | 30.5 | 115.8 |
| 4096 | 4217 | 4 | 4 | 0 | 0 | 2251.2 | 6993.8 | 1940.1 (1) | 33.5 | 130.8 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 7 | 7 | 0 | 0 | 0 | 39.5 | 703 | 2602 | 29.8 | 32.4 | 120.2 | 1 | 5154 |
| 1 | 20 | 7 | 7 | 0 | 0 | 0 | 56.5 | 806 | 1904 | 31.7 | 132.4 | 140.7 | 0 | 5154 |
| 2 | 40 | 10 | 7 | 0 | 0 | 3 | 52.0 | 486 | 1435 | 29.3 | 98.9 | 131.5 | 0 | 5155 |
| 3 | 60 | 12 | 11 | 0 | 0 | 1 | 67.3 | 4495 | 9694 | 33.0 | 132.2 | 140.8 | 4 | 5156 |
| 4 | 80 | 11 | 11 | 0 | 0 | 0 | 75.5 | 639 | 1267 | 31.5 | 122.7 | 135.0 | 1 | 5156 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 22.6 | - | - | 29.9 | 32.3 | 33.0 | 0 | 5156 |

## Memory

source `footprint_mb`, 35 samples; start 5154 MB, end 5156 MB, max 5156 MB (end - start 2 MB).

Growth fit over t >= 20 s (28 samples, 1.4 min): Theil-Sen slope 1.32 MB/min (OLS 1.82, r2 0.18), fitted growth 2 MB, projected 79 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 20499 MB, end 20499 MB, max 20499 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

35 samples. running: max 4.0 / mean 2.57. queue depth: max 4.0 / mean 0.46. step EWMA ms: max 110.8 / mean 45.82. KV bytes used: max 1827422208.0 / mean 839226865.37 of 4608 MB. rejected (delta): 0. server TTFT mean 780.2 ms, ITL mean 39.8 ms. prefill chunks (delta) 241.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 635 | 25.7% |
| 2 | 686 | 27.8% |
| 3 | 595 | 24.1% |
| 4 | 555 | 22.5% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |

mean rows per decode pass 2.43

## SLO

no thresholds given (informational run).
