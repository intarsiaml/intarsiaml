# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T07:23:43-04:00  wall 114 s  planned rate 0.7 req/s over 100.0 s  seed 12

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 83 | 74 | 9 | 0 (0/0) | 0 | 0 |

offered 0.830 req/s, completed 0.665 req/s, output 95.3 tok/s (over the 111 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 10.84%, goodput 89.16%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 79 | 4196.8 | 4163.6 | 8682.6 | 9886.9 | 10479.7 |
| ITL (ms) | 10524 | 62.7 | 45.3 | 132.6 | 149.4 | 180.5 |
| per-stream decode (tok/s) | 74 | 17.2 | 15.5 | 28.6 | 35.2 | 35.5 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (8 such prompts): n 2233, p50 49.5, p95 142.1, p99 156.3, max 180.5 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 1, p50 381.5, p95 381.5, max 381.5 ms; all other requests: n 78, p50 4274.5, p95 8684.0 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 144 | 11 | 11 | 0 | 0 | 2545.2 | 7470.2 | 71.5 (1) | 45.3 | 133.4 |
| 512 | 526 | 17 | 15 | 0 | 0 | 5194.5 | 7782.6 | - (0) | 45.7 | 132.5 |
| 1024 | 1074 | 31 | 26 | 0 | 0 | 3807.1 | 9333.0 | 381.5 (1) | 45.2 | 133.2 |
| 2048 | 2058 | 15 | 14 | 0 | 0 | 6408.7 | 8512.3 | - (0) | 45.5 | 133.0 |
| 4096 | 4200 | 9 | 8 | 0 | 0 | 2375.1 | 7335.5 | - (0) | 44.2 | 127.0 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 14 | 13 | 0 | 0 | 1 | 63.7 | 830 | 2678 | 35.8 | 107.9 | 136.9 | 0 | 9764 |
| 1 | 20 | 18 | 17 | 0 | 0 | 1 | 96.5 | 5523 | 8544 | 48.6 | 133.6 | 155.1 | 7 | 9765 |
| 2 | 40 | 19 | 19 | 0 | 0 | 0 | 121.6 | 7000 | 9796 | 47.1 | 129.1 | 139.3 | 8 | 9765 |
| 3 | 60 | 9 | 8 | 0 | 0 | 1 | 105.5 | 4115 | 6784 | 45.7 | 129.7 | 139.9 | 7 | 9764 |
| 4 | 80 | 23 | 17 | 0 | 0 | 6 | 79.7 | 1592 | 7036 | 46.8 | 145.1 | 167.8 | 3 | 9766 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 63.1 | - | - | 40.1 | 134.0 | 142.8 | 3 | 9757 |

## Memory

source `footprint_mb`, 37 samples; start 9762 MB, end 9757 MB, max 9766 MB (end - start -5 MB).

Growth fit over t >= 20 s (30 samples, 1.5 min): Theil-Sen slope 1.23 MB/min (OLS 0.26, r2 0.01), fitted growth 2 MB, projected 74 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 25107 MB, end 25107 MB, max 25107 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

38 samples. running: max 8.0 / mean 6.29. queue depth: max 8.0 / mean 2.53. step EWMA ms: max 120.0 / mean 68.53. KV bytes used: max 2666004480.0 / mean 1708848181.89 of 9216 MB. rejected (delta): 0. server TTFT mean 852.7 ms, ITL mean 63.2 ms. prefill chunks (delta) 451.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 246 | 13.4% |
| 2 | 179 | 9.8% |
| 3 | 73 | 4.0% |
| 4 | 57 | 3.1% |
| 5 | 43 | 2.3% |
| 6 | 147 | 8.0% |
| 7 | 389 | 21.2% |
| 8 | 697 | 38.1% |

mean rows per decode pass 5.71

## SLO

no thresholds given (informational run).
