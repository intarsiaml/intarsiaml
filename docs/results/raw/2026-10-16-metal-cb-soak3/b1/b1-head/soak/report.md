# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T13:40:32-04:00  wall 360 s  planned rate 0.3 req/s over 360.0 s  seed 31

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 80 | 80 | 0 | 0 (0/0) | 0 | 0 |

offered 0.222 req/s, completed 0.237 req/s, output 9.8 tok/s (over the 337 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 80 | 612.0 | 302.5 | 2420.6 | 3145.7 | 3822.9 |
| ITL (ms) | 3214 | 28.0 | 28.0 | 28.7 | 29.1 | 55.6 |
| per-stream decode (tok/s) | 80 | 35.7 | 35.7 | 36.1 | 36.3 | 36.3 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (3 such prompts): n 70, p50 28.2, p95 28.7, p99 28.8, max 28.9 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 21, p50 174.5, p95 489.2, max 1169.7 ms; all other requests: n 59, p50 516.4, p95 2481.7 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 143 | 19 | 19 | 0 | 0 | 84.1 | 1204.6 | 81.0 (12) | 27.8 | 28.3 |
| 512 | 532 | 27 | 27 | 0 | 0 | 176.6 | 1084.9 | 160.9 (17) | 27.9 | 28.4 |
| 1024 | 1074 | 19 | 19 | 0 | 0 | 302.7 | 2481.7 | 289.9 (14) | 28.0 | 28.5 |
| 2048 | 2119 | 12 | 12 | 0 | 0 | 572.1 | 2223.0 | 532.9 (9) | 28.4 | 28.9 |
| 4096 | 4102 | 3 | 3 | 0 | 0 | 1169.7 | 3557.6 | 1103.5 (2) | 28.8 | 29.2 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 12 | 12 | 0 | 0 | 0 | 8.3 | 296 | 3130 | 28.1 | 28.9 | 29.2 | 2 | 1691 |
| 1 | 60 | 13 | 13 | 0 | 0 | 0 | 7.4 | 302 | 789 | 27.8 | 28.5 | 28.9 | 0 | 1547 |
| 2 | 120 | 11 | 11 | 0 | 0 | 0 | 8.2 | 486 | 1799 | 27.9 | 28.4 | 28.6 | 1 | 1693 |
| 3 | 180 | 14 | 14 | 0 | 0 | 0 | 9.3 | 309 | 1742 | 28.1 | 28.9 | 29.2 | 0 | 1548 |
| 4 | 240 | 12 | 12 | 0 | 0 | 0 | 8.8 | 263 | 894 | 28.0 | 28.5 | 28.8 | 1 | 1550 |
| 5 | 300 | 18 | 18 | 0 | 0 | 0 | 12.8 | 304 | 1849 | 28.0 | 28.6 | 28.9 | 0 | 1549 |
| 6 | 360 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 72 samples; start 1689 MB, end 1549 MB, max 1694 MB (end - start -139 MB).

Growth fit over t >= 72 s (57 samples, 4.7 min): Theil-Sen slope 0.51 MB/min (OLS 10.41, r2 0.04), fitted growth 2 MB, projected 31 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

72 samples. running: max 1.0 / mean 0.26. queue depth: max 2.0 / mean 0.08. step EWMA ms: -. KV bytes used: max 646594560.0 / mean 166420480.00 of 1152 MB. rejected (delta): 0. server TTFT mean 257.7 ms, ITL mean 28.0 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
