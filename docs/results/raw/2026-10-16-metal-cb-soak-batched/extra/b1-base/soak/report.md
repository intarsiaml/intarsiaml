# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T10:43:51-04:00  wall 360 s  planned rate 0.3 req/s over 360.0 s  seed 31

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 80 | 80 | 0 | 0 (0/0) | 0 | 0 |

offered 0.222 req/s, completed 0.237 req/s, output 9.8 tok/s (over the 338 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 80 | 739.0 | 432.6 | 2500.0 | 3389.9 | 4013.8 |
| ITL (ms) | 3214 | 28.3 | 28.3 | 28.9 | 29.3 | 55.4 |
| per-stream decode (tok/s) | 80 | 35.4 | 35.4 | 36.0 | 36.0 | 36.1 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (3 such prompts): n 76, p50 28.3, p95 28.7, p99 29.2, max 29.5 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 21, p50 317.2, p95 642.4, max 1289.0 ms; all other requests: n 59, p50 611.1, p95 2589.2 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 143 | 19 | 19 | 0 | 0 | 274.8 | 1412.6 | 262.6 (12) | 28.1 | 28.5 |
| 512 | 532 | 27 | 27 | 0 | 0 | 321.2 | 1249.1 | 312.5 (17) | 28.2 | 28.8 |
| 1024 | 1074 | 19 | 19 | 0 | 0 | 431.6 | 2589.2 | 413.1 (14) | 28.3 | 28.9 |
| 2048 | 2119 | 12 | 12 | 0 | 0 | 641.3 | 2431.1 | 639.2 (9) | 28.5 | 29.1 |
| 4096 | 4102 | 3 | 3 | 0 | 0 | 1289.0 | 3741.3 | 1233.0 (2) | 28.9 | 29.4 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 12 | 12 | 0 | 0 | 0 | 8.2 | 465 | 3279 | 28.3 | 29.0 | 29.4 | 3 | 1690 |
| 1 | 60 | 13 | 13 | 0 | 0 | 0 | 7.5 | 402 | 835 | 28.2 | 28.9 | 29.2 | 0 | 1546 |
| 2 | 120 | 11 | 11 | 0 | 0 | 0 | 8.3 | 559 | 1883 | 28.2 | 28.7 | 29.2 | 1 | 1691 |
| 3 | 180 | 14 | 14 | 0 | 0 | 0 | 9.3 | 414 | 2019 | 28.3 | 28.9 | 29.2 | 0 | 1547 |
| 4 | 240 | 12 | 12 | 0 | 0 | 0 | 8.8 | 381 | 1056 | 28.1 | 28.6 | 28.9 | 1 | 1692 |
| 5 | 300 | 18 | 18 | 0 | 0 | 0 | 12.8 | 438 | 2007 | 28.3 | 28.9 | 29.3 | 0 | 1548 |
| 6 | 360 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 72 samples; start 1689 MB, end 1548 MB, max 1693 MB (end - start -142 MB).

Growth fit over t >= 72 s (57 samples, 4.7 min): Theil-Sen slope 0.56 MB/min (OLS 10.25, r2 0.04), fitted growth 3 MB, projected 33 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

72 samples. running: max 1.0 / mean 0.32. queue depth: max 3.0 / mean 0.11. step EWMA ms: -. KV bytes used: max 646594560.0 / mean 172996608.00 of 1152 MB. rejected (delta): 0. server TTFT mean 340.1 ms, ITL mean 28.3 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
