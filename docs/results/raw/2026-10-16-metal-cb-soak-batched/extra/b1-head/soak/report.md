# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T10:50:41-04:00  wall 360 s  planned rate 0.3 req/s over 360.0 s  seed 31

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 80 | 80 | 0 | 0 (0/0) | 0 | 0 |

offered 0.222 req/s, completed 0.237 req/s, output 9.8 tok/s (over the 338 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 80 | 721.2 | 417.1 | 2470.4 | 3329.3 | 3998.9 |
| ITL (ms) | 3214 | 28.0 | 28.0 | 28.8 | 29.4 | 56.3 |
| per-stream decode (tok/s) | 80 | 35.7 | 35.8 | 36.2 | 36.3 | 36.3 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (3 such prompts): n 75, p50 28.6, p95 29.1, p99 29.3, max 29.3 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 21, p50 300.7, p95 639.8, max 1272.9 ms; all other requests: n 59, p50 595.3, p95 2579.6 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 143 | 19 | 19 | 0 | 0 | 272.7 | 1312.5 | 257.3 (12) | 27.8 | 28.4 |
| 512 | 532 | 27 | 27 | 0 | 0 | 312.7 | 1203.4 | 294.2 (17) | 27.8 | 28.4 |
| 1024 | 1074 | 19 | 19 | 0 | 0 | 416.8 | 2579.6 | 411.0 (14) | 28.0 | 28.7 |
| 2048 | 2119 | 12 | 12 | 0 | 0 | 643.5 | 2396.4 | 634.0 (9) | 28.2 | 28.8 |
| 4096 | 4102 | 3 | 3 | 0 | 0 | 1272.9 | 3726.3 | 1213.8 (2) | 29.1 | 29.6 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 12 | 12 | 0 | 0 | 0 | 8.2 | 454 | 3275 | 28.4 | 29.4 | 29.6 | 3 | 1688 |
| 1 | 60 | 13 | 13 | 0 | 0 | 0 | 7.5 | 427 | 852 | 28.2 | 28.8 | 29.1 | 0 | 1544 |
| 2 | 120 | 11 | 11 | 0 | 0 | 0 | 8.3 | 567 | 1849 | 27.8 | 28.3 | 28.6 | 1 | 1689 |
| 3 | 180 | 14 | 14 | 0 | 0 | 0 | 9.3 | 415 | 1922 | 27.9 | 28.6 | 28.9 | 0 | 1546 |
| 4 | 240 | 12 | 12 | 0 | 0 | 0 | 8.8 | 352 | 1006 | 27.8 | 28.4 | 28.6 | 1 | 1691 |
| 5 | 300 | 18 | 18 | 0 | 0 | 0 | 12.8 | 414 | 1956 | 27.8 | 28.5 | 28.7 | 0 | 1546 |
| 6 | 360 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 72 samples; start 1687 MB, end 1546 MB, max 1691 MB (end - start -140 MB).

Growth fit over t >= 72 s (57 samples, 4.7 min): Theil-Sen slope 0.76 MB/min (OLS 13.11, r2 0.06), fitted growth 4 MB, projected 46 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

72 samples. running: max 1.0 / mean 0.31. queue depth: max 3.0 / mean 0.11. step EWMA ms: -. KV bytes used: max 646594560.0 / mean 172128256.00 of 1152 MB. rejected (delta): 0. server TTFT mean 332.3 ms, ITL mean 28.0 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
