# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T13:33:41-04:00  wall 360 s  planned rate 0.3 req/s over 360.0 s  seed 31

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 80 | 80 | 0 | 0 (0/0) | 0 | 0 |

offered 0.222 req/s, completed 0.237 req/s, output 9.8 tok/s (over the 338 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 80 | 694.1 | 386.5 | 2500.2 | 3257.9 | 3880.2 |
| ITL (ms) | 3214 | 28.0 | 27.9 | 28.6 | 28.9 | 55.4 |
| per-stream decode (tok/s) | 80 | 35.8 | 35.8 | 36.4 | 36.4 | 36.4 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (3 such prompts): n 73, p50 27.9, p95 28.3, p99 28.7, max 28.8 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 21, p50 276.1, p95 584.2, max 1248.1 ms; all other requests: n 59, p50 576.7, p95 2523.3 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 143 | 19 | 19 | 0 | 0 | 214.6 | 1328.6 | 209.0 (12) | 27.7 | 28.2 |
| 512 | 532 | 27 | 27 | 0 | 0 | 294.6 | 1214.4 | 258.3 (17) | 27.8 | 28.4 |
| 1024 | 1074 | 19 | 19 | 0 | 0 | 387.6 | 2523.3 | 381.4 (14) | 28.0 | 28.6 |
| 2048 | 2119 | 12 | 12 | 0 | 0 | 609.2 | 2319.2 | 584.2 (9) | 28.3 | 28.7 |
| 4096 | 4102 | 3 | 3 | 0 | 0 | 1248.1 | 3617.0 | 1187.9 (2) | 28.6 | 29.1 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 12 | 12 | 0 | 0 | 0 | 8.3 | 379 | 3192 | 27.8 | 28.7 | 28.9 | 3 | 1683 |
| 1 | 60 | 13 | 13 | 0 | 0 | 0 | 7.4 | 388 | 815 | 27.8 | 28.5 | 29.1 | 0 | 1539 |
| 2 | 120 | 11 | 11 | 0 | 0 | 0 | 8.3 | 567 | 1886 | 27.9 | 28.5 | 28.7 | 1 | 1684 |
| 3 | 180 | 14 | 14 | 0 | 0 | 0 | 9.3 | 363 | 1879 | 27.8 | 28.5 | 29.1 | 0 | 1540 |
| 4 | 240 | 12 | 12 | 0 | 0 | 0 | 8.8 | 334 | 991 | 27.9 | 28.4 | 28.6 | 1 | 1685 |
| 5 | 300 | 18 | 18 | 0 | 0 | 0 | 12.8 | 388 | 1957 | 28.0 | 28.7 | 29.0 | 0 | 1541 |
| 6 | 360 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 72 samples; start 1680 MB, end 1541 MB, max 1686 MB (end - start -139 MB).

Growth fit over t >= 72 s (57 samples, 4.7 min): Theil-Sen slope 0.62 MB/min (OLS 11.66, r2 0.05), fitted growth 3 MB, projected 37 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

72 samples. running: max 1.0 / mean 0.28. queue depth: max 3.0 / mean 0.10. step EWMA ms: -. KV bytes used: max 646594560.0 / mean 170041344.00 of 1152 MB. rejected (delta): 0. server TTFT mean 312.8 ms, ITL mean 27.9 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
