# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T13:54:12-04:00  wall 360 s  planned rate 0.3 req/s over 360.0 s  seed 31

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 80 | 80 | 0 | 0 (0/0) | 0 | 0 |

offered 0.222 req/s, completed 0.237 req/s, output 9.8 tok/s (over the 337 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 80 | 690.1 | 384.6 | 2446.9 | 3291.7 | 3917.9 |
| ITL (ms) | 3214 | 27.7 | 27.5 | 28.7 | 29.6 | 54.4 |
| per-stream decode (tok/s) | 80 | 36.2 | 36.2 | 36.8 | 36.9 | 36.9 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (3 such prompts): n 74, p50 27.7, p95 28.9, p99 29.4, max 29.7 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 21, p50 265.7, p95 581.8, max 1270.2 ms; all other requests: n 59, p50 570.5, p95 2555.8 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 143 | 19 | 19 | 0 | 0 | 216.0 | 1280.2 | 204.8 (12) | 27.4 | 28.8 |
| 512 | 532 | 27 | 27 | 0 | 0 | 287.6 | 1175.9 | 262.0 (17) | 27.4 | 28.4 |
| 1024 | 1074 | 19 | 19 | 0 | 0 | 386.7 | 2555.8 | 372.4 (14) | 27.6 | 28.8 |
| 2048 | 2119 | 12 | 12 | 0 | 0 | 611.8 | 2361.3 | 594.0 (9) | 27.7 | 28.3 |
| 4096 | 4102 | 3 | 3 | 0 | 0 | 1270.2 | 3653.1 | 1192.2 (2) | 28.4 | 29.8 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 12 | 12 | 0 | 0 | 0 | 8.3 | 394 | 3226 | 28.0 | 29.2 | 30.1 | 3 | 1690 |
| 1 | 60 | 13 | 13 | 0 | 0 | 0 | 7.4 | 387 | 810 | 27.4 | 28.7 | 29.9 | 0 | 1546 |
| 2 | 120 | 11 | 11 | 0 | 0 | 0 | 8.3 | 563 | 1825 | 27.5 | 28.4 | 29.4 | 1 | 1691 |
| 3 | 180 | 14 | 14 | 0 | 0 | 0 | 9.3 | 374 | 1907 | 27.6 | 28.9 | 29.6 | 0 | 1547 |
| 4 | 240 | 12 | 12 | 0 | 0 | 0 | 8.8 | 322 | 957 | 27.5 | 28.2 | 28.7 | 1 | 1692 |
| 5 | 300 | 18 | 18 | 0 | 0 | 0 | 12.8 | 366 | 1936 | 27.5 | 28.2 | 28.7 | 0 | 1547 |
| 6 | 360 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 72 samples; start 1688 MB, end 1547 MB, max 1692 MB (end - start -141 MB).

Growth fit over t >= 72 s (57 samples, 4.7 min): Theil-Sen slope 0.39 MB/min (OLS 17.00, r2 0.11), fitted growth 2 MB, projected 24 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

72 samples. running: max 1.0 / mean 0.32. queue depth: max 3.0 / mean 0.11. step EWMA ms: -. KV bytes used: max 646594560.0 / mean 175020032.00 of 1152 MB. rejected (delta): 0. server TTFT mean 315.1 ms, ITL mean 27.7 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
