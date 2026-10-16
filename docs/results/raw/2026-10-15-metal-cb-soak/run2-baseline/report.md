# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-15T19:55:01-04:00  wall 204 s  planned rate 0.2 req/s over 200.0 s  seed 2

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 47 | 47 | 0 | 0 (0/0) | 0 | 0 |

offered 0.235 req/s, completed 0.234 req/s, output 7.2 tok/s (over the 201 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 47 | 2048.9 | 1688.7 | 4969.0 | 6289.5 | 6342.8 |
| ITL (ms) | 1409 | 57.5 | 50.2 | 91.8 | 103.3 | 215.9 |
| per-stream decode (tok/s) | 47 | 18.4 | 16.8 | 31.3 | 36.0 | 36.1 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (13 such prompts): n 236, p50 50.4, p95 90.3, p99 101.5, max 106.1 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 138 | 14 | 14 | 0 | 0 | 288.4 | 3641.4 | 261.5 (9) | 52.9 | 97.7 |
| 512 | 513 | 5 | 5 | 0 | 0 | 4186.0 | 5919.8 | 317.2 (2) | 48.0 | 85.9 |
| 1024 | 1064 | 8 | 8 | 0 | 0 | 623.1 | 2862.1 | 589.7 (6) | 51.3 | 88.3 |
| 2048 | 2033 | 7 | 7 | 0 | 0 | 1175.9 | 4064.7 | 1036.9 (4) | 48.6 | 91.8 |
| 4096 | 4256 | 13 | 13 | 0 | 0 | 2795.8 | 5550.4 | 1909.5 (6) | 49.8 | 87.3 |

## Per 30 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 7 | 7 | 0 | 0 | 0 | 4.8 | 1721 | 3047 | 42.8 | 76.9 | 156.3 | 1 | 1550 |
| 1 | 30 | 5 | 5 | 0 | 0 | 0 | 2.3 | 1033 | 3008 | 66.8 | 89.6 | 93.9 | 0 | 1695 |
| 2 | 60 | 9 | 9 | 0 | 0 | 0 | 12.2 | 4186 | 6296 | 49.6 | 90.9 | 102.4 | 3 | 1695 |
| 3 | 90 | 8 | 8 | 0 | 0 | 0 | 7.8 | 590 | 1861 | 49.6 | 88.1 | 94.9 | 1 | 1695 |
| 4 | 120 | 6 | 6 | 0 | 0 | 0 | 7.5 | 3167 | 4978 | 49.9 | 92.1 | 106.5 | 1 | 1695 |
| 5 | 150 | 6 | 6 | 0 | 0 | 0 | 7.5 | 1800 | 4539 | 50.1 | 93.7 | 105.2 | 1 | 1695 |
| 6 | 180 | 6 | 6 | 0 | 0 | 0 | 6.4 | 1532 | 3077 | 56.9 | 97.5 | 100.8 | 1 | 1695 |

## Memory

source `footprint_mb`, 68 samples; start 1694 MB, end 1695 MB, max 1695 MB (end - start 1 MB).

Growth fit over t >= 40 s (54 samples, 2.7 min): Theil-Sen slope 0.35 MB/min (OLS 1.84, r2 0.00), fitted growth 1 MB, projected 21 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

68 samples. running: max 1.0 / mean 0.62. queue depth: max 3.0 / mean 0.29. step EWMA ms: -. KV bytes used: max 765591552.0 / mean 260964592.94 of 1152 MB. rejected (delta): 0. server TTFT mean 881.0 ms, ITL mean 57.5 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
