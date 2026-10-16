# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T13:47:22-04:00  wall 360 s  planned rate 0.3 req/s over 360.0 s  seed 31

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 80 | 80 | 0 | 0 (0/0) | 0 | 0 |

offered 0.222 req/s, completed 0.237 req/s, output 9.8 tok/s (over the 337 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 80 | 680.4 | 386.2 | 2435.4 | 3238.7 | 3859.0 |
| ITL (ms) | 3214 | 27.5 | 27.5 | 28.2 | 28.5 | 55.8 |
| per-stream decode (tok/s) | 80 | 36.3 | 36.4 | 36.7 | 36.9 | 37.0 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (3 such prompts): n 73, p50 27.8, p95 28.3, p99 28.4, max 28.6 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 21, p50 258.7, p95 587.6, max 1260.0 ms; all other requests: n 59, p50 568.1, p95 2499.5 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 143 | 19 | 19 | 0 | 0 | 219.6 | 1289.2 | 209.3 (12) | 27.3 | 27.8 |
| 512 | 532 | 27 | 27 | 0 | 0 | 276.3 | 1148.4 | 252.1 (17) | 27.4 | 27.9 |
| 1024 | 1074 | 19 | 19 | 0 | 0 | 387.7 | 2499.5 | 377.2 (14) | 27.6 | 28.1 |
| 2048 | 2119 | 12 | 12 | 0 | 0 | 614.5 | 2302.5 | 587.6 (9) | 27.7 | 28.2 |
| 4096 | 4102 | 3 | 3 | 0 | 0 | 1260.0 | 3599.1 | 1186.8 (2) | 28.3 | 28.9 |

## Per 60 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 12 | 12 | 0 | 0 | 0 | 8.3 | 381 | 3183 | 27.7 | 28.4 | 28.6 | 3 | 1692 |
| 1 | 60 | 13 | 13 | 0 | 0 | 0 | 7.4 | 389 | 815 | 27.4 | 27.9 | 28.4 | 0 | 1547 |
| 2 | 120 | 11 | 11 | 0 | 0 | 0 | 8.3 | 536 | 1806 | 27.4 | 27.9 | 28.3 | 1 | 1693 |
| 3 | 180 | 14 | 14 | 0 | 0 | 0 | 9.3 | 387 | 1841 | 27.6 | 28.1 | 28.9 | 0 | 1548 |
| 4 | 240 | 12 | 12 | 0 | 0 | 0 | 8.8 | 325 | 936 | 27.4 | 27.9 | 28.2 | 1 | 1693 |
| 5 | 300 | 18 | 18 | 0 | 0 | 0 | 12.8 | 386 | 1864 | 27.5 | 28.0 | 28.3 | 0 | 1549 |
| 6 | 360 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 72 samples; start 1689 MB, end 1549 MB, max 1694 MB (end - start -140 MB).

Growth fit over t >= 72 s (57 samples, 4.7 min): Theil-Sen slope 0.41 MB/min (OLS 13.10, r2 0.06), fitted growth 2 MB, projected 25 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

72 samples. running: max 1.0 / mean 0.28. queue depth: max 3.0 / mean 0.10. step EWMA ms: -. KV bytes used: max 646594560.0 / mean 170041344.00 of 1152 MB. rejected (delta): 0. server TTFT mean 310.6 ms, ITL mean 27.5 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
