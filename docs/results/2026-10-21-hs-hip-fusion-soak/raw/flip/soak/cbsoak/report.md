# cbsoak report

http://127.0.0.1:8090  model `qwen36-35b-hs`  started 2026-10-21T19:41:20-04:00  wall 840 s  planned rate 0.3 req/s over 840.0 s  seed 7

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 279 | 253 | 26 | 0 (0/0) | 0 | 0 |

offered 0.332 req/s, completed 0.303 req/s, output 43.4 tok/s (over the 836 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 9.32%, goodput 90.68%; chunks per completion token 0.997 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 266 | 1263.3 | 421.0 | 5155.0 | 8300.6 | 8972.4 |
| ITL (ms) | 35882 | 5.5 | 5.5 | 5.6 | 5.7 | 39.4 |
| per-stream decode (tok/s) | 253 | 182.2 | 182.5 | 185.8 | 186.3 | 186.6 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (10 such prompts): n 1298, p50 5.4, p95 5.5, p99 5.5, max 16.4 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 67, p50 128.1, p95 2798.1, max 4368.5 ms; all other requests: n 199, p50 682.0, p95 6274.9 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 141 | 61 | 57 | 0 | 0 | 166.0 | 5236.0 | 61.5 (39) | 5.4 | 5.5 |
| 512 | 535 | 86 | 84 | 0 | 0 | 352.8 | 5396.6 | 96.4 (53) | 5.5 | 5.5 |
| 1024 | 1050 | 72 | 62 | 0 | 0 | 570.3 | 5543.6 | 133.4 (40) | 5.5 | 5.6 |
| 2048 | 2169 | 48 | 41 | 0 | 0 | 366.4 | 3629.8 | 219.9 (31) | 5.5 | 5.6 |
| 4096 | 4149 | 12 | 9 | 0 | 0 | 990.4 | 5824.9 | 665.8 (6) | 5.6 | 5.7 |

## Per 30 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 9 | 9 | 0 | 0 | 0 | 40.5 | 133 | 3407 | 5.5 | 5.6 | 5.6 | 2 | - |
| 1 | 30 | 7 | 7 | 0 | 0 | 0 | 39.4 | 222 | 820 | 5.5 | 5.6 | 5.7 | 0 | - |
| 2 | 60 | 8 | 7 | 0 | 0 | 1 | 38.9 | 115 | 1073 | 5.5 | 5.6 | 5.6 | 2 | - |
| 3 | 90 | 12 | 10 | 0 | 0 | 2 | 43.5 | 285 | 917 | 5.4 | 5.5 | 5.6 | 1 | - |
| 4 | 120 | 9 | 9 | 0 | 0 | 0 | 47.4 | 270 | 2391 | 5.5 | 5.6 | 5.7 | 3 | - |
| 5 | 150 | 6 | 4 | 0 | 0 | 2 | 18.1 | 498 | 3065 | 5.5 | 5.6 | 5.6 | 7 | - |
| 6 | 180 | 12 | 10 | 0 | 0 | 2 | 43.2 | 1585 | 3990 | 5.4 | 5.6 | 5.6 | 9 | - |
| 7 | 210 | 12 | 11 | 0 | 0 | 1 | 49.8 | 161 | 1563 | 5.5 | 5.6 | 5.7 | 2 | - |
| 8 | 240 | 8 | 7 | 0 | 0 | 1 | 46.0 | 1406 | 2154 | 5.4 | 5.5 | 5.6 | 2 | - |
| 9 | 270 | 8 | 8 | 0 | 0 | 0 | 49.4 | 129 | 1245 | 5.5 | 5.6 | 5.6 | 2 | - |
| 10 | 300 | 13 | 13 | 0 | 0 | 0 | 44.6 | 435 | 6928 | 5.5 | 5.6 | 5.6 | 1 | - |
| 11 | 330 | 14 | 12 | 0 | 0 | 2 | 54.3 | 3932 | 8199 | 5.4 | 5.5 | 5.5 | 12 | - |
| 12 | 360 | 7 | 6 | 0 | 0 | 1 | 41.8 | 1098 | 2331 | 5.4 | 5.6 | 5.7 | 6 | - |
| 13 | 390 | 7 | 7 | 0 | 0 | 0 | 39.4 | 570 | 1065 | 5.5 | 5.6 | 5.6 | 1 | - |
| 14 | 420 | 9 | 9 | 0 | 0 | 0 | 33.8 | 124 | 1995 | 5.5 | 5.6 | 5.7 | 3 | - |
| 15 | 450 | 12 | 12 | 0 | 0 | 0 | 49.1 | 340 | 573 | 5.5 | 5.7 | 5.7 | 0 | - |
| 16 | 480 | 10 | 10 | 0 | 0 | 0 | 52.9 | 788 | 2622 | 5.4 | 5.6 | 5.6 | 3 | - |
| 17 | 510 | 11 | 11 | 0 | 0 | 0 | 43.5 | 682 | 3397 | 5.5 | 5.5 | 5.6 | 4 | - |
| 18 | 540 | 11 | 9 | 0 | 0 | 2 | 40.8 | 3321 | 5744 | 5.4 | 5.5 | 5.6 | 9 | - |
| 19 | 570 | 12 | 11 | 0 | 0 | 1 | 44.9 | 192 | 914 | 5.5 | 5.6 | 5.7 | 2 | - |
| 20 | 600 | 13 | 12 | 0 | 0 | 1 | 71.5 | 395 | 1578 | 5.4 | 5.5 | 5.6 | 1 | - |
| 21 | 630 | 11 | 10 | 0 | 0 | 1 | 39.3 | 1849 | 3712 | 5.5 | 5.6 | 5.6 | 5 | - |
| 22 | 660 | 7 | 7 | 0 | 0 | 0 | 29.9 | 129 | 330 | 5.5 | 5.6 | 5.7 | 0 | - |
| 23 | 690 | 16 | 11 | 0 | 0 | 5 | 36.3 | 5745 | 8644 | 5.5 | 5.6 | 5.6 | 12 | - |
| 24 | 720 | 9 | 7 | 0 | 0 | 2 | 39.3 | 1614 | 2914 | 5.4 | 5.6 | 5.6 | 8 | - |
| 25 | 750 | 11 | 10 | 0 | 0 | 1 | 59.5 | 204 | 1330 | 5.5 | 5.7 | 5.7 | 2 | - |
| 26 | 780 | 10 | 9 | 0 | 0 | 1 | 41.1 | 127 | 329 | 5.5 | 5.7 | 5.7 | 1 | - |
| 27 | 810 | 5 | 5 | 0 | 0 | 0 | 26.5 | 133 | 1163 | 5.5 | 5.6 | 5.6 | 0 | - |
| 28 | 840 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

no process memory samples (run without --pid, or the process could not be sampled).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 11554 MB, end 11554 MB, max 11586 MB; slope 0.00 MB/min (r2 0.02), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

168 samples. running: max 1.0 / mean 0.56. queue depth: max 12.0 / mean 1.32. step EWMA ms: -. KV bytes used: max 162283520.0 / mean 96103619.05 of 223 MB. rejected (delta): 0. server TTFT mean 152.8 ms, ITL mean 6.3 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
