# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T12:46:31-04:00  wall 100 s  planned rate 0.3 req/s over 100.0 s  seed 42

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 29 | 28 | 1 | 0 (0/0) | 0 | 0 |

offered 0.290 req/s, completed 0.303 req/s, output 38.7 tok/s (over the 92 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 3.45%, goodput 96.55%; chunks per completion token 0.998 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 29 | 485.4 | 291.3 | 1696.6 | 2356.8 | 2440.5 |
| ITL (ms) | 3537 | 32.4 | 28.9 | 33.2 | 136.0 | 149.5 |
| per-stream decode (tok/s) | 28 | 31.8 | 32.8 | 35.8 | 36.1 | 36.2 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (2 such prompts): n 60, p50 137.9, p95 149.4, p99 149.5, max 149.5 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 3, p50 339.6, p95 960.3, max 1029.3 ms; all other requests: n 26, p50 290.2, p95 1860.8 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 134 | 6 | 6 | 0 | 0 | 105.9 | 1716.0 | 106.0 (1) | 28.8 | 30.6 |
| 512 | 515 | 12 | 11 | 0 | 0 | 198.8 | 668.8 | 154.0 (6) | 29.0 | 34.7 |
| 1024 | 1130 | 8 | 8 | 0 | 0 | 393.4 | 489.2 | 323.8 (3) | 28.1 | 29.6 |
| 2048 | 2378 | 1 | 1 | 0 | 0 | 1011.7 | 1011.7 | - (0) | 30.5 | 130.9 |
| 4096 | 4226 | 2 | 2 | 0 | 0 | 1734.9 | 2369.9 | 1029.3 (1) | 29.0 | 32.3 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 4 | 3 | 0 | 0 | 1 | 18.6 | 159 | 268 | 28.3 | 29.5 | 38.2 | 0 | 5006 |
| 1 | 20 | 5 | 5 | 0 | 0 | 0 | 27.5 | 340 | 489 | 27.9 | 29.2 | 94.8 | 0 | 5151 |
| 2 | 40 | 7 | 7 | 0 | 0 | 0 | 53.1 | 383 | 1017 | 29.9 | 33.7 | 114.6 | 0 | 5143 |
| 3 | 60 | 8 | 8 | 0 | 0 | 0 | 36.9 | 359 | 2336 | 29.9 | 129.4 | 148.8 | 1 | 5152 |
| 4 | 80 | 5 | 5 | 0 | 0 | 0 | 42.2 | 324 | 394 | 28.9 | 30.2 | 94.3 | 0 | 5007 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 33 samples; start 5150 MB, end 5007 MB, max 5152 MB (end - start -143 MB).

Growth fit over t >= 20 s (26 samples, 1.3 min): Theil-Sen slope 1.08 MB/min (OLS 14.50, r2 0.01), fitted growth 1 MB, projected 65 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 20499 MB, end 20499 MB, max 20499 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

34 samples. running: max 4.0 / mean 1.21. queue depth: max 1.0 / mean 0.03. step EWMA ms: max 118.1 / mean 32.90. KV bytes used: max 1509064704.0 / mean 758079969.88 of 4608 MB. rejected (delta): 0. server TTFT mean 398.9 ms, ITL mean 32.5 ms. prefill chunks (delta) 84.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 1042 | 52.4% |
| 2 | 451 | 22.7% |
| 3 | 374 | 18.8% |
| 4 | 120 | 6.0% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |

mean rows per decode pass 1.78

## SLO

no thresholds given (informational run).
