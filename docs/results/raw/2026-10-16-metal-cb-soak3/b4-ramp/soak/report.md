# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T12:44:48-04:00  wall 103 s  planned rate 0.2 req/s over 100.0 s  seed 41

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 22 | 22 | 0 | 0 (0/0) | 0 | 0 |

offered 0.220 req/s, completed 0.231 req/s, output 37.9 tok/s (over the 95 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 0.00%, goodput 100.00%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 22 | 595.5 | 306.3 | 1714.1 | 2666.8 | 2912.3 |
| ITL (ms) | 3594 | 32.2 | 28.5 | 32.6 | 113.1 | 125.2 |
| per-stream decode (tok/s) | 22 | 31.1 | 33.1 | 35.9 | 36.1 | 36.2 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (2 such prompts): n 17, p50 99.2, p95 104.5, p99 104.9, max 105.0 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 5, p50 303.5, p95 971.8, max 1137.5 ms; all other requests: n 17, p50 409.4, p95 1977.3 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 154 | 4 | 4 | 0 | 0 | 99.1 | 116.6 | 79.6 (1) | 28.1 | 31.7 |
| 512 | 505 | 6 | 6 | 0 | 0 | 218.0 | 258.4 | 154.8 (2) | 28.1 | 31.9 |
| 1024 | 1071 | 6 | 6 | 0 | 0 | 444.3 | 998.8 | 306.3 (2) | 30.1 | 99.1 |
| 2048 | 2102 | 4 | 4 | 0 | 0 | 856.8 | 2613.6 | - (0) | 29.6 | 32.3 |
| 4096 | 4056 | 2 | 2 | 0 | 0 | 1440.5 | 1713.2 | 1137.5 (1) | 28.6 | 30.9 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 4 | 4 | 0 | 0 | 0 | 21.5 | 551 | 1601 | 30.1 | 95.6 | 101.2 | 0 | 5003 |
| 1 | 20 | 1 | 1 | 0 | 0 | 0 | 16.9 | 1137 | 1137 | 28.2 | 29.1 | 29.6 | 0 | 5133 |
| 2 | 40 | 3 | 3 | 0 | 0 | 0 | 30.3 | 92 | 705 | 27.9 | 28.9 | 94.7 | 0 | 5149 |
| 3 | 60 | 6 | 6 | 0 | 0 | 0 | 32.7 | 287 | 462 | 27.9 | 29.3 | 98.2 | 0 | 5150 |
| 4 | 80 | 8 | 8 | 0 | 0 | 0 | 73.5 | 376 | 2297 | 30.1 | 103.9 | 119.2 | 0 | 5151 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 6.0 | - | - | 28.6 | 30.9 | 31.4 | 0 | 5151 |

## Memory

source `footprint_mb`, 34 samples; start 3995 MB, end 5151 MB, max 5151 MB (end - start 1156 MB).

Growth fit over t >= 20 s (27 samples, 1.3 min): Theil-Sen slope 2.79 MB/min (OLS 70.43, r2 0.24), fitted growth 4 MB, projected 168 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 20499 MB, end 20499 MB, max 20499 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

35 samples. running: max 4.0 / mean 1.14. queue depth: max 0.0 / mean 0.00. step EWMA ms: max 86.7 / mean 31.90. KV bytes used: max 1298497536.0 / mean 951984362.06 of 4608 MB. rejected (delta): 0. server TTFT mean 449.8 ms, ITL mean 32.0 ms. prefill chunks (delta) 90.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 1221 | 56.6% |
| 2 | 549 | 25.4% |
| 3 | 307 | 14.2% |
| 4 | 82 | 3.8% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |

mean rows per decode pass 1.65

## SLO

no thresholds given (informational run).
