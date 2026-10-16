# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-15T19:51:33-04:00  wall 180 s  planned rate 0.5 req/s over 180.0 s  seed 1

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 105 | 100 | 5 | 0 (0/0) | 0 | 0 |

offered 0.583 req/s, completed 0.557 req/s, output 25.6 tok/s (over the 180 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 4.76%, goodput 95.24%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 101 | 3240.9 | 3013.4 | 8444.6 | 9678.0 | 9903.6 |
| ITL (ms) | 4499 | 28.0 | 27.8 | 29.3 | 30.6 | 67.0 |
| per-stream decode (tok/s) | 100 | 35.6 | 35.9 | 36.5 | 36.7 | 36.8 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (7 such prompts): n 338, p50 27.9, p95 29.3, p99 30.3, max 43.2 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 136 | 11 | 11 | 0 | 0 | 1268.8 | 4283.1 | 239.6 (3) | 27.6 | 29.2 |
| 512 | 522 | 34 | 33 | 0 | 0 | 4030.1 | 8636.8 | 297.0 (2) | 27.8 | 29.3 |
| 1024 | 1086 | 34 | 31 | 0 | 0 | 2344.7 | 8485.2 | 333.5 (3) | 27.8 | 29.3 |
| 2048 | 2125 | 19 | 18 | 0 | 0 | 2729.9 | 6729.8 | 591.9 (3) | 28.0 | 29.2 |
| 4096 | 4230 | 7 | 7 | 0 | 0 | 3969.6 | 5817.6 | 1875.9 (1) | 28.7 | 29.9 |

## Per 30 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 20 | 19 | 0 | 0 | 1 | 28.3 | 3000 | 5671 | 27.7 | 28.6 | 29.0 | 2 | 1691 |
| 1 | 30 | 18 | 16 | 0 | 0 | 2 | 28.8 | 5476 | 9178 | 27.7 | 28.4 | 28.9 | 4 | 1692 |
| 2 | 60 | 18 | 17 | 0 | 0 | 1 | 21.3 | 846 | 3342 | 27.7 | 28.5 | 28.9 | 3 | 1693 |
| 3 | 90 | 15 | 15 | 0 | 0 | 0 | 22.1 | 1391 | 3170 | 27.9 | 29.3 | 29.8 | 2 | 1693 |
| 4 | 120 | 18 | 17 | 0 | 0 | 1 | 26.1 | 4406 | 9200 | 28.0 | 29.7 | 30.5 | 5 | 1694 |
| 5 | 150 | 16 | 16 | 0 | 0 | 0 | 26.6 | 4178 | 5837 | 28.2 | 30.5 | 37.2 | 4 | 1694 |
| 6 | 180 | 0 | 0 | 0 | 0 | 0 | 0.1 | - | - | 28.5 | 29.1 | 29.2 | - | - |

## Memory

source `footprint_mb`, 60 samples; start 1691 MB, end 1694 MB, max 1694 MB (end - start 4 MB).

Growth fit over t >= 36 s (48 samples, 2.4 min): Theil-Sen slope 1.14 MB/min (OLS 4.08, r2 0.02), fitted growth 3 MB, projected 68 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 17043 MB, end 17043 MB, max 17043 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

60 samples. running: max 1.0 / mean 0.90. queue depth: max 5.0 / mean 1.70. step EWMA ms: -. KV bytes used: max 666648576.0 / mean 176890675.20 of 1152 MB. rejected (delta): 0. server TTFT mean 370.0 ms, ITL mean 28.0 ms. prefill chunks (delta) -.

## SLO

no thresholds given (informational run).
