# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T12:52:37-04:00  wall 148 s  planned rate 0.8 req/s over 100.0 s  seed 45

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 94 | 83 | 11 | 0 (0/0) | 0 | 0 |

offered 0.940 req/s, completed 0.565 req/s, output 80.9 tok/s (over the 147 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 11.70%, goodput 88.30%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 85 | 20414.6 | 17336.7 | 46218.9 | 48981.4 | 50006.0 |
| ITL (ms) | 11792 | 41.9 | 32.0 | 123.9 | 139.0 | 153.2 |
| per-stream decode (tok/s) | 83 | 25.0 | 25.4 | 31.0 | 35.8 | 36.5 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (7 such prompts): n 5963, p50 31.9, p95 129.6, p99 139.3, max 150.2 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 1, p50 294.7, p95 294.7, max 294.7 ms; all other requests: n 84, p50 17475.1, p95 46293.1 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 134 | 11 | 8 | 0 | 0 | 13159.1 | 29967.6 | 61.5 (1) | 31.8 | 127.1 |
| 512 | 519 | 26 | 25 | 0 | 0 | 16885.2 | 45909.8 | - (0) | 31.6 | 120.9 |
| 1024 | 1041 | 27 | 24 | 0 | 0 | 21467.6 | 46903.1 | 294.7 (1) | 31.9 | 125.0 |
| 2048 | 2101 | 22 | 19 | 0 | 0 | 19219.1 | 45279.8 | - (0) | 32.2 | 122.0 |
| 4096 | 4390 | 8 | 7 | 0 | 0 | 26467.2 | 37347.0 | - (0) | 32.9 | 128.7 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 17 | 14 | 0 | 0 | 3 | 69.5 | 2272 | 4108 | 31.3 | 108.7 | 125.2 | 4 | 5157 |
| 1 | 20 | 20 | 18 | 0 | 0 | 2 | 93.0 | 11879 | 19315 | 31.5 | 118.3 | 139.4 | 9 | 5158 |
| 2 | 40 | 16 | 15 | 0 | 0 | 1 | 78.4 | 14322 | 17790 | 32.8 | 128.8 | 141.7 | 11 | 5158 |
| 3 | 60 | 23 | 21 | 0 | 0 | 2 | 86.2 | 26467 | 37143 | 31.8 | 123.5 | 138.9 | 20 | 5159 |
| 4 | 80 | 18 | 15 | 0 | 0 | 3 | 83.5 | 43486 | 49152 | 31.7 | 127.5 | 137.3 | 26 | 5160 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 65.5 | - | - | 33.2 | 136.2 | 146.5 | 23 | 5160 |
| 6 | 120 | 0 | 0 | 0 | 0 | 0 | 90.5 | - | - | 32.2 | 115.6 | 132.4 | 13 | 5160 |
| 7 | 140 | 0 | 0 | 0 | 0 | 0 | 27.4 | - | - | 29.8 | 32.3 | 118.4 | 0 | 5152 |

## Memory

source `footprint_mb`, 49 samples; start 5156 MB, end 5152 MB, max 5160 MB (end - start -4 MB).

Growth fit over t >= 20 s (42 samples, 2.1 min): Theil-Sen slope 1.71 MB/min (OLS 1.16, r2 0.23), fitted growth 4 MB, projected 103 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 20499 MB, end 20499 MB, max 20499 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

50 samples. running: max 4.0 / mean 3.68. queue depth: max 26.0 / mean 11.32. step EWMA ms: max 121.1 / mean 48.11. KV bytes used: max 1905721344.0 / mean 863248711.68 of 4608 MB. rejected (delta): 0. server TTFT mean 759.2 ms, ITL mean 42.3 ms. prefill chunks (delta) 475.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 173 | 5.3% |
| 2 | 120 | 3.7% |
| 3 | 481 | 14.8% |
| 4 | 2477 | 76.2% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |

mean rows per decode pass 3.62

## SLO

no thresholds given (informational run).
