# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T07:27:48-04:00  wall 131 s  planned rate 1.4 req/s over 100.0 s  seed 14

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 128 | 112 | 16 | 0 (0/0) | 0 | 0 |

offered 1.280 req/s, completed 0.861 req/s, output 116.3 tok/s (over the 130 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 12.50%, goodput 87.50%; chunks per completion token 0.998 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 112 | 14128.9 | 13983.3 | 23747.7 | 24007.5 | 24197.7 |
| ITL (ms) | 14998 | 59.3 | 42.6 | 123.5 | 133.4 | 256.2 |
| per-stream decode (tok/s) | 112 | 17.7 | 17.0 | 24.3 | 28.9 | 29.5 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (8 such prompts): n 6733, p50 43.0, p95 126.5, p99 137.0, max 256.2 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 1, p50 138.1, p95 138.1, max 138.1 ms; all other requests: n 111, p50 13998.2, p95 23749.9 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 145 | 23 | 21 | 0 | 0 | 13998.2 | 23392.0 | - (0) | 42.3 | 122.4 |
| 512 | 509 | 32 | 29 | 0 | 0 | 13113.5 | 23413.3 | 138.1 (1) | 42.4 | 124.1 |
| 1024 | 1033 | 39 | 30 | 0 | 0 | 14006.2 | 23229.0 | - (0) | 42.7 | 123.0 |
| 2048 | 2099 | 24 | 24 | 0 | 0 | 15592.9 | 23836.7 | - (0) | 42.7 | 124.1 |
| 4096 | 4059 | 10 | 8 | 0 | 0 | 11803.9 | 23286.9 | - (0) | 44.0 | 124.4 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 24 | 22 | 0 | 0 | 2 | 90.9 | 1868 | 7509 | 39.8 | 124.2 | 133.0 | 7 | 9769 |
| 1 | 20 | 29 | 23 | 0 | 0 | 6 | 110.0 | 10672 | 12947 | 44.4 | 124.7 | 138.1 | 15 | 9769 |
| 2 | 40 | 28 | 25 | 0 | 0 | 3 | 121.2 | 16044 | 23621 | 42.1 | 123.5 | 129.5 | 20 | 9770 |
| 3 | 60 | 24 | 21 | 0 | 0 | 3 | 121.0 | 16604 | 22547 | 42.9 | 123.8 | 136.9 | 17 | 9771 |
| 4 | 80 | 23 | 21 | 0 | 0 | 2 | 121.1 | 23136 | 23840 | 43.0 | 125.7 | 133.3 | 23 | 9771 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 131.3 | - | - | 42.9 | 118.5 | 136.6 | 21 | 9772 |
| 6 | 120 | 0 | 0 | 0 | 0 | 0 | 60.0 | - | - | 32.1 | 113.4 | 124.9 | 4 | 9763 |

## Memory

source `footprint_mb`, 43 samples; start 9767 MB, end 9763 MB, max 9772 MB (end - start -4 MB).

Growth fit over t >= 20 s (36 samples, 1.8 min): Theil-Sen slope 1.61 MB/min (OLS 0.79, r2 0.07), fitted growth 3 MB, projected 97 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 25107 MB, end 25107 MB, max 25107 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

44 samples. running: max 8.0 / mean 7.25. queue depth: max 23.0 / mean 11.82. step EWMA ms: max 114.8 / mean 66.28. KV bytes used: max 3347103744.0 / mean 1551049448.73 of 9216 MB. rejected (delta): 0. server TTFT mean 703.9 ms, ITL mean 59.9 ms. prefill chunks (delta) 557.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 4 | 0.2% |
| 2 | 61 | 2.8% |
| 3 | 129 | 6.0% |
| 4 | 72 | 3.3% |
| 5 | 62 | 2.9% |
| 6 | 181 | 8.4% |
| 7 | 443 | 20.5% |
| 8 | 1208 | 55.9% |

mean rows per decode pass 6.93

## SLO

no thresholds given (informational run).
