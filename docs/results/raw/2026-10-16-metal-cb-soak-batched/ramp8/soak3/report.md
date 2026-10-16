# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T07:25:35-04:00  wall 109 s  planned rate 1.0 req/s over 100.0 s  seed 13

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 74 | 68 | 6 | 0 (0/0) | 0 | 0 |

offered 0.740 req/s, completed 0.633 req/s, output 91.0 tok/s (over the 107 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 8.11%, goodput 91.89%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 73 | 3826.9 | 2823.9 | 9715.9 | 10923.3 | 12100.4 |
| ITL (ms) | 9696 | 59.1 | 44.2 | 130.4 | 149.8 | 246.7 |
| per-stream decode (tok/s) | 68 | 18.2 | 17.4 | 32.0 | 33.2 | 34.7 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (7 such prompts): n 2746, p50 47.5, p95 142.3, p99 161.1, max 246.7 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 1, p50 219.6, p95 219.6, max 219.6 ms; all other requests: n 72, p50 2851.7, p95 9733.7 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 132 | 10 | 10 | 0 | 0 | 2215.8 | 8858.8 | 219.6 (1) | 43.6 | 123.2 |
| 512 | 535 | 17 | 16 | 0 | 0 | 1960.3 | 10003.6 | - (0) | 44.8 | 129.8 |
| 1024 | 1063 | 25 | 21 | 0 | 0 | 1033.0 | 9327.4 | - (0) | 42.7 | 132.5 |
| 2048 | 2163 | 14 | 14 | 0 | 0 | 3389.1 | 10064.8 | - (0) | 44.9 | 129.6 |
| 4096 | 4236 | 8 | 7 | 0 | 0 | 5362.7 | 8948.6 | - (0) | 45.6 | 133.7 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 14 | 13 | 0 | 0 | 1 | 59.7 | 387 | 931 | 30.0 | 92.5 | 107.1 | 0 | 9766 |
| 1 | 20 | 15 | 13 | 0 | 0 | 2 | 76.2 | 738 | 1852 | 37.5 | 128.4 | 143.1 | 0 | 9767 |
| 2 | 40 | 21 | 18 | 0 | 0 | 3 | 93.6 | 5768 | 8814 | 48.2 | 144.3 | 164.5 | 8 | 9767 |
| 3 | 60 | 11 | 11 | 0 | 0 | 0 | 122.9 | 8896 | 11283 | 45.6 | 123.7 | 154.4 | 6 | 9768 |
| 4 | 80 | 13 | 13 | 0 | 0 | 0 | 98.2 | 4619 | 10084 | 45.6 | 123.7 | 141.4 | 5 | 9768 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 37.9 | - | - | 33.3 | 135.0 | 145.3 | 0 | 9759 |

## Memory

source `footprint_mb`, 36 samples; start 9765 MB, end 9759 MB, max 9768 MB (end - start -6 MB).

Growth fit over t >= 20 s (29 samples, 1.5 min): Theil-Sen slope 1.27 MB/min (OLS 0.18, r2 0.00), fitted growth 2 MB, projected 76 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 25107 MB, end 25107 MB, max 25107 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

37 samples. running: max 8.0 / mean 5.76. queue depth: max 8.0 / mean 1.95. step EWMA ms: max 126.1 / mean 61.31. KV bytes used: max 3036266496.0 / mean 1864003251.89 of 9216 MB. rejected (delta): 0. server TTFT mean 1041.4 ms, ITL mean 60.0 ms. prefill chunks (delta) 415.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 288 | 14.7% |
| 2 | 208 | 10.6% |
| 3 | 160 | 8.2% |
| 4 | 240 | 12.3% |
| 5 | 140 | 7.2% |
| 6 | 161 | 8.2% |
| 7 | 200 | 10.2% |
| 8 | 558 | 28.5% |

mean rows per decode pass 4.95

## SLO

no thresholds given (informational run).
