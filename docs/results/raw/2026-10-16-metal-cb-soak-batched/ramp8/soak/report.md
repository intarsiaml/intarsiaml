# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T07:21:46-04:00  wall 107 s  planned rate 0.4 req/s over 100.0 s  seed 11

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 42 | 39 | 3 | 0 (0/0) | 0 | 0 |

offered 0.420 req/s, completed 0.376 req/s, output 59.1 tok/s (over the 104 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 7.14%, goodput 92.86%; chunks per completion token 1.000 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 42 | 685.6 | 535.2 | 1762.9 | 2481.7 | 2535.5 |
| ITL (ms) | 6087 | 40.8 | 31.4 | 110.2 | 133.1 | 493.0 |
| per-stream decode (tok/s) | 39 | 25.7 | 26.6 | 33.5 | 35.1 | 35.6 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (4 such prompts): n 208, p50 117.7, p95 153.8, p99 188.7, max 189.2 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 2, p50 266.4, p95 298.3, max 301.8 ms; all other requests: n 40, p50 554.5, p95 1795.0 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 142 | 6 | 5 | 0 | 0 | 139.8 | 521.3 | 231.1 (1) | 32.2 | 119.3 |
| 512 | 544 | 6 | 5 | 0 | 0 | 305.3 | 716.4 | 301.8 (1) | 30.3 | 110.5 |
| 1024 | 1051 | 20 | 20 | 0 | 0 | 519.1 | 796.2 | 317.0 (2) | 31.9 | 108.0 |
| 2048 | 1903 | 6 | 5 | 0 | 0 | 865.1 | 1114.5 | - (0) | 31.1 | 102.5 |
| 4096 | 4132 | 4 | 4 | 0 | 0 | 2083.6 | 2515.8 | - (0) | 31.7 | 86.6 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 6 | 6 | 0 | 0 | 0 | 33.6 | 430 | 706 | 28.9 | 31.9 | 101.9 | 1 | 7310 |
| 1 | 20 | 10 | 9 | 0 | 0 | 1 | 46.4 | 409 | 778 | 29.0 | 94.7 | 99.6 | 0 | 7456 |
| 2 | 40 | 10 | 10 | 0 | 0 | 0 | 77.8 | 515 | 1763 | 31.2 | 105.3 | 120.5 | 0 | 8609 |
| 3 | 60 | 10 | 10 | 0 | 0 | 0 | 68.7 | 624 | 1946 | 35.0 | 112.5 | 150.5 | 0 | 9762 |
| 4 | 80 | 6 | 4 | 0 | 0 | 2 | 63.8 | 956 | 2190 | 37.1 | 127.3 | 156.8 | 0 | 9762 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 16.2 | - | - | 30.1 | 31.7 | 32.1 | 0 | 9754 |

## Memory

source `footprint_mb`, 35 samples; start 3998 MB, end 9754 MB, max 9762 MB (end - start 5756 MB).

Growth fit over t >= 20 s (28 samples, 1.4 min): Theil-Sen slope 2013.35 MB/min (OLS 2145.75, r2 0.89), fitted growth 2821 MB, projected 120801 MB/h. Leak suspect: **YES** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 25107 MB, end 25107 MB, max 25107 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

36 samples. running: max 7.0 / mean 2.39. queue depth: max 1.0 / mean 0.03. step EWMA ms: max 98.6 / mean 42.26. KV bytes used: max 2520465408.0 / mean 1413681152.00 of 9216 MB. rejected (delta): 0. server TTFT mean 659.1 ms, ITL mean 41.2 ms. prefill chunks (delta) 219.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 535 | 22.7% |
| 2 | 830 | 35.2% |
| 3 | 457 | 19.4% |
| 4 | 335 | 14.2% |
| 5 | 98 | 4.2% |
| 6 | 51 | 2.2% |
| 7 | 49 | 2.1% |
| 8 | 0 | 0.0% |

mean rows per decode pass 2.57

## SLO

no thresholds given (informational run).
