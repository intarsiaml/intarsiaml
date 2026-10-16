# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T12:48:17-04:00  wall 103 s  planned rate 0.45 req/s over 100.0 s  seed 43

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 52 | 48 | 4 | 0 (0/0) | 0 | 0 |

offered 0.520 req/s, completed 0.468 req/s, output 53.8 tok/s (over the 102 s from first send to last completion); shed rate 0.00%, error rate 0.00%, cancel rate 7.69%, goodput 92.31%; chunks per completion token 0.999 (1.0 means ITL is per token).

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 51 | 746.1 | 355.1 | 2477.4 | 2630.4 | 2687.9 |
| ITL (ms) | 5458 | 35.5 | 30.4 | 98.2 | 134.4 | 218.6 |
| per-stream decode (tok/s) | 48 | 29.4 | 30.5 | 36.2 | 36.4 | 36.6 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (4 such prompts): n 202, p50 117.5, p95 140.9, p99 148.3, max 218.6 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 1, p50 140.6, p95 140.6, max 140.6 ms; all other requests: n 50, p50 372.3, p95 2481.5 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 155 | 10 | 9 | 0 | 0 | 103.0 | 2378.5 | 81.4 (1) | 29.1 | 32.3 |
| 512 | 510 | 20 | 19 | 0 | 0 | 238.1 | 994.0 | 139.3 (4) | 29.8 | 93.4 |
| 1024 | 1046 | 10 | 10 | 0 | 0 | 445.4 | 2521.1 | 367.0 (2) | 31.4 | 117.8 |
| 2048 | 2057 | 8 | 6 | 0 | 0 | 770.6 | 1098.4 | 630.6 (4) | 29.9 | 94.9 |
| 4096 | 4193 | 4 | 4 | 0 | 0 | 2456.7 | 2564.7 | - (0) | 32.3 | 116.2 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 10 | 9 | 0 | 0 | 1 | 46.8 | 229 | 638 | 28.7 | 31.4 | 97.0 | 0 | 5152 |
| 1 | 20 | 11 | 11 | 0 | 0 | 0 | 58.8 | 410 | 1968 | 30.9 | 111.4 | 135.3 | 0 | 5153 |
| 2 | 40 | 10 | 10 | 0 | 0 | 0 | 48.1 | 150 | 681 | 29.3 | 32.2 | 107.9 | 0 | 5153 |
| 3 | 60 | 14 | 11 | 0 | 0 | 3 | 58.2 | 964 | 2586 | 31.1 | 116.2 | 136.5 | 2 | 5154 |
| 4 | 80 | 7 | 7 | 0 | 0 | 0 | 53.2 | 1156 | 2493 | 32.0 | 108.3 | 135.2 | 1 | 5154 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 10.3 | - | - | 29.9 | 30.6 | 30.7 | 0 | 5154 |

## Memory

source `footprint_mb`, 34 samples; start 5152 MB, end 5154 MB, max 5154 MB (end - start 2 MB).

Growth fit over t >= 20 s (27 samples, 1.3 min): Theil-Sen slope 1.14 MB/min (OLS 1.17, r2 0.77), fitted growth 2 MB, projected 69 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 20499 MB, end 20499 MB, max 20499 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

35 samples. running: max 4.0 / mean 2.06. queue depth: max 2.0 / mean 0.14. step EWMA ms: max 109.2 / mean 37.86. KV bytes used: max 1260601344.0 / mean 764133844.11 of 4608 MB. rejected (delta): 0. server TTFT mean 489.3 ms, ITL mean 35.7 ms. prefill chunks (delta) 187.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 907 | 35.1% |
| 2 | 891 | 34.5% |
| 3 | 421 | 16.3% |
| 4 | 363 | 14.1% |
| 5 | 0 | 0.0% |
| 6 | 0 | 0.0% |
| 7 | 0 | 0.0% |
| 8 | 0 | 0.0% |

mean rows per decode pass 2.09

## SLO

no thresholds given (informational run).
