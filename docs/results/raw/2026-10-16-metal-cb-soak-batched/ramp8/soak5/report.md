# cbsoak report

http://127.0.0.1:8107  model `Qwen3-8B`  started 2026-10-16T07:30:31-04:00  wall 160 s  planned rate 1.8 req/s over 100.0 s  seed 15

## Totals

| sent | ok | cancelled | shed (429/503) | error | client_dropped |
|---|---|---|---|---|---|
| 189 | 131 | 19 | 17 (17/0) | 22 | 0 |

offered 1.890 req/s, completed 0.827 req/s, output 110.5 tok/s (over the 158 s from first send to last completion); shed rate 8.99%, error rate 11.64%, cancel rate 10.05%, goodput 69.31%; chunks per completion token 1.000 (1.0 means ITL is per token).

Error kinds: `still running after --drain-s 60.0` x22

## Latency

| metric | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| TTFT (ms) | 133 | 34586.6 | 37017.8 | 67464.4 | 70748.5 | 72478.0 |
| ITL (ms) | 17363 | 62.5 | 43.5 | 126.8 | 138.4 | 255.1 |
| per-stream decode (tok/s) | 131 | 16.4 | 16.3 | 21.9 | 23.8 | 24.0 |

ITL of running streams during the prefill of an arriving >= 3000-token prompt (16 such prompts): n 16561, p50 43.5, p95 127.0, p99 138.7, max 255.1 ms (gaps overlapping [send, send + TTFT] of another request; with max_batch 1 the window includes queue wait and the running stream is not stalled, so this is ordinary ITL there).

TTFT of requests that arrived after > 3 s with no request open (server idle: weights are re-wired, ~170 ms extra): n 1, p50 65.6, p95 65.6, max 65.6 ms; all other requests: n 132, p50 37411.8, p95 67495.8 ms.

## By prompt length

solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).

| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 141 | 38 | 28 | 4 | 4 | 30899.7 | 61571.2 | 65.6 (1) | 43.2 | 125.8 |
| 512 | 523 | 55 | 37 | 5 | 6 | 42415.8 | 62533.9 | - (0) | 42.9 | 126.1 |
| 1024 | 1063 | 44 | 28 | 6 | 7 | 38824.6 | 66128.7 | - (0) | 43.8 | 128.3 |
| 2048 | 2122 | 31 | 22 | 0 | 4 | 38216.1 | 67685.0 | - (0) | 43.6 | 126.9 |
| 4096 | 4125 | 21 | 16 | 2 | 1 | 33605.5 | 67881.7 | - (0) | 44.1 | 128.5 |

## Per 20 s window (requests by send time, ITL and tok/s by chunk arrival time)

| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 41 | 34 | 0 | 0 | 7 | 80.2 | 7538 | 17996 | 45.1 | 132.8 | 145.6 | 12 | 9772 |
| 1 | 20 | 36 | 34 | 0 | 0 | 2 | 111.0 | 26978 | 37807 | 44.2 | 126.9 | 135.3 | 35 | 9774 |
| 2 | 40 | 33 | 28 | 0 | 0 | 5 | 142.2 | 43238 | 47467 | 42.0 | 118.5 | 134.8 | 40 | 9775 |
| 3 | 60 | 35 | 30 | 0 | 3 | 2 | 111.2 | 59451 | 67559 | 43.9 | 125.7 | 154.9 | 56 | 9776 |
| 4 | 80 | 44 | 5 | 17 | 19 | 3 | 119.3 | 69897 | 72212 | 43.2 | 125.1 | 135.2 | 64 | 9777 |
| 5 | 100 | 0 | 0 | 0 | 0 | 0 | 109.4 | - | - | 43.0 | 126.5 | 135.7 | 62 | 9777 |
| 6 | 120 | 0 | 0 | 0 | 0 | 0 | 138.2 | - | - | 42.5 | 125.9 | 134.5 | 47 | 9777 |
| 7 | 140 | 0 | 0 | 0 | 0 | 0 | 63.4 | - | - | 44.2 | 134.0 | 143.4 | 30 | 9777 |
| 8 | 160 | 0 | 0 | 0 | 0 | 0 | 0.0 | - | - | - | - | - | - | - |

## Memory

source `footprint_mb`, 53 samples; start 9770 MB, end 9777 MB, max 9777 MB (end - start 7 MB).

Growth fit over t >= 20 s (46 samples, 2.3 min): Theil-Sen slope 2.26 MB/min (OLS 2.23, r2 0.89), fitted growth 5 MB, projected 136 MB/h. Leak suspect: **no** (growth >= 100 MB and r2 >= 0.6).

Device memory from /metrics (total - free, system-wide; the process footprint above does not include GPU buffers): start 25107 MB, end 25107 MB, max 25107 MB; slope 0.00 MB/min (r2 1.00), fitted growth 0 MB. Leak suspect: **no**.

## Server (/metrics)

54 samples. running: max 8.0 / mean 7.72. queue depth: max 64.0 / mean 37.11. step EWMA ms: max 117.0 / mean 69.65. KV bytes used: max 2619703296.0 / mean 1631302997.33 of 9216 MB. rejected (delta): 17. server TTFT mean 777.8 ms, ITL mean 62.5 ms. prefill chunks (delta) 742.

Batch-size histogram (decode passes by rows, delta over the run):

| rows | passes | share |
|---|---|---|
| 1 | 17 | 0.7% |
| 2 | 2 | 0.1% |
| 3 | 14 | 0.6% |
| 4 | 7 | 0.3% |
| 5 | 9 | 0.4% |
| 6 | 116 | 4.8% |
| 7 | 578 | 24.0% |
| 8 | 1663 | 69.1% |

mean rows per decode pass 7.56

## SLO

no thresholds given (informational run).
