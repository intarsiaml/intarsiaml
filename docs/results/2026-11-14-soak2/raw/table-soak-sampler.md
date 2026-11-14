| replay | start min | requests | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s median (bimodal: see doc) | decode tok/s mean | agg output tok/s | cached frac | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S61 | 0.0 | 48 | 0 | 67.9 | 1.38 | 1.54 | 40.6 | 41.2 | 59.1 | 0.732 | 127 |
| S62 | 2.1 | 48 | 0 | 66.5 | 1.38 | 1.52 | 40.4 | 41.0 | 59.5 | 0.732 | 126 |
| S63 | 4.3 | 48 | 0 | 71.1 | 1.38 | 2.11 | 41.2 | 42.8 | 59.2 | 0.732 | 124 |
| S64 | 6.4 | 48 | 0 | 67.5 | 1.37 | 1.55 | 40.5 | 41.6 | 59.3 | 0.732 | 125 |
| S65 | 8.5 | 48 | 0 | 71.1 | 1.43 | 1.96 | 42.5 | 42.7 | 59.7 | 0.732 | 124 |
| S66 | 10.6 | 48 | 0 | 73.6 | 1.44 | 2.37 | 42.2 | 43.5 | 59.9 | 0.732 | 125 |
| S67 | 12.7 | 48 | 0 | 67.9 | 1.39 | 1.54 | 41.9 | 41.6 | 60.1 | 0.732 | 126 |
| S68 | 14.9 | 48 | 0 | 70.0 | 1.35 | 2.21 | 42.4 | 42.9 | 59.2 | 0.732 | 124 |
| S69 | 16.9 | 48 | 0 | 74.9 | 1.41 | 2.49 | 42.9 | 44.3 | 60.5 | 0.732 | 124 |
| S70 | 19.0 | 48 | 0 | 66.8 | 1.35 | 1.54 | 40.8 | 42.2 | 59.5 | 0.732 | 124 |
| S71 | 21.2 | 48 | 0 | 69.0 | 1.38 | 1.67 | 41.2 | 42.4 | 60.3 | 0.732 | 125 |
| S72 | 23.3 | 48 | 0 | 69.2 | 1.38 | 2.21 | 42.4 | 42.7 | 59.8 | 0.732 | 124 |
| S73 | 25.4 | 48 | 0 | 74.3 | 1.42 | 2.38 | 43.1 | 44.2 | 60.1 | 0.732 | 124 |
| S74 | 27.5 | 48 | 0 | 73.6 | 1.43 | 2.20 | 43.0 | 44.0 | 60.7 | 0.732 | 125 |
| S75 | 29.6 | 48 | 0 | 65.9 | 1.37 | 1.50 | 41.1 | 41.2 | 59.7 | 0.732 | 125 |
| S76 | 31.7 | 48 | 0 | 74.9 | 1.41 | 2.33 | 42.5 | 44.4 | 59.6 | 0.732 | 123 |
| S77 | 33.8 | 48 | 0 | 75.4 | 1.36 | 2.71 | 42.1 | 43.5 | 58.3 | 0.732 | 126 |
| S78 | 35.9 | 48 | 0 | 78.8 | 1.54 | 2.59 | 42.5 | 44.4 | 58.2 | 0.732 | 127 |
- ttft_sum: first half mean 70.053, second half mean 71.989, min 65.894, max 78.801
- ttft_p50: first half mean 1.392, second half mean 1.404, min 1.346, max 1.544
- dec_p50: first half mean 41.621, second half mean 42.089, min 40.372, max 43.133
- dec_mean: first half mean 42.405, second half mean 43.220, min 40.966, max 44.364
- cached: first half mean 0.732, second half mean 0.732, min 0.732, max 0.732
- wall: first half mean 125.078, second half mean 124.657, min 122.860, max 126.900

sampler: 40 samples, 00:37:32 to 01:16:38
| signal | first 5 mean | last 5 mean | min | max | slope/min, all | slope/min, from sample 6 (store full) |
|---|---:|---:|---:|---:|---:|---:|
| mem_avail_mb | 47724.8 | 44489.4 | 42050.0 | 53727.0 | -22.03 | 62.87 |
| mem_used_mb | 76884.6 | 80119.6 | 70882.0 | 82559.0 | 22.03 | -62.87 |
| swap_used_mb | 6325.0 | 6325.0 | 6325.0 | 6325.0 | 0.00 | 0.00 |
| rss_mb | 162.0 | 162.0 | 162.0 | 162.0 | 0.00 | 0.00 |
| intarsia_kv_bytes_used | 887.4 | 928.9 | 233.8 | 1428.2 | 1.17 | 2.51 |
| intarsia_device_memory_free_bytes | 50043.5 | 46651.1 | 44092.9 | 56337.9 | -23.10 | 65.92 |
| intarsia_prefix_store_tier_bytes{tier="host"} | 6415.2 | 8537.0 | 669.5 | 8587.2 | 36.77 | -0.20 |
| intarsia_prefix_store_entries | 90.4 | 102.8 | 15.0 | 140.0 | 0.27 | 0.01 |
| intarsia_prefix_store_tier_objects{tier="host"} | 820.0 | 1162.4 | 79.0 | 1334.0 | 5.80 | 0.04 |

- store host bytes peak 8587 MB against budget 8590 MB (within budget)
- lookups 909, hits 852, misses 57, host evictions 3550, rejected 0, inserts 3648
- requests 909, rejected 0, engine_failed 0, no_progress_s 0, finished length/stop 410/497
- first host eviction seen in sample 2 (of 40; 60 s apart)
- intarsia_batch_paused_steps_total (prefill pause default ran when > 0): 1698
- server gaps (batch_itl) total 136835: above 0.25 s 1, above 0.5 s 0
