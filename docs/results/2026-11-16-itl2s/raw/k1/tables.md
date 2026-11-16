## client: gaps between arrivals (ms), pooled over the cell's requests of all visits of the arm

| cell | arm | visits | combined tok/s (median pair; per visit) | tok/s per stream | gap p50 | p90 | p99 | p99.9 | max | localeval ITL p99 (median of requests) | per-visit gap p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cjk-nt T=0 s2 | X1 | 1 | 147.6 (148) | 74.1 | 24.5 | 25.5 | 26.4 | 90.6 | 96.2 | 26 | 26 |
| code T=0 s2 | X1 | 1 | 154.4 (154) | 77.2 | 24.3 | 25.3 | 49.6 | 95.0 | 105.2 | 98 | 50 |
| en T=0 s2 | X1 | 1 | 151.3 (151) | 75.5 | 24.6 | 25.5 | 50.2 | 89.2 | 95.0 | 101 | 50 |
| code T=1 s2 | X1 | 1 | 147.2 (147) | 73.8 | 24.4 | 25.5 | 49.8 | 94.1 | 106.3 | 97 | 50 |

## server (Emit lines): per visit and cell

| visit | cell | iterations | emission gap ms p50 / p99 / max | per-token spread p99 | 2-seq verify pass ms p50 / p99 | draft (prepare) ms p50 / p99 | decode step ms p50 / p99 | tokens per emission | dup iters |
|---|---|---|---|---|---|---|---|---|---|
| X1.a | en T=0 s2 | 1695 | 24.5 / 26.0 / 94.9 | 25.6 | 21.9 / 23.2 (n=1619) | 2.6 / 15.8 | 11.7 / 11.9 | 1.85 | 0 |
| X1.a | code T=0 s2 | 1652 | 24.3 / 25.6 / 104.9 | 25.1 | 21.6 / 22.8 (n=1614) | 2.6 / 15.5 | 11.7 / 12.3 | 1.87 | 0 |
| X1.a | cjk-nt T=0 s2 | 1783 | 24.5 / 26.0 / 96.4 | 25.7 | 21.8 / 23.3 (n=1623) | 2.6 / 15.7 | 11.7 / 12.0 | 1.80 | 0 |
| X1.a | code T=1 s2 | 1728 | 24.4 / 26.0 / 106.3 | 25.3 | 21.7 / 22.9 (n=1660) | 2.6 / 17.9 | 11.8 / 12.7 | 1.80 | 0 |

## server: what the gaps above 60 ms (emission gaps, both streams running) were made of, per visit

| visit | gaps | gaps > thr | class of the iteration that ended the gap (V / V+D / D) | prefill chunk inside | first 16 tokens of the reply | alone on the server | gap p99 over gaps with 2 streams running |
|---|---|---|---|---|---|---|---|
| X1.a | 13333 | 32 | 32 / 0 / 0 | 32 | 32 | 0 | 25.9 |

## server: 2-seq verify pass wall time by row count (ms), per visit

| visit | rows | passes | p50 | p99 | max |
|---|---|---|---|---|---|
| X1.a | 1 seqs / 2 rows | 27 | 15.6 | 16.6 | 16.6 |
| X1.a | 2 seqs / 4 rows | 6907 | 21.8 | 23.1 | 27.1 |

## server: tokens per emission (share of emissions with 1 / 2 / 3 tokens) and the longest run of 1-token emissions of a request, per visit

| visit | cell | emissions | 1 token | 2 tokens | 3 tokens | longest run of 1-token emissions (p50 / max over requests) | share of tokens delivered in 1-token emissions |
|---|---|---|---|---|---|---|---|
