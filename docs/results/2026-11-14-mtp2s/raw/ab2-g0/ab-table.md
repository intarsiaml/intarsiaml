| visit | streams | decode tok/s per stream, median (min-max) | combined tok/s | ITL p99 ms | chunk gap p99 ms (max) | MTP passes | accepted / drafted | tokens per pass |
|---|---|---|---|---|---|---|---|---|
| O1 | 1 | 88.49 (87.64-88.87) | 88.49 | 92.45 | 78.83 (128.95) | 0 | 0 / 0 | None |
| O1 | 2 | 69.83 (68.7-70.18) | 139.63 | 141.16 | 85.16 (131.51) | 0 | 0 / 0 | None |

| arm | visits | streams | combined tok/s (each visit) | mean | vs A | per stream median (mean) | ITL p99 ms (mean) | acceptance | replies identical to A |
|---|---|---|---|---|---|---|---|---|---|
| O | 1 | 1 | 88.5 | 88.5 | nan | 88.5 | 92.5 | None | - |
| O | 1 | 2 | 139.6 | 139.6 | nan | 69.8 | 141.2 | None | - |
