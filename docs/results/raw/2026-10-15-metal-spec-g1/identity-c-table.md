| case | prompt tok | finish / tokens | off == base | on4 == base | on6 == base | acceptance on4 / on6 | tokens per pass on4 / on6 | verified tokens dropped at the stop, on4 / on6 | min_tokens cuts on4 / on6 | decode tok/s base / off / on4 / on6 |
|---|---|---|---|---|---|---|---|---|---|---|
| golden-docs-48 | 4326 | length / 48 | yes | yes | yes | 7% / 5% | 1.27 / 1.27 | 1 / 1 | 0 / 0 | 34.2 / 34.4 / 38.5 / 35.3 |
| golden-docs-256 | 4326 | length / 256 | yes | yes | yes | 6% / 4% | 1.25 / 1.25 | 0 / 0 | 0 / 0 | 34.0 / 33.7 / 36.1 / 31.8 |
| golden-code-48 | 3596 | length / 48 | yes | yes | yes | 12% / 8% | 1.48 / 1.48 | 0 / 0 | 0 / 0 | 33.4 / 33.2 / 43.4 / 39.7 |
| golden-code-256 | 3596 | length / 256 | yes | yes | yes | 33% / 25% | 2.30 / 2.48 | 1 / 3 | 0 / 0 | 33.6 / 33.4 / 66.1 / 54.5 |
| repeat-eos | 35 | stop / 16 | yes | yes | yes | 100% / 100% | 5.00 / 7.00 | 1 / 0 | 0 / 0 | 34.6 / 34.4 / 131.2 / 162.8 |

on4: 335 verify passes, 1340 proposed, 213 accepted (15.9%)
on6: 327 verify passes, 1962 proposed, 222 accepted (11.3%)
