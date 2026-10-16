| pass | build | 1k TTFT s | 4096-row prompt / TTFT s / marginal vs 1k tok/s | 4108-row prompt / TTFT s / marginal vs 1k tok/s | localeval's own marginal (4096 vs 1k) | run id |
|---|---|---|---|---|---|---|
| p1-cand-c2048 | cand | 0.25 | 4087 / 1.07 / 3795 | 4112 / 1.30 / 2979 | 3795.3 | 20261015-075606-adhoc-p3noise-p1-cand-c2048 |
| p2-fix-c2048 | fix | 0.25 | 4088 / 1.09 / 3690 | 4113 / 1.14 / 3511 | 3690.1 | 20261015-075830-adhoc-p3noise-p2-fix-c2048 |
| p3-fix-c2048 | fix | 0.25 | 4088 / 1.11 / 3625 | 4112 / 1.13 / 3545 | 3624.6 | 20261015-080053-adhoc-p3noise-p3-fix-c2048 |
| p4-cand-c2048 | cand | 0.25 | 4088 / 1.09 / 3713 | 4110 / 1.28 / 3049 | 3713.4 | 20261015-080311-adhoc-p3noise-p4-cand-c2048 |
| p5-cand-c2048 | cand | 0.25 | 4088 / 1.10 / 3655 | 4110 / 1.26 / 3084 | 3655.3 | 20261015-080529-adhoc-p3noise-p5-cand-c2048 |
| p6-fix-c2048 | fix | 0.25 | 4087 / 1.08 / 3753 | 4113 / 1.13 / 3564 | 3752.7 | 20261015-080756-adhoc-p3noise-p6-fix-c2048 |
| p7-fix-c2048 | fix | 0.25 | 4088 / 1.06 / 3828 | 4115 / 1.13 / 3554 | 3828.2 | 20261015-081020-adhoc-p3noise-p7-fix-c2048 |
| p8-cand-c2048 | cand | 0.25 | 4090 / 1.07 / 3804 | 4112 / 1.28 / 3049 | 3803.7 | 20261015-081237-adhoc-p3noise-p8-cand-c2048 |

- cand: 4096-row marginal-vs-1k per pass [3795, 3713, 3655, 3804] (median 3754); 4108-row [2979, 3049, 3084, 3049] (median 3049)
- fix: 4096-row marginal-vs-1k per pass [3690, 3625, 3753, 3828] (median 3721); 4108-row [3511, 3545, 3564, 3554] (median 3549)
