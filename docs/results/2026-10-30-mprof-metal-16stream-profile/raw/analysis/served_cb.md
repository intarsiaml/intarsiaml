| rows | steps | cmd buffers / step | dispatches / step | GPU busy ms | GPU idle inside step ms | GPU idle before next step ms | step period ms (first cb start -> next first cb start) | host encode ms (sum ho->hc) | commit ms (sum hc->hr) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 496 | 7.0 | 764 | 7.90 | 0.024 | 0.190 | 8.11 | 0.74 | 0.01 |
| 8 | 488 | 8.0 | 804 | 21.63 | 0.204 | 0.295 | 22.13 | 1.13 | 0.02 |
| 16 | 480 | 10.0 | 1125 | 40.14 | 0.312 | 0.390 | 40.84 | 1.70 | 0.03 |
