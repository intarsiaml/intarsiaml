| B | cmd buffers / step | dispatches / step | step wall (first cb host-open -> last cb GPU end) ms | GPU busy ms | GPU idle inside step ms | host encode (sum ho->hc) ms | commit calls (sum hc->hr) ms | idle between steps (last cb end -> next first cb start) ms | first-cb GPU start minus its host-open ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.0 | 764 | 8.72 | 8.23 | 0.204 | 1.39 | 0.03 | 0.586 | 0.288 |
| 4 | 8.0 | 804 | 15.01 | 14.12 | 0.285 | 1.93 | 0.05 | 1.182 | 0.602 |
| 8 | 8.0 | 804 | 21.05 | 20.24 | 0.310 | 2.19 | 0.05 | 0.788 | 0.502 |
| 9 | 10.0 | 1125 | 27.11 | 26.52 | 0.277 | 3.01 | 0.06 | 0.608 | 0.314 |
| 12 | 10.0 | 1125 | 32.37 | 31.58 | 0.394 | 3.20 | 0.07 | 0.834 | 0.401 |
| 16 | 10.0 | 1125 | 38.98 | 37.89 | 0.638 | 3.50 | 0.06 | 1.752 | 0.458 |
