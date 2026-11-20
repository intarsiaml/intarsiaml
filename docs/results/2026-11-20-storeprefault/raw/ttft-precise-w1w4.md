TTFT s from localeval result.json (`ttft_s.median` of each size row, to the ms, instead of the 10 ms of the console lines), windows w1 + w4; ratio to off in brackets

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 0.5900 (1.0000) | 2.5125 (1.0000) | 5.7133 (1.0000) |
| smacro | 0.7625 (1.2924) | 3.0360 (1.2084) | 6.6540 (1.1647) |
| pf | 0.6700 (1.1356) | 2.6248 (1.0447) | 5.8403 (1.0222) |
| pf4k | 0.6710 (1.1373) | 2.6815 (1.0673) | 5.9530 (1.0420) |

pf / off over every (pf visit, off visit) pairing, min .. max, and the visit values:

- 4096: 1.1191 .. 1.1465; pf 0.667, 0.673, 0.668, 0.672; off 0.588, 0.596, 0.589, 0.587
- 16384: 1.0369 .. 1.0511; pf 2.633, 2.626, 2.616, 2.624; off 2.523, 2.505, 2.509, 2.513
- 32768: 1.0148 .. 1.0283; pf 5.856, 5.835, 5.837, 5.833; off 5.748, 5.700, 5.695, 5.710
