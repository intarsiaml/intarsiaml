Fast-clock threshold: a pass is FAST when the median GPU MHz over its three 4k prefill request windows is >= 1450 (the 1k windows are only 0.25-0.43 s long = 2-3 macmon samples each and include the clock ramp-up from the preceding decode, so they are shown but not used for the flag)

## Per-pass table

| pass | localeval run id | idle util % | zoom | GPU MHz 1k pf | GPU MHz 4k pf | pf W 4k | MHz decode 1k | fast? | dec 128 | dec 1k | dec 4k | marg pf 1k | marg pf 4k | TTFT 1k s | TTFT 4k s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clk-safe-r1 | 20261015-045103-adhoc-clk-safe-r1 | 1 | no | 1554 | 1082 | 25 | 1389 | slow | 36.8 | 36.6 | 35.0 | 4046 | 2196 | 0.268 | 1.680 |
| clk-cand-r1 | 20261015-045257-adhoc-clk-cand-r1 | 1 | no | 1568 | 1533 | 83 | 1384 | FAST | 36.7 | 36.5 | 35.4 | 4345 | 3719 | 0.252 | 1.085 |
| clk-mlx-lm-r1 | 20261015-045418-adhoc-clk-mlx-lm-r1 | 0 | no | 1512 | 1586 | 94 | 1286 | FAST | 34.5 | 33.8 | 32.6 | 4272 | 3406 | 0.426 | 1.336 |
| clk-llamacpp-r1 | 20261015-045548-adhoc-clk-llamacpp-r1 | 1 | no | 1564 | 1618 | 87 | 1159 | FAST | 35.4 | 35.1 | 33.8 | 2950 | 2472 | 0.386 | 1.639 |
| clk-mlx-lm-r2 | 20261015-045714-adhoc-clk-mlx-lm-r2 | 1 | no | 1342 | 1614 | 98 | 1282 | FAST | 34.4 | 33.8 | 32.6 | 4268 | 3459 | 0.426 | 1.322 |
| clk-llamacpp-r2 | 20261015-045843-adhoc-clk-llamacpp-r2 | 1 | no | 1597 | 1617 | 87 | 1162 | FAST | 35.2 | 35.0 | 34.0 | 2963 | 2470 | 0.385 | 1.639 |
| clk-safe-r2 | 20261015-050042-adhoc-clk-safe-r2 | 1 | no | 1580 | 1617 | 91 | 1404 | FAST | 36.7 | 36.6 | 35.6 | 4056 | 3501 | 0.267 | 1.152 |
| clk-cand-r2 | 20261015-050234-adhoc-clk-cand-r2 | 1 | no | 1542 | 1606 | 94 | 1398 | FAST | 36.7 | 36.5 | 35.6 | 4385 | 3143 | 0.252 | 1.245 |
| clk-cand-r3 | 20261015-050427-adhoc-clk-cand-r3 | 1 | no | 1552 | 1595 | 96 | 1398 | FAST | 36.7 | 36.5 | 35.6 | 4373 | 3869 | 0.252 | 1.052 |
| clk-llamacpp-r3 | 20261015-050548-adhoc-clk-llamacpp-r3 | 0 | no | 1617 | 1617 | 87 | 1151 | FAST | 35.4 | 35.0 | 33.7 | 2963 | 2469 | 0.386 | 1.641 |
| clk-safe-r3 | 20261015-050746-adhoc-clk-safe-r3 | 0 | no | 1580 | 1617 | 91 | 1402 | FAST | 36.7 | 36.6 | 35.6 | 4060 | 3494 | 0.267 | 1.153 |
| clk-mlx-lm-r3 | 20261015-050907-adhoc-clk-mlx-lm-r3 | 1 | no | 1404 | 1606 | 98 | 1273 | FAST | 34.5 | 33.9 | 32.7 | 4191 | 3490 | 0.428 | 1.316 |

## All-pass medians (median, (min-max, n)); decode tok/s, prefill tok/s

| engine | passes | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k |
|---|---|---|---|---|---|---|
| SAFE | 3 | 36.7 (36.7-36.8, n=3) | 36.6 (36.6-36.6, n=3) | 35.6 (35.0-35.6, n=3) | 4056 (4046-4060, n=3) | 3494 (2196-3501, n=3) |
| CAND | 3 | 36.7 (36.7-36.7, n=3) | 36.5 (36.5-36.5, n=3) | 35.6 (35.4-35.6, n=3) | 4373 (4345-4385, n=3) | 3719 (3143-3869, n=3) |
| mlx-lm | 3 | 34.5 (34.4-34.5, n=3) | 33.8 (33.8-33.9, n=3) | 32.6 (32.6-32.7, n=3) | 4268 (4191-4272, n=3) | 3459 (3406-3490, n=3) |
| llama.cpp | 3 | 35.4 (35.2-35.4, n=3) | 35.0 (35.0-35.1, n=3) | 33.8 (33.7-34.0, n=3) | 2963 (2950-2963, n=3) | 2470 (2469-2472, n=3) |

## Fast-clock passes only (median GPU >= 1450 MHz over each pass 4k prefill windows)

| engine | passes | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k |
|---|---|---|---|---|---|---|
| SAFE | 2 | 36.7 (36.7-36.7, n=2) | 36.6 (36.6-36.6, n=2) | 35.6 (35.6-35.6, n=2) | 4058 (4056-4060, n=2) | 3498 (3494-3501, n=2) |
| CAND | 3 | 36.7 (36.7-36.7, n=3) | 36.5 (36.5-36.5, n=3) | 35.6 (35.4-35.6, n=3) | 4373 (4345-4385, n=3) | 3719 (3143-3869, n=3) |
| mlx-lm | 3 | 34.5 (34.4-34.5, n=3) | 33.8 (33.8-33.9, n=3) | 32.6 (32.6-32.7, n=3) | 4268 (4191-4272, n=3) | 3459 (3406-3490, n=3) |
| llama.cpp | 3 | 35.4 (35.2-35.4, n=3) | 35.0 (35.0-35.1, n=3) | 33.8 (33.7-34.0, n=3) | 2963 (2950-2963, n=3) | 2470 (2469-2472, n=3) |

## Ratios, all-pass medians

| metric | SAFE/mlx-lm | CAND/mlx-lm | SAFE/llama.cpp | CAND/llama.cpp | SAFE %floor | CAND %floor |
|---|---|---|---|---|---|---|
| decode 128 | 106.6% | 106.5% | 104.0% | 103.9% | 90.5% | 90.4% |
| decode 1024 | 108.1% | 108.0% | 104.4% | 104.3% | 90.1% | 90.0% |
| decode 4096 | 109.0% | 109.0% | 105.1% | 105.1% | 87.6% | 87.6% |
| marginal prefill 1024 | 95.0% | 102.5% | 136.9% | 147.6% | n/a | n/a |
| marginal prefill 4096 | 101.0% | 107.5% | 141.5% | 150.6% | n/a | n/a |

## Ratios, fast-clock passes only

| metric | SAFE/mlx-lm | CAND/mlx-lm | SAFE/llama.cpp | CAND/llama.cpp | SAFE %floor | CAND %floor |
|---|---|---|---|---|---|---|
| decode 128 | 106.6% | 106.5% | 103.9% | 103.9% | 90.5% | 90.4% |
| decode 1024 | 108.1% | 108.0% | 104.4% | 104.3% | 90.1% | 90.0% |
| decode 4096 | 109.0% | 109.0% | 105.2% | 105.1% | 87.7% | 87.6% |
| marginal prefill 1024 | 95.1% | 102.5% | 137.0% | 147.6% | n/a | n/a |
| marginal prefill 4096 | 101.1% | 107.5% | 141.6% | 150.6% | n/a | n/a |

## Verdicts: all-pass medians

| criterion | SAFE | CAND |
|---|---|---|
| prefill 1024 vs mlx-lm | 95.0% of mlx-lm: stretch (>=85%) | 102.5% of mlx-lm: stretch (>=85%) |
| prefill 4096 vs mlx-lm | 101.0% of mlx-lm: stretch (>=85%) | 107.5% of mlx-lm: stretch (>=85%) |
| decode 128 | >=85% floor: PASS (90.5%); >=90% mlx-lm: PASS (106.6%); >= llama.cpp: PASS (36.7 vs 35.4) | >=85% floor: PASS (90.4%); >=90% mlx-lm: PASS (106.5%); >= llama.cpp: PASS (36.7 vs 35.4) |
| decode 1024 | >=85% floor: PASS (90.1%); >=90% mlx-lm: PASS (108.1%); >= llama.cpp: PASS (36.6 vs 35.0) | >=85% floor: PASS (90.0%); >=90% mlx-lm: PASS (108.0%); >= llama.cpp: PASS (36.5 vs 35.0) |
| decode 4096 | >=85% floor: PASS (87.6%); >=90% mlx-lm: PASS (109.0%); >= llama.cpp: PASS (35.6 vs 33.8) | >=85% floor: PASS (87.6%); >=90% mlx-lm: PASS (109.0%); >= llama.cpp: PASS (35.6 vs 33.8) |

## Verdicts: fast-clock passes only

| criterion | SAFE | CAND |
|---|---|---|
| prefill 1024 vs mlx-lm | 95.1% of mlx-lm: stretch (>=85%) | 102.5% of mlx-lm: stretch (>=85%) |
| prefill 4096 vs mlx-lm | 101.1% of mlx-lm: stretch (>=85%) | 107.5% of mlx-lm: stretch (>=85%) |
| decode 128 | >=85% floor: PASS (90.5%); >=90% mlx-lm: PASS (106.6%); >= llama.cpp: PASS (36.7 vs 35.4) | >=85% floor: PASS (90.4%); >=90% mlx-lm: PASS (106.5%); >= llama.cpp: PASS (36.7 vs 35.4) |
| decode 1024 | >=85% floor: PASS (90.1%); >=90% mlx-lm: PASS (108.1%); >= llama.cpp: PASS (36.6 vs 35.0) | >=85% floor: PASS (90.0%); >=90% mlx-lm: PASS (108.0%); >= llama.cpp: PASS (36.5 vs 35.0) |
| decode 4096 | >=85% floor: PASS (87.7%); >=90% mlx-lm: PASS (109.0%); >= llama.cpp: PASS (35.6 vs 33.8) | >=85% floor: PASS (87.6%); >=90% mlx-lm: PASS (109.0%); >= llama.cpp: PASS (35.6 vs 33.8) |

## Per-round marginal prefill 4k (each round's engines ran minutes apart, in rotated order)

| round | order | SAFE | CAND | mlx-lm | llama.cpp |
|---|---|---|---|---|---|
| 1 | SAFE > CAND > mlx-lm > llama.cpp | 2196 @ 1082 MHz | 3719 @ 1533 MHz | 3406 @ 1586 MHz | 2472 @ 1618 MHz |
| 2 | mlx-lm > llama.cpp > SAFE > CAND | 3501 @ 1617 MHz | 3143 @ 1606 MHz | 3459 @ 1614 MHz | 2470 @ 1617 MHz |
| 3 | CAND > llama.cpp > SAFE > mlx-lm | 3494 @ 1617 MHz | 3869 @ 1595 MHz | 3490 @ 1606 MHz | 2469 @ 1617 MHz |

## Clock vs 4k-prefill TTFT per engine (pass level)

| engine | pairs (MHz, TTFT 4k s) |
|---|---|
| SAFE | 1082, 1.68; 1617, 1.15; 1617, 1.15 |
| CAND | 1533, 1.08; 1606, 1.25; 1595, 1.05 |
| mlx-lm | 1586, 1.34; 1614, 1.32; 1606, 1.32 |
| llama.cpp | 1618, 1.64; 1617, 1.64; 1617, 1.64 |

## Per-request (prefill window): pass, size, rep, TTFT, median MHz, median W, samples

| pass | size | rep | TTFT s | MHz | W | n |
|---|---|---|---|---|---|---|
| clk-safe-r1 | 1024 | 1 | 0.27 | 1550 | 65 | 2 |
| clk-safe-r1 | 1024 | 2 | 0.27 | 1564 | 71 | 2 |
| clk-safe-r1 | 1024 | 3 | 0.27 | 1482 | 56 | 3 |
| clk-safe-r1 | 4096 | 1 | 1.24 | 1442 | 62 | 8 |
| clk-safe-r1 | 4096 | 2 | 1.68 | 940 | 24 | 10 |
| clk-safe-r1 | 4096 | 3 | 1.86 | 934 | 22 | 12 |
| clk-cand-r1 | 1024 | 1 | 0.25 | 1530 | 65 | 2 |
| clk-cand-r1 | 1024 | 2 | 0.25 | 1568 | 76 | 2 |
| clk-cand-r1 | 1024 | 3 | 0.25 | 1472 | 50 | 2 |
| clk-cand-r1 | 4096 | 1 | 1.07 | 1554 | 88 | 7 |
| clk-cand-r1 | 4096 | 2 | 1.09 | 1533 | 83 | 7 |
| clk-cand-r1 | 4096 | 3 | 1.09 | 1514 | 79 | 7 |
| clk-mlx-lm-r1 | 1024 | 1 | 0.43 | 1489 | 34 | 3 |
| clk-mlx-lm-r1 | 1024 | 2 | 0.42 | 1576 | 71 | 2 |
| clk-mlx-lm-r1 | 1024 | 3 | 0.43 | 1343 | 54 | 3 |
| clk-mlx-lm-r1 | 4096 | 1 | 1.32 | 1597 | 96 | 8 |
| clk-mlx-lm-r1 | 4096 | 2 | 1.34 | 1581 | 91 | 9 |
| clk-mlx-lm-r1 | 4096 | 3 | 1.34 | 1578 | 92 | 8 |
| clk-llamacpp-r1 | 1024 | 1 | 0.38 | 1564 | 73 | 2 |
| clk-llamacpp-r1 | 1024 | 2 | 0.39 | 1617 | 79 | 3 |
| clk-llamacpp-r1 | 1024 | 3 | 0.39 | 1492 | 65 | 3 |
| clk-llamacpp-r1 | 4096 | 1 | 1.64 | 1618 | 87 | 11 |
| clk-llamacpp-r1 | 4096 | 2 | 1.63 | 1618 | 87 | 10 |
| clk-llamacpp-r1 | 4096 | 3 | 1.64 | 1618 | 88 | 10 |
| clk-mlx-lm-r2 | 1024 | 1 | 0.43 | 1318 | 51 | 3 |
| clk-mlx-lm-r2 | 1024 | 2 | 0.43 | 1346 | 29 | 4 |
| clk-mlx-lm-r2 | 1024 | 3 | 0.43 | 1367 | 55 | 3 |
| clk-mlx-lm-r2 | 4096 | 1 | 1.31 | 1616 | 99 | 8 |
| clk-mlx-lm-r2 | 4096 | 2 | 1.32 | 1608 | 97 | 8 |
| clk-mlx-lm-r2 | 4096 | 3 | 1.32 | 1612 | 98 | 8 |
| clk-llamacpp-r2 | 1024 | 1 | 0.39 | 1597 | 75 | 3 |
| clk-llamacpp-r2 | 1024 | 2 | 0.39 | 1564 | 74 | 3 |
| clk-llamacpp-r2 | 1024 | 3 | 0.39 | 1614 | 77 | 3 |
| clk-llamacpp-r2 | 4096 | 1 | 1.64 | 1617 | 86 | 11 |
| clk-llamacpp-r2 | 4096 | 2 | 1.64 | 1617 | 88 | 10 |
| clk-llamacpp-r2 | 4096 | 3 | 1.64 | 1617 | 87 | 10 |
| clk-safe-r2 | 1024 | 1 | 0.27 | 1552 | 68 | 2 |
| clk-safe-r2 | 1024 | 2 | 0.27 | 1558 | 68 | 2 |
| clk-safe-r2 | 1024 | 3 | 0.27 | 1580 | 74 | 2 |
| clk-safe-r2 | 4096 | 1 | 1.15 | 1617 | 90 | 7 |
| clk-safe-r2 | 4096 | 2 | 1.15 | 1617 | 91 | 7 |
| clk-safe-r2 | 4096 | 3 | 1.15 | 1617 | 91 | 7 |
| clk-cand-r2 | 1024 | 1 | 0.25 | 1476 | 48 | 2 |
| clk-cand-r2 | 1024 | 2 | 0.25 | 1562 | 74 | 2 |
| clk-cand-r2 | 1024 | 3 | 0.25 | 1472 | 47 | 2 |
| clk-cand-r2 | 4096 | 1 | 1.24 | 1611 | 99 | 8 |
| clk-cand-r2 | 4096 | 2 | 1.25 | 1602 | 95 | 8 |
| clk-cand-r2 | 4096 | 3 | 1.25 | 1584 | 92 | 8 |
| clk-cand-r3 | 1024 | 1 | 0.25 | 1507 | 56 | 2 |
| clk-cand-r3 | 1024 | 2 | 0.25 | 1558 | 73 | 2 |
| clk-cand-r3 | 1024 | 3 | 0.25 | 1547 | 73 | 2 |
| clk-cand-r3 | 4096 | 1 | 1.04 | 1607 | 99 | 7 |
| clk-cand-r3 | 4096 | 2 | 1.05 | 1586 | 95 | 6 |
| clk-cand-r3 | 4096 | 3 | 1.06 | 1572 | 92 | 7 |
| clk-llamacpp-r3 | 1024 | 1 | 0.39 | 1617 | 82 | 3 |
| clk-llamacpp-r3 | 1024 | 2 | 0.39 | 1617 | 81 | 3 |
| clk-llamacpp-r3 | 1024 | 3 | 0.39 | 1563 | 71 | 3 |
| clk-llamacpp-r3 | 4096 | 1 | 1.64 | 1617 | 87 | 10 |
| clk-llamacpp-r3 | 4096 | 2 | 1.64 | 1617 | 87 | 10 |
| clk-llamacpp-r3 | 4096 | 3 | 1.64 | 1617 | 88 | 10 |
| clk-safe-r3 | 1024 | 1 | 0.27 | 1558 | 68 | 2 |
| clk-safe-r3 | 1024 | 2 | 0.27 | 1556 | 66 | 2 |
| clk-safe-r3 | 1024 | 3 | 0.27 | 1580 | 75 | 2 |
| clk-safe-r3 | 4096 | 1 | 1.15 | 1617 | 91 | 7 |
| clk-safe-r3 | 4096 | 2 | 1.15 | 1617 | 91 | 7 |
| clk-safe-r3 | 4096 | 3 | 1.15 | 1616 | 91 | 7 |
| clk-mlx-lm-r3 | 1024 | 1 | 0.43 | 1404 | 73 | 3 |
| clk-mlx-lm-r3 | 1024 | 2 | 0.43 | 1295 | 68 | 3 |
| clk-mlx-lm-r3 | 1024 | 3 | 0.42 | 1602 | 75 | 3 |
| clk-mlx-lm-r3 | 4096 | 1 | 1.32 | 1609 | 99 | 8 |
| clk-mlx-lm-r3 | 4096 | 2 | 1.32 | 1608 | 99 | 8 |
| clk-mlx-lm-r3 | 4096 | 3 | 1.31 | 1600 | 95 | 8 |
