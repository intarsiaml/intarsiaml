Fast-clock threshold: median GPU MHz over the three 4k prefill request windows >= 1450

| pass | localeval run | idle util % | MHz 4k prefill | MHz decode 128/1k/4k | fast? | decode 128 | decode 1k | decode 4k (median of 3) | marginal prefill 1k / 4k tok/s |
|---|---|---|---|---|---|---|---|---|---|
| p1-off | 20261015-122505-adhoc-p1-off | 4 | 1617 | 1285/1362/1398 | FAST | 36.0 (36.0/36.0/35.5) | 36.1 (36.1/35.9/36.3) | 35.3 (35.0/35.3/35.3) | 4385 / 3875 |
| p2-on | 20261015-122729-adhoc-p2-on | 1 | 1453 | 1316/1409/1386 | FAST | 37.7 (36.4/38.1/37.7) | 38.5 (38.3/38.5/38.9) | 39.3 (38.6/39.3/39.5) | 4318 / 3445 |
| p3-on | 20261015-122933-adhoc-p3-on | 1 | 1494 | 1372/1408/1391 | FAST | 37.6 (37.8/36.7/37.6) | 38.6 (38.2/38.6/40.1) | 39.2 (40.3/39.2/39.1) | 4346 / 3596 |
| p4-off | 20261015-123127-adhoc-p4-off | 2 | 1576 | 1328/1372/1360 | FAST | 36.4 (35.9/36.4/36.4) | 36.0 (36.4/36.0/35.7) | 34.8 (34.8/35.1/34.7) | 4368 / 3839 |

### All passes (ABBA, no clock filter)

| size | spec off tok/s | spec on tok/s | on / off | spec off vs 36.7/36.6/35.6 | passes (off, on) used |
|---|---|---|---|---|---|
| 128 | 36.0 (n=6, 35.5-36.4) | 37.7 (n=6, 36.4-38.1) | **1.05x** | -1.9% | p1-off,p2-on,p3-on,p4-off |
| 1024 | 36.0 (n=6, 35.7-36.4) | 38.5 (n=6, 38.2-40.1) | **1.07x** | -1.5% | p1-off,p2-on,p3-on,p4-off |
| 4096 | 35.0 (n=6, 34.7-35.3) | 39.2 (n=6, 38.6-40.3) | **1.12x** | -1.5% | p1-off,p2-on,p3-on,p4-off |

### Passes with 4k-prefill clock >= 1450 MHz only

| size | spec off tok/s | spec on tok/s | on / off | spec off vs 36.7/36.6/35.6 | passes (off, on) used |
|---|---|---|---|---|---|
| 128 | 36.0 (n=6, 35.5-36.4) | 37.7 (n=6, 36.4-38.1) | **1.05x** | -1.9% | p1-off,p2-on,p3-on,p4-off |
| 1024 | 36.0 (n=6, 35.7-36.4) | 38.5 (n=6, 38.2-40.1) | **1.07x** | -1.5% | p1-off,p2-on,p3-on,p4-off |
| 4096 | 35.0 (n=6, 34.7-35.3) | 39.2 (n=6, 38.6-40.3) | **1.12x** | -1.5% | p1-off,p2-on,p3-on,p4-off |

### spec counters from /metrics (spec-on passes, per prompt size: 3 reps x 256 tokens)

| pass | size | verify passes | proposed | accepted | acceptance | tokens per pass (1 + accepted/passes) |
|---|---|---|---|---|---|---|
| p2-on | 128 | 83 | 332 | 32 | 9.6% | 1.39 |
| p2-on | 1024 | 171 | 684 | 64 | 9.4% | 1.37 |
| p2-on | 4096 | 211 | 844 | 110 | 13.0% | 1.52 |
| p3-on | 128 | 80 | 320 | 23 | 7.2% | 1.29 |
| p3-on | 1024 | 171 | 684 | 67 | 9.8% | 1.39 |
| p3-on | 4096 | 218 | 872 | 115 | 13.2% | 1.53 |
