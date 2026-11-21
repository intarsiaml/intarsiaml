=== s2-T0.7-k20-warm

s2-T0.7-k20-warm: combined tok/s over the overlap window (reps)
| window | 4096 | 32768 | 65536 |
|---|---|---|---|
| none a1 | 151.4 (8) | 122.5 (8) | 105.5 (8) |
| none a4 | 151.8 (8) | 122.7 (8) | 105.1 (8) |
| fast a2 | 152.4 (8) | 126.7 (8) | 102.8 (8) |
| fast a3 | 152.5 (8) | 128.8 (8) | 101.6 (8) |
| **none pooled** | 151.6 (16) | 122.6 (16) | 105.3 (16) |
| **fast pooled** | 152.5 (16) | 127.7 (16) | 102.2 (16) |
| fast vs none | **+0.6 %** (2 SE 1.7 %) | **+4.2 %** (2 SE 4.5 %) | **-2.9 %** (2 SE 4.1 %) |

s2-T0.7-k20-warm: combined tok/s over the whole request (reps)
| window | 4096 | 32768 | 65536 |
|---|---|---|---|
| none a1 | 149.7 (8) | 120.5 (8) | 102.1 (8) |
| none a4 | 149.0 (8) | 120.8 (8) | 102.6 (8) |
| fast a2 | 150.9 (8) | 124.5 (8) | 100.9 (8) |
| fast a3 | 150.2 (8) | 126.2 (8) | 99.5 (8) |
| **none pooled** | 149.4 (16) | 120.6 (16) | 102.3 (16) |
| **fast pooled** | 150.6 (16) | 125.4 (16) | 100.2 (16) |
| fast vs none | **+0.8 %** (2 SE 2.0 %) | **+4.0 %** (2 SE 4.5 %) | **-2.1 %** (2 SE 3.9 %) |

s2-T0.7-k20-warm: per-stream tok/s in the overlap (reps)
| window | 4096 | 32768 | 65536 |
|---|---|---|---|
| none a1 | 75.7 (16) | 61.2 (16) | 52.8 (16) |
| none a4 | 75.9 (16) | 61.4 (16) | 52.5 (16) |
| fast a2 | 76.2 (16) | 63.3 (16) | 51.4 (16) |
| fast a3 | 76.2 (16) | 64.4 (16) | 50.8 (16) |
| **none pooled** | 75.8 (32) | 61.3 (32) | 52.7 (32) |
| **fast pooled** | 76.2 (32) | 63.9 (32) | 51.1 (32) |
| fast vs none | **+0.6 %** (2 SE 1.8 %) | **+4.2 %** (2 SE 3.9 %) | **-2.9 %** (2 SE 3.8 %) |

tokens per verify pass (cell counters) and ms per verify pass and sequence, per window
| window | 4096 | 32768 | 65536 |
|---|---|---|---|
| none a1 | 2.21 tok, 29.2 ms | 2.19 tok, 35.7 ms | 2.30 tok, 43.6 ms |
| none a4 | 2.21 tok, 29.2 ms | 2.20 tok, 35.8 ms | 2.28 tok, 43.4 ms |
| fast a2 | 2.21 tok, 28.9 ms | 2.26 tok, 35.7 ms | 2.23 tok, 43.3 ms |
| fast a3 | 2.22 tok, 29.1 ms | 2.31 tok, 35.9 ms | 2.21 tok, 43.5 ms |

solo top-k / keyed graph / fallback / total spec steps per window (sum over the cell's sizes)
none a1: solo 0 graph 0 fallback 0 steps 10978
none a4: solo 0 graph 0 fallback 0 steps 10996
fast a2: solo 265 graph 247 fallback 0 steps 10983
fast a3: solo 395 graph 344 fallback 0 steps 10887
=== s2-T0.7-k20

s2-T0.7-k20: combined tok/s over the overlap window (reps)
| window | 4096 |
|---|---|
| none a1 | 153.8 (8) |
| none a4 | 149.2 (8) |
| fast a2 | 150.0 (8) |
| fast a3 | 152.0 (8) |
| **none pooled** | 151.5 (16) |
| **fast pooled** | 151.0 (16) |
| fast vs none | **-0.3 %** (2 SE 2.3 %) |

s2-T0.7-k20: combined tok/s over the whole request (reps)
| window | 4096 |
|---|---|
| none a1 | 134.7 (8) |
| none a4 | 131.2 (8) |
| fast a2 | 133.4 (8) |
| fast a3 | 134.1 (8) |
| **none pooled** | 132.9 (16) |
| **fast pooled** | 133.8 (16) |
| fast vs none | **+0.6 %** (2 SE 1.8 %) |

s2-T0.7-k20: per-stream tok/s in the overlap (reps)
| window | 4096 |
|---|---|
| none a1 | 76.9 (16) |
| none a4 | 74.6 (16) |
| fast a2 | 75.0 (16) |
| fast a3 | 76.0 (16) |
| **none pooled** | 75.8 (32) |
| **fast pooled** | 75.5 (32) |
| fast vs none | **-0.4 %** (2 SE 2.5 %) |

tokens per verify pass (cell counters) and ms per verify pass and sequence, per window
| window | 4096 |
|---|---|
| none a1 | 2.25 tok, 29.2 ms |
| none a4 | 2.20 tok, 29.5 ms |
| fast a2 | 2.17 tok, 28.9 ms |
| fast a3 | 2.21 tok, 29.0 ms |

solo top-k / keyed graph / fallback / total spec steps per window (sum over the cell's sizes)
none a1: solo 0 graph 0 fallback 0 steps 3620
none a4: solo 0 graph 0 fallback 0 steps 3690
fast a2: solo 111 graph 105 fallback 0 steps 3755
fast a3: solo 152 graph 147 fallback 0 steps 3690
=== s1-T0.7-k0

s1-T0.7-k0: combined tok/s over the overlap window (reps)
| window | 4096 |
|---|---|
| none a1 | 85.6 (8) |
| none a4 | 85.5 (8) |
| fast a2 | 96.5 (8) |
| fast a3 | 99.6 (8) |
| **none pooled** | 85.6 (16) |
| **fast pooled** | 98.1 (16) |
| fast vs none | **+14.6 %** (2 SE 4.4 %) |

s1-T0.7-k0: combined tok/s over the whole request (reps)
| window | 4096 |
|---|---|
| none a1 | 85.6 (8) |
| none a4 | 85.5 (8) |
| fast a2 | 96.5 (8) |
| fast a3 | 99.7 (8) |
| **none pooled** | 85.6 (16) |
| **fast pooled** | 98.1 (16) |
| fast vs none | **+14.6 %** (2 SE 4.4 %) |

s1-T0.7-k0: per-stream tok/s in the overlap (reps)
| window | 4096 |
|---|---|
| none a1 | 85.6 (8) |
| none a4 | 85.5 (8) |
| fast a2 | 96.5 (8) |
| fast a3 | 99.6 (8) |
| **none pooled** | 85.6 (16) |
| **fast pooled** | 98.1 (16) |
| fast vs none | **+14.6 %** (2 SE 4.4 %) |

tokens per verify pass (cell counters) and ms per verify pass and sequence, per window
| window | 4096 |
|---|---|
| none a1 | 2.21 tok, 25.8 ms |
| none a4 | 2.23 tok, 26.1 ms |
| fast a2 | 2.19 tok, 22.7 ms |
| fast a3 | 2.33 tok, 23.4 ms |

solo top-k / keyed graph / fallback / total spec steps per window (sum over the cell's sizes)
none a1: solo 0 graph 0 fallback 0 steps 1853
none a4: solo 0 graph 0 fallback 0 steps 1829
fast a2: solo 0 graph 1863 fallback 1863 steps 1863
fast a3: solo 0 graph 1749 fallback 1753 steps 1753
=== s2-T0-k-

s2-T0-k-: combined tok/s over the overlap window (reps)
| window | 4096 |
|---|---|
| none a1 | 159.7 (8) |
| none a4 | 160.5 (8) |
| fast a2 | 158.5 (8) |
| fast a3 | 160.3 (8) |
| **none pooled** | 160.1 (16) |
| **fast pooled** | 159.4 (16) |
| fast vs none | **-0.4 %** (2 SE 3.0 %) |

s2-T0-k-: combined tok/s over the whole request (reps)
| window | 4096 |
|---|---|
| none a1 | 139.7 (8) |
| none a4 | 141.0 (8) |
| fast a2 | 138.8 (8) |
| fast a3 | 140.5 (8) |
| **none pooled** | 140.3 (16) |
| **fast pooled** | 139.6 (16) |
| fast vs none | **-0.5 %** (2 SE 2.1 %) |

s2-T0-k-: per-stream tok/s in the overlap (reps)
| window | 4096 |
|---|---|
| none a1 | 79.8 (16) |
| none a4 | 80.2 (16) |
| fast a2 | 79.2 (16) |
| fast a3 | 80.2 (16) |
| **none pooled** | 80.0 (32) |
| **fast pooled** | 79.7 (32) |
| fast vs none | **-0.4 %** (2 SE 3.8 %) |

tokens per verify pass (cell counters) and ms per verify pass and sequence, per window
| window | 4096 |
|---|---|
| none a1 | 2.29 tok, 28.7 ms |
| none a4 | 2.31 tok, 28.8 ms |
| fast a2 | 2.29 tok, 28.9 ms |
| fast a3 | 2.30 tok, 28.6 ms |

solo top-k / keyed graph / fallback / total spec steps per window (sum over the cell's sizes)
none a1: solo 0 graph 0 fallback 0 steps 3549
none a4: solo 0 graph 0 fallback 0 steps 3522
fast a2: solo 0 graph 0 fallback 0 steps 3558
fast a3: solo 0 graph 0 fallback 0 steps 3543
