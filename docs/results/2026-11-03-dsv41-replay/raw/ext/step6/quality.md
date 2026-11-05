Paired off / on: 95 items scored in both arms (5 shards; 0 cases with a call error left out; 0 items scored in one arm only).

**Overall**

| set | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| all | 95 | 72 (0.758) | 69 (0.726) | 3 | 0 | 0.250 |

**Criterion**: at least 85 items scored in both arms; on not worse than off by 0.1 or more, and not worse by more than 0.05 with McNemar p < 0.05. This detects a large loss only: with this many items a loss of a few points is inside the noise, and the items depend on compressed KV far more than on the recent window (see the doc). Met (off 0.758, on 0.726, p 0.250).

**By depth of the EARLIEST fact (fraction of the document)**

| bin | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| < 0.10 | 41 | 32 (0.780) | 30 (0.732) | 2 | 0 | 0.500 |
| 0.10 to 0.20 | 35 | 26 (0.743) | 25 (0.714) | 1 | 0 | 1.000 |
| >= 0.20 | 19 | 14 (0.737) | 14 (0.737) | 0 | 0 | 1.000 |

**By depth of the LATEST fact**

| bin | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| < 0.90 (answer needs nothing near the end) | 30 | 24 (0.800) | 23 (0.767) | 1 | 0 | 1.000 |
| >= 0.90 | 65 | 48 (0.738) | 46 (0.708) | 2 | 0 | 0.500 |

**By number of facts in the first third of the document**

| bin | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| 0 to 1 | 42 | 38 (0.905) | 37 (0.881) | 1 | 0 | 1.000 |
| 2 to 3 | 34 | 24 (0.706) | 23 (0.676) | 1 | 0 | 1.000 |
| 4 or more | 19 | 10 (0.526) | 9 (0.474) | 1 | 0 | 1.000 |

No item has all its facts early: smallest latest-fact depth in the scored items 0.615.

**By kind**

| kind | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| hop2 | 19 | 19 (1.000) | 18 (0.947) | 1 | 0 | 1.000 |
| hop3 | 13 | 12 (0.923) | 12 (0.923) | 0 | 0 | 1.000 |
| agg_sum | 28 | 9 (0.321) | 8 (0.286) | 1 | 0 | 1.000 |
| agg_max | 20 | 17 (0.850) | 16 (0.800) | 1 | 0 | 1.000 |
| repo_sum | 15 | 15 (1.000) | 15 (1.000) | 0 | 0 | 1.000 |

**By size**

| size | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 75 | 59 (0.787) | 58 (0.773) | 1 | 0 | 1.000 |
| 65K | 20 | 13 (0.650) | 11 (0.550) | 2 | 0 | 0.500 |

Discordant items: `agg_sum-L4-32k-009` (depths [0.067, 0.259, 0.44, 0.44, 0.609, 0.651, 0.678, 0.751, 0.792, 0.888, 0.953, 0.999]; off right, on wrong), `agg_max-L3-65k-003` (depths [0.085, 0.09, 0.138, 0.299, 0.36, 0.539, 0.583, 0.661]; off right, on wrong), `hop2-L3-65k-004` (depths [0.133, 0.932]; off right, on wrong)

Reply text identical off vs on for 78 of 95 items.
