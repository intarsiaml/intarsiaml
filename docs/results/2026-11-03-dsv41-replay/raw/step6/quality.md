Paired off / on: 60 items scored in both arms (3 shards; 0 cases with a call error left out; 0 items scored in one arm only).

**Overall**

| set | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| all | 60 | 47 (0.783) | 47 (0.783) | 0 | 0 | 1.000 |

**Criterion**: at least 54 items scored in both arms; on not worse than off by 0.1 or more, and not worse by more than 0.05 with McNemar p < 0.05. This detects a large loss only: with this many items a loss of a few points is inside the noise, and the items depend on compressed KV far more than on the recent window (see the doc). Met (off 0.783, on 0.783, p 1.000).

**By depth of the EARLIEST fact (fraction of the document)**

| bin | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| < 0.10 | 31 | 24 (0.774) | 24 (0.774) | 0 | 0 | 1.000 |
| 0.10 to 0.20 | 19 | 16 (0.842) | 16 (0.842) | 0 | 0 | 1.000 |
| >= 0.20 | 10 | 7 (0.700) | 7 (0.700) | 0 | 0 | 1.000 |

**By depth of the LATEST fact**

| bin | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| < 0.90 (answer needs nothing near the end) | 21 | 16 (0.762) | 16 (0.762) | 0 | 0 | 1.000 |
| >= 0.90 | 39 | 31 (0.795) | 31 (0.795) | 0 | 0 | 1.000 |

**By number of facts in the first third of the document**

| bin | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| 0 to 1 | 28 | 27 (0.964) | 27 (0.964) | 0 | 0 | 1.000 |
| 2 to 3 | 20 | 11 (0.550) | 11 (0.550) | 0 | 0 | 1.000 |
| 4 or more | 12 | 9 (0.750) | 9 (0.750) | 0 | 0 | 1.000 |

No item has all its facts early: smallest latest-fact depth in the scored items 0.671.

**By kind**

| kind | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| hop2 | 11 | 10 (0.909) | 10 (0.909) | 0 | 0 | 1.000 |
| hop3 | 11 | 11 (1.000) | 11 (1.000) | 0 | 0 | 1.000 |
| agg_sum | 20 | 9 (0.450) | 9 (0.450) | 0 | 0 | 1.000 |
| agg_max | 9 | 8 (0.889) | 8 (0.889) | 0 | 0 | 1.000 |
| repo_sum | 9 | 9 (1.000) | 9 (1.000) | 0 | 0 | 1.000 |

**By size**

| size | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 50 | 40 (0.800) | 40 (0.800) | 0 | 0 | 1.000 |
| 65K | 10 | 7 (0.700) | 7 (0.700) | 0 | 0 | 1.000 |

Discordant items: none

Reply text identical off vs on for 56 of 60 items.
