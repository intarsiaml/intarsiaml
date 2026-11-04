### Accuracy per arm

| arm | items | correct | accuracy | Wilson 95% CI | exact-parse correct | no FINAL line | truncated (>= 400 tokens) | call errors | cached tokens (sum) | mean TTFT 32K / 65K (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 (ab0) | 180 | 133 | 0.739 | 0.670 to 0.798 | 133 | 1 | 1 | 0 | 0 | 52.7 / 111.4 |
| 2048 (ab1) | 180 | 134 | 0.744 | 0.676 to 0.803 | 134 | 0 | 0 | 0 | 0 | 40.2 / 84.8 |
| 512-hc0 (ab2) | 180 | 125 | 0.694 | 0.624 to 0.757 | 125 | 1 | 1 | 0 | 0 | 52.3 / 110.3 |

### Paired: 512 (reference) vs 2048: 180 items scored in both arms (equal prompt token count in 180)

| both right | only 512 right | only 2048 right | both wrong | exact McNemar p | 2048 - 512 | Newcombe 95% CI |
|---|---|---|---|---|---|---|
| 127 | 6 | 7 | 40 | 1.000 | +0.006 (+1.0 items) | -0.035 to +0.046 |

Largest loss of 2048 against 512 that the 95% CI allows: 3.5 percentage points (6.2 of 180 items).

Reply text differs for 103 of 180 items; the final answer differs for 31 of 180.

**By kind**

| kind | items | 512 correct | 2048 correct | only 512 | only 2048 | both wrong | final answer differs |
|---|---|---|---|---|---|---|---|
| hop2 | 36 | 35 | 35 | 1 | 1 | 0 | 2 |
| hop3 | 24 | 23 | 23 | 1 | 1 | 0 | 2 |
| agg_sum | 60 | 20 | 20 | 3 | 3 | 37 | 24 |
| agg_max | 36 | 31 | 32 | 1 | 2 | 3 | 3 |
| repo_sum | 24 | 24 | 24 | 0 | 0 | 0 | 0 |

**By level**

| level | items | 512 correct | 2048 correct | only 512 | only 2048 | both wrong |
|---|---|---|---|---|---|---|
| 3 | 115 | 97 | 99 | 3 | 5 | 13 |
| 4 | 65 | 36 | 35 | 3 | 2 | 27 |

**By size**

| size | items | 512 correct | 2048 correct | only 512 | only 2048 | both wrong |
|---|---|---|---|---|---|---|
| 32 | 150 | 113 | 113 | 4 | 4 | 33 |
| 65 | 30 | 20 | 21 | 2 | 3 | 7 |

**By span of the facts (fraction of the document)**

| span of the facts (fraction of the document) | items | 512 correct | 2048 correct | only 512 | only 2048 | both wrong |
|---|---|---|---|---|---|---|
| span < 0.7 | 41 | 32 | 33 | 1 | 2 | 7 |
| 0.7 to 0.85 | 84 | 62 | 65 | 2 | 5 | 17 |
| >= 0.85 | 55 | 39 | 36 | 3 | 0 | 16 |

Discordant items: `hop2-L3-32k-005` (512 wrong, 2048 right), `agg_sum-L3-32k-007` (512 wrong, 2048 right), `agg_sum-L4-32k-007` (512 wrong, 2048 right), `agg_sum-L4-32k-009` (512 right, 2048 wrong), `agg_sum-L4-32k-014` (512 right, 2048 wrong), `hop3-L3-32k-015` (512 right, 2048 wrong), `agg_max-L3-32k-020` (512 right, 2048 wrong), `agg_max-L3-32k-028` (512 wrong, 2048 right), `agg_sum-L4-65k-001` (512 wrong, 2048 right), `agg_max-L3-65k-002` (512 wrong, 2048 right), `hop3-L3-65k-003` (512 wrong, 2048 right), `hop2-L3-65k-004` (512 right, 2048 wrong), `agg_sum-L4-65k-005` (512 right, 2048 wrong)


### Paired: 512 (reference) vs 512-hc0: 180 items scored in both arms (equal prompt token count in 180)

| both right | only 512 right | only 512-hc0 right | both wrong | exact McNemar p | 512-hc0 - 512 | Newcombe 95% CI |
|---|---|---|---|---|---|---|
| 121 | 12 | 4 | 43 | 0.077 | -0.044 (-8.0 items) | -0.088 to -0.001 |

Largest loss of 512-hc0 against 512 that the 95% CI allows: 8.8 percentage points (15.9 of 180 items).

Reply text differs for 107 of 180 items; the final answer differs for 30 of 180.

**By kind**

| kind | items | 512 correct | 512-hc0 correct | only 512 | only 512-hc0 | both wrong | final answer differs |
|---|---|---|---|---|---|---|---|
| hop2 | 36 | 35 | 35 | 1 | 1 | 0 | 2 |
| hop3 | 24 | 23 | 22 | 1 | 0 | 1 | 1 |
| agg_sum | 60 | 20 | 16 | 6 | 2 | 38 | 22 |
| agg_max | 36 | 31 | 28 | 4 | 1 | 4 | 5 |
| repo_sum | 24 | 24 | 24 | 0 | 0 | 0 | 0 |

**By level**

| level | items | 512 correct | 512-hc0 correct | only 512 | only 512-hc0 | both wrong |
|---|---|---|---|---|---|---|
| 3 | 115 | 97 | 94 | 7 | 4 | 14 |
| 4 | 65 | 36 | 31 | 5 | 0 | 29 |

**By size**

| size | items | 512 correct | 512-hc0 correct | only 512 | only 512-hc0 | both wrong |
|---|---|---|---|---|---|---|
| 32 | 150 | 113 | 108 | 8 | 3 | 34 |
| 65 | 30 | 20 | 17 | 4 | 1 | 9 |

**By span of the facts (fraction of the document)**

| span of the facts (fraction of the document) | items | 512 correct | 512-hc0 correct | only 512 | only 512-hc0 | both wrong |
|---|---|---|---|---|---|---|
| span < 0.7 | 41 | 32 | 32 | 2 | 2 | 7 |
| 0.7 to 0.85 | 84 | 62 | 58 | 6 | 2 | 20 |
| >= 0.85 | 55 | 39 | 35 | 4 | 0 | 16 |

Discordant items: `agg_sum-L4-32k-001` (512 right, 512-hc0 wrong), `hop2-L3-32k-005` (512 wrong, 512-hc0 right), `agg_max-L3-32k-006` (512 right, 512-hc0 wrong), `hop3-L3-32k-007` (512 right, 512-hc0 wrong), `agg_sum-L3-32k-007` (512 wrong, 512-hc0 right), `agg_sum-L4-32k-009` (512 right, 512-hc0 wrong), `agg_sum-L4-32k-013` (512 right, 512-hc0 wrong), `agg_max-L3-32k-013` (512 right, 512-hc0 wrong), `agg_sum-L4-32k-014` (512 right, 512-hc0 wrong), `agg_max-L3-32k-027` (512 right, 512-hc0 wrong), `agg_max-L3-32k-028` (512 wrong, 512-hc0 right), `agg_sum-L3-65k-001` (512 right, 512-hc0 wrong), `agg_sum-L3-65k-003` (512 wrong, 512-hc0 right), `agg_max-L3-65k-003` (512 right, 512-hc0 wrong), `hop2-L3-65k-004` (512 right, 512-hc0 wrong), `agg_sum-L4-65k-005` (512 right, 512-hc0 wrong)


### Is the shift of 2048 larger than the noise control 512-hc0? (180 items scored in all three arms)

| measure against 512 | 2048 | 512-hc0 | only 2048 moves | only 512-hc0 moves | sign-test p (two-sided) |
|---|---|---|---|---|---|
| reply text differs | 103 | 107 | 19 | 23 | 0.644 |
| final answer differs | 31 | 30 | 13 | 12 | 1.000 |
| correctness flips | 13 | 16 | 6 | 9 | 0.607 |

Correct on these items: 512 133/180, 2048 134/180, 512-hc0 125/180.

2048 against 512-hc0: only 2048 right 12, only 512-hc0 right 3, McNemar p 0.035; 2048 - 512-hc0 +0.050, Newcombe 95% CI +0.008 to +0.092.

