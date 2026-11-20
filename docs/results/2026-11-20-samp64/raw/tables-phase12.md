
### sampled T=0.7: decode tok/s, median of 5 reps (min-max)

| config | 4k | 32k | 64k |
|---|---|---|---|
| spec on, k=2 (main) | 95.4 (92-97) | 69.1 (68-73) | 52.0 (51-54) |
| `--spec off` | 84.6 (83-85) | 69.5 (69-70) | 57.6 (57-58) |
| `--spec-k 1` | 91.0 (90-96) | 71.6 (70-72) | 56.5 (53-57) |
| `INTARSIA_SPEC_K_MIN=1` | 97.4 (96-126) | 71.9 (67-73) | 52.7 (50-60) |
| `INTARSIA_SPEC_COSTGATE=1` | 94.2 (94-97) | 70.7 (69-76) | 55.0 (50-57) |

MTP acceptance (accepted / proposed) and tokens per verify pass, sampled T=0.7:

| config | 4k | 32k | 64k |
|---|---|---|---|
| spec on, k=2 (main) | 56.5%, 2.13 | 55.3%, 2.10 | 54.3%, 2.08 |
| `--spec-k 1` | 73.2%, 1.73 | 71.9%, 1.72 | 71.0%, 1.71 |
| `INTARSIA_SPEC_K_MIN=1` | 65.5%, 2.29 | 57.1%, 2.13 | 57.8%, 2.14 |
| `INTARSIA_SPEC_COSTGATE=1` | 57.0%, 2.14 | 58.8%, 2.17 | 58.1%, 2.16 |

### greedy: decode tok/s, median of 5 reps (min-max)

| config | 4k | 32k | 64k |
|---|---|---|---|
| spec on, k=2 (main) | 120.3 (116-146) | 81.3 (79-86) | 60.4 (60-63) |
| `--spec off` | 85.8 (86-89) | 71.7 (71-72) | 59.1 (59-60) |
| `--spec-k 1` | 119.0 (112-124) | 84.8 (84-94) | 65.5 (64-68) |
| `INTARSIA_SPEC_K_MIN=1` | 118.9 (112-151) | 82.6 (81-85) | 60.2 (57-68) |
| `INTARSIA_SPEC_COSTGATE=1` | 115.3 (114-135) | 81.4 (80-83) | 59.1 (57-67) |

MTP acceptance (accepted / proposed) and tokens per verify pass, greedy:

| config | 4k | 32k | 64k |
|---|---|---|---|
| spec on, k=2 (main) | 63.8%, 2.27 | 58.4%, 2.17 | 61.2%, 2.22 |
| `--spec-k 1` | 81.2%, 1.81 | 77.9%, 1.78 | 79.0%, 1.79 |
| `INTARSIA_SPEC_K_MIN=1` | 68.2%, 2.35 | 60.4%, 2.19 | 61.5%, 2.22 |
| `INTARSIA_SPEC_COSTGATE=1` | 62.4%, 2.25 | 58.0%, 2.16 | 60.6%, 2.21 |

### Derived cost per verify pass (ms per pass = tokens per pass x 1000 / tok/s) against the plain step of the spec-off run

| | 4k | 32k | 64k |
|---|---|---|---|
| sampled spec pass, ms | 22.3 | 30.4 | 40.1 |
| sampled plain step, ms | 11.8 | 14.4 | 17.4 |
| sampled pass / plain step (tokens per pass needed to break even, got) | 1.89 (needs 1.89, gets 2.13) | 2.11 (needs 2.11, gets 2.10) | 2.31 (needs 2.31, gets 2.08) |
| greedy spec pass, ms | 18.9 | 26.6 | 36.8 |
| greedy plain step, ms | 11.7 | 14.0 | 16.9 |
| greedy pass / plain step (tokens per pass needed to break even, got) | 1.62 (needs 1.62, gets 2.27) | 1.91 (needs 1.91, gets 2.17) | 2.17 (needs 2.17, gets 2.22) |
