# step 7: recent-window items, off vs on: no defect per the reading rule

## chunk 2048: 144 pairs scored in both arms (off ran 144 rows, on 144)

| length | distance bucket | pairs | off right | on right | b: off right, on wrong | c: on right, off wrong | one-sided p(b) |
|---|---|---:|---:|---:|---:|---:|---:|
| 4k | <=64 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 4k | 65-128 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 4k | 129-256 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 4k | 257-1024 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 4k | 1025-4096 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 4k | far (>= half) | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 32k | <=64 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 32k | 65-128 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 32k | 129-256 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 32k | 257-1024 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 32k | 1025-4096 | 12 | 12 | 12 | 0 | 0 | 1.000 |
| 32k | far (>= half) | 12 | 12 | 12 | 0 | 0 | 1.000 |

**Pooled over lengths (the cells the reading rule reads)**

| distance bucket | pairs | off right | on right | b | c | one-sided p(b) |
|---|---:|---:|---:|---:|---:|---:|
| <=64 | 24 | 24 | 24 | 0 | 0 | 1.000 |
| 65-128 | 24 | 24 | 24 | 0 | 0 | 1.000 |
| 129-256 | 24 | 24 | 24 | 0 | 0 | 1.000 |
| 257-1024 | 24 | 24 | 24 | 0 | 0 | 1.000 |
| 1025-4096 | 24 | 24 | 24 | 0 | 0 | 1.000 |
| far (>= half) | 24 | 24 | 24 | 0 | 0 | 1.000 |

**Reading rule, chunk 2048**

- no defect bucket <=64: n 24, off right 24, off right/on wrong b=0, on right/off wrong c=0, p1=1.000
- no defect bucket 65-128: n 24, off right 24, off right/on wrong b=0, on right/off wrong c=0, p1=1.000
- no defect bucket 129-256: n 24, off right 24, off right/on wrong b=0, on right/off wrong c=0, p1=1.000
- no defect bucket 257-1024: n 24, off right 24, off right/on wrong b=0, on right/off wrong c=0, p1=1.000
- no defect bucket 1025-4096: n 24, off right 24, off right/on wrong b=0, on right/off wrong c=0, p1=1.000
- no defect bucket far (>= half): n 24, off right 24, off right/on wrong b=0, on right/off wrong c=0, p1=1.000
- no defect pooled buckets 1-3: n 72, b=0, c=0, one-sided p=1.0000 (needs <= 0.01)
- off: 144 right; wrong answer is a decoy's code: 0; no code in the reply: 0; reply names more than one code: 0
- on: 144 right; wrong answer is a decoy's code: 0; no code in the reply: 0; reply names more than one code: 0
- localeval's own 'contains expected' score and this scorer disagree on 0 replies (a hedged reply that also names the code)

Discordant items (distance of the fact line's first token, floor-chunk flags at this chunk): none

