### Accuracy per arm

| arm | items | correct | accuracy | Wilson 95% CI | truncated replies (>= 48 tokens) | call errors | cached tokens (sum / max per item) | prompt tokens (min-max) |
|---|---|---|---|---|---|---|---|---|
| 512 (ab0) | 52 | 38 | 0.731 | 0.597 to 0.832 | 1 | 0 | 0 / 0 | 30309-31213 |
| 2048 (ab8) | 52 | 39 | 0.750 | 0.618 to 0.848 | 0 | 0 | 0 / 0 | 30309-31213 |

### Paired: 512 (reference) vs 2048, 52 items scored in both arms (prompt token count equal in 52)

| both right | only 512 right | only 2048 right | both wrong | exact McNemar p (two-sided) | 2048 - 512 accuracy | Newcombe 95% CI of that difference |
|---|---|---|---|---|---|---|
| 38 | 0 | 1 | 13 | 1.000 | +0.019 (+1.0 items) | -0.028 to +0.070 |

Largest loss of 2048 against 512 that the 95% CI still allows: 2.8 percentage points (1.5 of 52 items).

Sensitivity (quote-insensitive scoring of the string constants, same replies): 512 41/52, 2048 42/52; only 512 right 0, only 2048 right 1; McNemar p 1.000; 2048 - 512 +0.019, Newcombe 95% CI -0.033 to +0.076.

Items whose reply text differs between the arms (after strip): 4 of 52: needle_hop-00-d08, needle_hop-01-d55, needle_hop-06-d80, needle_hop-09-d71.

Discordant items:

- `needle_hop-09-d71` expected `TZ-7633-QA`; 512: `Lumder` (wrong); 2048: `TZ-7633-QA` (right)

**By kind**

| kind | items | 512 correct | 2048 correct | only 512 | only 2048 | both wrong |
|---|---|---|---|---|---|---|
| needle | 12 | 12 | 12 | 0 | 0 | 0 |
| needle_dist | 12 | 12 | 12 | 0 | 0 | 0 |
| needle_hop | 12 | 2 | 3 | 0 | 1 | 9 |
| qa_const | 16 | 12 | 12 | 0 | 0 | 4 |

**By depth of the answer-bearing text** (fraction of the document before it; qa_const: before the asked file)

| depth | items | 512 correct | 2048 correct | only 512 | only 2048 | both wrong |
|---|---|---|---|---|---|---|
| early (< 0.33) | 16 | 11 | 11 | 0 | 0 | 5 |
| middle (0.33 to 0.67) | 18 | 14 | 14 | 0 | 0 | 4 |
| late (>= 0.67) | 18 | 13 | 14 | 0 | 1 | 4 |

**By kind and depth, reference arm only** (correct / items)

| kind | early | middle | late |
|---|---|---|---|
| needle | 4/4 | 4/4 | 4/4 |
| needle_dist | 4/4 | 4/4 | 4/4 |
| needle_hop | 0/4 | 1/4 | 1/4 |
| qa_const | 3/4 | 5/6 | 4/6 |
