# Pilot (set seed 'pilot', not part of the final set)

Format `FINAL: [answer]`: the model dropped the brackets in every reply, so localeval's score was 0/33; the replies were re-scored offline (exact parse of the FINAL line). ab0 = chunk 512, ab1 = chunk 2048 (65K items only, memory check).

| arm | kind | level | size | correct/items |
|---|---|---|---|---|
| ab0 | agg_max | L1 | 32K | 2/2 |
| ab0 | agg_max | L2 | 32K | 1/2 |
| ab0 | agg_max | L3 | 32K | 2/2 |
| ab0 | agg_sum | L1 | 32K | 2/2 |
| ab0 | agg_sum | L2 | 32K | 2/2 |
| ab0 | agg_sum | L2 | 65K | 1/1 |
| ab0 | agg_sum | L3 | 32K | 0/2 |
| ab0 | hop2 | L1 | 32K | 2/2 |
| ab0 | hop2 | L2 | 32K | 2/2 |
| ab0 | hop2 | L3 | 32K | 1/2 |
| ab0 | hop3 | L1 | 32K | 2/2 |
| ab0 | hop3 | L2 | 32K | 1/2 |
| ab0 | hop3 | L2 | 65K | 0/1 |
| ab0 | hop3 | L3 | 32K | 2/2 |
| ab0 | repo_sum | L1 | 32K | 2/2 |
| ab0 | repo_sum | L2 | 32K | 2/2 |
| ab0 | repo_sum | L2 | 65K | 1/1 |
| ab0 | repo_sum | L3 | 32K | 2/2 |
| ab1 | agg_sum | L2 | 65K | 1/1 |
| ab1 | hop3 | L2 | 65K | 1/1 |
| ab1 | repo_sum | L2 | 65K | 1/1 |

- ab0 32K: 25/30 correct; localeval score 0; mean TTFT 52.0 s; mean completion tokens 85
- ab0 65K: 2/3 correct; localeval score 0; mean TTFT 109.7 s; mean completion tokens 86
- ab1 65K: 3/3 correct; localeval score 0; mean TTFT 84.4 s; mean completion tokens 90
- ab0 32K level 1: 10/10
- ab0 32K level 2: 8/10
- ab0 32K level 3: 7/10
