# Pilot 2 (set seed 'pilot2', chunk 512, format `FINAL: <answer>`, max_tokens 160; localeval scores)

| kind | level | correct/items | replies cut at 160 tokens |
|---|---|---|---|
| agg_max | L3 | 2/3 | 1 |
| agg_sum | L3 | 1/3 | 0 |
| agg_sum | L4 | 0/3 | 3 |
| hop2 | L3 | 3/3 | 0 |
| hop3 | L3 | 3/3 | 0 |
| repo_sum | L3 | 2/2 | 0 |
| repo_sum | L4 | 4/4 | 0 |

Total 15/21 = 0.71. The format works (localeval score equals the exact parse of the FINAL line for all 21). Four replies ran out of the 160-token budget before the FINAL line (agg_sum L4 three times, agg_max L3 once) and were scored wrong: the main window uses max_tokens 400. hop2/hop3 L3 and repo_sum L3/L4 were right in 12 of 12 (the model is good at them at 512), agg_sum L3/L4 is the hard kind (1 right of 6; 3 of the 5 failures were cut replies, the others a missed shard in the sum), so the final set is weighted to agg_sum.
