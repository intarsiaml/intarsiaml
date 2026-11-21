| run | n | score | 2 SE | truncated | finish stop / length / other | thinking tok mean (median, p90, max) | answer tok mean | total tok mean | empty / no-thinking / inline `<think>` | wall s | tokenizer sum vs server |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 200 | 0.985 (197) | 0.017 | 0 | 200 / 0 / 0 | 1949.6 (1608.0, 3277, 8569) | 22.5 | 1975.1 | 0 / 0 / 0 | 4964.6 | 3.0 |
| A2 | 200 | 0.985 (197) | 0.017 | 0 | 200 / 0 / 0 | 2178.4 (1962.5, 3568, 11407) | 28.0 | 2209.4 | 0 / 0 / 0 | 5557.3 | 3 |
| B | 200 | 0.985 (197) | 0.017 | 0 | 200 / 0 / 0 | 2094.0 (1767.0, 3450, 8125) | 33.9 | 2131.9 | 0 / 0 / 0 | 2933.5 | 4 |

| run | number-only final answers | worked final answers (anything else) | worked: right of count |
|---|---|---|---|
| A1 | 165 | 35 | 34 of 35 |
| A2 | 152 | 48 | 47 of 48 |
| B | 143 | 57 | 56 of 57 |

| pair | same correctness | only first right | only second right | both right | both wrong | McNemar p | total tok: ratio of means 2nd/1st (95 % bootstrap) | second longer on / shorter on (sign test p) | thinking tok ratio |
|---|---|---|---|---|---|---|---|---|---|
| A1 vs A2 | 200 of 200 | 0 | 0 | 197 | 3 | 1.000 | 1.119 (1.041 to 1.207) | 117 / 83 (0.019) | 1.117 |
| A1 vs B | 198 of 200 | 1 | 1 | 196 | 2 | 1.000 | 1.079 (1.003 to 1.161) | 113 / 87 (0.077) | 1.074 |
| A2 vs B | 198 of 200 | 1 | 1 | 196 | 2 | 1.000 | 0.965 (0.892 to 1.042) | 96 / 104 (0.621) | 0.961 |

### disagreements A1 vs A2: 0 ids
### disagreements A1 vs B: 2 ids
- gsm8k-00119: gold '95200'; A1: extracted '99077' score 0 tokens 8787 (thinking 8569); B: extracted '95200' score 1 tokens 8361 (thinking 8125)
- gsm8k-00182: gold '23'; A1: extracted '23' score 1 tokens 3207 (thinking 3096); B: extracted '24' score 0 tokens 2673 (thinking 2667)
### disagreements A2 vs B: 2 ids
- gsm8k-00119: gold '95200'; A2: extracted '99076.92' score 0 tokens 11418 (thinking 11407); B: extracted '95200' score 1 tokens 8361 (thinking 8125)
- gsm8k-00182: gold '23'; A2: extracted '23' score 1 tokens 3621 (thinking 3616); B: extracted '24' score 0 tokens 2673 (thinking 2667)
### largest length differences A1 vs A2
- gsm8k-00050: A1 6295 tok (score 1), A2 1598 tok (score 1)
- gsm8k-00058: A1 2743 tok (score 1), A2 6207 tok (score 1)
- gsm8k-00198: A1 1134 tok (score 1), A2 4191 tok (score 1)
- gsm8k-00002: A1 5255 tok (score 1), A2 2272 tok (score 1)
- gsm8k-00144: A1 1275 tok (score 1), A2 4209 tok (score 1)
### largest length differences A1 vs B
- gsm8k-00050: A1 6295 tok (score 1), B 1777 tok (score 1)
- gsm8k-00045: A1 2355 tok (score 1), B 6318 tok (score 1)
- gsm8k-00074: A1 5102 tok (score 1), B 1307 tok (score 1)
- gsm8k-00075: A1 2685 tok (score 1), B 6170 tok (score 1)
- gsm8k-00042: A1 4049 tok (score 1), B 1252 tok (score 1)
### largest length differences A2 vs B
- gsm8k-00058: A2 6207 tok (score 1), B 1627 tok (score 1)
- gsm8k-00075: A2 2172 tok (score 1), B 6170 tok (score 1)
- gsm8k-00002: A2 2272 tok (score 1), B 5942 tok (score 1)
- gsm8k-00045: A2 2786 tok (score 1), B 6318 tok (score 1)
- gsm8k-00119: A2 11418 tok (score 0), B 8361 tok (score 1)
