| run | n | score | 2 SE | truncated | finish stop / length / other | thinking tok mean (median, p90, max) | answer tok mean | total tok mean | empty / no-thinking / inline `<think>` | wall s | tokenizer sum vs server |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 150 | 0.933 (140) | 0.041 | 1 | 149 / 1 / 0 | 1963.2 (1550.5, 3002, 16383) | 6.8 | 1973.1 | 1 / 0 / 0 | 4179.8 | 3.1 |
| A2 | 150 | 0.920 (138) | 0.044 | 1 | 149 / 1 / 0 | 2031.3 (1580.5, 2999, 16317) | 6.8 | 2041.6 | 1 / 0 / 0 | 4307.3 | 3.5 |
| B | 150 | 0.913 (137) | 0.046 | 2 | 148 / 2 / 0 | 1977.5 (1466.5, 2684, 16384) | 6.8 | 1988.2 | 2 / 0 / 0 | 2248.0 | 4.0 |


| task | A1: score / truncated / mean tok | A2: score / truncated / mean tok | B: score / truncated / mean tok |
|---|---|---|---|
| dyck_languages | 26/30 / 1 / 3253 | 26/30 / 1 / 3427 | 26/30 / 2 / 3518 |
| geometric_shapes | 24/30 / 0 / 2037 | 22/30 / 0 / 1952 | 22/30 / 0 / 1871 |
| logical_deduction_seven_objects | 30/30 / 0 / 1691 | 30/30 / 0 / 1796 | 30/30 / 0 / 1558 |
| multistep_arithmetic_two | 30/30 / 0 / 1270 | 30/30 / 0 / 1355 | 30/30 / 0 / 1424 |
| tracking_shuffled_objects_seven_objects | 30/30 / 0 / 1614 | 30/30 / 0 / 1677 | 29/30 / 0 / 1570 |

| pair | same correctness | only first right | only second right | both right | both wrong | McNemar p | total tok: ratio of means 2nd/1st (95 % bootstrap) | second longer on / shorter on (sign test p) | thinking tok ratio |
|---|---|---|---|---|---|---|---|---|---|
| A1 vs A2 | 146 of 150 | 3 | 1 | 137 | 9 | 0.625 | 1.035 (0.973 to 1.103) | 78 / 70 (0.565) | 1.035 |
| A1 vs B | 143 of 150 | 5 | 2 | 135 | 8 | 0.453 | 1.008 (0.940 to 1.091) | 71 / 78 (0.623) | 1.007 |
| A2 vs B | 145 of 150 | 3 | 2 | 135 | 10 | 1.000 | 0.974 (0.920 to 1.030) | 68 / 81 (0.326) | 0.973 |

### disagreements A1 vs A2: 4 ids
- bbh-00002:dyck_languages: gold '}'; A1: extracted '}' score 1 tokens 6962 (thinking 6954); A2: extracted '' score 0 tokens 11111 (thinking 11099)
- bbh-00014:dyck_languages: gold ') ) }'; A1: extracted ') }' score 0 tokens 9801 (thinking 9792); A2: extracted ') ) }' score 1 tokens 5258 (thinking 5248)
- bbh-00092:geometric_shapes: gold '(K)'; A1: extracted '(K)' score 1 tokens 3767 (thinking 3757); A2: extracted '(H)' score 0 tokens 4667 (thinking 4657)
- bbh-00108:geometric_shapes: gold '(K)'; A1: extracted '(K)' score 1 tokens 3505 (thinking 3495); A2: extracted '(H)' score 0 tokens 3210 (thinking 3200)
### disagreements A1 vs B: 7 ids
- bbh-00013:dyck_languages: gold ') >'; A1: extracted ') >' score 1 tokens 14103 (thinking 14087); B: extracted '' score 0 tokens 16384 (thinking 16383)
- bbh-00014:dyck_languages: gold ') ) }'; A1: extracted ') }' score 0 tokens 9801 (thinking 9792); B: extracted ') ) }' score 1 tokens 6600 (thinking 6589)
- bbh-00019:dyck_languages: gold '} > ]'; A1: extracted '] >' score 0 tokens 3787 (thinking 3776); B: extracted '} > ]' score 1 tokens 3344 (thinking 3333)
- bbh-00022:dyck_languages: gold '> )'; A1: extracted '> )' score 1 tokens 5459 (thinking 5450); B: extracted '> ) )' score 0 tokens 14714 (thinking 14703)
- bbh-00092:geometric_shapes: gold '(K)'; A1: extracted '(K)' score 1 tokens 3767 (thinking 3757); B: extracted '(H)' score 0 tokens 3766 (thinking 3755)
### disagreements A2 vs B: 5 ids
- bbh-00002:dyck_languages: gold '}'; A2: extracted '' score 0 tokens 11111 (thinking 11099); B: extracted '}' score 1 tokens 8340 (thinking 8331)
- bbh-00013:dyck_languages: gold ') >'; A2: extracted ') >' score 1 tokens 14082 (thinking 14076); B: extracted '' score 0 tokens 16384 (thinking 16383)
- bbh-00019:dyck_languages: gold '} > ]'; A2: extracted '> ]' score 0 tokens 4928 (thinking 4919); B: extracted '} > ]' score 1 tokens 3344 (thinking 3333)
- bbh-00022:dyck_languages: gold '> )'; A2: extracted '> )' score 1 tokens 10473 (thinking 10464); B: extracted '> ) )' score 0 tokens 14714 (thinking 14703)
- bbh-00133:tracking_shuffled_objects_seven_objects: gold '(B)'; A2: extracted '(B)' score 1 tokens 1742 (thinking 1732); B: extracted '(G)' score 0 tokens 1910 (thinking 1899)
### largest length differences A1 vs A2
- bbh-00022:dyck_languages: A1 5459 tok (score 1), A2 10473 tok (score 1)
- bbh-00014:dyck_languages: A1 9801 tok (score 0), A2 5258 tok (score 1)
- bbh-00002:dyck_languages: A1 6962 tok (score 1), A2 11111 tok (score 0)
- bbh-00086:logical_deduction_seven_objects: A1 2843 tok (score 1), A2 5889 tok (score 1)
- bbh-00012:dyck_languages: A1 8784 tok (score 1), A2 6212 tok (score 1)
### largest length differences A1 vs B
- bbh-00022:dyck_languages: A1 5459 tok (score 1), B 14714 tok (score 0)
- bbh-00014:dyck_languages: A1 9801 tok (score 0), B 6600 tok (score 1)
- bbh-00013:dyck_languages: A1 14103 tok (score 1), B 16384 tok (score 0)
- bbh-00012:dyck_languages: A1 8784 tok (score 1), B 6559 tok (score 1)
- bbh-00076:logical_deduction_seven_objects: A1 3159 tok (score 1), B 1680 tok (score 1)
### largest length differences A2 vs B
- bbh-00022:dyck_languages: A2 10473 tok (score 1), B 14714 tok (score 0)
- bbh-00086:logical_deduction_seven_objects: A2 5889 tok (score 1), B 2677 tok (score 1)
- bbh-00002:dyck_languages: A2 11111 tok (score 0), B 8340 tok (score 1)
- bbh-00013:dyck_languages: A2 14082 tok (score 1), B 16384 tok (score 0)
- bbh-00144:tracking_shuffled_objects_seven_objects: A2 3009 tok (score 1), B 1247 tok (score 1)
