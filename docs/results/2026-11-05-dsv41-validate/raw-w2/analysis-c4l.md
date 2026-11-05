# check 4 (temperature 0.7, seeds fixed per request): default against off

| kind | arm | n | correct | mean tokens | mean repeated-4-gram fraction | max token run | degenerate | finish=length |
|---|---|---|---|---|---|---|---|---|
| freelong | default | 16 | - | 237.1 | 0.005 | 1 | 0 | 7 |
| freelong | off | 16 | - | 232.6 | 0.007 | 1 | 0 | 3 |

prefix-cache hits in either arm (must be none): []

identical replies (same seed) 0 of 16; correct 0 vs 0; free-form length ratio 1.02, repeated-4-gram 0.005 vs 0.007

PASS
