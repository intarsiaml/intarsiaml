# check 4 (temperature 0.7, seeds fixed per request): default against off

| kind | arm | n | correct | mean tokens | mean repeated-4-gram fraction | max token run | degenerate | finish=length |
|---|---|---|---|---|---|---|---|---|
| answer | default | 16 | 16 | 6.1 | 0.000 | 1 | 0 | 0 |
| answer | off | 16 | 16 | 6.1 | 0.000 | 1 | 0 | 0 |
| free | default | 8 | - | 221.8 | 0.002 | 1 | 0 | 0 |
| free | off | 8 | - | 221.8 | 0.002 | 1 | 0 | 0 |

prefix-cache hits in either arm (must be none): []

identical replies (same seed) 24 of 24; correct 16 vs 16; free-form length ratio 1.00, repeated-4-gram 0.002 vs 0.002

PASS
