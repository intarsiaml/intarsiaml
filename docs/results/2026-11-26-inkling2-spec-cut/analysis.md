| class | pairs | tok/s A (gate 0.8 k3) | tok/s B (cut 0.5 k3) | B/A median [95% CI] | B/A geomean [95% CI] | cluster-bootstrap geomean CI | pairs > 1.0 | prompts >= +2% (median per prompt) | prompts below 0.99 | min per-prompt median |
|---|---:|---:|---:|---|---|---|---:|---|---|---|
| short | 300 | 84.8 | 83.9 | 1.007 [0.997-1.014] | 1.006 [1.000-1.012] | [1.003-1.009] | 163/300 | 6/15 | 7/15 | 0.928 |
| len1k | 120 | 82.1 | 86.1 | 1.050 [1.036-1.058] | 1.042 [1.035-1.050] | [1.038-1.047] | 95/120 | 2/3 | 0/3 | 0.993 |
| len4k | 84 | 74.6 | 83.2 | 1.116 [1.105-1.135] | 1.119 [1.110-1.128] | [1.115-1.123] | 84/84 | 3/3 | 0/3 | 1.074 |
| len16k | 48 | 89.7 | 94.2 | 1.042 [1.028-1.053] | 1.032 [1.022-1.041] | [1.029-1.035] | 40/48 | 2/3 | 0/3 | 0.999 |

ids of 1200 requests (both arms, all cycles incl. warm-up) vs --spec off: 1200 identical, 0 differ

len16k c5: verify passes 2286, proposed 5184, accepted 4248 (82%), 2.86 tok/pass
len16k g3: verify passes 1998, proposed 4374, accepted 3942 (90%), 2.97 tok/pass
len1k c5: verify passes 5082, proposed 11130, accepted 7980 (72%), 2.57 tok/pass
len1k g3: verify passes 4032, proposed 8652, accepted 6930 (80%), 2.72 tok/pass
len4k c5: verify passes 3540, proposed 7050, accepted 5730 (81%), 2.62 tok/pass
len4k g3: verify passes 2430, proposed 4710, accepted 4320 (92%), 2.78 tok/pass
short c5: verify passes 16236, proposed 37092, accepted 21406 (58%), 2.32 tok/pass
short g3: verify passes 12694, proposed 27192, accepted 19382 (71%), 2.53 tok/pass
