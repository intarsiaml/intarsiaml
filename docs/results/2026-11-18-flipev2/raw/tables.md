## cell 1: 16 requests on a shared 16448-token prefix, `--max-batch 16` (mean over cells, range)

| herd | cells | TTFT sum s | TTFT p50 s | TTFT last s | wall s |
|---|---|---|---|---|---|
| off (N) | 6 | 1497.87 (1489.58 to 1516.10) | 93.72 (93.10 to 94.97) | 175.43 (174.55 to 176.68) | 176.52 (175.62 to 177.66) |
| on (H) | 6 | 247.15 (246.87 to 247.51) | 15.34 (15.32 to 15.37) | 20.08 (20.04 to 20.10) | 27.91 (27.82 to 28.00) |

per seed, TTFT sum s (off / on): 121: 1490.1 / 247.1; 122: 1508.7 / 247.2; 123: 1494.8 / 247.2

## cell 2: 8 requests on a shared 16448-token prefix + one unrelated 1000-token prompt at +2 s, `--max-batch 16` (mean over cells, range)

| arm | cells | herd-8 TTFT sum s | herd-8 TTFT last s | short TTFT s | short end s (abs) | wall s | cached tokens of short |
|---|---|---|---|---|---|---|---|
| neither | 6 | 398.97 (397.38 to 400.31) | 88.21 (87.87 to 88.58) | 86.96 (86.61 to 87.35) | 90.62 (90.16 to 91.03) | 90.62 (90.16 to 91.03) | [0] |
| herd | 6 | 104.88 (104.42 to 105.64) | 15.11 (15.04 to 15.23) | 13.98 (13.89 to 14.12) | 21.30 (21.18 to 21.40) | 21.30 (21.18 to 21.40) | [0] |
| sjf | 6 | 413.29 (409.58 to 420.14) | 89.92 (89.38 to 90.94) | 2.34 (2.32 to 2.37) | 20.53 (19.29 to 21.22) | 91.11 (90.43 to 92.42) | [0] |
| herd + sjf | 6 | 114.19 (113.69 to 115.30) | 16.29 (16.21 to 16.46) | 2.34 (2.31 to 2.39) | 18.07 (17.88 to 18.45) | 21.60 (21.36 to 21.99) | [0] |

## cell 4: 4 requests on a 16448-token prefix + 2 unrelated 8192-token prompts, released together (thread order S U S S U S), `--max-batch 16`

per cell: TTFT of the two unrelated prompts (index 1 and index 4)

| arm | visit | seed | U1 (idx 1) s | U2 (idx 4) s | pair sum s | slower of two s | shared four (mean) s |
|---|---|---|---|---|---|---|---|
| H | H2 | 131 | 10.30 | 4.95 | 15.25 | 10.30 | 24.64 |
| H | H2 | 132 | 23.90 | 4.96 | 28.86 | 23.90 | 23.11 |
| H | H2 | 133 | 18.05 | 24.61 | 42.66 | 24.61 | 13.40 |
| H | H2 | 134 | 4.97 | 24.33 | 29.30 | 24.33 | 21.39 |
| H | H7 | 131 | 4.94 | 23.91 | 28.85 | 23.91 | 23.10 |
| H | H7 | 132 | 16.72 | 22.52 | 39.24 | 22.52 | 20.49 |
| H | H7 | 133 | 17.28 | 23.70 | 40.97 | 23.70 | 16.19 |
| H | H7 | 134 | 10.28 | 4.95 | 15.24 | 10.28 | 24.87 |
| N | N1 | 131 | 4.96 | 10.30 | 15.27 | 10.30 | 42.11 |
| N | N1 | 132 | 4.97 | 23.55 | 28.52 | 23.55 | 40.47 |
| N | N1 | 133 | 4.97 | 48.22 | 53.19 | 48.22 | 37.58 |
| N | N1 | 134 | 35.75 | 4.97 | 40.73 | 35.75 | 39.08 |
| N | N8 | 131 | 27.67 | 34.21 | 61.89 | 34.21 | 33.69 |
| N | N8 | 132 | 48.19 | 4.97 | 53.15 | 48.19 | 37.68 |
| N | N8 | 133 | 57.66 | 29.90 | 87.56 | 57.66 | 31.73 |
| N | N8 | 134 | 4.97 | 36.04 | 41.02 | 36.04 | 39.13 |

summary (mean +- sd over cells)

| arm | cells | U1 s | U2 s | pair sum s | slower of two s | faster of two s | shared four mean s |
|---|---|---|---|---|---|---|---|
| neither | 8 | 23.64 +- 21.77 | 24.02 +- 15.96 | 47.66 +- 21.92 | 36.74 +- 15.03 | 10.92 +- 11.04 | 37.68 +- 3.44 |
| herd | 8 | 13.31 +- 6.76 | 16.74 +- 9.78 | 30.05 +- 10.72 | 20.44 +- 6.30 | 9.60 +- 6.43 | 20.90 +- 4.11 |

per visit, pair sum s (mean over seeds): H/H2: 29.02; H/H7: 31.08; N/N1: 34.42; N/N8: 60.90

## ids: every reply of every concurrent cell against the solo reply of the same prompt set (first visit that ran it)

- herd8 / herd / arm H: 96 / 96 equal
- herd8 / herd / arm N: 96 / 96 equal
- hs / hs / arm B: 54 / 54 equal
- hs / hs / arm H: 54 / 54 equal
- hs / hs / arm N: 54 / 54 equal
- hs / hs / arm S: 54 / 54 equal
- mix / mix / arm H: 48 / 48 equal
- mix / mix / arm N: 48 / 48 equal
- solo / solo:herd / arm H: 96 / 96 equal
- solo / solo:herd / arm N: 96 / 96 equal
- solo / solo:hs / arm B: 54 / 54 equal
- solo / solo:hs / arm H: 54 / 54 equal
- solo / solo:hs / arm N: 54 / 54 equal
- solo / solo:hs / arm S: 54 / 54 equal
- solo / solo:mix / arm H: 48 / 48 equal
- solo / solo:mix / arm N: 48 / 48 equal
- errors: 0; reply lengths (tokens: replies): {48: 1008}
- total: 1008 / 1008 equal

## cell 4, fixed arrival order (20 ms apart): 4 requests on a 16448-token prefix (S) + 2 unrelated 8192-token prompts (U), `--max-batch 16`

### order A: S0 U1 S2 S3 U4 S5 (flipev's order)

| request | kind | herd off TTFT s | herd on TTFT s | on - off s | off sd | on sd |
|---|---|---|---|---|---|---|
| idx 0 (arrives 1th) | S | 11.13 | 11.13 | -0.00 | 0.02 | 0.02 |
| idx 1 (arrives 2th) | U | 16.79 | 16.82 | +0.03 | 0.05 | 0.09 |
| idx 2 (arrives 3th) | S | 28.65 | 17.34 | -11.31 | 0.07 | 0.09 |
| idx 3 (arrives 4th) | S | 39.55 | 17.87 | -21.68 | 0.08 | 0.09 |
| idx 4 (arrives 5th) | U | 45.33 | 23.73 | -21.60 | 0.19 | 0.08 |
| idx 5 (arrives 6th) | S | 56.49 | 24.27 | -32.22 | 0.10 | 0.08 |

- pair sum of the two unrelated: off 62.12 +- 0.23 (n 6), on 40.54 +- 0.16 (n 6), on - off -21.57

- slower of the two unrelated: off 45.33 +- 0.19 (n 6), on 23.73 +- 0.08 (n 6), on - off -21.60

- faster of the two unrelated: off 16.79 +- 0.05 (n 6), on 16.82 +- 0.09 (n 6), on - off +0.03

  per visit, U pair sum / slower / faster (s): H/H2: 40.66 / 23.78 / 16.87; H/H3: 40.43 / 23.67 / 16.76; N/N1: 62.18 / 45.39 / 16.78; N/N4: 62.06 / 45.27 / 16.79

  per cell (seed, arm, visit): U1 / U4 TTFT: 141 H/H2: 16.91 / 23.84; 141 H/H3: 16.78 / 23.74; 141 N/N1: 16.84 / 45.49; 141 N/N4: 16.77 / 45.35; 142 H/H2: 16.91 / 23.79; 142 H/H3: 16.84 / 23.63; 142 N/N1: 16.70 / 45.14; 142 N/N4: 16.77 / 45.08; 143 H/H2: 16.80 / 23.72; 143 H/H3: 16.68 / 23.64; 143 N/N1: 16.81 / 45.55; 143 N/N4: 16.82 / 45.37

### order B: U1 U4 S0 S2 S3 S5 (the unrelated pair first)

| request | kind | herd off TTFT s | herd on TTFT s | on - off s | off sd | on sd |
|---|---|---|---|---|---|---|
| idx 0 (arrives 3th) | S | 23.69 | 23.99 | +0.30 | 0.28 | 0.20 |
| idx 1 (arrives 1th) | U | 4.95 | 4.95 | +0.00 | 0.01 | 0.01 |
| idx 2 (arrives 4th) | S | 35.87 | 24.54 | -11.33 | 0.34 | 0.21 |
| idx 3 (arrives 5th) | S | 47.98 | 25.10 | -22.88 | 0.33 | 0.22 |
| idx 4 (arrives 2th) | U | 10.27 | 10.28 | +0.01 | 0.02 | 0.02 |
| idx 5 (arrives 6th) | S | 60.06 | 25.67 | -34.39 | 0.35 | 0.22 |

- pair sum of the two unrelated: off 15.22 +- 0.03 (n 6), on 15.23 +- 0.03 (n 6), on - off +0.01

- slower of the two unrelated: off 10.27 +- 0.02 (n 6), on 10.28 +- 0.02 (n 6), on - off +0.01

- faster of the two unrelated: off 4.95 +- 0.01 (n 6), on 4.95 +- 0.01 (n 6), on - off +0.00

  per visit, U pair sum / slower / faster (s): H/H2: 15.23 / 10.28 / 4.96; H/H3: 15.23 / 10.27 / 4.95; N/N1: 15.21 / 10.26 / 4.95; N/N4: 15.23 / 10.27 / 4.96

  per cell (seed, arm, visit): U1 / U4 TTFT: 151 H/H2: 4.96 / 10.28; 151 H/H3: 4.93 / 10.24; 151 N/N1: 4.96 / 10.28; 151 N/N4: 4.95 / 10.25; 152 H/H2: 4.95 / 10.26; 152 H/H3: 4.96 / 10.29; 152 N/N1: 4.94 / 10.25; 152 N/N4: 4.96 / 10.27; 153 H/H2: 4.96 / 10.29; 153 H/H3: 4.96 / 10.30; 153 N/N1: 4.95 / 10.25; 153 N/N4: 4.97 / 10.29

ids: 288 / 288 replies (concurrent and solo) equal the first solo reply of the same prompts; errors 0

## diag: 3 cold 16448-token prompts released together (FIFO prefill), TTFT per prompt s; the step between sorted TTFTs is one prompt's prefill + first token

| config | seed | TTFTs sorted s | steps s |
|---|---|---|---|
| mb16-store | 701 | 10.94, 23.36, 36.29 | 10.94, 12.42, 12.92 |
| mb16-store | 702 | 10.88, 23.52, 36.73 | 10.88, 12.64, 13.21 |
| mb8-store | 701 | 10.94, 23.46, 36.32 | 10.94, 12.52, 12.86 |
| mb8-store | 702 | 10.89, 23.66, 37.17 | 10.89, 12.77, 13.51 |
| mb16 | 701 | 10.58, 22.44, 34.31 | 10.58, 11.86, 11.87 |
| mb16 | 702 | 10.53, 22.37, 34.30 | 10.53, 11.84, 11.94 |
| mb1 | 701 | 9.91, 20.84, 31.59 | 9.91, 10.92, 10.76 |
| mb1 | 702 | 9.83, 20.95, 31.68 | 9.83, 11.13, 10.73 |

## cell 3 (AGE sensitivity, 2 decoders + 40K + 2 shorts, `--max-batch 4`): ITL of the ongoing decodes while the 40K prompt prefills (event gaps in the window, pooled over runs)

runs dropped: 0 with an error, 0 void (a decoder finished before the long prompt's first token)

| decoders | scenario | arm | runs | gaps n | mean ms | p99 ms | max ms | gaps >200 ms | long TTFT s | short TTFT s (s1 / s2) | dec tokens in window | dec tok/s/stream | window s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | longshort | fifo | 6 | 1235 | 326.6 | 793.8 | 813.3 | 701 | 33.97 (33.91 to 34.09) | 33.25 / 32.16 | 206 | 3.0 | 33.97 |
| 2 | longshort | sjf age 4 | 6 | 1112 | 381.1 | 808.7 | 817.2 | 752 | 35.48 (35.42 to 35.58) | 2.03 / 7.61 | 185 | 2.6 | 35.48 |
| 2 | longshort | sjf age 8 | 6 | 1097 | 383.8 | 808.5 | 813.6 | 749 | 35.44 (35.36 to 35.51) | 2.03 / 7.62 | 183 | 2.6 | 35.44 |
| 2 | longshort | sjf age 16 | 6 | 1100 | 385.0 | 806.5 | 816.7 | 752 | 35.47 (35.39 to 35.57) | 2.01 / 7.60 | 183 | 2.6 | 35.47 |

per-run (arm visit rep): long TTFT / short TTFTs / dec tokens in window / max window gap ms

- d2 longshort 16 164 rep1: 35.574 / 2.021 7.603 / 181 / 817
- d2 longshort 16 164 rep2: 35.561 / 2.02 7.62 / 188 / 813
- d2 longshort 16 164 rep3: 35.445 / 2.014 7.614 / 182 / 816
- d2 longshort 16 165 rep1: 35.421 / 2.03 7.607 / 181 / 810
- d2 longshort 16 165 rep2: 35.435 / 1.985 7.561 / 181 / 816
- d2 longshort 16 165 rep3: 35.389 / 2.017 7.585 / 187 / 807
- d2 longshort 4 42 rep1: 35.464 / 2.054 7.63 / 188 / 809
- d2 longshort 4 42 rep2: 35.432 / 2.014 7.587 / 188 / 809
- d2 longshort 4 42 rep3: 35.42 / 1.99 7.541 / 180 / 814
- d2 longshort 4 47 rep1: 35.578 / 2.071 7.674 / 188 / 814
- d2 longshort 4 47 rep2: 35.533 / 2.008 7.624 / 182 / 817
- d2 longshort 4 47 rep3: 35.46 / 2.016 7.587 / 186 / 812
- d2 longshort 8 83 rep1: 35.442 / 2.044 7.626 / 181 / 809
- d2 longshort 8 83 rep2: 35.362 / 2.007 7.577 / 181 / 810
- d2 longshort 8 83 rep3: 35.514 / 2.032 7.634 / 186 / 810
- d2 longshort 8 86 rep1: 35.504 / 2.062 7.654 / 187 / 810
- d2 longshort 8 86 rep2: 35.436 / 2.013 7.622 / 181 / 811
- d2 longshort 8 86 rep3: 35.389 / 2.014 7.588 / 181 / 814
- d2 longshort F F1 rep1: 33.908 / 33.196 32.093 / 204 / 807
- d2 longshort F F1 rep2: 33.911 / 33.187 32.086 / 205 / 807
- d2 longshort F F1 rep3: 33.914 / 33.192 32.095 / 206 / 808
- d2 longshort F F8 rep1: 34.065 / 33.358 32.258 / 204 / 813
- d2 longshort F F8 rep2: 33.934 / 33.213 32.122 / 205 / 808
- d2 longshort F F8 rep3: 34.094 / 33.376 32.285 / 211 / 813

cell 3 (AGE sensitivity, 2 decoders + 40K + 2 shorts, `--max-batch 4`) ids: arm 16: 30 / 30 contended replies equal their own server's solo reply; arm 4: 30 / 30 contended replies equal their own server's solo reply; arm 8: 30 / 30 contended replies equal their own server's solo reply; arm F: 30 / 30 contended replies equal their own server's solo reply
distinct id sequences per prompt over all runs and arms (1 = all equal): d0: 1, d1: 1, long: 1, s1: 1, s2: 1
reply lengths: {('d', 800): 48, ('long', 16): 24, ('s1', 16): 24, ('s2', 16): 24}

## cell 5 (3 / 4 decoders, `--max-batch 8`): ITL of the ongoing decodes while the 40K prompt prefills (event gaps in the window, pooled over runs)

runs dropped: 0 with an error, 0 void (a decoder finished before the long prompt's first token)

| decoders | scenario | arm | runs | gaps n | mean ms | p99 ms | max ms | gaps >200 ms | long TTFT s | short TTFT s (s1 / s2) | dec tokens in window | dec tok/s/stream | window s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | long | fifo | 4 | 730 | 548.8 | 777.1 | 787.4 | 703 | 33.63 (33.54 to 33.79) | - | 182 | 1.8 | 33.63 |
| 3 | long | sjf (age 8) | 4 | 728 | 550.3 | 776.3 | 783.9 | 704 | 33.58 (33.49 to 33.63) | - | 182 | 1.8 | 33.58 |
| 3 | longshort | fifo | 4 | 730 | 548.8 | 778.1 | 785.4 | 703 | 33.63 (33.51 to 33.75) | 32.93 / 31.24 | 182 | 1.8 | 33.63 |
| 3 | longshort | sjf (age 8) | 4 | 783 | 538.3 | 774.0 | 780.7 | 753 | 35.25 (35.17 to 35.35) | 1.92 / 1.58 | 196 | 1.9 | 35.25 |
| 4 | long | fifo | 4 | 975 | 553.7 | 783.7 | 784.5 | 939 | 33.91 (33.82 to 33.98) | - | 244 | 1.8 | 33.91 |
| 4 | long | sjf (age 8) | 4 | 980 | 546.2 | 781.9 | 784.3 | 932 | 33.90 (33.80 to 34.05) | - | 245 | 1.8 | 33.90 |
| 4 | longshort | fifo | 3 | 736 | 550.1 | 784.5 | 786.6 | 704 | 33.92 (33.84 to 34.02) | 33.22 / 31.55 | 245 | 1.8 | 33.92 |
| 4 | longshort | sjf (age 8) | 4 | 1044 | 543.8 | 781.6 | 787.1 | 1004 | 35.62 (35.43 to 35.91) | 1.97 / 1.66 | 261 | 1.8 | 35.62 |

per-run (arm visit rep): long TTFT / short TTFTs / dec tokens in window / max window gap ms

- d3 long F F1 rep1: 33.794 / - / 184 / 787
- d3 long F F1 rep2: 33.656 / - / 182 / 777
- d3 long F F4 rep1: 33.539 / - / 181 / 774
- d3 long F F4 rep2: 33.537 / - / 183 / 779
- d3 long S S2 rep1: 33.598 / - / 181 / 779
- d3 long S S2 rep2: 33.617 / - / 183 / 784
- d3 long S S3 rep1: 33.629 / - / 181 / 777
- d3 long S S3 rep2: 33.488 / - / 183 / 774
- d3 longshort F F1 rep1: 33.748 / 33.053 31.371 / 181 / 785
- d3 longshort F F1 rep2: 33.668 / 32.954 31.255 / 186 / 779
- d3 longshort F F4 rep1: 33.513 / 32.812 31.123 / 181 / 778
- d3 longshort F F4 rep2: 33.606 / 32.895 31.197 / 182 / 778
- d3 longshort S S2 rep1: 35.22 / 1.921 1.584 / 198 / 781
- d3 longshort S S2 rep2: 35.272 / 1.914 1.577 / 193 / 775
- d3 longshort S S3 rep1: 35.354 / 1.916 1.587 / 195 / 778
- d3 longshort S S3 rep2: 35.168 / 1.914 1.559 / 197 / 774
- d4 long F F1 rep1: 33.977 / - / 242 / 784
- d4 long F F1 rep2: 33.951 / - / 244 / 784
- d4 long F F4 rep1: 33.89 / - / 246 / 784
- d4 long F F4 rep2: 33.822 / - / 243 / 784
- d4 long S S2 rep1: 33.92 / - / 247 / 784
- d4 long S S2 rep2: 34.046 / - / 244 / 784
- d4 long S S3 rep1: 33.85 / - / 245 / 782
- d4 long S S3 rep2: 33.8 / - / 244 / 782
- d4 longshort F F1 rep1: 33.837 / 33.135 31.461 / 247 / 780
- d4 longshort F F4 rep1: 33.903 / 33.202 31.53 / 248 / 785
- d4 longshort F F4 rep2: 34.02 / 33.326 31.646 / 241 / 787
- d4 longshort S S2 rep1: 35.644 / 1.961 1.652 / 263 / 785
- d4 longshort S S2 rep2: 35.906 / 2.023 1.73 / 262 / 787
- d4 longshort S S3 rep1: 35.428 / 1.944 1.627 / 259 / 778
- d4 longshort S S3 rep2: 35.517 / 1.952 1.632 / 260 / 782

cell 5 (3 / 4 decoders, `--max-batch 8`) ids: arm F: 81 / 81 contended replies equal their own server's solo reply; arm S: 88 / 88 contended replies equal their own server's solo reply
distinct id sequences per prompt over all runs and arms (1 = all equal): d0: 1, d1: 1, d2: 1, d3: 1, long: 1, s1: 1, s2: 1
reply lengths: {('d', 800): 108, ('long', 16): 31, ('s1', 16): 15, ('s2', 16): 15}

## cell 3b (AGE with a burst of 12 shorts, 2 decoders, `--max-batch 16`): ITL of the ongoing decodes while the 40K prompt prefills (event gaps in the window, pooled over runs)

runs dropped: 0 with an error, 0 void (a decoder finished before the long prompt's first token)

| decoders | scenario | arm | runs | gaps n | mean ms | p99 ms | max ms | gaps >200 ms | long TTFT s | short TTFT s (s1 / s2) | dec tokens in window | dec tok/s/stream | window s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | longburst | fifo | 4 | 824 | 330.0 | 793.0 | 815.6 | 472 | 33.98 (33.88 to 34.08) | burst mean 36.59 / max 39.14 (11 shorts; s2 37.57) | 206 | 3.0 | 33.98 |
| 2 | longburst | sjf age 2 | 4 | 795 | 422.3 | 1772.5 | 1792.5 | 517 | 42.19 (42.10 to 42.26) | burst mean 6.02 / max 9.35 (11 shorts; s2 7.76) | 199 | 2.4 | 42.19 |
| 2 | longburst | sjf age 8 | 4 | 806 | 416.8 | 1779.1 | 1788.6 | 518 | 42.14 (42.05 to 42.19) | burst mean 5.33 / max 7.86 (11 shorts; s2 6.27) | 202 | 2.4 | 42.14 |
| 2 | longburst | sjf age 32 | 4 | 807 | 417.6 | 1775.4 | 1801.9 | 519 | 42.20 (42.04 to 42.36) | burst mean 5.34 / max 7.88 (11 shorts; s2 6.28) | 202 | 2.4 | 42.20 |

per-run (arm visit rep): long TTFT / short TTFTs / dec tokens in window / max window gap ms

- d2 longburst 2 22 rep1: 42.263 / 3.024 7.8 8.042 3.023 8.045 9.371 4.796 3.025 4.795 4.79 9.373 8.043 / 197 / 1792
- d2 longburst 2 22 rep2: 42.156 / 9.318 7.718 9.317 4.756 4.754 7.999 2.996 8.004 4.753 2.995 8.002 2.998 / 204 / 1783
- d2 longburst 2 27 rep1: 42.236 / 8.089 7.827 4.833 8.086 9.417 8.084 3.062 9.409 3.057 3.062 4.829 4.832 / 196 / 1791
- d2 longburst 2 27 rep2: 42.102 / 7.996 7.712 4.748 4.748 4.754 7.994 7.995 2.99 9.307 2.992 2.993 9.31 / 198 / 1784
- d2 longburst 32 324 rep1: 42.252 / 4.8 6.339 6.6 6.598 7.92 6.593 3.026 4.802 3.026 3.022 4.801 7.924 / 202 / 1796
- d2 longburst 32 324 rep2: 42.15 / 4.742 6.229 6.516 7.835 4.738 6.522 7.834 2.987 6.521 2.986 4.74 2.984 / 201 / 1777
- d2 longburst 32 325 rep1: 42.356 / 4.821 6.358 3.048 3.049 7.948 6.62 6.619 6.624 4.822 4.817 7.948 3.051 / 202 / 1802
- d2 longburst 32 325 rep2: 42.042 / 2.964 6.202 4.717 6.491 7.802 4.722 6.49 6.495 2.964 4.718 2.964 7.802 / 202 / 1775
- d2 longburst 8 83 rep1: 42.126 / 6.552 6.291 7.873 4.781 3.015 7.875 4.779 3.015 3.011 6.557 4.777 6.559 / 202 / 1779
- d2 longburst 8 83 rep2: 42.052 / 4.742 6.223 2.985 4.74 6.52 7.823 2.986 6.518 7.826 2.988 6.52 4.739 / 201 / 1779
- d2 longburst 8 86 rep1: 42.173 / 7.877 6.286 2.989 2.988 6.561 6.557 7.876 4.77 2.989 4.772 6.561 4.772 / 202 / 1789
- d2 longburst 8 86 rep2: 42.19 / 4.77 6.266 6.55 7.864 3.007 4.766 6.556 7.867 3.005 3.006 6.552 4.767 / 201 / 1785
- d2 longburst F F1 rep1: 33.88 / 37.717 37.481 35.912 34.14 35.912 37.717 34.141 35.917 39.039 39.043 34.141 37.714 / 206 / 807
- d2 longburst F F1 rep2: 33.907 / 34.162 37.463 34.16 35.929 34.164 39.036 37.713 37.707 35.927 35.928 39.038 37.711 / 206 / 809
- d2 longburst F F8 rep1: 34.077 / 37.916 37.692 34.34 39.253 34.34 36.122 36.118 36.118 37.914 37.918 34.339 39.25 / 206 / 810
- d2 longburst F F8 rep2: 34.059 / 34.321 37.65 37.892 37.897 39.217 34.321 36.102 36.104 34.323 39.217 37.891 36.104 / 206 / 816

cell 3b (AGE with a burst of 12 shorts, 2 decoders, `--max-batch 16`) ids: arm 2: 60 / 60 contended replies equal their own server's solo reply; arm 32: 60 / 60 contended replies equal their own server's solo reply; arm 8: 60 / 60 contended replies equal their own server's solo reply; arm F: 60 / 60 contended replies equal their own server's solo reply
distinct id sequences per prompt over all runs and arms (1 = all equal): d0: 1, d1: 1, long: 1, s1: 1, s10: 1, s11: 1, s12: 1, s2: 1, s3: 1, s4: 1, s5: 1, s6: 1, s7: 1, s8: 1, s9: 1
reply lengths: {('d', 800): 32, ('long', 16): 16, ('s1', 16): 16, ('s2', 16): 16, ('s3', 16): 16, ('s4', 16): 16, ('s5', 16): 16, ('s6', 16): 16, ('s7', 16): 16, ('s8', 16): 16, ('s9', 16): 16, ('s10', 16): 16, ('s11', 16): 16, ('s12', 16): 16}
