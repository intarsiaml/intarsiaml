clock filter: rounds with min GPU clock >= 1450 MHz (macmon 100 ms)

| case | a8 | rounds valid/total | clock mean (MHz) over valid rounds |
|---|---|---|---|
| down | 0 | 34/60 | 1586 |
| down | 1 | 27/30 | 1596 |
| gu | 0 | 34/60 | 1573 |
| gu | 1 | 25/30 | 1584 |
| kv | 0 | 51/60 | 1607 |
| kv | 1 | 27/30 | 1611 |
| qo | 0 | 48/60 | 1592 |
| qo | 1 | 27/30 | 1601 |

### qo (16-byte-aligned weights), valid rounds 48

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 58.5 | 58.4 (cfg 1) | 1.002 [0.986-1.016] | - | - |  |
| 2 | 58.5 | 58.5 (cfg 1) | 1.002 [0.981-1.021] | 407.2 | 6.97 | tiled GEMM |
| 4 | 58.5 | 59.5 (cfg 1) | 1.017 [0.995-1.038] | 411.1 | 7.05 | tiled GEMM |
| 6 | 58.5 | 61.1 (cfg 1) | 1.042 [1.021-1.081] | 424.0 | 7.25 | tiled GEMM |
| 8 | 58.5 | 64.3 (cfg 1) | 1.097 [1.068-1.125] | 430.5 | 7.36 | tiled GEMM |
| 16 | 58.5 | - | - | 114.4 | 1.96 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.00, P2=1.02, P4=1.09; R=3: ; R=4: P1=1.02, P2=1.07, P4=1.15; R=5: ; R=6: P1=1.04, P1s2=1.19, P1s3=1.15, P2=1.16, P4=1.66; R=7: ; R=8: P1=1.10, P1s2=1.50, P1s4=1.38, P2=1.46, P2s4=1.76, P4=2.52; R=16: 

other tiled-GEMM row counts: M=15: 478.5 us (8.17x)

### down (16-byte-aligned weights), valid rounds 34

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 171.6 | 171.5 (cfg 1) | 1.003 [0.991-1.019] | - | - |  |
| 2 | 171.6 | 173.1 (cfg 1) | 1.001 [0.982-1.030] | 1246.3 | 7.26 | tiled GEMM |
| 4 | 171.6 | 172.5 (cfg 1) | 1.003 [0.977-1.020] | 1262.7 | 7.35 | tiled GEMM |
| 6 | 171.6 | 174.0 (cfg 1) | 1.007 [0.987-1.036] | 1329.5 | 7.74 | tiled GEMM |
| 8 | 171.6 | 205.0 (cfg 1) | 1.190 [1.157-1.219] | 1355.2 | 7.89 | tiled GEMM |
| 16 | 171.6 | - | - | 335.8 | 1.95 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.00, P2=1.02, P4=1.03; R=3: ; R=4: P1=1.00, P2=1.02, P4=1.11; R=5: ; R=6: P1=1.01, P1s2=1.23, P1s3=1.13, P2=1.06, P4=1.53; R=7: ; R=8: P1=1.19, P1s2=1.59, P1s4=1.49, P2=1.35, P2s4=1.50, P4=2.39; R=16: 

other tiled-GEMM row counts: M=15: 1525.5 us (8.87x)

### kv (16-byte-aligned weights), valid rounds 51

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 30.2 | 30.4 (cfg 1) | 1.006 [0.986-1.022] | - | - |  |
| 2 | 30.2 | 30.3 (cfg 1) | 1.004 [0.972-1.016] | 767.9 | 25.41 | tiled GEMM |
| 4 | 30.2 | 30.8 (cfg 1) | 1.017 [0.980-1.038] | 775.0 | 25.61 | tiled GEMM |
| 6 | 30.2 | 32.4 (cfg 1) | 1.073 [1.019-1.108] | 797.9 | 26.39 | tiled GEMM |
| 8 | 30.2 | 35.3 (cfg 1) | 1.174 [1.107-1.197] | 804.4 | 26.62 | tiled GEMM |
| 16 | 30.2 | - | - | 79.9 | 2.63 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.00, P2=1.04, P4=1.11; R=3: ; R=4: P1=1.02, P2=1.14, P4=1.34; R=5: ; R=6: P1=1.07, P1s2=1.27, P1s3=1.24, P2=1.28, P4=1.83; R=7: ; R=8: P1=1.17, P1s2=1.57, P1s4=1.51, P2=1.59, P2s4=1.91, P4=2.73; R=16: 

other tiled-GEMM row counts: M=15: 872.2 us (28.91x)

### gu (16-byte-aligned weights), valid rounds 34

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 346.6 | 347.9 (cfg 1) | 1.002 [0.989-1.017] | - | - |  |
| 2 | 346.6 | 347.0 (cfg 1) | 0.997 [0.985-1.016] | 1705.1 | 4.91 | tiled GEMM |
| 4 | 346.6 | 349.0 (cfg 1) | 1.007 [0.979-1.032] | 1721.1 | 4.97 | tiled GEMM |
| 6 | 346.6 | 351.1 (cfg 1) | 1.015 [0.992-1.040] | 1790.6 | 5.16 | tiled GEMM |
| 8 | 346.6 | 397.4 (cfg 1) | 1.149 [1.117-1.172] | 1809.0 | 5.21 | tiled GEMM |
| 16 | 346.6 | - | - | 546.2 | 1.58 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.00, P2=1.00, P4=1.02; R=3: ; R=4: P1=1.01, P2=1.01, P4=1.03; R=5: ; R=6: P1=1.02, P1s2=1.33, P1s3=1.20, P2=1.05, P4=1.43; R=7: ; R=8: P1=1.15, P1s2=1.90, P1s4=1.54, P2=1.34, P2s4=1.82, P4=2.31; R=16: 

other tiled-GEMM row counts: M=15: 2021.6 us (5.82x)

### qo (8-byte-aligned weights), valid rounds 27

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 57.7 | 57.9 (cfg 1) | 1.003 [0.984-1.024] | - | - |  |
| 2 | 57.7 | 57.8 (cfg 1) | 1.003 [0.972-1.021] | 406.4 | 7.05 | tiled GEMM |
| 4 | 57.7 | 59.0 (cfg 1) | 1.021 [0.992-1.034] | 410.9 | 7.12 | tiled GEMM |
| 6 | 57.7 | 62.7 (cfg 1) | 1.087 [1.055-1.110] | 423.0 | 7.33 | tiled GEMM |
| 8 | 57.7 | 75.6 (cfg 1) | 1.312 [1.266-1.323] | 428.4 | 7.43 | tiled GEMM |
| 16 | 57.7 | - | - | 112.6 | 1.95 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.00, P2=1.02, P4=1.05; R=3: ; R=4: P1=1.02, P2=1.03, P4=1.11; R=5: ; R=6: P1=1.09, P1s2=1.21, P1s3=1.19, P2=1.13, P4=1.61; R=7: ; R=8: P1=1.31, P1s2=1.55, P1s4=1.48, P2=1.45, P2s4=1.76, P4=2.50; R=16: 

other tiled-GEMM row counts: M=15: 476.6 us (8.27x)

### down (8-byte-aligned weights), valid rounds 27

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 170.2 | 170.8 (cfg 1) | 1.003 [0.991-1.027] | - | - |  |
| 2 | 170.2 | 170.7 (cfg 1) | 1.002 [0.985-1.019] | 1245.4 | 7.32 | tiled GEMM |
| 4 | 170.2 | 170.8 (cfg 1) | 1.008 [0.993-1.021] | 1262.4 | 7.42 | tiled GEMM |
| 6 | 170.2 | 175.6 (cfg 1) | 1.029 [1.014-1.055] | 1331.6 | 7.83 | tiled GEMM |
| 8 | 170.2 | 229.7 (cfg 1) | 1.355 [1.323-1.380] | 1356.1 | 7.97 | tiled GEMM |
| 16 | 170.2 | - | - | 331.6 | 1.95 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.00, P2=1.01, P4=1.03; R=3: ; R=4: P1=1.01, P2=1.03, P4=1.11; R=5: ; R=6: P1=1.03, P1s2=1.29, P1s3=1.16, P2=1.06, P4=1.53; R=7: ; R=8: P1=1.36, P1s2=1.65, P1s4=1.53, P2=1.37, P2s4=1.52, P4=2.39; R=16: 

other tiled-GEMM row counts: M=15: 1525.4 us (8.96x)

### kv (8-byte-aligned weights), valid rounds 27

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 30.0 | 29.9 (cfg 1) | 1.005 [0.995-1.017] | - | - |  |
| 2 | 30.0 | 30.2 (cfg 1) | 1.012 [0.992-1.053] | 765.1 | 25.55 | tiled GEMM |
| 4 | 30.0 | 30.7 (cfg 1) | 1.030 [1.002-1.053] | 772.3 | 25.78 | tiled GEMM |
| 6 | 30.0 | 33.6 (cfg 1) | 1.130 [1.102-1.158] | 795.0 | 26.58 | tiled GEMM |
| 8 | 30.0 | 42.4 (cfg 1) | 1.421 [1.382-1.453] | 801.2 | 26.77 | tiled GEMM |
| 16 | 30.0 | - | - | 78.3 | 2.63 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.01, P2=1.04, P4=1.08; R=3: ; R=4: P1=1.03, P2=1.06, P4=1.23; R=5: ; R=6: P1=1.13, P1s2=1.28, P1s3=1.25, P2=1.19, P4=1.77; R=7: ; R=8: P1=1.42, P1s2=1.60, P1s4=1.64, P2=1.54, P2s4=1.84, P4=2.66; R=16: 

other tiled-GEMM row counts: M=15: 870.4 us (29.10x)

### gu (8-byte-aligned weights), valid rounds 25

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 1 | 338.1 | 339.0 (cfg 1) | 1.003 [0.992-1.010] | - | - |  |
| 2 | 338.1 | 338.5 (cfg 1) | 1.002 [0.990-1.012] | 1674.7 | 4.96 | tiled GEMM |
| 4 | 338.1 | 339.7 (cfg 2) | 1.006 [0.994-1.036] | 1694.4 | 5.01 | tiled GEMM |
| 6 | 338.1 | 361.9 (cfg 2) | 1.071 [1.051-1.098] | 1772.4 | 5.24 | tiled GEMM |
| 8 | 338.1 | 480.4 (cfg 2) | 1.423 [1.383-1.440] | 1798.2 | 5.31 | tiled GEMM |
| 16 | 338.1 | - | - | 537.0 | 1.59 | NAX GEMM |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: P1=1.00, P2=1.00, P4=1.01; R=3: ; R=4: P1=1.01, P2=1.01, P4=1.02; R=5: ; R=6: P1=1.08, P1s2=1.39, P1s3=1.26, P2=1.07, P4=1.46; R=7: ; R=8: P1=1.42, P1s2=2.00, P1s4=1.65, P2=1.42, P2s4=1.86, P4=2.36; R=16: 

other tiled-GEMM row counts: M=15: 2004.9 us (5.93x)

### Layer sum (2 x q/o + down + k/v pair + gate/up pair), GEMV time only

| R | alignment | T_layer(1) serving (us) | T_layer(R) multi-row (us) | ratio | T_layer(R) existing path (us) | existing ratio |
|---|---|---|---|---|---|---|
| 2 | 16-byte | 665.5 | 667.4 | 1.003 | 4533.7 | 6.81 |
| 4 | 16-byte | 665.5 | 671.3 | 1.009 | 4581.0 | 6.88 |
| 6 | 16-byte | 665.5 | 679.7 | 1.021 | 4765.9 | 7.16 |
| 8 | 16-byte | 665.5 | 766.3 | 1.152 | 4829.5 | 7.26 |
| 2 | 8-byte | 653.6 | 655.0 | 1.002 | 4497.9 | 6.88 |
| 4 | 8-byte | 653.6 | 659.1 | 1.009 | 4550.7 | 6.96 |
| 6 | 8-byte | 653.6 | 696.5 | 1.066 | 4745.1 | 7.26 |
| 8 | 8-byte | 653.6 | 903.8 | 1.383 | 4812.3 | 7.36 |

### Census-weighted (Qwen3-8B: 7-8 of 36 layers per shape sit in the 8-mod-16 shard), per layer, GEMV time

| R | T_layer(1) (us) | T_layer(R) multi-row | ratio | existing | existing ratio |
|---|---|---|---|---|---|
| 2 | 663.1 | 664.9 | 1.003 | 4526.6 | 6.83 |
| 4 | 663.1 | 668.9 | 1.009 | 4575.0 | 6.90 |
| 6 | 663.1 | 683.0 | 1.030 | 4761.7 | 7.18 |
| 8 | 663.1 | 793.6 | 1.197 | 4826.0 | 7.28 |

### Per verify step, GEMVs only: 36 x layer + lm_head (all R rows through the head), 16-byte-aligned weights

| R | step(1) (ms) | step(R) multi-row (ms) | step(R)/step(1) | step(R) existing (ms) | existing/step(1) | per-token cost multi-row (ms/row) |
|---|---|---|---|---|---|---|

### Per verify step, GEMVs only, census-weighted layers (16-byte / 8-mod-16 mix of the real shards) + lm_head

| R | step(1) (ms) | step(R) multi-row (ms) | step(R)/step(1) |
|---|---|---|---|

bitwise checks in the bench: True (264 checks)
