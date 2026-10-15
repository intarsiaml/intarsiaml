clock filter: rounds with min GPU clock >= 1450 MHz (macmon 100 ms)

| case | a8 | rounds valid/total | clock mean (MHz) over valid rounds |
|---|---|---|---|
| down | 0 | 55/60 | 1584 |
| down | 1 | 29/30 | 1524 |
| gu | 0 | 54/60 | 1589 |
| gu | 1 | 13/30 | 1591 |
| kv | 0 | 58/60 | 1615 |
| kv | 1 | 26/30 | 1602 |
| lm | 0 | 59/60 | 1596 |
| qo | 0 | 53/60 | 1600 |
| qo | 1 | 26/30 | 1568 |

### qo (16-byte-aligned weights), valid rounds 53

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 57.9 | 58.7 (cfg 1) | 1.014 [1.003-1.047] | - | - |  |
| 5 | 57.9 | 59.3 (cfg 1) | 1.025 [1.012-1.039] | - | - |  |
| 6 | 57.9 | 60.4 (cfg 1) | 1.045 [1.029-1.057] | - | - |  |
| 7 | 57.9 | 61.7 (cfg 1) | 1.064 [1.051-1.079] | - | - |  |
| 8 | 57.9 | 63.6 (cfg 1) | 1.100 [1.083-1.121] | - | - |  |
| 16 | 57.9 | 165.8 (cfg 1s8) | 2.875 [2.838-2.923] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.01; R=5: P1=1.02; R=6: P1=1.04, P1s2=1.19, P1s3=1.15; R=7: P1=1.06; R=8: P1=1.10, P1s2=1.52, P1s4=1.41, P2s4=1.78; R=16: P1s4=3.10, P1s8=2.87

### down (16-byte-aligned weights), valid rounds 55

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 170.0 | 170.9 (cfg 1) | 1.005 [0.994-1.013] | - | - |  |
| 5 | 170.0 | 170.7 (cfg 1) | 1.004 [0.998-1.011] | - | - |  |
| 6 | 170.0 | 171.0 (cfg 1) | 1.006 [0.999-1.015] | - | - |  |
| 7 | 170.0 | 174.6 (cfg 1) | 1.027 [1.016-1.038] | - | - |  |
| 8 | 170.0 | 203.8 (cfg 1) | 1.198 [1.188-1.233] | - | - |  |
| 16 | 170.0 | 538.4 (cfg 1s8) | 3.165 [3.115-3.260] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.01; R=5: P1=1.00; R=6: P1=1.01, P1s2=1.23, P1s3=1.13; R=7: P1=1.03; R=8: P1=1.20, P1s2=1.60, P1s4=1.50, P2s4=1.51; R=16: P1s4=3.49, P1s8=3.17

### kv (16-byte-aligned weights), valid rounds 58

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 29.7 | 30.3 (cfg 1) | 1.020 [1.008-1.027] | - | - |  |
| 5 | 29.7 | 30.9 (cfg 1) | 1.041 [1.030-1.049] | - | - |  |
| 6 | 29.7 | 31.9 (cfg 1) | 1.076 [1.061-1.086] | - | - |  |
| 7 | 29.7 | 32.8 (cfg 1) | 1.106 [1.094-1.117] | - | - |  |
| 8 | 29.7 | 35.0 (cfg 1) | 1.178 [1.164-1.187] | - | - |  |
| 16 | 29.7 | 80.0 (cfg 1s8) | 2.700 [2.646-2.716] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.02; R=5: P1=1.04; R=6: P1=1.08, P1s2=1.29, P1s3=1.25; R=7: P1=1.11; R=8: P1=1.18, P1s2=1.59, P1s4=1.53, P2s4=1.94; R=16: P1s4=2.91, P1s8=2.70

### gu (16-byte-aligned weights), valid rounds 54

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 336.3 | 338.4 (cfg 1) | 1.007 [0.996-1.012] | - | - |  |
| 5 | 336.3 | 339.3 (cfg 1) | 1.009 [1.001-1.020] | - | - |  |
| 6 | 336.3 | 341.5 (cfg 1) | 1.016 [1.005-1.033] | - | - |  |
| 7 | 336.3 | 348.8 (cfg 1) | 1.038 [1.028-1.051] | - | - |  |
| 8 | 336.3 | 392.7 (cfg 1) | 1.169 [1.159-1.195] | - | - |  |
| 16 | 336.3 | 1150.0 (cfg 1s8) | 3.422 [3.389-3.525] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.01; R=5: P1=1.01; R=6: P1=1.02, P1s2=1.35, P1s3=1.22; R=7: P1=1.04; R=8: P1=1.17, P1s2=1.94, P1s4=1.57, P2s4=1.84; R=16: P1s4=3.73, P1s8=3.42

### lm (16-byte-aligned weights), valid rounds 59

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 2089.2 | 2099.7 (cfg 1) | 1.006 [0.998-1.011] | - | - |  |
| 5 | 2089.2 | 2110.9 (cfg 1) | 1.011 [1.001-1.018] | - | - |  |
| 6 | 2089.2 | 2125.1 (cfg 1s3) | 1.017 [1.008-1.024] | - | - |  |
| 7 | 2089.2 | 2258.8 (cfg 1) | 1.082 [1.072-1.118] | - | - |  |
| 8 | 2089.2 | 2233.5 (cfg 1s4) | 1.068 [1.060-1.081] | - | - |  |
| 16 | 2089.2 | 3957.3 (cfg 1s4) | 1.900 [1.877-1.975] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.01; R=5: P1=1.01; R=6: P1=1.04, P1s2=1.02, P1s3=1.02; R=7: P1=1.08; R=8: P1=1.20, P1s2=1.12, P1s4=1.07, P2s4=1.28; R=16: P1s4=1.90, P1s8=2.39

### qo (8-byte-aligned weights), valid rounds 26

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 58.1 | 58.6 (cfg 1) | 1.012 [0.990-1.029] | - | - |  |
| 5 | 58.1 | 60.7 (cfg 1) | 1.045 [1.028-1.069] | - | - |  |
| 6 | 58.1 | 63.0 (cfg 1) | 1.092 [1.074-1.117] | - | - |  |
| 7 | 58.1 | 67.5 (cfg 1) | 1.173 [1.153-1.224] | - | - |  |
| 8 | 58.1 | 75.6 (cfg 1) | 1.316 [1.292-1.384] | - | - |  |
| 16 | 58.1 | 175.0 (cfg 1s8) | 3.051 [2.995-3.218] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.01; R=5: P1=1.05; R=6: P1=1.09, P1s2=1.22, P1s3=1.18; R=7: P1=1.17; R=8: P1=1.32, P1s2=1.56, P1s4=1.49, P2s4=1.77; R=16: P1s4=3.21, P1s8=3.05

### down (8-byte-aligned weights), valid rounds 29

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 169.5 | 170.0 (cfg 1) | 1.003 [0.997-1.015] | - | - |  |
| 5 | 169.5 | 170.5 (cfg 1) | 1.006 [0.997-1.014] | - | - |  |
| 6 | 169.5 | 175.6 (cfg 1) | 1.035 [1.023-1.045] | - | - |  |
| 7 | 169.5 | 214.3 (cfg 1) | 1.262 [1.201-1.274] | - | - |  |
| 8 | 169.5 | 243.9 (cfg 1) | 1.435 [1.362-1.460] | - | - |  |
| 16 | 169.5 | 692.6 (cfg 1s8) | 4.070 [3.880-4.168] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.00; R=5: P1=1.01; R=6: P1=1.04, P1s2=1.36, P1s3=1.21; R=7: P1=1.26; R=8: P1=1.44, P1s2=1.77, P1s4=1.61, P2s4=1.60; R=16: P1s4=4.50, P1s8=4.07

### kv (8-byte-aligned weights), valid rounds 26

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 30.1 | 31.1 (cfg 1) | 1.026 [1.010-1.045] | - | - |  |
| 5 | 30.1 | 32.7 (cfg 1) | 1.079 [1.065-1.105] | - | - |  |
| 6 | 30.1 | 34.2 (cfg 1) | 1.131 [1.105-1.162] | - | - |  |
| 7 | 30.1 | 37.7 (cfg 1) | 1.256 [1.221-1.277] | - | - |  |
| 8 | 30.1 | 42.6 (cfg 1) | 1.425 [1.376-1.458] | - | - |  |
| 16 | 30.1 | 89.2 (cfg 1s8) | 2.978 [2.883-3.018] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.03; R=5: P1=1.08; R=6: P1=1.13, P1s2=1.28, P1s3=1.24; R=7: P1=1.26; R=8: P1=1.42, P1s2=1.60, P1s4=1.65, P2s4=1.83; R=16: P1s4=2.98, P1s8=2.98

### gu (8-byte-aligned weights), valid rounds 13

| R | T(1) serving (us) | T(R) multi-row (us) | ratio mr/T(1) [p10-p90] | T(R) existing (us) | existing/T(1) | existing path |
|---|---|---|---|---|---|---|
| 4 | 340.9 | 342.6 (cfg 1) | 1.005 [0.985-1.019] | - | - |  |
| 5 | 340.9 | 344.3 (cfg 1) | 1.018 [0.993-1.023] | - | - |  |
| 6 | 340.9 | 367.6 (cfg 1) | 1.080 [1.053-1.106] | - | - |  |
| 7 | 340.9 | 430.2 (cfg 1) | 1.249 [1.233-1.284] | - | - |  |
| 8 | 340.9 | 483.1 (cfg 1) | 1.419 [1.378-1.463] | - | - |  |
| 16 | 340.9 | 1312.6 (cfg 1s8) | 3.848 [3.722-4.052] | - | - |  |
P sweep (output rows per threadgroup; ratio to T(1)): R=2: ; R=3: ; R=4: P1=1.01; R=5: P1=1.02; R=6: P1=1.08, P1s2=1.39, P1s3=1.25; R=7: P1=1.25; R=8: P1=1.42, P1s2=1.99, P1s4=1.63, P2s4=1.85; R=16: P1s4=3.98, P1s8=3.85

### Layer sum (2 x q/o + down + k/v pair + gate/up pair), GEMV time only

| R | alignment | T_layer(1) serving (us) | T_layer(R) multi-row (us) | ratio | T_layer(R) existing path (us) | existing ratio |
|---|---|---|---|---|---|---|
| 4 | 16-byte | 651.9 | 657.0 | 1.008 | - | - |
| 5 | 16-byte | 651.9 | 659.5 | 1.012 | - | - |
| 6 | 16-byte | 651.9 | 665.2 | 1.020 | - | - |
| 7 | 16-byte | 651.9 | 679.7 | 1.043 | - | - |
| 8 | 16-byte | 651.9 | 758.7 | 1.164 | - | - |
| 16 | 16-byte | 651.9 | 2100.0 | 3.221 | - | - |
| 4 | 8-byte | 656.8 | 660.9 | 1.006 | - | - |
| 5 | 8-byte | 656.8 | 668.8 | 1.018 | - | - |
| 6 | 8-byte | 656.8 | 703.4 | 1.071 | - | - |
| 7 | 8-byte | 656.8 | 817.2 | 1.244 | - | - |
| 8 | 8-byte | 656.8 | 920.9 | 1.402 | - | - |
| 16 | 8-byte | 656.8 | 2444.5 | 3.722 | - | - |

### Census-weighted (Qwen3-8B: 7-8 of 36 layers per shape sit in the 8-mod-16 shard), per layer, GEMV time

| R | T_layer(1) (us) | T_layer(R) multi-row | ratio | existing | existing ratio |
|---|---|---|---|---|---|
| 4 | 652.9 | 657.8 | 1.008 | - | - |
| 5 | 652.9 | 661.4 | 1.013 | - | - |
| 6 | 652.9 | 672.8 | 1.031 | - | - |
| 7 | 652.9 | 706.7 | 1.083 | - | - |
| 8 | 652.9 | 790.8 | 1.211 | - | - |
| 16 | 652.9 | 2167.5 | 3.320 | - | - |

### Per verify step, GEMVs only: 36 x layer + lm_head (all R rows through the head), 16-byte-aligned weights

| R | step(1) (ms) | step(R) multi-row (ms) | step(R)/step(1) | step(R) existing (ms) | existing/step(1) | per-token cost multi-row (ms/row) |
|---|---|---|---|---|---|---|
| 4 | 25.56 | 25.75 | 1.008 | - | - | 6.44 |
| 5 | 25.56 | 25.85 | 1.012 | - | - | 5.17 |
| 6 | 25.56 | 26.07 | 1.020 | - | - | 4.35 |
| 7 | 25.56 | 26.73 | 1.046 | - | - | 3.82 |
| 8 | 25.56 | 29.55 | 1.156 | - | - | 3.69 |
| 16 | 25.56 | 79.56 | 3.113 | - | - | 4.97 |

bitwise checks in the bench: True (476 checks)
