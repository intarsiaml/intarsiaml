### Component times (ms, median)

| pass | ctx 512 | ctx 1024 | ctx 4096 |
|---|---:|---:|---:|
| plain1 | 28.01 | 27.79 | 28.30 |
| plain2 | 28.63 | 28.22 | 29.94 |
| plain3 | 29.64 | 28.96 | 31.53 |
| plain4 | 30.31 | 30.23 | 33.29 |
| plain5 | 31.60 | 31.53 | 35.53 |
| plain6 | 33.22 | 33.55 | 38.30 |
| verify2 | 27.98 | 28.24 | 29.20 |
| verify3 | 28.25 | 28.72 | 29.85 |
| verify4 | 28.83 | 29.10 | 30.65 |
| verify5 | 29.52 | 29.62 | 31.54 |
| verify6 | 30.93 | 31.01 | 33.64 |

### Today's iterations: measured composite against the sum of its parts (ms, ctx 512 / 1024 / 4096)

| iteration | measured | sum of parts | ratio |
|---|---|---|---|
| iter_b2_k2 | 56.19 / 56.37 / 58.08 | 56.26 / 56.50 / 58.16 | 0.999 / 0.998 / 0.999 |
| iter_b2_k3 | 56.76 / 56.52 / 58.71 | 56.83 / 56.89 / 58.95 | 0.999 / 0.993 / 0.996 |
| iter_b2_k4 | 57.64 / 57.02 / 59.47 | 57.53 / 57.41 / 59.84 | 1.002 / 0.993 / 0.994 |
| iter_b3_k2 | 56.99 / 56.91 / 59.32 | 56.88 / 56.94 / 59.79 | 1.002 / 1.000 / 0.992 |
| iter_b3_k3 | 57.53 / 57.39 / 60.45 | 57.46 / 57.32 / 60.58 | 1.001 / 1.001 / 0.998 |
| iter_b3_k4 | 58.08 / 58.13 / 61.80 | 58.15 / 57.84 / 61.47 | 0.999 / 1.005 / 1.005 |
| iter_vv2_k4 | 59.09 / 59.47 / 63.62 | 59.05 / 59.24 / 63.07 | 1.001 / 1.004 / 1.009 |
| iter_vv3_k4 | 88.90 / 89.41 / 95.72 | 88.57 / 88.86 / 94.61 | 1.004 / 1.006 / 1.012 |

### Attention: what a q-row verify group saves against q single-row launches inside a plain step (ms per pass)

| q | g0b-derived ctx 512 | g0b-derived ctx 1024 | g0b-derived ctx 4096 | measured P(q)-V(q) ctx 512 | measured P(q)-V(q) ctx 1024 | measured P(q)-V(q) ctx 4096 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.37 | 0.14 | 0.76 | 0.65 | 0.00 | 0.74 |
| 3 | 0.61 | 0.36 | 1.61 | 1.38 | 0.25 | 1.68 |
| 4 | 0.86 | 0.79 | 2.34 | 1.48 | 1.13 | 2.65 |
| 5 | 1.10 | 1.18 | 3.12 | 2.08 | 1.91 | 4.00 |
| 6 | 1.40 | 1.58 | 3.82 | 2.30 | 2.53 | 4.66 |

### g0b_bench attn, us per layer (this slot): T1 = serving 1-row; T(q) = rows kernel

| ctx | T1 | T(2) | T(3) | T(4) | T(5) | T(6) |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 11.3 | 12.2 | 16.8 | 21.3 | 25.8 | 28.8 |
| 1024 | 14.6 | 25.3 | 33.7 | 36.3 | 39.9 | 43.4 |
| 4096 | 36.6 | 52.2 | 65.2 | 81.5 | 96.5 | 113.5 |

### One speculating sequence among B (what the brief describes): today vs one batched pass, ms, ctx 512 / 1024 / 4096

| B | k | rows R=B+k | today V(1+k)+P(B-1) | batched pass P(R) - corr(g0b) | saving |
|---|---|---|---|---|---|
| 2 | 2 | 4 | 56.3 / 56.5 / 58.2 | 29.7 / 29.9 / 31.7 | 47% / 47% / 46% |
| 2 | 3 | 5 | 56.8 / 56.9 / 59.0 | 30.7 / 30.7 / 33.2 | 46% / 46% / 44% |
| 2 | 4 | 6 | 57.5 / 57.4 / 59.8 | 32.1 / 32.4 / 35.2 | 44% / 44% / 41% |
| 3 | 2 | 5 | 56.9 / 56.9 / 59.8 | 31.0 / 31.2 / 33.9 | 46% / 45% / 43% |
| 3 | 3 | 6 | 57.5 / 57.3 / 60.6 | 32.4 / 32.8 / 36.0 | 44% / 43% / 41% |
| 3 | 4 | 7 | does not fit one pass (R > 6): 2 passes | | |

### Analytic, not traced and not a pre-registered scenario: one grounded stream beside open-ended stream(s), spec on for the grounded one (k = 4)

Per grounded verify pass of t = 3.95 tokens (506 tokens / 128 passes, grounded-n2 counters): the grounded stream verifies in 1 iteration and drains its other t-1 tokens with no row; each open stream needs a plain row in every iteration.

| B | ctx | today ms per cycle | batched-verify ms per cycle | tokens per cycle | today tok/s | batched tok/s | gain |
|---|---|---|---|---|---|---|---|
| 2 | 512 | 140.2 | 114.7 | 7.90 | 56.4 | 68.8 | +22% |
| 2 | 1024 | 139.4 | 114.3 | 7.90 | 56.7 | 69.1 | +22% |
| 2 | 4096 | 143.3 | 118.7 | 7.90 | 55.1 | 66.6 | +21% |
| 3 | 512 | 142.6 | 144.6 | 11.85 | 83.1 | 82.0 | -1% |
| 3 | 1024 | 141.1 | 143.4 | 11.85 | 84.0 | 82.6 | -2% |
| 3 | 4096 | 149.8 | 151.8 | 11.85 | 79.1 | 78.1 | -1% |
