step wall 29.58 -> 25.50 ms, kernel busy 26.24 -> 22.16, idle in step 3.34 -> 3.35, launches 1075 -> 1075

| kernel | before calls x us | before ms/step | after calls x us | after ms/step |
|---|---|---:|---|---:|
| `gemv_kernel<16` | 100 x 22.73; 40 x 22.58 | 3.18 | - | 0.00 |
| `gemv_rows_grp_kernel` | - | 0.00 | 100 x 3.27; 40 x 5.18 | 0.53 |
| `shared_glu_rows` | 40 x 24.74 | 0.99 | 40 x 7.89 | 0.32 |
| `shared_down_combine_rows` | 40 x 21.48 | 0.86 | 40 x 21.72 | 0.87 |
| `ar2_kernel` | 40 x 43.39 | 1.74 | - | 0.00 |
| `ar2ll_kernel` | 40 x 11.70 | 0.47 | 40 x 26.24; 40 x 10.31 | 1.46 |
