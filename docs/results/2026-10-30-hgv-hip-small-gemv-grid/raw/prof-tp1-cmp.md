step wall 40.24 -> 36.63 ms, kernel busy 37.03 -> 33.42, idle in step 3.21 -> 3.21, launches 1001 -> 1001

| kernel | before calls x us | before ms/step | after calls x us | after ms/step |
|---|---|---:|---|---:|
| `gemv_kernel<16` | 60 x 23.06; 40 x 24.33; 40 x 23.83 | 3.31 | - | 0.00 |
| `gemv_rows_grp_kernel` | - | 0.00 | 40 x 5.46; 60 x 3.62; 40 x 3.05 | 0.56 |
| `shared_glu_rows` | 40 x 28.95 | 1.16 | 40 x 13.85 | 0.55 |
| `shared_down_combine_rows` | 40 x 22.27 | 0.89 | 40 x 22.28 | 0.89 |
