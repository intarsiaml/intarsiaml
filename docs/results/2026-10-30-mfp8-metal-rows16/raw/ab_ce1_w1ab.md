| op | arm | B=8 | B=9 | B=12 | B=16 |
|---|---|---|---|---|---|
| TOTAL | base | 20.41 | 27.32 | 31.36 | 36.73 |
| TOTAL | r16 | 20.42 | 28.74 | 32.87 | 38.27 |
| GROUP GDN layers | base | 1.60 | 1.81 | 2.25 | 2.94 |
| GROUP GDN layers | r16 | 1.59 | 1.81 | 2.27 | 2.97 |
| GROUP MoE routed experts | base | 8.29 | 9.28 | 11.62 | 14.49 |
| GROUP MoE routed experts | r16 | 8.28 | 9.36 | 11.70 | 14.53 |
| GROUP dense fp8 projections | base | 5.47 | 10.00 | 10.02 | 10.08 |
| GROUP dense fp8 projections | r16 | 5.49 | 11.30 | 11.39 | 11.53 |
| GROUP full attention (core) | base | 1.03 | 1.11 | 1.41 | 1.86 |
| GROUP full attention (core) | r16 | 1.03 | 1.12 | 1.42 | 1.87 |
| GROUP lm_head + argmax | base | 2.07 | 2.50 | 3.20 | 4.12 |
| GROUP lm_head + argmax | r16 | 2.07 | 2.50 | 3.20 | 4.12 |
| GROUP norms / embedding | base | 0.49 | 0.49 | 0.49 | 0.50 |
| GROUP norms / embedding | r16 | 0.49 | 0.49 | 0.49 | 0.50 |
| GROUP router + top-k | base | 0.63 | 0.76 | 0.82 | 0.93 |
| GROUP router + top-k | r16 | 0.63 | 0.77 | 0.84 | 0.94 |
| GROUP shared expert | base | 0.83 | 1.36 | 1.54 | 1.80 |
| GROUP shared expert | r16 | 0.84 | 1.38 | 1.56 | 1.81 |
| fp8 2048x4096 (GDN out x30, attn o x10) | base | 1.89 | 3.31 | 3.17 | 3.20 |
| fp8 2048x4096 (GDN out x30, attn o x10) | r16 | 1.92 | 3.36 | 3.41 | 3.51 |
| fp8 4096x2048 (GDN z x30) | base | 1.03 | 1.91 | 1.92 | 1.93 |
| fp8 4096x2048 (GDN z x30) | r16 | 1.02 | 2.18 | 2.19 | 2.19 |
| fp8 512x2048 (attn k, v x10 each) | base | 0.25 | 0.56 | 0.48 | 0.48 |
| fp8 512x2048 (attn k, v x10 each) | r16 | 0.25 | 0.32 | 0.32 | 0.32 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | base | 2.30 | 4.22 | 4.45 | 4.46 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | r16 | 2.30 | 5.44 | 5.47 | 5.50 |
