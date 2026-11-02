| op | arm | B=8 | B=9 | B=12 | B=16 |
|---|---|---|---|---|---|
| TOTAL | base | 20.47 | 27.26 | 31.70 | 38.14 |
| TOTAL | r16 | 20.37 | 42.19 | 46.32 | 51.76 |
| TOTAL | nt4 | 20.37 | 89.73 | 93.80 | 99.25 |
| GROUP GDN layers | base | 1.59 | 1.81 | 2.27 | 3.03 |
| GROUP GDN layers | r16 | 1.57 | 1.85 | 2.31 | 3.05 |
| GROUP GDN layers | nt4 | 1.58 | 2.14 | 2.72 | 3.56 |
| GROUP MoE routed experts | base | 8.34 | 9.27 | 11.71 | 14.98 |
| GROUP MoE routed experts | r16 | 8.22 | 9.51 | 11.75 | 14.52 |
| GROUP MoE routed experts | nt4 | 8.28 | 9.66 | 11.77 | 14.39 |
| GROUP dense fp8 projections | base | 5.48 | 9.96 | 10.13 | 10.47 |
| GROUP dense fp8 projections | r16 | 5.50 | 24.55 | 24.73 | 24.95 |
| GROUP dense fp8 projections | nt4 | 5.42 | 71.51 | 71.65 | 71.92 |
| GROUP full attention (core) | base | 1.04 | 1.12 | 1.43 | 1.92 |
| GROUP full attention (core) | r16 | 1.04 | 1.13 | 1.43 | 1.88 |
| GROUP full attention (core) | nt4 | 1.04 | 1.24 | 1.56 | 2.02 |
| GROUP lm_head + argmax | base | 2.07 | 2.50 | 3.26 | 4.37 |
| GROUP lm_head + argmax | r16 | 2.06 | 2.50 | 3.19 | 4.12 |
| GROUP lm_head + argmax | nt4 | 2.06 | 2.50 | 3.19 | 4.12 |
| GROUP norms / embedding | base | 0.49 | 0.48 | 0.49 | 0.51 |
| GROUP norms / embedding | r16 | 0.48 | 0.49 | 0.49 | 0.49 |
| GROUP norms / embedding | nt4 | 0.48 | 0.50 | 0.50 | 0.50 |
| GROUP router + top-k | base | 0.63 | 0.76 | 0.84 | 0.97 |
| GROUP router + top-k | r16 | 0.63 | 0.78 | 0.84 | 0.95 |
| GROUP router + top-k | nt4 | 0.63 | 0.79 | 0.84 | 0.93 |
| GROUP shared expert | base | 0.84 | 1.36 | 1.57 | 1.88 |
| GROUP shared expert | r16 | 0.88 | 1.38 | 1.56 | 1.79 |
| GROUP shared expert | nt4 | 0.87 | 1.38 | 1.57 | 1.80 |
| fp8 2048x4096 (GDN out x30, attn o x10) | base | 1.91 | 3.30 | 3.16 | 3.27 |
| fp8 2048x4096 (GDN out x30, attn o x10) | r16 | 1.93 | 6.55 | 6.69 | 6.88 |
| fp8 2048x4096 (GDN out x30, attn o x10) | nt4 | 1.85 | 18.47 | 18.60 | 18.86 |
| fp8 4096x2048 (GDN z x30) | base | 1.02 | 1.89 | 1.95 | 2.01 |
| fp8 4096x2048 (GDN z x30) | r16 | 1.02 | 4.74 | 4.74 | 4.75 |
| fp8 4096x2048 (GDN z x30) | nt4 | 1.02 | 13.99 | 13.99 | 13.99 |
| fp8 512x2048 (attn k, v x10 each) | base | 0.25 | 0.56 | 0.49 | 0.51 |
| fp8 512x2048 (attn k, v x10 each) | r16 | 0.25 | 0.55 | 0.55 | 0.55 |
| fp8 512x2048 (attn k, v x10 each) | nt4 | 0.25 | 1.50 | 1.50 | 1.50 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | base | 2.30 | 4.21 | 4.53 | 4.69 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | r16 | 2.29 | 12.71 | 12.74 | 12.77 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | nt4 | 2.30 | 37.54 | 37.55 | 37.57 |
