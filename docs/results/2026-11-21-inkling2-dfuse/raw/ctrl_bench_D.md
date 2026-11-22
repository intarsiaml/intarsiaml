| what | T | us per call (median) | p10 | p90 |
|---|---:|---:|---:|---:|
| router pair (logits_bf16_gemv + moe_route_f32) | 1 | 9.65 | 9.62 | 9.68 |
| ink_route_bf16 (fused) | 1 | 8.60 | 8.58 | 8.62 |
| loop_blend S=3 (4 dispatches) | 1 | 24.75 | 24.58 | 25.14 |
| loop_blend S=3 (fused tail, 2 dispatches) | 1 | 32.33 | 32.08 | 32.87 |
| loop_blend S=1 (4 dispatches) | 1 | 20.65 | 20.50 | 20.78 |
| loop_blend S=1 (fused tail, 2 dispatches) | 1 | 20.94 | 20.73 | 21.05 |
| conv_mix attn side, plain: off | 1 | 8.36 | 8.33 | 8.39 |
| conv_mix attn side, plain: hoist | 1 | 5.96 | 5.93 | 5.99 |
| conv_mix attn side, blend S=1: off | 1 | 27.81 | 27.58 | 27.99 |
| conv_mix attn side, blend S=1: hoist | 1 | 25.28 | 25.11 | 25.53 |
| conv_mix attn side, blend S=1: norm_c | 1 | 22.38 | 22.20 | 22.85 |
| conv_mix attn side, blend S=1: hoist + norm_c | 1 | 20.07 | 19.96 | 20.33 |
| conv_mix mlp side, plain (Q 3): off | 1 | 11.25 | 11.23 | 11.28 |
| conv_mix mlp side, plain (Q 3): hoist | 1 | 9.99 | 9.97 | 10.02 |
| conv_mix mlp side, blend S=3: off | 1 | 35.80 | 35.61 | 36.09 |
| conv_mix mlp side, blend S=3: hoist | 1 | 36.54 | 36.34 | 36.77 |
| conv_mix mlp side, blend S=3: norm_c | 1 | 31.37 | 30.93 | 31.64 |
| conv_mix mlp side, blend S=3: hoist + norm_c | 1 | 31.70 | 31.46 | 32.22 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 4 | 10.65 | 10.61 | 10.70 |
| ink_route_bf16 (fused) | 4 | 8.82 | 8.80 | 8.83 |
| loop_blend S=3 (4 dispatches) | 4 | 30.66 | 30.52 | 30.84 |
| loop_blend S=3 (fused tail, 2 dispatches) | 4 | 66.07 | 65.61 | 66.58 |
| loop_blend S=1 (4 dispatches) | 4 | 24.95 | 24.87 | 25.09 |
| loop_blend S=1 (fused tail, 2 dispatches) | 4 | 45.37 | 45.16 | 45.70 |
| conv_mix attn side, plain: off | 4 | 8.53 | 8.51 | 8.56 |
| conv_mix attn side, plain: hoist | 4 | 6.13 | 6.11 | 6.14 |
| conv_mix attn side, blend S=1: off | 4 | 31.00 | 30.89 | 31.16 |
| conv_mix attn side, blend S=1: hoist | 4 | 29.79 | 29.56 | 30.07 |
| conv_mix attn side, blend S=1: norm_c | 4 | 25.95 | 25.89 | 26.05 |
| conv_mix attn side, blend S=1: hoist + norm_c | 4 | 24.23 | 24.18 | 24.27 |
| conv_mix mlp side, plain (Q 3): off | 4 | 11.49 | 11.45 | 11.52 |
| conv_mix mlp side, plain (Q 3): hoist | 4 | 10.22 | 10.20 | 10.24 |
| conv_mix mlp side, blend S=3: off | 4 | 42.38 | 42.19 | 42.63 |
| conv_mix mlp side, blend S=3: hoist | 4 | 40.91 | 40.78 | 41.15 |
| conv_mix mlp side, blend S=3: norm_c | 4 | 34.82 | 34.61 | 35.08 |
| conv_mix mlp side, blend S=3: hoist + norm_c | 4 | 35.93 | 35.82 | 36.22 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 16 | 24.19 | 23.88 | 25.06 |
| ink_route_bf16 (fused) | 16 | 13.06 | 13.04 | 13.09 |
| loop_blend S=3 (4 dispatches) | 16 | 63.75 | 63.48 | 63.97 |
| loop_blend S=3 (fused tail, 2 dispatches) | 16 | 207.81 | 203.82 | 211.70 |
| loop_blend S=1 (4 dispatches) | 16 | 48.84 | 48.53 | 49.30 |
| loop_blend S=1 (fused tail, 2 dispatches) | 16 | 146.35 | 145.73 | 147.03 |
| conv_mix attn side, plain: off | 16 | 8.81 | 8.77 | 8.85 |
| conv_mix attn side, plain: hoist | 16 | 6.30 | 6.28 | 6.33 |
| conv_mix attn side, blend S=1: off | 16 | 53.76 | 53.54 | 53.93 |
| conv_mix attn side, blend S=1: hoist | 16 | 52.61 | 52.48 | 52.71 |
| conv_mix attn side, blend S=1: norm_c | 16 | 50.10 | 49.90 | 50.28 |
| conv_mix attn side, blend S=1: hoist + norm_c | 16 | 48.42 | 48.26 | 48.67 |
| conv_mix mlp side, plain (Q 3): off | 16 | 12.67 | 12.61 | 12.72 |
| conv_mix mlp side, plain (Q 3): hoist | 16 | 10.80 | 10.77 | 10.84 |
| conv_mix mlp side, blend S=3: off | 16 | 92.01 | 91.48 | 92.62 |
| conv_mix mlp side, blend S=3: hoist | 16 | 74.12 | 73.95 | 74.33 |
| conv_mix mlp side, blend S=3: norm_c | 16 | 70.92 | 70.65 | 71.33 |
| conv_mix mlp side, blend S=3: hoist + norm_c | 16 | 72.21 | 71.96 | 72.57 |
