| what | T | us per call (median) | p10 | p90 |
|---|---:|---:|---:|---:|
| router pair (logits_bf16_gemv + moe_route_f32) | 1 | 9.63 | 9.61 | 9.68 |
| ink_route_bf16 (fused) | 1 | 8.62 | 8.59 | 8.64 |
| ink_route_bf16 (fused, INTARSIA_INK_ROUTE_PRELOAD) | 1 | 9.97 | 9.95 | 10.02 |
| loop_blend S=3 (4 dispatches): off | 1 | 25.37 | 25.13 | 25.68 |
| loop_blend S=3 (4 dispatches): norm_c | 1 | 19.75 | 19.70 | 19.90 |
| loop_blend S=1 (4 dispatches): off | 1 | 20.10 | 19.98 | 20.22 |
| loop_blend S=1 (4 dispatches): norm_c | 1 | 14.85 | 14.77 | 14.98 |
| conv_mix attn side, plain: off | 1 | 8.65 | 8.61 | 8.67 |
| conv_mix attn side, plain: hoist | 1 | 6.28 | 6.18 | 6.29 |
| conv_mix attn side, blend S=1: off | 1 | 28.35 | 28.26 | 28.55 |
| conv_mix attn side, blend S=1: hoist | 1 | 26.18 | 26.08 | 26.58 |
| conv_mix attn side, blend S=1: norm_c | 1 | 22.84 | 22.78 | 23.12 |
| conv_mix attn side, blend S=1: hoist + norm_c | 1 | 20.71 | 20.68 | 20.81 |
| conv_mix mlp side, plain (Q 3): off | 1 | 12.11 | 12.08 | 12.14 |
| conv_mix mlp side, plain (Q 3): hoist | 1 | 11.95 | 11.91 | 11.99 |
| conv_mix mlp side, blend S=3: off | 1 | 36.42 | 36.33 | 36.65 |
| conv_mix mlp side, blend S=3: hoist | 1 | 37.85 | 37.72 | 38.04 |
| conv_mix mlp side, blend S=3: norm_c | 1 | 31.74 | 31.55 | 31.92 |
| conv_mix mlp side, blend S=3: hoist + norm_c | 1 | 33.24 | 33.09 | 33.47 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 4 | 10.64 | 10.60 | 10.69 |
| ink_route_bf16 (fused) | 4 | 8.84 | 8.82 | 8.88 |
| ink_route_bf16 (fused, INTARSIA_INK_ROUTE_PRELOAD) | 4 | 9.84 | 9.82 | 9.88 |
| loop_blend S=3 (4 dispatches): off | 4 | 30.33 | 30.25 | 30.56 |
| loop_blend S=3 (4 dispatches): norm_c | 4 | 25.82 | 25.79 | 25.96 |
| loop_blend S=1 (4 dispatches): off | 4 | 25.01 | 24.96 | 25.06 |
| loop_blend S=1 (4 dispatches): norm_c | 4 | 19.96 | 19.88 | 20.09 |
| conv_mix attn side, plain: off | 4 | 8.91 | 8.87 | 8.94 |
| conv_mix attn side, plain: hoist | 4 | 6.30 | 6.26 | 6.33 |
| conv_mix attn side, blend S=1: off | 4 | 31.66 | 31.56 | 31.74 |
| conv_mix attn side, blend S=1: hoist | 4 | 30.44 | 30.37 | 30.55 |
| conv_mix attn side, blend S=1: norm_c | 4 | 26.72 | 26.63 | 26.89 |
| conv_mix attn side, blend S=1: hoist + norm_c | 4 | 25.48 | 25.36 | 25.69 |
| conv_mix mlp side, plain (Q 3): off | 4 | 12.42 | 12.39 | 12.45 |
| conv_mix mlp side, plain (Q 3): hoist | 4 | 12.59 | 12.57 | 12.61 |
| conv_mix mlp side, blend S=3: off | 4 | 42.92 | 42.79 | 43.04 |
| conv_mix mlp side, blend S=3: hoist | 4 | 42.76 | 42.72 | 42.83 |
| conv_mix mlp side, blend S=3: norm_c | 4 | 35.35 | 35.29 | 35.43 |
| conv_mix mlp side, blend S=3: hoist + norm_c | 4 | 38.52 | 38.45 | 38.72 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 16 | 23.61 | 23.56 | 23.92 |
| ink_route_bf16 (fused) | 16 | 13.12 | 13.08 | 13.16 |
| ink_route_bf16 (fused, INTARSIA_INK_ROUTE_PRELOAD) | 16 | 12.11 | 12.08 | 12.14 |
| loop_blend S=3 (4 dispatches): off | 16 | 53.90 | 53.81 | 54.08 |
| loop_blend S=3 (4 dispatches): norm_c | 16 | 53.13 | 53.06 | 53.29 |
| loop_blend S=1 (4 dispatches): off | 16 | 30.44 | 30.39 | 30.48 |
| loop_blend S=1 (4 dispatches): norm_c | 16 | 25.90 | 25.88 | 25.95 |
| conv_mix attn side, plain: off | 16 | 9.04 | 9.03 | 9.08 |
| conv_mix attn side, plain: hoist | 16 | 6.39 | 6.38 | 6.40 |
| conv_mix attn side, blend S=1: off | 16 | 38.80 | 38.61 | 38.93 |
| conv_mix attn side, blend S=1: hoist | 16 | 36.08 | 36.05 | 36.15 |
| conv_mix attn side, blend S=1: norm_c | 16 | 32.39 | 32.35 | 32.42 |
| conv_mix attn side, blend S=1: hoist + norm_c | 16 | 31.24 | 31.19 | 31.31 |
| conv_mix mlp side, plain (Q 3): off | 16 | 13.04 | 13.02 | 13.08 |
| conv_mix mlp side, plain (Q 3): hoist | 16 | 13.31 | 13.27 | 13.36 |
| conv_mix mlp side, blend S=3: off | 16 | 80.07 | 79.07 | 80.67 |
| conv_mix mlp side, blend S=3: hoist | 16 | 69.43 | 69.32 | 69.53 |
| conv_mix mlp side, blend S=3: norm_c | 16 | 62.21 | 62.11 | 62.36 |
| conv_mix mlp side, blend S=3: hoist + norm_c | 16 | 66.82 | 66.72 | 66.92 |
