| what | T | us per call (median) | p10 | p90 |
|---|---:|---:|---:|---:|
| router pair (logits_bf16_gemv + moe_route_f32) | 1 | 9.68 | 9.59 | 9.73 |
| ink_route_bf16 (fused) | 1 | 9.47 | 9.43 | 9.50 |
| loop_blend S=3 (4 dispatches) | 1 | 25.37 | 25.07 | 25.56 |
| loop_blend S=3 (fused tail, 2 dispatches) | 1 | 32.46 | 32.25 | 32.88 |
| loop_blend S=1 (4 dispatches) | 1 | 19.99 | 19.81 | 20.43 |
| loop_blend S=1 (fused tail, 2 dispatches) | 1 | 20.61 | 20.49 | 20.86 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 2 | 9.79 | 9.75 | 9.83 |
| ink_route_bf16 (fused) | 2 | 9.38 | 9.35 | 9.42 |
| loop_blend S=3 (4 dispatches) | 2 | 28.03 | 27.47 | 28.31 |
| loop_blend S=3 (fused tail, 2 dispatches) | 2 | 44.16 | 43.79 | 44.44 |
| loop_blend S=1 (4 dispatches) | 2 | 21.54 | 21.49 | 21.74 |
| loop_blend S=1 (fused tail, 2 dispatches) | 2 | 28.55 | 28.46 | 28.75 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 3 | 10.02 | 9.97 | 10.06 |
| ink_route_bf16 (fused) | 3 | 9.49 | 9.46 | 9.51 |
| loop_blend S=3 (4 dispatches) | 3 | 29.62 | 29.43 | 29.92 |
| loop_blend S=3 (fused tail, 2 dispatches) | 3 | 55.50 | 55.05 | 55.98 |
| loop_blend S=1 (4 dispatches) | 3 | 23.29 | 23.09 | 23.47 |
| loop_blend S=1 (fused tail, 2 dispatches) | 3 | 37.01 | 36.92 | 37.09 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 4 | 10.64 | 10.58 | 10.69 |
| ink_route_bf16 (fused) | 4 | 9.54 | 9.51 | 9.59 |
| loop_blend S=3 (4 dispatches) | 4 | 30.44 | 30.28 | 30.75 |
| loop_blend S=3 (fused tail, 2 dispatches) | 4 | 66.62 | 66.21 | 67.05 |
| loop_blend S=1 (4 dispatches) | 4 | 24.95 | 24.87 | 25.11 |
| loop_blend S=1 (fused tail, 2 dispatches) | 4 | 45.70 | 45.47 | 46.20 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 16 | 23.73 | 23.41 | 24.50 |
| ink_route_bf16 (fused) | 16 | 12.97 | 12.95 | 13.06 |
| loop_blend S=3 (4 dispatches) | 16 | 76.20 | 75.78 | 76.71 |
| loop_blend S=3 (fused tail, 2 dispatches) | 16 | 209.00 | 205.30 | 212.12 |
| loop_blend S=1 (4 dispatches) | 16 | 47.48 | 47.38 | 47.56 |
| loop_blend S=1 (fused tail, 2 dispatches) | 16 | 148.17 | 147.55 | 148.74 |
