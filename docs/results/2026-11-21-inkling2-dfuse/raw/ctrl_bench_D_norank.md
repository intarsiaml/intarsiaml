| router pair (logits_bf16_gemv + moe_route_f32) | 1 | 9.61 | 9.54 | 9.69 |
| ink_route_bf16 (fused) | 1 | 9.55 | 9.52 | 9.60 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 4 | 10.63 | 10.60 | 10.67 |
| ink_route_bf16 (fused) | 4 | 9.72 | 9.70 | 9.74 |
| router pair (logits_bf16_gemv_rows + moe_route_f32) | 16 | 23.54 | 23.49 | 23.83 |
| ink_route_bf16 (fused) | 16 | 13.06 | 13.04 | 13.08 |
